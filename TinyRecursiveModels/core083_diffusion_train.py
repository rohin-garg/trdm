
#!/usr/bin/env python
"""
MAIN LOGIC

Implements the core-083 diffusion + one-step distillation pipeline exactly as in the blueprint:
  * Base TRM (84 cores, ACT off, fixed 4 steps) loaded from HF checkpoint.
  * Noisy core: per-sample scalar RMS over flattened core input; noise ~ 0.02 * RMS * N(0, 1).
  * Diffusion (epsilon-pred): input concat([x_clean, x_t, t_token]) of length 97+97+1=195, same core block stack as TRM, then a linear over the sequence dim to project back to length 97.
  * One-step: input concat([x_clean, diffusion_noise]) length 194, same core block stack, linear over sequence dim to length 97.
  * DDIM (eta=0) starts from diffusion_noise ~ N(0, 1) with no x_clean leakage. Distill DDIM outputs into one-step. All conditioning is via concatenation; no cross-sample mixing.
  * Streaming mode: each batch performs teacher inference -> diffusion update -> one-step distillation, then is discarded (no serialized buffers).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1Block
from models.layers import RotaryEmbedding

from streaming_core_train import (
    ArchConfig,
    LossConfig,
    PretrainConfig,
    _build_pretrain_config,
    _load_arch_config,
    build_teacher,
    success_mask,
)
from pretrain import create_dataloader, create_model

# Checkpoint cadence (by correct puzzles) to avoid long uninterrupted runs.
DEFAULT_CORRECT_CHECKPOINT_INTERVAL = 200_000


def init_distributed() -> Tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


class _CoreStack(nn.Module):
    """Stack of TRM core blocks with rotary positional encodings."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        expansion: float,
        num_layers: int,
        rms_norm_eps: float,
        rope_theta: float,
        seq_len: int,
    ) -> None:
        super().__init__()
        cfg = type(
            "BlockCfg",
            (),
            {
                "hidden_size": hidden_dim,
                "num_heads": num_heads,
                "expansion": expansion,
                "rms_norm_eps": rms_norm_eps,
                "mlp_t": False,
                "puzzle_emb_len": 0,
                "puzzle_emb_ndim": 0,
            },
        )()
        self.layers = nn.ModuleList([TinyRecursiveReasoningModel_ACTV1Block(cfg) for _ in range(num_layers)])
        self.rotary_emb = RotaryEmbedding(dim=hidden_dim // num_heads, max_position_embeddings=seq_len, base=rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos_sin = self.rotary_emb()
        h = x
        for layer in self.layers:
            h = layer(cos_sin=cos_sin, hidden_states=h)
        return h


class CoreDiffusionModel(nn.Module):
    """Predict diffusion epsilon given (x_clean, x_t, timestep token) using the core architecture."""

    def __init__(
        self,
        seq_len_clean: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        expansion: float,
        rms_norm_eps: float,
        rope_theta: float,
        num_steps: int,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.seq_len_clean = seq_len_clean
        self.seq_len_noisy = seq_len_clean
        self.seq_len_t = 1
        total_len = self.seq_len_clean + self.seq_len_noisy + self.seq_len_t
        self.stack = _CoreStack(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            num_layers=num_layers,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            seq_len=total_len,
        )
        # Learned timestep embedding; indexed by t_idx and broadcast to the t token.
        self.t_embed = nn.Embedding(num_steps, hidden_dim)
        # Project full sequence output (total_len) back to target length (seq_len_noisy)
        self.out_proj = nn.Linear(total_len, self.seq_len_noisy)

    def forward(self, x_clean: torch.Tensor, noisy: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        # x_clean: [B, 97, D], noisy=x_t: [B, 97, D], t_idx: [B]
        t_vec = self.t_embed(t_idx)[:, None, :]  # [B,1,D]
        t_token = t_vec.expand(-1, self.seq_len_t, -1)  # [B,1,D]
        concat = torch.cat([x_clean, noisy, t_token], dim=1)  # [B, 195, D]
        h = self.stack(concat)  # [B, 195, D]
        # Project over the sequence dimension (195 -> 97) to match core output length.
        h_perm = h.permute(0, 2, 1)
        h_proj = self.out_proj(h_perm)  # [B, D, seq_len_noisy]
        return h_proj.permute(0, 2, 1)


class OneStepCoreModel(nn.Module):
    """One-step distilled model: (x_clean, diffusion_noise) -> deterministic DDIM output, core architecture."""

    def __init__(
        self,
        seq_len_clean: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        expansion: float,
        rms_norm_eps: float,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.seq_len_clean = seq_len_clean
        self.seq_len_noise = seq_len_clean
        total_len = self.seq_len_clean + self.seq_len_noise
        self.stack = _CoreStack(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            num_layers=num_layers,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            seq_len=total_len,
        )
        self.out_proj = nn.Linear(total_len, self.seq_len_noise)

    def forward(self, x_clean: torch.Tensor, noise_init: torch.Tensor) -> torch.Tensor:
        # x_clean: [B,97,D], noise_init: [B,97,D]
        concat = torch.cat([x_clean, noise_init], dim=1)  # [B,194,D]
        h = self.stack(concat)  # [B,194,D]
        h_perm = h.permute(0, 2, 1)
        h_proj = self.out_proj(h_perm)  # [B,D,97]
        return h_proj.permute(0, 2, 1)


def make_ddim_schedule(
    steps: int,
    schedule: str = "linear",
    alpha_start: float = 0.9,
    alpha_end: float = 0.1,
) -> torch.Tensor:
    """Return alpha_cumprod schedule."""
    if schedule == "linear":
        return torch.linspace(alpha_start, alpha_end, steps)
    if schedule == "cosine":
        # Nichol & Dhariwal cosine schedule with offset s=0.008
        s = 0.008
        t = torch.linspace(0, 1, steps)
        f = torch.cos(((t + s) / (1 + s)) * (0.5 * torch.pi)) ** 2
        bar_alpha = f / f[0]
        return bar_alpha
    raise ValueError(f"Unknown schedule: {schedule}")


def ddim_sample(
    model: CoreDiffusionModel,
    x_clean: torch.Tensor,
    noise_init: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Deterministic DDIM (eta=0) for an epsilon-predictor."""
    device = x_clean.device
    noisy = noise_init
    steps = alphas_cumprod.numel()
    for i in reversed(range(steps)):
        alpha_t = alphas_cumprod[i].to(device)
        t_idx = torch.full((x_clean.shape[0],), i, device=device, dtype=torch.long)
        eps_pred = model(x_clean, noisy, t_idx)
        x0_pred = (noisy - (1 - alpha_t).sqrt() * eps_pred) / alpha_t.sqrt()
        if i == 0:
            noisy = x0_pred
        else:
            alpha_prev = alphas_cumprod[i - 1].to(device)
            noisy = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * eps_pred
    return noisy


def prepare_loader(cfg, split: str, rank: int, world_size: int, batch_size: int):
    cfg = cfg.copy(update={"global_batch_size": batch_size * world_size})
    loader, metadata = create_dataloader(
        cfg,
        split=split,
        rank=rank,
        world_size=world_size,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=cfg.global_batch_size,
    )
    return loader, metadata


def main(args: argparse.Namespace) -> None:
    rank, world_size, local_rank, device = init_distributed()
    torch.manual_seed(1234 + rank)
    os.environ.setdefault("DISABLE_COMPILE", "1")
    diffusion_data_dir = args.diffusion_data_dir or args.data_dir
    distill_data_dir = args.distill_data_dir or diffusion_data_dir
    eval_data_dir = args.eval_data_dir or diffusion_data_dir
    if rank == 0:
        print(
            "[config] "
            f"diffusion_data={diffusion_data_dir} "
            f"eval_data={eval_data_dir} "
            f"distill_data(unused_streaming)={distill_data_dir}"
        )

    # Build teacher (frozen)
    teacher, loader, metadata = build_teacher(
        checkpoint=args.checkpoint,
        data_dir=diffusion_data_dir,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    seq_len = teacher.inner.config.seq_len + teacher.inner.puzzle_emb_len  # type: ignore[attr-defined]
    hidden_dim = teacher.inner.config.hidden_size  # type: ignore[attr-defined]

    # Diffusion model and optimizer (match TRM core architecture; only seq_len differs)
    core_layers = teacher.inner.config.L_layers  # type: ignore[attr-defined]
    diffusion = CoreDiffusionModel(
        seq_len_clean=seq_len,
        hidden_dim=hidden_dim,
        num_layers=core_layers,
        num_heads=teacher.inner.config.num_heads,  # type: ignore[attr-defined]
        expansion=teacher.inner.config.expansion,  # type: ignore[attr-defined]
        rms_norm_eps=teacher.inner.config.rms_norm_eps,  # type: ignore[attr-defined]
        rope_theta=teacher.inner.config.rope_theta,  # type: ignore[attr-defined]
        num_steps=args.diffusion_steps,
    ).to(device)
    if world_size > 1:
        diffusion = DDP(diffusion, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    diff_opt = torch.optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    alphas_cumprod = make_ddim_schedule(
        args.diffusion_steps, args.schedule, args.alpha_start, args.alpha_end
    ).to(device)

    # Optionally load an existing diffusion checkpoint and skip diffusion training
    global_step = 0
    total_correct = 0
    core_model = diffusion.module if isinstance(diffusion, DDP) else diffusion
    train_diffusion = args.load_diffusion is None or args.train_diffusion
    if args.load_diffusion is not None:
        state = torch.load(args.load_diffusion, map_location=device)
        diffusion_state = state.get("diffusion", state)
        (diffusion.module if isinstance(diffusion, DDP) else diffusion).load_state_dict(diffusion_state)
        if "alphas_cumprod" in state:
            alphas_cumprod = state["alphas_cumprod"].to(device)
        if rank == 0:
            print(f"[info] loaded diffusion weights from {args.load_diffusion}")
        global_step = state.get("steps", global_step)
        total_correct = state.get("correct", total_correct)

    eval_loader, _ = prepare_loader(
        _build_pretrain_config(_load_arch_config(), eval_data_dir, args.batch_size * world_size, args.checkpoint),
        split="test",
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
    )

    # Quick sanity evals on teacher and identity replacement (no replacements)
    def _quick_eval(eval_loader, replacements=None, max_puzzles: int = args.eval_puzzles, label: str = "eval"):
        total = 0
        correct = 0
        it = iter(eval_loader)
        while total < max_puzzles:
            try:
                _, batch, _ = next(it)
            except StopIteration:
                it = iter(eval_loader)
                _, batch, _ = next(it)
            take = min(batch["inputs"].shape[0], max_puzzles - total)
            if take < batch["inputs"].shape[0]:
                batch = {k: v[:take].clone() for k, v in batch.items()}
            total += batch["inputs"].shape[0]
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.no_grad():
                logits, _ = teacher.streaming_forward(
                    batch=batch,
                    steps=4,
                    noise_scale=args.noise_scale,
                    collect_core_io=False,
                    replacements=replacements,
                )
            mask = success_mask(logits, batch["labels"])
            correct += int(mask.sum().item())
        return correct / max(1, total)

    if rank == 0:
        base_pass = _quick_eval(eval_loader, None, label="teacher")
        ident_pass = _quick_eval(eval_loader, {83: (lambda x, noise: x)}, label="identity")
        print(f"[sanity] teacher pass@1={base_pass:.4f} identity_core083 pass@1={ident_pass:.4f}")
    if dist.is_initialized():
        dist.barrier()

    # Streaming state
    loader_it = iter(loader)
    min_steps = max(10, args.min_steps) if train_diffusion else 0
    next_milestone = 100_000
    next_ckpt = args.correct_ckpt_interval
    if rank == 0:
        print(
            "[info] starting streaming training; "
            f"target_correct={args.target_correct} min_steps={min_steps} max_steps={args.max_steps}"
        )

    # One-step model (trained in-stream alongside diffusion; no serialized buffers)
    one_step = OneStepCoreModel(
        seq_len_clean=seq_len,
        hidden_dim=hidden_dim,
        num_layers=core_layers,
        num_heads=teacher.inner.config.num_heads,  # type: ignore[attr-defined]
        expansion=teacher.inner.config.expansion,  # type: ignore[attr-defined]
        rms_norm_eps=teacher.inner.config.rms_norm_eps,  # type: ignore[attr-defined]
        rope_theta=teacher.inner.config.rope_theta,  # type: ignore[attr-defined]
    ).to(device)
    if world_size > 1:
        one_step = DDP(one_step, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    one_opt = torch.optim.AdamW(one_step.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    one_step_updates = 0

    # Streaming loop: teacher -> diffusion update -> one-step distill, then drop batch
    if rank == 0:
        print("[stream] starting joint streaming loop (no serialized buffers)")
    while total_correct < args.target_correct and global_step < args.max_steps:
        try:
            _, batch, _ = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            _, batch, _ = next(loader_it)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits, records = teacher.streaming_forward(
            batch=batch,
            steps=4,
            noise_scale=args.noise_scale,
            collect_core_io=True,
            replacements=None,
        )
        success = success_mask(logits, batch["labels"])
        local_success = int(success.sum().item())
        global_success = local_success
        if dist.is_initialized():
            success_tensor = torch.tensor([local_success], device=device, dtype=torch.long)
            dist.all_reduce(success_tensor, op=dist.ReduceOp.SUM)
            global_success = int(success_tensor.item())
        if rank == 0 and global_step % args.log_interval == 0:
            print(
                f"[info] loop step={global_step} batch_success={global_success} "
                f"total_correct={total_correct} one_step_updates={one_step_updates}"
            )
        if global_success > 0:
            total_correct += global_success
            if rank == 0:
                while total_correct >= next_milestone:
                    print(f"[info] reached {next_milestone} correct puzzles")
                    next_milestone += 100_000
                if total_correct >= next_ckpt:
                    out_dir = Path(args.output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = out_dir / f"core083_stream_step{global_step}_correct{total_correct}.pt"
                    torch.save(
                        {
                            "diffusion": (diffusion.module if isinstance(diffusion, DDP) else diffusion).state_dict(),
                            "one_step": (one_step.module if isinstance(one_step, DDP) else one_step).state_dict(),
                            "alphas_cumprod": alphas_cumprod.cpu(),
                            "steps": global_step,
                            "one_step_updates": one_step_updates,
                            "correct": total_correct,
                        },
                        ckpt_path,
                    )
                    print(f"[checkpoint] saved {ckpt_path}")
                    next_ckpt += args.correct_ckpt_interval

        core_records = [(ci, xc, out) for ci, xc, _noise, out in records if ci == 83]
        if not core_records:
            continue

        _, x_clean, target = core_records[0]
        x_sel = x_clean[success].to(torch.float32).detach()
        target_sel = target[success].to(torch.float32).detach()
        if x_sel.numel() == 0:
            continue

        # Diffusion update (epsilon prediction)
        if train_diffusion:
            diff_noise = torch.randn_like(target_sel)
            t_idx = torch.randint(0, args.diffusion_steps, (x_sel.shape[0],), device=device)
            alpha_t = alphas_cumprod[t_idx].view(-1, 1, 1)
            noisy_out = (alpha_t.sqrt() * target_sel) + ((1 - alpha_t).sqrt() * diff_noise)
            pred = diffusion(x_sel, noisy_out, t_idx)
            loss = F.mse_loss(pred, diff_noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), args.max_grad_norm)
            diff_opt.step()
            diff_opt.zero_grad(set_to_none=True)
            global_step += 1
            if rank == 0 and global_step % args.log_interval == 0:
                print(
                    f"[train-diff] step={global_step} loss={float(loss):.6f} "
                    f"success_batch={global_success} total_correct={total_correct}"
                )

        # One-step distillation directly from current diffusion (DDIM, eta=0)
        with torch.no_grad():
            noise_init = torch.randn_like(x_sel)
            sample_out = ddim_sample(core_model, x_sel, noise_init, alphas_cumprod)
        pred_one = one_step(x_sel, noise_init)
        loss_one = F.mse_loss(pred_one, sample_out)
        loss_one.backward()
        torch.nn.utils.clip_grad_norm_(one_step.parameters(), args.max_grad_norm)
        one_opt.step()
        one_opt.zero_grad(set_to_none=True)
        one_step_updates += 1
        if rank == 0 and one_step_updates % args.log_interval == 0:
            # Log a normalized view to make scale interpretable
            norm = sample_out.numel()
            print(
                f"[train-one] step={one_step_updates} loss={float(loss_one):.6f} "
                f"loss_per_elem={float(loss_one) / max(1, norm):.6e} "
                f"total_correct={total_correct}"
            )

        if total_correct >= args.target_correct and (not train_diffusion or global_step >= min_steps):
            break

    if rank == 0:
        print(
            f"[done] collected_correct={total_correct} diffusion_steps={global_step} "
            f"one_step_updates={one_step_updates}"
        )
    if dist.is_initialized():
        dist.barrier()

    # Eval by swapping core 083 with one-step
    def replacement_fn(x_clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # Use the provided noise for consistency with the teacher sampling.
        eps0 = noise.to(torch.float32)
        core = one_step.module if isinstance(one_step, DDP) else one_step
        with torch.no_grad():
            return core(x_clean.to(torch.float32), eps0)

    replacements = {83: replacement_fn}
    eval_loader, _ = prepare_loader(
        _build_pretrain_config(_load_arch_config(), eval_data_dir, args.batch_size * world_size, args.checkpoint),
        split="test",
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
    )
    total = 0
    correct = 0
    it = iter(eval_loader)
    while total < args.eval_puzzles:
        try:
            _, batch, _ = next(it)
        except StopIteration:
            it = iter(eval_loader)
            _, batch, _ = next(it)
        take = min(batch["inputs"].shape[0], args.eval_puzzles - total)
        if take < batch["inputs"].shape[0]:
            batch = {k: v[:take].clone() for k, v in batch.items()}
        total += batch["inputs"].shape[0]
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if batch["labels"].numel() == 0:
            if rank == 0:
                print("[eval] empty batch encountered; skipping")
            continue
        logits, _ = teacher.streaming_forward(
            batch=batch,
            steps=4,
            noise_scale=args.noise_scale,
            collect_core_io=False,
            replacements=replacements,
        )
        mask = success_mask(logits, batch["labels"])
        if rank == 0:
            print(f"[eval] batch_size={batch['inputs'].shape[0]} correct_batch={int(mask.sum().item())}")
        correct += int(mask.sum().item())
    if rank == 0:
        print(f"[eval] puzzles={total} pass@1={correct / max(1,total):.4f}")

    if rank == 0:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "diffusion": (diffusion.module if isinstance(diffusion, DDP) else diffusion).state_dict(),
                "one_step": (one_step.module if isinstance(one_step, DDP) else one_step).state_dict(),
                "alphas_cumprod": alphas_cumprod.cpu(),
                "steps": global_step,
                "one_step_updates": one_step_updates,
                "correct": total_correct,
            },
            out_dir / "core083_diffusion.pt",
        )
        print(f"[saved] {out_dir / 'core083_diffusion.pt'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Core 083 diffusion + distillation prototype")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--diffusion-data-dir", type=str, default=None, help="Dataset for diffusion training (train split) and baseline eval; defaults to --data-dir.")
    p.add_argument(
        "--distill-data-dir",
        type=str,
        default=None,
        help="Deprecated: streaming distillation uses the diffusion dataset; kept for CLI compatibility.",
    )
    p.add_argument("--eval-data-dir", type=str, default=None, help="Dataset for evaluation (test split); defaults to --diffusion-data-dir.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--target-correct", type=int, default=100_000)
    p.add_argument("--max-steps", type=int, default=20_000)
    p.add_argument("--min-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--alpha-start", type=float, default=0.999, help="Starting alpha_cumprod for diffusion schedule.")
    p.add_argument("--alpha-end", type=float, default=0.001, help="Ending alpha_cumprod for diffusion schedule.")
    p.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--diffusion-layers", type=int, default=4)
    p.add_argument("--one-step-layers", type=int, default=4)
    p.add_argument(
        "--one-step-epochs",
        type=int,
        default=3,
        help="Deprecated: one-step is trained streaming; value ignored.",
    )
    p.add_argument(
        "--distill-samples",
        type=int,
        default=2048,
        help="Deprecated: streaming mode performs one-step updates every batch.",
    )
    p.add_argument("--eval-puzzles", type=int, default=2048)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--load-diffusion", type=str, default=None, help="Optional diffusion checkpoint to load and skip diffusion training.")
    p.add_argument("--train-diffusion", action="store_true", help="If set with --load-diffusion, continue training the diffusion model instead of skipping.")
    p.add_argument("--noise-scale", type=float, default=0.02, help="RMS-scaled noise std for teacher streaming_forward.")
    p.add_argument(
        "--correct-ckpt-interval",
        type=int,
        default=DEFAULT_CORRECT_CHECKPOINT_INTERVAL,
        help="Save checkpoints every this many correct puzzles.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
