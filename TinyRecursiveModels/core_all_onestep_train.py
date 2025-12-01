#!/usr/bin/env python
"""
Train a shared one-step core to replace all TRM cores.

Objective: onestep(x_clean, eps) -> teacher_core(x_clean + rms*noise_scale*eps)
where rms is per-sample RMS over x_clean (flattened), eps ~ N(0,I) same shape as x_clean.
[rms is of shape (B)]
No rejection sampling; stream batches from the teacher [size 512], collect per-core IO [get 84*512 (input, output) pairs total], and
update a single shared core across all core indices; train with batch size 16384; do 20 gradient steps for each puzzle. Periodically checkpoint by
samples_seen. At the end, evaluate by swapping all cores with the learned
one-step core.

Base model: TinyRecursiveReasoningModel (TRM) with: 84 cores [4 * 3 * 7 (per H-cycle, we have 6 = arch.L_cycles L-steps and one H-step, and there's 3 = arch.H_cycles H-cycles.)], ACT disabled, 4 fixed steps. Take it from the hf checkpoint "Sanjin2024/TinyRecursiveModels-Sudoku-Extreme-att" (load the weights locally). 
noise scale 0.20.
train on a new dataset with 1e5 base puzzles + 1e3 augmentations per base puzzle (1e8 + 1e5 total puzzles).
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

from streaming_core_train import (
    _build_pretrain_config,
    _load_arch_config,
    build_teacher,
    success_mask,
)
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1Block
from models.layers import RotaryEmbedding, rms_norm
from pretrain import create_dataloader


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


class OneStepCoreModel(nn.Module):
    """One-step core: (x_clean, eps) -> deterministic core output."""

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
        self.seq_len_eps = seq_len_clean
        total_len = self.seq_len_clean + self.seq_len_eps
        self.norm_eps = rms_norm_eps
        self.stack = _CoreStack(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            num_layers=num_layers,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            seq_len=total_len,
        )
        self.out_proj = nn.Linear(total_len, self.seq_len_eps)

    def forward(self, x_clean: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([x_clean, eps], dim=1)  # [B, seq_clean+seq_eps, D]
        h = self.stack(concat)
        h_perm = h.permute(0, 2, 1)
        h_proj = self.out_proj(h_perm)
        out = h_proj.permute(0, 2, 1)
        out = rms_norm(out, variance_epsilon=self.norm_eps)
        return out


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

    if rank == 0:
        print(
            f"[config] data={args.data_dir} eval_data={args.eval_data_dir or args.data_dir} "
            f"noise_scale={args.noise_scale} batch_size={args.batch_size} target_samples={args.target_samples}"
        )

    # Teacher
    teacher, loader, _ = build_teacher(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    seq_len = teacher.inner.config.seq_len + teacher.inner.puzzle_emb_len  # type: ignore[attr-defined]
    hidden_dim = teacher.inner.config.hidden_size  # type: ignore[attr-defined]
    core_layers = teacher.inner.config.L_layers  # type: ignore[attr-defined]
    num_cores = args.steps * teacher.inner.config.H_cycles * (teacher.inner.config.L_cycles + 1)  # type: ignore[attr-defined]
    if rank == 0:
        print(f"[info] training shared one-step for all cores; num_cores={num_cores}")

    # Model
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

    # Training loop
    loader_it = iter(loader)
    samples_seen = 0
    global_step = 0
    next_ckpt = args.sample_ckpt_interval
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print("[info] starting training loop")
    while samples_seen < args.target_samples and global_step < args.max_steps:
        try:
            _, batch, _ = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            _, batch, _ = next(loader_it)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # Single teacher forward (no grad) to collect all cores.
        with torch.no_grad():
            _, records = teacher.streaming_forward(
                batch=batch,
                steps=args.steps,
                noise_scale=args.noise_scale,
                collect_core_io=True,
                core_idxs=list(range(num_cores)),
                replacements=None,
            )

        if not records:
            continue

        # Stack all records into big tensors
        xs: List[torch.Tensor] = []
        eps_list: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for _, x_clean, noise, out in records:
            x = x_clean.to(torch.float32)
            noise = noise.to(torch.float32)
            out_t = out.to(torch.float32)
            rms = torch.sqrt(torch.mean(x ** 2, dim=(1, 2)) + 1e-6)
            eps = noise / (args.noise_scale * rms.view(-1, 1, 1) + 1e-6)
            xs.append(x)
            eps_list.append(eps)
            ys.append(out_t)

        x_all = torch.cat(xs, dim=0)
        eps_all = torch.cat(eps_list, dim=0)
        y_all = torch.cat(ys, dim=0)
        total = x_all.shape[0]

        # Train for multiple epochs over the collected batch
        for epoch in range(args.epochs_per_collect):
            perm = torch.randperm(total, device=device)
            x_shuf = x_all[perm]
            eps_shuf = eps_all[perm]
            y_shuf = y_all[perm]

            for start in range(0, total, args.train_batch_size):
                one_opt.zero_grad(set_to_none=True)
                x_mb = x_shuf[start : start + args.train_batch_size]
                eps_mb = eps_shuf[start : start + args.train_batch_size]
                y_mb = y_shuf[start : start + args.train_batch_size]

                pred = one_step(x_mb, eps_mb)
                loss = F.mse_loss(pred, y_mb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(one_step.parameters(), args.max_grad_norm)
                one_opt.step()

                global_step += 1
                samples_seen += x_mb.shape[0]

                if rank == 0 and global_step % args.log_interval == 0:
                    per_elem = float(loss)
                    print(
                        f"[train-one] step={global_step} samples_seen={samples_seen} "
                        f"loss={float(loss):.6f} loss_per_elem={per_elem:.6e}",
                        flush=True,
                    )

                if samples_seen >= args.target_samples or global_step >= args.max_steps:
                    break
            if samples_seen >= args.target_samples or global_step >= args.max_steps:
                break

        del x_all, eps_all, y_all, xs, eps_list, ys
        torch.cuda.empty_cache()

        if samples_seen >= next_ckpt and rank == 0:
            ckpt_path = out_dir / f"core_all_one_step_samples{samples_seen}_step{global_step}.pt"
            torch.save(
                {
                    "one_step": (one_step.module if isinstance(one_step, DDP) else one_step).state_dict(),
                    "steps": global_step,
                    "samples": samples_seen,
                },
                ckpt_path,
            )
            print(f"[checkpoint] saved {ckpt_path}", flush=True)
            next_ckpt += args.sample_ckpt_interval

    # Eval: replace all cores with one-step
    def replacement_fn(x_clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x_clean.to(torch.float32) ** 2, dim=(1, 2)) + 1e-6)
        eps = noise.to(torch.float32) / (args.noise_scale * rms.view(-1, 1, 1) + 1e-6)
        core = one_step.module if isinstance(one_step, DDP) else one_step
        with torch.no_grad():
            return core(x_clean.to(torch.float32), eps)

    replacements = {idx: replacement_fn for idx in range(num_cores)}
    eval_loader, _ = prepare_loader(
        _build_pretrain_config(_load_arch_config(), args.eval_data_dir or args.data_dir, args.batch_size * world_size, args.checkpoint),
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
        logits, _ = teacher.streaming_forward(
            batch=batch,
            steps=args.steps,
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
        final_path = out_dir / "core_all_one_step_final.pt"
        torch.save(
            {
                "one_step": (one_step.module if isinstance(one_step, DDP) else one_step).state_dict(),
                "steps": global_step,
                "samples": samples_seen,
            },
            final_path,
        )
        print(f"[saved] {final_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shared one-step core training for all TRM cores")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--eval-data-dir", type=str, default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=512, help="Teacher batch size.")
    p.add_argument("--steps", type=int, default=4, help="Number of halting steps in teacher streaming forward.")
    p.add_argument("--target-samples", type=int, default=100_000_000, help="Total (core-level) samples to train on.")
    p.add_argument("--max-steps", type=int, default=2_000_000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--noise-scale", type=float, default=0.20, help="RMS-scaled noise std for teacher streaming_forward.")
    p.add_argument("--sample-ckpt-interval", type=int, default=2_000_000, help="Checkpoint every this many samples.")
    p.add_argument("--eval-puzzles", type=int, default=10_000)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--train-batch-size", type=int, default=1_024, help="Batch size for onestep updates (post-collection).")
    p.add_argument("--epochs-per-collect", type=int, default=10, help="Number of epochs over each collected batch.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
