#!/usr/bin/env python
"""
MAIN LOGIC

Streaming one-step denoiser training for all 84 decoupled TRM cores.

This keeps everything on-GPU:
  * Load the HF Sudoku checkpoint on each GPU.
  * For each batch, run halting-off inference for 4 steps; each step has 21 cores
    (H_cycles=3, L_cycles=6), totaling 84 distinct cores. Add 2% RMS noise to the
    post-injection activations of every core.
  * Keep only puzzles solved correctly (rejection sampling).
  * Train 84 independent one-step denoisers jointly (single backward pass).
  * Repeat until the requested number of (puzzle, core) samples is reached.
  * Evaluate every `eval_interval` samples on 8,192 held-out puzzles.

Example launch (4×H200):
torchrun --standalone --nproc_per_node=4 TinyRecursiveModels/streaming_core_train.py \
  --checkpoint activationanalysis/checkpoints/trm_sudoku_att_step21700.pt \
  --data-dir TinyRecursiveModels/data/sudoku-extreme-1k-aug-1000 \
  --output-dir trdm/analysis/streaming_run1 \
  --total-samples 10000000 \
  --steps 4 --batch-size 64 --eval-interval 500000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from models.core_denoiser import CoreDenoiserSet
from pretrain import (
    ArchConfig,
    LossConfig,
    PretrainConfig,
    create_dataloader,
    create_model,
)
import yaml


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


def success_mask(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    valid = labels != -100
    equal = torch.where(valid, preds == labels, torch.ones_like(preds, dtype=torch.bool))
    return equal.all(dim=1)


def build_teacher(checkpoint: str, data_dir: str, batch_size: int, rank: int, world_size: int, device: torch.device):
    arch_cfg = _load_arch_config()
    cfg = _build_pretrain_config(
        arch_cfg=arch_cfg,
        data_path=data_dir,
        global_batch_size=batch_size * world_size,
        checkpoint=checkpoint,
    )
    loader, metadata = create_dataloader(
        cfg,
        split="train",
        rank=rank,
        world_size=world_size,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=cfg.global_batch_size,
    )
    loss_wrapper, _, _ = create_model(cfg, metadata, rank=rank, world_size=world_size)
    loss_wrapper = loss_wrapper.to(device)
    loss_wrapper.eval()
    for p in loss_wrapper.parameters():
        p.requires_grad_(False)
    model = loss_wrapper.model  # type: ignore[attr-defined]
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, loader, metadata


def make_replacements(core_model: CoreDenoiserSet):
    def _fn(core_idx: int):
        def _apply(x_clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            # Denoiser consumes clean hidden+injection plus the sampled noise.
            return core_model.forward_core(core_idx, x_clean.to(torch.float32), noise.to(torch.float32))

        return _apply

    return {idx: _fn(idx) for idx in range(len(core_model))}


@torch.no_grad()
def run_eval(
    teacher,
    denoisers: Optional[CoreDenoiserSet],
    loader,
    device: torch.device,
    steps: int,
    noise_scale: float,
    max_puzzles: int,
) -> Dict[str, float]:
    total = 0
    correct = 0
    it = iter(loader)
    replacements = make_replacements(denoisers) if denoisers is not None else None
    while total < max_puzzles:
        try:
            _, batch, _ = next(it)
        except StopIteration:
            it = iter(loader)
            _, batch, _ = next(it)
        take = min(batch["inputs"].shape[0], max_puzzles - total)
        if take < batch["inputs"].shape[0]:
            batch = {k: v[:take].clone() for k, v in batch.items()}
        total += batch["inputs"].shape[0]
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits, _ = teacher.streaming_forward(
            batch=batch,
            steps=steps,
            noise_scale=noise_scale,
            collect_core_io=False,
            replacements=replacements,
        )
        mask = success_mask(logits, batch["labels"])
        correct += int(mask.sum().item())
    acc = correct / max(1, total)
    return {"puzzles": total, "pass@1": acc}


def train_loop(args: argparse.Namespace) -> None:
    rank, world_size, local_rank, device = init_distributed()
    torch.manual_seed(42 + rank)

    teacher, loader, metadata = build_teacher(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    seq_len = teacher.inner.config.seq_len + teacher.inner.puzzle_emb_len  # type: ignore[attr-defined]
    expected_cores = args.steps * teacher.inner.config.H_cycles * (teacher.inner.config.L_cycles + 1)  # type: ignore[attr-defined]
    if args.num_cores != expected_cores and rank == 0:
        print(f"[warn] num_cores={args.num_cores} does not match expected {expected_cores} from config; using provided value.")
    denoisers = CoreDenoiserSet(
        num_cores=args.num_cores,
        seq_len=seq_len,
        hidden_dim=teacher.inner.config.hidden_size,  # type: ignore[attr-defined]
        num_layers=args.core_layers,
        num_heads=teacher.inner.config.num_heads,  # type: ignore[attr-defined]
        expansion=teacher.inner.config.expansion,  # type: ignore[attr-defined]
    ).to(device)
    if world_size > 1:
        denoisers = DDP(denoisers, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(denoisers.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    max_samples = args.total_samples
    eval_interval = args.eval_interval
    sample_counter = 0
    global_step = 0
    loader_it = iter(loader)
    next_eval_sample = args.eval_interval

    os.makedirs(args.output_dir, exist_ok=True)

    if rank == 0:
        baseline = run_eval(
            teacher=teacher,
            denoisers=None,
            loader=loader,
            device=device,
            steps=args.steps,
            noise_scale=args.noise_scale,
            max_puzzles=args.eval_puzzles,
        )
        print(f"[baseline] noisy teacher pass@1={baseline['pass@1']:.4f} puzzles={baseline['puzzles']}")
    if dist.is_initialized():
        dist.barrier()

    while sample_counter < max_samples:
        try:
            _, batch, _ = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            _, batch, _ = next(loader_it)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits, records = teacher.streaming_forward(
            batch=batch,
            steps=args.steps,
            noise_scale=args.noise_scale,
            collect_core_io=True,
            core_idxs=list(range(args.num_cores)),
            replacements=None,
        )
        success = success_mask(logits, batch["labels"])
        success_count = int(success.sum().item())
        if success_count == 0:
            continue

        core_model = denoisers.module if isinstance(denoisers, DDP) else denoisers
        total_loss = None
        used_cores = 0
        for core_idx, x_clean, noise, target_out in records:
            x_sel = (x_clean + noise)[success].to(torch.float32)
            n_sel = noise[success].to(torch.float32)
            t_sel = target_out[success].to(torch.float32)
            if x_sel.numel() == 0:
                continue
            pred = core_model.forward_core(core_idx, x_sel, n_sel)
            loss = F.mse_loss(pred, t_sel)
            total_loss = loss if total_loss is None else total_loss + loss
            used_cores += 1

        if total_loss is None or used_cores == 0:
            continue

        total_loss = total_loss / used_cores
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(denoisers.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        local_samples = success_count * used_cores
        global_increment = local_samples
        if dist.is_initialized():
            sample_tensor = torch.tensor([local_samples], device=device, dtype=torch.long)
            dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
            global_increment = int(sample_tensor.item())
        sample_counter = min(sample_counter + global_increment, max_samples)

        if rank == 0 and global_step % args.log_interval == 0:
            print(f"[train] step={global_step} loss={float(total_loss):.6f} success={success_count} cores={used_cores} samples={sample_counter}/{max_samples}")

        if sample_counter >= max_samples:
            break

        while sample_counter >= next_eval_sample:
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                denoisers.eval()
                metrics = run_eval(
                    teacher=teacher,
                    denoisers=core_model,
                    loader=loader,
                    device=device,
                    steps=args.steps,
                    noise_scale=args.noise_scale,
                    max_puzzles=args.eval_puzzles,
                )
                ckpt = {
                    "denoisers": core_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "samples": sample_counter,
                    "metrics": metrics,
                }
                ckpt_path = Path(args.output_dir) / f"denoisers_step{global_step}.pt"
                torch.save(ckpt, ckpt_path)
                print(f"[eval] step={global_step} samples={sample_counter} pass@1={metrics['pass@1']:.4f} saved={ckpt_path}")
                denoisers.train()
            if dist.is_initialized():
                dist.barrier()
            next_eval_sample += eval_interval

    if rank == 0:
        core_model = denoisers.module if isinstance(denoisers, DDP) else denoisers
        ckpt = {
            "denoisers": core_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "samples": sample_counter,
        }
        final_path = Path(args.output_dir) / "denoisers_final.pt"
        torch.save(ckpt, final_path)
        print(f"[done] saved {final_path} samples={sample_counter}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming TRM core denoiser training (all 84 cores).")
    parser.add_argument("--checkpoint", required=True, help="Path to HF TRM checkpoint.")
    parser.add_argument("--data-dir", required=True, help="Sudoku dataset directory.")
    parser.add_argument("--output-dir", required=True, help="Where to save checkpoints/metrics.")
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size.")
    parser.add_argument("--steps", type=int, default=4, help="Fixed halting steps (21 cores per step -> 84 cores).")
    parser.add_argument("--num-cores", type=int, default=84)
    parser.add_argument("--noise-scale", type=float, default=0.02, help="Noise std = scale * RMS per token.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--core-layers", type=int, default=6, help="Transformer layers per denoiser core.")
    parser.add_argument("--total-samples", type=int, default=10_000_000, help="(puzzle, core) pairs to train on.")
    parser.add_argument("--eval-interval", type=int, default=500_000, help="Evaluate every N samples (k * interval).")
    parser.add_argument("--eval-puzzles", type=int, default=8192)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("DISABLE_COMPILE", "1")
    args = parse_args()
    train_loop(args)

# --- minimal inline helpers from diffusion_streaming.py ---
def _load_arch_config() -> ArchConfig:
    """Load the TRM architecture config used for trajectory capture."""
    cfg_path = Path(__file__).resolve().parent / "config" / "arch" / "trm.yaml"
    with open(cfg_path, "r", encoding="utf-8") as handle:
        arch_dict = yaml.safe_load(handle)
    loss_cfg = LossConfig(**arch_dict.pop("loss"))
    if isinstance(arch_dict.get("puzzle_emb_ndim"), str):
        arch_dict["puzzle_emb_ndim"] = arch_dict["hidden_size"]
    return ArchConfig(loss=loss_cfg, **arch_dict)


def _build_pretrain_config(
    arch_cfg: ArchConfig,
    data_path: str,
    global_batch_size: int,
    checkpoint: str,
) -> PretrainConfig:
    """Create a frozen-pretrain config for running TRM inference."""
    return PretrainConfig(
        arch=arch_cfg,
        data_paths=[data_path],
        data_paths_test=[],
        evaluators=[],
        global_batch_size=global_batch_size,
        epochs=1,
        lr=1e-4,
        lr_min_ratio=1.0,
        lr_warmup_steps=2000,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        puzzle_emb_lr=1e-2,
        puzzle_emb_weight_decay=0.1,
        project_name=None,
        run_name=None,
        load_checkpoint=checkpoint,
        resume_step=None,
        checkpoint_path=None,
        seed=0,
        checkpoint_every_eval=False,
        eval_interval=None,
        min_eval_interval=0,
        eval_save_outputs=[],
        ema=False,
        ema_rate=0.999,
        freeze_weights=True,
        grad_clip_norm=None,
        fullrec_weight_decay=None,
    )
