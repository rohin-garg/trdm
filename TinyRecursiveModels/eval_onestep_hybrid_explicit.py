#!/usr/bin/env python
"""
Minimal explicit hybrid eval:
  - Load teacher checkpoint for everything EXCEPT the core stack.
  - Load one-step core weights separately.
  - During streaming_forward, swap EVERY core with the one-step core, sampling fresh eps ~ N(0, I).
  - No teacher core weights are used (LM head, embeddings, inits are from teacher).

Example:
python TinyRecursiveModels/eval_onestep_hybrid_explicit.py \
  --teacher-checkpoint TinyRecursiveModels/checkpoints/trm_sudoku_att_step21700.pt \
  --onestep-checkpoint /home/lerchen/orcd/pool/trdm/analysis/core_all_onestep/core_all_one_step_final.pt \
  --data-dir TinyRecursiveModels/data/sudoku-extreme-100k \
  --batch-size 512 --eval-puzzles 10000 --steps 4 --noise-scale 0.2
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import torch

# Disable torch.compile in pretrain.py
os.environ.setdefault("DISABLE_COMPILE", "1")
# Mock adam_atan2_backend if compiled extension is unavailable
sys.modules["adam_atan2_backend"] = types.SimpleNamespace(adam_atan2_cuda_impl_=lambda *args, **kwargs: None)

ROOT = Path(__file__).resolve().parent
sys.path.append(ROOT.as_posix())

from streaming_core_train import build_teacher, _build_pretrain_config, _load_arch_config, success_mask  # type: ignore  # noqa: E402
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1Block  # type: ignore  # noqa: E402
from models.layers import RotaryEmbedding, rms_norm  # type: ignore  # noqa: E402
from pretrain import create_dataloader  # type: ignore  # noqa: E402


class _CoreStack(torch.nn.Module):
    def __init__(self, *, hidden_dim, num_heads, expansion, num_layers, rms_norm_eps, rope_theta, seq_len):
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
        self.layers = torch.nn.ModuleList([TinyRecursiveReasoningModel_ACTV1Block(cfg) for _ in range(num_layers)])
        self.rotary_emb = RotaryEmbedding(dim=hidden_dim // num_heads, max_position_embeddings=seq_len, base=rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos_sin = self.rotary_emb()
        h = x
        for layer in self.layers:
            h = layer(cos_sin=cos_sin, hidden_states=h)
        return h


class OneStepCoreModel(torch.nn.Module):
    def __init__(self, seq_len_clean: int, hidden_dim: int, num_layers: int, num_heads: int, expansion: float, rms_norm_eps: float, rope_theta: float):
        super().__init__()
        self.seq_len_clean = seq_len_clean
        total_len = seq_len_clean * 2
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
        self.out_proj = torch.nn.Linear(total_len, self.seq_len_clean)

    def forward(self, x_clean: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([x_clean, eps], dim=1)
        h = self.stack(concat)
        h_perm = h.permute(0, 2, 1)
        h_proj = self.out_proj(h_perm)
        out = h_proj.permute(0, 2, 1)
        out = rms_norm(out, variance_epsilon=self.norm_eps)
        return out

    @property
    def seq_len_clean(self) -> int:
        return self._seq_len_clean

    @seq_len_clean.setter
    def seq_len_clean(self, v: int) -> None:
        self._seq_len_clean = v


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explicit hybrid eval (teacher everything except core replaced by one-step)")
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--onestep-checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-puzzles", type=int, default=10000)
    p.add_argument("--noise-scale", type=float, default=0.2)
    p.add_argument("--steps", type=int, default=4)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Teacher (embeddings, heads, inits)
    teacher, _, _ = build_teacher(
        checkpoint=args.teacher_checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        rank=0,
        world_size=1,
        device=device,
    )
    seq_len = teacher.inner.config.seq_len + teacher.inner.puzzle_emb_len  # type: ignore[attr-defined]
    hidden_dim = teacher.inner.config.hidden_size  # type: ignore[attr-defined]
    core_layers = teacher.inner.config.L_layers  # type: ignore[attr-defined]
    num_cores = args.steps * teacher.inner.config.H_cycles * (teacher.inner.config.L_cycles + 1)  # type: ignore[attr-defined]

    # One-step core weights (explicit)
    one_step = OneStepCoreModel(
        seq_len_clean=seq_len,
        hidden_dim=hidden_dim,
        num_layers=core_layers,
        num_heads=teacher.inner.config.num_heads,  # type: ignore[attr-defined]
        expansion=teacher.inner.config.expansion,  # type: ignore[attr-defined]
        rms_norm_eps=teacher.inner.config.rms_norm_eps,  # type: ignore[attr-defined]
        rope_theta=teacher.inner.config.rope_theta,  # type: ignore[attr-defined]
    ).to(device)
    state = torch.load(args.onestep_checkpoint, map_location=device)
    one_step.load_state_dict(state["one_step"])
    one_step.eval()

    # Eval loader
    cfg = _build_pretrain_config(_load_arch_config(), args.data_dir, args.batch_size, args.teacher_checkpoint)
    eval_loader, _ = create_dataloader(
        cfg,
        split="test",
        rank=0,
        world_size=1,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=cfg.global_batch_size,
    )

    @torch.no_grad()
    def replacement_fn(x_clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # Sample fresh eps ~ N(0, I). Do NOT use teacher core weights.
        eps = torch.randn_like(x_clean)
        return one_step(x_clean.to(torch.float32), eps)

    replacements = {idx: replacement_fn for idx in range(num_cores)}

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
            noise_scale=0.0,  # cores are replaced, so disable teacher noise
            collect_core_io=False,
            replacements=replacements,
        )
        mask = success_mask(logits, batch["labels"])
        correct += int(mask.sum().item())
        print(f"[eval] total={total} correct_batch={int(mask.sum().item())}")

    print(f"[eval] puzzles={total} pass@1={correct / max(1,total):.4f}")


if __name__ == "__main__":
    main(parse_args())
