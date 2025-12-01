#!/usr/bin/env python
"""
Pairwise L2 diagnostics for core083 variants on a fresh batch of successes:
 1) mean ||noisy_core||_2
 2) mean ||raw_core||_2
 3) mean ||noisy_core - raw_core||_2
 4) mean ||noisy_core - diffusion_core||_2 (DDIM)
 5) mean ||noisy_core - onestep_core||_2
 6) mean ||diffusion_core - onestep_core||_2

Requires: teacher ckpt/data, diffusion ckpt with both diffusion/one_step weights.
"""
from __future__ import annotations
import argparse
import torch
from streaming_core_train import build_teacher, success_mask
from core083_diffusion_train import CoreDiffusionModel, OneStepCoreModel, ddim_sample

def core_forward_block(layers, x_input, cos_sin=None):
    h = x_input
    for layer in layers:
        h = layer(cos_sin=cos_sin, hidden_states=h)
    return h

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher, loader, _ = build_teacher(
        checkpoint=args.teacher_ckpt,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        rank=0,
        world_size=1,
        device=device,
    )
    seq_len = teacher.inner.config.seq_len + teacher.inner.puzzle_emb_len  # type: ignore[attr-defined]
    hidden_dim = teacher.inner.config.hidden_size  # type: ignore[attr-defined]

    state = torch.load(args.diffusion_ckpt, map_location=device)
    alphas_cumprod = state["alphas_cumprod"].to(device)
    diffusion = CoreDiffusionModel(
        seq_len_clean=seq_len,
        hidden_dim=hidden_dim,
        num_layers=teacher.inner.config.L_layers,  # type: ignore[attr-defined]
        num_heads=teacher.inner.config.num_heads,  # type: ignore[attr-defined]
        expansion=teacher.inner.config.expansion,  # type: ignore[attr-defined]
        rms_norm_eps=teacher.inner.config.rms_norm_eps,  # type: ignore[attr-defined]
        rope_theta=teacher.inner.config.rope_theta,  # type: ignore[attr-defined]
        num_steps=alphas_cumprod.numel(),
    ).to(device)
    diffusion.load_state_dict(state["diffusion"])
    diffusion.eval()

    one_step = OneStepCoreModel(
        seq_len_clean=seq_len,
        hidden_dim=hidden_dim,
        num_layers=teacher.inner.config.L_layers,  # type: ignore[attr-defined]
        num_heads=teacher.inner.config.num_heads,  # type: ignore[attr-defined]
        expansion=teacher.inner.config.expansion,  # type: ignore[attr-defined]
        rms_norm_eps=teacher.inner.config.rms_norm_eps,  # type: ignore[attr-defined]
        rope_theta=teacher.inner.config.rope_theta,  # type: ignore[attr-defined]
    ).to(device)
    one_step.load_state_dict(state["one_step"])
    one_step.eval()

    _, batch, _ = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        logits, records = teacher.streaming_forward(
            batch=batch,
            steps=4,
            noise_scale=args.noise_scale,
            collect_core_io=True,
            replacements=None,
        )
    mask = success_mask(logits, batch["labels"])
    core_records = [(ci, xc, out) for ci, xc, _noise, out in records if ci == 83]
    if not core_records:
        print("no core083 records")
        return
    _, x_clean, noisy_out = core_records[0]
    cos_sin = teacher.inner.rotary_emb() if hasattr(teacher.inner, "rotary_emb") else None  # type: ignore[attr-defined]
    with torch.no_grad():
        raw_out = core_forward_block(teacher.inner.L_level.layers, x_clean, cos_sin=cos_sin)  # type: ignore[attr-defined]

    x = x_clean[mask].to(torch.float32)
    noisy_out = noisy_out[mask].to(torch.float32)
    raw_out = raw_out[mask].to(torch.float32)
    if x.numel() == 0:
        print("no successes")
        return

    noise_init = torch.randn_like(raw_out)
    with torch.no_grad():
        diff_out = ddim_sample(diffusion, x, noise_init, alphas_cumprod)
        one_out = one_step(x, noise_init)

    def mean_l2(t):
        return torch.sqrt(torch.sum(t * t, dim=(1, 2))).mean().item()

    def mean_l2_diff(a, b):
        return torch.sqrt(torch.sum((a - b) ** 2, dim=(1, 2))).mean().item()

    print({
        "num_success": int(mask.sum()),
        "mean_l2(noisy_core)": mean_l2(noisy_out),
        "mean_l2(raw_core)": mean_l2(raw_out),
        "mean_l2(noisy - raw)": mean_l2_diff(noisy_out, raw_out),
        "mean_l2(noisy - diffusion)": mean_l2_diff(noisy_out, diff_out),
        "mean_l2(noisy - one_step)": mean_l2_diff(noisy_out, one_out),
        "mean_l2(diffusion - one_step)": mean_l2_diff(diff_out, one_out),
    })

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--diffusion-ckpt", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--noise-scale", type=float, default=0.02)
    args = p.parse_args()
    main(args)
