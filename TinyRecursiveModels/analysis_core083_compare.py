#!/usr/bin/env python
"""
sanity 

Distribution comparison for core 083:
  * Grab one noisy teacher forward to get x_clean at core 083.
  * Resample the teacher core 083 multiple times with fresh noise (conditional on that x_clean).
  * Sample DDIM from the diffusion model multiple times for the same x_clean.
  * Report MSEs vs raw core output, mean differences, and simple PCA variance.
Single-GPU sanity probe.
"""
from __future__ import annotations
import argparse
import torch
from streaming_core_train import build_teacher
from core083_diffusion_train import CoreDiffusionModel, make_ddim_schedule, ddim_sample

def core_forward_block(layers, x_input, cos_sin=None):
    h = x_input
    for layer in layers:
        h = layer(cos_sin=cos_sin, hidden_states=h)
    return h


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher, loader, _ = build_teacher(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        rank=0,
        world_size=1,
        device=device,
    )
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
    core_records = [(ci, xc, out) for ci, xc, _noise, out in records if ci == 83]
    if not core_records:
        print("No core 083 record found")
        return
    _, x_clean, target_noisy = core_records[0]
    x_clean = x_clean.to(torch.float32)
    target_noisy = target_noisy.to(torch.float32)
    cos_sin = teacher.inner.rotary_emb() if hasattr(teacher.inner, "rotary_emb") else None  # type: ignore[attr-defined]
    target_raw = core_forward_block(teacher.inner.L_level.layers, x_clean, cos_sin=cos_sin)  # type: ignore[attr-defined]

    # Teacher resamples (fresh noise per sample, conditional on x_clean)
    teacher_samples = []
    for _ in range(args.num_samples):
        rms = torch.sqrt(torch.mean(x_clean ** 2, dim=(1, 2), keepdim=True) + 1e-6)
        noise = torch.randn_like(x_clean) * (args.noise_scale * rms)
        x_noisy = x_clean + noise
        out = core_forward_block(teacher.inner.L_level.layers, x_noisy, cos_sin=cos_sin)  # type: ignore[attr-defined]
        teacher_samples.append(out.detach())
    teacher_samples = torch.stack(teacher_samples, dim=0)

    # Load diffusion
    state = torch.load(args.diffusion_ckpt, map_location=device)
    alphas_cumprod = state.get(
        "alphas_cumprod",
        make_ddim_schedule(args.diffusion_steps, args.schedule, args.alpha_start, args.alpha_end),
    ).to(device)
    seq_len = teacher.inner.config.seq_len + teacher.inner.puzzle_emb_len  # type: ignore[attr-defined]
    hidden_dim = teacher.inner.config.hidden_size  # type: ignore[attr-defined]
    diffusion = CoreDiffusionModel(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=args.diffusion_layers,
        num_heads=teacher.inner.config.num_heads,  # type: ignore[attr-defined]
        expansion=teacher.inner.config.expansion,  # type: ignore[attr-defined]
        num_steps=alphas_cumprod.numel(),
    ).to(device)
    diffusion.load_state_dict(state.get("diffusion", state))
    diffusion.eval()

    diff_samples = []
    for _ in range(args.num_samples):
        noise_init = torch.randn_like(x_clean)
        with torch.no_grad():
            out = ddim_sample(diffusion, x_clean, noise_init, alphas_cumprod)
        diff_samples.append(out.detach())
    diff_samples = torch.stack(diff_samples, dim=0)

    teacher_mean = teacher_samples.mean(dim=0)
    diff_mean = diff_samples.mean(dim=0)
    mse_ddim_vs_raw = torch.mean((diff_mean - target_raw) ** 2).item()
    mse_teacher_vs_raw = torch.mean((teacher_mean - target_raw) ** 2).item()
    mse_ddim_vs_teacher_mean = torch.mean((diff_mean - teacher_mean) ** 2).item()

    all_samples = torch.cat([teacher_samples, diff_samples], dim=0)
    flat = all_samples.reshape(all_samples.shape[0], -1)
    flat = flat - flat.mean(dim=0, keepdim=True)
    u, s, v = torch.svd_lowrank(flat, q=min(3, flat.shape[1]))
    var_explained = (s ** 2) / (flat.shape[0] - 1)

    print({
        "mse_ddim_vs_raw": mse_ddim_vs_raw,
        "mse_teacher_mean_vs_raw": mse_teacher_vs_raw,
        "mse_ddim_vs_teacher_mean": mse_ddim_vs_teacher_mean,
        "pca_var_explained_top3": var_explained[:3].tolist(),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--diffusion-ckpt", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--noise-scale", type=float, default=0.02)
    p.add_argument("--diffusion-steps", type=int, default=20)
    p.add_argument("--diffusion-layers", type=int, default=4)
    p.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--alpha-start", type=float, default=0.9)
    p.add_argument("--alpha-end", type=float, default=0.1)
    args = p.parse_args()
    run(args)
