import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch

from pretrain import (
    create_dataloader,
    create_model,
)
from train_trm_diffusion import (
    _load_arch_config,
    _build_pretrain_config,
    _to_device,
    _move_carry_to_device,
)


def _compute_norm_stats(matrix: torch.Tensor) -> Dict[str, float]:
    norms = torch.linalg.vector_norm(matrix, dim=1)
    return {
        "mean": norms.mean().item(),
        "std": norms.std(unbiased=False).item(),
        "min": norms.min().item(),
        "max": norms.max().item(),
    }


def _compute_pca(matrix: torch.Tensor, k: int = 8) -> Dict[str, List[float]]:
    if matrix.size(0) < 2:
        return {"singular_values": [], "explained_variance": []}
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    q = min(k, matrix.size(0), matrix.size(1))
    if q < 1:
        return {"singular_values": [], "explained_variance": []}
    _, s, _ = torch.pca_lowrank(matrix, q=q)
    var = (s ** 2) / (matrix.size(0) - 1)
    total_var = var.sum().item()
    explained = (var / var.sum()).tolist() if total_var > 0 else [0.0 for _ in var]
    return {
        "singular_values": s.tolist(),
        "explained_variance": explained,
    }


def _trim_stack(tensors: List[torch.Tensor], limit: int) -> torch.Tensor:
    stacked = torch.cat(tensors, dim=0)
    if stacked.size(0) > limit:
        stacked = stacked[:limit]
    return stacked


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract TRM latents/logits and compute statistics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("DISABLE_COMPILE", "1")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    arch_cfg = _load_arch_config()
    pretrain_cfg = _build_pretrain_config(args.data_dir, args.batch_size, args.checkpoint, arch_cfg)

    dataloader, metadata = create_dataloader(
        pretrain_cfg,
        split="train",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=pretrain_cfg.global_batch_size,
        rank=0,
        world_size=1,
    )

    model, _, _ = create_model(pretrain_cfg, metadata, rank=0, world_size=1)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    prefix_len = model.model.inner.puzzle_emb_len  # type: ignore[attr-defined]

    gathered_h: List[torch.Tensor] = []
    gathered_l: List[torch.Tensor] = []
    gathered_logits: List[torch.Tensor] = []

    samples_collected = 0
    return_keys = ("logits", "z_H", "z_L", "preds")

    for _set_name, batch, _ in dataloader:
        batch_gpu = _to_device(batch, device)
        carry = model.initial_carry(batch_gpu)
        carry = _move_carry_to_device(carry, device)
        outputs = None
        iteration = 0
        while True:
            iteration += 1
            with torch.no_grad():
                carry, _loss, _metrics, outputs, all_done = model(
                    return_keys=return_keys,
                    carry=carry,
                    batch=batch_gpu,
                    return_latents=True,
                )
            if args.max_steps is not None and iteration >= args.max_steps:
                all_done = True
            if all_done:
                break

        assert outputs is not None
        z_h = carry.inner_carry.z_H  # type: ignore[attr-defined]
        z_l = carry.inner_carry.z_L  # type: ignore[attr-defined]
        logits = outputs["logits"]

        z_h = z_h.to(torch.float32).detach().cpu()
        z_l = z_l.to(torch.float32).detach().cpu()
        logits = logits.to(torch.float32).detach().cpu()

        if prefix_len > 0:
            z_h = z_h[:, prefix_len:, :]
            z_l = z_l[:, prefix_len:, :]

        gathered_h.append(z_h)
        gathered_l.append(z_l)
        gathered_logits.append(logits)

        samples_collected += z_h.size(0)
        if samples_collected >= args.num_samples:
            break

    if samples_collected == 0:
        raise RuntimeError("No samples collected for analysis.")

    z_h_all = _trim_stack(gathered_h, args.num_samples)
    z_l_all = _trim_stack(gathered_l, args.num_samples)
    logits_all = _trim_stack(gathered_logits, args.num_samples)

    z_h_flat = z_h_all.view(z_h_all.size(0), -1)
    z_l_flat = z_l_all.view(z_l_all.size(0), -1)
    logits_flat = logits_all.view(logits_all.size(0), -1)

    summary = {
        "num_samples": int(z_h_flat.size(0)),
        "z_H": {
            "shape_per_sample": list(z_h_all.shape[1:]),
            "norms": _compute_norm_stats(z_h_flat),
            "pca": _compute_pca(z_h_flat, k=8),
        },
        "z_L": {
            "shape_per_sample": list(z_l_all.shape[1:]),
            "norms": _compute_norm_stats(z_l_flat),
            "pca": _compute_pca(z_l_flat, k=8),
        },
        "logits": {
            "shape_per_sample": list(logits_all.shape[1:]),
            "norms": _compute_norm_stats(logits_flat),
            "pca": _compute_pca(logits_flat, k=8),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
