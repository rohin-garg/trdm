"""
deprecated

Utilities for streaming TRM trajectories directly into diffusion training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch import Tensor

from pretrain import (
    ArchConfig,
    LossConfig,
    PretrainConfig,
    create_dataloader,
    create_model,
)

# Type aliases
LayerExample = Dict[str, Tensor]
HCycleExample = Dict[str, Tensor]


def _load_arch_config() -> ArchConfig:
    """Load the TRM architecture config used for trajectory capture."""
    import yaml  # Local import to avoid global dependency when unused

    from pathlib import Path

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


@dataclass
class TrajectoryStreamerConfig:
    checkpoint_path: str
    data_dir: str
    global_batch_size: int
    device: torch.device
    rank: int
    world_size: int
    max_batches_per_epoch: Optional[int] = None
    dtype: torch.dtype = torch.float32
    split: str = "train"


class _BaseTrajectoryStreamer:
    """Base class that lazily yields TRM trajectories without persisting to disk."""

    def __init__(self, config: TrajectoryStreamerConfig):
        self.config = config
        self._arch_cfg: Optional[ArchConfig] = None
        self._pretrain_cfg: Optional[PretrainConfig] = None
        self._loss_wrapper: Optional[torch.nn.Module] = None
        self._model: Optional[torch.nn.Module] = None
        self._metadata = None
        self._dataloader = None
        self._loader_iter: Optional[Iterator] = None
        self._buffer: List[Dict[str, Tensor]] = []
        self._batches_consumed = 0

    @property
    def metadata(self):
        if self._metadata is None:
            raise RuntimeError("Metadata is not available before the first batch.")
        return self._metadata

    def _ensure_model_and_loader(self) -> None:
        if self._arch_cfg is None or self._pretrain_cfg is None:
            os.environ.setdefault("DISABLE_COMPILE", "1")
            arch_cfg = _load_arch_config()
            pretrain_cfg = _build_pretrain_config(
                arch_cfg,
                data_path=self.config.data_dir,
                global_batch_size=self.config.global_batch_size,
                checkpoint=self.config.checkpoint_path,
            )
            self._arch_cfg = arch_cfg
            self._pretrain_cfg = pretrain_cfg

        if self._dataloader is None or self._loader_iter is None:
            assert self._pretrain_cfg is not None
            loader, metadata = create_dataloader(
                self._pretrain_cfg,
                self.config.split,
                rank=self.config.rank,
                world_size=self.config.world_size,
                test_set_mode=True,
                epochs_per_iter=1,
                global_batch_size=self._pretrain_cfg.global_batch_size,
            )
            self._metadata = metadata
            self._dataloader = loader
            self._loader_iter = iter(loader)
            self._batches_consumed = 0

        if self._model is None:
            assert self._pretrain_cfg is not None
            assert self._metadata is not None
            loss_wrapper, _, _ = create_model(
                self._pretrain_cfg,
                self._metadata,
                rank=self.config.rank,
                world_size=self.config.world_size,
            )
            loss_wrapper = loss_wrapper.to(self.config.device)
            loss_wrapper.eval()
            for param in loss_wrapper.parameters():
                param.requires_grad_(False)
            self._loss_wrapper = loss_wrapper
            model = loss_wrapper.model  # type: ignore[attr-defined]
            model = model.to(self.config.device)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self._model = model

    def _reset_loader(self) -> None:
        self._loader_iter = None
        self._dataloader = None
        self._batches_consumed = 0

    def _next_trajectory_batch(self, auto_reset: bool = True) -> Dict[str, Tensor]:
        self._ensure_model_and_loader()
        assert self._loader_iter is not None
        assert self._model is not None

        if (
            not auto_reset
            and self.config.max_batches_per_epoch is not None
            and self._batches_consumed >= self.config.max_batches_per_epoch
        ):
            raise StopIteration

        if (
            auto_reset
            and self.config.max_batches_per_epoch is not None
            and self._batches_consumed >= self.config.max_batches_per_epoch
        ):
            self._reset_loader()
            self._ensure_model_and_loader()

        try:
            _, batch, _ = next(self._loader_iter)
        except StopIteration:
            if not auto_reset:
                self._reset_loader()
                raise
            self._reset_loader()
            self._ensure_model_and_loader()
            _, batch, _ = next(self._loader_iter)

        self._batches_consumed += 1

        device_batch = {k: v.to(self.config.device, non_blocking=True) for k, v in batch.items()}
        model = self._model

        with torch.no_grad():
            carry = model.initial_carry(device_batch)
            carry.inner_carry.z_H = carry.inner_carry.z_H.to(self.config.device)
            carry.inner_carry.z_L = carry.inner_carry.z_L.to(self.config.device)
            carry.steps = carry.steps.to(self.config.device)
            carry.halted = carry.halted.to(self.config.device)
            carry.current_data = {k: v.to(self.config.device) for k, v in carry.current_data.items()}

            carry, outputs = model(
                carry=carry,
                batch=device_batch,
                store_diffusion_trajectory=True,
            )

        trajectory = outputs["trajectory"]

        cpu_trajectory = {
            "all_h_cycles": [
                {
                    "z_L_list": [z.to(self.config.dtype).cpu() for z in h_cycle["z_L_list"]],
                    "z_H": h_cycle["z_H"].to(self.config.dtype).cpu(),
                    "input_embeddings": h_cycle["input_embeddings"].to(self.config.dtype).cpu(),
                }
                for h_cycle in trajectory["all_h_cycles"]
            ],
            "initial_z_H": trajectory["initial_z_H"].to(self.config.dtype).cpu(),
            "initial_z_L": trajectory["initial_z_L"].to(self.config.dtype).cpu(),
        }

        return cpu_trajectory

    def _examples_from_batch(self, batch: Dict[str, Tensor]) -> List[Dict[str, Tensor]]:
        raise NotImplementedError

    def fetch_examples(self, num_examples: int) -> List[Dict[str, Tensor]]:
        while len(self._buffer) < num_examples:
            batch = self._next_trajectory_batch()
            self._buffer.extend(self._examples_from_batch(batch))
        examples = self._buffer[:num_examples]
        self._buffer = self._buffer[num_examples:]
        return examples

    def reset(self) -> None:
        """Clear buffers and restart the underlying dataloader iterator."""
        self._buffer.clear()
        self._reset_loader()

    def iter_epoch(self, max_batches: Optional[int] = None) -> Iterator[List[Dict[str, Tensor]]]:
        """Yield flattened examples for each TRM batch without auto-wrapping."""
        self.reset()
        batches_yielded = 0
        while max_batches is None or batches_yielded < max_batches:
            try:
                batch = self._next_trajectory_batch(auto_reset=False)
            except StopIteration:
                break
            batches_yielded += 1
            yield self._examples_from_batch(batch)


class LayerUpdateStreamer(_BaseTrajectoryStreamer):
    """Produces per-sample TRM layer update tuples for diffusion-916 training."""

    def _examples_from_batch(self, batch: Dict[str, Tensor]) -> List[LayerExample]:
        seq_examples: List[LayerExample] = []
        initial_z_H = batch["initial_z_H"]
        initial_z_L = batch["initial_z_L"]

        for h_idx, h_cycle in enumerate(batch["all_h_cycles"]):
            z_L_list = h_cycle["z_L_list"]
            z_H = h_cycle["z_H"]
            x_embed = h_cycle["input_embeddings"]

            if h_idx == 0:
                prev_z_H = initial_z_H
                prev_z_L = initial_z_L
            else:
                prev_cycle = batch["all_h_cycles"][h_idx - 1]
                prev_z_H = prev_cycle["z_H"]
                prev_z_L = prev_cycle["z_L_list"][-1]

            batch_size = z_H.shape[0]
            for sample_idx in range(batch_size):
                sample_prev_z_L = prev_z_L[sample_idx]
                sample_prev_z_H = prev_z_H[sample_idx]
                sample_x = x_embed[sample_idx]

                running_prev = sample_prev_z_L
                for z_next in z_L_list:
                    sample_z_next = z_next[sample_idx]
                    seq_examples.append(
                        {
                            "target": sample_z_next.clone(),
                            "inputs": (
                                running_prev.clone(),
                                sample_prev_z_H.clone(),
                                sample_x.clone(),
                            ),
                            "type": torch.tensor(0, dtype=torch.int8),
                        }
                    )
                    running_prev = sample_z_next.clone()

                seq_examples.append(
                    {
                        "target": z_H[sample_idx].clone(),
                        "inputs": (
                            sample_prev_z_H.clone(),
                            z_L_list[-1][sample_idx].clone(),
                        ),
                        "type": torch.tensor(1, dtype=torch.int8),
                    }
                )

        return seq_examples


class HCycleStreamer(_BaseTrajectoryStreamer):
    """Produces per-sample H-cycle trajectory tensors for diffusion-9160 training."""

    def _examples_from_batch(self, batch: Dict[str, Tensor]) -> List[HCycleExample]:
        seq_examples: List[HCycleExample] = []
        initial_z_H = batch["initial_z_H"]
        initial_z_L = batch["initial_z_L"]

        for h_idx, h_cycle in enumerate(batch["all_h_cycles"]):
            z_L_list = h_cycle["z_L_list"]
            z_H = h_cycle["z_H"]
            x_embed = h_cycle["input_embeddings"]

            if h_idx == 0:
                prev_z_H = initial_z_H
                prev_z_L = initial_z_L
            else:
                prev_cycle = batch["all_h_cycles"][h_idx - 1]
                prev_z_H = prev_cycle["z_H"]
                prev_z_L = prev_cycle["z_L_list"][-1]

            batch_size = z_H.shape[0]
            for sample_idx in range(batch_size):
                sample_x = x_embed[sample_idx].clone()
                sample_prev_z_H = prev_z_H[sample_idx].clone()
                sample_prev_z_L = prev_z_L[sample_idx].clone()
                z_stack = torch.stack([z[sample_idx].clone() for z in z_L_list] + [z_H[sample_idx].clone()])
                seq_examples.append(
                    {
                        "target": z_stack,
                        "x": sample_x,
                        "y_init": sample_prev_z_H,
                        "z_init": sample_prev_z_L,
                    }
                )

        return seq_examples


def collate_layer_examples(batch: List[LayerExample]) -> Tuple[Tuple[Tensor, ...], Tensor, Tensor]:
    """Collate function for diffusion-916 training batches."""
    targets = torch.stack([item["target"] for item in batch], dim=0)
    max_len = max(len(item["inputs"]) for item in batch)
    padded_inputs: List[List[Tensor]] = []
    for item in batch:
        inputs_list = list(item["inputs"])
        if len(inputs_list) < max_len:
            pad_tensor = torch.zeros_like(inputs_list[0])
            inputs_list.extend([pad_tensor.clone() for _ in range(max_len - len(inputs_list))])
        padded_inputs.append(inputs_list)
    stacked = [torch.stack([inputs[i] for inputs in padded_inputs], dim=0) for i in range(max_len)]
    inputs = tuple(stacked)
    types = torch.stack([item["type"] for item in batch], dim=0)
    return inputs, targets, types


def collate_hcycle_examples(batch: List[HCycleExample]) -> Tuple[Tuple[Tensor, ...], Tensor]:
    """Collate function for diffusion-9160 training batches."""
    targets = torch.stack([item["target"] for item in batch], dim=0)
    x = torch.stack([item["x"] for item in batch], dim=0)
    y_init = torch.stack([item["y_init"] for item in batch], dim=0)
    z_init = torch.stack([item["z_init"] for item in batch], dim=0)
    return (x, y_init, z_init), targets
