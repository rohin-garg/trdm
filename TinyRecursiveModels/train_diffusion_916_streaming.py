"""Distributed streaming trainer for the single-layer (916) diffusion model."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data

from diffusion_streaming import (
    LayerUpdateStreamer,
    TrajectoryStreamerConfig,
    collate_layer_examples,
    _load_arch_config,
    _build_pretrain_config,
)
from models.diffusion_916 import SingleLayerDiffusion, DiffusionSchedule
from evaluators.arc import ARC
from pretrain import create_dataloader, create_model
from dataset.common import PuzzleDatasetMetadata
from dataset.build_arc_dataset import inverse_aug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train streaming diffusion-916 with DDP")
    parser.add_argument("--checkpoint-path", required=True, help="Frozen TRM checkpoint path")
    parser.add_argument("--data-dir", required=True, help="ARC dataset directory")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints/logs")
    parser.add_argument("--trm-global-batch-size", type=int, default=96, help="Global batch size fed to TRM inference (across all ranks)")
    parser.add_argument("--train-batch-size", type=int, default=32, help="Diffusion batch size per optimization step")
    parser.add_argument("--num-epochs", type=int, default=50, help="Target number of epochs")
    parser.add_argument("--time-limit-seconds", type=int, default=7200, help="Wall-clock training budget in seconds")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--timesteps", type=int, default=30, help="Diffusion timesteps")
    parser.add_argument("--num-layers", type=int, default=6, help="Diffusion transformer layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Diffusion attention heads")
    parser.add_argument("--log-interval", type=int, default=50, help="Batches between log prints")
    parser.add_argument("--grad-accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--max-trm-batches-per-epoch", type=int, default=None, help="Override TRM batches per epoch for debugging")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16", help="Autocast dtype")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only (skip training)")
    parser.add_argument("--eval-checkpoint", help="Diffusion checkpoint to evaluate")
    parser.add_argument("--eval-trm-checkpoint", help="TRM checkpoint for evaluation (defaults to training checkpoint)")
    parser.add_argument("--eval-data-dir", help="Dataset directory for evaluation (defaults to --data-dir)")
    parser.add_argument("--eval-output-path", help="JSON path to store evaluation metrics")
    parser.add_argument("--eval-num-samples", type=int, default=1, help="Number of diffusion samples per augmented puzzle")
    parser.add_argument("--eval-trm-batch-size", type=int, default=96, help="Global TRM batch size during evaluation")
    parser.add_argument("--eval-pass-ks", type=int, nargs="+", default=[1, 2], help="pass@k values to compute")
    parser.add_argument("--eval-submission-k", type=int, default=2, help="Number of predictions to submit per puzzle")
    parser.add_argument("--eval-device", default="cuda:0", help="Device to run evaluation on")
    parser.add_argument("--eval-after-train", action="store_true", help="Run evaluation after training completes")
    parser.add_argument("--eval-interval-steps", type=int, default=0, help="Run evaluation every N training steps (0 disables)")
    parser.add_argument("--eval-num-aug-samples", type=int, default=0, help="Maximum augmentations per base puzzle to evaluate (0 = all)")
    parser.add_argument("--eval-max-puzzles", type=int, default=0, help="Limit number of base puzzles evaluated (0 = all)")
    parser.add_argument("--eval-time-limit-seconds", type=int, default=3600, help="Wall-clock budget for evaluation runs (0 disables)")
    parser.add_argument("--resume-checkpoint", help="Path to a diffusion checkpoint to resume training from")
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int]:
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank

    timeout_env = os.environ.get("TORCH_DISTRIBUTED_DEFAULT_TIMEOUT")
    if timeout_env is None:
        timeout_env = os.environ.get("TRAIN_DDP_TIMEOUT_SECONDS")
    try:
        timeout_seconds = float(timeout_env) if timeout_env is not None else 7200.0
    except ValueError:
        timeout_seconds = 7200.0
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_seconds))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    return rank, world_size, local_rank


def compute_trm_batches_per_epoch(data_dir: str, global_batch_size: int) -> int:
    dataset_json = Path(data_dir) / "train" / "dataset.json"
    with dataset_json.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    total_examples = int(round(metadata["mean_puzzle_examples"] * metadata["total_puzzles"]))
    return math.ceil(total_examples / global_batch_size)


def prepare_streamer(
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    world_size: int,
    batches_per_epoch: int,
) -> LayerUpdateStreamer:
    streamer_cfg = TrajectoryStreamerConfig(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        global_batch_size=args.trm_global_batch_size,
        device=device,
        rank=rank,
        world_size=world_size,
        max_batches_per_epoch=batches_per_epoch,
    )
    return LayerUpdateStreamer(streamer_cfg)


def create_model_and_optimizer(
    seq_len: int,
    hidden_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DDP, torch.optim.Optimizer, DiffusionSchedule]:
    model = SingleLayerDiffusion(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    ddp_model = DDP(model, device_ids=[device.index], output_device=device.index)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)
    schedule = DiffusionSchedule.create(args.timesteps, device)
    return ddp_model, optimizer, schedule


def _schedule_state_from_schedule(schedule: DiffusionSchedule) -> Dict[str, torch.Tensor]:
    return {
        "betas": schedule.betas.detach().cpu(),
        "alphas": schedule.alphas.detach().cpu(),
        "alphas_cumprod": schedule.alphas_cumprod.detach().cpu(),
    }


def _schedule_from_state(state: Optional[Dict[str, torch.Tensor]], timesteps: int, device: torch.device) -> DiffusionSchedule:
    if state is None:
        return DiffusionSchedule.create(timesteps, device)

    betas = state["betas"].to(device)
    alphas = state["alphas"].to(device)
    alphas_cumprod = state["alphas_cumprod"].to(device)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
        sqrt_recip_alphas=torch.sqrt(1.0 / alphas),
    )


def _load_diffusion_checkpoint(path: Path) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    ckpt = torch.load(path, map_location="cpu")
    if "model_state_dict" in ckpt:
        model_state = ckpt["model_state_dict"]
        schedule_state = ckpt.get("schedule")
    else:
        model_state = ckpt
        schedule_state = None
    return model_state, schedule_state


def _reverse_diffusion_layer(
    model: SingleLayerDiffusion,
    schedule: DiffusionSchedule,
    condition_inputs: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    device = condition_inputs[0].device
    x = torch.randn_like(condition_inputs[0])
    timesteps = schedule.betas.shape[0]
    for step in reversed(range(timesteps)):
        t = torch.full((x.shape[0],), step, device=device, dtype=torch.long)
        pred_noise = model(x, t, *condition_inputs)
        beta = schedule.betas[step]
        sqrt_recip_alpha = schedule.sqrt_recip_alphas[step]
        sqrt_one_minus = schedule.sqrt_one_minus_alphas_cumprod[step]
        x = sqrt_recip_alpha * (x - beta / sqrt_one_minus * pred_noise)
        if step > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta) * noise
    return x


def _prepare_trm_evaluator(
    data_dir: str,
    trm_checkpoint: str,
    batch_size: int,
    device: torch.device,
    rank: int,
    world_size: int,
) -> Tuple[torch.nn.Module, data.DataLoader, PuzzleDatasetMetadata, object]:
    os.environ.setdefault("DISABLE_COMPILE", "1")
    arch_cfg = _load_arch_config()
    pretrain_cfg = _build_pretrain_config(
        arch_cfg,
        data_path=data_dir,
        global_batch_size=batch_size,
        checkpoint=trm_checkpoint,
    )

    dataloader, metadata = create_dataloader(
        pretrain_cfg,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=pretrain_cfg.global_batch_size,
        rank=rank,
        world_size=world_size,
    )

    trm_model, _, _ = create_model(pretrain_cfg, metadata, rank=rank, world_size=world_size)
    trm_model = trm_model.to(device)
    trm_model.eval()
    for param in trm_model.parameters():
        param.requires_grad_(False)

    return trm_model, dataloader, metadata, arch_cfg


def _run_layer_evaluation(
    args: argparse.Namespace,
    *,
    model_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    schedule_state: Optional[Dict[str, torch.Tensor]] = None,
    checkpoint_path: Optional[Path] = None,
    step_label: str = "eval",
    device_override: Optional[str] = None,
    output_path: Optional[Path] = None,
    rank: int = 0,
    world_size: int = 1,
    group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, float]:
    eval_data_dir = args.eval_data_dir or args.data_dir
    trm_ckpt = args.eval_trm_checkpoint or args.checkpoint_path
    eval_device = torch.device(device_override or args.eval_device)

    trm_model, eval_loader, metadata, arch_cfg = _prepare_trm_evaluator(
        eval_data_dir,
        trm_ckpt,
        args.eval_trm_batch_size,
        eval_device,
        rank,
        world_size,
    )

    seq_len = metadata.seq_len + getattr(arch_cfg, "puzzle_emb_len", 0)
    hidden_dim = arch_cfg.hidden_size

    diffusion_model = SingleLayerDiffusion(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(eval_device)

    if model_state_dict is not None:
        diffusion_model.load_state_dict(model_state_dict)
    else:
        if checkpoint_path is None:
            raise ValueError("Evaluation requires a checkpoint path or in-memory model state.")
        state_dict, sched_state = _load_diffusion_checkpoint(checkpoint_path)
        diffusion_model.load_state_dict(state_dict)
        if schedule_state is None:
            schedule_state = sched_state

    schedule = _schedule_from_state(schedule_state, args.timesteps, eval_device)
    diffusion_model.eval()

    arc_evaluator = ARC(
        eval_data_dir,
        metadata,
        submission_K=args.eval_submission_k,
        pass_Ks=tuple(args.eval_pass_ks),
        aggregated_voting=True,
    )
    arc_evaluator.begin_eval()

    if args.eval_max_puzzles > 0:
        limited = dict(list(arc_evaluator.test_puzzles.items())[: args.eval_max_puzzles])
        arc_evaluator.test_puzzles = limited

    inner = trm_model.model.inner  # type: ignore[attr-defined]
    puzzle_emb_len = inner.puzzle_emb_len

    allowed_aug_counts: Optional[Dict[str, int]] = None
    if args.eval_num_aug_samples > 0:
        allowed_aug_counts = {}

    allowed_base_names: Optional[set[str]] = None
    if args.eval_max_puzzles > 0:
        allowed_base_names = set(arc_evaluator.test_puzzles.keys())

    total_puzzles = 0
    last_log = 0
    progress_interval = max(args.eval_trm_batch_size, 100)
    start_time = time.time()
    time_limit = getattr(args, "eval_time_limit_seconds", 0) or 0

    with torch.no_grad():
        for split_name, batch, _ in eval_loader:
            # PuzzleDataset yields per-set names for the test split, so do not filter on the label.

            if time_limit and time.time() - start_time > time_limit:
                raise TimeoutError(
                    f"Evaluation exceeded time limit of {time_limit} seconds while processing split {split_name}."
                )

            full_inputs = batch["inputs"]
            full_ids = batch["puzzle_identifiers"]
            batch_size_full = full_inputs.size(0)

            keep_indices: Optional[List[int]] = None
            if allowed_aug_counts is not None or allowed_base_names is not None:
                selected: List[int] = []
                for idx in range(batch_size_full):
                    pid = int(full_ids[idx].item())
                    if pid == arc_evaluator.blank_identifier_id:
                        continue
                    name = arc_evaluator.identifier_map[pid]
                    base_name, _ = inverse_aug(name)
                    if allowed_base_names is not None and base_name not in allowed_base_names:
                        continue
                    if allowed_aug_counts is not None:
                        count = allowed_aug_counts.get(base_name, 0)
                        if count >= args.eval_num_aug_samples:
                            continue
                        allowed_aug_counts[base_name] = count + 1
                    selected.append(idx)
                if not selected:
                    continue
                keep_indices = selected

            if keep_indices is not None:
                keep_tensor = torch.tensor(keep_indices, dtype=torch.long)
                inputs = full_inputs.index_select(0, keep_tensor).to(eval_device)
                puzzle_ids = full_ids.index_select(0, keep_tensor).to(eval_device)
                subset_batch = {
                    k: (v.index_select(0, keep_tensor) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                batch_size = inputs.size(0)
            else:
                inputs = full_inputs.to(eval_device)
                puzzle_ids = full_ids.to(eval_device)
                subset_batch = batch
                batch_size = inputs.size(0)

            batch_device = {k: v.to(eval_device) for k, v in subset_batch.items() if isinstance(v, torch.Tensor)}
            carry = trm_model.initial_carry(batch_device)

            inner_carry = carry.inner_carry
            inner_carry.z_H = inner_carry.z_H.to(eval_device)
            inner_carry.z_L = inner_carry.z_L.to(eval_device)
            reset_mask = torch.ones((batch_size,), device=eval_device, dtype=torch.bool)
            seeded_carry = inner.reset_carry(reset_mask, inner_carry)

            embeddings = inner._input_embeddings(inputs, puzzle_ids)

            diffusion_dtype = diffusion_model.time_mlp[0].weight.dtype
            embeddings = embeddings.to(device=eval_device, dtype=diffusion_dtype)
            init_z_H = seeded_carry.z_H.to(device=eval_device, dtype=diffusion_dtype)
            init_z_L = seeded_carry.z_L.to(device=eval_device, dtype=diffusion_dtype)

            base_batch = {
                "inputs": subset_batch["inputs"].cpu(),
                "puzzle_identifiers": subset_batch["puzzle_identifiers"].cpu(),
            }

            for _ in range(args.eval_num_samples):
                z_H = init_z_H.clone()
                z_L = init_z_L.clone()

                for _ in range(inner.config.H_cycles):
                    running_z = z_L.clone()
                    for _ in range(inner.config.L_cycles):
                        running_z = _reverse_diffusion_layer(
                            diffusion_model,
                            schedule,
                            (running_z, z_H, embeddings),
                        )
                    z_L = running_z
                    z_H = _reverse_diffusion_layer(
                        diffusion_model,
                        schedule,
                        (z_H, z_L),
                    )

                final_z = z_H
                final_z_for_head = final_z.to(inner.lm_head.weight.dtype)
                logits = inner.lm_head(final_z_for_head)[:, puzzle_emb_len:]
                preds = logits.argmax(dim=-1)
                q_logits = inner.q_head(final_z_for_head[:, 0])[:, 0]

                arc_evaluator.update_batch(
                    base_batch,
                    {
                        "preds": preds.cpu(),
                        "q_halt_logits": q_logits.cpu(),
                    },
                )

            total_puzzles += batch_size
            if rank == 0 and total_puzzles - last_log >= progress_interval:
                elapsed_now = time.time() - start_time
                print(
                    f"[Eval][{step_label}] processed {total_puzzles} puzzles "
                    f"in {elapsed_now:.1f}s on rank 0"
                )
                last_log = total_puzzles

    save_dir = Path(args.output_dir) / "eval_artifacts"
    save_dir.mkdir(parents=True, exist_ok=True)
    results = arc_evaluator.result(str(save_dir), rank=rank, world_size=world_size, group=group)

    if eval_device.type == "cuda":
        torch.cuda.synchronize(eval_device)

    elapsed = time.time() - start_time
    global_puzzles = total_puzzles
    global_elapsed = elapsed

    if dist.is_initialized() and world_size > 1:
        total_tensor = torch.tensor([float(total_puzzles)], device=eval_device)
        elapsed_tensor = torch.tensor([float(elapsed)], device=eval_device)
        dist.reduce(total_tensor, dst=0, op=dist.ReduceOp.SUM, group=group)
        dist.reduce(elapsed_tensor, dst=0, op=dist.ReduceOp.MAX, group=group)
        if rank == 0:
            global_puzzles = int(total_tensor.item())
            global_elapsed = float(elapsed_tensor.item())

    metrics: Dict[str, float] = {
        "puzzles_evaluated": int(global_puzzles),
        "num_samples_per_aug": args.eval_num_samples,
        "elapsed_seconds": float(global_elapsed),
    }
    if results:
        metrics.update(results)
    metrics["step"] = step_label

    if rank == 0:
        output_file = output_path or (Path(args.output_dir) / f"eval_{step_label}.json")
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump({"metrics": metrics, "args": vars(args)}, handle, indent=2)

    return metrics


def train_batch(
    ddp_model: DDP,
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    inputs: Sequence[torch.Tensor],
    targets: torch.Tensor,
    args: argparse.Namespace,
    step: int,
    device: torch.device,
) -> float:
    batch_size = targets.shape[0]
    timesteps = torch.randint(0, schedule.betas.shape[0], (batch_size,), device=device, dtype=torch.long)
    noise = torch.randn_like(targets, device=device)

    sqrt_alpha = schedule.sqrt_alphas_cumprod[timesteps].view(batch_size, 1, 1)
    sqrt_one_minus = schedule.sqrt_one_minus_alphas_cumprod[timesteps].view(batch_size, 1, 1)
    noisy_target = sqrt_alpha * targets + sqrt_one_minus * noise

    autocast_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
        pred_noise = ddp_model(noisy_target, timesteps, *inputs)
        loss = F.mse_loss(pred_noise, noise)
        scaled_loss = loss / args.grad_accumulation

    scaled_loss.backward()

    if (step + 1) % args.grad_accumulation == 0:
        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return float(loss.detach().cpu())


def main():
    args = parse_args()

    if args.eval_only:
        eval_rank = 0
        eval_world = 1
        eval_group: Optional[dist.ProcessGroup] = None
        local_rank_env: Optional[int] = None

        world_env = int(os.environ.get("WORLD_SIZE", "1"))
        if not dist.is_initialized() and world_env > 1:
            eval_rank, eval_world, local_rank_env = init_distributed()
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank_env)
            eval_group = dist.group.WORLD
        elif dist.is_initialized():
            eval_rank = dist.get_rank()
            eval_world = dist.get_world_size()
            eval_group = dist.group.WORLD
            local_rank_env = int(os.environ.get("LOCAL_RANK", eval_rank % max(torch.cuda.device_count(), 1)))

        device_override = args.eval_device
        if local_rank_env is not None:
            device_override = f"cuda:{local_rank_env}"

        checkpoint = args.eval_checkpoint
        if checkpoint is None:
            checkpoint = Path(args.output_dir) / "final_checkpoint.pt"
        metrics = _run_layer_evaluation(
            args,
            checkpoint_path=Path(checkpoint),
            step_label="eval_only",
            device_override=device_override,
            output_path=Path(args.eval_output_path) if args.eval_output_path else None,
            rank=eval_rank,
            world_size=eval_world,
            group=eval_group,
        )
        print(json.dumps(metrics, indent=2))
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    rank, world_size, local_rank = init_distributed()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(42 + rank)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    dist.barrier()

    if args.max_trm_batches_per_epoch is not None:
        batches_per_epoch = args.max_trm_batches_per_epoch
    else:
        batches_per_epoch = compute_trm_batches_per_epoch(args.data_dir, args.trm_global_batch_size)

    streamer = prepare_streamer(args, device, rank, world_size, batches_per_epoch)

    warm_examples = streamer.fetch_examples(1)
    seq_len, hidden_dim = warm_examples[0]["target"].shape
    streamer.reset()

    ddp_model, optimizer, schedule = create_model_and_optimizer(seq_len, hidden_dim, args, device)
    optimizer.zero_grad(set_to_none=True)

    resume_step = 0
    if args.resume_checkpoint:
        if rank == 0:
            print(f"Resuming diffusion training from {args.resume_checkpoint}")
        ckpt_payload = torch.load(args.resume_checkpoint, map_location="cpu")
        model_state = ckpt_payload.get("model_state_dict") or ckpt_payload
        ddp_model.module.load_state_dict(model_state)
        opt_state = ckpt_payload.get("optimizer_state_dict")
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        schedule_state_cpu = ckpt_payload.get("schedule")
        if schedule_state_cpu is not None:
            schedule = _schedule_from_state(schedule_state_cpu, args.timesteps, device)
        resume_step = int(ckpt_payload.get("global_step", 0))
        torch.cuda.synchronize(device)

    if rank == 0:
        config_path = Path(args.output_dir) / "config.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "args": vars(args),
                    "seq_len": seq_len,
                    "hidden_dim": hidden_dim,
                    "world_size": world_size,
                    "batches_per_epoch": batches_per_epoch,
                },
                handle,
                indent=2,
            )

    job_start = time.time()
    global_step = resume_step
    time_limit = args.time_limit_seconds
    eval_interval = args.eval_interval_steps
    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()
        ddp_model.train()
        pending_examples: List = []
        epoch_loss = 0.0
        epoch_batches = 0
        time_limit_reached = False

        for batch_examples in streamer.iter_epoch(batches_per_epoch):
            pending_examples.extend(batch_examples)
            while len(pending_examples) >= args.train_batch_size:
                examples_slice = pending_examples[:args.train_batch_size]
                del pending_examples[:args.train_batch_size]
                inputs_cpu, targets_cpu, _ = collate_layer_examples(examples_slice)
                targets = targets_cpu.to(device)
                inputs = tuple(inp.to(device) for inp in inputs_cpu)

                loss_value = train_batch(
                    ddp_model,
                    optimizer,
                    schedule,
                    inputs,
                    targets,
                    args,
                    global_step,
                    device,
                )
                epoch_loss += loss_value
                epoch_batches += 1
                global_step += 1

                if epoch_batches % args.log_interval == 0 and rank == 0:
                    print(
                        f"Epoch {epoch} | batch {epoch_batches} | loss {loss_value:.6f} | pending {len(pending_examples)}"
                    )

                should_eval = eval_interval and (global_step % eval_interval == 0)

                if should_eval:
                    torch.cuda.synchronize(device)
                    if dist.is_initialized():
                        dist.barrier(device_ids=[local_rank])
                    state_dict_cpu = {k: v.detach().cpu() for k, v in ddp_model.module.state_dict().items()}
                    sched_state = _schedule_state_from_schedule(schedule)
                    label = f"step_{global_step}"
                    checkpoint_path = Path(args.output_dir) / f"checkpoint_{label}.pt"
                    if rank == 0:
                        torch.save(
                            {
                                "model_state_dict": state_dict_cpu,
                                "optimizer_state_dict": optimizer.state_dict(),
                                "global_step": global_step,
                                "args": vars(args),
                                "schedule": sched_state,
                            },
                            checkpoint_path,
                        )
                        print(f"Saved checkpoint to {checkpoint_path}")
                    if dist.is_initialized():
                        dist.barrier(device_ids=[local_rank])
                    eval_output_path = Path(args.output_dir) / f"eval_{label}.json" if rank == 0 else None
                    eval_group = dist.group.WORLD if dist.is_initialized() else None
                    metrics = _run_layer_evaluation(
                        args,
                        model_state_dict=state_dict_cpu,
                        schedule_state=sched_state,
                        step_label=label,
                        device_override=f"cuda:{local_rank}",
                        output_path=eval_output_path,
                        rank=rank,
                        world_size=world_size,
                        group=eval_group,
                    )
                    if rank == 0:
                        print(json.dumps(metrics, indent=2))
                    if dist.is_initialized():
                        dist.barrier(device_ids=[local_rank])
                    ddp_model.train()

                if time.time() - job_start >= time_limit:
                    time_limit_reached = True
                    break

            if time_limit_reached:
                break

        if not time_limit_reached and pending_examples:
            inputs_cpu, targets_cpu, _ = collate_layer_examples(pending_examples)
            targets = targets_cpu.to(device)
            inputs = tuple(inp.to(device) for inp in inputs_cpu)
            loss_value = train_batch(
                ddp_model,
                optimizer,
                schedule,
                inputs,
                targets,
                args,
                global_step,
                device,
            )
            epoch_loss += loss_value
            epoch_batches += 1
            global_step += 1
            pending_examples.clear()

        epoch_time = time.time() - epoch_start
        mean_loss = epoch_loss / max(epoch_batches, 1)

        if rank == 0:
            log_path = Path(args.output_dir) / "training.log"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"epoch={epoch}, batches={epoch_batches}, mean_loss={mean_loss:.6f}, time={epoch_time:.2f}s\n"
                )
            print(
                f"Epoch {epoch} complete on rank 0 | batches={epoch_batches} | mean_loss={mean_loss:.6f} | time={epoch_time:.2f}s"
            )

        if time_limit_reached:
            if rank == 0:
                print("Time limit reached, stopping training.")
            break

    dist.barrier()

    final_sched_state = _schedule_state_from_schedule(schedule)

    if rank == 0:
        checkpoint_path = Path(args.output_dir) / "final_checkpoint.pt"
        torch.save(
            {
                "model_state_dict": ddp_model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "args": vars(args),
                "schedule": final_sched_state,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")

    if args.eval_after_train:
        state_dict_cpu = {k: v.detach().cpu() for k, v in ddp_model.module.state_dict().items()}
        if dist.is_initialized():
            dist.barrier(device_ids=[local_rank])
        final_output_path = Path(args.output_dir) / "eval_final.json" if rank == 0 else None
        eval_group = dist.group.WORLD if dist.is_initialized() else None
        final_metrics = _run_layer_evaluation(
            args,
            model_state_dict=state_dict_cpu,
            schedule_state=final_sched_state,
            step_label="final",
            device_override=f"cuda:{local_rank}",
            output_path=final_output_path,
            rank=rank,
            world_size=world_size,
            group=eval_group,
        )
        if rank == 0:
            print(json.dumps(final_metrics, indent=2))
        if dist.is_initialized():
            dist.barrier(device_ids=[local_rank])

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
