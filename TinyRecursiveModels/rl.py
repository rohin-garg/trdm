from __future__ import annotations

import argparse
from cgitb import small
import itertools
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from core_all_onestep_train import OneStepCoreModel, _build_pretrain_config, _load_arch_config
from streaming_core_train import build_teacher, success_mask
from pretrain import create_dataloader


# Global flag for graceful shutdown on SIGTERM
_SHUTDOWN_REQUESTED = False


def _sigterm_handler(signum, frame):
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    print("[signal] SIGTERM received, will checkpoint and exit gracefully...")


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


@dataclass
class TrainMetrics:
    """Metrics accumulated during training for logging."""
    global_step: int = 0
    samples_seen: int = 0
    wall_time: float = 0.0
    
    # Per-step metrics (averaged over log_interval)
    loss: float = 0.0
    success_rate: float = 0.0
    grad_norm_before_clip: float = 0.0
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_min: float = 0.0
    advantage_max: float = 0.0
    positive_advantage_frac: float = 0.0  # fraction of samples with positive advantage
    
    # Additional useful metrics
    lr: float = 0.0
    batch_success_count: int = 0
    batch_size: int = 0
    step_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class EvalMetrics:
    """Metrics from evaluation runs."""
    global_step: int = 0
    samples_seen: int = 0
    wall_time: float = 0.0
    pass_at_1: float = 0.0
    eval_puzzles: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsLogger:
    """Logger for training and evaluation metrics."""
    
    def __init__(self, output_dir: Path, rank: int = 0):
        self.output_dir = output_dir
        self.rank = rank
        self.train_log_path = output_dir / "train_metrics.jsonl"
        self.eval_log_path = output_dir / "eval_metrics.jsonl"
        
    def log_train(self, metrics: TrainMetrics):
        if self.rank != 0:
            return
        with open(self.train_log_path, "a") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")
            
    def log_eval(self, metrics: EvalMetrics):
        if self.rank != 0:
            return
        with open(self.eval_log_path, "a") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")


def save_checkpoint(
    path: Path,
    one_step: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    samples_seen: int,
    rng_state: Dict[str, Any],
    args: argparse.Namespace,
    extra: Optional[Dict[str, Any]] = None,
):
    """Save a checkpoint for resuming training."""
    model_state = one_step.module.state_dict() if isinstance(one_step, DDP) else one_step.state_dict()
    ckpt = {
        "one_step": model_state,
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "samples_seen": samples_seen,
        "rng_state": rng_state,
        "args": vars(args),
    }
    if extra:
        ckpt.update(extra)
    
    # Save to temp file first, then rename for atomicity
    tmp_path = path.with_suffix(".tmp")
    torch.save(ckpt, tmp_path)
    tmp_path.rename(path)
    print(f"[checkpoint] saved {path}")


def load_checkpoint(
    path: Path,
    one_step: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, int, Dict[str, Any]]:
    """Load a checkpoint and return (global_step, samples_seen, rng_state)."""
    ckpt = torch.load(path, map_location=device)
    
    model = one_step.module if isinstance(one_step, DDP) else one_step
    model.load_state_dict(ckpt["one_step"])
    optimizer.load_state_dict(ckpt["optimizer"])
    
    return ckpt["global_step"], ckpt["samples_seen"], ckpt.get("rng_state", {})


def get_rng_state(device: torch.device) -> Dict[str, Any]:
    """Get current RNG state for checkpointing."""
    state = {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None,
    }
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def set_rng_state(state: Dict[str, Any], device: torch.device):
    """Restore RNG state from checkpoint."""
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"], device)
    try:
        import numpy as np
        if "numpy" in state:
            np.random.set_state(state["numpy"])
    except ImportError:
        pass


@torch.no_grad()
def run_eval(
    teacher,
    core_models: List[nn.Module],
    eval_loader,
    device: torch.device,
    steps: int,
    noise_scale: float,
    max_puzzles: int,
    num_cores: int,
) -> float:
    """Run evaluation and return pass@1 accuracy."""
    for core_model in core_models:
        core_model.eval()
    
    
    def replacement_fn(x_clean: torch.Tensor, noise: torch.Tensor, cur_core_idx: int) -> torch.Tensor:
        core = core_models[cur_core_idx].module if isinstance(core_models[cur_core_idx], DDP) else core_models[cur_core_idx]
        rms = torch.sqrt(torch.mean(x_clean.to(torch.float32) ** 2, dim=(1, 2)) + 1e-6)
        eps = noise.to(torch.float32) / (noise_scale * rms.view(-1, 1, 1) + 1e-6)
        return core(x_clean.to(torch.float32), eps)
    
    replacements = {idx: replacement_fn for idx in range(num_cores)}
    
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
        
        logits, _ = teacher.streaming_forward_for_rl(
            batch=batch,
            steps=steps,
            noise_scale=0.0,  # no additional noise during eval
            collect_core_io=False,
            replacements=replacements,
        )
        mask = success_mask(logits, batch["labels"])
        correct += int(mask.sum().item())
    
    for core_model in core_models:
        core_model.train()
    return correct / max(1, total)


def compute_grad_norm(model: nn.Module) -> float:
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def main(args: argparse.Namespace) -> None:
    global _SHUTDOWN_REQUESTED
    
    # Set up signal handler for graceful preemption
    signal.signal(signal.SIGTERM, _sigterm_handler)
    
    rank, world_size, local_rank, device = init_distributed()
    torch.manual_seed(1234 + rank)
    os.environ.setdefault("DISABLE_COMPILE", "1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics logger
    logger = MetricsLogger(out_dir, rank=rank)
    
    # Check for existing checkpoint to resume from
    latest_ckpt = out_dir / "latest.pt"
    resume_from = None
    if latest_ckpt.exists():
        resume_from = latest_ckpt
        if rank == 0:
            print(f"[resume] found checkpoint at {latest_ckpt}")
    elif args.resume_from and Path(args.resume_from).exists():
        resume_from = Path(args.resume_from)
        if rank == 0:
            print(f"[resume] loading from {resume_from}")

    # Build teacher and data
    teacher, loader, _ = build_teacher(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    seq_len = teacher.inner.config.seq_len + teacher.inner.puzzle_emb_len
    hidden_dim = teacher.inner.config.hidden_size
    core_layers = teacher.inner.config.L_layers
    num_cores = args.steps * teacher.inner.config.H_cycles * (teacher.inner.config.L_cycles + 1)
    
    if rank == 0:
        print(f"[info] training shared one-step for all cores; num_cores={num_cores}")

    core_models = [
        OneStepCoreModel(
            seq_len_clean=seq_len,
            hidden_dim=hidden_dim,
            num_layers=core_layers,
            num_heads=teacher.inner.config.num_heads,
            expansion=teacher.inner.config.expansion,
            rms_norm_eps=teacher.inner.config.rms_norm_eps,
            rope_theta=teacher.inner.config.rope_theta,
        ).to(device)
        for _ in range(num_cores)
    ]
    if args.onestep_checkpoint is not None:
        state = torch.load(args.onestep_checkpoint, map_location=device)
        for i, core_model in enumerate(core_models):
            core_model.load_state_dict(state["one_step"])

    if world_size > 1:
        core_models = [DDP(core_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) for core_model in core_models]
    one_opt = torch.optim.AdamW(
        itertools.chain.from_iterable(core_model.parameters() for core_model in core_models),
        lr=args.lr, weight_decay=args.weight_decay
    )

    print("========== args.num_fixed_batches: ", args.num_fixed_batches)
    if args.num_fixed_batches > 0:
        if rank == 0:
            print(f"[info] buffering {args.num_fixed_batches} batches for fixed-set training")
        fixed_batches = []
        temp_it = iter(loader)
        for _ in range(args.num_fixed_batches):
            try:
                item = next(temp_it)
                fixed_batches.append(item)
            except StopIteration:
                if rank == 0:
                    print(f"[warn] requested {args.num_fixed_batches} batches but dataset only had {len(fixed_batches)}")
                break
        if len(fixed_batches) == 0:
            raise ValueError("Dataset empty or failed to load any batches")
        loader_it = itertools.cycle(fixed_batches)
    else:
        loader_it = iter(loader)

    samples_seen = 0
    global_step = 0
    start_time = time.time()
    
    # Resume from checkpoint if available
    # if resume_from:
    #     global_step, samples_seen, rng_state = load_checkpoint(
    #         resume_from, one_step, one_opt, device
    #     )
    #     if rng_state:
    #         set_rng_state(rng_state, device)
    #     if rank == 0:
    #         print(f"[resume] restored global_step={global_step}, samples_seen={samples_seen}")
    
    # Prepare eval loader
    eval_loader = None
    if args.eval_interval > 0:
        eval_cfg = _build_pretrain_config(
            _load_arch_config(), 
            args.eval_data_dir or args.data_dir, 
            args.batch_size * world_size, 
            args.checkpoint
        )
        eval_loader, _ = create_dataloader(
            eval_cfg,
            split="test",
            rank=rank,
            world_size=world_size,
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=eval_cfg.global_batch_size,
        )
    
    next_ckpt_step = global_step + args.ckpt_interval
    next_eval_step = global_step + args.eval_interval if args.eval_interval > 0 else float('inf')
    
    # Accumulators for averaging metrics
    accum_loss = 0.0
    accum_grad_norm_before = 0.0
    accum_grad_norm_after = 0.0
    accum_success_rate = 0.0
    accum_advantage_mean = 0.0
    accum_advantage_std = 0.0
    accum_advantage_min = float('inf')
    accum_advantage_max = float('-inf')
    accum_positive_adv_frac = 0.0
    accum_reward_mean = 0.0
    accum_reward_std = 0.0
    accum_steps = 0
    log_start_time = time.time()

    # Run initial evaluation before training starts
    if eval_loader is not None and args.eval_interval > 0 and args.eval_first:
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            print(f"[eval] running initial evaluation at step {global_step}...")
            pass_at_1 = run_eval(
                teacher=teacher,
                core_models=core_models,
                eval_loader=eval_loader,
                device=device,
                steps=args.steps,
                noise_scale=args.noise_scale,
                max_puzzles=args.eval_puzzles,
                num_cores=num_cores,
            )
            eval_metrics = EvalMetrics(
                global_step=global_step,
                samples_seen=samples_seen,
                wall_time=time.time() - start_time,
                pass_at_1=pass_at_1,
                eval_puzzles=args.eval_puzzles,
            )
            logger.log_eval(eval_metrics)
            print(f"[eval] step={global_step} pass@1={pass_at_1:.4f}")
        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        print("[info] starting training loop")

    while global_step < args.max_steps:
        # Check for shutdown signal
        if _SHUTDOWN_REQUESTED:
            if rank == 0:
                print("[signal] saving checkpoint before exit...")
                save_checkpoint(
                    out_dir / "preempt.pt",
                    core_models, one_opt, global_step, samples_seen,
                    get_rng_state(device), args
                )
                # Also update latest
                save_checkpoint(
                    latest_ckpt,
                    core_models, one_opt, global_step, samples_seen,
                    get_rng_state(device), args
                )
            if dist.is_initialized():
                dist.barrier()
            sys.exit(0)
        
        step_start_time = time.time()
        
        if args.num_fixed_batches > 0:
            _, batch, _ = next(loader_it)
        else:
            try:
                _, batch, _ = next(loader_it)
            except StopIteration:
                loader_it = iter(loader)
                _, batch, _ = next(loader_it)
        
        # Original batch shape: [B, ...]
        small_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        # print(f"inputs shape: {small_batch['inputs'].shape} {small_batch['inputs'].sum().item()}")
        # print(f"inputs: {small_batch['inputs'].view(small_batch['inputs'].shape[0], -1)[:20, :20]}")
        # small_batch = {k: v[0].expand_as(v) for k, v in small_batch.items()}
        # print(f"inputs 0: {small_batch['inputs'][0].sum().item()} inputs 1: {small_batch['inputs'][1].sum().item()} inputs 2: {small_batch['inputs'][2].sum().item()} inputs 3: {small_batch['inputs'][3].sum().item()}")
        batch = {k: v.repeat_interleave(args.g, dim=0) for k, v in small_batch.items()}
        
        core_idx = torch.randint(0, num_cores, (1,)).item()
        def replacement_fn(x_clean: torch.Tensor, noise: torch.Tensor, cur_core_idx: int) -> torch.Tensor:
            rms = torch.sqrt(torch.mean(x_clean.to(torch.float32) ** 2, dim=(1, 2)) + 1e-6)
            eps = noise.to(torch.float32) / (args.noise_scale * rms.view(-1, 1, 1) + 1e-6)
            core = core_models[cur_core_idx].module if isinstance(core_models[cur_core_idx], DDP) else core_models[cur_core_idx]
            if cur_core_idx == core_idx: # only want grads for the current core we're injecting noise into
                return core(x_clean.to(torch.float32), eps)
            else:
                with torch.no_grad():
                    return core(x_clean.to(torch.float32), eps)

        replacements = {core_idx: replacement_fn}
        fixed_noise_seeds = {idx: torch.randint(0, 1000000, (1,)) for idx in range(num_cores)} # fix the inner noise for all the cores
        with torch.no_grad():
            logits, records = teacher.streaming_forward_for_rl(
                batch=batch,
                steps=args.steps,
                noise_scale=args.noise_scale,
                collect_core_io=True,
                core_idxs=[core_idx],
                replacements=replacements,
                fixed_noise_seeds=fixed_noise_seeds,
                expand_fixed_noise=args.g,
                post_process_noise_scale=args.post_process_noise_scale,
            )

        _, base_records = teacher.streaming_forward_for_rl( # no **postprocessed** noise records
            batch=small_batch,
            steps=args.steps,
            noise_scale=args.noise_scale,
            collect_core_io=True,
            core_idxs=[core_idx],
            replacements=replacements,
            fixed_noise_seeds=fixed_noise_seeds,
            post_process_noise_scale=None,
        )

        
        with torch.no_grad(): # computes the advantages for every generation
            
            success = success_mask(logits, batch["labels"]).view(-1, args.g).to(torch.float32) # [B, g]
            # print(success.shape)
            # print(f"success: {success}")
            advantage = (success - success.mean(dim=1).unsqueeze(1)) / (success.std(dim=1).unsqueeze(1) + 1e-6) # [B, g]
            advantage = advantage.view(-1) # [B*g]

            success_rate = success.mean().item()
            adv_mean = advantage.mean().item()
            adv_std = advantage.std().item()
            adv_min = advantage.min().item()
            adv_max = advantage.max().item()
            positive_adv_frac = (advantage > 0).float().mean().item()
            # print("frac of positive ROW advantages: ", (success.std(dim=1).unsqueeze(1) > 0).float().mean().item())

        loss = torch.tensor(0.0, requires_grad=True, device=device)
        assert len(records) == 1
        assert len(base_records) == 1
        # sum_diffs = 0
        for idx, ((_, x_clean, post_processed_noise, out), (_, _, _, base_out)) in enumerate(zip(records, base_records)):
            rms = torch.sqrt(torch.mean((out - post_processed_noise) ** 2, dim=(1, 2)) + 1e-6)
            std_dev = args.noise_scale * rms.view(-1) # [B*g]
            out = out.reshape(out.shape[0], -1)
            base_out = base_out.reshape(base_out.shape[0], -1).repeat_interleave(args.g, dim=0)
            assert out.shape == base_out.shape
            logprob = -((out - base_out)**2).mean(dim=1) / (2 * std_dev**2)
            # print(logprob.min().item(), (logprob * advantage).mean().item(), (logprob * advantage).std().item())
            loss = loss + (logprob * advantage).mean()
            # sum_diffs += ((out - base_out)**2).mean(dim=1).sum().item()

        loss = -loss / len(records)
        one_opt.zero_grad(set_to_none=True)
        loss.backward()
        
        # Compute gradient norm before clipping
        grad_norm_before = compute_grad_norm(core_models[core_idx])
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(core_models[core_idx].parameters(), args.max_grad_norm)
        one_opt.step()

        # new_sum_diffs = 0
        # _, base_records = teacher.streaming_forward_for_rl( # no **postprocessed** noise records
        #     batch=small_batch,
        #     steps=args.steps,
        #     noise_scale=args.noise_scale,
        #     collect_core_io=True,
        #     core_idxs=[core_idx],
        #     replacements=replacements,
        #     fixed_noise_seeds=fixed_noise_seeds,
        #     post_process_noise_scale=None,
        # )
        # for idx, ((_, x_clean, post_processed_noise, out), (_, _, _, base_out)) in enumerate(zip(records, base_records)):
        #     out = out.reshape(out.shape[0], -1)
        #     base_out = base_out.reshape(base_out.shape[0], -1).repeat_interleave(args.g, dim=0)
        #     assert out.shape == base_out.shape
        #     new_sum_diffs += ((out - base_out)**2).mean(dim=1).sum().item()
        # print(f"sum_diffs: {sum_diffs} new_sum_diffs: {new_sum_diffs} diff: {new_sum_diffs - sum_diffs}")

        global_step += 1
        samples_seen += batch["inputs"].shape[0]
        step_time = time.time() - step_start_time
        
        # Accumulate metrics
        accum_loss += loss.item()
        accum_grad_norm_before += grad_norm_before
        accum_success_rate += success_rate
        accum_advantage_mean += adv_mean
        accum_advantage_std += adv_std
        accum_advantage_min = min(accum_advantage_min, adv_min)
        accum_advantage_max = max(accum_advantage_max, adv_max)
        accum_positive_adv_frac += positive_adv_frac
        accum_steps += 1

        # Log metrics
        if global_step % args.log_interval == 0 and rank == 0:
            n = accum_steps
            metrics = TrainMetrics(
                global_step=global_step,
                samples_seen=samples_seen,
                wall_time=time.time() - start_time,
                loss=accum_loss / n,
                success_rate=accum_success_rate / n,
                grad_norm_before_clip=accum_grad_norm_before / n,
                advantage_mean=accum_advantage_mean / n,
                advantage_std=accum_advantage_std / n,
                advantage_min=accum_advantage_min,
                advantage_max=accum_advantage_max,
                positive_advantage_frac=accum_positive_adv_frac / n,
                lr=one_opt.param_groups[0]['lr'],
                batch_success_count=int(success.sum().item()),
                batch_size=batch["inputs"].shape[0] * args.g,
                step_time=(time.time() - log_start_time) / n,
            )
            logger.log_train(metrics)
            
            print(
                f"[train] step={global_step} samples={samples_seen} "
                f"loss={metrics.loss:.4f} success={metrics.success_rate:.3f} "
                f"grad={metrics.grad_norm_before_clip:.2f} "
                f"adv={metrics.advantage_mean:.3f}±{metrics.advantage_std:.3f} "
                f"t={metrics.step_time:.2f}s"
            )
            
            # Reset accumulators
            accum_loss = 0.0
            accum_grad_norm_before = 0.0
            accum_success_rate = 0.0
            accum_advantage_mean = 0.0
            accum_advantage_std = 0.0
            accum_advantage_min = float('inf')
            accum_advantage_max = float('-inf')
            accum_positive_adv_frac = 0.0
            accum_steps = 0
            log_start_time = time.time()

        # Evaluation
        if global_step >= next_eval_step and eval_loader is not None:
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                print(f"[eval] running evaluation at step {global_step}...")
                pass_at_1 = run_eval(
                    teacher=teacher,
                    core_models=core_models,
                    eval_loader=eval_loader,
                    device=device,
                    steps=args.steps,
                    noise_scale=args.noise_scale,
                    max_puzzles=args.eval_puzzles,
                    num_cores=num_cores,
                )
                eval_metrics = EvalMetrics(
                    global_step=global_step,
                    samples_seen=samples_seen,
                    wall_time=time.time() - start_time,
                    pass_at_1=pass_at_1,
                    eval_puzzles=args.eval_puzzles,
                )
                logger.log_eval(eval_metrics)
                print(f"[eval] step={global_step} pass@1={pass_at_1:.4f}")
            if dist.is_initialized():
                dist.barrier()
            next_eval_step += args.eval_interval

        # Checkpointing
        if global_step >= next_ckpt_step and rank == 0:
            ckpt_path = out_dir / f"step_{global_step}.pt"
            save_checkpoint(
                ckpt_path,
                one_step, one_opt, global_step, samples_seen,
                get_rng_state(device), args
            )
            # Also save as latest for easy resumption
            save_checkpoint(
                latest_ckpt,
                one_step, one_opt, global_step, samples_seen,
                get_rng_state(device), args
            )
            next_ckpt_step += args.ckpt_interval

    # Final checkpoint
    if rank == 0:
        save_checkpoint(
            out_dir / "final.pt",
            one_step, one_opt, global_step, samples_seen,
            get_rng_state(device), args
        )
        save_checkpoint(
            latest_ckpt,
            one_step, one_opt, global_step, samples_seen,
            get_rng_state(device), args
        )
        print(f"[done] training complete, final step={global_step}, samples={samples_seen}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL fine-tuning for one-step core model")
    p.add_argument("--checkpoint", required=True, help="Teacher TRM checkpoint path")
    p.add_argument("--onestep-checkpoint", required=True, help="Pretrained one-step checkpoint to fine-tune")
    p.add_argument("--data-dir", required=True, help="Training data directory")
    p.add_argument("--eval-data-dir", type=str, default=None, help="Evaluation data directory (defaults to data-dir)")
    p.add_argument("--output-dir", required=True, help="Output directory for checkpoints and logs")
    p.add_argument("--resume-from", type=str, default=None, help="Checkpoint to resume from (auto-detects latest.pt)")
    
    p.add_argument("--g", type=int, default=16, help="Number of generations per puzzle for REINFORCE")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate (typically lower for RL)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=4, help="Number of halting steps in teacher streaming forward")
    p.add_argument("--noise-scale", type=float, default=0.02, help="RMS-scaled noise std for teacher streaming_forward")
    p.add_argument("--post-process-noise-scale", type=float, default=0.02, help="RMS-scaled noise std for post-processed noise")
    p.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size (each sample generates g completions)")
    p.add_argument("--max-steps", type=int, default=100_000, help="Maximum training steps")
    p.add_argument("--num-fixed-batches", type=int, default=0, help="If > 0, cache this many batches and cycle them")
    
    p.add_argument("--log-interval", type=int, default=10, help="Log metrics every N steps")
    p.add_argument("--ckpt-interval", type=int, default=1000, help="Save checkpoint every N steps")
    p.add_argument("--eval-interval", type=int, default=500, help="Run evaluation every N steps (0 to disable)")
    p.add_argument("--eval-puzzles", type=int, default=2048, help="Number of puzzles for evaluation")
    p.add_argument("--eval-first", action="store_true", help="Run evaluation before training starts")
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
