#!/usr/bin/env python
"""
Plotting utility for RL training metrics.

Usage:
    python TinyRecursiveModels/plot_rl_metrics.py --log-dir analysis/rl_run1 --output plots.png
    
    # Plot specific metrics
    python TinyRecursiveModels/plot_rl_metrics.py --log-dir analysis/rl_run1 --metrics loss,success_rate,pass_at_1
    
    # Plot multiple runs for comparison
    python TinyRecursiveModels/plot_rl_metrics.py --log-dirs run1:analysis/rl_run1,run2:analysis/rl_run2
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    if not path.exists():
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply exponential moving average smoothing."""
    if window <= 1 or len(values) <= 1:
        return values
    alpha = 2 / (window + 1)
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def plot_train_metrics(
    log_dir: Path,
    output_path: Optional[Path] = None,
    metrics: Optional[List[str]] = None,
    smooth_window: int = 10,
    x_axis: str = "global_step",
    run_name: str = "",
) -> Tuple[plt.Figure, Dict[str, np.ndarray]]:
    """
    Plot training metrics from a log directory.
    
    Args:
        log_dir: Directory containing train_metrics.jsonl and eval_metrics.jsonl
        output_path: Path to save the figure (optional)
        metrics: List of metric names to plot (default: all standard metrics)
        smooth_window: Window size for smoothing
        x_axis: X-axis variable ('global_step', 'samples_seen', or 'wall_time')
        run_name: Name for this run in legends
        
    Returns:
        Figure and dict of extracted metrics
    """
    train_path = log_dir / "train_metrics.jsonl"
    eval_path = log_dir / "eval_metrics.jsonl"
    
    train_data = load_jsonl(train_path)
    eval_data = load_jsonl(eval_path)
    
    if not train_data and not eval_data:
        raise ValueError(f"No data found in {log_dir}")
    
    # Default metrics to plot
    if metrics is None:
        metrics = [
            "loss",
            "success_rate", 
            "grad_norm_before_clip",
            "advantage_mean",
            "positive_advantage_frac",
            "pass_at_1",  # from eval
        ]
    
    # Extract data
    extracted = {}
    
    if train_data:
        train_x = np.array([d[x_axis] for d in train_data])
        for m in metrics:
            if m in train_data[0] and m != "pass_at_1":
                extracted[m] = (train_x, np.array([d[m] for d in train_data]))
    
    if eval_data and "pass_at_1" in metrics:
        eval_x = np.array([d[x_axis] for d in eval_data])
        extracted["pass_at_1"] = (eval_x, np.array([d["pass_at_1"] for d in eval_data]))
    
    # Create figure
    n_metrics = len([m for m in metrics if m in extracted])
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each metric
    plot_idx = 0
    for m in metrics:
        if m not in extracted:
            continue
        
        ax = axes[plot_idx]
        x, y = extracted[m]
        
        # Plot raw data with low alpha
        label = f"{run_name} {m}" if run_name else m
        ax.plot(x, y, alpha=0.3, linewidth=0.5)
        
        # Plot smoothed data
        y_smooth = smooth(y, smooth_window)
        ax.plot(x, y_smooth, linewidth=2, label=label)
        
        ax.set_xlabel(x_axis.replace("_", " ").title())
        ax.set_ylabel(m.replace("_", " ").title())
        ax.set_title(m.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        
        if m == "pass_at_1":
            ax.set_ylim(0, 1)
        
        plot_idx += 1
    
    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    
    return fig, extracted


def plot_comparison(
    log_dirs: Dict[str, Path],
    output_path: Optional[Path] = None,
    metrics: Optional[List[str]] = None,
    smooth_window: int = 10,
    x_axis: str = "global_step",
) -> plt.Figure:
    """
    Plot comparison of metrics across multiple runs.
    
    Args:
        log_dirs: Dict mapping run names to log directories
        output_path: Path to save the figure
        metrics: List of metric names to plot
        smooth_window: Window size for smoothing
        x_axis: X-axis variable
        
    Returns:
        Figure
    """
    if metrics is None:
        metrics = ["loss", "success_rate", "pass_at_1"]
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_dirs)))
    
    for (run_name, log_dir), color in zip(log_dirs.items(), colors):
        train_data = load_jsonl(log_dir / "train_metrics.jsonl")
        eval_data = load_jsonl(log_dir / "eval_metrics.jsonl")
        
        for ax, m in zip(axes, metrics):
            data = eval_data if m == "pass_at_1" else train_data
            if not data or m not in data[0]:
                continue
            
            x = np.array([d[x_axis] for d in data])
            y = np.array([d[m] for d in data])
            y_smooth = smooth(y, smooth_window)
            
            ax.plot(x, y_smooth, label=run_name, color=color, linewidth=2)
            ax.set_xlabel(x_axis.replace("_", " ").title())
            ax.set_ylabel(m.replace("_", " ").title())
            ax.set_title(m.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    
    return fig


def print_summary(log_dir: Path):
    """Print summary statistics from a run."""
    train_data = load_jsonl(log_dir / "train_metrics.jsonl")
    eval_data = load_jsonl(log_dir / "eval_metrics.jsonl")
    
    print(f"\n{'='*60}")
    print(f"Summary for: {log_dir}")
    print(f"{'='*60}")
    
    if train_data:
        latest = train_data[-1]
        print(f"\nTraining Progress:")
        print(f"  Global Step: {latest['global_step']:,}")
        print(f"  Samples Seen: {latest['samples_seen']:,}")
        print(f"  Wall Time: {latest['wall_time']/3600:.2f} hours")
        
        # Recent averages (last 100 entries)
        recent = train_data[-100:]
        print(f"\nRecent Metrics (last {len(recent)} logs):")
        print(f"  Loss: {np.mean([d['loss'] for d in recent]):.4f}")
        print(f"  Success Rate: {np.mean([d['success_rate'] for d in recent]):.3f}")
        print(f"  Grad Norm (before clip): {np.mean([d['grad_norm_before_clip'] for d in recent]):.2f}")
        print(f"  Advantage Mean: {np.mean([d['advantage_mean'] for d in recent]):.3f}")
    
    if eval_data:
        latest_eval = eval_data[-1]
        best_eval = max(eval_data, key=lambda d: d['pass_at_1'])
        print(f"\nEvaluation:")
        print(f"  Latest pass@1: {latest_eval['pass_at_1']:.4f} (step {latest_eval['global_step']})")
        print(f"  Best pass@1: {best_eval['pass_at_1']:.4f} (step {best_eval['global_step']})")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Plot RL training metrics")
    parser.add_argument("--log-dir", type=str, help="Single log directory")
    parser.add_argument("--log-dirs", type=str, help="Multiple log dirs: name1:path1,name2:path2")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path for plot")
    parser.add_argument("--metrics", type=str, default=None, help="Comma-separated metric names")
    parser.add_argument("--smooth", type=int, default=10, help="Smoothing window size")
    parser.add_argument("--x-axis", type=str, default="global_step", 
                        choices=["global_step", "samples_seen", "wall_time"])
    parser.add_argument("--summary", action="store_true", help="Print summary statistics")
    parser.add_argument("--no-show", action="store_true", help="Don't show plot interactively")
    
    args = parser.parse_args()
    
    metrics = args.metrics.split(",") if args.metrics else None
    output_path = Path(args.output) if args.output else None
    
    if args.log_dirs:
        # Multiple runs comparison
        log_dirs = {}
        for item in args.log_dirs.split(","):
            name, path = item.split(":")
            log_dirs[name] = Path(path)
        
        if args.summary:
            for name, path in log_dirs.items():
                print_summary(path)
        
        plot_comparison(
            log_dirs,
            output_path=output_path,
            metrics=metrics,
            smooth_window=args.smooth,
            x_axis=args.x_axis,
        )
    elif args.log_dir:
        # Single run
        log_dir = Path(args.log_dir)
        
        if args.summary:
            print_summary(log_dir)
        
        plot_train_metrics(
            log_dir,
            output_path=output_path,
            metrics=metrics,
            smooth_window=args.smooth,
            x_axis=args.x_axis,
        )
    else:
        parser.error("Must specify --log-dir or --log-dirs")
    
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

