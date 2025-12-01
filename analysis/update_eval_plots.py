import json
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = ROOT / "TinyRecursiveModels" / "checkpoints" / "arc1concept" / "world_arcagi2_from_world_fresh"
SUBMISSION_PREFIX = "evaluator_ARC_step_"
CHALLENGES_PATH = ROOT / "TinyRecursiveModels" / "kaggle" / "combined" / "arc-agi_evaluation2_challenges.json"
SOLUTIONS_PATH = ROOT / "TinyRecursiveModels" / "kaggle" / "combined" / "arc-agi_evaluation2_solutions.json"
CSV_PATH = ROOT / "analysis" / "eval_pass_metrics.csv"
PLOT_PATH = ROOT / "analysis" / "eval_metrics.png"

PASS_KS = (1, 2)


def load_dataset() -> Dict[str, Dict[str, List]]:
    with CHALLENGES_PATH.open("r") as f:
        challenges = json.load(f)
    with SOLUTIONS_PATH.open("r") as f:
        solutions = json.load(f)
    return {
        pid: {"test": puzzle["test"], "solutions": solutions[pid]}
        for pid, puzzle in challenges.items()
        if pid in solutions
    }


def compute_pass_metrics(submission_path: Path, dataset: Dict[str, Dict[str, List]]) -> Dict[int, float]:
    with submission_path.open("r") as f:
        submission = json.load(f)

    totals = {k: 0.0 for k in PASS_KS}
    puzzle_ids = sorted(dataset.keys())

    for pid in puzzle_ids:
        puzzle = dataset[pid]
        tests = puzzle["test"]
        solutions = puzzle["solutions"]
        preds = submission.get(pid, [])

        per_puzzle_correct = {k: 0 for k in PASS_KS}
        for idx in range(len(tests)):
            gt = solutions[idx]
            attempt_dict = preds[idx] if idx < len(preds) else {}
            attempt_grids = []
            if isinstance(attempt_dict, dict):
                for attempt_idx in range(1, max(PASS_KS) + 1):
                    grid = attempt_dict.get(f"attempt_{attempt_idx}")
                    if grid is not None:
                        attempt_grids.append(grid)
            if attempt_grids:
                while len(attempt_grids) < max(PASS_KS):
                    attempt_grids.append(attempt_grids[0])

            for k in PASS_KS:
                limit = min(k, len(attempt_grids))
                ok = any(attempt_grids[i] == gt for i in range(limit)) if limit > 0 else False
                per_puzzle_correct[k] += 1 if ok else 0

        for k in PASS_KS:
            if len(tests) == 0:
                continue
            totals[k] += per_puzzle_correct[k] / len(tests)

    puzzle_count = len(puzzle_ids)
    if puzzle_count == 0:
        return {k: 0.0 for k in PASS_KS}

    return {k: totals[k] / puzzle_count for k in PASS_KS}


def collect_metrics(dataset: Dict[str, Dict[str, List]]):
    rows = []
    for path in sorted(EVAL_ROOT.glob(f"{SUBMISSION_PREFIX}*/submission.json")):
        step_str = path.parent.name.replace(SUBMISSION_PREFIX, "")
        try:
            step = int(step_str)
        except ValueError:
            continue
        metrics = compute_pass_metrics(path, dataset)
        rows.append((step, metrics[1], metrics[2]))

    rows.sort(key=lambda x: x[0])
    return rows


def save_csv(rows):
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "ARC/pass@1", "ARC/pass@2"])
        for step, pass1, pass2 in rows:
            writer.writerow([step, f"{pass1:.6f}", f"{pass2:.6f}"])


def save_plot(rows):
    if not rows:
        PLOT_PATH.unlink(missing_ok=True)
        return

    steps = [row[0] for row in rows]
    pass1 = [row[1] for row in rows]
    pass2 = [row[2] for row in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, pass1, label="ARC/pass@1", marker="o")
    plt.plot(steps, pass2, label="ARC/pass@2", marker="o")
    plt.xlabel("Step")
    plt.ylabel("Pass rate")
    plt.title("ARC Evaluation Pass Rates")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()


def main():
    dataset = load_dataset()
    rows = collect_metrics(dataset)
    save_csv(rows)
    save_plot(rows)


if __name__ == "__main__":
    main()
