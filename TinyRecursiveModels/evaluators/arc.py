from typing import Dict, Sequence, Optional
import os
import json
from pathlib import Path
import time

import torch
import numpy as np
from numba import njit
import torch.distributed as dist

from dataset.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np
from dataset.common import PuzzleDatasetMetadata

@njit
def _crop(grid: np.ndarray):
    """Find maximum-sized rectangle without any EOS token inside. """
    grid = grid.reshape(30, 30)

    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape
    
    num_c = nc
    for num_r in range(1, nr + 1):
        # Scan for maximum c
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break
        
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)

    return (grid[:max_size[0], :max_size[1]] - 2).astype(np.uint8)


class ARC:
    required_outputs = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds"}
    
    def __init__(self, data_path: str, 
        eval_metadata: PuzzleDatasetMetadata, 
        submission_K: int = 2, 
        pass_Ks: Sequence[int] = (1, 2, 5, 10, 100, 1000), 
        aggregated_voting: bool = True):
        super().__init__()
        self.pass_Ks = pass_Ks
        self.submission_K = submission_K
        self.aggregated_voting = aggregated_voting
        self.blank_identifier_id = eval_metadata.blank_identifier_id

        # Load identifiers and test puzzles
        with open(os.path.join(data_path, "identifiers.json"), "r") as f:
            self.identifier_map = json.load(f)
        with open(os.path.join(data_path, "test_puzzles.json"), "r") as f:
            self.test_puzzles = json.load(f)
            
        # States
        self._local_hmap = {}
        self._local_preds = {}
        
    def begin_eval(self):
        if not self.aggregated_voting:
            # Clear previous predictions
            self._local_hmap = {}
            self._local_preds = {}
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        # Collect required outputs to CPU
        outputs = {}
        q_values = None

        for collection in (batch, preds):
            for k, v in collection.items():
                if k in self.required_outputs:
                    if k == "q_halt_logits":
                        q_values = v.to(torch.float64).sigmoid().cpu()
                    else:
                        outputs[k] = v.cpu()
                        
        assert q_values is not None

        # Remove padding from outputs
        mask = outputs["puzzle_identifiers"] != self.blank_identifier_id
        outputs = {k: v[mask] for k, v in outputs.items()}

        # Get predictions
        for identifier, input, pred, q in zip(outputs["puzzle_identifiers"].numpy(), outputs["inputs"].numpy(), outputs["preds"].numpy(), q_values.numpy()):
            name = self.identifier_map[identifier]
            orig_name, _inverse_fn = inverse_aug(name)
            
            input_hash = grid_hash(_inverse_fn(_crop(input)))
            
            pred = _inverse_fn(_crop(pred))
            assert np.all((pred >= 0) & (pred <= 9)), f"Puzzle {name}'s prediction out of 0-9 range."  # Sanity check

            # Store into local state
            pred_hash = grid_hash(pred)

            self._local_hmap[pred_hash] = pred
            
            self._local_preds.setdefault(orig_name, {})
            self._local_preds[orig_name].setdefault(input_hash, [])
            self._local_preds[orig_name][input_hash].append((pred_hash, float(q)))
    
    def result(self, save_path: Optional[str], rank: int, world_size: int, group: Optional[torch.distributed.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        if save_path is not None:
            output_dir = Path(save_path) / "rank_outputs"
        else:
            output_dir = Path.cwd() / "arc_rank_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        rank_file = output_dir / f"rank_{rank}.json"
        print(f"[ARC evaluator] Rank {rank} writing {rank_file}")
        payload = {
            "hmap": {str(k): v.tolist() for k, v in self._local_hmap.items()},
            "preds": {
                puzzle: {
                    str(inp_hash): [[str(pred_hash), float(q)] for pred_hash, q in entries]
                    for inp_hash, entries in inputs.items()
                }
                for puzzle, inputs in self._local_preds.items()
            },
        }
        with open(rank_file, "w") as f:
            json.dump(payload, f)

        if dist.is_initialized():
            dist.barrier(group=group)

        if rank != 0:
            return

        global_hmap_preds = []
        for r in range(world_size):
            rank_path = output_dir / f"rank_{r}.json"
            wait_start = time.time()
            while not rank_path.exists():
                time.sleep(1.0)
                if time.time() - wait_start > 300:
                    raise RuntimeError(f"Timed out waiting for {rank_path}")
            print(f"[ARC evaluator] Rank 0 loading {rank_path}")
            with open(rank_path, "r") as f:
                data = json.load(f)
            hmap = {k: np.asarray(v) for k, v in data["hmap"].items()}
            preds = {
                puzzle: {
                    inp_hash: [(pred_hash, float(q)) for pred_hash, q in entries]
                    for inp_hash, entries in inputs.items()
                }
                for puzzle, inputs in data["preds"].items()
            }
            global_hmap_preds.append((hmap, preds))

        submission = {}
        correct = [0.0 for _ in range(len(self.pass_Ks))]

        for name, puzzle in self.test_puzzles.items():
            # Process test examples in this puzzle
            submission[name] = []
            num_test_correct = [0 for _ in range(len(self.pass_Ks))]
            for pair in puzzle["test"]:
                input_hash = grid_hash(arc_grid_to_np(pair["input"]))
                label_hash = grid_hash(arc_grid_to_np(pair["output"]))
                
                p_map = {}
                for hmap, preds in global_hmap_preds:  # type: ignore
                    for h, q in preds.get(name, {}).get(input_hash, {}):
                        p_map.setdefault(h, [0, 0])
                        p_map[h][0] += 1
                        p_map[h][1] += q
                        
                if not len(p_map):
                    print (f"Puzzle {name} has no predictions.")
                    continue

                for h, stats in p_map.items():
                    stats[1] /= stats[0]
                    
                p_map = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

                # vote for different Ks
                for i, k in enumerate(self.pass_Ks):
                    ok = False
                    for h, stats in p_map[:k]:
                        ok |= h == label_hash
                        
                    num_test_correct[i] += ok
                    
                # Query grids
                pred_grids = []
                for h, stats in p_map[:self.submission_K]:
                    for hmap, preds in global_hmap_preds:  # type: ignore
                        if h in hmap:
                            pred_grids.append(hmap[h])
                            break
                        
                # Pad to K
                while len(pred_grids) < self.submission_K:
                    pred_grids.append(pred_grids[0])
                
                submission[name].append({f"attempt_{i + 1}": np.asarray(grid).tolist() for i, grid in enumerate(pred_grids)})

            # Total correctness
            for i in range(len(self.pass_Ks)):
                correct[i] += num_test_correct[i] / len(puzzle["test"])

        # Save submission
        if save_path is not None:
            with open(os.path.join(save_path, "submission.json"), "w") as f:
                json.dump(submission, f)

            for r in range(world_size):
                try:
                    os.remove(output_dir / f"rank_{r}.json")
                except FileNotFoundError:
                    pass

        # Final result
        all_results = {f"ARC/pass@{k}": correct[i] / len(self.test_puzzles) for i, k in enumerate(self.pass_Ks)}

        return all_results
