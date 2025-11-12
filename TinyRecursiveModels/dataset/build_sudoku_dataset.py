from typing import Optional
import os
import csv
import json
import shutil
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-extreme-full"

    subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_aug: int = 0
    batch_size: int = 1000


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    
    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def convert_subset(set_name: str, config: DataProcessConfig):
    # Read CSV and collect valid puzzles (keep memory low by just storing indices)
    valid_puzzles = []

    with open(hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset"), newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for idx, (source, q, a, rating) in enumerate(reader):
            if (config.min_difficulty is None) or (int(rating) >= config.min_difficulty):
                assert len(q) == 81 and len(a) == 81
                valid_puzzles.append((q, a))

    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(valid_puzzles)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            valid_puzzles = [valid_puzzles[i] for i in indices]

    # Generate dataset in batches
    num_augments = config.num_aug if set_name == "train" else 0
    batch_size = min(config.batch_size, len(valid_puzzles))

    # Create save directory
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    # Temporary directory for batches
    temp_dir = os.path.join(save_dir, "temp_batches")
    os.makedirs(temp_dir, exist_ok=True)

    all_batch_files = []
    total_examples = 0
    total_puzzles = 0

    # Process in batches
    for batch_start in tqdm(range(0, len(valid_puzzles), batch_size), desc=f"Processing {set_name} batches"):
        batch_end = min(batch_start + batch_size, len(valid_puzzles))
        batch_puzzles = valid_puzzles[batch_start:batch_end]

        results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
        batch_example_id = 0
        batch_puzzle_id = 0

        results["puzzle_indices"].append(0)
        results["group_indices"].append(0)

        for q, a in batch_puzzles:
            # Convert to numpy arrays
            orig_inp = np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
            orig_out = np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0')

            for aug_idx in range(1 + num_augments):
                # First index is not augmented
                if aug_idx == 0:
                    inp, out = orig_inp, orig_out
                else:
                    inp, out = shuffle_sudoku(orig_inp, orig_out)

                # Push puzzle (only single example)
                results["inputs"].append(inp)
                results["labels"].append(out)
                batch_example_id += 1
                batch_puzzle_id += 1

                results["puzzle_indices"].append(batch_example_id)
                results["puzzle_identifiers"].append(0)

            # Push group
            results["group_indices"].append(batch_puzzle_id)

        # Convert batch to numpy arrays
        def _seq_to_numpy(seq):
            arr = np.concatenate(seq).reshape(len(seq), -1)
            assert np.all((arr >= 0) & (arr <= 9))
            return arr + 1

        batch_results = {
            "inputs": _seq_to_numpy(results["inputs"]),
            "labels": _seq_to_numpy(results["labels"]),
            "group_indices": np.array(results["group_indices"], dtype=np.int32),
            "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
            "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
        }

        # Save batch to temporary file
        batch_file = os.path.join(temp_dir, f"batch_{batch_start:06d}_{batch_end:06d}.npz")
        np.savez(batch_file, **batch_results)
        all_batch_files.append(batch_file)

        total_examples += len(batch_results["inputs"])
        total_puzzles += len(batch_results["group_indices"]) - 1

    # Concatenate all batches
    print(f"Concatenating {len(all_batch_files)} batches for {set_name}...")

    all_inputs = []
    all_labels = []
    all_puzzle_identifiers = []
    all_puzzle_indices = []
    all_group_indices = []

    cumulative_examples = 0
    cumulative_puzzles = 0

    for batch_file in tqdm(all_batch_files, desc="Loading batches"):
        batch_data = np.load(batch_file)

        all_inputs.append(batch_data["inputs"])
        all_labels.append(batch_data["labels"])
        all_puzzle_identifiers.append(batch_data["puzzle_identifiers"])
        all_puzzle_indices.append(batch_data["puzzle_indices"] + cumulative_examples)
        all_group_indices.append(batch_data["group_indices"] + cumulative_puzzles)

        cumulative_examples += len(batch_data["inputs"])
        cumulative_puzzles += len(batch_data["group_indices"]) - 1

    # Concatenate final arrays
    final_results = {
        "inputs": np.concatenate(all_inputs),
        "labels": np.concatenate(all_labels),
        "puzzle_identifiers": np.concatenate(all_puzzle_identifiers),
        "puzzle_indices": np.concatenate(all_puzzle_indices),
        "group_indices": np.concatenate(all_group_indices),
    }

    # Fix the first elements to be 0
    final_results["puzzle_indices"][0] = 0
    final_results["group_indices"][0] = 0

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=10 + 1,  # PAD + "0" ... "9"
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=total_puzzles,
        mean_puzzle_examples=1,
        total_puzzles=total_puzzles,
        sets=["all"]
    )

    # Save metadata as JSON.
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for k, v in final_results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    # Clean up temporary files
    shutil.rmtree(temp_dir)

    print(f"Completed processing {set_name}: {total_puzzles} puzzles, {total_examples} examples")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
