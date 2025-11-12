from typing import Optional
import os
import csv
import json
import numpy as np
from numpy.lib.format import open_memmap

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
    batch_size = min(config.batch_size, len(valid_puzzles)) if len(valid_puzzles) else 0

    # Create save directory
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    # Allocate append-only memmaps for final arrays
    total_puzzles_expected = len(valid_puzzles)
    total_examples_expected = total_puzzles_expected * (1 + num_augments)

    inputs_mm = open_memmap(
        os.path.join(save_dir, "all__inputs.npy"),
        mode="w+",
        dtype=np.uint8,
        shape=(total_examples_expected, 81),
    )
    labels_mm = open_memmap(
        os.path.join(save_dir, "all__labels.npy"),
        mode="w+",
        dtype=np.uint8,
        shape=(total_examples_expected, 81),
    )
    puzzle_identifiers_mm = open_memmap(
        os.path.join(save_dir, "all__puzzle_identifiers.npy"),
        mode="w+",
        dtype=np.int32,
        shape=(total_examples_expected,),
    )
    puzzle_indices_mm = open_memmap(
        os.path.join(save_dir, "all__puzzle_indices.npy"),
        mode="w+",
        dtype=np.int32,
        shape=(total_examples_expected + 1,),
    )
    group_indices_mm = open_memmap(
        os.path.join(save_dir, "all__group_indices.npy"),
        mode="w+",
        dtype=np.int32,
        shape=(total_puzzles_expected + 1,),
    )

    if total_examples_expected:
        puzzle_identifiers_mm[:] = 0
    puzzle_indices_mm[0] = 0
    group_indices_mm[0] = 0

    # Process in batches streaming results into memmaps
    example_offset = 0
    group_offset = 0

    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1

    for batch_start in tqdm(range(0, len(valid_puzzles), batch_size or 1), desc=f"Processing {set_name} batches"):
        batch_end = min(batch_start + batch_size, len(valid_puzzles)) if batch_size else len(valid_puzzles)
        batch_puzzles = valid_puzzles[batch_start:batch_end]

        batch_inputs = []
        batch_labels = []
        batch_puzzle_counts = []

        for q, a in batch_puzzles:
            orig_inp = np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
            orig_out = np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0')

            examples_for_puzzle = 0
            for aug_idx in range(1 + num_augments):
                if aug_idx == 0:
                    inp, out = orig_inp, orig_out
                else:
                    inp, out = shuffle_sudoku(orig_inp, orig_out)

                batch_inputs.append(inp)
                batch_labels.append(out)
                examples_for_puzzle += 1

            batch_puzzle_counts.append(examples_for_puzzle)

        if not batch_inputs:
            continue

        batch_inputs_arr = _seq_to_numpy(batch_inputs)
        batch_labels_arr = _seq_to_numpy(batch_labels)
        batch_example_count = batch_inputs_arr.shape[0]

        start = example_offset
        stop = example_offset + batch_example_count

        inputs_mm[start:stop] = batch_inputs_arr
        labels_mm[start:stop] = batch_labels_arr
        puzzle_indices_mm[start + 1: stop + 1] = np.arange(start + 1, stop + 1, dtype=np.int32)

        running_examples = start
        for puzzle_count in batch_puzzle_counts:
            running_examples += puzzle_count
            group_offset += 1
            group_indices_mm[group_offset] = running_examples

        example_offset = stop

    total_examples = example_offset
    total_puzzles = group_offset

    assert total_examples == total_examples_expected
    assert total_puzzles == total_puzzles_expected

    inputs_mm.flush()
    labels_mm.flush()
    puzzle_identifiers_mm.flush()
    puzzle_indices_mm.flush()
    group_indices_mm.flush()

    del inputs_mm, labels_mm, puzzle_identifiers_mm, puzzle_indices_mm, group_indices_mm

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

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    print(f"Completed processing {set_name}: {total_puzzles} puzzles, {total_examples} examples")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
