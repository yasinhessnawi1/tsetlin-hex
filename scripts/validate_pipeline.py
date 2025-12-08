"""
Pipeline validator to catch false-high accuracies.

Checks:
- Train/test file identity (same path or size) and sample-level overlap.
- Class balance for train/test.
- Duplicate samples within train/test.

Supports raw NPZ (states_at_<stage>) and GTM PKL (graphs/labels).

Usage examples:
  python scripts/validate_pipeline.py --board-size 15 --stages end,-2,-5
  python scripts/validate_pipeline.py --board-size 7 --stages 0 --data-dir data/kaggle_eval --gtm
"""

import argparse
import hashlib
import os
import pickle
from pathlib import Path
import numpy as np


def sha1_array(arr: np.ndarray) -> str:
    """Hash a numpy array (content + shape + dtype) deterministically."""
    h = hashlib.sha1()
    h.update(arr.shape.__repr__().encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def load_raw_stage(npz_path: Path, stage: str):
    data = np.load(npz_path)
    winners = data["winners"].astype(np.int8)
    boards = data[f"states_at_{stage}"].astype(np.int8)
    return boards, winners


def load_gtm_stage(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["graphs"], data["labels"]


def summarize(labels: np.ndarray, name: str):
    total = len(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(uniq, counts)}
    print(f"{name}: {total} samples | class dist: {dist}")


def main():
    parser = argparse.ArgumentParser(description="Validate data pipeline for leakage/overlap.")
    parser.add_argument("--board-size", type=int, required=True, help="Board size.")
    parser.add_argument("--stages", type=str, default="end,-2,-5", help='Comma-separated stages, e.g., "end,-2,-5" or "0"')
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory (root for train/test files).")
    parser.add_argument("--gtm", action="store_true", help="Validate GTM pickles instead of raw NPZ.")
    parser.add_argument("--max-check", type=int, default=50000, help="Cap samples when hashing (per split) for speed.")
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    data_dir = Path(args.data_dir)

    for stage in stages:
        print("\n" + "=" * 60)
        print(f"STAGE: {stage}")
        print("=" * 60)

        if args.gtm:
            train_path = data_dir / f"train_gtm_{args.board_size}x{args.board_size}_{stage}.pkl"
            test_path = data_dir / f"test_gtm_{args.board_size}x{args.board_size}_{stage}.pkl"
            if not train_path.exists() or not test_path.exists():
                print(f"[WARN] Missing GTM files: {train_path} or {test_path}")
                continue
            train_graphs, train_labels = load_gtm_stage(train_path)
            test_graphs, test_labels = load_gtm_stage(test_path)

            summarize(train_labels, "Train labels")
            summarize(test_labels, "Test labels")

            # Hash graphs by serialized bytes of (nodes/edges/messages). Rough check using repr of obj.
            def graph_hash(g):  # type: ignore
                return hashlib.sha1(repr(g).encode()).hexdigest()

            train_hashes = [graph_hash(g) for g in train_graphs[: args.max_check]]
            test_hashes = [graph_hash(g) for g in test_graphs[: args.max_check]]
        else:
            # raw npz
            train_path = data_dir / f"train_games_{args.board_size}x{args.board_size}.npz"
            test_path = data_dir / f"test_games_{args.board_size}x{args.board_size}.npz"
            if not train_path.exists() or not test_path.exists():
                print(f"[WARN] Missing raw NPZ files: {train_path} or {test_path}")
                continue
            train_boards, train_winners = load_raw_stage(train_path, stage if stage != "end" else "0")
            test_boards, test_winners = load_raw_stage(test_path, stage if stage != "end" else "0")

            summarize(train_winners, "Train winners")
            summarize(test_winners, "Test winners")

            train_hashes = [sha1_array(b) for b in train_boards[: args.max_check]]
            test_hashes = [sha1_array(b) for b in test_boards[: args.max_check]]

        # Basic file equality check
        if train_path.resolve() == test_path.resolve():
            print("[ERROR] Train and test paths are identical!")

        # Overlap checks
        train_set = set(train_hashes)
        test_set = set(test_hashes)
        overlap = train_set.intersection(test_set)
        print(f"Overlap (first {args.max_check} per split): {len(overlap)} samples")

        # Duplicate checks
        dup_train = len(train_hashes) - len(train_set)
        dup_test = len(test_hashes) - len(test_set)
        if dup_train > 0:
            print(f"[WARN] Duplicates in train (within first {args.max_check}): {dup_train}")
        if dup_test > 0:
            print(f"[WARN] Duplicates in test (within first {args.max_check}): {dup_test}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()



