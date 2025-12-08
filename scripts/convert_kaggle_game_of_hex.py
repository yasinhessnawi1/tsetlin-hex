"""
Convert the Kaggle "game-of-hex" dataset to the GTM raw format for 7x7 boards.

Output:
  data/train_games_7x7.npz
  data/test_games_7x7.npz

These files mimic the C-generated format expected by 1b_build_gtm_datasets.py:
  - winners            (int8)
  - stages             (int32, values like [0])
  - states_at_0        (int8, shape: [N, 7, 7])
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np


def find_npz(dataset_dir: Path):
    npz_files = list(dataset_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {dataset_dir}. "
                                "Please unzip the Kaggle dataset here.")
    # Prefer a file that looks like it has boards
    for name_hint in ["hex", "board", "game"]:
        for f in npz_files:
            if name_hint in f.name.lower():
                return f
    return npz_files[0]


def pick_key(keys, candidates):
    for cand in candidates:
        if cand in keys:
            return cand
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert Kaggle game-of-hex dataset to GTM raw npz (7x7).")
    parser.add_argument("--dataset-dir", type=str, default="data/kaggle_game_of_hex",
                        help="Directory containing the unzipped Kaggle dataset")
    parser.add_argument("--board-size", type=int, default=7, help="Board size (expected 7)")
    parser.add_argument("--train-output", type=str, default="data/train_games_7x7.npz",
                        help="Output path for train npz")
    parser.add_argument("--test-output", type=str, default="data/test_games_7x7.npz",
                        help="Output path for test npz")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Test split fraction (default 0.2)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"ERROR: dataset directory not found: {dataset_dir}")
        sys.exit(1)

    npz_path = find_npz(dataset_dir)
    print(f"[INFO] Loading source npz: {npz_path}")
    data = np.load(npz_path)
    keys = set(data.keys())
    print(f"[INFO] Keys found: {keys}")

    board_key = pick_key(keys, ["states", "boards", "board_states", "positions"])
    winner_key = pick_key(keys, ["winners", "winner", "result", "labels"])

    if board_key is None or winner_key is None:
        print("ERROR: Could not find board/winner keys in the npz.")
        print("Expected one of board keys: states, boards, board_states, positions")
        print("Expected one of winner keys: winners, winner, result, labels")
        sys.exit(1)

    boards = data[board_key]
    winners = data[winner_key]

    if boards.ndim == 3:
        # shape: (N, 7, 7)
        pass
    elif boards.ndim == 4 and boards.shape[-1] == 1:
        boards = boards[..., 0]
    else:
        print(f"ERROR: Unexpected board shape {boards.shape}. Expected (N,7,7) or (N,7,7,1).")
        sys.exit(1)

    if boards.shape[1] != args.board_size or boards.shape[2] != args.board_size:
        print(f"ERROR: Board size mismatch: {boards.shape[1:]} vs expected {args.board_size}x{args.board_size}")
        sys.exit(1)

    if winners.shape[0] != boards.shape[0]:
        print("ERROR: winners length does not match number of boards.")
        sys.exit(1)

    num_samples = boards.shape[0]
    print(f"[INFO] Samples: {num_samples}")

    # Basic sanity: ensure winners are 0/1
    unique_w = np.unique(winners)
    print(f"[INFO] Unique winners: {unique_w}")
    if not set(unique_w.tolist()).issubset({0, 1}):
        print("WARNING: Winners not in {0,1}. Please verify label encoding.")

    # Shuffle and split
    rng = np.random.default_rng(seed=42)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    test_count = max(1, int(num_samples * args.test_split))
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    train_boards = boards[train_idx]
    train_winners = winners[train_idx]
    test_boards = boards[test_idx]
    test_winners = winners[test_idx]

    stages = np.array([0], dtype=np.int32)

    Path("data").mkdir(exist_ok=True)
    np.savez_compressed(
        args.train_output,
        num_games=len(train_winners),
        board_size=args.board_size,
        winners=train_winners.astype(np.int8),
        stages=stages,
        states_at_0=train_boards.astype(np.int8),
    )
    np.savez_compressed(
        args.test_output,
        num_games=len(test_winners),
        board_size=args.board_size,
        winners=test_winners.astype(np.int8),
        stages=stages,
        states_at_0=test_boards.astype(np.int8),
    )

    print(f"[OK] Wrote train npz: {args.train_output} (games: {len(train_winners)})")
    print(f"[OK] Wrote test  npz: {args.test_output} (games: {len(test_winners)})")
    print("\nNext: run 1b_build_gtm_datasets.py --board-size 7 --stages 0")


if __name__ == "__main__":
    main()

