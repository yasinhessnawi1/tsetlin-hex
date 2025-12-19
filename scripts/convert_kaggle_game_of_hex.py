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
import zipfile
import numpy as np


def find_source_file(dataset_dir: Path):
    """Find a source file (npz or csv). Auto-unzip zip files if present."""
    # Try unzipping any zip files in the directory
    zip_files = list(dataset_dir.glob("*.zip"))
    for zf in zip_files:
        try:
            print(f"[INFO] Unzipping {zf} ...")
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(dataset_dir)
        except Exception as exc:
            print(f"[WARN] Failed to unzip {zf}: {exc}")

    npz_files = list(dataset_dir.glob("*.npz"))
    csv_files = list(dataset_dir.glob("*.csv"))

    candidates = npz_files + csv_files
    if not candidates:
        raise FileNotFoundError(
            f"No .npz or .csv files found in {dataset_dir}. "
            "Please place or unzip the Kaggle dataset here."
        )

    # Prefer filenames with hints
    hints = ["hex", "board", "game"]
    for hint in hints:
        for f in candidates:
            if hint in f.name.lower():
                return f

    return candidates[0]


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
    parser.add_argument("--all-to-test", action="store_true",
                        help="Put all samples into test set (train set gets full copy as well for compatibility)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (default: all)")
    parser.add_argument("--subset-seed", type=int, default=42,
                        help="Random seed for subset selection (default: 42)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"ERROR: dataset directory not found: {dataset_dir}")
        sys.exit(1)

    source_path = find_source_file(dataset_dir)
    print(f"[INFO] Using source: {source_path}")

    if source_path.suffix.lower() == ".npz":
        data = np.load(source_path)
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
            pass
        elif boards.ndim == 4 and boards.shape[-1] == 1:
            boards = boards[..., 0]
        else:
            print(f"ERROR: Unexpected board shape {boards.shape}. Expected (N,7,7) or (N,7,7,1).")
            sys.exit(1)

    elif source_path.suffix.lower() == ".csv":
        print("[INFO] Detected CSV; attempting to parse (winner column + 49 cells).")
        # Detect header
        with open(source_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        parts = first_line.split(",")
        header = False
        try:
            [float(p) for p in parts]
        except Exception:
            header = True

        data = np.genfromtxt(source_path, delimiter=",", skip_header=1 if header else 0)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        expected_cols = args.board_size * args.board_size + 1
        if data.shape[1] < expected_cols:
            print(f"ERROR: CSV has {data.shape[1]} columns; expected at least {expected_cols}.")
            sys.exit(1)

        # Parse according to header if present
        if header:
            header_cols = [h.strip() for h in parts]
            lower_cols = [h.lower() for h in header_cols]

            # Winner column: prefer explicit name, otherwise assume last
            winner_idx = None
            if "winner" in lower_cols:
                winner_idx = lower_cols.index("winner")
            else:
                winner_idx = len(header_cols) - 1

            # Board columns: names starting with "cell"
            board_indices = [i for i, name in enumerate(lower_cols) if name.startswith("cell")]
            board_indices.sort()

            if len(board_indices) != args.board_size * args.board_size:
                # fallback: assume all except winner
                board_indices = [i for i in range(data.shape[1]) if i != winner_idx]

            if len(board_indices) != args.board_size * args.board_size:
                print(f"ERROR: Could not map board cells from header; found {len(board_indices)} vs expected {args.board_size * args.board_size}")
                sys.exit(1)

            winners = data[:, winner_idx].astype(np.int8)
            boards_flat = data[:, board_indices]
        else:
            # No header: assume winner is last column, cells are first N
            winners = data[:, -1].astype(np.int8)
            boards_flat = data[:, : args.board_size * args.board_size]

        boards = boards_flat.reshape(-1, args.board_size, args.board_size).astype(np.int8)

    else:
        print(f"ERROR: Unsupported source file type: {source_path}")
        sys.exit(1)

    # Normalize board values from Kaggle encoding:
    # Kaggle:    1 = first player (red),  -1 = second player (blue), 0 = empty
    # Synthetic: 1 = Player0 (first mover), 2 = Player1 (second mover), 0 = empty
    # Mapping to align with synthetic:
    #   Kaggle  1 (first player)  -> GTM 1 (Player0)
    #   Kaggle -1 (second player) -> GTM 2 (Player1)
    #   Kaggle  0 (empty)         -> GTM 0 (Empty)
    uniq_board_vals = np.unique(boards)
    expected_vals = {-1, 0, 1}
    if not set(uniq_board_vals.tolist()).issubset(expected_vals):
        print(f"[ERROR] Unexpected board values: {uniq_board_vals}. Expected subset of {-1,0,1}.")
        sys.exit(1)

    mapped = np.empty_like(boards, dtype=np.int8)
    mapped[boards == 0] = 0          # empty
    mapped[boards == 1] = 1          # Kaggle 1 (first player) -> Player0
    mapped[boards == -1] = 2         # Kaggle -1 (second player) -> Player1
    boards = mapped

    uniq_board_vals = np.unique(boards)
    print(f"[INFO] Board values after normalization (GTM): {uniq_board_vals} (0=Empty, 1=Player0, 2=Player1)")

    # Normalize winners to match synthetic convention:
    # Kaggle:    winner=1 means first player, winner=-1 means second player
    # Synthetic: winner=0 means first player (Player0), winner=1 means second player (Player1)
    # Mapping: Kaggle 1 -> 0, Kaggle -1 -> 1
    uniq_winners = np.unique(winners)
    print(f"[INFO] Winners before normalization: {uniq_winners}")
    if set(uniq_winners.tolist()).issubset({-1, 1}):
        # Kaggle format: 1 = first player, -1 = second player
        mapped_winners = np.empty_like(winners, dtype=np.int8)
        mapped_winners[winners == 1] = 0   # first player -> Player0 (label 0)
        mapped_winners[winners == -1] = 1  # second player -> Player1 (label 1)
        winners = mapped_winners
    elif not set(uniq_winners.tolist()).issubset({0, 1}):
        print(f"[ERROR] Unexpected winner values: {uniq_winners}. Expected {-1,1} or {0,1}.")
        sys.exit(1)
    uniq_winners = np.unique(winners)
    print(f"[INFO] Winners after normalization: {uniq_winners} (0=Player0/first, 1=Player1/second)")

    # Apply subset selection if requested
    if args.max_samples is not None and args.max_samples < len(winners):
        print(f"[INFO] Selecting subset of {args.max_samples} games from {len(winners)} total")
        np.random.seed(args.subset_seed)
        indices = np.random.choice(len(winners), size=args.max_samples, replace=False)
        winners = winners[indices]
        boards = boards[indices]
        print(f"[INFO] Subset selected with seed {args.subset_seed}")

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

    if args.all_to_test:
        test_idx = indices
        train_idx = indices  # duplicate full set for compatibility
    else:
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

