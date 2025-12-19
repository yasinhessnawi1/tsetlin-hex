"""
Merge synthetic 7x7 data with a subset of Kaggle data and write mixed NPZs.

Default behavior:
- Reads synthetic NPZs from data/train_games_7x7.npz and data/test_games_7x7.npz
- Reads Kaggle NPZs from data/kaggle_eval/train_games_7x7.npz and data/kaggle_eval/test_games_7x7.npz
- Takes a subset of Kaggle samples (default 250k train, 50k test)
- Concatenates with synthetic, shuffles, and overwrites the synthetic NPZs

Example (PowerShell):
  python scripts/merge_kaggle_with_synthetic.py `
    --board-size 7 `
    --kaggle-train-take 250000 `
    --kaggle-test-take 50000
"""

import argparse
from pathlib import Path
import numpy as np


def load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    data = np.load(path)
    boards = data["states_at_0"]
    winners = data["winners"]
    board_size = data["board_size"].item() if "board_size" in data else boards.shape[1]
    return boards, winners, board_size


def take_subset(boards, winners, count: int, rng: np.random.Generator):
    count = min(count, len(winners))
    idx = rng.choice(len(winners), size=count, replace=False)
    return boards[idx], winners[idx]


def save_npz(path: Path, boards, winners, board_size: int):
    stages = np.array([0], dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        num_games=len(winners),
        board_size=board_size,
        winners=winners.astype(np.int8),
        stages=stages,
        states_at_0=boards.astype(np.int8),
    )


def summarize(name: str, boards, winners):
    uniq_b, cnt_b = np.unique(boards, return_counts=True)
    uniq_w, cnt_w = np.unique(winners, return_counts=True)
    print(f"\n=== {name} ===")
    print(f"samples: {len(winners)}")
    print(f"board values: {dict(zip(uniq_b.tolist(), cnt_b.tolist()))}")
    print(f"winner values: {dict(zip(uniq_w.tolist(), cnt_w.tolist()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-size", type=int, default=7)
    ap.add_argument("--synthetic-dir", type=str, default="data")
    ap.add_argument("--kaggle-dir", type=str, default="data/kaggle_eval")
    ap.add_argument("--kaggle-train-take", type=int, default=250_000)
    ap.add_argument("--kaggle-test-take", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-output", type=str, default=None,
                    help="Output train npz (default: overwrite synthetic train npz)")
    ap.add_argument("--test-output", type=str, default=None,
                    help="Output test npz (default: overwrite synthetic test npz)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    synth_train_path = Path(args.synthetic_dir) / f"train_games_{args.board_size}x{args.board_size}.npz"
    synth_test_path = Path(args.synthetic_dir) / f"test_games_{args.board_size}x{args.board_size}.npz"
    kaggle_train_path = Path(args.kaggle_dir) / f"train_games_{args.board_size}x{args.board_size}.npz"
    kaggle_test_path = Path(args.kaggle_dir) / f"test_games_{args.board_size}x{args.board_size}.npz"

    print("Loading synthetic train/test...")
    s_train_b, s_train_w, bsz_train = load_npz(synth_train_path)
    s_test_b, s_test_w, bsz_test = load_npz(synth_test_path)

    print("Loading Kaggle train/test...")
    k_train_b, k_train_w, bsz_ktrain = load_npz(kaggle_train_path)
    k_test_b, k_test_w, bsz_ktest = load_npz(kaggle_test_path)

    if not (bsz_train == bsz_test == bsz_ktrain == bsz_ktest == args.board_size):
        raise ValueError(
            f"Board size mismatch: synth_train={bsz_train}, synth_test={bsz_test}, "
            f"kaggle_train={bsz_ktrain}, kaggle_test={bsz_ktest}, expected={args.board_size}"
        )

    k_train_b_sub, k_train_w_sub = take_subset(k_train_b, k_train_w, args.kaggle_train_take, rng)
    k_test_b_sub, k_test_w_sub = take_subset(k_test_b, k_test_w, args.kaggle_test_take, rng)

    train_boards = np.concatenate([s_train_b, k_train_b_sub], axis=0)
    train_winners = np.concatenate([s_train_w, k_train_w_sub], axis=0)
    test_boards = np.concatenate([s_test_b, k_test_b_sub], axis=0)
    test_winners = np.concatenate([s_test_w, k_test_w_sub], axis=0)

    def shuffle_pair(b, w):
        idx = rng.permutation(len(w))
        return b[idx], w[idx]

    train_boards, train_winners = shuffle_pair(train_boards, train_winners)
    test_boards, test_winners = shuffle_pair(test_boards, test_winners)

    train_out = Path(args.train_output) if args.train_output else synth_train_path
    test_out = Path(args.test_output) if args.test_output else synth_test_path

    save_npz(train_out, train_boards, train_winners, args.board_size)
    save_npz(test_out, test_boards, test_winners, args.board_size)

    summarize("TRAIN MIXED", train_boards, train_winners)
    summarize("TEST  MIXED", test_boards, test_winners)

    print("\n[OK] Mixed datasets written:")
    print(f"  Train: {train_out}")
    print(f"  Test : {test_out}")
    print("You can now rebuild GTM with 1b_build_gtm_datasets.py and retrain.")


if __name__ == "__main__":
    main()

