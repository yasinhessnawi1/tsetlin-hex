"""
Debug helper to inspect Kaggle-to-GTM pipeline and model behavior.

Run from repo root, e.g.:
  python scripts/debug_kaggle_pipeline.py ^
    --npz-train data/train_games_7x7.npz ^
    --npz-test data/test_games_7x7.npz ^
    --gtm-test data/test_gtm_7x7_0.pkl ^
    --model models/gtm_7x7_end.pkl ^
    --sample 5000
"""

import argparse
import os
import sys
import pickle
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.models import HexGraphTM, Predictor  # noqa: E402


def summarize_npz(path: str, board_size: int, name: str):
    if not path:
        return
    if not os.path.exists(path):
        print(f"[WARN] NPZ not found: {path}")
        return
    data = np.load(path)
    boards = data["states_at_0"]
    winners = data["winners"]
    print(f"\n=== NPZ SUMMARY ({name}) ===")
    print(f"path: {path}")
    print(f"boards shape: {boards.shape}, dtype: {boards.dtype}")
    print(f"winners shape: {winners.shape}, dtype: {winners.dtype}")
    uniq_b, cnt_b = np.unique(boards, return_counts=True)
    print(f"board values: {dict(zip(uniq_b.tolist(), cnt_b.tolist()))}")
    uniq_w, cnt_w = np.unique(winners, return_counts=True)
    print(f"winner values: {dict(zip(uniq_w.tolist(), cnt_w.tolist()))}")
    if boards.shape[1:] != (board_size, board_size):
        print(f"[WARN] board size mismatch: {boards.shape[1:]} vs {board_size}x{board_size}")
    # small sample stone counts
    sample_n = min(3, boards.shape[0])
    for i in range(sample_n):
        b = boards[i]
        print(
            f"sample {i}: empty={(b==0).sum()}, p0={(b==1).sum()}, p1={(b==2).sum()}, winner={winners[i]}"
        )


def summarize_gtm(path: str, name: str, labels_only: bool = False):
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[WARN] GTM file not found: {path}")
        return None
    print(f"\n=== GTM SUMMARY ({name}) ===")
    print(f"path: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    labels = np.array(data["labels"])
    uniq_w, cnt_w = np.unique(labels, return_counts=True)
    print(f"labels: total={len(labels)}, values={dict(zip(uniq_w.tolist(), cnt_w.tolist()))}")
    graphs = data.get("graphs")
    if graphs is not None and not labels_only:
        # Graphs object may not expose num_graphs attr; fall back to len(labels)
        num_graphs = getattr(graphs, "num_graphs", len(labels))
        print(f"graphs object present; num_graphs={num_graphs}")
    return data


def evaluate_model(model_path: str, gtm_test: dict, sample: int):
    if not model_path or not os.path.exists(model_path):
        print(f"[WARN] Model not found: {model_path}")
        return
    if gtm_test is None:
        print("[WARN] No GTM test data provided; skip evaluation")
        return
    print(f"\n=== MODEL EVAL (sample={sample}) ===")
    model = HexGraphTM()
    model.load(model_path)
    graphs = gtm_test["graphs"]
    labels = gtm_test["labels"]
    if sample and sample > 0 and sample < len(labels):
        idx = np.random.default_rng(seed=123).choice(len(labels), size=sample, replace=False)
        if hasattr(graphs, "subset"):
            graphs_sample = graphs.subset(idx.tolist())
            labels_sample = labels[idx]
        else:
            print("[WARN] Graphs object has no subset(); evaluating full dataset instead.")
            graphs_sample = graphs
            labels_sample = labels
    else:
        graphs_sample = graphs
        labels_sample = labels
    predictor = Predictor(model)
    predictor.evaluate_detailed(graphs_sample, labels_sample, name="Test (sample)")
    preds = model.predict(graphs_sample)
    uniq_p, cnt_p = np.unique(preds, return_counts=True)
    print(f"pred counts: {dict(zip(uniq_p.tolist(), cnt_p.tolist()))}")
    try:
        scores = model.tm.score(graphs_sample)
        print(
            f"scores class0 range: [{scores[:,0].min():.2f}, {scores[:,0].max():.2f}], "
            f"class1 range: [{scores[:,1].min():.2f}, {scores[:,1].max():.2f}]"
        )
    except Exception as exc:
        print(f"[WARN] score() failed: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-size", type=int, default=7)
    ap.add_argument("--npz-train", type=str, default="data/train_games_7x7.npz")
    ap.add_argument("--npz-test", type=str, default="data/test_games_7x7.npz")
    ap.add_argument("--gtm-train", type=str, default=None)
    ap.add_argument("--gtm-test", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--sample", type=int, default=5000, help="sample size for model eval; 0 = full")
    args = ap.parse_args()

    summarize_npz(args.npz_train, args.board_size, "train npz")
    summarize_npz(args.npz_test, args.board_size, "test npz")

    gtm_train = summarize_gtm(args.gtm_train, "train GTM") if args.gtm_train else None
    gtm_test = summarize_gtm(args.gtm_test, "test GTM") if args.gtm_test else None

    if args.model and gtm_test:
        evaluate_model(args.model, gtm_test, args.sample)


if __name__ == "__main__":
    main()

