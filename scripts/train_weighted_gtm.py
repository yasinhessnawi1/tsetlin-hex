"""
Train Weighted GTM to handle class imbalance.

This script demonstrates how to use WeightedGTM for your Hex winner prediction task.
The weighted approach helps handle the Player 0 vs Player 1 class imbalance issue.
"""

import sys
import os
import time
import numpy as np
import pickle

# CUDA setup
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import HexGraphTM


def balance_data_by_undersampling(graphs, labels):
    """
    Balance classes by removing excess majority class samples.

    This reduces total training data but ensures equal class representation.
    """
    # Count classes
    unique_classes, class_counts = np.unique(labels, return_counts=True)

    print(f"\nOriginal class distribution:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Player {cls}: {count} samples ({100*count/len(labels):.1f}%)")

    # Find minority class size
    min_count = class_counts.min()

    print(f"\nBalancing strategy: Undersample to {min_count} samples per class")

    # Get indices for each class
    balanced_indices = []
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        # Randomly select min_count samples
        selected = np.random.choice(cls_indices, min_count, replace=False)
        balanced_indices.append(selected)

    # Combine and shuffle
    balanced_indices = np.concatenate(balanced_indices)
    np.random.shuffle(balanced_indices)

    # Create balanced labels
    balanced_labels = labels[balanced_indices]

    # Update graphs object
    graphs.number_of_graphs = len(balanced_indices)

    print(f"\nBalanced class distribution:")
    for cls in unique_classes:
        count = np.sum(balanced_labels == cls)
        print(f"  Player {cls}: {count} samples ({100*count/len(balanced_labels):.1f}%)")

    print(f"\nTotal samples: {len(labels)} → {len(balanced_labels)} (-{len(labels) - len(balanced_labels)})")

    return graphs, balanced_labels, balanced_indices


def main():
    # Configuration
    board_size = 5
    stage = "end"
    train_samples = 10000
    test_samples = 2000

    # Weighted GTM parameters
    clauses = 200  # Same as your best baseline
    T = 10000      # Same as your best baseline
    s = 10.0       # Same as your best baseline
    depth = 3      # Same as your best baseline
    epochs = 200

    print("="*60)
    print("BALANCED GTM TRAINING (UNDERSAMPLING)")
    print("="*60)
    print(f"Board size: {board_size}x{board_size}")
    print(f"Stage: {stage}")
    print(f"Training samples: {train_samples}")
    print(f"Test samples: {test_samples}")
    print()
    print("GTM Parameters:")
    print(f"  Clauses: {clauses}")
    print(f"  T: {T}")
    print(f"  s: {s}")
    print(f"  Depth: {depth}")
    print(f"  Epochs: {epochs}")
    print()
    print("Balancing Strategy: Remove excess majority class samples")
    print("="*60)

    # Load data
    print("\nLoading data...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    train_path = os.path.join(data_dir, f"train_gtm_{board_size}x{board_size}_{stage}.pkl")
    test_path = os.path.join(data_dir, f"test_gtm_{board_size}x{board_size}_{stage}.pkl")

    print(f"Loading {train_path}...")
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)

    print(f"Loading {test_path}...")
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    # Extract graphs and labels
    train_graphs = train_data['graphs']
    train_labels = train_data['labels']
    test_graphs = test_data['graphs']
    test_labels = test_data['labels']

    # Limit samples if requested
    if train_samples and len(train_labels) > train_samples:
        train_graphs.number_of_graphs = train_samples
        train_labels = train_labels[:train_samples]

    if test_samples and len(test_labels) > test_samples:
        test_graphs.number_of_graphs = test_samples
        test_labels = test_labels[:test_samples]

    print(f"Training samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")

    # Check class distribution
    train_p0 = np.sum(train_labels == 0)
    train_p1 = np.sum(train_labels == 1)
    test_p0 = np.sum(test_labels == 0)
    test_p1 = np.sum(test_labels == 1)

    # Balance training data by undersampling majority class
    train_graphs, train_labels, _ = balance_data_by_undersampling(train_graphs, train_labels)

    # Update class counts after balancing
    train_p0 = np.sum(train_labels == 0)
    train_p1 = np.sum(train_labels == 1)

    # Test data remains unbalanced (for fair evaluation)
    print(f"\nTest set (unbalanced for fair evaluation):")
    print(f"  Player 0: {test_p0} ({100*test_p0/len(test_labels):.1f}%)")
    print(f"  Player 1: {test_p1} ({100*test_p1/len(test_labels):.1f}%)")

    # Create GTM model
    print("\n" + "="*60)
    print("CREATING GTM MODEL")
    print("="*60)

    model = HexGraphTM(
        number_of_clauses=clauses,
        T=T,
        s=s,
        depth=depth,
        message_size=256,
        message_bits=2,
        max_included_literals=255,
        grid=(208, 1, 1),
        block=(128, 1, 1)
    )

    print(model)

    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    start_time = time.time()

    model.fit(train_graphs, train_labels, epochs=epochs)

    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per epoch: {total_time/epochs:.2f}s")

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    train_acc = model.evaluate(train_graphs, train_labels)
    test_acc = model.evaluate(test_graphs, test_labels)

    print(f"\nFinal Training Accuracy: {train_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # Check per-class accuracy
    train_preds = model.predict(train_graphs)
    test_preds = model.predict(test_graphs)

    train_p0_acc = 100 * np.sum((train_preds == 0) & (train_labels == 0)) / train_p0
    train_p1_acc = 100 * np.sum((train_preds == 1) & (train_labels == 1)) / train_p1
    test_p0_acc = 100 * np.sum((test_preds == 0) & (test_labels == 0)) / test_p0
    test_p1_acc = 100 * np.sum((test_preds == 1) & (test_labels == 1)) / test_p1

    print("\nPer-class Accuracy (Train):")
    print(f"  Player 0: {train_p0_acc:.2f}%")
    print(f"  Player 1: {train_p1_acc:.2f}%")
    print(f"  Gap: {abs(train_p0_acc - train_p1_acc):.2f}%")

    print("\nPer-class Accuracy (Test):")
    print(f"  Player 0: {test_p0_acc:.2f}%")
    print(f"  Player 1: {test_p1_acc:.2f}%")
    print(f"  Gap: {abs(test_p0_acc - test_p1_acc):.2f}%")

    print("="*60)


if __name__ == "__main__":
    main()
