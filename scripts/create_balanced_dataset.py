"""
Create balanced dataset by oversampling minority class.

This script takes existing training data and creates a balanced version
by replicating minority class samples.
"""

import os
import sys
import pickle
import numpy as np

# CUDA setup
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from GraphTsetlinMachine.graphs import Graphs


def balance_dataset(input_path, output_path):
    """
    Balance dataset by oversampling minority class.

    Args:
        input_path: Path to input .pkl file
        output_path: Path to output balanced .pkl file
    """
    print(f"\nLoading {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    graphs = data['graphs']
    labels = data['labels']

    print(f"Original dataset: {len(labels)} samples")

    # Count classes
    unique_classes, class_counts = np.unique(labels, return_counts=True)

    print(f"\nClass distribution (before):")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Player {cls}: {count} samples ({100*count/len(labels):.1f}%)")

    # Find minority and majority classes
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]

    minority_indices = np.where(labels == minority_class)[0]
    majority_indices = np.where(labels == majority_class)[0]

    n_minority = len(minority_indices)
    n_majority = len(majority_indices)

    # Calculate balanced indices
    balanced_indices = []

    # Add all majority samples
    balanced_indices.append(majority_indices)

    # Replicate minority samples to match majority
    n_replications = n_majority // n_minority
    remainder = n_majority % n_minority

    print(f"\nBalancing strategy:")
    print(f"  Minority class (Player {minority_class}): {n_minority} samples")
    print(f"  Majority class (Player {majority_class}): {n_majority} samples")
    print(f"  Replication factor: {n_replications}x + {remainder} extra")

    # Full replications
    replicated = np.tile(minority_indices, n_replications)
    balanced_indices.append(replicated)

    # Add remainder
    if remainder > 0:
        extra = np.random.choice(minority_indices, remainder, replace=False)
        balanced_indices.append(extra)

    # Combine and shuffle
    balanced_indices = np.concatenate(balanced_indices)
    np.random.shuffle(balanced_indices)

    # Create balanced labels
    balanced_labels = labels[balanced_indices]

    print(f"\nBalanced dataset: {len(balanced_labels)} samples")
    print(f"\nClass distribution (after):")
    for cls in unique_classes:
        count = np.sum(balanced_labels == cls)
        print(f"  Player {cls}: {count} samples ({100*count/len(balanced_labels):.1f}%)")

    # Save balanced dataset
    balanced_data = {
        'graphs': graphs,  # Note: graphs object is shared, indices handled at training time
        'labels': balanced_labels,
        'indices': balanced_indices,  # Store indices for reference
        'original_size': len(labels),
        'balanced': True
    }

    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(balanced_data, f)

    print(f"[OK] Balanced dataset saved!")
    print(f"\nNote: The graphs object is not duplicated (memory efficient).")
    print(f"Use 'indices' to map balanced samples to original graphs during training.")


def main():
    board_size = 5
    stage = "end"

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    input_path = os.path.join(data_dir, f"train_gtm_{board_size}x{board_size}_{stage}.pkl")
    output_path = os.path.join(data_dir, f"train_gtm_{board_size}x{board_size}_{stage}_balanced.pkl")

    print("="*60)
    print("CREATE BALANCED DATASET")
    print("="*60)
    print(f"Board size: {board_size}x{board_size}")
    print(f"Stage: {stage}")
    print("="*60)

    balance_dataset(input_path, output_path)

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\nBalanced dataset: {output_path}")
    print(f"\nTo use this dataset, modify your training script to load:")
    print(f"  {os.path.basename(output_path)}")
    print("="*60)


if __name__ == "__main__":
    main()
