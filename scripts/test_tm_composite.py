"""
Test TM Composite (Ensemble Strategy)

Phase 2 Experiment: Compare single model vs ensemble of specialists
"""

import argparse
import os
import sys
import pickle

# CUDA setup
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import (
    HexGraphTM,
    HexTMComposite,
    create_depth_diverse_composite,
    create_specificity_diverse_composite,
    create_mixed_composite
)


def load_data(board_size, stage, data_dir='data'):
    """Load training and test data."""
    train_path = f"{data_dir}/train_gtm_{board_size}x{board_size}_{stage}.pkl"
    test_path = f"{data_dir}/test_gtm_{board_size}x{board_size}_{stage}.pkl"

    print(f"Loading {train_path}...")
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)

    print(f"Loading {test_path}...")
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    return (train_data['graphs'], train_data['labels'],
            test_data['graphs'], test_data['labels'])


def test_baseline(train_graphs, train_labels, test_graphs, test_labels,
                  clauses=200, epochs=100):
    """Test baseline single model."""
    print(f"\n{'='*60}")
    print(f"BASELINE: Single Model ({clauses} clauses)")
    print(f"{'='*60}\n")

    model = HexGraphTM(
        number_of_clauses=clauses,
        T=15000,
        s=10.0,
        depth=3,
        message_size=256,
        message_bits=2,
        max_included_literals=255
    )

    print(f"Training for {epochs} epochs...")
    model.fit(train_graphs, train_labels, epochs=epochs)

    train_preds = model.predict(train_graphs)
    test_preds = model.predict(test_graphs)

    train_acc = 100.0 * (train_preds == train_labels).sum() / len(train_labels)
    test_acc = 100.0 * (test_preds == test_labels).sum() / len(test_labels)

    print(f"\nBaseline Results:")
    print(f"  Train Accuracy: {train_acc:.2f}%")
    print(f"  Test Accuracy:  {test_acc:.2f}%")
    print(f"  Total Clauses:  {clauses}")

    return test_acc


def test_composite(train_graphs, train_labels, test_graphs, test_labels,
                   composite_type='depth', base_clauses=50, epochs=100):
    """Test composite model."""
    print(f"\n{'='*60}")
    print(f"COMPOSITE: {composite_type.upper()} Diverse")
    print(f"{'='*60}\n")

    if composite_type == 'depth':
        composite = create_depth_diverse_composite(base_clauses=base_clauses)
    elif composite_type == 'specificity':
        composite = create_specificity_diverse_composite(base_clauses=base_clauses)
    elif composite_type == 'mixed':
        composite = create_mixed_composite(base_clauses=base_clauses)
    else:
        raise ValueError(f"Unknown composite type: {composite_type}")

    print(f"\nTraining composite for {epochs} epochs...")
    composite.fit(train_graphs, train_labels, epochs=epochs)

    # Evaluate
    train_acc = composite.evaluate(train_graphs, train_labels)
    test_acc = composite.evaluate(test_graphs, test_labels)

    print(f"\nComposite Results:")
    print(f"  Train Accuracy: {train_acc:.2f}%")
    print(f"  Test Accuracy:  {test_acc:.2f}%")
    print(f"  Total Clauses:  {composite.total_clauses()}")

    # Show specialist performance
    composite.print_specialist_performance(test_graphs, test_labels)

    return test_acc


def main():
    parser = argparse.ArgumentParser(description='Test TM Composite vs Single Model')
    parser.add_argument('--board-size', type=int, default=5,
                        help='Board size (default: 5)')
    parser.add_argument('--stage', type=str, default='end',
                        help='Game stage (default: end)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--baseline-clauses', type=int, default=200,
                        help='Clauses for baseline model (default: 200)')
    parser.add_argument('--composite-clauses', type=int, default=50,
                        help='Clauses per specialist (default: 50)')
    parser.add_argument('--composite-type', type=str, default='depth',
                        choices=['depth', 'specificity', 'mixed'],
                        help='Type of composite (default: depth)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory (default: data)')

    args = parser.parse_args()

    # Load data
    print(f"\n{'='*60}")
    print(f"PHASE 2: TM COMPOSITE EXPERIMENT")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    print(f"  Board: {args.board_size}x{args.board_size}")
    print(f"  Stage: {args.stage}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Baseline clauses: {args.baseline_clauses}")
    print(f"  Composite type: {args.composite_type}")
    print(f"  Clauses per specialist: {args.composite_clauses}")

    train_graphs, train_labels, test_graphs, test_labels = load_data(
        args.board_size, args.stage, args.data_dir
    )

    print(f"\nDataset:")
    print(f"  Training samples: {len(train_labels)}")
    print(f"  Test samples: {len(test_labels)}")

    # Test baseline
    baseline_acc = test_baseline(
        train_graphs, train_labels, test_graphs, test_labels,
        clauses=args.baseline_clauses,
        epochs=args.epochs
    )

    # Test composite
    composite_acc = test_composite(
        train_graphs, train_labels, test_graphs, test_labels,
        composite_type=args.composite_type,
        base_clauses=args.composite_clauses,
        epochs=args.epochs
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\nBaseline Single Model:")
    print(f"  Clauses: {args.baseline_clauses}")
    print(f"  Test Accuracy: {baseline_acc:.2f}%")

    if args.composite_type == 'depth':
        n_specialists = 4
    elif args.composite_type == 'specificity':
        n_specialists = 4
    elif args.composite_type == 'mixed':
        n_specialists = 5

    total_composite_clauses = args.composite_clauses * n_specialists

    print(f"\nComposite Ensemble ({args.composite_type}):")
    print(f"  Specialists: {n_specialists}")
    print(f"  Clauses per specialist: {args.composite_clauses}")
    print(f"  Total Clauses: {total_composite_clauses}")
    print(f"  Test Accuracy: {composite_acc:.2f}%")

    acc_diff = composite_acc - baseline_acc
    clause_reduction = (1 - total_composite_clauses / args.baseline_clauses) * 100

    print(f"\nImprovement:")
    print(f"  Accuracy change: {acc_diff:+.2f}%")
    print(f"  Clause reduction: {clause_reduction:.1f}%")

    if composite_acc >= baseline_acc and total_composite_clauses < args.baseline_clauses:
        print(f"\n[SUCCESS] Composite achieves similar/better accuracy with fewer clauses!")
    elif composite_acc >= baseline_acc:
        print(f"\n[GOOD] Composite achieves better accuracy (but more clauses)")
    elif total_composite_clauses < args.baseline_clauses:
        print(f"\n[TRADE-OFF] Composite uses fewer clauses (but lower accuracy)")
    else:
        print(f"\n[POOR] Composite underperforms baseline")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
