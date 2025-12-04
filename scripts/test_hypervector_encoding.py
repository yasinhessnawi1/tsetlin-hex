"""
Phase 3: Hypervector Encoding Optimization

Tests different hypervector configurations to find optimal encoding.
According to HVTM paper, this can give 5-20x clause reduction.

NOTE: Requires rebuilding datasets with different hypervector settings!
"""

import argparse
import os
import sys
import pickle
import subprocess

# CUDA setup
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import HexGraphTM


def test_configuration(
    board_size,
    stage,
    hv_size,
    hv_bits,
    clauses,
    epochs,
    data_dir='data'
):
    """
    Test a specific hypervector configuration.

    NOTE: Dataset must already exist with these HV settings!
    """
    print(f"\n{'='*60}")
    print(f"TESTING: HV_size={hv_size}, HV_bits={hv_bits}, clauses={clauses}")
    print(f"{'='*60}\n")

    # Load data (must exist!)
    train_path = f"{data_dir}/train_gtm_{board_size}x{board_size}_{stage}.pkl"
    test_path = f"{data_dir}/test_gtm_{board_size}x{board_size}_{stage}.pkl"

    if not os.path.exists(train_path):
        print(f"ERROR: Dataset not found: {train_path}")
        print(f"You need to rebuild dataset with HV_size={hv_size}, HV_bits={hv_bits}")
        return None

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    train_graphs, train_labels = train_data['graphs'], train_data['labels']
    test_graphs, test_labels = test_data['graphs'], test_data['labels']

    # Check if HV settings match
    actual_hv_size = train_graphs.hypervector_size
    if actual_hv_size != hv_size:
        print(f"WARNING: Dataset has HV_size={actual_hv_size}, not {hv_size}")
        print(f"Skipping this configuration.")
        return None

    # Create model
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

    # Evaluate
    train_preds = model.predict(train_graphs)
    test_preds = model.predict(test_graphs)

    train_acc = 100.0 * (train_preds == train_labels).sum() / len(train_labels)
    test_acc = 100.0 * (test_preds == test_labels).sum() / len(test_labels)

    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc:.2f}%")
    print(f"  Test Accuracy:  {test_acc:.2f}%")
    print(f"  HV Size: {hv_size}")
    print(f"  HV Bits: {hv_bits}")
    print(f"  Clauses: {clauses}")

    return {
        'hv_size': hv_size,
        'hv_bits': hv_bits,
        'clauses': clauses,
        'train_acc': train_acc,
        'test_acc': test_acc
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test hypervector encoding configurations',
        epilog="""
NOTE: This script tests different hypervector configurations.
Your datasets are built with specific HV settings (size=256, bits=4 currently).

To test different settings, you need to rebuild datasets with:
    python scripts/1b_build_gtm_datasets.py --hypervector-size SIZE --hypervector-bits BITS

Then run this script to compare results.
        """
    )

    parser.add_argument('--board-size', type=int, default=5,
                        help='Board size (default: 5)')
    parser.add_argument('--stage', type=str, default='end',
                        help='Game stage (default: end)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--clauses', type=int, default=200,
                        help='Number of clauses (default: 200)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory (default: data)')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"PHASE 3: HYPERVECTOR ENCODING OPTIMIZATION")
    print(f"{'='*60}\n")

    print("Current dataset info:")
    train_path = f"{args.data_dir}/train_gtm_{args.board_size}x{args.board_size}_{args.stage}.pkl"
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)

    graphs = train_data['graphs']
    print(f"  Hypervector size: {graphs.hypervector_size}")
    print(f"  Number of samples: {len(train_data['labels'])}")

    # Test current configuration
    result = test_configuration(
        args.board_size,
        args.stage,
        hv_size=graphs.hypervector_size,
        hv_bits=4,  # Can't detect from graphs, assume 4
        clauses=args.clauses,
        epochs=args.epochs,
        data_dir=args.data_dir
    )

    print(f"\n{'='*60}")
    print("TO TEST MORE CONFIGURATIONS:")
    print(f"{'='*60}")
    print("\n1. Rebuild dataset with different HV settings:")
    print("   python scripts/1b_build_gtm_datasets.py --board-size 5 \\")
    print("      --hypervector-size 512 --hypervector-bits 8")
    print("\n2. Test the new configuration:")
    print("   python scripts/test_hypervector_encoding.py --clauses 200")
    print("\n3. Compare results to find optimal HV encoding")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
