"""
Train TM Composite (Ensemble Strategy) for Hex winner prediction.

This script trains ensemble models using multiple specialized Tsetlin Machines.
Three composite strategies are available:
- Depth-diverse: Specialists with different message passing depths (1, 2, 3, 4)
- Specificity-diverse: Specialists with different specificity values (5, 10, 15, 20)
- Mixed: Combination of depth and specificity diversity (5 specialists)

Usage:
    python scripts/4_train_composite.py --board-size 5 --stage 0 --composite-type depth
    python scripts/4_train_composite.py --board-size 5 --stage all --composite-type mixed
"""

import argparse
import os
import sys
import pickle
import numpy as np

# IMPORTANT: Set CUDA device BEFORE any imports that use CUDA
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# CRITICAL: Disable CUDA cache to prevent nvcc compilation issues
os.environ['CUDA_CACHE_DISABLE'] = '1'

# CRITICAL: Force CUDA initialization BEFORE importing GraphTsetlinMachine
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except Exception as e:
    print(f"[WARNING] CUDA pre-initialization failed: {e}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import (
    HexTMComposite,
    create_depth_diverse_composite,
    create_specificity_diverse_composite,
    create_mixed_composite
)


def load_data(board_size, stage, data_dir='data'):
    """Load training and test data with fallback for stage naming."""
    train_path = f"{data_dir}/train_gtm_{board_size}x{board_size}_{stage}.pkl"
    test_path = f"{data_dir}/test_gtm_{board_size}x{board_size}_{stage}.pkl"

    # Fallback: if stage is "end" or "0" and file not found, try the other
    if not os.path.exists(train_path) and stage in ["end", "0"]:
        alt_stage = "0" if stage == "end" else "end"
        alt_train_path = f"{data_dir}/train_gtm_{board_size}x{board_size}_{alt_stage}.pkl"
        if os.path.exists(alt_train_path):
            print(f"[INFO] File not found with stage '{stage}', using '{alt_stage}' instead")
            train_path = alt_train_path
            test_path = f"{data_dir}/test_gtm_{board_size}x{board_size}_{alt_stage}.pkl"

    if not os.path.exists(train_path):
        print(f"\nERROR: Training dataset not found at {train_path}")
        print("Please run 1b_build_gtm_datasets.py first!")
        return None, None, None, None

    if not os.path.exists(test_path):
        print(f"\nERROR: Test dataset not found at {test_path}")
        print("Please run 1b_build_gtm_datasets.py first!")
        return None, None, None, None

    print(f"Loading {train_path}...")
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)

    print(f"Loading {test_path}...")
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    return (train_data['graphs'], train_data['labels'],
            test_data['graphs'], test_data['labels'])


def train_composite_stage(
    stage_name: str,
    board_size: int,
    composite_type: str,
    base_clauses: int,
    epochs: int,
    T: int,
    s: float,
    depth: int,
    message_size: int,
    message_bits: int,
    data_dir: str = 'data',
    save_model: bool = False
):
    """
    Train a composite model for a specific game stage.

    Args:
        stage_name: Name of the stage (e.g., 'end', '0', '-2', '-5')
        board_size: Size of the board
        composite_type: Type of composite ('depth', 'specificity', 'mixed')
        base_clauses: Number of clauses per specialist
        epochs: Number of training epochs
        T: Threshold parameter
        s: Specificity parameter (baseline for composite)
        depth: Depth parameter (baseline for composite)
        message_size: Message size
        message_bits: Message bits
        data_dir: Data directory
        save_model: Whether to save the trained model

    Returns:
        Dictionary with training results
    """
    print("\n" + "="*70)
    print(f"TRAINING COMPOSITE MODEL FOR STAGE: {stage_name}")
    print("="*70)

    # Load data
    train_graphs, train_labels, test_graphs, test_labels = load_data(
        board_size, stage_name, data_dir
    )

    if train_graphs is None:
        return None

    print(f"\nDataset loaded:")
    print(f"  Training samples: {len(train_labels)}")
    print(f"  Test samples: {len(test_labels)}")

    # Show class distribution
    train_p0 = np.sum(train_labels == 0)
    train_p1 = np.sum(train_labels == 1)
    test_p0 = np.sum(test_labels == 0)
    test_p1 = np.sum(test_labels == 1)

    print(f"\nClass distribution:")
    print(f"  Train: P0={train_p0} ({100*train_p0/len(train_labels):.1f}%), "
          f"P1={train_p1} ({100*train_p1/len(train_labels):.1f}%)")
    print(f"  Test:  P0={test_p0} ({100*test_p0/len(test_labels):.1f}%), "
          f"P1={test_p1} ({100*test_p1/len(test_labels):.1f}%)")

    # Create composite model
    print(f"\n{'='*70}")
    print(f"INITIALIZING {composite_type.upper()} COMPOSITE")
    print(f"{'='*70}")

    if composite_type == 'depth':
        composite = create_depth_diverse_composite(
            base_clauses=base_clauses,
            T=T,
            s=s,
            message_size=message_size,
            message_bits=message_bits
        )
        print(f"\nDepth-Diverse Composite (4 specialists):")
        print(f"  Specialist 1: depth=1, {base_clauses} clauses (very local)")
        print(f"  Specialist 2: depth=2, {base_clauses} clauses (local)")
        print(f"  Specialist 3: depth=3, {base_clauses} clauses (regional)")
        print(f"  Specialist 4: depth=4, {base_clauses} clauses (long-range)")

    elif composite_type == 'specificity':
        composite = create_specificity_diverse_composite(
            base_clauses=base_clauses,
            T=T,
            depth=depth,
            message_size=message_size,
            message_bits=message_bits
        )
        print(f"\nSpecificity-Diverse Composite (4 specialists):")
        print(f"  Specialist 1: s=5.0, {base_clauses} clauses (coarse patterns)")
        print(f"  Specialist 2: s=10.0, {base_clauses} clauses (medium specificity)")
        print(f"  Specialist 3: s=15.0, {base_clauses} clauses (fine patterns)")
        print(f"  Specialist 4: s=20.0, {base_clauses} clauses (very fine patterns)")

    elif composite_type == 'mixed':
        composite = create_mixed_composite(
            base_clauses=base_clauses,
            T=T,
            message_size=message_size,
            message_bits=message_bits
        )
        print(f"\nMixed Composite (5 specialists):")
        print(f"  Specialist 1: depth=2, s=5.0, {base_clauses} clauses (shallow + general)")
        print(f"  Specialist 2: depth=2, s=10.0, {base_clauses} clauses (shallow + balanced)")
        print(f"  Specialist 3: depth=3, s=10.0, {base_clauses} clauses (medium + balanced)")
        print(f"  Specialist 4: depth=3, s=15.0, {base_clauses} clauses (medium + specific)")
        print(f"  Specialist 5: depth=4, s=10.0, {base_clauses} clauses (deep + balanced)")

    else:
        raise ValueError(f"Unknown composite type: {composite_type}")

    print(f"\nTotal clauses across all specialists: {composite.total_clauses()}")

    # Train composite
    print(f"\n{'='*70}")
    print(f"TRAINING COMPOSITE ({epochs} epochs)")
    print(f"{'='*70}\n")

    composite.fit(train_graphs, train_labels, epochs=epochs)

    # Evaluate
    print(f"\n{'='*70}")
    print(f"EVALUATION")
    print(f"{'='*70}\n")

    train_acc = composite.evaluate(train_graphs, train_labels)
    test_acc = composite.evaluate(test_graphs, test_labels)

    # Per-class accuracy
    test_preds = composite.predict(test_graphs)
    test_p0_acc = 100.0 * np.sum((test_preds == 0) & (test_labels == 0)) / test_p0 if test_p0 > 0 else 0
    test_p1_acc = 100.0 * np.sum((test_preds == 1) & (test_labels == 1)) / test_p1 if test_p1 > 0 else 0
    gap = abs(test_p0_acc - test_p1_acc)

    print(f"Overall Results:")
    print(f"  Train Accuracy: {train_acc:.2f}%")
    print(f"  Test Accuracy:  {test_acc:.2f}%")
    print(f"  Total Clauses:  {composite.total_clauses()}")

    print(f"\nPer-Class Results (Test):")
    print(f"  P0 Accuracy: {test_p0_acc:.2f}%")
    print(f"  P1 Accuracy: {test_p1_acc:.2f}%")
    print(f"  Gap:         {gap:.2f}%")

    # Show specialist performance
    print(f"\n{'='*70}")
    print(f"SPECIALIST PERFORMANCE")
    print(f"{'='*70}\n")
    composite.print_specialist_performance(test_graphs, test_labels)

    # Save model (disabled due to CUDA pickling issues)
    if save_model:
        print("\n[INFO] Model saving disabled (PyCUDA pickling not supported)")
        print("[INFO] For competition, only accuracy numbers are needed")

    return {
        'composite': composite,
        'stage': stage_name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'p0_acc': test_p0_acc,
        'p1_acc': test_p1_acc,
        'gap': gap,
        'total_clauses': composite.total_clauses()
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train TM Composite models for Hex winner prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train depth-diverse composite for end game
  python scripts/4_train_composite.py --board-size 5 --stage 0 --composite-type depth

  # Train mixed composite for all stages
  python scripts/4_train_composite.py --board-size 5 --stage all --composite-type mixed --epochs 200

  # Train specificity-diverse composite with custom parameters
  python scripts/4_train_composite.py --board-size 5 --stage 0 --composite-type specificity \\
      --clauses-per-specialist 100 --T 15000 --s 10.0 --depth 3 --epochs 200
        """
    )

    # Data parameters
    parser.add_argument('--board-size', type=int, default=5,
                        help='Size of the Hex board (default: 5)')
    parser.add_argument('--stage', type=str, default='0',
                        help='Stage to train: 0, end, -2, -5, or all (default: 0)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing GTM datasets (default: data)')

    # Composite parameters
    parser.add_argument('--composite-type', type=str, default='depth',
                        choices=['depth', 'specificity', 'mixed'],
                        help='Type of composite ensemble (default: depth)')
    parser.add_argument('--clauses-per-specialist', type=int, default=200,
                        help='Number of clauses per specialist (default: 200)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--T', type=int, default=10000,
                        help='Threshold T (default: 10000)')
    parser.add_argument('--s', type=float, default=10.0,
                        help='Baseline specificity s (default: 10.0)')
    parser.add_argument('--depth', type=int, default=3,
                        help='Baseline message passing depth (default: 3)')

    # Advanced parameters
    parser.add_argument('--message-size', type=int, default=256,
                        help='Message size (default: 256)')
    parser.add_argument('--message-bits', type=int, default=2,
                        help='Message bits (default: 2)')

    # Informational only (hypervectors set at dataset build time)
    parser.add_argument('--hypervector-size', type=int, default=None,
                        help='INFO ONLY: Hypervector size used in dataset (set at build time)')
    parser.add_argument('--hypervector-bits', type=int, default=None,
                        help='INFO ONLY: Hypervector bits used in dataset (set at build time)')

    args = parser.parse_args()

    # Print configuration
    print("\n" + "="*70)
    print("TM COMPOSITE TRAINING CONFIGURATION")
    print("="*70)
    print(f"\nData:")
    print(f"  Board size: {args.board_size}x{args.board_size}")
    print(f"  Stage(s): {args.stage}")
    print(f"  Data directory: {args.data_dir}")

    print(f"\nComposite Configuration:")
    print(f"  Type: {args.composite_type}")
    print(f"  Clauses per specialist: {args.clauses_per_specialist}")

    if args.composite_type == 'depth':
        n_specialists = 4
    elif args.composite_type == 'specificity':
        n_specialists = 4
    elif args.composite_type == 'mixed':
        n_specialists = 5

    total_clauses = args.clauses_per_specialist * n_specialists
    print(f"  Number of specialists: {n_specialists}")
    print(f"  Total clauses: {total_clauses}")

    print(f"\nTraining Parameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  T: {args.T}")
    print(f"  Baseline s: {args.s}")
    print(f"  Baseline depth: {args.depth}")

    print(f"\nAdvanced Parameters:")
    print(f"  Message size: {args.message_size}")
    print(f"  Message bits: {args.message_bits}")

    # Print hypervector info if provided (informational only)
    if args.hypervector_size or args.hypervector_bits:
        print(f"\nHypervector Settings (from dataset build):")
        if args.hypervector_size:
            print(f"  Hypervector size: {args.hypervector_size}")
        if args.hypervector_bits:
            print(f"  Hypervector bits: {args.hypervector_bits}")
        print("  Note: These were set when building the dataset (.pkl files)")

    print("="*70)

    # Determine which stages to train
    if args.stage == 'all':
        stages = ['0', '-2', '-5']
    else:
        stages = [args.stage]

    print(f"\nTraining stages: {stages}\n")

    # Train each stage
    results = {}
    for stage in stages:
        result = train_composite_stage(
            stage_name=stage,
            board_size=args.board_size,
            composite_type=args.composite_type,
            base_clauses=args.clauses_per_specialist,
            epochs=args.epochs,
            T=args.T,
            s=args.s,
            depth=args.depth,
            message_size=args.message_size,
            message_bits=args.message_bits,
            data_dir=args.data_dir,
            save_model=False
        )

        if result is not None:
            results[stage] = result

        if len(stages) > 1 and stage != stages[-1]:
            print(f"\n{'='*70}")
            print(f"Completed {len(results)}/{len(stages)} stages")
            print(f"{'='*70}\n")

    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)

    if results:
        print(f"\n{'Stage':<10} {'Train Acc':<12} {'Test Acc':<12} {'P0 Acc':<10} {'P1 Acc':<10} {'Gap':<8} {'Clauses':<10}")
        print("-" * 80)

        for stage, result in results.items():
            print(f"{stage:<10} {result['train_acc']:>10.2f}%  {result['test_acc']:>10.2f}%  "
                  f"{result['p0_acc']:>8.2f}%  {result['p1_acc']:>8.2f}%  "
                  f"{result['gap']:>6.2f}%  {result['total_clauses']:>8}")

        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"\nComposite Type: {args.composite_type}")
        print(f"Specialists: {n_specialists}")
        print(f"Clauses per specialist: {args.clauses_per_specialist}")
        print(f"Total clauses: {total_clauses}")
        print("="*70 + "\n")
    else:
        print("\n[ERROR] No models were successfully trained!")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
