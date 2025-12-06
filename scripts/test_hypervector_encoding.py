"""
HVTM Experiment: Hypervector Size & Reasoning by Elimination

Tests 5 different hypervector sizes with RbE mode (s=1.0):
- 128, 256, 512, 1024, 2048

For each size:
1. Generate balanced game data
2. Build GTM graphs with specific HV size
3. Train model with 800 clauses, depth=3, s=1.0, T=5000, 30 epochs
4. Compare results

Expected: Larger HV sizes should need fewer clauses (5-20x reduction per HVTM paper)
"""

import os
import sys
import subprocess
import pickle

# CUDA setup
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

from src.models import HexGraphTM


def run_command(cmd, desc):
    """Run a command via subprocess."""
    print(f"\n{'='*60}")
    print(f"{desc}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n[ERROR] {desc} failed with code {result.returncode}")
        return False

    print(f"\n[OK] {desc} completed")
    return True


def test_hvtm_configuration(hv_size, hv_bits=2, board_size=5, stage="0"):
    """
    Test one HVTM configuration:
    1. Generate balanced games
    2. Build graphs with specified HV size
    3. Train and evaluate model

    Returns test accuracy or None if failed
    """
    print(f"\n{'='*70}")
    print(f"HVTM EXPERIMENT: HV_size={hv_size}, HV_bits={hv_bits}")
    print(f"{'='*70}")

    # Step 1: Generate balanced games
    if not run_command(
        ['python', 'scripts/1_generate_games.py',
         '--board-size', str(board_size),
         '--num-train', '10000',
         '--num-test', '2000',
         '--save-states', stage],
        f"Generate balanced games"
    ):
        return None

    # Step 2: Build GTM graphs with specific HV settings
    if not run_command(
        ['python', 'scripts/1b_build_gtm_datasets.py',
         '--board-size', str(board_size),
         '--hypervector-size', str(hv_size),
         '--hypervector-bits', str(hv_bits),
         '--stages', stage],
        f"Build GTM graphs (HV_size={hv_size}, HV_bits={hv_bits})"
    ):
        return None

    # Step 3: Load data
    train_path = f'data/train_gtm_{board_size}x{board_size}_{stage}.pkl'
    test_path = f'data/test_gtm_{board_size}x{board_size}_{stage}.pkl'

    print(f"\nLoading {train_path}...")
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)

    print(f"Loading {test_path}...")
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    train_graphs = train_data['graphs']
    train_labels = train_data['labels']
    test_graphs = test_data['graphs']
    test_labels = test_data['labels']

    # Verify HV settings
    actual_hv_size = train_graphs.hypervector_size
    if actual_hv_size != hv_size:
        print(f"[ERROR] Dataset has HV_size={actual_hv_size}, expected {hv_size}")
        return None

    print(f"\nDataset loaded:")
    print(f"  Train samples: {len(train_labels)}")
    print(f"  Test samples: {len(test_labels)}")
    print(f"  HV size: {actual_hv_size}")

    # Step 4: Train model with HVTM settings
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL (HVTM Configuration)")
    print(f"{'='*60}")
    print(f"  Clauses: 800")
    print(f"  T: 5000")
    print(f"  s: 1.0 (Reasoning by Elimination)")
    print(f"  Depth: 3")
    print(f"  Epochs: 30")
    print(f"  HV size: {hv_size}")
    print(f"  HV bits: {hv_bits}")

    model = HexGraphTM(
        number_of_clauses=800,
        T=5000,
        s=1.0,  # RbE mode
        depth=3,
        message_size=hv_size,
        message_bits=hv_bits,
        max_included_literals=255,
        grid=(208, 1, 1),
        block=(128, 1, 1)
    )

    print(f"\nTraining...")
    model.fit(train_graphs, train_labels, epochs=30)

    # Step 5: Evaluate
    train_preds = model.predict(train_graphs)
    test_preds = model.predict(test_graphs)

    import numpy as np
    train_acc = 100.0 * np.sum(train_preds == train_labels) / len(train_labels)
    test_acc = 100.0 * np.sum(test_preds == test_labels) / len(test_labels)

    # Per-class accuracy
    test_p0 = np.sum(test_labels == 0)
    test_p1 = np.sum(test_labels == 1)
    test_p0_acc = 100.0 * np.sum((test_preds == 0) & (test_labels == 0)) / test_p0
    test_p1_acc = 100.0 * np.sum((test_preds == 1) & (test_labels == 1)) / test_p1

    print(f"\n{'='*60}")
    print(f"RESULTS: HV_size={hv_size}")
    print(f"{'='*60}")
    print(f"  Train Accuracy: {train_acc:.2f}%")
    print(f"  Test Accuracy:  {test_acc:.2f}%")
    print(f"  P0 Accuracy:    {test_p0_acc:.2f}%")
    print(f"  P1 Accuracy:    {test_p1_acc:.2f}%")
    print(f"  Gap:            {abs(test_p0_acc - test_p1_acc):.2f}%")

    return {
        'hv_size': hv_size,
        'hv_bits': hv_bits,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'p0_acc': test_p0_acc,
        'p1_acc': test_p1_acc,
        'gap': abs(test_p0_acc - test_p1_acc)
    }


def main():
    print("="*70)
    print("HVTM EXPERIMENT: HYPERVECTOR SIZE OPTIMIZATION")
    print("="*70)
    print()
    print("Testing 5 hypervector sizes with Reasoning by Elimination (s=1.0)")
    print("Sizes: 128, 256, 512, 1024, 2048")
    print()
    print("Configuration:")
    print("  Clauses: 800")
    print("  T: 5000")
    print("  s: 1.0 (RbE mode)")
    print("  Depth: 3")
    print("  Epochs: 30")
    print("  Data: BALANCED 50/50 P0/P1")
    print()
    print("Expected: Larger HV sizes should achieve similar accuracy")
    print("          with potentially fewer clauses needed.")
    print("="*70)
    input("\nPress Enter to start experiments...")

    # Test configurations
    hv_sizes = [128, 256, 512, 1024, 2048]
    results = []

    for hv_size in hv_sizes:
        result = test_hvtm_configuration(
            hv_size=hv_size,
            hv_bits=8,
            board_size=5,
            stage="0"
        )

        if result:
            results.append(result)

        print(f"\n{'='*70}")
        print(f"Completed {len(results)}/{len(hv_sizes)} experiments")
        print(f"{'='*70}\n")

        if hv_size != hv_sizes[-1]:
            input("Press Enter to continue to next experiment...")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'HV Size':<10} {'Test Acc':<12} {'P0 Acc':<10} {'P1 Acc':<10} {'Gap':<8}")
    print("-" * 60)

    for r in results:
        print(f"{r['hv_size']:<10} {r['test_acc']:>10.2f}%  {r['p0_acc']:>8.2f}%  {r['p1_acc']:>8.2f}%  {r['gap']:>6.2f}%")

    print()

    # Find best
    if results:
        best = max(results, key=lambda x: x['test_acc'])
        print(f"Best configuration: HV_size={best['hv_size']} with {best['test_acc']:.2f}% test accuracy")

        # Compare to baseline
        baseline_acc = 71.40  # From balanced undersampling experiment
        improvement = best['test_acc'] - baseline_acc

        print(f"\nComparison to balanced baseline (71.40%):")
        print(f"  Best HVTM: {best['test_acc']:.2f}%")
        print(f"  Difference: {improvement:+.2f}%")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
