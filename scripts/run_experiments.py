"""
GTM Optimization Experiment Runner

This script runs systematic experiments to find optimal configurations
for Hex winner prediction with minimal clause count.

Based on: GTM_Optimization_Guide.md

Usage:
    python scripts/run_experiments.py --phase 1 --experiment 1
    python scripts/run_experiments.py --phase 1 --all
"""

import argparse
import os
import sys
import pickle
import json
import time
from datetime import datetime
from pathlib import Path
from itertools import product

# CUDA setup - MUST be before any CUDA imports
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f"INFO: Setting CUDA_VISIBLE_DEVICES=0")
else:
    print(f"INFO: Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Force CUDA initialization before importing GraphTsetlinMachine
try:
    import pycuda.driver as cuda
    cuda.init()
    device = cuda.Device(0)
    print(f"CUDA Device: {device.name()}")
    print(f"  Memory: {device.total_memory() / (1024**3):.2f} GB")
except Exception as e:
    print(f"WARNING: CUDA initialization issue: {e}")
    print("Will attempt to proceed anyway...")

from src.models import HexGraphTM, Predictor
from src.utils import Config


class ExperimentRunner:
    """Runner for GTM optimization experiments."""

    def __init__(self, output_dir: str = "experiments", board_size: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.board_size = board_size
        self.results = []

        # Create timestamped session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{timestamp}"
        self.session_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"EXPERIMENT SESSION: {timestamp}")
        print(f"Output directory: {self.session_dir}")
        print(f"{'='*60}\n")

    def load_gtm_dataset(self, filepath: str):
        """Load a GTM dataset from pickle file."""
        print(f"Loading dataset: {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"  ✓ Loaded {len(data['labels'])} samples")
        return data['graphs'], data['labels']

    def run_single_experiment(
        self,
        name: str,
        stage: str,
        model_params: dict,
        epochs: int = 100,
        test_every: int = 10,
        data_dir: str = "data"
    ):
        """
        Run a single experiment and record results.

        Args:
            name: Experiment name
            stage: Game stage ('end', '-2', '-5')
            model_params: Model hyperparameters
            epochs: Number of training epochs
            test_every: Evaluate every N epochs
            data_dir: Directory containing datasets

        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {name}")
        print(f"Stage: {stage}")
        print(f"{'='*60}")
        print(f"Parameters:")
        for key, value in model_params.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}")

        # Load data
        train_path = f"{data_dir}/train_gtm_{self.board_size}x{self.board_size}_{stage}.pkl"
        test_path = f"{data_dir}/test_gtm_{self.board_size}x{self.board_size}_{stage}.pkl"

        if not os.path.exists(train_path):
            print(f"ERROR: Training dataset not found: {train_path}")
            return None

        if not os.path.exists(test_path):
            print(f"ERROR: Test dataset not found: {test_path}")
            return None

        train_graphs, train_labels = self.load_gtm_dataset(train_path)
        test_graphs, test_labels = self.load_gtm_dataset(test_path)

        # Create model
        print(f"\nInitializing Graph Tsetlin Machine...")
        model = HexGraphTM(**model_params)
        predictor = Predictor(model)

        # Train with timing
        print(f"\nTraining for {epochs} epochs...")
        start_time = time.time()

        train_acc, test_acc = predictor.train(
            train_graphs=train_graphs,
            train_labels=train_labels,
            test_graphs=test_graphs,
            test_labels=test_labels,
            epochs=epochs,
            test_every=test_every
        )

        train_time = time.time() - start_time

        # Results
        result = {
            'name': name,
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'params': model_params,
            'epochs': epochs,
            'train_time': train_time,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'clauses': model_params.get('number_of_clauses', 'N/A'),
            'efficiency': test_acc / model_params.get('number_of_clauses', 1)
        }

        self.results.append(result)

        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"  Train Accuracy: {train_acc:.2f}%")
        print(f"  Test Accuracy:  {test_acc:.2f}%")
        print(f"  Training Time:  {train_time:.2f}s")
        print(f"  Clauses:        {result['clauses']}")
        print(f"  Efficiency:     {result['efficiency']:.4f} acc%/clause")
        print(f"{'='*60}\n")

        return result

    def run_phase_1_experiment_1(self, stage: str = 'end'):
        """
        Phase 1, Experiment 1: Find minimum clauses baseline
        Vary: clauses = [100, 200, 300, 400, 500]
        Fixed: s=10.0, T=15000, depth=3
        """
        print(f"\n{'#'*60}")
        print(f"PHASE 1 - EXPERIMENT 1: Minimum Clauses Baseline")
        print(f"{'#'*60}\n")

        clause_counts = [100, 200, 300, 400, 500]
        base_params = {
            's': 10.0,
            'T': 15000,
            'depth': 3,
            'message_size': 256,
            'message_bits': 2,
            'max_included_literals': 255
        }

        for clauses in clause_counts:
            params = base_params.copy()
            params['number_of_clauses'] = clauses

            self.run_single_experiment(
                name=f"P1E1_clauses_{clauses}",
                stage=stage,
                model_params=params,
                epochs=100,
                test_every=10
            )

    def run_phase_1_experiment_2(self, stage: str = 'end', optimal_clauses: int = 200):
        """
        Phase 1, Experiment 2: Optimize specificity
        Vary: s = [5.0, 10.0, 15.0, 20.0, 25.0]
        Fixed: clauses=optimal_clauses (from Exp 1), T=15000, depth=3
        """
        print(f"\n{'#'*60}")
        print(f"PHASE 1 - EXPERIMENT 2: Optimize Specificity (s)")
        print(f"Using clauses={optimal_clauses} from Experiment 1")
        print(f"{'#'*60}\n")

        s_values = [5.0, 10.0, 15.0, 20.0, 25.0]
        base_params = {
            'number_of_clauses': optimal_clauses,
            'T': 15000,
            'depth': 3,
            'message_size': 256,
            'message_bits': 2,
            'max_included_literals': 255
        }

        for s in s_values:
            params = base_params.copy()
            params['s'] = s

            self.run_single_experiment(
                name=f"P1E2_s_{s}",
                stage=stage,
                model_params=params,
                epochs=100,
                test_every=10
            )

    def run_phase_1_experiment_3(self, stage: str = 'end', optimal_clauses: int = 200, optimal_s: float = 10.0):
        """
        Phase 1, Experiment 3: Optimize threshold
        Vary: T = [5000, 10000, 15000, 20000]
        Fixed: clauses, s (from previous experiments), depth=3
        """
        print(f"\n{'#'*60}")
        print(f"PHASE 1 - EXPERIMENT 3: Optimize Threshold (T)")
        print(f"Using clauses={optimal_clauses}, s={optimal_s}")
        print(f"{'#'*60}\n")

        T_values = [5000, 10000, 15000, 20000]
        base_params = {
            'number_of_clauses': optimal_clauses,
            's': optimal_s,
            'depth': 3,
            'message_size': 256,
            'message_bits': 2,
            'max_included_literals': 255
        }

        for T in T_values:
            params = base_params.copy()
            params['T'] = T

            self.run_single_experiment(
                name=f"P1E3_T_{T}",
                stage=stage,
                model_params=params,
                epochs=100,
                test_every=10
            )

    def run_phase_1_experiment_4(self, stage: str = 'end', optimal_clauses: int = 200,
                                   optimal_s: float = 10.0, optimal_T: int = 15000):
        """
        Phase 1, Experiment 4: Optimize depth
        Vary: depth = [2, 3, 4, 5, 6]
        Fixed: clauses, s, T (from previous experiments)
        """
        print(f"\n{'#'*60}")
        print(f"PHASE 1 - EXPERIMENT 4: Optimize Depth")
        print(f"Using clauses={optimal_clauses}, s={optimal_s}, T={optimal_T}")
        print(f"{'#'*60}\n")

        depth_values = [2, 3, 4, 5, 6]
        base_params = {
            'number_of_clauses': optimal_clauses,
            's': optimal_s,
            'T': optimal_T,
            'message_size': 256,
            'message_bits': 2,
            'max_included_literals': 255
        }

        for depth in depth_values:
            params = base_params.copy()
            params['depth'] = depth

            self.run_single_experiment(
                name=f"P1E4_depth_{depth}",
                stage=stage,
                model_params=params,
                epochs=100,
                test_every=10
            )

    def save_results(self, filename: str = "results.json"):
        """Save all results to JSON file."""
        filepath = self.session_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to: {filepath}")
        return filepath

    def print_summary(self):
        """Print summary of all experiments."""
        if not self.results:
            print("\nNo results to summarize.")
            return

        print(f"\n{'='*60}")
        print(f"EXPERIMENT SUMMARY ({len(self.results)} experiments)")
        print(f"{'='*60}")

        # Sort by test accuracy
        sorted_results = sorted(self.results, key=lambda x: x['test_accuracy'], reverse=True)

        print(f"\n{'Name':<25} {'Clauses':<10} {'Train%':<10} {'Test%':<10} {'Eff':<10}")
        print('-' * 65)

        for r in sorted_results:
            print(f"{r['name']:<25} {r['clauses']:<10} "
                  f"{r['train_accuracy']:<10.2f} {r['test_accuracy']:<10.2f} "
                  f"{r['efficiency']:<10.4f}")

        # Best results
        best_acc = sorted_results[0]
        print(f"\n{'='*60}")
        print(f"BEST ACCURACY: {best_acc['test_accuracy']:.2f}%")
        print(f"  Config: {best_acc['name']}")
        print(f"  Clauses: {best_acc['clauses']}")
        print(f"  Parameters: {best_acc['params']}")

        # Most efficient
        sorted_by_eff = sorted(self.results, key=lambda x: x['efficiency'], reverse=True)
        best_eff = sorted_by_eff[0]
        print(f"\nMOST EFFICIENT: {best_eff['efficiency']:.4f} acc%/clause")
        print(f"  Config: {best_eff['name']}")
        print(f"  Clauses: {best_eff['clauses']}, Accuracy: {best_eff['test_accuracy']:.2f}%")

        # 100% accuracy configs
        perfect = [r for r in self.results if r['test_accuracy'] >= 99.5]
        if perfect:
            min_clauses = min(perfect, key=lambda x: x['clauses'])
            print(f"\nMINIMUM CLAUSES FOR ≥99.5% ACCURACY:")
            print(f"  Config: {min_clauses['name']}")
            print(f"  Clauses: {min_clauses['clauses']}")
            print(f"  Accuracy: {min_clauses['test_accuracy']:.2f}%")

        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run GTM optimization experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Phase 1, Experiment 1 (find minimum clauses)
  python scripts/run_experiments.py --phase 1 --experiment 1

  # Run all Phase 1 experiments
  python scripts/run_experiments.py --phase 1 --all

  # Run specific experiment with custom parameters
  python scripts/run_experiments.py --phase 1 --experiment 2 --optimal-clauses 300
        """
    )

    parser.add_argument('--phase', type=int, required=True,
                        help='Experiment phase (1, 2, 3, etc.)')
    parser.add_argument('--experiment', type=int,
                        help='Experiment number within phase')
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments in the phase')
    parser.add_argument('--board-size', type=int, default=10,
                        help='Board size (default: 10)')
    parser.add_argument('--stage', type=str, default='end',
                        help='Game stage: end, -2, -5 (default: end)')
    parser.add_argument('--output-dir', type=str, default='experiments',
                        help='Output directory (default: experiments)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory (default: data)')

    # For sequential experiments
    parser.add_argument('--optimal-clauses', type=int, default=200,
                        help='Optimal clauses from Experiment 1 (default: 200)')
    parser.add_argument('--optimal-s', type=float, default=10.0,
                        help='Optimal s from Experiment 2 (default: 10.0)')
    parser.add_argument('--optimal-T', type=int, default=15000,
                        help='Optimal T from Experiment 3 (default: 15000)')

    args = parser.parse_args()

    # Create runner
    runner = ExperimentRunner(output_dir=args.output_dir, board_size=args.board_size)

    # Run experiments
    if args.phase == 1:
        if args.all or args.experiment == 1:
            runner.run_phase_1_experiment_1(stage=args.stage)

        if args.all or args.experiment == 2:
            runner.run_phase_1_experiment_2(
                stage=args.stage,
                optimal_clauses=args.optimal_clauses
            )

        if args.all or args.experiment == 3:
            runner.run_phase_1_experiment_3(
                stage=args.stage,
                optimal_clauses=args.optimal_clauses,
                optimal_s=args.optimal_s
            )

        if args.all or args.experiment == 4:
            runner.run_phase_1_experiment_4(
                stage=args.stage,
                optimal_clauses=args.optimal_clauses,
                optimal_s=args.optimal_s,
                optimal_T=args.optimal_T
            )

    else:
        print(f"Phase {args.phase} experiments not yet implemented.")
        print("Available: Phase 1 (baseline hyperparameter optimization)")
        return

    # Summary and save
    runner.print_summary()
    results_file = runner.save_results()

    print(f"\n{'='*60}")
    print(f"SESSION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {results_file}")
    print(f"Session directory: {runner.session_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
