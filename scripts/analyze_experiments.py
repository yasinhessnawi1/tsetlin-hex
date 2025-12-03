"""
Analyze GTM Optimization Experiment Results

This script parses the log files from run_gtm_experiments.bat and
generates a summary report with visualizations.

Usage:
    python scripts/analyze_experiments.py --results-dir experiments/phase1_20241203_123456
"""

import argparse
import re
import json
from pathlib import Path
from datetime import datetime


class ExperimentAnalyzer:
    """Analyze experiment results from log files."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results = []

    def parse_log_file(self, log_file: Path):
        """Parse a single log file and extract results."""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract experiment parameters from filename
            filename = log_file.stem
            parts = filename.split('_')

            result = {
                'file': log_file.name,
                'experiment': '_'.join(parts[:2]),  # e.g., exp1_1
                'param_name': parts[2] if len(parts) > 2 else 'unknown',
                'param_value': parts[3] if len(parts) > 3 else 'unknown'
            }

            # Extract configuration from log
            config_match = re.search(r'Number of clauses:\s*(\d+)', content)
            if config_match:
                result['clauses'] = int(config_match.group(1))

            depth_match = re.search(r'Depth:\s*(\d+)', content)
            if depth_match:
                result['depth'] = int(depth_match.group(1))

            s_match = re.search(r'Specificity \(s\):\s*([\d.]+)', content)
            if s_match:
                result['s'] = float(s_match.group(1))

            T_match = re.search(r'Threshold \(T\):\s*(\d+)', content)
            if T_match:
                result['T'] = int(T_match.group(1))

            # Extract final accuracy
            # Look for "Training Accuracy: XX.XX%" and "Test Accuracy: XX.XX%"
            train_acc_match = re.search(r'Training Accuracy:\s*([\d.]+)%', content)
            if train_acc_match:
                result['train_accuracy'] = float(train_acc_match.group(1))

            test_acc_match = re.search(r'Test Accuracy:\s*([\d.]+)%', content)
            if test_acc_match:
                result['test_accuracy'] = float(test_acc_match.group(1))

            # Extract training time if available
            time_match = re.search(r'Total training time:\s*([\d.]+)\s*seconds', content)
            if time_match:
                result['training_time'] = float(time_match.group(1))

            # Calculate efficiency
            if 'test_accuracy' in result and 'clauses' in result:
                result['efficiency'] = result['test_accuracy'] / result['clauses']

            self.results.append(result)
            return result

        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
            return None

    def parse_all_logs(self):
        """Parse all log files in the results directory."""
        log_files = list(self.results_dir.glob('*.log'))
        print(f"\nFound {len(log_files)} log files in {self.results_dir}")
        print("\nParsing logs...")

        for log_file in sorted(log_files):
            print(f"  {log_file.name}...", end=' ')
            result = self.parse_log_file(log_file)
            if result and 'test_accuracy' in result:
                print(f"✓ (Test Acc: {result['test_accuracy']:.2f}%)")
            else:
                print("✗ (no results found)")

    def generate_report(self):
        """Generate a comprehensive analysis report."""
        if not self.results:
            print("\nNo results to analyze!")
            return

        print("\n" + "="*70)
        print("GTM OPTIMIZATION EXPERIMENT ANALYSIS")
        print("="*70)

        # Group by experiment
        experiments = {}
        for r in self.results:
            exp_name = r['experiment']
            if exp_name not in experiments:
                experiments[exp_name] = []
            experiments[exp_name].append(r)

        # Analyze each experiment
        for exp_name in sorted(experiments.keys()):
            exp_results = experiments[exp_name]
            print(f"\n{'='*70}")
            print(f"EXPERIMENT {exp_name.upper()}")
            print(f"{'='*70}")

            # Sort by test accuracy
            sorted_results = sorted(
                exp_results,
                key=lambda x: x.get('test_accuracy', 0),
                reverse=True
            )

            # Print table
            print(f"\n{'Config':<20} {'Train%':<10} {'Test%':<10} {'Clauses':<10} {'Efficiency':<12}")
            print("-" * 70)

            for r in sorted_results:
                config = f"{r.get('param_name', 'N/A')}={r.get('param_value', 'N/A')}"
                train_acc = r.get('train_accuracy', 0)
                test_acc = r.get('test_accuracy', 0)
                clauses = r.get('clauses', 'N/A')
                eff = r.get('efficiency', 0)

                print(f"{config:<20} {train_acc:<10.2f} {test_acc:<10.2f} "
                      f"{clauses if isinstance(clauses, int) else clauses:<10} "
                      f"{eff:<12.4f}")

            # Best result
            best = sorted_results[0]
            print(f"\n✓ BEST: {best.get('param_name', 'N/A')}={best.get('param_value', 'N/A')}")
            print(f"  Test Accuracy: {best.get('test_accuracy', 0):.2f}%")
            print(f"  Train Accuracy: {best.get('train_accuracy', 0):.2f}%")
            print(f"  Clauses: {best.get('clauses', 'N/A')}")

        # Overall summary
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY")
        print(f"{'='*70}")

        # Best overall accuracy
        best_acc = max(self.results, key=lambda x: x.get('test_accuracy', 0))
        print(f"\nBEST ACCURACY: {best_acc.get('test_accuracy', 0):.2f}%")
        print(f"  Experiment: {best_acc['experiment']}")
        print(f"  Configuration:")
        print(f"    Clauses: {best_acc.get('clauses', 'N/A')}")
        print(f"    Depth: {best_acc.get('depth', 'N/A')}")
        print(f"    s: {best_acc.get('s', 'N/A')}")
        print(f"    T: {best_acc.get('T', 'N/A')}")

        # Most efficient
        best_eff = max(
            self.results,
            key=lambda x: x.get('efficiency', 0)
        )
        print(f"\nMOST EFFICIENT: {best_eff.get('efficiency', 0):.4f} acc%/clause")
        print(f"  Experiment: {best_eff['experiment']}")
        print(f"  Test Accuracy: {best_eff.get('test_accuracy', 0):.2f}%")
        print(f"  Clauses: {best_eff.get('clauses', 'N/A')}")

        # Perfect or near-perfect results
        threshold = 99.0
        perfect = [r for r in self.results if r.get('test_accuracy', 0) >= threshold]
        if perfect:
            min_clauses = min(perfect, key=lambda x: x.get('clauses', float('inf')))
            print(f"\nMINIMUM CLAUSES FOR ≥{threshold}% ACCURACY:")
            print(f"  Clauses: {min_clauses.get('clauses', 'N/A')}")
            print(f"  Test Accuracy: {min_clauses.get('test_accuracy', 0):.2f}%")
            print(f"  Configuration: depth={min_clauses.get('depth', 'N/A')}, "
                  f"s={min_clauses.get('s', 'N/A')}, T={min_clauses.get('T', 'N/A')}")

        # Recommendations
        print(f"\n{'='*70}")
        print("RECOMMENDATIONS")
        print(f"{'='*70}")

        # Find optimal from each experiment
        print("\nOptimal parameters from each experiment:")
        for exp_name in sorted(experiments.keys()):
            exp_results = experiments[exp_name]
            best = max(exp_results, key=lambda x: x.get('test_accuracy', 0))
            param = f"{best.get('param_name', 'N/A')}={best.get('param_value', 'N/A')}"
            print(f"  {exp_name}: {param} ({best.get('test_accuracy', 0):.2f}%)")

        print("\nNext Steps:")
        print("  1. Use the optimal configuration for Phase 2 experiments")
        print("  2. Consider TM Composites (ensemble of specialists)")
        print("  3. Explore CoTM (Coalesced TM with clause sharing)")
        print("  4. Test weighted clauses if supported by library")

        print(f"\n{'='*70}\n")

    def save_json_summary(self, output_file: str = "summary.json"):
        """Save results to JSON for further analysis."""
        output_path = self.results_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ JSON summary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze GTM optimization experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/analyze_experiments.py --results-dir experiments/phase1_20241203_123456

This will parse all .log files in the directory and generate:
  1. Console report with experiment summaries
  2. JSON file with all parsed results
        """
    )

    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing experiment log files')
    parser.add_argument('--output', type=str, default='summary.json',
                        help='Output JSON filename (default: summary.json)')

    args = parser.parse_args()

    # Check directory exists
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Directory not found: {results_dir}")
        return

    # Create analyzer
    analyzer = ExperimentAnalyzer(results_dir)

    # Parse logs
    analyzer.parse_all_logs()

    # Generate report
    analyzer.generate_report()

    # Save JSON
    analyzer.save_json_summary(args.output)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
