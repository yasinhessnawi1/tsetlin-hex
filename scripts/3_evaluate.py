"""
Script to evaluate trained GTM models on test data.

This script loads trained models and evaluates them comprehensively,
showing accuracy at different game stages.

Usage:
    python scripts/3_evaluate.py --board-size 10
"""

import argparse
import os
import sys
import pickle

# PATCH: Allow unsupported compiler versions (VS2022)
try:
    import pycuda.compiler
    pycuda.compiler.DEFAULT_NVCC_FLAGS.append('-allow-unsupported-compiler')
    pycuda.compiler.DEFAULT_NVCC_FLAGS.append('-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH')
except ImportError:
    pass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import HexGraphTM, Predictor
from src.utils import Config, find_latest_run


def load_gtm_dataset(filepath: str):
    """Load a GTM dataset from pickle file."""
    print(f"Loading dataset from {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data['labels'])} samples")
    return data['graphs'], data['labels']


def evaluate_stage(stage_name: str, config: Config, run_folder: str = None, use_latest: bool = False):
    """
    Evaluate a model for a specific game stage.

    Args:
        stage_name: Name of the stage (e.g., 'end', '-2', '-5')
       config: Configuration object
        run_folder: Specific run folder to evaluate (None = use config.get_model_path)
        use_latest: If True, find and use the latest training run
    """
    print("\n" + "="*60)
    print(f"EVALUATING MODEL FOR STAGE: {stage_name}")
    print("="*60)

    # Determine model path
    if use_latest:
        # Find latest training run
        latest_run = find_latest_run(
            base_dir=f"{config.models_dir}/training_runs",
            stage=stage_name,
            board_size=config.board_size
        )
        if latest_run is None:
            print(f"\nERROR: No training run found for stage '{stage_name}' and board size {config.board_size}")
            return None
        model_path = os.path.join(latest_run, "model.pkl")
        print(f"\nUsing latest training run: {os.path.basename(latest_run)}")
    elif run_folder is not None:
        # Use specified run folder
        model_path = os.path.join(run_folder, "model.pkl")
        print(f"\nUsing specified run folder: {os.path.basename(run_folder)}")
    else:
        # Use old-style model path (backward compatibility)
        model_path = config.get_model_path(stage_name)

    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found at {model_path}")
        print("Please train the model first using 2_train_model.py!")
        return None

    print(f"\nLoading model from {model_path}...")
    model = HexGraphTM()
    model.load(model_path)

    # Load test dataset
    test_path = f"{config.data_dir}/test_gtm_{config.board_size}x{config.board_size}_{stage_name}.pkl"

    if not os.path.exists(test_path):
        print(f"\nERROR: Test dataset not found at {test_path}")
        return None

    test_graphs, test_labels = load_gtm_dataset(test_path)

    # Create predictor
    predictor = Predictor(model)

    # Evaluate
    results = predictor.evaluate_detailed(
        test_graphs,
        test_labels,
        name=f"Test Set (Stage: {stage_name})"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate GTM models')

    parser.add_argument('--board-size', type=int, default=10,
                        help='Size of the Hex board (default: 10)')
    parser.add_argument('--stage', type=str, default='all',
                        help='Stage to evaluate: end, -2, -5, or all (default: all)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing GTM datasets (default: data)')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory containing trained models (default: models)')
    parser.add_argument('--latest', action='store_true',
                        help='Automatically use the most recent training run')
    parser.add_argument('--run-folder', type=str, default=None,
                        help='Specific training run folder to evaluate (absolute path)')

    args = parser.parse_args()

    # Create config
    config = Config()
    config.board_size = args.board_size
    config.data_dir = args.data_dir
    config.models_dir = args.models_dir

    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"Board size: {config.board_size}x{config.board_size}")
    print(f"Data directory: {config.data_dir}")
    print(f"Models directory: {config.models_dir}")

    # Determine which stages to evaluate
    if args.stage == 'all':
        stages = ['end', '-2', '-5']
    else:
        stages = [args.stage]

    print(f"Evaluating stages: {stages}")

    # Evaluate each stage
    results = {}
    for stage in stages:
        result = evaluate_stage(stage, config, 
                               run_folder=args.run_folder, 
                               use_latest=args.latest)
        if result is not None:
            results[stage] = result

    # Comprehensive summary
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*60)

    if len(results) == 0:
        print("\nNo results to summarize!")
        return

    print(f"\nBoard Size: {config.board_size}x{config.board_size}")
    print(f"\n{'Stage':<10} {'Accuracy':<12} {'P0 Acc':<12} {'P1 Acc':<12}")
    print("-" * 50)

    for stage, result in results.items():
        stage_label = {
            'end': 'End',
            '-2': '2 Before',
            '-5': '5 Before'
        }.get(stage, stage)

        print(f"{stage_label:<10} {result['accuracy']:>10.2f}% "
              f"{result['player0_accuracy']:>10.2f}% "
              f"{result['player1_accuracy']:>10.2f}%")

    print("\n" + "="*60)

    # Analysis
    print("\nANALYSIS:")

    if len(results) >= 2:
        # Compare end vs earlier predictions
        if 'end' in results and '-2' in results:
            end_acc = results['end']['accuracy']
            before2_acc = results['-2']['accuracy']
            diff = end_acc - before2_acc
            print(f"  End vs 2-Before: {diff:+.2f}% difference")

        if 'end' in results and '-5' in results:
            end_acc = results['end']['accuracy']
            before5_acc = results['-5']['accuracy']
            diff = end_acc - before5_acc
            print(f"  End vs 5-Before: {diff:+.2f}% difference")

    # Check if we achieved 100% on end-game
    if 'end' in results:
        end_acc = results['end']['accuracy']
        if end_acc >= 100.0:
            print(f"\n  ✓ PERFECT! Achieved 100% accuracy on end-game prediction!")
            print(f"    Ready to scale to larger board sizes.")
        elif end_acc >= 99.0:
            print(f"\n  Near perfect! {end_acc:.2f}% accuracy on end-game.")
            print(f"    Consider more training or hyperparameter tuning.")
        else:
            print(f"\n  Current accuracy: {end_acc:.2f}%")
            print(f"    Suggestions:")
            print(f"      - Increase number of clauses (current: {config.number_of_clauses})")
            print(f"      - Increase message passing depth (current: {config.depth})")
            print(f"      - Generate more training data")
            print(f"      - Adjust T and s parameters")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
