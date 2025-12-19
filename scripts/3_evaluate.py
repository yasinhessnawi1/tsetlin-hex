"""
Script to evaluate trained GTM models on test data.

End-to-end behavior:
  - If test datasets are missing, it will generate raw games (via 1_generate_games.py)
    and build GTM datasets (via 1b_build_gtm_datasets.py) automatically.
  - Then loads the requested model(s) and reports accuracy.

Usage:
    python scripts/3_evaluate.py --board-size 10 --stage all
    python scripts/3_evaluate.py --board-size 11 --stage end --num-train 10000 --num-test 3000
"""

import argparse
import os
import sys
import pickle
import subprocess
import shutil
from pathlib import Path
import os
import sys
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import HexGraphTM, Predictor
from src.utils import Config
from src.utils.visualization import plot_confusion_matrix, plot_stage_comparison, plot_training_history

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_STAGES = [0, -2, -5]


def load_gtm_dataset(filepath: str):
    """Load a GTM dataset from pickle file."""
    print(f"Loading dataset from {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data['labels'])} samples")
    return data['graphs'], data['labels']


def parse_stages_arg(stages_str: str):
    """Parse stages argument into a list of stage identifiers (strings)."""
    if stages_str.lower() == "all":
        return [str(s) for s in DEFAULT_STAGES]
    return [s.strip() for s in stages_str.split(",") if s.strip()]


def stage_to_dataset_label(stage: str) -> str:
    """Map human stage names to dataset labels."""
    if stage == "end":
        return "0"
    return stage


def game_npz_paths(board_size: int, data_dir: Path):
    train_npz = data_dir / f"train_games_{board_size}x{board_size}.npz"
    test_npz = data_dir / f"test_games_{board_size}x{board_size}.npz"
    return train_npz, test_npz


def gtm_dataset_paths(board_size: int, stages: list, data_dir: Path):
    return [
        (
            data_dir / f"train_gtm_{board_size}x{board_size}_{stage}.pkl",
            data_dir / f"test_gtm_{board_size}x{board_size}_{stage}.pkl",
        )
        for stage in stages
    ]


def ensure_raw_games(board_size: int, num_train: int, num_test: int, stages_arg: str, data_dir: Path):
    """Ensure raw game npz files exist; generate if missing."""
    train_npz, test_npz = game_npz_paths(board_size, data_dir)
    default_data_dir = REPO_ROOT / "data"
    default_train, default_test = game_npz_paths(board_size, default_data_dir)

    if not (train_npz.exists() and test_npz.exists()):
        if data_dir != default_data_dir and default_train.exists() and default_test.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(default_train, train_npz)
            shutil.copy2(default_test, test_npz)
            print(f"[INFO] Copied existing raw games from {default_data_dir} to {data_dir}")

    if train_npz.exists() and test_npz.exists():
        print(f"[SKIP] Raw games already exist at {train_npz} and {test_npz}")
        return

    gen_script = SCRIPTS_DIR / "1_generate_games.py"
    cmd = [
        sys.executable,
        str(gen_script),
        "--board-size",
        str(board_size),
        "--num-train",
        str(num_train),
        "--num-test",
        str(num_test),
        "--save-states",
        stages_arg,
        "--no-balance",
    ]
    print(f"[RUN] Generating games: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    # Copy back to custom dir if needed
    if data_dir != default_data_dir and default_train.exists() and default_test.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(default_train, train_npz)
        shutil.copy2(default_test, test_npz)

    if not (train_npz.exists() and test_npz.exists()):
        raise FileNotFoundError("Game generation did not produce expected npz files.")
    print(f"[OK] Generated raw games at {train_npz} and {test_npz}")


def ensure_gtm_datasets(board_size: int, stages: list, stages_arg: str, data_dir: Path):
    """Ensure GTM train/test pickles exist; build if missing."""
    expected = gtm_dataset_paths(board_size, stages, data_dir)
    default_data_dir = REPO_ROOT / "data"
    default_expected = gtm_dataset_paths(board_size, stages, default_data_dir)

    # Try to reuse default datasets
    if not all(tr.exists() and te.exists() for tr, te in expected):
        if data_dir != default_data_dir and all(tr.exists() and te.exists() for tr, te in default_expected):
            data_dir.mkdir(parents=True, exist_ok=True)
            for (src_tr, src_te), (dst_tr, dst_te) in zip(default_expected, expected):
                shutil.copy2(src_tr, dst_tr)
                shutil.copy2(src_te, dst_te)
            print(f"[INFO] Copied existing GTM datasets from {default_data_dir} to {data_dir}")

    if all(tr.exists() and te.exists() for tr, te in expected):
        print("[SKIP] GTM datasets already exist for requested stages.")
        return

    build_script = SCRIPTS_DIR / "1b_build_gtm_datasets.py"
    cmd = [
        sys.executable,
        str(build_script),
        "--board-size",
        str(board_size),
        "--stages",
        stages_arg if stages_arg.lower() != "all" else "all",
    ]
    print(f"[RUN] Building GTM datasets: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    # Copy back to custom dir if needed
    if data_dir != default_data_dir:
        data_dir.mkdir(parents=True, exist_ok=True)
        for (src_tr, src_te), (dst_tr, dst_te) in zip(default_expected, expected):
            if src_tr.exists() and not dst_tr.exists():
                shutil.copy2(src_tr, dst_tr)
            if src_te.exists() and not dst_te.exists():
                shutil.copy2(src_te, dst_te)

    expected = gtm_dataset_paths(board_size, stages, data_dir)
    missing = [(tr, te) for tr, te in expected if not (tr.exists() and te.exists())]
    if missing:
        raise FileNotFoundError(f"GTM dataset build missing files: {missing}")
    print("[OK] GTM datasets ready.")


def evaluate_stage(stage_name: str, config: Config, explicit_test_file: str = None):
    """
    Evaluate a model for a specific game stage.

    Args:
        stage_name: Name of the stage (e.g., 'end', '-2', '-5')
        config: Configuration object
    """
    print("\n" + "="*60)
    print(f"EVALUATING MODEL FOR STAGE: {stage_name}")
    print("="*60)

    # Load model
    model_path = config.get_model_path(stage_name)

    if not os.path.exists(model_path):
        history_path = model_path.replace('.pkl', '_history.pkl')
        if os.path.exists(history_path):
            print(f"\n[WARN] Model not found at {model_path}, but history exists: {history_path}")
            print("       Model was likely not saved due to PyCUDA pickling limits.")
            print("       Skipping evaluation for this stage.")
        else:
            print(f"\n[WARN] Model not found at {model_path}. Skipping evaluation for this stage.")
            print("       Train first (scripts/2_train_model.py) to enable evaluation.")
        return None

    print(f"\nLoading model from {model_path}...")
    model = HexGraphTM()
    model.load(model_path)

    # Load test dataset (with optional override and fallback stage label)
    if explicit_test_file:
        test_path = explicit_test_file
        if not os.path.exists(test_path):
            print(f"\nERROR: Test dataset not found at {test_path}")
            return None
    else:
        test_path = f"{config.data_dir}/test_gtm_{config.board_size}x{config.board_size}_{stage_name}.pkl"
        if not os.path.exists(test_path):
            alt_stage = stage_to_dataset_label(stage_name)
            alt_test = f"{config.data_dir}/test_gtm_{config.board_size}x{config.board_size}_{alt_stage}.pkl"
            if os.path.exists(alt_test):
                print(f"[INFO] Using alternate stage label file: {alt_test}")
                test_path = alt_test
            else:
                print(f"\nERROR: Test dataset not found at {test_path}")
                print(f"Also tried: {alt_test}")
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


def generate_visualizations(results: dict, config: Config, viz_dir: str, show_plots: bool):
    """
    Generate visualizations for evaluation results.

    Args:
        results: Dictionary of evaluation results by stage
        config: Configuration object
        viz_dir: Directory to save plots
        show_plots: Whether to display plots interactively
    """
    import os
    import matplotlib.pyplot as plt

    # Create visualization directory
    viz_path = Path(viz_dir)
    viz_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    print(f"Plots will be saved to: {viz_path.absolute()}")

    # Generate confusion matrix for each stage
    for stage, result in results.items():
        stage_label = {
            'end': 'End',
            '-2': '2_Before',
            '-5': '5_Before'
        }.get(stage, stage)

        # Confusion matrix plot
        title = f"Confusion Matrix - Stage: {stage_label} ({config.board_size}x{config.board_size})"
        save_path = viz_path / f"confusion_matrix_{config.board_size}x{config.board_size}_{stage_label.lower()}.png"
        fig, ax = plot_confusion_matrix(result['confusion_matrix'], title=title, save_path=str(save_path))

        if not show_plots:
            plt.close(fig)

    # Generate stage comparison plot if multiple stages
    if len(results) > 1:
        title = f"Accuracy Comparison Across Stages ({config.board_size}x{config.board_size})"
        save_path = viz_path / f"stage_comparison_{config.board_size}x{config.board_size}.png"
        fig, ax = plot_stage_comparison(results, save_path=str(save_path))

        if not show_plots:
            plt.close(fig)

    # Try to generate training history plots if available
    training_histories = load_training_histories(results.keys(), config)
    for stage, history in training_histories.items():
        if history:
            stage_label = {
                'end': 'End',
                '-2': '2_Before',
                '-5': '5_Before'
            }.get(stage, stage)

            title = f"Training History - Stage: {stage_label} ({config.board_size}x{config.board_size})"
            save_path = viz_path / f"training_history_{config.board_size}x{config.board_size}_{stage_label.lower()}.png"
            fig, ax = plot_training_history(history, save_path=str(save_path))

            if not show_plots:
                plt.close(fig)

    if show_plots:
        plt.show()

    print(f"\nVisualization generation complete!")
    print(f"All plots saved to: {viz_path.absolute()}")
    print(f"{'='*60}\n")


def load_training_histories(stages: list, config: Config) -> dict:
    """
    Load training histories for the evaluated stages.

    Args:
        stages: List of stage names
        config: Configuration object

    Returns:
        Dictionary mapping stage names to training histories (or None if not found)
    """
    histories = {}

    for stage in stages:
        history_path = config.get_model_path(stage).replace('.pkl', '_history.pkl')
        if os.path.exists(history_path):
            try:
                with open(history_path, 'rb') as f:
                    history = pickle.load(f)
                histories[stage] = history
                print(f"Loaded training history for stage {stage}: {history_path}")
            except Exception as e:
                print(f"Warning: Could not load training history for stage {stage}: {e}")
                histories[stage] = None
        else:
            histories[stage] = None

    return histories


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
    parser.add_argument('--test-file', type=str, default=None,
                        help='Explicit path to test GTM pickle (overrides stage-based path)')
    parser.add_argument('--num-train', type=int, default=10000,
                        help='Number of training games to generate if missing (default: 10000)')
    parser.add_argument('--num-test', type=int, default=3000,
                        help='Number of test games to generate if missing (default: 3000)')
    parser.add_argument('--gen-stages', type=str, default='all',
                        help='Stages to generate/build datasets for (default: all -> 0,-2,-5)')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualizations for evaluation results (default: True)')
    parser.add_argument('--no-visualize', action='store_true', default=False,
                        help='Disable visualization generation')
    parser.add_argument('--viz-dir', type=str, default='evaluation_plots',
                        help='Directory to save visualization plots (default: evaluation_plots). Visualizations are generated automatically.')
    parser.add_argument('--show-plots', action='store_true', default=False,
                        help='Display plots interactively (blocks execution)')

    args = parser.parse_args()

    # Create config
    config = Config()
    config.board_size = args.board_size
    config.data_dir = args.data_dir
    config.models_dir = args.models_dir
    data_dir = (REPO_ROOT / config.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

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

    # Prepare generation stages (for ensuring data)
    generation_stages = parse_stages_arg(args.gen_stages)
    generation_stage_arg = args.gen_stages if args.gen_stages.lower() != "all" else ",".join(generation_stages)

    # Map evaluation stages to dataset labels to ensure correct files exist
    dataset_stage_labels = list({stage_to_dataset_label(s) for s in stages})

    # Ensure raw games and GTM datasets exist
    ensure_raw_games(
        board_size=args.board_size,
        num_train=args.num_train,
        num_test=args.num_test,
        stages_arg=generation_stage_arg,
        data_dir=data_dir,
    )

    ensure_gtm_datasets(
        board_size=args.board_size,
        stages=dataset_stage_labels,
        stages_arg=generation_stage_arg,
        data_dir=data_dir,
    )

    # Evaluate each stage
    results = {}
    for stage in stages:
        result = evaluate_stage(stage, config, explicit_test_file=args.test_file)
        if result is not None:
            results[stage] = result

    # Generate visualizations by default (unless disabled)
    if (args.visualize and not args.no_visualize) and len(results) > 0:
        generate_visualizations(results, config, args.viz_dir, args.show_plots)

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
