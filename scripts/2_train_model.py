"""
Script to train Graph Tsetlin Machine models for Hex winner prediction.

This script trains separate models for different game stages:
- End of game
- 2 moves before end
- 5 moves before end

Usage:
    python scripts/2_train_model.py --board-size 10 --stage end
    python scripts/2_train_model.py --board-size 10 --stage all
"""

import argparse
import os
import sys
import pickle

# IMPORTANT: Set CUDA device BEFORE any imports that use CUDA
# This ensures PyCUDA uses the correct GPU, especially important for MIG mode
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    # Default to device 0 if not set
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f"INFO: Setting CUDA_VISIBLE_DEVICES=0")
else:
    print(f"INFO: Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import HexGraphTM, Predictor
from src.utils import Config


def load_gtm_dataset(filepath: str):
    """Load a GTM dataset from pickle file."""
    print(f"Loading dataset from {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data['labels'])} samples")
    return data['graphs'], data['labels']


def train_stage(
    stage_name: str,
    config: Config,
    save_model: bool = False  # Disabled due to CUDA pickling issues
):
    """
    Train a model for a specific game stage.

    Args:
        stage_name: Name of the stage (e.g., 'end', '-2', '-5')
        config: Configuration object
        save_model: Whether to save the trained model
    """
    print("\n" + "="*60)
    print(f"TRAINING MODEL FOR STAGE: {stage_name}")
    print("="*60)

    # Load datasets
    train_path = f"{config.data_dir}/train_gtm_{config.board_size}x{config.board_size}_{stage_name}.pkl"
    test_path = f"{config.data_dir}/test_gtm_{config.board_size}x{config.board_size}_{stage_name}.pkl"

    if not os.path.exists(train_path):
        print(f"\nERROR: Training dataset not found at {train_path}")
        print("Please run 1b_build_gtm_datasets.py first!")
        return None

    if not os.path.exists(test_path):
        print(f"\nERROR: Test dataset not found at {test_path}")
        print("Please run 1b_build_gtm_datasets.py first!")
        return None

    train_graphs, train_labels = load_gtm_dataset(train_path)
    test_graphs, test_labels = load_gtm_dataset(test_path)

    # Create model
    print("\nInitializing Graph Tsetlin Machine...")
    model = HexGraphTM(
        number_of_clauses=config.number_of_clauses,
        T=config.T,
        s=config.s,
        depth=config.depth,
        message_size=config.message_size,
        message_bits=config.message_bits,
        max_included_literals=config.max_included_literals,
        grid=config.grid,
        block=config.block
    )

    # Create predictor
    predictor = Predictor(model)

    # Train
    train_acc, test_acc = predictor.train(
        train_graphs=train_graphs,
        train_labels=train_labels,
        test_graphs=test_graphs,
        test_labels=test_labels,
        epochs=config.epochs,
        test_every=config.test_every
    )

    # Detailed evaluation
    train_results = predictor.evaluate_detailed(
        train_graphs,
        train_labels,
        name=f"Training Set (Stage: {stage_name})"
    )

    test_results = predictor.evaluate_detailed(
        test_graphs,
        test_labels,
        name=f"Test Set (Stage: {stage_name})"
    )

    # Save model
    if save_model:
        os.makedirs(config.models_dir, exist_ok=True)
        model_path = config.get_model_path(stage_name)
        model.save(model_path)

        # Save training history
        history_path = model_path.replace('.pkl', '_history.pkl')
        predictor.save_training_history(history_path)

    return {
        'model': model,
        'predictor': predictor,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_results': train_results,
        'test_results': test_results
    }


def main():
    parser = argparse.ArgumentParser(description='Train GTM models for Hex winner prediction')

    parser.add_argument('--board-size', type=int, default=10,
                        help='Size of the Hex board (default: 10)')
    parser.add_argument('--stage', type=str, default='all',
                        help='Stage to train: end, -2, -5, or all (default: all)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing GTM datasets (default: data)')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to save models (default: models)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--clauses', type=int, default=1000,
                        help='Number of clauses (default: 500)')
    parser.add_argument('--depth', type=int, default=3,
                        help='Message passing depth (default: 3)')

    args = parser.parse_args()

    # Create config
    config = Config()
    config.board_size = args.board_size
    config.data_dir = args.data_dir
    config.models_dir = args.models_dir
    config.epochs = args.epochs
    config.number_of_clauses = args.clauses
    config.depth = args.depth

    # Print configuration
    config.print_config()

    # Determine which stages to train
    if args.stage == 'all':
        stages = ['end', '-2', '-5']
    else:
        stages = [args.stage]

    print(f"\nTraining stages: {stages}\n")

    # Train each stage
    results = {}
    for stage in stages:
        result = train_stage(stage, config, save_model=True)
        if result is not None:
            results[stage] = result

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    for stage, result in results.items():
        print(f"\nStage: {stage}")
        print(f"  Training Accuracy: {result['train_acc']:.2f}%")
        print(f"  Test Accuracy: {result['test_acc']:.2f}%")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModels saved to: {config.models_dir}/")
    print(f"You can now evaluate the models:")
    print(f"  python scripts/3_evaluate.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
