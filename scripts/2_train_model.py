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

# PATCH: Allow unsupported compiler versions (VS2022)
try:
    import pycuda.compiler
    pycuda.compiler.DEFAULT_NVCC_FLAGS.append('-allow-unsupported-compiler')
    pycuda.compiler.DEFAULT_NVCC_FLAGS.append('-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH')
    print("INFO: Applied PyCUDA compiler flags for VS2022 compatibility")
except ImportError:
    print("WARNING: Could not import pycuda.compiler. Make sure pycuda is installed.")

# Add src topath
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import HexGraphTM, Predictor
from src.utils import Config, TrainingLogger
from src.data_generation import DatasetBuilder


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
    save_model: bool = True  # Now enabled with proper get_state/set_state serialization
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

    # Create training logger for structured logging
    logger = TrainingLogger(
        base_dir=f"{config.models_dir}/training_runs",
        stage=stage_name,
        board_size=config.board_size
    )
    print(f"Training run folder: {logger.run_folder}")

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

    # Create predictor with logger
    predictor = Predictor(model, logger=logger)

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

    # Save model using new structured logging
    if save_model:
        print("\n" + "="*60)
        print("SAVING MODEL AND TRAINING ARTIFACTS")
        print("="*60)
        
        # Save model to logger's run folder
        model_path = logger.get_model_path()
        model.save(model_path)
        
        # Save configuration
        logger.set_config(
            model_hyperparameters={
                'number_of_clauses': config.number_of_clauses,
                'T': config.T,
                's': config.s,
                'depth': config.depth,
                'message_size': config.message_size,
                'message_bits': config.message_bits,
                'max_included_literals': config.max_included_literals
            },
            training_config={
                'epochs': config.epochs,
                'board_size': config.board_size,
                'stage': stage_name,
                'test_every': config.test_every
            },
            cuda_config={
                'grid': list(config.grid),
                'block': list(config.block)
            },
            dataset_info={
                'train_path': train_path,
                'test_path': test_path,
                'train_samples': len(train_labels),
                'test_samples': len(test_labels)
            }
        )
        logger.save_config()
        
        # Save training history
        logger.save_training_history()
        
        # Save summary
        logger.save_summary(train_acc, test_acc)
        
        # Extract and save human-readable rules
        print("\nExtracting human-readable rules...")
        from src.utils import RuleExtractor
        from src.data_generation import DatasetBuilder
        
        # Get symbol names from the dataset builder
        builder = DatasetBuilder(board_size=config.board_size)
        symbol_names = builder.symbols
        
        # Create rule extractor and save rules
        rule_extractor = RuleExtractor(model, symbol_names)
        rules_path = os.path.join(logger.run_path, "rules.txt")
        rule_extractor.save_rules(rules_path, max_rules=200)  # Save top 200 rules
        
        # Extract and save message passing information
        print("\nExtracting message passing patterns...")
        messages_path = os.path.join(logger.run_path, "messages.txt")
        rule_extractor.save_messages(messages_path, num_edge_types=1)  # 1 edge type for Hex
        
        print(f"\nAll artifacts saved to: {logger.run_path}")
        print("="*60)

    return {
        'model': model,
        'predictor': predictor,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_results': train_results,
        'test_results': test_results,
        'logger': logger
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

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--clauses', type=int, default=10000,
                        help='Number of clauses (default: 1000)')
    parser.add_argument('--depth', type=int, default=6,
                        help='Message passing depth (default: 3)')
    parser.add_argument('--T', type=int, default=15000,
                        help='Threshold T (default: 5000)')
    parser.add_argument('--s', type=float, default=0.01,
                        help='Specificity s (default: 0.01)')

    # Advanced parameters
    parser.add_argument('--message-size', type=int, default=256,
                        help='Message size (default: 256)')
    parser.add_argument('--message-bits', type=int, default=2,
                        help='Message bits (default: 2)')
    parser.add_argument('--max-included-literals', type=int, default=255,
                        help='Max included literals (default: 255)')
    parser.add_argument('--test-every', type=int, default=5,
                        help='Test every N epochs (default: 5)')

    # Informational only (hypervectors set at dataset build time)
    parser.add_argument('--hypervector-size', type=int, default=None,
                        help='INFO ONLY: Hypervector size used in dataset (set at build time)')
    parser.add_argument('--hypervector-bits', type=int, default=None,
                        help='INFO ONLY: Hypervector bits used in dataset (set at build time)')

    args = parser.parse_args()

    # Create config
    config = Config()
    config.board_size = args.board_size
    config.data_dir = args.data_dir
    config.models_dir = args.models_dir

    # Training parameters
    config.epochs = args.epochs
    config.number_of_clauses = args.clauses
    config.depth = args.depth
    config.T = args.T
    config.s = args.s
    config.test_every = args.test_every

    # Advanced parameters
    config.message_size = args.message_size
    config.message_bits = args.message_bits
    config.max_included_literals = args.max_included_literals

    # Print configuration
    config.print_config()

    # Print hypervector info if provided (informational only)
    if args.hypervector_size or args.hypervector_bits:
        print("\n" + "="*60)
        print("HYPERVECTOR SETTINGS (from dataset build)")
        print("="*60)
        if args.hypervector_size:
            print(f"Hypervector size: {args.hypervector_size}")
        if args.hypervector_bits:
            print(f"Hypervector bits: {args.hypervector_bits}")
        print("Note: These were set when building the dataset (.pkl files)")
        print("="*60)

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
