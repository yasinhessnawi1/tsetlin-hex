"""
Script to train Graph Tsetlin Machine models for Hex winner prediction.

FIXED VERSION - Uses correct GTM parameters and binary classification.

This script trains separate models for different game stages:
- End of game
- 2 moves before end
- 5 moves before end

Usage:
    python scripts/2_train_model.py --board-size 10 --stage end
    python scripts/2_train_model.py --board-size 10 --stage all
    
CRITICAL FIXES:
    - T values now 15-50 (was 5000-15000)
    - s values now 3.0-5.0 or tuples (was 0.01-10.0)
    - Clauses increased to 1000-4000 (was 100-500)
    - max_included_literals now None (was 255)
    - Using binary GraphTsetlinMachine (not MultiClass)
"""

import argparse
import os
import sys
import pickle

# IMPORTANT: Set CUDA device BEFORE any imports that use CUDA
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f"INFO: Setting CUDA_VISIBLE_DEVICES=0")
else:
    print(f"INFO: Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import HexGraphTM, Predictor
from src.utils import Config


def get_auto_params(board_size: int, depth: int):
    """Get recommended parameters based on board size and depth."""
    # Clause recommendations
    clause_map = {
        5: 1000,
        6: 1500,
        7: 2000,
        8: 2500,
        9: 3000,
        10: 4000,
        11: 5000,
    }
    clauses = clause_map.get(board_size, 2000 + (board_size - 7) * 500)
    if depth >= 3:
        clauses = int(clauses * 1.5)
    
    # T recommendations
    T_map = {
        5: 15,
        6: 20,
        7: 25,
        8: 30,
        9: 35,
        10: 40,
        11: 50,
    }
    T = T_map.get(board_size, 25)
    
    # s recommendations (tuple for depth > 1)
    if depth == 1:
        s = 3.0
    elif depth == 2:
        s = (4.0, 2.5)
    elif depth == 3:
        s = (5.0, 3.0, 2.0)
    else:
        s = tuple(5.0 - i * 0.7 for i in range(depth))
    
    return clauses, T, s


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
    save_model: bool = False
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

    # Load datasets with fallback for "end" -> "0" naming
    train_path = f"{config.data_dir}/train_gtm_{config.board_size}x{config.board_size}_{stage_name}.pkl"
    test_path = f"{config.data_dir}/test_gtm_{config.board_size}x{config.board_size}_{stage_name}.pkl"

    # Fallback: if stage is "end" and file not found, try "0"
    if not os.path.exists(train_path) and stage_name == "end":
        alt_train_path = f"{config.data_dir}/train_gtm_{config.board_size}x{config.board_size}_0.pkl"
        if os.path.exists(alt_train_path):
            print(f"[INFO] Using alternative naming: '0' instead of 'end'")
            train_path = alt_train_path
            test_path = f"{config.data_dir}/test_gtm_{config.board_size}x{config.board_size}_0.pkl"

    if not os.path.exists(train_path):
        print(f"\nERROR: Training dataset not found at {train_path}")
        if stage_name == "end":
            print(f"Also tried: {config.data_dir}/train_gtm_{config.board_size}x{config.board_size}_0.pkl")
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
        board_size=config.board_size,  # Added board_size parameter
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

    # Train with validation
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

    # Save model if requested
    if save_model:
        print("\n[INFO] Saving model...")
        os.makedirs(config.models_dir, exist_ok=True)
        model_path = config.get_model_path(stage_name)
        try:
            model.save(model_path)
            history_path = model_path.replace('.pkl', '_history.pkl')
            predictor.save_training_history(history_path)
            print(f"[OK] Model saved to {model_path}")
        except Exception as e:
            print(f"[WARNING] Could not save model: {e}")
            print("[INFO] This is okay - only accuracy numbers are needed for evaluation")

    return {
        'model': model,
        'predictor': predictor,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_results': train_results,
        'test_results': test_results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train GTM models for Hex winner prediction (FIXED VERSION)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT PARAMETER CHANGES:
  The default parameters have been FIXED to use correct values:
  - T: Now 15-50 (was 5000-15000) 
  - s: Now 3.0-5.0 or tuples (was 0.01-10.0)
  - Clauses: Now auto-set 1000-4000 based on board size
  - max_included_literals: Now None (was 255)
  
Examples:
  # Auto-configure for 7x7 board (RECOMMENDED)
  python scripts/2_train_model.py --board-size 7 --stage end --auto-params
  
  # Manual configuration for experiments
  python scripts/2_train_model.py --board-size 7 --stage end --clauses 2000 --T 25 --s 4.0 --depth 2
        """
    )

    parser.add_argument('--board-size', type=int, default=10,
                        help='Size of the Hex board (default: 10)')
    parser.add_argument('--stage', type=str, default='all',
                        help='Stage to train: end, -2, -5, or all (default: all)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing GTM datasets (default: data)')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to save models (default: models)')

    # Auto-configuration
    parser.add_argument('--auto-params', action='store_true',
                        help='Use automatic parameter configuration based on board size (RECOMMENDED)')

    # Training parameters with FIXED defaults
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--clauses', type=int, default=None,
                        help='Number of clauses (default: auto-set based on board size)')
    parser.add_argument('--depth', type=int, default=2,
                        help='Message passing depth (default: 2, was 6)')
    parser.add_argument('--T', type=int, default=None,
                        help='Threshold T (default: auto-set 15-50, was 5000)')
    parser.add_argument('--s', type=float, default=None,
                        help='Specificity s - single value (default: auto-set based on depth)')
    
    # Advanced s configuration
    parser.add_argument('--s-tuple', type=str, default=None,
                        help='Specificity as tuple for multi-depth, e.g., "4.0,2.5" for depth=2')

    # Advanced parameters
    parser.add_argument('--message-size', type=int, default=256,
                        help='Message size (default: 256)')
    parser.add_argument('--message-bits', type=int, default=2,
                        help='Message bits (default: 2)')
    parser.add_argument('--max-included-literals', type=int, default=None,
                        help='Max included literals (default: None = no limit, was 255)')
    parser.add_argument('--test-every', type=int, default=5,
                        help='Test every N epochs (default: 5)')

    args = parser.parse_args()

    # Create config
    config = Config()
    config.board_size = args.board_size
    config.data_dir = args.data_dir
    config.models_dir = args.models_dir
    config.epochs = args.epochs
    config.test_every = args.test_every
    config.message_size = args.message_size
    config.message_bits = args.message_bits

    # Determine depth
    depth = args.depth

    # Auto-params or manual configuration
    if args.auto_params or (args.clauses is None and args.T is None and args.s is None):
        print("\n[AUTO-CONFIGURATION] Using recommended parameters for board size", args.board_size)
        auto_clauses, auto_T, auto_s = get_auto_params(args.board_size, depth)
        
        config.number_of_clauses = auto_clauses
        config.T = auto_T
        config.s = auto_s
        config.depth = depth
        config.max_included_literals = None
        
        print(f"  Auto-configured:")
        print(f"    Clauses: {auto_clauses}")
        print(f"    T: {auto_T}")
        print(f"    s: {auto_s}")
        print(f"    depth: {depth}")
        print(f"    max_included_literals: None")
    else:
        # Manual configuration with validation
        config.number_of_clauses = args.clauses if args.clauses is not None else get_auto_params(args.board_size, depth)[0]
        config.T = args.T if args.T is not None else get_auto_params(args.board_size, depth)[1]
        config.depth = depth
        config.max_included_literals = args.max_included_literals
        
        # Handle s parameter (can be single float or tuple)
        if args.s_tuple:
            # Parse tuple from string "4.0,2.5"
            config.s = tuple(float(x.strip()) for x in args.s_tuple.split(','))
            print(f"[INFO] Using s as tuple: {config.s}")
        elif args.s is not None:
            config.s = args.s
        else:
            config.s = get_auto_params(args.board_size, depth)[2]
        
        # Validate parameters
        if config.T > 100:
            print(f"\n[WARNING] T={config.T} is very high! Recommended: 15-50")
            print(f"          This may prevent the model from learning properly.")
            print(f"          Consider using --auto-params or setting T=15-50")
        
        if isinstance(config.s, float) and config.s > 20:
            print(f"\n[WARNING] s={config.s} is very high! Recommended: 3.0-5.0")
            print(f"          This may cause unstable learning.")
        
        if config.number_of_clauses < 500:
            print(f"\n[WARNING] {config.number_of_clauses} clauses may be too few!")
            print(f"          Recommended: 1000-4000 for board size {args.board_size}")

    # Print configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    config.print_config()
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
    print(f"\nYou can now evaluate the models:")
    print(f"  python scripts/3_evaluate.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()