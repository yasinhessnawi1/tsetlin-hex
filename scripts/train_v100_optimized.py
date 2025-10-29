"""
V100-optimized training script.
Uses larger grid sizes and more clauses to fully utilize 32GB V100.
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


def load_gtm_dataset(filepath: str):
    """Load a GTM dataset from pickle file."""
    print(f"Loading dataset from {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data['labels'])} samples")
    return data['graphs'], data['labels']


def main():
    parser = argparse.ArgumentParser(description='V100-Optimized GTM Training')
    
    parser.add_argument('--board-size', type=int, default=5)
    parser.add_argument('--stage', type=str, default='end')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--clauses', type=int, default=500)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--grid-x', type=int, default=1024,
                        help='CUDA grid X dimension (default: 1024 for V100)')
    parser.add_argument('--message-size', type=int, default=512,
                        help='Message size (default: 512)')
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    config.board_size = args.board_size
    config.epochs = args.epochs
    config.number_of_clauses = args.clauses
    config.depth = args.depth
    config.message_size = args.message_size
    
    # V100-optimized CUDA settings
    # V100 has 80 SMs, can run many blocks concurrently
    config.grid = (args.grid_x, 1, 1)
    config.block = (128, 1, 1)
    
    print("\n" + "="*60)
    print("V100-OPTIMIZED CONFIGURATION")
    print("="*60)
    print(f"Clauses: {config.number_of_clauses}")
    print(f"Depth: {config.depth}")
    print(f"T: {config.T}, s: {config.s}")
    print(f"Message size: {config.message_size}")
    print(f"CUDA Grid: {config.grid}")
    print(f"CUDA Block: {config.block}")
    print(f"Total threads: {config.grid[0] * config.block[0]:,}")
    print("="*60 + "\n")
    
    # Load datasets
    train_path = f"{config.data_dir}/train_gtm_{config.board_size}x{config.board_size}_{args.stage}.pkl"
    test_path = f"{config.data_dir}/test_gtm_{config.board_size}x{config.board_size}_{args.stage}.pkl"
    
    if not os.path.exists(train_path):
        print(f"\nERROR: Training dataset not found at {train_path}")
        return 1
    
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
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    train_acc, test_acc = predictor.train(
        train_graphs=train_graphs,
        train_labels=train_labels,
        test_graphs=test_graphs,
        test_labels=test_labels,
        epochs=config.epochs,
        test_every=10
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Training Accuracy: {train_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

