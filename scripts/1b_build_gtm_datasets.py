"""
Build GTM-compatible datasets from C-generated game data.
Converts raw board states into Graph Tsetlin Machine (GTM) graph objects.
"""

import argparse
import os
import sys
import pickle
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation import DatasetBuilder
from src.utils import Config

def main():
    parser = argparse.ArgumentParser(description='Build GTM-compatible datasets from C-generated games')
    parser.add_argument('--board-size', type=int, default=5, help='Board size')
    parser.add_argument('--hypervector-size', type=int, default=128, help='Hypervector size')
    parser.add_argument('--hypervector-bits', type=int, default=4, help='Hypervector bits')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("BUILDING GTM DATASETS FROM C-GENERATED DATA")
    print("="*60)
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Hypervector size: {args.hypervector_size}")
    print(f"Hypervector bits: {args.hypervector_bits}")
    print("="*60)

    # Load C-generated game data
    train_file = f'data/train_games_{args.board_size}x{args.board_size}.npz'
    test_file = f'data/test_games_{args.board_size}x{args.board_size}.npz'

    if not os.path.exists(train_file):
        print(f"\nERROR: Training data not found at {train_file}")
        print("Please run: python scripts/1_generate_games.py first!")
        sys.exit(1)

    if not os.path.exists(test_file):
        print(f"\nERROR: Test data not found at {test_file}")
        print("Please run: python scripts/1_generate_games.py first!")
        sys.exit(1)

    print(f"\nLoading C-generated game data...")
    train_data = np.load(train_file)
    test_data = np.load(test_file)

    train_boards = train_data['states_at_end']
    train_winners = train_data['winners']
    test_boards = test_data['states_at_end']
    test_winners = test_data['winners']

    print(f"Train: {len(train_winners)} games")
    print(f"Test: {len(test_winners)} games")
    
    # Initialize dataset builder
    builder = DatasetBuilder(board_size=args.board_size)

    print("\n" + "="*60)
    print("BUILDING TRAIN AND TEST DATASETS WITH COMPATIBLE ENCODINGS")
    print("="*60)
    print("NOTE: Using fixed random seed to ensure train/test encoding compatibility!")
    
    # Fixed seed for reproducible hypervectors
    HYPERVECTOR_SEED = 42
    
    # Create TRAINING graphs with seed
    print(f"\nBuilding training graphs (seed={HYPERVECTOR_SEED})...")
    np.random.seed(HYPERVECTOR_SEED)
    train_graphs, train_labels = builder.create_graphs_from_game_data(
        train_boards,
        train_winners,
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits,
        verbose=True
    )
    
    # Create TEST graphs with SAME seed (identical encoding!)
    print(f"\nBuilding test graphs (seed={HYPERVECTOR_SEED})...")
    np.random.seed(HYPERVECTOR_SEED)
    test_graphs, test_labels = builder.create_graphs_from_game_data(
        test_boards,
        test_winners,
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits,
        verbose=True
    )
    
    print(f"\n[OK] Train and test graphs created with COMPATIBLE encodings!")

    # Save datasets
    print(f"\n" + "="*60)
    print("SAVING GTM DATASETS")
    print("="*60)
    
    train_output = f'data/train_gtm_{args.board_size}x{args.board_size}_end.pkl'
    test_output = f'data/test_gtm_{args.board_size}x{args.board_size}_end.pkl'
    
    print(f"\nSaving training dataset to {train_output}...")
    with open(train_output, 'wb') as f:
        pickle.dump({'graphs': train_graphs, 'labels': train_labels}, f)
    print(f"  Saved! ({len(train_labels)} samples)")
    
    print(f"\nSaving test dataset to {test_output}...")
    with open(test_output, 'wb') as f:
        pickle.dump({'graphs': test_graphs, 'labels': test_labels}, f)
    print(f"  Saved! ({len(test_labels)} samples)")

    print("\n" + "="*60)
    print("SUCCESS! GTM datasets built and saved.")
    print("="*60)
    print(f"Training data: {train_output}")
    print(f"Test data: {test_output}")
    print("\nYou can now train the model:")
    print(f"  python scripts/2_train_model.py --board-size {args.board_size}")
    print("="*60)


if __name__ == '__main__':
    main()
