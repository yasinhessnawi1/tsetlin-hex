"""
Build GTM-compatible datasets from C-generated game data with multi-stage support.
Converts raw board states into Graph Tsetlin Machine (GTM) graph objects.
"""

import argparse
import os
import sys
import pickle
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation import DatasetBuilder
from src.utils import Config

def main():
    parser = argparse.ArgumentParser(description='Build GTM-compatible datasets from C-generated games')
    parser.add_argument('--board-size', type=int, default=5, help='Board size')
    parser.add_argument('--hypervector-size', type=int, default=64, help='Hypervector size')
    parser.add_argument('--hypervector-bits', type=int, default=4, help='Hypervector bits')
    parser.add_argument('--stages', type=str, default='all',
                        help='Which stages to process: "all" or comma-separated like "0,-2,-5"')
    parser.add_argument('--train-file', type=str, default=None,
                        help='Optional path to train npz (default: data/train_games_<board>x<board>.npz)')
    parser.add_argument('--test-file', type=str, default=None,
                        help='Optional path to test npz (default: data/test_games_<board>x<board>.npz)')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Directory to write GTM pkl outputs (default: data)')
    parser.add_argument('--output-prefix', type=str, default='',
                        help='Prefix for output GTM filenames (default: "")')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("BUILDING GTM DATASETS FROM C-GENERATED DATA")
    print("="*60)
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Hypervector size: {args.hypervector_size}")
    print(f"Hypervector bits: {args.hypervector_bits}")
    print(f"Stages to process: {args.stages}")
    print("="*60)

    # Load C-generated game data
    train_file = args.train_file or f'data/train_games_{args.board_size}x{args.board_size}.npz'
    test_file = args.test_file or f'data/test_games_{args.board_size}x{args.board_size}.npz'

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

    # Detect available stages
    available_stages = []
    for key in train_data.keys():
        if key.startswith('states_at_'):
            stage = key.replace('states_at_', '')
            available_stages.append(stage)

    if not available_stages:
        print("ERROR: No stage data found in training file!")
        sys.exit(1)

    print(f"\nAvailable stages: {available_stages}")

    # Determine which stages to process
    if args.stages == 'all':
        stages_to_process = available_stages
    else:
        stages_to_process = [s.strip() for s in args.stages.split(',')]
        # Validate stages exist
        for stage in stages_to_process:
            if stage not in available_stages:
                print(f"ERROR: Stage '{stage}' not found in data!")
                print(f"Available: {available_stages}")
                sys.exit(1)

    print(f"Processing stages: {stages_to_process}")

    train_winners = train_data['winners']
    test_winners = test_data['winners']

    print(f"Train: {len(train_winners)} games")
    print(f"Test: {len(test_winners)} games")

    # Initialize dataset builder
    builder = DatasetBuilder(board_size=args.board_size)

    # Fixed seed for reproducible hypervectors
    HYPERVECTOR_SEED = 42

    # Process each stage
    for stage in stages_to_process:
        print("\n" + "="*60)
        print(f"PROCESSING STAGE: {stage}")
        print("="*60)

        train_boards = train_data[f'states_at_{stage}']
        test_boards = test_data[f'states_at_{stage}']

        print(f"Train boards shape: {train_boards.shape}")
        print(f"Test boards shape: {test_boards.shape}")

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

        # Save datasets for this stage
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        prefix = args.output_prefix
        train_output = output_dir / f'{prefix}train_gtm_{args.board_size}x{args.board_size}_{stage}.pkl'
        test_output = output_dir / f'{prefix}test_gtm_{args.board_size}x{args.board_size}_{stage}.pkl'

        print(f"\nSaving training dataset to {train_output}...")
        with open(train_output, 'wb') as f:
            pickle.dump({'graphs': train_graphs, 'labels': train_labels}, f)
        print(f"  Saved! ({len(train_labels)} samples)")

        print(f"\nSaving test dataset to {test_output}...")
        with open(test_output, 'wb') as f:
            pickle.dump({'graphs': test_graphs, 'labels': test_labels}, f)
        print(f"  Saved! ({len(test_labels)} samples)")

    print("\n" + "="*60)
    print("SUCCESS! GTM datasets built and saved for all stages.")
    print("="*60)
    print("\nYou can now train models for each stage:")
    for stage in stages_to_process:
        print(f"  python scripts/2_train_model.py --board-size {args.board_size} --stage {stage}")
    print("="*60)


if __name__ == '__main__':
    main()
