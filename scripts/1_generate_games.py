"""
Generate Hex game datasets using the OFFICIAL C implementation (hex_datagen).
This ensures we use the exact same game generation as the benchmark.
"""

import argparse
import subprocess
import numpy as np
import os
from pathlib import Path

def generate_games_with_c(board_size: int, num_games: int, output_file: str):
    """Generate games using C executable and save to npz format."""
    
    print(f"\n{'='*70}")
    print(f"GENERATING {num_games} GAMES USING C IMPLEMENTATION")
    print(f"Board size: {board_size}x{board_size}")
    print(f"{'='*70}")
    
    # Choose the right executable
    if board_size == 5:
        exe_name = "hex_datagen_5x5.exe"
    elif board_size == 10:
        exe_name = "hex_datagen_10x10.exe"
    elif board_size == 11:
        exe_name = "hex_datagen_11x11.exe"
    else:
        print(f"ERROR: Unsupported board size {board_size}")
        return False
    if not os.path.exists(exe_name):
        print(f"ERROR: {exe_name} not found!")
        print("Please compile with: compile_datagen.bat")
        return False
    
    print(f"\nRunning {exe_name}...")
    print(f"This will generate {num_games} games...")
    
    # Run the C program and capture output
    try:
        result = subprocess.run(
            [exe_name, str(num_games)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        lines = result.stdout.strip().split('\n')
        
        if len(lines) != num_games:
            print(f"WARNING: Expected {num_games} lines, got {len(lines)}")
        
        print(f"Successfully generated {len(lines)} games")
        
        # Parse game data
        winners = []
        boards_end = []
        
        for i, line in enumerate(lines):
            parts = line.strip().split(',')
            if len(parts) != board_size * board_size + 1:
                print(f"WARNING: Line {i} has wrong format (got {len(parts)} parts, expected {board_size*board_size + 1})")
                continue
            
            winner = int(parts[0])
            board_flat = [int(x) for x in parts[1:]]
            board = np.array(board_flat, dtype=np.int8).reshape(board_size, board_size)
            
            winners.append(winner)
            boards_end.append(board)
            
            if (i + 1) % 1000 == 0:
                print(f"  Parsed {i + 1} games...")
        
        winners = np.array(winners, dtype=np.int8)
        boards_end = np.array(boards_end, dtype=np.int8)
        
        print(f"\nData summary:")
        print(f"  Total games: {len(winners)}")
        print(f"  Player 0 wins: {np.sum(winners == 0)} ({100*np.sum(winners == 0)/len(winners):.1f}%)")
        print(f"  Player 1 wins: {np.sum(winners == 1)} ({100*np.sum(winners == 1)/len(winners):.1f}%)")
        print(f"  Board shape: {boards_end.shape}")
        
        # Save to npz format (compatible with existing code)
        print(f"\nSaving to {output_file}...")
        np.savez_compressed(
            output_file,
            num_games=len(winners),
            board_size=board_size,
            winners=winners,
            states_at_end=boards_end,
            # Note: C version doesn't track intermediate states
            # If needed, we can modify hex_datagen.c to output them
        )
        
        print(f"[OK] Saved {len(winners)} games to {output_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {exe_name}:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate Hex games using C implementation')
    parser.add_argument('--board-size', type=int, default=5, choices=[5, 10, 11],
                        help='Board size (5, 10, or 11)')
    parser.add_argument('--num-train', type=int, default=10000,
                        help='Number of training games')
    parser.add_argument('--num-test', type=int, default=2000,
                        help='Number of test games')
    
    args = parser.parse_args()
    
    # Create data directory
    Path('data').mkdir(exist_ok=True)
    
    print("="*70)
    print("HEX GAME GENERATION USING C IMPLEMENTATION")
    print("="*70)
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Training games: {args.num_train}")
    print(f"Test games: {args.num_test}")
    print("="*70)
    
    # Generate training data
    train_file = f'data/train_games_{args.board_size}x{args.board_size}.npz'
    success_train = generate_games_with_c(args.board_size, args.num_train, train_file)
    
    if not success_train:
        print("\nFailed to generate training data!")
        return
    
    # Generate test data
    test_file = f'data/test_games_{args.board_size}x{args.board_size}.npz'
    success_test = generate_games_with_c(args.board_size, args.num_test, test_file)
    
    if not success_test:
        print("\nFailed to generate test data!")
        return
    
    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"Training data: {train_file}")
    print(f"Test data: {test_file}")
    print("\nNext step:")
    print(f"  python scripts/1b_build_gtm_datasets.py --board-size {args.board_size}")
    print("="*70)


if __name__ == '__main__':
    main()

