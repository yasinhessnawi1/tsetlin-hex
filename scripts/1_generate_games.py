"""
Generate Hex game datasets with multi-stage support using hex_datagen_stages.c
Captures board states at multiple points during the game (e.g., end, -2, -5)
"""

import argparse
import subprocess
import numpy as np
import os
from pathlib import Path

def generate_games_with_stages(board_size: int, num_games: int, output_file: str, stages: list):
    """Generate games using C executable with stage tracking and save to npz format."""

    print(f"\n{'='*70}")
    print(f"GENERATING {num_games} GAMES WITH MULTI-STAGE TRACKING")
    print(f"Board size: {board_size}x{board_size}")
    print(f"Stages: {stages}")
    print(f"{'='*70}")

    # Choose the right executable (cross-platform)
    import platform
    is_windows = platform.system() == 'Windows'

    if board_size == 5:
        exe_name = os.path.join("hex_binaries", "hex_datagen_5x5.exe" if is_windows else "hex_datagen_5x5")
    elif board_size == 7:
        exe_name = os.path.join("hex_binaries", "hex_datagen_7x7.exe" if is_windows else "hex_datagen_7x7")
    elif board_size == 10:
        exe_name = os.path.join("hex_binaries", "hex_datagen_10x10.exe" if is_windows else "hex_datagen_10x10")
    elif board_size == 11:
        exe_name = os.path.join("hex_binaries", "hex_datagen_11x11.exe" if is_windows else "hex_datagen_11x11")
    else:
        print(f"ERROR: Unsupported board size {board_size}")
        return False

    if not os.path.exists(exe_name):
        print(f"ERROR: {exe_name} not found!")
        if is_windows:
            print("Please compile with: hex_binaries\\compile_datagen.bat")
        else:
            print("Please compile with: ./hex_binaries/compile_linux.sh")
        return False

    print(f"\nRunning {exe_name}...")
    print(f"This will generate {num_games} games with {len(stages)} stages each...")

    # Run the C program with stage arguments
    # Format: ./hex_datagen_5x5 <num_games> <stage1> <stage2> <stage3> ...
    cmd = [exe_name, str(num_games)] + [str(s) for s in stages]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Parse the output
        # Format: winner,stage0_board,stage1_board,stage2_board,...
        lines = result.stdout.strip().split('\n')

        if len(lines) != num_games:
            print(f"WARNING: Expected {num_games} lines, got {len(lines)}")

        print(f"Successfully generated {len(lines)} games")

        # Parse game data
        winners = []
        boards_by_stage = [[] for _ in stages]  # One list per stage

        expected_parts = 1 + len(stages) * board_size * board_size  # winner + (stages * board cells)

        for i, line in enumerate(lines):
            parts = line.strip().split(',')
            if len(parts) != expected_parts:
                print(f"WARNING: Line {i} has wrong format (got {len(parts)} parts, expected {expected_parts})")
                continue

            winner = int(parts[0])
            winners.append(winner)

            # Parse each stage's board state
            offset = 1
            for stage_idx in range(len(stages)):
                board_flat = [int(x) for x in parts[offset:offset + board_size*board_size]]
                board = np.array(board_flat, dtype=np.int8).reshape(board_size, board_size)
                boards_by_stage[stage_idx].append(board)
                offset += board_size * board_size

            if (i + 1) % 1000 == 0:
                print(f"  Parsed {i + 1} games...")

        winners = np.array(winners, dtype=np.int8)

        # Convert boards to numpy arrays
        boards_arrays = [np.array(boards, dtype=np.int8) for boards in boards_by_stage]

        print(f"\nData summary:")
        print(f"  Total games: {len(winners)}")
        print(f"  Player 0 wins: {np.sum(winners == 0)} ({100*np.sum(winners == 0)/len(winners):.1f}%)")
        print(f"  Player 1 wins: {np.sum(winners == 1)} ({100*np.sum(winners == 1)/len(winners):.1f}%)")
        print(f"  Stages tracked: {len(stages)}")
        for stage_idx, stage in enumerate(stages):
            print(f"    Stage {stage}: {boards_arrays[stage_idx].shape}")

        # Save to npz format with stage labels
        print(f"\nSaving to {output_file}...")
        save_dict = {
            'num_games': len(winners),
            'board_size': board_size,
            'winners': winners,
            'stages': np.array(stages, dtype=np.int32)
        }

        # Add each stage's data with descriptive key
        for stage_idx, stage in enumerate(stages):
            key = f'states_at_{stage}'
            save_dict[key] = boards_arrays[stage_idx]

        np.savez_compressed(output_file, **save_dict)

        print(f"[OK] Saved {len(winners)} games to {output_file}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR running {exe_name}:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_stages(stages_str):
    """Parse stages argument: '0,-2,-5' -> [0, -2, -5]"""
    if not stages_str:
        return [0]  # Default: end only
    return [int(s.strip()) for s in stages_str.split(',')]


def main():
    parser = argparse.ArgumentParser(description='Generate Hex games with multi-stage tracking')
    parser.add_argument('--board-size', type=int, default=5, choices=[5, 7, 10, 11],
                        help='Board size (5, 7, 10, or 11)')
    parser.add_argument('--num-train', type=int, default=10000,
                        help='Number of training games')
    parser.add_argument('--num-test', type=int, default=2000,
                        help='Number of test games')
    parser.add_argument('--save-states', type=str, default='0,-2,-5',
                        help='Stages to save (comma-separated). 0=end, -2=2 before end, etc. Default: "0,-2,-5"')

    args = parser.parse_args()

    # Parse stages
    stages = parse_stages(args.save_states)

    # Create data directory
    Path('data').mkdir(exist_ok=True)

    print("="*70)
    print("HEX GAME GENERATION WITH MULTI-STAGE TRACKING")
    print("="*70)
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Training games: {args.num_train}")
    print(f"Test games: {args.num_test}")
    print(f"Stages: {stages}")
    print("="*70)

    # Generate training data
    train_file = f'data/train_games_{args.board_size}x{args.board_size}.npz'
    success_train = generate_games_with_stages(args.board_size, args.num_train, train_file, stages)

    if not success_train:
        print("\nFailed to generate training data!")
        return

    # Generate test data
    test_file = f'data/test_games_{args.board_size}x{args.board_size}.npz'
    success_test = generate_games_with_stages(args.board_size, args.num_test, test_file, stages)

    if not success_test:
        print("\nFailed to generate test data!")
        return

    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"Training data: {train_file}")
    print(f"Test data: {test_file}")
    print("\nNext step:")
    print(f"  python scripts/1b_build_gtm_datasets.py --board-size {args.board_size} --stages all")
    print("="*70)


if __name__ == '__main__':
    main()
