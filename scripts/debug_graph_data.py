"""
Debug: Inspect actual graph data to see what GTM is receiving
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np

print("="*70)
print("DEBUGGING GRAPH DATA")
print("="*70)

# Load datasets
with open('data/train_gtm_5x5_end.pkl', 'rb') as f:
    train_data = pickle.load(f)

train_graphs = train_data['graphs']
train_labels = train_data['labels']

print(f"\nDataset info:")
print(f"  Graphs: {train_graphs.number_of_graphs}")
print(f"  Labels: {len(train_labels)}")
print(f"  Label distribution: {np.bincount(train_labels)}")

# Check a few specific games
print("\n" + "="*70)
print("INSPECTING INDIVIDUAL GAMES")
print("="*70)

# Load the original board states to compare
from src.data_generation import GameGenerator
generator = GameGenerator(board_size=5)
game_data = generator.load_dataset('data/train_games_5x5.npz')

print(f"\nOriginal game data:")
print(f"  Games: {game_data['num_games']}")
print(f"  Winners: {len(game_data['winners'])}")

# Look at first 5 games
for i in range(5):
    board = game_data['states_at_end'][i]
    winner = game_data['winners'][i]
    label = train_labels[i]
    
    print(f"\n--- Game {i} ---")
    print(f"Winner (from game): {winner}")
    print(f"Label (in GTM): {label}")
    print(f"Match: {winner == label}")
    
    print(f"\nBoard state:")
    print(board)
    
    # Count pieces
    player0_pieces = np.sum(board == 1)
    player1_pieces = np.sum(board == 2)
    empty = np.sum(board == 0)
    
    print(f"Player 0 pieces: {player0_pieces}")
    print(f"Player 1 pieces: {player1_pieces}")
    print(f"Empty cells: {empty}")
    print(f"Total: {player0_pieces + player1_pieces + empty} (should be 25)")
    
    # Check if this is a finished game
    if empty > 0:
        print(f"[WARNING] Game has {empty} empty cells - not finished!")

print("\n" + "="*70)
print("CHECKING FOR DATA ISSUES")
print("="*70)

# Check all games for common issues
all_boards = game_data['states_at_end']
all_winners = game_data['winners']

# Issue 1: Empty games
empty_games = []
for i in range(len(all_boards)):
    if np.sum(all_boards[i] > 0) < 5:  # Less than 5 pieces
        empty_games.append(i)

if empty_games:
    print(f"\n[ERROR] Found {len(empty_games)} nearly empty games!")
    print(f"First few: {empty_games[:10]}")
else:
    print(f"\n[OK] No empty games found")

# Issue 2: Unfinished games
unfinished = []
for i in range(len(all_boards)):
    if np.sum(all_boards[i] == 0) > 10:  # More than 10 empty cells
        unfinished.append(i)

if unfinished:
    print(f"\n[WARNING] Found {len(unfinished)} games with >10 empty cells")
    print(f"These might be unfinished games")
else:
    print(f"\n[OK] All games appear finished")

# Issue 3: Label distribution
print(f"\n[INFO] Label distribution:")
print(f"  Player 0 wins: {np.sum(all_winners == 0)} ({100*np.sum(all_winners == 0)/len(all_winners):.1f}%)")
print(f"  Player 1 wins: {np.sum(all_winners == 1)} ({100*np.sum(all_winners == 1)/len(all_winners):.1f}%)")

# Issue 4: Check if graphs have edges
print(f"\n[INFO] Graph structure:")
print(f"  Number of graphs: {train_graphs.number_of_graphs}")
print(f"  This is a Graphs object - cannot easily inspect internal structure")
print(f"  But we know each graph should have 25 nodes with hex adjacency edges")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("If you see any [ERROR] or [WARNING] above, that could be the issue.")
print("Otherwise, the data looks correct and the problem is with GTM parameters.")
print("="*70)

