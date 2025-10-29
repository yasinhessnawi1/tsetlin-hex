"""
Diagnostic script for C-generated Hex game data.
Validates that the C implementation is generating correct game data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pickle

print("="*70)
print("DIAGNOSTIC: C-GENERATED HEX GAME DATA")
print("="*70)

# Load C-generated data
print("\nLoading C-generated data...")
train_data = np.load('data/train_games_5x5.npz')
test_data = np.load('data/test_games_5x5.npz')

train_boards = train_data['states_at_end']
train_winners = train_data['winners']
test_boards = test_data['states_at_end']
test_winners = test_data['winners']

print(f"Train games: {len(train_winners)}")
print(f"Test games: {len(test_winners)}")

# Check 1: Board dimensions
print("\n" + "="*70)
print("CHECK 1: BOARD DIMENSIONS")
print("="*70)
expected_shape = (5, 5)
if train_boards.shape[1:] == expected_shape:
    print(f"[OK] Board shape correct: {expected_shape}")
else:
    print(f"[ERROR] Board shape wrong: {train_boards.shape[1:]}, expected {expected_shape}")

# Check 2: Valid cell values
print("\n" + "="*70)
print("CHECK 2: CELL VALUES")
print("="*70)
all_boards = np.concatenate([train_boards, test_boards])
unique_values = np.unique(all_boards)
print(f"Unique values in boards: {unique_values}")
if set(unique_values).issubset({0, 1, 2}):
    print("[OK] All values are valid (0=Empty, 1=Player0, 2=Player1)")
else:
    print(f"[ERROR] Invalid values found: {unique_values}")

# Check 3: Winner labels
print("\n" + "="*70)
print("CHECK 3: WINNER LABELS")
print("="*70)
unique_winners = np.unique(np.concatenate([train_winners, test_winners]))
print(f"Unique winner labels: {unique_winners}")
if set(unique_winners).issubset({0, 1}):
    print("[OK] All winner labels are valid (0 or 1)")
else:
    print(f"[ERROR] Invalid winner labels: {unique_winners}")

# Check 4: Class distribution
print("\n" + "="*70)
print("CHECK 4: CLASS DISTRIBUTION")
print("="*70)
train_p0 = np.sum(train_winners == 0)
train_p1 = np.sum(train_winners == 1)
test_p0 = np.sum(test_winners == 0)
test_p1 = np.sum(test_winners == 1)

print(f"Training set:")
print(f"  Player 0 wins: {train_p0} ({100*train_p0/len(train_winners):.1f}%)")
print(f"  Player 1 wins: {train_p1} ({100*train_p1/len(train_winners):.1f}%)")
print(f"\nTest set:")
print(f"  Player 0 wins: {test_p0} ({100*test_p0/len(test_winners):.1f}%)")
print(f"  Player 1 wins: {test_p1} ({100*test_p1/len(test_winners):.1f}%)")

imbalance = abs(train_p0 - train_p1) / len(train_winners)
if imbalance < 0.15:
    print(f"\n[OK] Class distribution is balanced (imbalance: {100*imbalance:.1f}%)")
else:
    print(f"\n[WARNING] Significant class imbalance: {100*imbalance:.1f}%")

# Check 5: Board occupancy
print("\n" + "="*70)
print("CHECK 5: BOARD OCCUPANCY (Empty cells at game end)")
print("="*70)
empty_cells = []
for board in train_boards[:100]:  # Sample first 100
    empty = np.sum(board == 0)
    empty_cells.append(empty)

print(f"Empty cells at game end (first 100 games):")
print(f"  Min: {np.min(empty_cells)}")
print(f"  Max: {np.max(empty_cells)}")
print(f"  Average: {np.mean(empty_cells):.1f}")
print(f"  Median: {np.median(empty_cells):.0f}")

if np.max(empty_cells) > 15:
    print("\n[WARNING] Some games have many empty cells (>15)")
    print("This means games ended very early - check if C code stops at winner correctly")
elif np.mean(empty_cells) < 3:
    print("\n[OK] Games have few empty cells - most cells filled before winner")
else:
    print("\n[OK] Reasonable number of empty cells for Hex end states")

# Check 6: Sample game validation
print("\n" + "="*70)
print("CHECK 6: SAMPLE GAME VALIDATION")
print("="*70)

def check_connectivity(board, player, start_edge, end_edge):
    """Simple flood fill to check if player has a winning path."""
    visited = np.zeros_like(board, dtype=bool)
    
    # Get starting positions
    if start_edge == 'top':
        queue = [(0, j) for j in range(board.shape[1]) if board[0, j] == player + 1]
    elif start_edge == 'left':
        queue = [(i, 0) for i in range(board.shape[0]) if board[i, 0] == player + 1]
    else:
        return False
    
    # Hex neighbors (6 directions)
    neighbors = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    
    while queue:
        r, c = queue.pop(0)
        
        if visited[r, c]:
            continue
        visited[r, c] = True
        
        # Check if reached goal edge
        if end_edge == 'bottom' and r == board.shape[0] - 1:
            return True
        if end_edge == 'right' and c == board.shape[1] - 1:
            return True
        
        # Add neighbors
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
                if board[nr, nc] == player + 1 and not visited[nr, nc]:
                    queue.append((nr, nc))
    
    return False

# Check first 20 games
print("Checking first 20 games for correct winners...")
errors = 0
for i in range(min(20, len(train_boards))):
    board = train_boards[i]
    winner = train_winners[i]
    
    # Player 0 connects top-bottom, Player 1 connects left-right
    p0_wins = check_connectivity(board, 0, 'top', 'bottom')
    p1_wins = check_connectivity(board, 1, 'left', 'right')
    
    if winner == 0 and not p0_wins:
        print(f"  Game {i}: [ERROR] Winner labeled as P0 but no P0 path found!")
        errors += 1
    elif winner == 1 and not p1_wins:
        print(f"  Game {i}: [ERROR] Winner labeled as P1 but no P1 path found!")
        errors += 1

if errors == 0:
    print("[OK] All sampled games have correct winners!")
else:
    print(f"\n[ERROR] Found {errors} games with incorrect winner labels!")

# Check 7: GTM dataset validity
print("\n" + "="*70)
print("CHECK 7: GTM DATASET VALIDITY")
print("="*70)

try:
    with open('data/train_gtm_5x5_end.pkl', 'rb') as f:
        gtm_data = pickle.load(f)
    
    gtm_graphs = gtm_data['graphs']
    gtm_labels = gtm_data['labels']
    
    print(f"GTM graphs: {gtm_graphs.number_of_graphs}")
    print(f"GTM labels: {len(gtm_labels)}")
    
    if gtm_graphs.number_of_graphs == len(gtm_labels):
        print("[OK] GTM dataset size matches")
    else:
        print("[ERROR] GTM dataset size mismatch!")
    
    if len(gtm_labels) == len(train_winners):
        print("[OK] GTM labels count matches training data")
    else:
        print("[ERROR] GTM labels don't match training data count!")
    
    if np.array_equal(gtm_labels, train_winners):
        print("[OK] GTM labels match original winners")
    else:
        print("[WARNING] GTM labels differ from original winners")
        
except Exception as e:
    print(f"[ERROR] Could not load GTM dataset: {e}")

# Final summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("C-generated data validation complete.")
print("\nKey findings:")
print(f"  - {len(train_winners)} training games, {len(test_winners)} test games")
print(f"  - Average empty cells: {np.mean(empty_cells):.1f}")
print(f"  - Class balance: {train_p0} vs {train_p1}")
if errors == 0:
    print(f"  - All sampled games have correct connectivity")
else:
    print(f"  - {errors} games with potential winner errors")
print("="*70)

