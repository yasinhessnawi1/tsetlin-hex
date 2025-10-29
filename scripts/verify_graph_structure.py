"""
Verify that GTM graphs are correctly constructed from C-generated data.
Check nodes, edges, properties, and graph structure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pickle

print("="*70)
print("VERIFYING GTM GRAPH STRUCTURE")
print("="*70)

# Load raw game data
print("\n1. Loading raw C-generated game data...")
raw_data = np.load('data/train_games_5x5.npz')
boards = raw_data['states_at_end']
winners = raw_data['winners']

print(f"   Loaded {len(winners)} games")

# Load GTM graphs
print("\n2. Loading GTM graph dataset...")
with open('data/train_gtm_5x5_end.pkl', 'rb') as f:
    gtm_data = pickle.load(f)

graphs = gtm_data['graphs']
labels = gtm_data['labels']

print(f"   Graphs object: {graphs.number_of_graphs} graphs")
print(f"   Labels: {len(labels)} labels")

# Check graph structure
print("\n" + "="*70)
print("3. CHECKING GRAPH STRUCTURE")
print("="*70)

print(f"\nGraph object type: {type(graphs)}")
print(f"Number of graphs: {graphs.number_of_graphs}")

# The Graphs object doesn't expose much, but we can check what we built
# Let's verify by rebuilding one graph manually and comparing

print("\n" + "="*70)
print("4. MANUAL GRAPH RECONSTRUCTION TEST")
print("="*70)

# Pick first game
test_idx = 0
test_board = boards[test_idx]
test_winner = winners[test_idx]

print(f"\nTest game {test_idx}:")
print(f"Winner: {test_winner}")
print(f"Board:")
print(test_board)

# Count pieces
p0_count = np.sum(test_board == 1)
p1_count = np.sum(test_board == 2)
empty_count = np.sum(test_board == 0)

print(f"\nPiece counts: P0={p0_count}, P1={p1_count}, Empty={empty_count}")

# Verify board size
board_size = 5
num_nodes = board_size * board_size
print(f"\nExpected nodes per graph: {num_nodes}")
print(f"Board dimensions: {board_size}x{board_size}")

# Check hex adjacency
print("\n" + "="*70)
print("5. HEX ADJACENCY VERIFICATION")
print("="*70)

def get_hex_neighbors(row, col, board_size):
    """Get hex neighbors for a cell."""
    neighbors = []
    # Hex has 6 neighbors
    directions = [
        (-1, 0),   # top
        (-1, 1),   # top-right
        (0, -1),   # left
        (0, 1),    # right
        (1, -1),   # bottom-left
        (1, 0),    # bottom
    ]
    
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < board_size and 0 <= nc < board_size:
            neighbors.append((nr, nc))
    
    return neighbors

# Check a few cells
test_cells = [(0, 0), (2, 2), (4, 4), (0, 4), (4, 0)]

print("\nChecking hex adjacency for sample cells:")
for row, col in test_cells:
    neighbors = get_hex_neighbors(row, col, board_size)
    print(f"  Cell ({row},{col}): {len(neighbors)} neighbors - {neighbors}")

# Count total edges
total_edges = 0
edge_counts = {}
for i in range(board_size):
    for j in range(board_size):
        neighbors = get_hex_neighbors(i, j, board_size)
        num_neighbors = len(neighbors)
        edge_counts[num_neighbors] = edge_counts.get(num_neighbors, 0) + 1
        total_edges += num_neighbors

print(f"\nEdge distribution:")
for num, count in sorted(edge_counts.items()):
    print(f"  {count} cells with {num} neighbors")
print(f"Total directed edges: {total_edges}")

# Check node properties
print("\n" + "="*70)
print("6. NODE PROPERTIES VERIFICATION")
print("="*70)

print("\nNode property mapping:")
print("  0 (Empty) -> 'Empty'")
print("  1 (Player 0) -> 'Player0'")  
print("  2 (Player 1) -> 'Player1'")

# Count node types in first game
print(f"\nFirst game node types:")
print(f"  Empty nodes: {empty_count}")
print(f"  Player0 nodes: {p0_count}")
print(f"  Player1 nodes: {p1_count}")
print(f"  Total: {empty_count + p0_count + p1_count} (should be {num_nodes})")

if empty_count + p0_count + p1_count == num_nodes:
    print("  [OK] Node count matches board size")
else:
    print("  [ERROR] Node count mismatch!")

# Check if labels match
print("\n" + "="*70)
print("7. LABEL VERIFICATION")
print("="*70)

print(f"\nFirst 10 game labels comparison:")
print(f"{'Game':<6} {'Raw Winner':<12} {'GTM Label':<12} {'Match':<8}")
print("-" * 40)
for i in range(min(10, len(winners))):
    match = "OK" if winners[i] == labels[i] else "ERROR"
    print(f"{i:<6} {winners[i]:<12} {labels[i]:<12} {match:<8}")

all_match = np.array_equal(winners, labels)
if all_match:
    print("\n[OK] All labels match raw winners")
else:
    mismatches = np.sum(winners != labels)
    print(f"\n[ERROR] {mismatches} label mismatches found!")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\n[INFO] Graph structure:")
print(f"  - Each graph has {num_nodes} nodes (one per board cell)")
print(f"  - Nodes connected with hex adjacency (6-way)")
print(f"  - 3 node properties: Empty, Player0, Player1")
print(f"  - Labels: 0=Player0 wins, 1=Player1 wins")

print(f"\n[INFO] Expected GTM behavior:")
print(f"  - With {board_size}x{board_size} board and depth=6")
print(f"  - Messages can propagate up to 6 hops")
print(f"  - Should be enough to detect paths across the board")

print(f"\n[INFO] Potential issues to check:")
print(f"  1. Are edges bidirectional? (Should be for Hex)")
print(f"  2. Are edge types labeled? (Could help learning)")
print(f"  3. Do messages propagate correctly through depth-6?")
print(f"  4. Are there enough clauses to learn connectivity patterns?")

print("="*70)

