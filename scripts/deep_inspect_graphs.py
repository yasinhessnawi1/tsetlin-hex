"""
Deep inspection of graph objects to understand what's happening.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data_generation import DatasetBuilder

print("="*70)
print("DEEP GRAPH INSPECTION")
print("="*70)

# Create two IDENTICAL boards - all Player 0
board1 = np.ones((5, 5), dtype=np.int32)
board2 = np.ones((5, 5), dtype=np.int32)

# Create two DIFFERENT boards - one all Player 0, one all Player 1
board3 = np.ones((5, 5), dtype=np.int32)
board4 = np.full((5, 5), 2, dtype=np.int32)

print("\nCreating graphs from boards:")
print(f"Board 1 (all 1s): shape={board1.shape}, unique values={np.unique(board1)}")
print(f"Board 2 (all 1s): shape={board2.shape}, unique values={np.unique(board2)}")
print(f"Board 3 (all 1s): shape={board3.shape}, unique values={np.unique(board3)}")
print(f"Board 4 (all 2s): shape={board4.shape}, unique values={np.unique(board4)}")

builder = DatasetBuilder(board_size=5)

# Test 1: Two identical boards should produce "similar" graphs
print("\n" + "="*70)
print("TEST 1: Two identical boards")
print("="*70)

boards_identical = np.array([board1, board2])
labels_identical = np.array([0, 0], dtype=np.uint32)

graphs_identical, _ = builder.create_graphs_from_game_data(
    boards_identical,
    labels_identical,
    hypervector_size=128,
    hypervector_bits=4,
    verbose=False
)

print(f"Created {graphs_identical.number_of_graphs} graphs")

# Test 2: Two different boards should produce different graphs
print("\n" + "="*70)
print("TEST 2: Two different boards (all 1s vs all 2s)")
print("="*70)

boards_different = np.array([board3, board4])
labels_different = np.array([0, 1], dtype=np.uint32)

graphs_different, _ = builder.create_graphs_from_game_data(
    boards_different,
    labels_different,
    hypervector_size=128,
    hypervector_bits=4,
    verbose=False
)

print(f"Created {graphs_different.number_of_graphs} graphs")

# Test 3: Check what happens when we create train and test graphs separately
print("\n" + "="*70)
print("TEST 3: Train and Test created separately")
print("="*70)

train_boards = np.ones((3, 5, 5), dtype=np.int32)
test_boards = np.ones((2, 5, 5), dtype=np.int32)

train_labels = np.array([0, 0, 0], dtype=np.uint32)
test_labels = np.array([0, 0], dtype=np.uint32)

print("Building TRAIN graphs...")
graphs_train, _ = builder.create_graphs_from_game_data(
    train_boards,
    train_labels,
    hypervector_size=128,
    hypervector_bits=4,
    verbose=False
)

print("Building TEST graphs...")
graphs_test, _ = builder.create_graphs_from_game_data(
    test_boards,
    test_labels,
    hypervector_size=128,
    hypervector_bits=4,
    verbose=False
)

print(f"\nTrain graphs: {graphs_train.number_of_graphs}")
print(f"Test graphs: {graphs_test.number_of_graphs}")

# HYPOTHESIS: The hypervectors are RANDOMLY generated each time!
# This means train and test graphs have INCOMPATIBLE encodings!

print("\n" + "="*70)
print("CRITICAL HYPOTHESIS")
print("="*70)
print("The GraphTsetlinMachine Graphs class may generate RANDOM hypervectors")
print("for each symbol during initialization. If train and test graphs are")
print("created separately, they would have INCOMPATIBLE encodings!")
print("")
print("This would explain:")
print("  - 100% training accuracy (learns the training encoding)")
print("  - 0% test accuracy (test encoding is completely different)")
print("  - Real games stuck at 57% (random baseline)")
print("")
print("SOLUTION: Create ALL graphs (train + test) at once, then split them!")
print("="*70)

