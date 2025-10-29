"""Check if edges are bidirectional"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation import DatasetBuilder

builder = DatasetBuilder(board_size=5)

# Check neighbor map
print("Checking edge directionality:")
print("\nSample: R2C2 (center cell)")
print(f"  Neighbors of R2C2: {builder._neighbor_map['R2C2']}")

# Check if edges are bidirectional
print("\nBidirectionality check:")
for node, neighbors in list(builder._neighbor_map.items())[:5]:
    print(f"\n{node} -> {neighbors}")
    for neighbor in neighbors:
        if node in builder._neighbor_map[neighbor]:
            print(f"  [OK] {neighbor} -> {node} (bidirectional)")
        else:
            print(f"  [ERROR] {neighbor} -> {node} (MISSING!)")

