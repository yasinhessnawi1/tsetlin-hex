#!/bin/bash

# This script trains the GTM model with optimized parameters
# Use this after generating data with generate_5x5_data.sh

python scripts/2_train_model.py --board-size 5 --stage end --epochs 50 --clauses 100 --depth 6

read -p "Press Enter to exit..."
