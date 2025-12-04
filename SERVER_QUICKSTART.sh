#!/bin/bash
# Quick-start script for server training
# Run this on your Linux server (V100/A100)

echo "============================================================"
echo "SERVER QUICK-START - Competition Training"
echo "============================================================"
echo ""
echo "This script will:"
echo "1. Check data integrity (P0 vs P1 wins)"
echo "2. Regenerate data if needed"
echo "3. Train with optimal settings for 100% accuracy"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "STEP 1: Checking Data Integrity"
echo "============================================================"
echo ""

# Check if data exists
if [ ! -f "data/train_games_5x5.npz" ]; then
    echo -e "${RED}ERROR: Data not found!${NC}"
    echo "Please generate data first:"
    echo "  ./generate_5x5_data.sh"
    exit 1
fi

# Check P0 vs P1 distribution
python3 -c "
import numpy as np
import sys

data = np.load('data/train_games_5x5.npz')
total = len(data['winners'])
p0_wins = sum(data['winners'] == 0)
p1_wins = sum(data['winners'] == 1)

print(f'Total games: {total}')
print(f'P0 wins: {p0_wins} ({100.0*p0_wins/total:.1f}%)')
print(f'P1 wins: {p1_wins} ({100.0*p1_wins/total:.1f}%)')

# Check for bug
if p1_wins == 0:
    print('')
    print('❌ BUG DETECTED: No P1 wins!')
    print('Data was generated with buggy C code.')
    print('You MUST regenerate!')
    sys.exit(1)
elif p1_wins < total * 0.35 or p1_wins > total * 0.50:
    print('')
    print('⚠️  WARNING: Unusual P1 win rate!')
    print(f'Expected: 40-48%, Got: {100.0*p1_wins/total:.1f}%')
    print('Data may be corrupted. Consider regenerating.')
    sys.exit(2)
else:
    print('')
    print('✓ Data looks good!')
    sys.exit(0)
"

DATA_CHECK=$?

if [ $DATA_CHECK -eq 1 ]; then
    echo ""
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}DATA BUG DETECTED - MUST REGENERATE!${NC}"
    echo -e "${RED}============================================================${NC}"
    echo ""
    read -p "Regenerate data now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting old data..."
        rm -f data/*5x5*

        echo "Recompiling C code with bug fix..."
        ./hex_binaries/compile_linux.sh

        echo "Generating 1M training games..."
        ./generate_5x5_data.sh
    else
        echo "Cannot proceed with buggy data. Exiting."
        exit 1
    fi
elif [ $DATA_CHECK -eq 2 ]; then
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "============================================================"
echo "STEP 2: Training for 100% Accuracy"
echo "============================================================"
echo ""
echo "Starting with Stage 0 (easiest)..."
echo "Using 2000 clauses, 150 epochs, s=0.01"
echo ""
read -p "Press Enter to start training..."

# Train Stage 0
echo ""
echo "============================================================"
echo "TRAINING STAGE 0 (END)"
echo "============================================================"
python3 scripts/2_train_model.py \
    --board-size 5 \
    --stage 0 \
    --epochs 150 \
    --clauses 2000 \
    --depth 6 \
    --T 1500 \
    --s 0.01

echo ""
echo -e "${YELLOW}Check the accuracy above.${NC}"
echo "If < 100%, increase clauses and retry:"
echo "  python3 scripts/2_train_model.py --board-size 5 --stage 0 --epochs 150 --clauses 3000 --depth 6 --T 1500 --s 0.01"
echo ""
read -p "Press Enter to continue to Stage -2..."

# Train Stage -2
echo ""
echo "============================================================"
echo "TRAINING STAGE -2 (2 BEFORE END)"
echo "============================================================"
python3 scripts/2_train_model.py \
    --board-size 5 \
    --stage -2 \
    --epochs 200 \
    --clauses 3000 \
    --depth 6 \
    --T 1500 \
    --s 0.01

echo ""
echo -e "${YELLOW}Check the accuracy above.${NC}"
echo "If < 100%, increase clauses and retry:"
echo "  python3 scripts/2_train_model.py --board-size 5 --stage -2 --epochs 200 --clauses 4000 --depth 6 --T 1500 --s 0.01"
echo ""
read -p "Press Enter to continue to Stage -5 (hardest)..."

# Train Stage -5
echo ""
echo "============================================================"
echo "TRAINING STAGE -5 (5 BEFORE END - HARDEST)"
echo "============================================================"
python3 scripts/2_train_model.py \
    --board-size 5 \
    --stage -5 \
    --epochs 200 \
    --clauses 5000 \
    --depth 6 \
    --T 1500 \
    --s 0.01

echo ""
echo "============================================================"
echo "TRAINING COMPLETE!"
echo "============================================================"

