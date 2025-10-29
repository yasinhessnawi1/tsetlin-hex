#!/bin/bash

echo "============================================================"
echo "TRAINING GTM ON V100 32GB - OPTIMIZED"
echo "============================================================"
echo ""
echo "Optimizations:"
echo "  - Increased clauses: 500 (use more GPU)"
echo "  - Larger grid: 1024 blocks (80 SMs × 12)"
echo "  - Message size: 512 (more capacity)"
echo "  - Using all 32GB memory"
echo ""

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

# Train with V100-optimized settings
# V100 has 80 SMs, each can handle multiple blocks
# Grid size = 80 SMs × 12-16 blocks per SM = ~1000-1280 blocks
python3 scripts/train_v100_optimized.py \
    --board-size 5 \
    --stage end \
    --epochs 100 \
    --clauses 500 \
    --depth 6 \
    --grid-x 1024 \
    --message-size 512

echo ""
echo "============================================================"
echo "TRAINING COMPLETE!"
echo "============================================================"
echo ""

