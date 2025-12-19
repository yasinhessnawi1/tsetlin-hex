#!/bin/bash

echo "============================================================"
echo "TRAINING GTM ON A100 32GB"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  - Board: 5x5"
echo "  - Clauses: 200 (optimized for A100)"
echo "  - Depth: 6"
echo "  - Epochs: 100 (leverage massive data)"
echo "  - T: 500, s: 5.0"
echo ""
echo "With 1M training samples, expect 90-95% accuracy!"
echo ""

# Set CUDA device to use the MIG instance
# MIG device 0 (GPU 0, GI 6, CI 0) with 14 SMs and ~20GB memory
export CUDA_VISIBLE_DEVICES=0

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

# Train with A100-optimized settings
python3 scripts/2_train_model.py \
    --board-size 7 \
    --stage -2,-5 \
    --epochs 1 \
    --clauses 200 \
    --depth 3
    --T 200
    --s 1.0
    --message-size 512
    --message-bits 4

echo ""
echo "============================================================"
echo "TRAINING COMPLETE!"
echo "============================================================"
echo ""
read -p "Press Enter to exit..."
