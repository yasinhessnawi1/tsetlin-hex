#!/bin/bash

echo "============================================================"
echo "TRAINING GTM ON A100 80GB"
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

# Train with A100-optimized settings
python3 scripts/2_train_model.py \
    --board-size 5 \
    --stage end \
    --epochs 100 \
    --clauses 200 \
    --depth 6

echo ""
echo "============================================================"
echo "TRAINING COMPLETE!"
echo "============================================================"
echo ""
read -p "Press Enter to exit..."
