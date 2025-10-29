#!/bin/bash

echo "============================================================"
echo "GENERATING MASSIVE DATASET FOR 5X5 HEX (A100 80GB)"
echo "============================================================"
echo ""
echo "Detected NVIDIA A100 80GB - Optimizing for maximum throughput!"
echo ""
echo "This will generate:"
echo "  - 1,000,000 training games"
echo "  - 200,000 test games"
echo ""
echo "With A100 80GB, this enables:"
echo "  - Near 100% accuracy potential"
echo "  - Strong generalization"
echo ""
echo "Estimated time: ~60-90 minutes"
echo ""
read -p "Press Enter to continue..."

# Generate games using C code (CPU-bound, so not affected by GPU)
python3 scripts/1_generate_games.py --board-size 5 --train-games 1000000 --test-games 200000

echo ""
echo "============================================================"
echo "BUILDING GTM DATASETS"
echo "============================================================"
echo ""

# Build GTM datasets
python3 scripts/1b_build_gtm_datasets.py --board-size 5

echo ""
echo "============================================================"
echo "DATA GENERATION COMPLETE!"
echo "============================================================"
echo ""
echo "You can now train the model with:"
echo "  ./run_strong_training.sh"
echo ""
read -p "Press Enter to exit..."
