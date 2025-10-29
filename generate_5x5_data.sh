#!/bin/bash

echo "============================================================"
echo "GENERATING LARGE DATASET FOR 5X5 HEX"
echo "============================================================"
echo ""
echo "This will generate 500,000 training games and 80,000 test games"
echo "for 5x5 Hex boards to achieve near-perfect accuracy."
echo ""
echo "Estimated time: ~30-60 minutes depending on your CPU"
echo ""
read -p "Press Enter to continue..."

# Generate games using C code
python scripts/1_generate_games.py --board-size 5 --num-train 500000 --num-test 80000

echo ""
echo "============================================================"
echo "BUILDING GTM DATASETS"
echo "============================================================"
echo ""

# Build GTM datasets
python scripts/1b_build_gtm_datasets.py --board-size 5

echo ""
echo "============================================================"
echo "DATA GENERATION COMPLETE!"
echo "============================================================"
echo ""
echo "You can now train the model with:"
echo "  ./run_strong_training.sh"
echo ""
read -p "Press Enter to exit..."
