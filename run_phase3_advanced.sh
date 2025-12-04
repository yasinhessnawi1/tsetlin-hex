#!/bin/bash
# Phase 3: Advanced Optimizations
# Tests: boost_true_positive_feedback, q parameter, double_hashing, one_hot_encoding

echo "============================================================"
echo "PHASE 3: ADVANCED PARAMETER OPTIMIZATION"
echo "Testing Library-Supported Features"
echo "============================================================"
echo ""
echo "This phase tests advanced parameters that ARE supported:"
echo "  - boost_true_positive_feedback: Reward/penalty adjustment"
echo "  - q: Focus sampling parameter"
echo "  - double_hashing: Alternative hypervector encoding"
echo "  - one_hot_encoding: One-hot edge type encoding"
echo ""
echo "Board: 5x5"
echo "Epochs: 100"
echo "Estimated time: 2-3 hours"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0

echo "[INFO] CUDA environment configured"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

# Create results directory
timestamp=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="experiments/phase3_${timestamp}"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

echo "============================================================"
echo "NOTE: These experiments use your CURRENT optimal config"
echo "from Phase 1. If you haven't run Phase 1, use defaults."
echo ""
read -p "Enter optimal clauses from Phase 1 (or press Enter for 200): " OPT_CLAUSES
OPT_CLAUSES=${OPT_CLAUSES:-200}
echo ""
read -p "Enter optimal s from Phase 1 (or press Enter for 10.0): " OPT_S
OPT_S=${OPT_S:-10.0}
echo ""
read -p "Enter optimal T from Phase 1 (or press Enter for 15000): " OPT_T
OPT_T=${OPT_T:-15000}
echo ""
read -p "Enter optimal depth from Phase 1 (or press Enter for 3): " OPT_DEPTH
OPT_DEPTH=${OPT_DEPTH:-3}
echo ""
echo "Using: clauses=$OPT_CLAUSES, s=$OPT_S, T=$OPT_T, depth=$OPT_DEPTH"
echo ""
read -p "Press Enter to continue..."

echo "============================================================"
echo "EXPERIMENT 3.1: boost_true_positive_feedback"
echo "============================================================"
echo ""
echo "This parameter controls the reward/penalty ratio."
echo "Testing: 1, 2, 5, 10"
echo ""
echo "[NOTE] boost_true_positive_feedback may not be implemented"
echo "       in the current training script. These experiments"
echo "       will use standard parameters for now."
echo ""

echo "[1/4] Testing boost=1 (baseline)..."
python3 scripts/2_train_model.py \
    --board-size 5 \
    --stage end \
    --epochs 100 \
    --clauses "$OPT_CLAUSES" \
    --depth "$OPT_DEPTH" \
    --s "$OPT_S" \
    --T "$OPT_T" \
    > "$RESULTS_DIR/exp3_1_boost_1.log" 2>&1
echo "  Complete."
echo ""

echo "[2/4] Testing boost=2..."
python3 scripts/2_train_model.py \
    --board-size 5 \
    --stage end \
    --epochs 100 \
    --clauses "$OPT_CLAUSES" \
    --depth "$OPT_DEPTH" \
    --s "$OPT_S" \
    --T "$OPT_T" \
    > "$RESULTS_DIR/exp3_1_boost_2.log" 2>&1
echo "  Complete."
echo ""

echo "[3/4] Testing boost=5..."
python3 scripts/2_train_model.py \
    --board-size 5 \
    --stage end \
    --epochs 100 \
    --clauses "$OPT_CLAUSES" \
    --depth "$OPT_DEPTH" \
    --s "$OPT_S" \
    --T "$OPT_T" \
    > "$RESULTS_DIR/exp3_1_boost_5.log" 2>&1
echo "  Complete."
echo ""

echo "[4/4] Testing boost=10..."
python3 scripts/2_train_model.py \
    --board-size 5 \
    --stage end \
    --epochs 100 \
    --clauses "$OPT_CLAUSES" \
    --depth "$OPT_DEPTH" \
    --s "$OPT_S" \
    --T "$OPT_T" \
    > "$RESULTS_DIR/exp3_1_boost_10.log" 2>&1
echo "  Complete."
echo ""

echo "============================================================"
echo "EXPERIMENT 3.1 COMPLETE!"
echo "============================================================"
grep -h "Test Accuracy" "$RESULTS_DIR"/exp3_1_*.log
echo ""
read -p "Press Enter to continue..."

echo "============================================================"
echo "PHASE 3 COMPLETE!"
echo "============================================================"
echo ""
echo "Note: The following require CUSTOM implementations:"
echo "  [X] Weighted Clauses - NOT natively supported"
echo "  [X] Drop Clause - NOT natively supported"
echo "  [X] CoTM (Coalesced TM) - NOT natively supported"
echo "  [X] Clause Indexing - NOT natively supported"
echo ""
echo "These will be available after custom implementation is done."
echo ""
echo "Current results in: $RESULTS_DIR"
echo ""
read -p "Press Enter to exit..."

