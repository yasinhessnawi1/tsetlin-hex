#!/bin/bash
# GTM Optimization Experiments - Phase 1
# Based on GTM_Optimization_Guide.md
# Linux version for 10x10 boards

echo "============================================================"
echo "GTM OPTIMIZATION EXPERIMENTS - PHASE 1"
echo "Finding Minimal Clause Configuration"
echo "============================================================"
echo ""
echo "This script runs systematic experiments to find the optimal"
echo "GTM configuration with minimal clauses while maintaining"
echo "high accuracy for Hex winner prediction."
echo ""
echo "Phase 1 Experiments:"
echo "  1.1: Minimum Clauses (100, 200, 300, 400, 500)"
echo "  1.2: Specificity s (5, 10, 15, 20, 25)"
echo "  1.3: Threshold T (5000, 10000, 15000, 20000)"
echo "  1.4: Message Depth (2, 3, 4, 5, 6)"
echo ""
echo "Board: 10x10"
echo "Stage: end"
echo "Epochs: 100 per experiment"
echo "Total experiments: ~19"
echo "Estimated time: 4-8 hours"
echo ""
read -p "Press Enter to continue..."

# Set CUDA environment (for Linux)
if [ -d "/usr/local/cuda" ]; then
    export CUDA_PATH=/usr/local/cuda
    export CUDA_HOME=$CUDA_PATH
    export PATH=$CUDA_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

echo "[INFO] CUDA environment configured"
if [ -n "$CUDA_PATH" ]; then
    echo "  CUDA: $CUDA_PATH"
fi
echo "  Device: 0"
echo ""

# Create results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="experiments/phase1_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

echo "============================================================"
echo "PHASE 1 - EXPERIMENT 1.1: Minimum Clauses Baseline"
echo "Fixed: s=10.0, T=15000, depth=3"
echo "Varying: clauses = [100, 200, 300, 400, 500]"
echo "============================================================"
echo ""

echo "[1/5] Testing 100 clauses..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses 100 --depth 3 --s 10.0 --T 15000 > "${RESULTS_DIR}/exp1_1_clauses_100.log" 2>&1
echo "  Complete. Log: ${RESULTS_DIR}/exp1_1_clauses_100.log"
echo ""

echo "[2/5] Testing 200 clauses..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses 200 --depth 3 --s 10.0 --T 15000 > "${RESULTS_DIR}/exp1_1_clauses_200.log" 2>&1
echo "  Complete. Log: ${RESULTS_DIR}/exp1_1_clauses_200.log"
echo ""

echo "[3/5] Testing 300 clauses..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses 300 --depth 3 --s 10.0 --T 15000 > "${RESULTS_DIR}/exp1_1_clauses_300.log" 2>&1
echo "  Complete. Log: ${RESULTS_DIR}/exp1_1_clauses_300.log"
echo ""

echo "[4/5] Testing 400 clauses..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses 400 --depth 3 --s 10.0 --T 15000 > "${RESULTS_DIR}/exp1_1_clauses_400.log" 2>&1
echo "  Complete. Log: ${RESULTS_DIR}/exp1_1_clauses_400.log"
echo ""

echo "[5/5] Testing 500 clauses..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses 500 --depth 3 --s 10.0 --T 15000 > "${RESULTS_DIR}/exp1_1_clauses_500.log" 2>&1
echo "  Complete. Log: ${RESULTS_DIR}/exp1_1_clauses_500.log"
echo ""

echo "============================================================"
echo "EXPERIMENT 1.1 COMPLETE!"
echo "============================================================"
echo ""
echo "Review the logs in: $RESULTS_DIR"
echo "Look for \"Test Accuracy\" in each log file to compare results."
echo ""
echo "Next: Determine optimal clause count, then run Experiment 1.2"
echo ""
echo "Commands to extract results:"
echo "  grep \"Test Accuracy\" ${RESULTS_DIR}/exp1_1_*.log"
echo ""
read -p "Press Enter to continue..."

echo "============================================================"
echo "PHASE 1 - EXPERIMENT 1.2: Optimize Specificity"
echo "Using optimal_clauses from Experiment 1.1"
echo "Fixed: clauses=[optimal], T=15000, depth=3"
echo "Varying: s = [5.0, 10.0, 15.0, 20.0, 25.0]"
echo "============================================================"
echo ""
echo -n "Enter optimal clause count from Experiment 1.1 (e.g., 200): "
read OPTIMAL_CLAUSES
echo ""
echo "Using $OPTIMAL_CLAUSES clauses for specificity tests..."
echo ""

echo "[1/5] Testing s=5.0..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses $OPTIMAL_CLAUSES --depth 3 --s 5.0 --T 15000 > "${RESULTS_DIR}/exp1_2_s_5.0.log" 2>&1
echo "  Complete. Log: ${RESULTS_DIR}/exp1_2_s_5.0.log"
echo ""

echo "[2/5] Testing s=10.0..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses $OPTIMAL_CLAUSES --depth 3 --s 10.0 --T 15000 > "${RESULTS_DIR}/exp1_2_s_10.0.log" 2>&1
echo "  Complete. Log: ${RESULTS_DIR}/exp1_2_s_10.0.log"
echo ""

echo "[3/5] Testing s=15.0..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses $OPTIMAL_CLAUSES --depth 3 --s 15.0 --T 15000 > "${RESULTS_DIR}/exp1_2_s_15.0.log" 2>&1
echo "  Complete. Log: ${RESULTS_DIR}/exp1_2_s_15.0.log"
echo ""

echo "[4/5] Testing s=20.0..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses $OPTIMAL_CLAUSES --depth 3 --s 20.0 --T 15000 > "${RESULTS_DIR}/exp1_2_s_20.0.log" 2>&1
echo "  Complete. Log: ${RESULTS_DIR}/exp1_2_s_20.0.log"
echo ""

echo "[5/5] Testing s=25.0..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses $OPTIMAL_CLAUSES --depth 3 --s 25.0 --T 15000 > "${RESULTS_DIR}/exp1_2_s_25.0.log" 2>&1
echo "  Complete. Log: ${RESULTS_DIR}/exp1_2_s_25.0.log"
echo ""

echo "============================================================"
echo "EXPERIMENT 1.2 COMPLETE!"
echo "============================================================"
echo ""
grep "Test Accuracy" ${RESULTS_DIR}/exp1_2_*.log
echo ""
read -p "Press Enter to continue..."

echo "============================================================"
echo "PHASE 1 - EXPERIMENT 1.3: Optimize Threshold T"
echo "Using optimal parameters from Experiments 1.1 and 1.2"
echo "============================================================"
echo ""
echo -n "Enter optimal s value from Experiment 1.2 (e.g., 10.0): "
read OPTIMAL_S
echo ""
echo "Using clauses=$OPTIMAL_CLAUSES, s=$OPTIMAL_S"
echo "Varying: T = [5000, 10000, 15000, 20000]"
echo ""

echo "[1/4] Testing T=5000..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses $OPTIMAL_CLAUSES --depth 3 --s $OPTIMAL_S --T 5000 > "${RESULTS_DIR}/exp1_3_T_5000.log" 2>&1
echo "  Complete."
echo ""

echo "[2/4] Testing T=10000..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses $OPTIMAL_CLAUSES --depth 3 --s $OPTIMAL_S --T 10000 > "${RESULTS_DIR}/exp1_3_T_10000.log" 2>&1
echo "  Complete."
echo ""

echo "[3/4] Testing T=15000..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses $OPTIMAL_CLAUSES --depth 3 --s $OPTIMAL_S --T 15000 > "${RESULTS_DIR}/exp1_3_T_15000.log" 2>&1
echo "  Complete."
echo ""

echo "[4/4] Testing T=20000..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses $OPTIMAL_CLAUSES --depth 3 --s $OPTIMAL_S --T 20000 > "${RESULTS_DIR}/exp1_3_T_20000.log" 2>&1
echo "  Complete."
echo ""

echo "============================================================"
echo "EXPERIMENT 1.3 COMPLETE!"
echo "============================================================"
echo ""
grep "Test Accuracy" ${RESULTS_DIR}/exp1_3_*.log
echo ""
read -p "Press Enter to continue..."

echo "============================================================"
echo "PHASE 1 - EXPERIMENT 1.4: Optimize Depth"
echo "Using optimal parameters from all previous experiments"
echo "============================================================"
echo ""
echo -n "Enter optimal T value from Experiment 1.3 (e.g., 15000): "
read OPTIMAL_T
echo ""
echo "Using clauses=$OPTIMAL_CLAUSES, s=$OPTIMAL_S, T=$OPTIMAL_T"
echo "Varying: depth = [2, 3, 4, 5, 6]"
echo ""

echo "[1/5] Testing depth=2..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses $OPTIMAL_CLAUSES --depth 2 --s $OPTIMAL_S --T $OPTIMAL_T > "${RESULTS_DIR}/exp1_4_depth_2.log" 2>&1
echo "  Complete."
echo ""

echo "[2/5] Testing depth=3..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses $OPTIMAL_CLAUSES --depth 3 --s $OPTIMAL_S --T $OPTIMAL_T > "${RESULTS_DIR}/exp1_4_depth_3.log" 2>&1
echo "  Complete."
echo ""

echo "[3/5] Testing depth=4..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses $OPTIMAL_CLAUSES --depth 4 --s $OPTIMAL_S --T $OPTIMAL_T > "${RESULTS_DIR}/exp1_4_depth_4.log" 2>&1
echo "  Complete."
echo ""

echo "[4/5] Testing depth=5..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses $OPTIMAL_CLAUSES --depth 5 --s $OPTIMAL_S --T $OPTIMAL_T > "${RESULTS_DIR}/exp1_4_depth_5.log" 2>&1
echo "  Complete."
echo ""

echo "[5/5] Testing depth=6..."
python3 scripts/2_train_model.py --board-size 10 --stage end --epochs 100 --clauses $OPTIMAL_CLAUSES --depth 6 --s $OPTIMAL_S --T $OPTIMAL_T > "${RESULTS_DIR}/exp1_4_depth_6.log" 2>&1
echo "  Complete."
echo ""

echo "============================================================"
echo "EXPERIMENT 1.4 COMPLETE!"
echo "============================================================"
echo ""
grep "Test Accuracy" ${RESULTS_DIR}/exp1_4_*.log
echo ""

echo "============================================================"
echo "PHASE 1 COMPLETE - ALL EXPERIMENTS DONE!"
echo "============================================================"
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""
echo "OPTIMAL CONFIGURATION FOUND:"
echo "  Clauses: $OPTIMAL_CLAUSES"
echo "  Specificity (s): $OPTIMAL_S"
echo "  Threshold (T): $OPTIMAL_T"
echo "  Depth: [Check logs above]"
echo ""
echo "Next steps:"
echo "  1. Review all results in $RESULTS_DIR"
echo "  2. Document the Pareto-optimal configurations"
echo "  3. Move to Phase 2: Ensemble approaches (TM Composites)"
echo "  4. Consider advanced techniques (CoTM, weighted clauses, HVTM)"
echo ""
echo "To analyze all results:"
echo "  python3 scripts/analyze_experiments.py --results-dir $RESULTS_DIR"
echo ""




