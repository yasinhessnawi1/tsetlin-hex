#!/bin/bash
# Phase 2: TM Composites (Ensemble Strategy)

echo "============================================================"
echo "PHASE 2: TM COMPOSITE EXPERIMENTS"
echo "Ensemble of Specialized Models (BALANCED DATA)"
echo "============================================================"
echo ""
echo "This tests ensemble approaches on BALANCED 50/50 data where"
echo "multiple specialized models work together."
echo ""
echo "Data: BALANCED via undersampling (50% P0, 50% P1)"
echo "Parameters: 200 clauses, T=10000, s=10.0, depth=3, epochs=200"
echo ""
echo "Experiments:"
echo "  2.1: Depth-diverse composite (4 specialists: depth 1,2,3,4)"
echo "  2.2: Specificity-diverse composite (4 specialists: s 5,10,15,20)"
echo "  2.3: Mixed composite (5 specialists: various depth+s combinations)"
echo ""
echo "Target: Beat balanced baseline of 71.40%"
echo "Board: 5x5, Stage: 0 (end game)"
echo "Estimated time: 3-4 hours"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0

echo "[INFO] CUDA environment configured"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

# Create results directory
timestamp=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="experiments/phase2_${timestamp}"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

echo "============================================================"
echo "EXPERIMENT 2.1: Depth-Diverse Composite"
echo "============================================================"
echo ""
echo "Using BALANCED data (50/50 P0/P1)"
echo "Composite: 4 specialists x 50 clauses = 200 total"
echo "  - Depth 1: Very local patterns"
echo "  - Depth 2: Local patterns"
echo "  - Depth 3: Regional patterns"
echo "  - Depth 4: Long-range patterns"
echo ""
echo "Compare to balanced baseline: 71.40% (P0: 71.00%, P1: 71.80%, Gap: 0.80%)"
echo ""

python3 scripts/4_train_composite.py \
    --board-size 5 \
    --stage 0 \
    --epochs 200 \
    --clauses-per-specialist 50 \
    --composite-type depth \
    --T 10000 \
    --s 10.0 \
    --depth 3 \
    | tee "$RESULTS_DIR/exp2_1_depth_composite.log"

echo ""
echo "[OK] Experiment 2.1 complete"
echo "Log: $RESULTS_DIR/exp2_1_depth_composite.log"
echo ""
read -p "Press Enter to continue to next experiment..."

echo "============================================================"
echo "EXPERIMENT 2.2: Specificity-Diverse Composite"
echo "============================================================"
echo ""
echo "Using BALANCED data (50/50 P0/P1)"
echo "Composite: 4 specialists x 50 clauses = 200 total"
echo "  - s=5: Coarse patterns (general)"
echo "  - s=10: Medium specificity (balanced)"
echo "  - s=15: Fine patterns (specific)"
echo "  - s=20: Very fine patterns (very specific)"
echo ""
echo "Compare to balanced baseline: 71.40% (P0: 71.00%, P1: 71.80%, Gap: 0.80%)"
echo ""

python3 scripts/4_train_composite.py \
    --board-size 5 \
    --stage 0 \
    --epochs 200 \
    --clauses-per-specialist 50 \
    --composite-type specificity \
    --T 10000 \
    --s 10.0 \
    --depth 3 \
    | tee "$RESULTS_DIR/exp2_2_specificity_composite.log"

echo ""
echo "[OK] Experiment 2.2 complete"
echo "Log: $RESULTS_DIR/exp2_2_specificity_composite.log"
echo ""
read -p "Press Enter to continue to next experiment..."

echo "============================================================"
echo "EXPERIMENT 2.3: Mixed Composite (Best of Both)"
echo "============================================================"
echo ""
echo "Using BALANCED data (50/50 P0/P1)"
echo "Composite: 5 specialists x 40 clauses = 200 total"
echo "  - Depth 2 + s=5: Shallow + general"
echo "  - Depth 2 + s=10: Shallow + balanced"
echo "  - Depth 3 + s=10: Medium + balanced"
echo "  - Depth 3 + s=15: Medium + specific"
echo "  - Depth 4 + s=10: Deep + balanced"
echo ""
echo "Compare to balanced baseline: 71.40% (P0: 71.00%, P1: 71.80%, Gap: 0.80%)"
echo ""

python3 scripts/4_train_composite.py \
    --board-size 5 \
    --stage 0 \
    --epochs 200 \
    --clauses-per-specialist 40 \
    --composite-type mixed \
    --T 10000 \
    --s 10.0 \
    --depth 3 \
    | tee "$RESULTS_DIR/exp2_3_mixed_composite.log"

echo ""
echo "[OK] Experiment 2.3 complete"
echo "Log: $RESULTS_DIR/exp2_3_mixed_composite.log"
echo ""

echo "============================================================"
echo "PHASE 2 COMPLETE!"
echo "============================================================"
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""
echo "To view results:"
echo "  grep -E 'Test Accuracy|Clause reduction' $RESULTS_DIR/*.log"
echo ""
echo "Next: If composites show improvement, proceed to HVTM experiment"
echo ""
read -p "Press Enter to exit..."

