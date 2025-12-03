#!/bin/bash
# Phase 2: TM Composites (Ensemble Strategy)

echo "============================================================"
echo "PHASE 2: TM COMPOSITE EXPERIMENTS"
echo "Ensemble of Specialized Models"
echo "============================================================"
echo ""
echo "This tests ensemble approaches where multiple specialized"
echo "models work together, potentially achieving better accuracy"
echo "with fewer total clauses."
echo ""
echo "Experiments:"
echo "  2.1: Depth-diverse composite (4 specialists: depth 2,3,4,5)"
echo "  2.2: Specificity-diverse composite (4 specialists: s 5,10,15,20)"
echo "  2.3: Mixed composite (5 specialists: various depth+s)"
echo ""
echo "Each experiment compares:"
echo "  - Baseline: Single model with N clauses"
echo "  - Composite: Multiple specialists with N/4 clauses each"
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
RESULTS_DIR="experiments/phase2_${timestamp}"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

echo "============================================================"
echo "EXPERIMENT 2.1: Depth-Diverse Composite"
echo "============================================================"
echo ""
echo "Baseline: 200 clauses, depth=3"
echo "Composite: 4 specialists x 50 clauses = 200 total"
echo "  - Shallow (depth=2): Local patterns"
echo "  - Medium (depth=3): Regional patterns"
echo "  - Deep (depth=4): Long-range patterns"
echo "  - Very Deep (depth=5): Global connectivity"
echo ""

python3 scripts/test_tm_composite.py \
    --board-size 5 \
    --stage end \
    --epochs 100 \
    --baseline-clauses 200 \
    --composite-clauses 50 \
    --composite-type depth \
    > "$RESULTS_DIR/exp2_1_depth_composite.log" 2>&1

echo ""
echo "[OK] Experiment 2.1 complete"
echo "Log: $RESULTS_DIR/exp2_1_depth_composite.log"
echo ""
read -p "Press Enter to continue to next experiment..."

echo "============================================================"
echo "EXPERIMENT 2.2: Specificity-Diverse Composite"
echo "============================================================"
echo ""
echo "Baseline: 200 clauses, s=10.0"
echo "Composite: 4 specialists x 50 clauses = 200 total"
echo "  - Coarse (s=5): General patterns"
echo "  - Medium (s=10): Balanced"
echo "  - Fine (s=15): Specific patterns"
echo "  - Very Fine (s=20): Very specific patterns"
echo ""

python3 scripts/test_tm_composite.py \
    --board-size 5 \
    --stage end \
    --epochs 100 \
    --baseline-clauses 200 \
    --composite-clauses 50 \
    --composite-type specificity \
    > "$RESULTS_DIR/exp2_2_specificity_composite.log" 2>&1

echo ""
echo "[OK] Experiment 2.2 complete"
echo "Log: $RESULTS_DIR/exp2_2_specificity_composite.log"
echo ""
read -p "Press Enter to continue to next experiment..."

echo "============================================================"
echo "EXPERIMENT 2.3: Mixed Composite (Best of Both)"
echo "============================================================"
echo ""
echo "Baseline: 200 clauses"
echo "Composite: 5 specialists x 40 clauses = 200 total"
echo "  - Shallow + general"
echo "  - Medium + balanced"
echo "  - Deep + balanced"
echo "  - Medium + specific"
echo "  - Deep + specific"
echo ""

python3 scripts/test_tm_composite.py \
    --board-size 5 \
    --stage end \
    --epochs 100 \
    --baseline-clauses 200 \
    --composite-clauses 40 \
    --composite-type mixed \
    > "$RESULTS_DIR/exp2_3_mixed_composite.log" 2>&1

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
echo "Next: If composites show improvement, proceed to Phase 3"
echo "      (CoTM, Weighted Clauses, HVTM, etc.)"
echo ""
read -p "Press Enter to exit..."

