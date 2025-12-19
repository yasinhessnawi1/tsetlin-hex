#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Validate/evaluate a saved GTM model using scripts/3_evaluate.py.
# Auto-generates missing test data (games + GTM pkls) if needed.
#
# Usage:
#   ./validate_model.sh [BOARD_SIZE] [STAGE] [DATA_DIR] [MODELS_DIR] [NUM_TRAIN] [NUM_TEST] [GEN_STAGES]
#
# Defaults:
#   BOARD_SIZE: 11
#   STAGE: end
#   DATA_DIR: data
#   MODELS_DIR: models
#   NUM_TRAIN: 10000
#   NUM_TEST: 3000
#   GEN_STAGES: all   (0,-2,-5)
# ------------------------------------------------------------

BOARD_SIZE=${1:-11}
STAGE=${2:-end}
DATA_DIR=${3:-data}
MODELS_DIR=${4:-models}
NUM_TRAIN=${5:-10000}
NUM_TEST=${6:-3000}
GEN_STAGES=${7:-all}

# Optional CUDA selection (mirrors run_training.sh style)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo ""
echo "=============================================="
echo "Validating model via scripts/3_evaluate.py"
echo "  Board      : ${BOARD_SIZE}x${BOARD_SIZE}"
echo "  Stage      : ${STAGE}"
echo "  Data dir   : ${DATA_DIR}"
echo "  Models dir : ${MODELS_DIR}"
echo "  Num train  : ${NUM_TRAIN}"
echo "  Num test   : ${NUM_TEST}"
echo "  Gen stages : ${GEN_STAGES}"
echo "  CUDA device: ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python3 scripts/3_evaluate.py \
  --board-size "${BOARD_SIZE}" \
  --stage "${STAGE}" \
  --data-dir "${DATA_DIR}" \
  --models-dir "${MODELS_DIR}" \
  --num-train "${NUM_TRAIN}" \
  --num-test "${NUM_TEST}" \
  --gen-stages "${GEN_STAGES}" \
  --visualize \
  --viz-dir "evaluation_plots"

STATUS=$?
exit ${STATUS}



