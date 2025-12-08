#!/usr/bin/env bash
set -euo pipefail

# Full 7x7 validation pipeline using Kaggle dataset cholling/game-of-hex.
# Steps:
#   1) Download & unzip Kaggle dataset
#   2) Convert to GTM raw npz (train/test) for board 7x7
#   3) Build GTM graph datasets (pkl)
#   4) Evaluate saved model(s) with scripts/3_evaluate.py
#
# Requirements:
#   - kaggle CLI installed and authenticated
#   - Python deps for conversion/eval already installed
#
# Usage:
#   ./validate_kaggle_7x7.sh [STAGE] [MODELS_DIR] [DATA_DIR] [NUM_TRAIN] [NUM_TEST]
# Defaults:
#   STAGE: end
#   MODELS_DIR: models
#   DATA_DIR: data
#   NUM_TRAIN: 10000  (used only if 3_evaluate auto-gen kicks in; we prebuild below)
#   NUM_TEST: 3000    (used only if 3_evaluate auto-gen kicks in)

BOARD_SIZE=7
STAGE=${1:-end}
MODELS_DIR=${2:-models}
DATA_DIR=${3:-data}
NUM_TRAIN=${4:-10000}
NUM_TEST=${5:-3000}
GEN_STAGES="0"  # only end-state from Kaggle set

echo ""
echo "============================================================"
echo "Kaggle 7x7 Validation Pipeline"
echo "  Board      : ${BOARD_SIZE}x${BOARD_SIZE}"
echo "  Stage      : ${STAGE}"
echo "  Models dir : ${MODELS_DIR}"
echo "  Data dir   : ${DATA_DIR}"
echo "  Gen stages : ${GEN_STAGES}"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ensure_kaggle_cli() {
  if command -v kaggle >/dev/null 2>&1; then
    return 0
  fi
  echo "[INFO] kaggle CLI not found. Attempting to install with pip --user ..."
  if command -v python3 >/dev/null 2>&1; then
    python3 -m pip install --user kaggle >/dev/null 2>&1 || true
  elif command -v python >/dev/null 2>&1; then
    python -m pip install --user kaggle >/dev/null 2>&1 || true
  fi
  if command -v kaggle >/dev/null 2>&1; then
    return 0
  fi
  echo "[ERROR] kaggle CLI still not found. Please install and authenticate first:"
  echo "  pip install --user kaggle"
  echo "  mkdir -p ~/.kaggle && echo '{\"username\":\"<user>\",\"key\":\"<api-key>\"}' > ~/.kaggle/kaggle.json"
  echo "  chmod 600 ~/.kaggle/kaggle.json"
  exit 1
}

# If raw npz already exists, skip download/convert
RAW_TRAIN="${DATA_DIR}/train_games_${BOARD_SIZE}x${BOARD_SIZE}.npz"
RAW_TEST="${DATA_DIR}/test_games_${BOARD_SIZE}x${BOARD_SIZE}.npz"

# 1) Download Kaggle dataset
KAGGLE_DIR="${DATA_DIR}/kaggle_game_of_hex"
mkdir -p "${KAGGLE_DIR}"
if [[ -f "${RAW_TRAIN}" && -f "${RAW_TEST}" ]]; then
  echo "[SKIP] Found existing raw npz files: ${RAW_TRAIN}, ${RAW_TEST}"
else
  ensure_kaggle_cli
  echo "[RUN] Downloading Kaggle dataset to ${KAGGLE_DIR} ..."
  kaggle datasets download -d cholling/game-of-hex -p "${KAGGLE_DIR}" --unzip
fi

# 2) Convert to GTM raw npz (train/test)
echo "[RUN] Converting Kaggle dataset to GTM raw format..."
python3 scripts/convert_kaggle_game_of_hex.py \
  --dataset-dir "${KAGGLE_DIR}" \
  --board-size "${BOARD_SIZE}" \
  --train-output "${DATA_DIR}/train_games_${BOARD_SIZE}x${BOARD_SIZE}.npz" \
  --test-output "${DATA_DIR}/test_games_${BOARD_SIZE}x${BOARD_SIZE}.npz"

# 3) Build GTM graph datasets (pkl)
echo "[RUN] Building GTM datasets..."
python3 scripts/1b_build_gtm_datasets.py \
  --board-size "${BOARD_SIZE}" \
  --stages "${GEN_STAGES}"

# 4) Evaluate using saved model(s)
echo "[RUN] Evaluating..."
python3 scripts/3_evaluate.py \
  --board-size "${BOARD_SIZE}" \
  --stage "${STAGE}" \
  --data-dir "${DATA_DIR}" \
  --models-dir "${MODELS_DIR}" \
  --num-train "${NUM_TRAIN}" \
  --num-test "${NUM_TEST}" \
  --gen-stages "${GEN_STAGES}"

echo ""
echo "============================================================"
echo "Kaggle 7x7 Validation COMPLETE"
echo "============================================================"

