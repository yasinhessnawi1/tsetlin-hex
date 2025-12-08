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
DATA_DIR=${3:-data/kaggle_eval}
NUM_TRAIN=${4:-10000}
NUM_TEST=${5:-3000}
# Force Kaggle to use only end-state (stage 0)
GEN_STAGES="0"

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
  # Ensure ~/.local/bin is in PATH (common location for pip --user)
  if [ -d "$HOME/.local/bin" ] && [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  fi

  if command -v kaggle >/dev/null 2>&1; then
    return 0
  fi

  # Fallback: try running via python -m kaggle if available
  if command -v python3 >/dev/null 2>&1; then
    if python3 -c "import kaggle" >/dev/null 2>&1; then
      kaggle() { python3 -m kaggle "$@"; }
      export -f kaggle
      return 0
    fi
  elif command -v python >/dev/null 2>&1; then
    if python -c "import kaggle" >/dev/null 2>&1; then
      kaggle() { python -m kaggle "$@"; }
      export -f kaggle
      return 0
    fi
  fi

  echo "[ERROR] kaggle CLI not found. Please install and authenticate first:"
  echo "  pip install --user kaggle"
  echo "  mkdir -p ~/.kaggle && echo '{\"username\":\"<user>\",\"key\":\"<api-key>\"}' > ~/.kaggle/kaggle.json"
  echo "  chmod 600 ~/.kaggle/kaggle.json"
  echo "Also ensure ~/.local/bin is in PATH (export PATH=\$HOME/.local/bin:\$PATH)."
  exit 1
}

# 1) Download Kaggle dataset (skip if raw NPZ already exists)
KAGGLE_DIR="${DATA_DIR}/kaggle_game_of_hex"
RAW_TRAIN="${DATA_DIR}/train_games_${BOARD_SIZE}x${BOARD_SIZE}.npz"
RAW_TEST="${DATA_DIR}/test_games_${BOARD_SIZE}x${BOARD_SIZE}.npz"

if [[ -f "${RAW_TRAIN}" && -f "${RAW_TEST}" ]]; then
  echo "[SKIP] Found raw NPZ files: ${RAW_TRAIN}, ${RAW_TEST}"
else
  mkdir -p "${KAGGLE_DIR}"
  ensure_kaggle_cli
  echo "[RUN] Downloading Kaggle dataset to ${KAGGLE_DIR} ..."
  kaggle datasets download -d cholling/game-of-hex -p "${KAGGLE_DIR}" --unzip
fi

# 2) Convert to GTM raw npz (train/test)
if [[ -f "${RAW_TRAIN}" && -f "${RAW_TEST}" ]]; then
  echo "[SKIP] Raw NPZ already present; skipping conversion."
else
  echo "[RUN] Converting Kaggle dataset to GTM raw format..."
  python3 scripts/convert_kaggle_game_of_hex.py \
    --dataset-dir "${KAGGLE_DIR}" \
    --board-size "${BOARD_SIZE}" \
    --train-output "${RAW_TRAIN}" \
    --test-output "${RAW_TEST}" \
    --all-to-test
fi

# 3) Build GTM graph datasets (single full set; always rebuild to ensure full data)
FULL_GTM="${DATA_DIR}/full_test_gtm_${BOARD_SIZE}x${BOARD_SIZE}_${GEN_STAGES}.pkl"
echo "[RUN] Building GTM datasets (single full set)..."
python3 scripts/1b_build_gtm_datasets.py \
  --board-size "${BOARD_SIZE}" \
  --stages "${GEN_STAGES}" \
  --train-file "${RAW_TRAIN}" \
  --test-file "${RAW_TEST}" \
  --output-dir "${DATA_DIR}" \
  --output-prefix "full_" \
  --single-output

# 4) Evaluate using saved model(s)
echo "[RUN] Evaluating..."
python3 scripts/3_evaluate.py \
  --board-size "${BOARD_SIZE}" \
  --stage "${STAGE}" \
  --data-dir "${DATA_DIR}" \
  --models-dir "${MODELS_DIR}" \
  --num-train "${NUM_TRAIN}" \
  --num-test "${NUM_TEST}" \
  --gen-stages "${GEN_STAGES}" \
  --test-file "${FULL_GTM}"

echo ""
echo "============================================================"
echo "Kaggle 7x7 Validation COMPLETE"
echo "============================================================"

