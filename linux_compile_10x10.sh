#!/bin/bash
#
# Linux Full Pipeline Script for A100 GPU Server
# Compiles C code, generates data, trains GTM model, and evaluates
#
# Usage: bash linux_compile_10x10.sh
#

set -e  # Exit on error

echo "============================================================"
echo "GTM HEX TRAINING - FULL PIPELINE FOR A100 (Linux)"
echo "============================================================"

# ============================================================
# STEP 1: SYSTEM SETUP
# ============================================================
echo ""
echo "Step 1/6: Installing system dependencies..."
echo "============================================================"

# Update package lists
sudo apt-get update

# Install build essentials and Python
sudo apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    cuda-toolkit-12-8

echo "✓ System dependencies installed"

# ============================================================
# STEP 2: PYTHON ENVIRONMENT SETUP
# ============================================================
echo ""
echo "Step 2/6: Setting up Python virtual environment..."
echo "============================================================"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3.10 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

echo "✓ Python environment ready"

# ============================================================
# STEP 3: COMPILE C CODE FOR DATA GENERATION
# ============================================================
echo ""
echo "Step 3/6: Compiling C data generator for 10x10 board..."
echo "============================================================"

# Compile hex_datagen_stages.c for 10x10 board
gcc -O3 -DBOARD_DIM=10 hex_datagen_stages.c -o hex_datagen_10x10 -lm

if [ -f "hex_datagen_10x10" ]; then
    chmod +x hex_datagen_10x10
    echo "✓ hex_datagen_10x10 compiled successfully"
else
    echo "✗ ERROR: Compilation failed!"
    exit 1
fi

# ============================================================
# STEP 4: GENERATE TRAINING DATA
# ============================================================
echo ""
echo "Step 4/6: Generating 1M training + 200k test games..."
echo "============================================================"

python scripts/1_generate_games.py \
    --board-size 10 \
    --num-train 1000000 \
    --num-test 200000 \
    --save-states 0

if [ $? -eq 0 ]; then
    echo "✓ Game data generated successfully"
else
    echo "✗ ERROR: Data generation failed!"
    exit 1
fi

# ============================================================
# STEP 5: BUILD GTM DATASETS
# ============================================================
echo ""
echo "Step 5/6: Building GTM datasets with binary encoding..."
echo "============================================================"

python scripts/1b_build_gtm_datasets.py \
    --board-size 10 \
    --hypervector-size 256 \
    --hypervector-bits 4 \
    --stages end

if [ $? -eq 0 ]; then
    echo "✓ GTM datasets built successfully"
else
    echo "✗ ERROR: Dataset building failed!"
    exit 1
fi

# ============================================================
# STEP 6: TRAIN MODEL ON A100
# ============================================================
echo ""
echo "Step 6/6: Training GTM model on A100 GPU..."
echo "============================================================"

# A100 optimized hyperparameters
python scripts/2_train_model.py \
    --board-size 10 \
    --stage end \
    --epochs 100 \
    --T 8000 \
    --clauses 10000 \
    --s 100 \
    --depth 6

if [ $? -eq 0 ]; then
    echo "✓ Model training completed"
else
    echo "✗ ERROR: Training failed!"
    exit 1
fi

# ============================================================
# STEP 7: EVALUATE MODEL
# ============================================================
echo ""
echo "Step 7/7: Evaluating trained model..."
echo "============================================================"

python scripts/3_evaluate.py \
    --board-size 10 \
    --stage end \
    --latest

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo ""
echo "Results saved in: models/training_runs/"
echo "Check the latest folder for:"
echo "  - training_history.json (metrics)"
echo "  - rules.txt (learned patterns)"
echo "  - messages.txt (message passing)"
echo "  - model.pkl (trained model)"
echo ""
echo "To view results:"
echo "  cat models/training_runs/\$(ls -t models/training_runs/ | head -1)/summary.txt"
echo ""
