#!/bin/bash

# Change to hex_binaries directory (where this script is located)
cd "$(dirname "$0")"

echo "============================================================"
echo "COMPILING HEX GAME GENERATOR FOR LINUX"
echo "============================================================"
echo ""

# Check if gcc is available
if ! command -v gcc &> /dev/null; then
    echo "ERROR: gcc not found. Please install:"
    echo "  sudo apt-get install build-essential"
    exit 1
fi

echo "Compiling hex_datagen_stages.c for different board sizes..."
echo ""

# Compile for 5x5
echo "Compiling for 5x5 board..."
gcc -O3 -DBOARD_DIM=5 -o hex_datagen_5x5 hex_datagen_stages.c -lm
if [ $? -eq 0 ]; then
    echo "  ✓ hex_datagen_5x5 compiled successfully"
else
    echo "  ✗ Failed to compile hex_datagen_5x5"
    exit 1
fi

# Compile for 7x7
echo "Compiling for 7x7 board..."
gcc -O3 -DBOARD_DIM=7 -o hex_datagen_7x7 hex_datagen_stages.c -lm
if [ $? -eq 0 ]; then
    echo "  ✓ hex_datagen_7x7 compiled successfully"
else
    echo "  ✗ Failed to compile hex_datagen_7x7"
fi

# Compile for 10x10
echo "Compiling for 10x10 board..."
gcc -O3 -DBOARD_DIM=10 -o hex_datagen_10x10 hex_datagen_stages.c -lm
if [ $? -eq 0 ]; then
    echo "  ✓ hex_datagen_10x10 compiled successfully"
else
    echo "  ✗ Failed to compile hex_datagen_10x10"
fi

# Compile for 11x11
echo "Compiling for 11x11 board..."
gcc -O3 -DBOARD_DIM=11 -o hex_datagen_11x11 hex_datagen_stages.c -lm
if [ $? -eq 0 ]; then
    echo "  ✓ hex_datagen_11x11 compiled successfully"
else
    echo "  ✗ Failed to compile hex_datagen_11x11"
fi

echo ""
echo "============================================================"
echo "COMPILATION COMPLETE!"
echo "============================================================"
echo ""
echo "You can now generate data with:"
echo "  ./generate_5x5_data.sh"
echo ""

# Make executables executable (redundant but safe)
chmod +x hex_datagen_* 2>/dev/null

ls -lh hex_datagen_* 2>/dev/null
