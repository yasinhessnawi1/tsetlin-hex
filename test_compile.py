#!/usr/bin/env python3
"""Test script to verify hex_datagen compilation works."""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

HEX_BIN_DIR = Path(__file__).parent / "hex_binaries"

def compile_hex_datagen(board_size: int):
    """Compile hex_datagen_stages for an arbitrary board size."""
    if os.name == "nt":
        # Windows: use cl with proper MSVC environment setup
        import subprocess

        # Set up MSVC environment by calling vcvars64.bat
        vs_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community"
        vcvars_cmd = f'call "{vs_path}\\VC\\Auxiliary\\Build\\vcvars64.bat" && cl /O2 /DBOARD_DIM={board_size} hex_datagen_stages.c /Fe:hex_datagen_{board_size}x{board_size}.exe'

        try:
            subprocess.run(vcvars_cmd, shell=True, check=True, cwd=HEX_BIN_DIR)
            print(f"Successfully compiled hex_datagen_{board_size}x{board_size}.exe")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to compile with MSVC: {e}. Make sure Visual Studio 2022 Community is installed.")
            return False
    else:
        gcc = shutil.which("gcc")
        if not gcc:
            print("ERROR: gcc not found. Install build-essential.")
            return False
        cmd = [
            "gcc",
            "-O3",
            f"-DBOARD_DIM={board_size}",
            "hex_datagen_stages.c",
            f"-o hex_datagen_{board_size}x{board_size}",
        ]
        try:
            subprocess.run(cmd, check=True, cwd=HEX_BIN_DIR)
            print(f"Successfully compiled hex_datagen_{board_size}x{board_size}")
            return True
        except Exception as exc:
            print(f"ERROR: Failed to compile generator: {exc}")
            return False

if __name__ == "__main__":
    # Test compiling a 21x21 board (shouldn't exist)
    test_size = 21
    exe_path = HEX_BIN_DIR / f"hex_datagen_{test_size}x{test_size}.exe"

    if exe_path.exists():
        print(f"Removing existing {exe_path}")
        exe_path.unlink()

    print(f"Testing compilation for {test_size}x{test_size} board...")
    success = compile_hex_datagen(test_size)

    if success and exe_path.exists():
        print("✅ Compilation test PASSED!")
    else:
        print("❌ Compilation test FAILED!")
