@echo off
REM Train TM Composite Model

echo ============================================================
echo TM COMPOSITE TRAINING
echo ============================================================
echo.
echo This script trains a TM Composite (ensemble) model for
echo Hex winner prediction.
echo.
echo Default configuration:
echo   - Board: 5x5
echo   - Stage: 0 (end game)
echo   - Type: depth (depth-diverse composite)
echo   - Clauses per specialist: 333
echo   - Epochs: 200
echo   - T: 10000
echo   - s: 1.0
echo.
echo You can customize by editing this script or running directly:
echo   python scripts/4_train_composite.py --help
echo.
pause

REM Set CUDA environment
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_HOME=%CUDA_PATH%
set VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community
set MSVC_PATH=%VS_PATH%\VC\Tools\MSVC\14.39.33519
set PATH=%MSVC_PATH%\bin\Hostx64\x64;%CUDA_PATH%\bin;%PATH%
set CUDA_VISIBLE_DEVICES=0

echo [INFO] CUDA environment configured
echo.

REM Run training
python scripts/4_train_composite.py --board-size 5 --stage all --composite-type mixed --clauses-per-specialist 400 --epochs 200 --T 8000

echo.
echo ============================================================
echo TRAINING COMPLETE!
echo ============================================================
echo.
pause
