@echo off
REM Train Weighted GTM for class imbalance handling
echo ============================================================
echo WEIGHTED GTM TRAINING (CLASS BALANCING)
echo ============================================================
echo.
echo This trains a GTM with automatic class balancing to reduce
echo the accuracy gap between Player 0 and Player 1 predictions.
echo.
echo Your baseline results (5x5, 200 epochs):
echo   Player 0: 83.46%% accuracy
echo   Player 1: 62.06%% accuracy
echo   Gap: 21.40%%
echo.
echo Expected improvement with class balancing:
echo   Player 0: ~75-80%% (slightly lower)
echo   Player 1: ~70-75%% (MUCH better!)
echo   Gap: ~5-10%% (much smaller)
echo.
echo Board: 5x5
echo Epochs: 200
echo Estimated time: ~45 minutes
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

python scripts\train_weighted_gtm.py

echo.
echo ============================================================
echo TRAINING COMPLETE
echo ============================================================
pause
