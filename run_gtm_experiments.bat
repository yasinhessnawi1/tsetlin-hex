@echo off
REM GTM Optimization Experiments - Phase 1 (UPDATED)
REM Uses run_experiments.py for streamlined testing

echo ============================================================
echo GTM OPTIMIZATION EXPERIMENTS - PHASE 1
echo Optimized Clause and Threshold Search
echo ============================================================
echo.
echo This script runs systematic experiments to find the optimal
echo GTM configuration with minimal clauses while maintaining
echo high accuracy for Hex winner prediction.
echo.
echo NEW Phase 1 Strategy:
echo   - Test clause counts: 400, 800, 1200
echo   - Test threshold values: 5000, 10000, 15000, 20000
echo   - Test message sizes: 256, 512
echo   - Test message bits: 2, 4
echo   - Adaptive specificity: s = clauses/100
echo   - Fixed depth: 3 (known best)
echo.
echo Total experiments: 96 (3 clauses x 4 T x 2 msg_size x 2 msg_bits)
echo Board: 5x5 (FAST EXPERIMENTS)
echo Stage: end
echo Epochs: 30 per experiment (fast parameter search)
echo Estimated time: 3-5 hours
echo.
pause

REM Set CUDA environment (required for PyCUDA/nvcc on Windows)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_HOME=%CUDA_PATH%
set VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community
set MSVC_PATH=%VS_PATH%\VC\Tools\MSVC\14.39.33519
set PATH=%MSVC_PATH%\bin\Hostx64\x64;%CUDA_PATH%\bin;%PATH%
set CUDA_VISIBLE_DEVICES=0

echo [INFO] CUDA environment configured
echo   CUDA: %CUDA_PATH%
echo   MSVC: %MSVC_PATH%
echo   Device: 0
echo.

echo ============================================================
echo RUNNING PHASE 1 - EXPERIMENT 1
echo Clause Count and Threshold Optimization
echo ============================================================
echo.

python scripts/run_experiments.py --phase 1 --experiment 1 --board-size 5 --stage end

echo.
echo ============================================================
echo PHASE 1 COMPLETE!
echo ============================================================
echo.
echo Results saved to: experiments/session_[timestamp]/results.json
echo.
echo To view summary:
echo   Check the console output above for best configurations
echo.
echo Next steps:
echo   1. Review experiment summary (printed above)
echo   2. Note the best performing configurations
echo   3. Run Phase 2 (TM Composites) with optimal parameters
echo   4. Consider class balancing if per-class gap is large
echo.
pause
