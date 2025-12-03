@echo off
REM GTM Optimization Experiments - Phase 1
REM Based on GTM_Optimization_Guide.md

echo ============================================================
echo GTM OPTIMIZATION EXPERIMENTS - PHASE 1
echo Finding Minimal Clause Configuration
echo ============================================================
echo.
echo This script runs systematic experiments to find the optimal
echo GTM configuration with minimal clauses while maintaining
echo high accuracy for Hex winner prediction.
echo.
echo Phase 1 Experiments:
echo   1.1: Minimum Clauses (100, 200, 300, 400, 500)
echo   1.2: Specificity s (5, 10, 15, 20, 25)
echo   1.3: Threshold T (5000, 10000, 15000, 20000)
echo   1.4: Message Depth (2, 3, 4, 5, 6)
echo.
echo Board: 5x5 (FAST EXPERIMENTS)
echo Stage: end
echo Epochs: 100 per experiment
echo Total experiments: ~19
echo Estimated time: 1-2 hours (5x5 is 10-20x faster!)
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

REM Create results directory with timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set timestamp=%datetime:~0,8%_%datetime:~8,6%
set RESULTS_DIR=experiments\phase1_%timestamp%
mkdir %RESULTS_DIR% 2>nul

echo Results will be saved to: %RESULTS_DIR%
echo.

echo ============================================================
echo PHASE 1 - EXPERIMENT 1.1: Minimum Clauses Baseline
echo Fixed: s=10.0, T=15000, depth=3
echo Varying: clauses = [100, 200, 300, 400, 500]
echo ============================================================
echo.

echo [1/5] Testing 100 clauses...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses 100 --depth 3 --s 10.0 --T 15000 > %RESULTS_DIR%\exp1_1_clauses_100.log 2>&1
echo   Complete. Log: %RESULTS_DIR%\exp1_1_clauses_100.log
echo.

echo [2/5] Testing 200 clauses...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses 200 --depth 3 --s 10.0 --T 15000 > %RESULTS_DIR%\exp1_1_clauses_200.log 2>&1
echo   Complete. Log: %RESULTS_DIR%\exp1_1_clauses_200.log
echo.

echo [3/5] Testing 300 clauses...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses 300 --depth 3 --s 10.0 --T 15000 > %RESULTS_DIR%\exp1_1_clauses_300.log 2>&1
echo   Complete. Log: %RESULTS_DIR%\exp1_1_clauses_300.log
echo.

echo [4/5] Testing 400 clauses...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses 400 --depth 3 --s 10.0 --T 15000 > %RESULTS_DIR%\exp1_1_clauses_400.log 2>&1
echo   Complete. Log: %RESULTS_DIR%\exp1_1_clauses_400.log
echo.

echo [5/5] Testing 500 clauses...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses 500 --depth 3 --s 10.0 --T 15000 > %RESULTS_DIR%\exp1_1_clauses_500.log 2>&1
echo   Complete. Log: %RESULTS_DIR%\exp1_1_clauses_500.log
echo.

echo ============================================================
echo EXPERIMENT 1.1 COMPLETE!
echo ============================================================
echo.
echo Review the logs in: %RESULTS_DIR%
echo Look for "Test Accuracy" in each log file to compare results.
echo.
echo Next: Determine optimal clause count, then run Experiment 1.2
echo.
echo Commands to extract results:
echo   findstr "Test Accuracy" %RESULTS_DIR%\exp1_1_*.log
echo.
pause

echo ============================================================
echo PHASE 1 - EXPERIMENT 1.2: Optimize Specificity
echo Using optimal_clauses from Experiment 1.1
echo Fixed: clauses=[optimal], T=15000, depth=3
echo Varying: s = [5.0, 10.0, 15.0, 20.0, 25.0]
echo ============================================================
echo.
echo Enter optimal clause count from Experiment 1.1 (e.g., 200):
set /p OPTIMAL_CLAUSES=
echo.
echo Using %OPTIMAL_CLAUSES% clauses for specificity tests...
echo.

echo [1/5] Testing s=5.0...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPTIMAL_CLAUSES% --depth 3 --s 5.0 --T 15000 > %RESULTS_DIR%\exp1_2_s_5.0.log 2>&1
echo   Complete. Log: %RESULTS_DIR%\exp1_2_s_5.0.log
echo.

echo [2/5] Testing s=10.0...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPTIMAL_CLAUSES% --depth 3 --s 10.0 --T 15000 > %RESULTS_DIR%\exp1_2_s_10.0.log 2>&1
echo   Complete. Log: %RESULTS_DIR%\exp1_2_s_10.0.log
echo.

echo [3/5] Testing s=15.0...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPTIMAL_CLAUSES% --depth 3 --s 15.0 --T 15000 > %RESULTS_DIR%\exp1_2_s_15.0.log 2>&1
echo   Complete. Log: %RESULTS_DIR%\exp1_2_s_15.0.log
echo.

echo [4/5] Testing s=20.0...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPTIMAL_CLAUSES% --depth 3 --s 20.0 --T 15000 > %RESULTS_DIR%\exp1_2_s_20.0.log 2>&1
echo   Complete. Log: %RESULTS_DIR%\exp1_2_s_20.0.log
echo.

echo [5/5] Testing s=25.0...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPTIMAL_CLAUSES% --depth 3 --s 25.0 --T 15000 > %RESULTS_DIR%\exp1_2_s_25.0.log 2>&1
echo   Complete. Log: %RESULTS_DIR%\exp1_2_s_25.0.log
echo.

echo ============================================================
echo EXPERIMENT 1.2 COMPLETE!
echo ============================================================
echo.
findstr "Test Accuracy" %RESULTS_DIR%\exp1_2_*.log
echo.
pause

echo ============================================================
echo PHASE 1 - EXPERIMENT 1.3: Optimize Threshold T
echo Using optimal parameters from Experiments 1.1 and 1.2
echo ============================================================
echo.
echo Enter optimal s value from Experiment 1.2 (e.g., 10.0):
set /p OPTIMAL_S=
echo.
echo Using clauses=%OPTIMAL_CLAUSES%, s=%OPTIMAL_S%
echo Varying: T = [5000, 10000, 15000, 20000]
echo.

echo [1/4] Testing T=5000...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPTIMAL_CLAUSES% --depth 3 --s %OPTIMAL_S% --T 5000 > %RESULTS_DIR%\exp1_3_T_5000.log 2>&1
echo   Complete.
echo.

echo [2/4] Testing T=10000...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPTIMAL_CLAUSES% --depth 3 --s %OPTIMAL_S% --T 10000 > %RESULTS_DIR%\exp1_3_T_10000.log 2>&1
echo   Complete.
echo.

echo [3/4] Testing T=15000...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPTIMAL_CLAUSES% --depth 3 --s %OPTIMAL_S% --T 15000 > %RESULTS_DIR%\exp1_3_T_15000.log 2>&1
echo   Complete.
echo.

echo [4/4] Testing T=20000...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPTIMAL_CLAUSES% --depth 3 --s %OPTIMAL_S% --T 20000 > %RESULTS_DIR%\exp1_3_T_20000.log 2>&1
echo   Complete.
echo.

echo ============================================================
echo EXPERIMENT 1.3 COMPLETE!
echo ============================================================
echo.
findstr "Test Accuracy" %RESULTS_DIR%\exp1_3_*.log
echo.
pause

echo ============================================================
echo PHASE 1 - EXPERIMENT 1.4: Optimize Depth
echo Using optimal parameters from all previous experiments
echo ============================================================
echo.
echo Enter optimal T value from Experiment 1.3 (e.g., 15000):
set /p OPTIMAL_T=
echo.
echo Using clauses=%OPTIMAL_CLAUSES%, s=%OPTIMAL_S%, T=%OPTIMAL_T%
echo Varying: depth = [2, 3, 4, 5, 6]
echo.

echo [1/5] Testing depth=2...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPTIMAL_CLAUSES% --depth 2 --s %OPTIMAL_S% --T %OPTIMAL_T% > %RESULTS_DIR%\exp1_4_depth_2.log 2>&1
echo   Complete.
echo.

echo [2/5] Testing depth=3...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPTIMAL_CLAUSES% --depth 3 --s %OPTIMAL_S% --T %OPTIMAL_T% > %RESULTS_DIR%\exp1_4_depth_3.log 2>&1
echo   Complete.
echo.

echo [3/5] Testing depth=4...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPTIMAL_CLAUSES% --depth 4 --s %OPTIMAL_S% --T %OPTIMAL_T% > %RESULTS_DIR%\exp1_4_depth_4.log 2>&1
echo   Complete.
echo.

echo [4/5] Testing depth=5...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPTIMAL_CLAUSES% --depth 5 --s %OPTIMAL_S% --T %OPTIMAL_T% > %RESULTS_DIR%\exp1_4_depth_5.log 2>&1
echo   Complete.
echo.

echo [5/5] Testing depth=6...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPTIMAL_CLAUSES% --depth 6 --s %OPTIMAL_S% --T %OPTIMAL_T% > %RESULTS_DIR%\exp1_4_depth_6.log 2>&1
echo   Complete.
echo.

echo ============================================================
echo EXPERIMENT 1.4 COMPLETE!
echo ============================================================
echo.
findstr "Test Accuracy" %RESULTS_DIR%\exp1_4_*.log
echo.

echo ============================================================
echo PHASE 1 COMPLETE - ALL EXPERIMENTS DONE!
echo ============================================================
echo.
echo Results directory: %RESULTS_DIR%
echo.
echo OPTIMAL CONFIGURATION FOUND:
echo   Clauses: %OPTIMAL_CLAUSES%
echo   Specificity (s): %OPTIMAL_S%
echo   Threshold (T): %OPTIMAL_T%
echo   Depth: [Check logs above]
echo.
echo Next steps:
echo   1. Review all results in %RESULTS_DIR%
echo   2. Document the Pareto-optimal configurations
echo   3. Move to Phase 2: Ensemble approaches (TM Composites)
echo   4. Consider advanced techniques (CoTM, weighted clauses, HVTM)
echo.
echo To analyze all results:
echo   python scripts/analyze_experiments.py --results-dir %RESULTS_DIR%
echo.
pause
