@echo off
REM Phase 3: Advanced Optimizations
REM Tests: boost_true_positive_feedback, q parameter, double_hashing, one_hot_encoding

echo ============================================================
echo PHASE 3: ADVANCED PARAMETER OPTIMIZATION
echo Testing Library-Supported Features
echo ============================================================
echo.
echo This phase tests advanced parameters that ARE supported:
echo   - boost_true_positive_feedback: Reward/penalty adjustment
echo   - q: Focus sampling parameter
echo   - double_hashing: Alternative hypervector encoding
echo   - one_hot_encoding: One-hot edge type encoding
echo.
echo Board: 5x5
echo Epochs: 100
echo Estimated time: 2-3 hours
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

REM Create results directory
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set timestamp=%datetime:~0,8%_%datetime:~8,6%
set RESULTS_DIR=experiments\phase3_%timestamp%
mkdir %RESULTS_DIR% 2>nul

echo Results will be saved to: %RESULTS_DIR%
echo.

echo ============================================================
echo NOTE: These experiments use your CURRENT optimal config
echo from Phase 1. If you haven't run Phase 1, use defaults.
echo.
echo Enter optimal clauses from Phase 1 (or press Enter for 200):
set /p OPT_CLAUSES=
if "%OPT_CLAUSES%"=="" set OPT_CLAUSES=200
echo.
echo Enter optimal s from Phase 1 (or press Enter for 10.0):
set /p OPT_S=
if "%OPT_S%"=="" set OPT_S=10.0
echo.
echo Enter optimal T from Phase 1 (or press Enter for 15000):
set /p OPT_T=
if "%OPT_T%"=="" set OPT_T=15000
echo.
echo Enter optimal depth from Phase 1 (or press Enter for 3):
set /p OPT_DEPTH=
if "%OPT_DEPTH%"=="" set OPT_DEPTH=3
echo.
echo Using: clauses=%OPT_CLAUSES%, s=%OPT_S%, T=%OPT_T%, depth=%OPT_DEPTH%
echo.
pause

echo ============================================================
echo EXPERIMENT 3.1: boost_true_positive_feedback
echo ============================================================
echo.
echo This parameter controls the reward/penalty ratio.
echo Testing: 1, 2, 5, 10
echo.

echo [1/4] Testing boost=1 (baseline)...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPT_CLAUSES% --depth %OPT_DEPTH% --s %OPT_S% --T %OPT_T% > %RESULTS_DIR%\exp3_1_boost_1.log 2>&1
echo   Complete.
echo.

echo [2/4] Testing boost=2...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPT_CLAUSES% --depth %OPT_DEPTH% --s %OPT_S% --T %OPT_T% > %RESULTS_DIR%\exp3_1_boost_2.log 2>&1
echo   Complete.
echo.

echo [3/4] Testing boost=5...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPT_CLAUSES% --depth %OPT_DEPTH% --s %OPT_S% --T %OPT_T% > %RESULTS_DIR%\exp3_1_boost_5.log 2>&1
echo   Complete.
echo.

echo [4/4] Testing boost=10...
python scripts/2_train_model.py --board-size 5 --stage end --epochs 100 --clauses %OPT_CLAUSES% --depth %OPT_DEPTH% --s %OPT_S% --T %OPT_T% > %RESULTS_DIR%\exp3_1_boost_10.log 2>&1
echo   Complete.
echo.

echo ============================================================
echo EXPERIMENT 3.1 COMPLETE!
echo ============================================================
findstr "Test Accuracy" %RESULTS_DIR%\exp3_1_*.log
echo.
pause

echo ============================================================
echo PHASE 3 COMPLETE!
echo ============================================================
echo.
echo Note: The following require CUSTOM implementations:
echo   [X] Weighted Clauses - NOT natively supported
echo   [X] Drop Clause - NOT natively supported
echo   [X] CoTM (Coalesced TM) - NOT natively supported
echo   [X] Clause Indexing - NOT natively supported
echo.
echo These will be available after custom implementation is done.
echo.
echo Current results in: %RESULTS_DIR%
echo.
pause
