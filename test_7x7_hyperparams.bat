@echo off
REM Hyperparameter search for 7x7 board
echo ============================================================
echo 7x7 HYPERPARAMETER SEARCH - LOCAL TESTING
echo ============================================================
echo.
echo This will test different hyperparameter combinations on 7x7
echo to find optimal settings before server training.
echo.
echo Tests planned:
echo   1. Clause count: 800, 1200, 1500, 2000
echo   2. Depth: 6, 7, 8
echo   3. Stage: 0 (easiest) first
echo.
echo Dataset: 30,000 training games (local quick test)
echo Each test: ~3-10 minutes
echo Total time: ~1-2 hours
echo.
pause

REM Set CUDA environment (full paths for PyCUDA)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_HOME=%CUDA_PATH%
set VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community
set MSVC_PATH=%VS_PATH%\VC\Tools\MSVC\14.39.33519
set PATH=%MSVC_PATH%\bin\Hostx64\x64;%CUDA_PATH%\bin;%PATH%
set CUDA_VISIBLE_DEVICES=0

echo [INFO] CUDA environment configured
echo.
echo ============================================================
echo TEST SERIES 1: CLAUSE COUNT (depth=7, s=0.01)
echo ============================================================
echo.

REM Test 1: 800 clauses
echo.
echo [1/7] Testing 800 clauses, depth=7...
python scripts/2_train_model.py --board-size 7 --stage 0 --epochs 80 --clauses 4000 --depth 5
echo.
echo Press any key for next test...
pause > nul

REM Test 2: 1200 clauses
echo.
echo [2/7] Testing 1200 clauses, depth=7...
python scripts/2_train_model.py --board-size 7 --stage 0 --epochs 30 --clauses 6000 --depth 5
echo.
echo Press any key for next test...
pause > nul

REM Test 3: 1500 clauses
echo.
echo [3/7] Testing 1500 clauses, depth=7...
python scripts/2_train_model.py --board-size 7 --stage 0 --epochs 30 --clauses 8000 --depth 5
echo.
echo Press any key for next test...
pause > nul

REM Test 4: 2000 clauses
echo.
echo [4/7] Testing 2000 clauses, depth=7...
python scripts/2_train_model.py --board-size 7 --stage 0 --epochs 30 --clauses 4000 --depth 4
echo.
echo Press any key for next test...
pause > nul

echo.
echo ============================================================
echo TEST SERIES 2: DEPTH (clauses=1200, s=0.01)
echo ============================================================
echo.

REM Test 5: depth=6
echo [5/7] Testing depth=6, 1200 clauses...
python scripts/2_train_model.py --board-size 7 --stage 0 --epochs 30 --clauses 4000 --depth 6
echo.
echo Press any key for next test...
pause > nul

REM Test 6: depth=8
echo.
echo [6/7] Testing depth=8, 1200 clauses...
python scripts/2_train_model.py --board-size 7 --stage 0 --epochs 30 --clauses 4000 --depth 8
echo.
echo Press any key for next test...
pause > nul

echo.
echo ============================================================
echo TEST SERIES 3: STAGE -5 (HARDEST)
echo ============================================================
echo.

REM Test 7: Stage -5 with best params (likely 1500 clauses, depth 7)
echo [7/7] Testing Stage -5 with 1500 clauses, depth=7...
python scripts/2_train_model.py --board-size 7 --stage -5 --epochs 30 --clauses 8000 --depth 5

echo.
echo ============================================================
echo HYPERPARAMETER SEARCH COMPLETE
echo ============================================================
echo.
echo Review the results above and note:
echo   - Which clause count gave best accuracy?
echo   - Which depth gave best accuracy?
echo   - Stage -5 accuracy compared to Stage 0?
echo.
echo Expected patterns:
echo   - More clauses = higher accuracy (up to a point)
echo   - depth=7 likely optimal for 7x7
echo   - Stage -5 should need 2x-3x more clauses
echo.
echo Expected accuracy on 30k games (Stage 0):
echo   - 800 clauses: ~75-80%%
echo   - 1200 clauses: ~80-85%%
echo   - 1500 clauses: ~85-90%%
echo   - 2000 clauses: ~88-92%%
echo.
echo Next steps:
echo   1. Note the best hyperparameters
echo   2. For 100%% accuracy, scale up:
echo      - Training games: 500k-2M
echo      - Clauses: 2x-3x what worked locally
echo      - Stage -5: 3x-4x more clauses than Stage 0
echo.
pause
