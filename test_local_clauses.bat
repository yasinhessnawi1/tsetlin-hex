@echo off
REM Quick local testing to estimate clause requirements

echo ============================================================
echo LOCAL TESTING - Finding Clause Requirements
echo ============================================================
echo.
echo Dataset: 10,000 training games (local quick test)
echo Purpose: Estimate clause counts for server (1M games)
echo.
echo Testing Stage 0 (easiest) with different clauses:
echo - 200 clauses: Already tested = 93.6%%
echo - 500 clauses: Testing now...
echo - 1000 clauses: Then this...
echo - 2000 clauses: Then this...
echo.
pause

REM Set CUDA environment (same as run_strong_training.bat)
set CUDA_VISIBLE_DEVICES=0

echo.
echo ============================================================
echo [1/3] Testing with 500 clauses
echo ============================================================
python scripts/2_train_model.py --board-size 5 --stage 0 --epochs 50 --clauses 500 --depth 6

echo.
echo Press any key to test 1000 clauses...
pause

echo.
echo ============================================================
echo [2/3] Testing with 1000 clauses
echo ============================================================
python scripts/2_train_model.py --board-size 5 --stage 0 --epochs 50 --clauses 1000 --depth 6

echo.
echo Press any key to test 2000 clauses...
pause

echo.
echo ============================================================
echo [3/3] Testing with 2000 clauses
echo ============================================================
python scripts/2_train_model.py --board-size 5 --stage 0 --epochs 50 --clauses 2000 --depth 6

echo.
echo ============================================================
echo LOCAL TESTING COMPLETE
echo ============================================================
echo.
echo Results summary (Stage 0, 10k games):
echo - 200 clauses: 93.6%% (already tested)
echo - 500 clauses: (check above)
echo - 1000 clauses: (check above)
echo - 2000 clauses: (check above)
echo.
echo Next: Use these numbers to estimate server requirements
echo If 2000 clauses gets 96-98%% on 10k games,
echo then 2000 clauses should get 99-100%% on 1M games!
echo.
pause
