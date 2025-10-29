@echo off

echo ============================================================
echo GENERATING LARGE DATASET FOR 5X5 HEX
echo ============================================================
echo.
echo This will generate 100,000 training games and 20,000 test games
echo for 5x5 Hex boards to achieve near-perfect accuracy.
echo.
echo Estimated time: ~30-60 minutes depending on your CPU
echo.
pause

REM Generate games using C code
python scripts/1_generate_games.py --board-size 5 --num-train 500000 --num-test 80000

echo.
echo ============================================================
echo BUILDING GTM DATASETS
echo ============================================================
echo.

REM Build GTM datasets
python scripts/1b_build_gtm_datasets.py --board-size 5

echo.
echo ============================================================
echo DATA GENERATION COMPLETE!
echo ============================================================
echo.
echo You can now train the model with:
echo   run_strong_training.bat
echo.
pause
