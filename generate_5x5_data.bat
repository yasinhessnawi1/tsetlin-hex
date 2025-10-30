@echo off

echo ============================================================
echo GENERATING LARGE DATASET FOR 5X5 HEX
echo ============================================================
echo.
echo This will generate 50,000 training games and 10,000 test games
echo for 5x5 Hex boards to achieve near-perfect accuracy.
echo.
echo Estimated time: ~30-60 minutes depending on your CPU
echo.
pause

REM Generate games using C code with multi-stage tracking
python scripts/1_generate_games.py --board-size 5 --num-train 10000 --num-test 2000 --save-states 0,-2,-5

echo.
echo ============================================================
echo BUILDING GTM DATASETS FOR ALL STAGES
echo ============================================================
echo.

REM Build GTM datasets for all stages (0, -2, -5)
python scripts/1b_build_gtm_datasets.py --board-size 5 --stages all

echo.
echo ============================================================
echo DATA GENERATION COMPLETE!
echo ============================================================
echo.
echo You can now train the model with:
echo   run_strong_training.bat
echo.
pause
