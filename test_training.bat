@echo off

REM Navigate to script directory (makes script portable)
cd /d %~dp0

REM Setup Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Activate virtual environment
call .\.venv\Scripts\activate.bat

REM Generate small test data (100 games)
echo Generating test data...
python scripts\1_generate_games.py --board-size 10 --num-train 100 --num-test 20

REM Build GTM datasets
echo Building GTM datasets...
python scripts\1b_build_gtm_datasets.py --board-size 10

REM Run training test
python scripts\2_train_model.py --board-size 10 --stage end --epochs 1 --clauses 100 --depth 6
