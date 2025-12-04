@echo off
REM Generate small 10x10 dataset for local hyperparameter testing
echo ============================================================
echo GENERATING 10x10 DATASET - LOCAL TESTING
echo ============================================================
echo.
echo This will generate a SMALL dataset for quick local testing:
echo   - 5,000 training games (small for speed)
echo   - 1,000 test games
echo   - Stages: 0, -2, -5
echo.
echo Purpose: Find optimal hyperparameters before server training
echo Expected time: ~5-10 minutes
echo.
pause

echo.
echo ============================================================
echo STEP 1: COMPILING C CODE FOR 10x10
echo ============================================================

REM Set up MSVC environment
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1

echo Compiling hex_datagen_stages.c for 10x10...
cl /O2 /DBOARD_DIM=10 hex_binaries\hex_datagen_stages.c /Fe:hex_binaries\hex_datagen_10x10.exe > nul 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Compilation failed!
    pause
    exit /b 1
)

echo [OK] hex_datagen_10x10.exe compiled

echo.
echo ============================================================
echo STEP 2: GENERATING GAMES
echo ============================================================

python scripts/1_generate_games.py --board-size 10 --num-train 50000 --num-test 10000 --save-states 0,-2,-5

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Game generation failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo STEP 3: BUILDING GTM DATASETS
echo ============================================================
echo.
echo Using hypervector_size=256, bits=4 for 10x10 (23 symbols)
echo This gives better accuracy than default 128!
echo.

python scripts/1b_build_gtm_datasets.py --board-size 10 --hypervector-size 256 --hypervector-bits 4 --stages all

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: GTM dataset building failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS! 10x10 DATASET READY
echo ============================================================
echo.
echo Dataset location:
echo   - data/train_gtm_10x10_0.pkl
echo   - data/train_gtm_10x10_-2.pkl
echo   - data/train_gtm_10x10_-5.pkl
echo.
echo Next: Run hyperparameter search
echo   test_10x10_hyperparams.bat
echo.
pause
