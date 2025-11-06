@echo off
REM Generate small 7x7 dataset for local hyperparameter testing
echo ============================================================
echo GENERATING 7x7 DATASET - LOCAL TESTING
echo ============================================================
echo.
echo This will generate a MEDIUM dataset for local testing:
echo   - 30,000 training games (moderate size)
echo   - 6,000 test games
echo   - Stages: 0, -2, -5
echo.
echo Purpose: Find optimal hyperparameters before server training
echo Expected time: ~3-5 minutes
echo.
pause

echo.
echo ============================================================
echo STEP 1: CHECKING C EXECUTABLE FOR 7x7
echo ============================================================

if not exist "hex_datagen_7x7.exe" (
    echo Compiling hex_datagen_stages.c for 7x7...

    REM Set up MSVC environment
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
    call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1

    cl /O2 /DBOARD_DIM=7 hex_datagen_stages.c /Fe:hex_datagen_7x7.exe > nul 2>&1

    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Compilation failed!
        pause
        exit /b 1
    )

    echo [OK] hex_datagen_7x7.exe compiled
) else (
    echo [OK] hex_datagen_7x7.exe already exists
)

echo.
echo ============================================================
echo STEP 2: GENERATING GAMES
echo ============================================================

python scripts/1_generate_games.py --board-size 7 --num-train 70000 --num-test 14000 --save-states 0,-2,-5

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
echo Using hypervector_size=192, bits=4 for 7x7
echo This is optimized for ~15-18 symbols in 7x7 graphs
echo.

python scripts/1b_build_gtm_datasets.py --board-size 7 --hypervector-size 256 --hypervector-bits 4 --stages all

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: GTM dataset building failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS! 7x7 DATASET READY
echo ============================================================
echo.
echo Dataset location:
echo   - data/train_gtm_7x7_0.pkl
echo   - data/train_gtm_7x7_-2.pkl
echo   - data/train_gtm_7x7_-5.pkl
echo.
echo Next: Run hyperparameter search
echo   test_7x7_hyperparams.bat
echo.
pause
