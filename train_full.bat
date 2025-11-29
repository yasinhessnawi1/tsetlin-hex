@echo off

REM Navigate to script directory (makes script portable)
cd /d %~dp0

REM Set CUDA and MSVC paths explicitly
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=%CUDA_PATH%"
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
set "MSVC_PATH=%VS_PATH%\VC\Tools\MSVC\14.39.33519"

REM Setup Visual Studio environment
call "%VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Add CUDA and MSVC to PATH
set "PATH=%MSVC_PATH%\bin\Hostx64\x64;%CUDA_PATH%\bin;%PATH%"

REM Activate virtual environment
call .\.venv\Scripts\activate.bat

echo.
echo ============================================================
echo STEP 1: GENERATING TRAINING DATA (1M train + 200k test)
echo ============================================================
echo.

python scripts\1_generate_games.py --board-size 10 --num-train 1000000 --num-test 200000 --save-states 0

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Data generation failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo STEP 2: BUILDING GTM DATASETS (Binary Encoding)
echo ============================================================
echo.

python scripts\1b_build_gtm_datasets.py --board-size 10 --hypervector-size 256 --hypervector-bits 4 --stages end

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Dataset building failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo STEP 3: TRAINING MODEL
echo ============================================================
echo.

REM Run training with optimal hyperparameters
python scripts\2_train_model.py --board-size 10 --stage end --epochs 100 --T 8000 --clauses 10000 --s 100 --depth 6

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo STEP 4: EVALUATING MODEL
echo ============================================================
echo.

REM Run evaluation on the latest trained model
python scripts\3_evaluate.py --board-size 10 --stage end --latest

pause
