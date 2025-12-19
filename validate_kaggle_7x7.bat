@echo off
setlocal
REM Set CUDA and VS paths
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=%CUDA_PATH%"
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
set "MSVC_PATH=%VS_PATH%\VC\Tools\MSVC\14.39.33519"
set "PATH=%MSVC_PATH%\bin\Hostx64\x64;%CUDA_PATH%\bin;%PATH%"
REM Kaggle 7x7 validation (Windows)
REM Uses full Kaggle dataset, builds a single GTM file, then evaluates a model.
REM Usage: validate_kaggle_7x7.bat [STAGE] [MODELS_DIR] [DATA_DIR] [MAX_SAMPLES]
REM Defaults: STAGE=end, MODELS_DIR=models, DATA_DIR=data\kaggle_eval, MAX_SAMPLES=all
REM Examples:
REM   validate_kaggle_7x7.bat                    # Use all ~1M games
REM   validate_kaggle_7x7.bat end models data 50000  # Use 50K games subset
REM   validate_kaggle_7x7.bat -2 models data 100000  # Use 100K games for -2 stage
REM
REM NOTE: If you change MAX_SAMPLES, delete existing NPZ/PKL files to force recreation:
REM   del data\kaggle_eval\*.npz data\kaggle_eval\*.pkl

if "%~1"=="" (
  set "STAGE=end"
) else (
  set "STAGE=%~1"
)

if "%~2"=="" (
  set "MODELS_DIR=models"
) else (
  set "MODELS_DIR=%~2"
)

if "%~3"=="" (
  set "DATA_DIR=data\kaggle_eval"
) else (
  set "DATA_DIR=%~3"
)

if "%~4"=="" (
  set "MAX_SAMPLES="
) else (
  set "MAX_SAMPLES=%~4"
)

REM Hypervector settings (override via env if needed)
if "%HYPERVECTOR_SIZE%"=="" set "HYPERVECTOR_SIZE=64"
if "%HYPERVECTOR_BITS%"=="" set "HYPERVECTOR_BITS=4"

set "BOARD_SIZE=7"
set "GEN_STAGES=0"
set "KAGGLE_DIR=%DATA_DIR%\kaggle_game_of_hex"
set "RAW_TRAIN=%DATA_DIR%\train_games_%BOARD_SIZE%x%BOARD_SIZE%.npz"
set "RAW_TEST=%DATA_DIR%\test_games_%BOARD_SIZE%x%BOARD_SIZE%.npz"
set "FULL_GTM=%DATA_DIR%\full_train_gtm_%BOARD_SIZE%x%BOARD_SIZE%_%GEN_STAGES%.pkl"

echo.
echo ============================================================
echo Kaggle 7x7 Validation Pipeline (Windows)
echo   Board      : %BOARD_SIZE%x%BOARD_SIZE%
echo   Stage      : %STAGE%
echo   Models dir : %MODELS_DIR%
echo   Data dir   : %DATA_DIR%
echo   Gen stages : %GEN_STAGES%
echo   HV size    : %HYPERVECTOR_SIZE%
echo   HV bits    : %HYPERVECTOR_BITS%
if "%MAX_SAMPLES%"=="" (
  echo   Max samples: all
) else (
  echo   Max samples: %MAX_SAMPLES%
)
echo ============================================================


pushd "%~dp0"
set "PYTHONPATH=%CD%;%PYTHONPATH%"

REM 1) Download Kaggle dataset (skip if raw NPZ exists)
if exist "%RAW_TRAIN%" (
  if exist "%RAW_TEST%" (
    echo [SKIP] Found raw NPZ files: %RAW_TRAIN%, %RAW_TEST%
    if "%MAX_SAMPLES%" neq "" (
      echo [NOTE] delete existing files to recreate subset:
      echo [NOTE]   del data\kaggle_eval\*.npz data\kaggle_eval\*.pkl
    )
  ) else (
    goto :dl_kaggle
  )
) else (
:dl_kaggle
  if not exist "%KAGGLE_DIR%" mkdir "%KAGGLE_DIR%"
  echo [RUN] Downloading Kaggle dataset to %KAGGLE_DIR% ...
  kaggle datasets download -d cholling/game-of-hex -p "%KAGGLE_DIR%" --unzip
)

REM 2) Convert to GTM raw npz (skip if already present)
if exist "%RAW_TRAIN%" (
  if exist "%RAW_TEST%" (
    echo [SKIP] Raw NPZ already present; skipping conversion.
  ) else (
    goto :conv_kaggle
  )
) else (
:conv_kaggle
  echo [RUN] Converting Kaggle dataset to GTM raw format...
  set "CONV_CMD=python scripts/convert_kaggle_game_of_hex.py ^
    --dataset-dir "%KAGGLE_DIR%" ^
    --board-size %BOARD_SIZE% ^
    --train-output "%RAW_TRAIN%" ^
    --test-output "%RAW_TEST%" ^
    --all-to-test"
  if "%MAX_SAMPLES%" neq "" (
    set "CONV_CMD=%CONV_CMD% --max-samples %MAX_SAMPLES%"
  )
  %CONV_CMD%
)

REM 3) Build single full GTM (skip if exists)
if exist "%FULL_GTM%" (
  echo [SKIP] Full GTM already exists: %FULL_GTM%
  goto :evaluate
) else (
:build_gtm
  echo [RUN] Building GTM datasets (single full set)
  python scripts/1b_build_gtm_datasets.py ^
    --board-size %BOARD_SIZE% ^
    --stages %GEN_STAGES% ^
    --train-file "%RAW_TRAIN%" ^
    --test-file "%RAW_TEST%" ^
    --output-dir "%DATA_DIR%" ^
    --output-prefix "full_" ^
    --single-output ^
    --hypervector-size %HYPERVECTOR_SIZE% ^
    --hypervector-bits %HYPERVECTOR_BITS%
)

REM 4) Evaluate using saved model(s) against the full GTM set
if exist "%FULL_GTM%" (
  echo [RUN] Evaluating...
) else (
  echo [ERROR] Full GTM does not exist: %FULL_GTM%
  pause
  exit /b 1
)
:evaluate
  python scripts/3_evaluate.py ^
  --board-size %BOARD_SIZE% ^
  --stage %STAGE% ^
  --data-dir "%DATA_DIR%" ^
  --models-dir "%MODELS_DIR%" ^
  --test-file "%FULL_GTM%" ^
  --gen-stages %GEN_STAGES% ^
  --visualize ^
  --viz-dir "evaluation_plots"

echo.
echo ============================================================
echo Kaggle 7x7 Validation COMPLETE
echo ============================================================

popd
exit /b 0
