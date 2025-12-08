@echo off
setlocal

rem ------------------------------------------------------------
rem Environment setup (match other batch scripts)
rem ------------------------------------------------------------
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=%CUDA_PATH%"
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
set "MSVC_PATH=%VS_PATH%\VC\Tools\MSVC\14.39.33519"
set "PATH=%MSVC_PATH%\bin\Hostx64\x64;%CUDA_PATH%\bin;%PATH%"

rem ------------------------------------------------------------
rem Validate a saved GTM model using scripts/3_evaluate.py.
rem It will auto-generate missing test data (games + GTM pkls) if needed.
rem Usage:
rem   validate_model.bat [BOARD_SIZE] [STAGE] [DATA_DIR] [MODELS_DIR] [NUM_TRAIN] [NUM_TEST] [GEN_STAGES]
rem Defaults:
rem   BOARD_SIZE: 11
rem   STAGE: end
rem   DATA_DIR: data
rem   MODELS_DIR: models
rem   NUM_TRAIN: 10000
rem   NUM_TEST: 3000
rem   GEN_STAGES: all   (0,-2,-5)
rem ------------------------------------------------------------

set "BOARD_SIZE=%~1"
if "%BOARD_SIZE%"=="" set "BOARD_SIZE=11"

set "STAGE=%~2"
if "%STAGE%"=="" set "STAGE=end"

set "DATA_DIR=%~3"
if "%DATA_DIR%"=="" set "DATA_DIR=data"

set "MODELS_DIR=%~4"
if "%MODELS_DIR%"=="" set "MODELS_DIR=models"

set "NUM_TRAIN=%~5"
if "%NUM_TRAIN%"=="" set "NUM_TRAIN=10000"

set "NUM_TEST=%~6"
if "%NUM_TEST%"=="" set "NUM_TEST=3000"

set "GEN_STAGES=%~7"
if "%GEN_STAGES%"=="" set "GEN_STAGES=all"

echo.
echo ================================================
echo Validating model via scripts/3_evaluate.py
echo   Board      : %BOARD_SIZE%x%BOARD_SIZE%
echo   Stage      : %STAGE%
echo   Data dir   : %DATA_DIR%
echo   Models dir : %MODELS_DIR%
echo   Num train  : %NUM_TRAIN%
echo   Num test   : %NUM_TEST%
echo   Gen stages : %GEN_STAGES%
echo ================================================
echo.

pushd "%~dp0"
set "PYTHONPATH=%CD%;%PYTHONPATH%"

python scripts/3_evaluate.py ^
  --board-size %BOARD_SIZE% ^
  --stage %STAGE% ^
  --data-dir "%DATA_DIR%" ^
  --models-dir "%MODELS_DIR%" ^
  --num-train %NUM_TRAIN% ^
  --num-test %NUM_TEST% ^
  --gen-stages "%GEN_STAGES%"

set EXITCODE=%ERRORLEVEL%
popd
exit /b %EXITCODE%

