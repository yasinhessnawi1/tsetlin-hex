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

REM Run training
python scripts\2_train_model.py --board-size 10 --stage end --epochs 100 --T 15000 --clauses 12000 --depth 6

echo.
echo ============================================================
echo RUNNING EVALUATION
echo ============================================================
echo.

REM Run evaluation on the latest trained model
python scripts\3_evaluate.py --board-size 10 --stage end --latest

pause
