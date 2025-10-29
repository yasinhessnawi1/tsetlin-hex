@echo off
REM Configure CUDA environment and run training

echo ============================================
echo Configuring CUDA Environment
echo ============================================

REM Set CUDA paths
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=%CUDA_PATH%"

REM Set Visual Studio paths for cl.exe
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
set "MSVC_PATH=%VS_PATH%\VC\Tools\MSVC\14.39.33519"
set "PATH=%MSVC_PATH%\bin\Hostx64\x64;%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

echo CUDA_PATH: %CUDA_PATH%
echo VS_PATH: %VS_PATH%
echo.

REM Verify cl.exe is available
cl.exe >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: cl.exe not found! Please check Visual Studio installation.
    pause
    exit /b 1
)
echo cl.exe found successfully!

REM Verify nvcc is available
nvcc --version
if %errorlevel% neq 0 (
    echo ERROR: nvcc not found! Please check CUDA installation.
    pause
    exit /b 1
)

echo.
echo ============================================
echo Starting Model Training
echo ============================================
echo.

REM Run the training script with proper environment
python scripts/2_train_model.py --board-size 10 --stage end --epochs 100

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo Training Complete!
echo ============================================
pause
