@echo off
echo Setting up CUDA environment for PyCUDA...

REM Add CUDA 12.8 to PATH
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA_PATH%\bin;%PATH%"

echo CUDA_PATH set to: %CUDA_PATH%
echo.
echo Testing nvcc...
nvcc --version

echo.
echo Environment configured! You can now run:
echo   python scripts/2_train_model.py --board-size 10 --stage end --epochs 100
echo.
