@echo off

REM Set CUDA and VS paths
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=%CUDA_PATH%"
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
set "MSVC_PATH=%VS_PATH%\VC\Tools\MSVC\14.39.33519"
set "PATH=%MSVC_PATH%\bin\Hostx64\x64;%CUDA_PATH%\bin;%PATH%"

python scripts/2_train_model.py --board-size 5 --stage all --clauses 800 --depth 3 --s 8.0 --T 15000 --epochs 30 --message-size 512 --message-bits 4 --hypervector-size 512 --hypervector-bits 4


pause

