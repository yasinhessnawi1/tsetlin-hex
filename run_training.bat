@echo off

REM Set CUDA and VS paths
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=%CUDA_PATH%"
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
set "MSVC_PATH=%VS_PATH%\VC\Tools\MSVC\14.39.33519"
set "PATH=%MSVC_PATH%\bin\Hostx64\x64;%CUDA_PATH%\bin;%PATH%"

python scripts/2_train_model.py --board-size 7 --stage all --clauses 322 --depth 3 --s-tuple "1.6,1.2,1.0" --T 322 --epochs 100 --message-size 512 --message-bits 4 --no-balance --gen-stages 0


pause

