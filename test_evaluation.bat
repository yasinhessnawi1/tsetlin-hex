@echo off

REM Navigate to script directory (makes script portable)
cd /d %~dp0

REM Setup Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Activate virtual environment
call .\.venv\Scripts\activate.bat

REM Run evaluation test
python scripts\3_evaluate.py --board-size 10 --stage end --latest
