@echo off
echo Compiling hex_datagen_stages.c for multi-stage data generation...

REM Set up MSVC environment
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"

REM Compile for 5x5 board
echo Compiling for 5x5 board...
cl /O2 /DBOARD_DIM=5 hex_datagen_stages.c /Fe:hex_datagen_5x5.exe

echo.
echo Compiling for 7x7 board...
cl /O2 /DBOARD_DIM=7 hex_datagen_stages.c /Fe:hex_datagen_7x7.exe

echo.
echo Compiling for 10x10 board...
cl /O2 /DBOARD_DIM=10 hex_datagen_stages.c /Fe:hex_datagen_10x10.exe

echo.
echo Compiling for 11x11 board...
cl /O2 /DBOARD_DIM=11 hex_datagen_stages.c /Fe:hex_datagen_11x11.exe

if %ERRORLEVEL% EQU 0 (
    echo Success! Created multi-stage data generation executables
    echo.
    echo Test with: hex_datagen_5x5.exe 10 0 -2 -5
    echo This generates 10 games with 3 stages: end (0), -2 moves, -5 moves
) else (
    echo Failed to compile
)

pause


