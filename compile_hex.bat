@echo off
echo Compiling hex.c to shared library...

REM Set up MSVC environment
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
set "MSVC_PATH=%VS_PATH%\VC\Tools\MSVC\14.39.33519"
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"

REM Compile for 5x5 board first (for testing)
echo Compiling for 5x5 board...
cl /LD /DBOARD_DIM=5 hex.c /Fe:hex_5x5.dll

echo.
echo Compiling for 10x10 board...
cl /LD /DBOARD_DIM=10 hex.c /Fe:hex_10x10.dll

echo.
echo Compiling for 11x11 board (standard)...
cl /LD /DBOARD_DIM=11 hex.c /Fe:hex_11x11.dll

if %ERRORLEVEL% EQU 0 (
    echo Success! Created hex.dll
) else (
    echo Failed to compile
)

pause

