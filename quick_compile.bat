@echo off
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1

echo Compiling hex_datagen_stages.c for 5x5...
cl /O2 /DBOARD_DIM=5 hex_datagen_stages.c /Fe:hex_datagen_5x5.exe

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Success! Testing with 100 games...
    hex_datagen_5x5.exe 100 0 -2 -5 > test_100games.csv 2>&1

    echo.
    echo Analyzing results...
    python -c "data = open('test_100games.csv').read(); lines = [l for l in data.split('\n') if l.strip() and not 'Generated' in l and not 'Done' in l]; p0 = sum(1 for l in lines if l.startswith('0,')); p1 = sum(1 for l in lines if l.startswith('1,')); print(f'Total games: {len(lines)}'); print(f'Player 0 wins: {p0} ({100.0*p0/len(lines) if lines else 0:.1f}%%)'); print(f'Player 1 wins: {p1} ({100.0*p1/len(lines) if lines else 0:.1f}%%)')"
) else (
    echo Compilation failed!
)
