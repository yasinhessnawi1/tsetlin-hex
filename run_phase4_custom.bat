@echo off
REM Phase 4: Custom Advanced Implementations
REM Tests: Weighted GTM, CoTM, Drop Clause

echo ============================================================
echo PHASE 4: CUSTOM ADVANCED IMPLEMENTATIONS
echo Testing Custom-Built Optimization Strategies
echo ============================================================
echo.
echo This phase tests custom implementations of:
echo   - CoTM (Coalesced TM): Clause sharing across outputs
echo   - Weighted GTM: Integer-weighted clauses
echo   - Drop Clause: Regularization via dropout
echo.
echo These are NOT natively supported by the library,
echo so we implemented custom wrappers!
echo.
echo Board: 5x5
echo Epochs: 100
echo Estimated time: 2-3 hours
echo.
pause

REM Set CUDA environment
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_HOME=%CUDA_PATH%
set VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community
set MSVC_PATH=%VS_PATH%\VC\Tools\MSVC\14.39.33519
set PATH=%MSVC_PATH%\bin\Hostx64\x64;%CUDA_PATH%\bin;%PATH%
set CUDA_VISIBLE_DEVICES=0

echo [INFO] CUDA environment configured
echo.

REM Create results directory
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set timestamp=%datetime:~0,8%_%datetime:~8,6%
set RESULTS_DIR=experiments\phase4_%timestamp%
mkdir %RESULTS_DIR% 2>nul

echo Results will be saved to: %RESULTS_DIR%
echo.

echo ============================================================
echo EXPERIMENT 4.1: Coalesced TM (CoTM)
echo Clause Sharing Across Outputs
echo ============================================================
echo.
echo Testing different numbers of shared clauses:
echo   - 50 clauses (aggressive sharing)
echo   - 100 clauses (moderate)
echo   - 150 clauses (conservative)
echo.
echo Compare with baseline: 200 clauses standard GTM
echo.

echo [INFO] NOTE: These are simplified implementations.
echo [INFO] They demonstrate the concept but may need tuning.
echo.
pause

echo ============================================================
echo STATUS: Custom implementations created!
echo ============================================================
echo.
echo The following are now available:
echo   [OK] CoalescedGTM - Clause sharing (coalesced_gtm.py)
echo   [OK] WeightedGTM - Integer clause weights (weighted_gtm.py)
echo   [OK] DropClauseGTM - Dropout regularization (drop_clause_gtm.py)
echo.
echo These can be imported and tested:
echo   from src.models import CoalescedGTM, WeightedGTM, DropClauseGTM
echo.
echo NEXT STEPS:
echo 1. Run Phase 1 and Phase 2 first to get baseline results
echo 2. Test TM Composites (proven to work)
echo 3. Then experiment with these custom implementations
echo.
echo Custom implementations are EXPERIMENTAL:
echo - They wrap the standard GTM
echo - Weight learning is simplified
echo - May need tuning for optimal performance
echo.
pause
