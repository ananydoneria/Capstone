@echo off
setlocal
call "%~dp0_common.bat"
echo ================================================================
echo  FIX 06: regenerate price data + GNN  (ONLY if they did not copy over)
echo ================================================================
echo WARNING: this re-downloads prices from yfinance and RETRAINS the GNN.
echo If data\processed\*.parquet and src\models\gnn\weights\*.pt already exist,
echo you almost certainly do NOT want this - it replaces the shipped weights.
echo.
if exist "%ROOT%\src\models\gnn\weights\propagation_gnn.pt" echo   ^> GNN weights ARE present right now.
if exist "%ROOT%\data\processed\features.parquet" echo   ^> features.parquet IS present right now.
echo.
choice /C YN /M "Continue and regenerate anyway"
if errorlevel 2 ( echo Cancelled. & pause & exit /b 0 )
if not exist "%VENV_PY%" ( echo .venv missing - run fix_02_venv.bat first & pause & exit /b 1 )
"%VENV_PY%" scripts\run_phase1.py        || ( echo run_phase1 failed & pause & exit /b 1 )
"%VENV_PY%" scripts\fetch_market_data.py || ( echo fetch_market_data failed & pause & exit /b 1 )
"%VENV_PY%" scripts\train_gnn.py         || ( echo train_gnn failed & pause & exit /b 1 )
"%VENV_PY%" scripts\cache_gnn_scores.py  || ( echo cache_gnn_scores failed & pause & exit /b 1 )
echo.
"%VENV_PY%" -m pytest tests -q -W ignore
echo.
echo Next: strongpc\00_preflight.bat
pause
