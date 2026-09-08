@echo off
setlocal
call "%~dp0_common.bat"
echo ================================================================
echo  TESTS ONLY: full pytest + GNN test report (md + pdf), saved to a run folder
echo ================================================================
if not exist "%VENV_PY%" ( echo .venv missing - run fix_02_venv.bat first & pause & exit /b 1 )
"%VENV_PY%" "%ROOT%\strongpc\run_pipeline.py" --only tests_before,gnn_report,gnn_pdf,collect
pause
