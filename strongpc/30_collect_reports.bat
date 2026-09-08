@echo off
setlocal
call "%~dp0_common.bat"
echo ================================================================
echo  COLLECT: bundle all reports/results into reports\strongpc\run_*.zip
echo ================================================================
if not exist "%VENV_PY%" ( echo .venv missing - run fix_02_venv.bat first & pause & exit /b 1 )
"%VENV_PY%" "%ROOT%\strongpc\collect_reports.py" %*
echo.
echo Copy the newest reports\strongpc\run_*.zip back to the other machine.
pause
