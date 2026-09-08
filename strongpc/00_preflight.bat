@echo off
setlocal
call "%~dp0_common.bat"
echo ================================================================
echo  STRONG-PC PREFLIGHT  (checks + score; names the fix script per gap)
echo ================================================================
if "%PY%"=="" (
    echo [FAIL] No python found at all. Run strongpc\fix_01_python.bat first.
    pause & exit /b 1
)
rem Prefer the venv python; preflight itself needs only the standard library.
%PY% "%ROOT%\strongpc\preflight.py" %*
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
    echo RESULT: READY. Next: strongpc\10_run_pipeline.bat
) else (
    echo RESULT: NOT READY. Run the fix_*.bat scripts listed above, then re-run this.
)
echo Report saved under reports\strongpc\
pause
exit /b %RC%
