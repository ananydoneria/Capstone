@echo off
setlocal
echo ================================================================
echo  FIX 01: install Python 3.11 (needed to create the project venv)
echo ================================================================
where winget >nul 2>&1
if errorlevel 1 (
    echo winget is not available. Install Python 3.11 manually:
    echo   https://www.python.org/downloads/release/python-3119/
    echo   -- tick "Add python.exe to PATH" in the installer --
    start https://www.python.org/downloads/release/python-3119/
    pause & exit /b 1
)
winget install --id Python.Python.3.11 -e --source winget --accept-package-agreements --accept-source-agreements
echo.
echo Done. CLOSE this window and open a NEW one so PATH updates, then run
echo   strongpc\fix_02_venv.bat
pause
