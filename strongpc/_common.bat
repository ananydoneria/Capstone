@echo off
rem Shared prologue: cd to project root, pick the python to use.
rem Called via:  call "%~dp0_common.bat"
cd /d "%~dp0.."
set "ROOT=%CD%"
set "VENV_PY=%ROOT%\.venv\Scripts\python.exe"
set "PY="
if exist "%VENV_PY%" set "PY="%VENV_PY%""
if "%PY%"=="" (
    where py >nul 2>&1 && set "PY=py -3"
)
if "%PY%"=="" (
    where python >nul 2>&1 && set "PY=python"
)
if not exist "%ROOT%\reports\strongpc" mkdir "%ROOT%\reports\strongpc"
