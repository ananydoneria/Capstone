@echo off
setlocal
call "%~dp0_common.bat"
echo ================================================================
echo  FIX 02: create .venv and install requirements.txt
echo ================================================================
if exist "%ROOT%\.venv" (
    echo An existing .venv was found. If it was copied from another PC it is
    echo broken (venvs hard-code absolute paths) and must be recreated.
    choice /C YN /M "Delete .venv and recreate it"
    if errorlevel 2 ( echo Keeping .venv. & pause & exit /b 0 )
    rmdir /s /q "%ROOT%\.venv"
)
set "BASE="
py -3.11 -c "print()" >nul 2>&1 && set "BASE=py -3.11"
if "%BASE%"=="" py -3.12 -c "print()" >nul 2>&1 && set "BASE=py -3.12"
if "%BASE%"=="" py -3.10 -c "print()" >nul 2>&1 && set "BASE=py -3.10"
if "%BASE%"=="" (
    python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1 && set "BASE=python"
)
if "%BASE%"=="" (
    echo No Python 3.10-3.12 found. Run strongpc\fix_01_python.bat first.
    pause & exit /b 1
)
echo Using base interpreter: %BASE%
%BASE% -m venv "%ROOT%\.venv" || ( echo venv creation failed & pause & exit /b 1 )
set "PY=%ROOT%\.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip wheel
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo NVIDIA GPU detected: installing the CUDA 12.1 torch wheel FIRST so that
    echo requirements.txt does not pull the CPU-only build.
    "%PY%" -m pip install "torch>=2.3,<2.6" --index-url https://download.pytorch.org/whl/cu121
)
"%PY%" -m pip install -r "%ROOT%\requirements.txt" || ( echo pip install failed & pause & exit /b 1 )
echo.
echo Verifying imports...
"%PY%" -c "import torch, torch_geometric, stable_baselines3, langchain_ollama, tqdm, rich; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
echo.
echo Done. Next: strongpc\00_preflight.bat
pause
