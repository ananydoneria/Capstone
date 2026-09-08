@echo off
setlocal
call "%~dp0_common.bat"
echo ================================================================
echo  FIX 03: replace CPU-only torch with the CUDA 12.1 build (RTX 3090)
echo ================================================================
if not exist "%VENV_PY%" ( echo .venv missing - run fix_02_venv.bat first & pause & exit /b 1 )
where nvidia-smi >nul 2>&1 || (
    echo nvidia-smi not found: no NVIDIA driver. Run fix_08_nvidia_driver.bat first.
    pause & exit /b 1
)
"%VENV_PY%" -m pip uninstall -y torch
"%VENV_PY%" -m pip install "torch>=2.3,<2.6" --index-url https://download.pytorch.org/whl/cu121 || ( echo install failed & pause & exit /b 1 )
echo.
"%VENV_PY%" -c "import torch; print('torch', torch.__version__, '| cuda:', torch.version.cuda, '| available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU VISIBLE - check driver (fix_08)')"
echo.
echo Next: strongpc\00_preflight.bat
pause
