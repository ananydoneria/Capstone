@echo off
setlocal
echo ================================================================
echo  FIX 08: NVIDIA driver  (cannot be installed silently - manual step)
echo ================================================================
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo nvidia-smi not found: no NVIDIA driver is installed.
) else (
    echo Current driver:
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo The CUDA 12.1 torch wheels need driver 531 or newer.
)
echo.
echo 1. Download the latest Game Ready / Studio driver for the RTX 3090:
echo      https://www.nvidia.com/Download/index.aspx
echo 2. Install it (needs admin), REBOOT.
echo 3. Re-run strongpc\00_preflight.bat, then fix_03_torch_cuda.bat if torch
echo    still cannot see the GPU.
start https://www.nvidia.com/Download/index.aspx
pause
