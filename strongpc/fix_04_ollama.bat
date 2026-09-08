@echo off
setlocal
call "%~dp0_common.bat"
echo ================================================================
echo  FIX 04: install Ollama and make sure the server is running
echo ================================================================
set "OLLAMA="
where ollama >nul 2>&1 && set "OLLAMA=ollama"
if "%OLLAMA%"=="" if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if "%OLLAMA%"=="" (
    where winget >nul 2>&1
    if errorlevel 1 (
        echo winget not available. Install Ollama manually from https://ollama.com/download
        start https://ollama.com/download
        pause & exit /b 1
    )
    winget install --id Ollama.Ollama -e --source winget --accept-package-agreements --accept-source-agreements
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
    if "%OLLAMA%"=="" (
        echo Installed, but ollama.exe not on PATH yet. CLOSE this window, open a new one,
        echo and run this script again.
        pause & exit /b 1
    )
)
echo Ollama found: %OLLAMA%
curl -s -o nul -w "%%{http_code}" http://127.0.0.1:11434/api/tags 2>nul | findstr 200 >nul
if errorlevel 1 (
    echo Server not reachable - starting "ollama serve" in the background...
    start "ollama serve" /min "%OLLAMA%" serve
    timeout /t 8 /nobreak >nul
)
curl -s -o nul -w "%%{http_code}" http://127.0.0.1:11434/api/tags 2>nul | findstr 200 >nul
if errorlevel 1 (
    echo Still not reachable. Launch the Ollama app from the Start menu (it runs as a
    echo tray icon) and re-run preflight.
) else (
    echo Ollama server is up on port 11434.
)
echo.
echo Next: strongpc\fix_05_llama_model.bat  (pull the 8B model)
pause
