@echo off
setlocal
call "%~dp0_common.bat"
echo ================================================================
echo  FIX 05: pull the sentiment model  llama3:8b-instruct-q4_K_M  (~4.7 GB)
echo ================================================================
set "OLLAMA="
where ollama >nul 2>&1 && set "OLLAMA=ollama"
if "%OLLAMA%"=="" if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if "%OLLAMA%"=="" ( echo Ollama not installed - run fix_04_ollama.bat first & pause & exit /b 1 )
"%OLLAMA%" pull llama3:8b-instruct-q4_K_M || ( echo pull failed - is the Ollama server running? & pause & exit /b 1 )
echo.
"%OLLAMA%" list
echo.
echo Quick GPU check (the model should show 100%% GPU):
"%OLLAMA%" run llama3:8b-instruct-q4_K_M "Reply with the single word OK" 
"%OLLAMA%" ps
echo.
echo Next: strongpc\00_preflight.bat
pause
