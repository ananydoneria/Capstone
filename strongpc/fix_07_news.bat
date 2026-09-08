@echo off
setlocal
call "%~dp0_common.bat"
echo ================================================================
echo  FIX 07: news corpus backfill (Finnhub if key set, then Google News RSS)
echo ================================================================
if not exist "%VENV_PY%" ( echo .venv missing - run fix_02_venv.bat first & pause & exit /b 1 )
if "%FINNHUB_API_KEY%"=="" (
    echo FINNHUB_API_KEY is not set - Finnhub will be skipped (allowed).
    echo To include it:  set FINNHUB_API_KEY=your_key   then re-run this script.
)
"%VENV_PY%" scripts\fetch_finnhub_news.py
"%VENV_PY%" scripts\fetch_google_news.py
echo.
echo Both are resume-safe: if you saw WARN lines (rate limiting), re-run this
echo script later until every ticker reports 0 new month files.
echo Next: strongpc\00_preflight.bat
pause
