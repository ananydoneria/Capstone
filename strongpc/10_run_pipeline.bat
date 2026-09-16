@echo off
setlocal
call "%~dp0_common.bat"
echo ================================================================
echo  RUN PIPELINE: preflight -^> tests -^> news -^> sentiment -^> state gate
echo               -^> baselines -^> PPO x4 -^> evaluate -^> report -^> bundle
echo  Everything is logged under reports\strongpc\run_^<timestamp^>\
echo  Sentiment scoring is the slow stage (1-3 h on the GPU); PPO is minutes.
echo  Do not close this window.
echo ================================================================
if not exist "%VENV_PY%" ( echo .venv missing - run fix_02_venv.bat first & pause & exit /b 1 )
"%VENV_PY%" "%ROOT%\strongpc\run_pipeline.py" %*
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" ( echo PIPELINE FINISHED OK ) else ( echo PIPELINE STOPPED - read SUMMARY.md in the run folder )
pause
exit /b %RC%
