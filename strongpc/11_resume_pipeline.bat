@echo off
setlocal
call "%~dp0_common.bat"
echo ================================================================
echo  RESUME: re-run from a stage (already-trained PPO arms are skipped)
echo  Stages: preflight tests_before news_finnhub news_google sentiment
echo          build_state baselines ppo_full ppo_no-gnn ppo_no-sentiment
echo          ppo_neither evaluate report tests_after gnn_report gnn_pdf collect
echo ================================================================
set /p STAGE="Start from stage [default: sentiment]: "
if "%STAGE%"=="" set "STAGE=sentiment"
"%VENV_PY%" "%ROOT%\strongpc\run_pipeline.py" --from %STAGE% %*
pause
