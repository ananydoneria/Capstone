@echo off
setlocal
call "%~dp0_common.bat"
echo Opening TensorBoard for src\rl_agent\logs  (Ctrl+C here to stop)
start http://localhost:6006
"%VENV_PY%" -m tensorboard.main --logdir "%ROOT%\src\rl_agent\logs"
