@echo off
setlocal
echo ================================================================
echo  FIX 09: stop Windows from sleeping during the multi-hour PPO run
echo ================================================================
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 30
echo Sleep + hibernate disabled while on AC power (monitor still turns off).
echo To restore later:  powercfg /change standby-timeout-ac 30
pause
