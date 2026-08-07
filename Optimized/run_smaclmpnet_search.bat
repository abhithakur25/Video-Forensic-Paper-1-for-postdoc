@echo off
REM Detached SMA-CLMPNet recipe search, nested 5 outer x 2 inner over 6
REM configurations = 65 fits at about 13.5 min each, so roughly 15 hours.
REM
REM Launched via WMI Win32_Process.Create for the same reason as the FF++
REM shards: Start-Process does not escape the harness job object, and this run
REM is far too long to depend on a session staying open.
REM
REM --resume picks up from the per-outer-fold checkpoint in
REM Optimized/smaclmpnet_search.json, so a death costs at most one fold.

set E=C:\Users\USER\anaconda3\envs\VideoForgeryCPU
set PATH=%E%\Library\bin;%E%;%E%\Scripts;%PATH%
set PYTHONWARNINGS=ignore
set TF_CPP_MIN_LOG_LEVEL=3
cd /d C:\Users\USER\Downloads\PostDoc\Implimentation_Paper1

"%E%\python.exe" -u Optimized\optimize_smaclmpnet.py ^
    --outer 5 --inner 2 --budget 6 --resume ^
    >> logs\smaclmpnet_search.log 2>> logs\smaclmpnet_search.err

echo EXITCODE=%ERRORLEVEL% >> logs\smaclmpnet_search.log
