@echo off
REM Detached FF++ face-crop shard launcher.  Usage: run_ffpp_shard.bat <shard>
REM
REM Launched via WMI Win32_Process.Create so the process is owned by the WMI
REM service rather than the calling shell - the same reason run_kfold.bat
REM exists. The first attempt at this job (2026-08-07 15:38) was launched from
REM the session and all 8 shards died together at ~16:15 when that session
REM ended, 65 of 250 videos in.
REM
REM ffpp_prepare.py is resumable: a video whose crops already exist is skipped,
REM so a relaunch costs only a directory scan for the work already done.

set E=C:\Users\USER\anaconda3\envs\VideoForgeryCPU
set PATH=%E%\Library\bin;%E%;%E%\Scripts;%PATH%
set PYTHONWARNINGS=ignore
cd /d C:\Users\USER\Downloads\PostDoc\Implimentation_Paper1

"%E%\python.exe" -u Optimized\ffpp_prepare.py ^
    --methods Deepfakes --frames 32 --size 299 --margin 0.25 ^
    --shard %1 --nshards 8 ^
    >> logs\ffpp_shard%1.log 2>> logs\ffpp_shard%1.err

echo EXITCODE=%ERRORLEVEL% >> logs\ffpp_shard%1.log
