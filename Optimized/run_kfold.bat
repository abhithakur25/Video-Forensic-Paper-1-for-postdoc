@echo off
REM Detached k-fold launcher.
REM
REM Launched via WMI Win32_Process.Create so the process is owned by the WMI
REM service rather than the calling shell. Start-Process from an agent task
REM does NOT escape the harness job object: PID 29352 was killed the moment
REM its launching task was reaped, 6 model-fits in.
REM
REM --resume picks up from the per-k checkpoint in Analysis1\TRUE_KF.

set E=C:\Users\USER\anaconda3\envs\VideoForgeryCPU
set PATH=%E%\Library\bin;%E%;%E%\Scripts;%PATH%
set PYTHONWARNINGS=ignore
cd /d C:\Users\USER\Downloads\PostDoc\Implimentation_Paper1

"%E%\python.exe" -u Optimized\optimize_models.py ^
    --mode kfold --ks 6,7,8,9,10 --folds-per-k 1 ^
    --epochs 30 --batch-size 8 --epochs-baseline 10 ^
    --out Analysis1\TRUE_KF --resume ^
    >> logs\kfold_true.log 2>> logs\kfold_true.err

echo EXITCODE=%ERRORLEVEL% >> logs\kfold_true.log
