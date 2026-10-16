@echo off
setlocal EnableExtensions
set "PYTHONUNBUFFERED=1"

call conda activate lotus
if errorlevel 1 (
    echo ERROR: Could not activate conda environment lotus.
    exit /b 1
)

cd /d "%~dp0.."

set "RUNNER=%CD%\scripts\run_top1_llm_verify.py"

set "MODE="
if /I "%~1"=="dry" set "MODE=--dry-run"
if "%~1"=="" goto LAUNCH
if /I "%~1"=="dry" goto LAUNCH
echo Unknown argument "%~1"
echo Usage: %~nx0 [dry]
exit /b 2
:LAUNCH
REM ######################################################################
REM wt: Top-1 Semantic Retrieval + LLM Verification baseline.
REM
REM Run 0 completed under outputs\lotus_top1_llm_verify is intentionally
REM left untouched. This launcher runs four additional independent
REM repetitions (run1..run4), each into its own isolated output root.
REM All 32 datasets; join-value columns resolved from each dataset's rows.txt
REM (TabulaX/AutoFJ canonical config). No manual join-column overrides.
REM ######################################################################
set "RUN=1"
:REPETITION_LOOP
if %RUN% GTR 4 goto REPETITIONS_DONE
set "OUT=%CD%\outputs\lotus_top1_llm_verify_run%RUN%"
echo.
echo ============================================================
echo Family: wt  -  repetition run%RUN% of 4  -  output root: %OUT%
echo ============================================================
python "%RUNNER%" --family wt --dataset AJ-beatles-songs --dataset AJ-california-govs-1 --dataset AJ-california-govs-2 --dataset AJ-chinese-provinces --dataset AJ-christmas-songs-1 --dataset AJ-christmas-songs-2 --dataset AJ-duke-cs-profs --dataset AJ-fruits-1 --dataset AJ-fruits-2 --dataset AJ-fsu-name-to-username --dataset AJ-k12-name-to-email --dataset AJ-new-york-govs-1 --dataset AJ-new-york-govs-2 --dataset AJ-new-york-govs-3 --dataset AJ-new-york-govs-4 --dataset AJ-new-york-govs-5 --dataset AJ-park-to-state-1 --dataset AJ-park-to-state-2 --dataset AJ-sharif-email-to-url --dataset AJ-sharif-username-to-email --dataset AJ-texas-govs-1 --dataset AJ-texas-govs-2 --dataset AJ-uk-prime-ministers --dataset AJ-us-cities --dataset AJ-us-presidents-1 --dataset AJ-us-presidents-2 --dataset AJ-us-presidents-3 --dataset AJ-us-presidents-4 --dataset AJ-us-presidents-5 --dataset AJ-us-presidents-6 --dataset AJ-vegetables --dataset names200 --embedding-batch-size 64 --llm-batch-size 8 --max-retries 5 --retry-backoff-seconds 2.0 --continue-on-error %MODE% --output-root "%OUT%"
if errorlevel 1 exit /b %ERRORLEVEL%
set /a RUN+=1
goto REPETITION_LOOP
:REPETITIONS_DONE
echo All 4 repetitions finished for family wt.
exit /b 0
