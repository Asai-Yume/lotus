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
REM wt: CORRECTED fair baseline (BGE retrieval + Any-Join batch verification).
REM
REM   retrieval embedding : BAAI/bge-base-en-v1.5 (local SentenceTransformersRM)
REM   verification        : anyjoin_batch, 100 candidate pairs per logical request
REM   temperature         : 0.0 (supplied explicitly)
REM   K                   : 1 (unchanged Top-1 retrieval)
REM
REM Five corrected repetitions: run0 (base root) plus run1..run4.
REM Historical baseline outputs under outputs\lotus_top1_llm_verify* are
REM never written by this launcher.
REM wt: all 32 datasets; join-value columns resolved from each dataset's
REM rows.txt. No manual join-column overrides; empty-row policy unchanged.
REM ######################################################################
set "RUN=0"
:RUN_LOOP
if %RUN% GTR 4 goto RUNS_DONE
set "OUT=%CD%\outputs\lotus_top1_llm_verify_bge_batch100"
if %RUN% GTR 0 set "OUT=%OUT%_run%RUN%"
echo.
echo ============================================================
echo Family: wt  -  corrected repetition run%RUN% (0-4)  -  output root: %OUT%
echo ============================================================
python "%RUNNER%" --family wt --dataset AJ-beatles-songs --dataset AJ-california-govs-1 --dataset AJ-california-govs-2 --dataset AJ-chinese-provinces --dataset AJ-christmas-songs-1 --dataset AJ-christmas-songs-2 --dataset AJ-duke-cs-profs --dataset AJ-fruits-1 --dataset AJ-fruits-2 --dataset AJ-fsu-name-to-username --dataset AJ-k12-name-to-email --dataset AJ-new-york-govs-1 --dataset AJ-new-york-govs-2 --dataset AJ-new-york-govs-3 --dataset AJ-new-york-govs-4 --dataset AJ-new-york-govs-5 --dataset AJ-park-to-state-1 --dataset AJ-park-to-state-2 --dataset AJ-sharif-email-to-url --dataset AJ-sharif-username-to-email --dataset AJ-texas-govs-1 --dataset AJ-texas-govs-2 --dataset AJ-uk-prime-ministers --dataset AJ-us-cities --dataset AJ-us-presidents-1 --dataset AJ-us-presidents-2 --dataset AJ-us-presidents-3 --dataset AJ-us-presidents-4 --dataset AJ-us-presidents-5 --dataset AJ-us-presidents-6 --dataset AJ-vegetables --dataset names200 --embedding-backend sentence_transformers --embedding-model BAAI/bge-base-en-v1.5 --verification-mode anyjoin_batch --verification-batch-size 100 --omit-lm-completion-budget --embedding-batch-size 64 --llm-batch-size 8 --max-retries 5 --retry-backoff-seconds 2.0 --continue-on-error %MODE% --output-root "%OUT%"
if errorlevel 1 exit /b %ERRORLEVEL%
set /a RUN+=1
goto RUN_LOOP
:RUNS_DONE
echo All 5 corrected repetitions finished for family wt.
exit /b 0
