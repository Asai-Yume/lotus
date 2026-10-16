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
REM ss: CORRECTED fair baseline (BGE retrieval + Any-Join batch verification).
REM
REM   retrieval embedding : BAAI/bge-base-en-v1.5 (local SentenceTransformersRM)
REM   verification        : anyjoin_batch, 100 candidate pairs per logical request
REM   temperature         : 0.0 (supplied explicitly)
REM   K                   : 1 (unchanged Top-1 retrieval)
REM
REM Five corrected repetitions: run0 (base root) plus run1..run4.
REM Historical baseline outputs under outputs\lotus_top1_llm_verify* are
REM never written by this launcher.
REM ss: two python invocations per repetition (104 normal datasets, then 4
REM FF-name-combine variants with --gt-source-col source-value /
REM --gt-target-col target-value); both target the SAME repetition root.
REM ######################################################################
set "RUN=0"
:RUN_LOOP
if %RUN% GTR 4 goto RUNS_DONE
set "OUT=%CD%\outputs\lotus_top1_llm_verify_bge_batch100"
if %RUN% GTR 0 set "OUT=%OUT%_run%RUN%"
echo.
echo ============================================================
echo Family: ss  -  corrected repetition run%RUN% (0-4)  -  output root: %OUT%
echo ============================================================
python "%RUNNER%" --family ss --dataset FF-bikes --dataset FF-bikes-long --dataset FF-bikes-long-repeat --dataset FF-bikes-small --dataset FF-dr-name --dataset FF-dr-name-long --dataset FF-dr-name-long-repeat --dataset FF-dr-name-small --dataset FF-firstname --dataset FF-firstname-long --dataset FF-firstname-long-repeat --dataset FF-firstname-small --dataset FF-initials --dataset FF-initials-long --dataset FF-initials-long-repeat --dataset FF-initials-small --dataset FF-lastname --dataset FF-lastname-long --dataset FF-lastname-long-repeat --dataset FF-lastname-small --dataset FF-name-combine-2 --dataset FF-name-combine-2-long --dataset FF-name-combine-2-long-repeat --dataset FF-name-combine-2-short --dataset FF-name-combine-3 --dataset FF-name-combine-3-long --dataset FF-name-combine-3-long-repeat --dataset FF-name-combine-3-short --dataset FF-name-combine-4 --dataset FF-name-combine-4-long --dataset FF-name-combine-4-long-repeat --dataset FF-name-combine-4-short --dataset FF-phone --dataset FF-phone-1 --dataset FF-phone-1-long --dataset FF-phone-1-long-repeat --dataset FF-phone-1-short --dataset FF-phone-10 --dataset FF-phone-10-long --dataset FF-phone-10-long-repeat --dataset FF-phone-10-short --dataset FF-phone-2 --dataset FF-phone-2-long --dataset FF-phone-2-long-repeat --dataset FF-phone-2-short --dataset FF-phone-3 --dataset FF-phone-3-long --dataset FF-phone-3-long-repeat --dataset FF-phone-3-short --dataset FF-phone-4 --dataset FF-phone-4-long --dataset FF-phone-4-long-repeat --dataset FF-phone-4-short --dataset FF-phone-5 --dataset FF-phone-5-long --dataset FF-phone-5-long-repeat --dataset FF-phone-5-short --dataset FF-phone-6 --dataset FF-phone-6-long --dataset FF-phone-6-long-repeat --dataset FF-phone-6-short --dataset FF-phone-7 --dataset FF-phone-7-long --dataset FF-phone-7-long-repeat --dataset FF-phone-7-short --dataset FF-phone-8 --dataset FF-phone-8-long --dataset FF-phone-8-long-repeat --dataset FF-phone-8-short --dataset FF-phone-9 --dataset FF-phone-9-long --dataset FF-phone-9-long-repeat --dataset FF-phone-9-short --dataset FF-phone-long --dataset FF-phone-long-repeat --dataset FF-phone-short --dataset FF-reverse-name --dataset FF-reverse-name-long --dataset FF-reverse-name-long-repeat --dataset FF-reverse-name-short --dataset FF-univ-1 --dataset FF-univ-1-long --dataset FF-univ-1-long-repeat --dataset FF-univ-1-short --dataset FF-univ-2 --dataset FF-univ-2-long --dataset FF-univ-2-long-repeat --dataset FF-univ-2-short --dataset FF-univ-3 --dataset FF-univ-3-long --dataset FF-univ-3-long-repeat --dataset FF-univ-3-short --dataset FF-univ-4 --dataset FF-univ-4-long --dataset FF-univ-4-long-repeat --dataset FF-univ-4-short --dataset FF-univ-5 --dataset FF-univ-5-long --dataset FF-univ-5-long-repeat --dataset FF-univ-5-short --dataset FF-univ-6 --dataset FF-univ-6-long --dataset FF-univ-6-long-repeat --dataset FF-univ-6-short --embedding-backend sentence_transformers --embedding-model BAAI/bge-base-en-v1.5 --verification-mode anyjoin_batch --verification-batch-size 100 --omit-lm-completion-budget --embedding-batch-size 64 --llm-batch-size 8 --max-retries 5 --retry-backoff-seconds 2.0 --continue-on-error %MODE% --output-root "%OUT%"
if errorlevel 1 exit /b %ERRORLEVEL%

python "%RUNNER%" --family ss --dataset FF-name-combine --dataset FF-name-combine-long --dataset FF-name-combine-long-repeat --dataset FF-name-combine-short --gt-source-col source-value --gt-target-col target-value --embedding-backend sentence_transformers --embedding-model BAAI/bge-base-en-v1.5 --verification-mode anyjoin_batch --verification-batch-size 100 --omit-lm-completion-budget --embedding-batch-size 64 --llm-batch-size 8 --max-retries 5 --retry-backoff-seconds 2.0 --continue-on-error %MODE% --output-root "%OUT%"
if errorlevel 1 exit /b %ERRORLEVEL%
set /a RUN+=1
goto RUN_LOOP
:RUNS_DONE
echo All 5 corrected repetitions finished for family ss.
exit /b 0
