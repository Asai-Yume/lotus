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
REM autofj_overlap: Top-1 Semantic Retrieval + LLM Verification baseline.
REM
REM Run 0 completed under outputs\lotus_top1_llm_verify is intentionally
REM left untouched. This launcher runs four additional independent
REM repetitions (run1..run4), each into its own isolated output root.
REM ######################################################################
set "RUN=1"
:REPETITION_LOOP
if %RUN% GTR 4 goto REPETITIONS_DONE
set "OUT=%CD%\outputs\lotus_top1_llm_verify_run%RUN%"
echo.
echo ============================================================
echo Family: autofj_overlap  -  repetition run%RUN% of 4  -  output root: %OUT%
echo ============================================================
python "%RUNNER%" --family autofj_overlap --dataset Amphibian_f025 --dataset Amphibian_f050 --dataset Amphibian_f075 --dataset Amphibian_f100 --dataset Artwork_f025 --dataset Artwork_f050 --dataset Artwork_f075 --dataset Artwork_f100 --dataset BasketballTeam_f025 --dataset BasketballTeam_f050 --dataset BasketballTeam_f075 --dataset BasketballTeam_f100 --dataset Case_f025 --dataset Case_f050 --dataset Case_f075 --dataset Case_f100 --dataset ClericalAdministrativeRegion_f025 --dataset ClericalAdministrativeRegion_f050 --dataset ClericalAdministrativeRegion_f075 --dataset ClericalAdministrativeRegion_f100 --dataset Device_f025 --dataset Device_f050 --dataset Device_f075 --dataset Device_f100 --dataset Drug_f025 --dataset Drug_f050 --dataset Drug_f075 --dataset Drug_f100 --dataset Election_f025 --dataset Election_f050 --dataset Election_f075 --dataset Election_f100 --dataset GivenName_f025 --dataset GivenName_f050 --dataset GivenName_f075 --dataset GivenName_f100 --dataset GovernmentAgency_f025 --dataset GovernmentAgency_f050 --dataset GovernmentAgency_f075 --dataset GovernmentAgency_f100 --dataset MemberOfParliament_f025 --dataset MemberOfParliament_f050 --dataset MemberOfParliament_f075 --dataset MemberOfParliament_f100 --dataset MotorsportSeason_f025 --dataset MotorsportSeason_f050 --dataset MotorsportSeason_f075 --dataset MotorsportSeason_f100 --dataset Museum_f025 --dataset Museum_f050 --dataset Museum_f075 --dataset Museum_f100 --dataset RailwayLine_f025 --dataset RailwayLine_f050 --dataset RailwayLine_f075 --dataset RailwayLine_f100 --dataset ShoppingMall_f025 --dataset ShoppingMall_f050 --dataset ShoppingMall_f075 --dataset ShoppingMall_f100 --dataset SoccerLeague_f025 --dataset SoccerLeague_f050 --dataset SoccerLeague_f075 --dataset SoccerLeague_f100 --dataset Song_f025 --dataset Song_f050 --dataset Song_f075 --dataset Song_f100 --dataset SportsLeague_f025 --dataset SportsLeague_f050 --dataset SportsLeague_f075 --dataset SportsLeague_f100 --dataset Tournament_f025 --dataset Tournament_f050 --dataset Tournament_f075 --dataset Tournament_f100 --dataset Wrestler_f025 --dataset Wrestler_f050 --dataset Wrestler_f075 --dataset Wrestler_f100 --embedding-batch-size 64 --llm-batch-size 8 --max-retries 5 --retry-backoff-seconds 2.0 --continue-on-error %MODE% --output-root "%OUT%"
if errorlevel 1 exit /b %ERRORLEVEL%
set /a RUN+=1
goto REPETITION_LOOP
:REPETITIONS_DONE
echo All 4 repetitions finished for family autofj_overlap.
exit /b 0
