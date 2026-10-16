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
REM autofj: Top-1 Semantic Retrieval + LLM Verification baseline.
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
echo Family: autofj  -  repetition run%RUN% of 4  -  output root: %OUT%
echo ============================================================
python "%RUNNER%" --family autofj --dataset Amphibian --dataset ArtificialSatellite --dataset Artwork --dataset Award --dataset BasketballTeam --dataset Case --dataset ChristianBishop --dataset ClericalAdministrativeRegion --dataset Country --dataset Device --dataset Drug --dataset Election --dataset Enzyme --dataset EthnicGroup --dataset FootballLeagueSeason --dataset FootballMatch --dataset Galaxy --dataset GivenName --dataset GovernmentAgency --dataset HistoricBuilding --dataset Hospital --dataset Legislature --dataset Magazine --dataset MemberOfParliament --dataset Monarch --dataset MotorsportSeason --dataset Museum --dataset NationalFootballLeagueSeason --dataset NaturalEvent --dataset NCAATeamSeason --dataset Noble --dataset PoliticalParty --dataset Race --dataset RailwayLine --dataset Reptile --dataset RugbyLeague --dataset ShoppingMall --dataset SoccerClubSeason --dataset SoccerLeague --dataset SoccerTournament --dataset Song --dataset SportFacility --dataset SportsLeague --dataset Stadium --dataset TelevisionStation --dataset TennisTournament --dataset Tournament --dataset UnitOfWork --dataset Venue --dataset Wrestler --embedding-batch-size 64 --llm-batch-size 8 --max-retries 5 --retry-backoff-seconds 2.0 --continue-on-error %MODE% --output-root "%OUT%"
if errorlevel 1 exit /b %ERRORLEVEL%
set /a RUN+=1
goto REPETITION_LOOP
:REPETITIONS_DONE
echo All 4 repetitions finished for family autofj.
exit /b 0
