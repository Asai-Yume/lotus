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
REM autofj: CORRECTED fair baseline (BGE retrieval + Any-Join batch verification).
REM
REM   retrieval embedding : BAAI/bge-base-en-v1.5 (local SentenceTransformersRM)
REM   verification        : anyjoin_batch, 100 candidate pairs per logical request
REM   temperature         : 0.0 (supplied explicitly)
REM   K                   : 1 (unchanged Top-1 retrieval)
REM
REM Five corrected repetitions: run0 (base root) plus run1..run4.
REM Historical baseline outputs under outputs\lotus_top1_llm_verify* are
REM never written by this launcher.
REM ######################################################################
set "RUN=0"
:RUN_LOOP
if %RUN% GTR 4 goto RUNS_DONE
set "OUT=%CD%\outputs\lotus_top1_llm_verify_bge_batch100"
if %RUN% GTR 0 set "OUT=%OUT%_run%RUN%"
echo.
echo ============================================================
echo Family: autofj  -  corrected repetition run%RUN% (0-4)  -  output root: %OUT%
echo ============================================================
python "%RUNNER%" --family autofj --dataset Amphibian --dataset ArtificialSatellite --dataset Artwork --dataset Award --dataset BasketballTeam --dataset Case --dataset ChristianBishop --dataset ClericalAdministrativeRegion --dataset Country --dataset Device --dataset Drug --dataset Election --dataset Enzyme --dataset EthnicGroup --dataset FootballLeagueSeason --dataset FootballMatch --dataset Galaxy --dataset GivenName --dataset GovernmentAgency --dataset HistoricBuilding --dataset Hospital --dataset Legislature --dataset Magazine --dataset MemberOfParliament --dataset Monarch --dataset MotorsportSeason --dataset Museum --dataset NationalFootballLeagueSeason --dataset NaturalEvent --dataset NCAATeamSeason --dataset Noble --dataset PoliticalParty --dataset Race --dataset RailwayLine --dataset Reptile --dataset RugbyLeague --dataset ShoppingMall --dataset SoccerClubSeason --dataset SoccerLeague --dataset SoccerTournament --dataset Song --dataset SportFacility --dataset SportsLeague --dataset Stadium --dataset TelevisionStation --dataset TennisTournament --dataset Tournament --dataset UnitOfWork --dataset Venue --dataset Wrestler --embedding-backend sentence_transformers --embedding-model BAAI/bge-base-en-v1.5 --verification-mode anyjoin_batch --verification-batch-size 100 --omit-lm-completion-budget --embedding-batch-size 64 --llm-batch-size 8 --max-retries 5 --retry-backoff-seconds 2.0 --continue-on-error %MODE% --output-root "%OUT%"
if errorlevel 1 exit /b %ERRORLEVEL%
set /a RUN+=1
goto RUN_LOOP
:RUNS_DONE
echo All 5 corrected repetitions finished for family autofj.
exit /b 0
