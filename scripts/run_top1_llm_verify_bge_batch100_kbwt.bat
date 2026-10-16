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
REM kbwt: CORRECTED fair baseline (BGE retrieval + Any-Join batch verification).
REM
REM   retrieval embedding : BAAI/bge-base-en-v1.5 (local SentenceTransformersRM)
REM   verification        : anyjoin_batch, 100 candidate pairs per logical request
REM   temperature         : 0.0 (supplied explicitly)
REM   K                   : 1 (unchanged Top-1 retrieval)
REM
REM Five corrected repetitions: run0 (base root) plus run1..run4.
REM Historical baseline outputs under outputs\lotus_top1_llm_verify* are
REM never written by this launcher.
REM kbwt: all 81 datasets (CountryToTwoLettersISOCode included; literal 'NA'
REM is preserved via keep_default_na=False in the loader).
REM ######################################################################
set "RUN=0"
:RUN_LOOP
if %RUN% GTR 4 goto RUNS_DONE
set "OUT=%CD%\outputs\lotus_top1_llm_verify_bge_batch100"
if %RUN% GTR 0 set "OUT=%OUT%_run%RUN%"
echo.
echo ============================================================
echo Family: kbwt  -  corrected repetition run%RUN% (0-4)  -  output root: %OUT%
echo ============================================================
python "%RUNNER%" --family kbwt --dataset AirportcodeToCity --dataset AnimalToGenusSpeciesCommonNames --dataset ArenaTocurrentSoccerClubs --dataset AsciiToUnicode --dataset AthletesHoldingWorldRecordsPerCountry --dataset Author2Novels --dataset AwardToArtist --dataset BankToCity --dataset BankToSwiftCode --dataset CUSIPToCompanyName --dataset CharToAscii --dataset City2LongLat --dataset City2TallBuildings --dataset CityToCountry --dataset CityToZIp --dataset CompanyToBloombergID --dataset CompanyToCeo --dataset CompanyToIndustry --dataset CompanyToTicker --dataset Continent2Countries --dataset CountriesToContinents --dataset Country2Mountains7k --dataset Country2UnescoSites --dataset CountryCodeToCountry --dataset CountryToAreaSqKM --dataset CountryToCitizen --dataset CountryToCurrencies --dataset CountryToLanguage --dataset CountryToThreeLettersISOCode --dataset CountryToTwoLettersISOCode --dataset Driver2Champioships --dataset ElementToBP --dataset ElementToSymbol --dataset EnglishToGerman --dataset GregorianToHijri --dataset HexCodeToRGB --dataset IBANToBankName --dataset IPToCountry --dataset IPToDomain --dataset ISBNToAuthor --dataset ISBNToPublisher --dataset ISBNToTitle --dataset ISSNToTitle --dataset Lake2Countries --dataset ManufacturerToProducts --dataset Mountain7k2Countries --dataset MountainsOver7k2Range --dataset MountainsOver7k2feet --dataset MountainsOver7k2meters --dataset Movie2Cast --dataset Movie2year --dataset NBAClub2Roster --dataset PatentIDToName --dataset Person2Spouses --dataset PoundsToKg --dataset RGBToColor --dataset RegionToAreaCode-Single --dataset SoccerPlayer2Birthdate --dataset SoccerPlayer2Birthplace --dataset SoccerPlayer2NationalTeam --dataset SoccerPlayer2club --dataset SoccerplayerToAllclub --dataset StateToAbbrv --dataset SwiftToBankName --dataset SymbolToCompanyName --dataset TeamToCoach --dataset TeamToManager --dataset USStandardToMetric --dataset UnescoSite2Location --dataset UniversityTostate --dataset WorldRecordToAthlete --dataset Worldcupwinner2year --dataset YearNobelPrizeFieldLaureate --dataset YearNobelPrizeLaureate --dataset ZipToCity --dataset ZipToState --dataset airportToCountry --dataset bankToCountryDistinct --dataset fahrenheitToCelcius --dataset shoesizeUSEUR --dataset soccerTeam2Arena --embedding-backend sentence_transformers --embedding-model BAAI/bge-base-en-v1.5 --verification-mode anyjoin_batch --verification-batch-size 100 --omit-lm-completion-budget --embedding-batch-size 64 --llm-batch-size 8 --max-retries 5 --retry-backoff-seconds 2.0 --continue-on-error %MODE% --output-root "%OUT%"
if errorlevel 1 exit /b %ERRORLEVEL%
set /a RUN+=1
goto RUN_LOOP
:RUNS_DONE
echo All 5 corrected repetitions finished for family kbwt.
exit /b 0
