@echo off
echo Starting Filipino Tweet Preprocessing from Excel File...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install requirements if needed
echo Installing required packages...
pip install -r requirements.txt

REM Run the preprocessing script
echo.
echo Running preprocessing script on Excel file...
echo Processing 'tweets_split_id.xlsx' - only valid tweets (Status = 1)
python preprocess_tweets.py

echo.
echo Preprocessing completed!
echo Check 'tweets_split_id_processed.xlsx' for results
pause
