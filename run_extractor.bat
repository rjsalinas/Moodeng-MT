@echo off
REM TweetTaglish Twitter Extractor - Windows Batch File
REM This script runs the Twitter extraction tool

echo ========================================
echo TweetTaglish Twitter Extractor
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install requirements if needed
if not exist ".venv\Lib\site-packages\apify_client" (
    echo Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
)

REM Check if APIFY_TOKEN is set
if "%APIFY_TOKEN%"=="" (
    echo.
    echo WARNING: APIFY_TOKEN environment variable is not set
    echo Please set it before running the extractor:
    echo   set APIFY_TOKEN=your_token_here
    echo.
    echo Or run this command to set it temporarily:
    echo   set APIFY_TOKEN=your_token_here && run_extractor.bat
    echo.
    pause
    exit /b 1
)

REM Run the test setup first
echo Running setup test...
python scripts\test_setup.py
if errorlevel 1 (
    echo.
    echo Setup test failed. Please fix the issues above.
    pause
    exit /b 1
)

echo.
echo Setup test passed! Starting extraction...
echo.

REM Run the Twitter extractor
python scripts\twitter_extractor.py

echo.
echo Extraction completed!
pause
