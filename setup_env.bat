@echo off
REM Environment Setup Script for TweetTaglish
REM Run this script to set required environment variables

echo TweetTaglish Environment Setup
echo =============================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found. Please run 'python -m venv .venv' first.
    pause
    exit /b 1
)

REM Set environment variables for current session
echo Setting environment variables for current session...
echo.

REM You need to replace these with your actual API tokens
set APIFY_TOKEN=your_apify_token_here
set X_BEARER_TOKEN=your_twitter_bearer_token_here

echo.
echo ⚠️  IMPORTANT: You need to set your actual API tokens!
echo.
echo To get your Apify token:
echo   1. Go to https://console.apify.com/account/integrations
echo   2. Copy your API token
echo   3. Replace 'your_apify_token_here' in this script
echo.
echo To get your Twitter Bearer Token (optional):
echo   1. Go to https://developer.twitter.com/en/portal/dashboard
echo   2. Create an app and get your Bearer Token
echo   3. Replace 'your_twitter_bearer_token_here' in this script
echo.
echo After setting the tokens, run this script again to activate them.
echo.
echo Current environment variables:
echo   APIFY_TOKEN: %APIFY_TOKEN%
echo   X_BEARER_TOKEN: %X_BEARER_TOKEN%
echo.
echo Now you can run: python scripts\test_setup.py
echo.
pause
