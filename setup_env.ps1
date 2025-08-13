# Environment Setup Script for TweetTaglish
# Run this script to set required environment variables

Write-Host "TweetTaglish Environment Setup" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green
Write-Host ""

# Check if virtual environment is activated
if (-not (Test-Path ".venv")) {
    Write-Host "❌ Virtual environment not found. Please run 'python -m venv .venv' first." -ForegroundColor Red
    exit 1
}

# Set environment variables for current session
Write-Host "Setting environment variables for current session..." -ForegroundColor Yellow

# You need to replace these with your actual API tokens
$env:APIFY_TOKEN = "your_apify_token_here"
$env:X_BEARER_TOKEN = "your_twitter_bearer_token_here"

Write-Host ""
Write-Host "⚠️  IMPORTANT: You need to set your actual API tokens!" -ForegroundColor Red
Write-Host ""
Write-Host "To get your Apify token:" -ForegroundColor Cyan
Write-Host "  1. Go to https://console.apify.com/account/integrations" -ForegroundColor Cyan
Write-Host "  2. Copy your API token" -ForegroundColor Cyan
Write-Host "  3. Replace 'your_apify_token_here' in this script" -ForegroundColor Cyan
Write-Host ""
Write-Host "To get your Twitter Bearer Token (optional):" -ForegroundColor Cyan
Write-Host "  1. Go to https://developer.twitter.com/en/portal/dashboard" -ForegroundColor Cyan
Write-Host "  2. Create an app and get your Bearer Token" -ForegroundColor Cyan
Write-Host "  3. Replace 'your_twitter_bearer_token_here' in this script" -ForegroundColor Cyan
Write-Host ""
Write-Host "After setting the tokens, run this script again to activate them." -ForegroundColor Yellow
Write-Host ""
Write-Host "Current environment variables:" -ForegroundColor Green
Write-Host "  APIFY_TOKEN: $env:APIFY_TOKEN" -ForegroundColor White
Write-Host "  X_BEARER_TOKEN: $env:X_BEARER_TOKEN" -ForegroundColor White
Write-Host ""
Write-Host "Now you can run: python scripts\test_setup.py" -ForegroundColor Green
