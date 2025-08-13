# TweetTaglish Twitter Extractor - PowerShell Script
# This script runs the Twitter extraction tool

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TweetTaglish Twitter Extractor" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and try again" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    try {
        python -m venv .venv
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & .\.venv\Scripts\Activate.ps1
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install requirements if needed
if (-not (Test-Path ".venv\Lib\site-packages\apify_client")) {
    Write-Host "Installing requirements..." -ForegroundColor Yellow
    try {
        pip install -r requirements.txt
        Write-Host "✓ Requirements installed" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Failed to install requirements" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check if APIFY_TOKEN is set
if (-not $env:APIFY_TOKEN) {
    Write-Host ""
    Write-Host "WARNING: APIFY_TOKEN environment variable is not set" -ForegroundColor Yellow
    Write-Host "Please set it before running the extractor:" -ForegroundColor Yellow
    Write-Host "  `$env:APIFY_TOKEN = 'your_token_here'" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Or run this command to set it temporarily:" -ForegroundColor Yellow
    Write-Host "  `$env:APIFY_TOKEN = 'your_token_here'; .\run_extractor.ps1" -ForegroundColor Cyan
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Run the test setup first
Write-Host "Running setup test..." -ForegroundColor Yellow
try {
    python scripts\test_setup.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Setup test failed. Please fix the issues above." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} catch {
    Write-Host "ERROR: Failed to run setup test" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Setup test passed! Starting extraction..." -ForegroundColor Green
Write-Host ""

# Run the Twitter extractor
try {
    python scripts\twitter_extractor.py
    Write-Host ""
    Write-Host "Extraction completed!" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Extraction failed" -ForegroundColor Red
}

Read-Host "Press Enter to exit"
