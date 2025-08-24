Write-Host "Starting Filipino Tweet Preprocessing from Excel File..." -ForegroundColor Green
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion" -ForegroundColor Cyan
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install requirements if needed
Write-Host "Installing required packages..." -ForegroundColor Yellow
pip install -r requirements.txt

# Run the preprocessing script
Write-Host ""
Write-Host "Running preprocessing script on Excel file..." -ForegroundColor Yellow
Write-Host "Processing 'tweets_split_id.xlsx' - only valid tweets (Status = 1)" -ForegroundColor Cyan
python preprocess_tweets.py

Write-Host ""
Write-Host "Preprocessing completed!" -ForegroundColor Green
Write-Host "Check 'tweets_split_id_processed.xlsx' for results" -ForegroundColor Cyan
Read-Host "Press Enter to exit"
