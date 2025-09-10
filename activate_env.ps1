# PowerShell script to activate moodeng_env virtual environment
Write-Host "Activating moodeng_env virtual environment..." -ForegroundColor Green
& ".\moodeng_env\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Virtual environment activated!" -ForegroundColor Green
Write-Host "You can now run: python model_training_enhanced.py" -ForegroundColor Yellow
Write-Host ""
