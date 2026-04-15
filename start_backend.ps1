# Start the Astronomical Analysis Backend
Set-Location "$PSScriptRoot\backend"
Write-Host "[*] Activating Python 3.12 virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
Write-Host "[*] Starting FastAPI backend on http://localhost:8000 ..." -ForegroundColor Cyan
Write-Host "[*] API docs at http://localhost:8000/docs" -ForegroundColor Green
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
