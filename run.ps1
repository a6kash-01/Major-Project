Set-Location -Path $PSScriptRoot
if (Test-Path ".\.venv\Scripts\python.exe") {
    & ".\.venv\Scripts\python.exe" run.py
} elseif (Test-Path "$env:USERPROFILE\anaconda3\python.exe") {
    & "$env:USERPROFILE\anaconda3\python.exe" run.py
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python run.py
} else {
    Write-Host "Python not found. Please install Python or Anaconda." -ForegroundColor Red
}
