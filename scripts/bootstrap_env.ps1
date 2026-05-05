Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "Installing pinned dependencies from requirements.txt..."
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Environment setup complete."
Write-Host "Next, run:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\\run_demo.ps1"
Write-Host "or:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\\run_streamlit.ps1"
