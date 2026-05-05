Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"

streamlit run apps/streamlit_app.py
