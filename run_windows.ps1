# Starts the neuron explorer on Windows:  .\run_windows.ps1            (Qwen3.5-2B, http://localhost:8791)
#                                         .\run_windows.ps1 0.8B       (or 4B, with -NoModel for the 4B cache only)
param([string]$Size = "2B", [switch]$NoModel)

$env:PYTHONUTF8 = "1"                       # vocab.json and prompts hold non-Latin text; Windows defaults to cp1250
$env:EXPLORER_MODEL = "Qwen/Qwen3.5-$Size"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"   # less fragmentation on a 6 GB card
if ($NoModel) { $env:EXPLORER_NO_MODEL = "1" } else { Remove-Item Env:EXPLORER_NO_MODEL -ErrorAction SilentlyContinue }

Set-Location $PSScriptRoot
# a terminal opened before uv was installed doesn't see it yet: reload PATH, else use the project's venv directly
$env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
if (Get-Command uv -ErrorAction SilentlyContinue) { uv run python explorer/server.py }
elseif (Test-Path .venv\Scripts\python.exe) { .\.venv\Scripts\python.exe explorer/server.py }
else { Write-Host "uv not found: install it with  winget install astral-sh.uv  and run  uv sync" }
