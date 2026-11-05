# Builds the explorer's per-model caches on this machine (explorer/cache/models/<model>/):
#   .\build_cache_windows.ps1            (Qwen3.5-2B)      .\build_cache_windows.ps1 0.8B
# Steps already done are skipped, so it can be rerun after an interruption. Log: build_cache_<size>.log
param([string]$Size = "2B")

$env:PYTHONUTF8 = "1"
$env:EXPLORER_MODEL = "Qwen/Qwen3.5-$Size"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
Set-Location $PSScriptRoot
$env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
$name = "qwen3.5-$($Size.ToLower())"
$cache = if ($name -eq "qwen3.5-4b") { "explorer/cache" } else { "explorer/cache/models/$name" }

function Step($label, $out, [scriptblock]$run) {
    if ($out -and (Test-Path $out)) { Write-Host "== $label - already there ($out)"; return }
    Write-Host "== $label"; $t = Get-Date
    & $run
    if ($LASTEXITCODE -ne 0) { Write-Host "!! $label failed (exit $LASTEXITCODE)"; exit 1 }
    Write-Host ("   done in {0:N0} min" -f ((Get-Date) - $t).TotalMinutes)
}

Step "logit lens index" "$cache/lens_index.npz" {
    uv run python -c "import sys; sys.path.insert(0, 'explorer'); from runtime import Runtime, CACHE; import os; Runtime('stats').build_lens_index(os.path.join(CACHE, 'lens_index.npz'))" }
Step "activation ranges (10k web tokens)" "$cache/act_scale.npz" {
    uv run python -c "import sys; sys.path.insert(0, 'explorer'); from runtime import Runtime, CACHE; import os; Runtime('stats').build_act_scale(os.path.join(CACHE, 'act_scale.npz'))" }
Step "steer lens, every layer" "$cache/steer_index.npz" {
    uv run python sweep/steer_lens.py --neurons-per-batch 48
    if ($LASTEXITCODE -eq 0) { uv run python explorer/build_steer_index.py } }
Step "connections" "$cache/connections.npz" { uv run python explorer/build_connections.py }
Step "map layout (writes)" "$cache/layout_writes.npz" { uv run python explorer/build_layout.py --kind writes }
Write-Host "== all caches built in $cache"
