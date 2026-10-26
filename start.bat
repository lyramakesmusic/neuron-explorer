@echo off
rem Starts the neuron explorer: http://localhost:8791, and on this machine's Tailscale IP when Tailscale is up.
rem Whoever reaches the Tailscale address gets the whole API, the agent's shell included.
cd /d "%~dp0"
set "EXPLORER_HOST="
for /f %%i in ('tailscale ip -4 2^>nul ^| findstr /r "^100\."') do set "EXPLORER_HOST=%%i"
echo Local:     http://localhost:8791
if defined EXPLORER_HOST (echo Tailscale: http://%EXPLORER_HOST%:8791) else (echo Tailscale is not running: local only)
echo Ctrl+C stops the server.
echo.
uv run python explorer\server.py
pause
