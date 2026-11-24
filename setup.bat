@echo off
REM One-time setup for Windows: installs FFmpeg + Python packages.
cd /d "%~dp0"

where python >nul 2>nul || (
  echo Python not found. Install Python 3.10+ from https://www.python.org/downloads/ ^(tick "Add to PATH"^) and run this again.
  pause & exit /b 1
)

where ffmpeg >nul 2>nul || (
  echo Installing FFmpeg via winget...
  winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
  echo If ffmpeg is still not found, close this window and open a new one.
)

if not exist .venv python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist .env copy .env.example .env
echo.
echo Setup done. Open .env and fill in your Instagram + (optional) Anthropic keys, then run start.bat
pause
