@echo off
rem ULTRON one-click launcher (Windows) - the terminal version.
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Install Python 3.10+ from https://python.org
  echo ^(tick "Add python.exe to PATH" during install^), then run this again.
  pause & exit /b
)

if not exist venv (
  echo Creating environment ^(first run only^)...
  python -m venv venv
)

echo Installing dependencies...
venv\Scripts\python.exe -m pip install -q --upgrade pip >nul 2>&1
venv\Scripts\python.exe -m pip install -q -r requirements.txt

if defined ANTHROPIC_API_KEY goto run
if exist apikey.txt (
  set /p ANTHROPIC_API_KEY=<apikey.txt
  goto run
)
echo.
set /p ANTHROPIC_API_KEY=Paste your Anthropic API key (sk-ant-...):
> apikey.txt echo %ANTHROPIC_API_KEY%
echo (saved to apikey.txt - it won't ask again)

:run
echo.
venv\Scripts\python.exe cli.py %*
pause
