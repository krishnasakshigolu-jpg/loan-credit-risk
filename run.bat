@echo off
rem Double-click to start the Loan Risk dashboard. The first run sets up Python and trains the model.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Setting up for the first time. This takes a minute or two.
  python -m venv .venv || goto :nopython
  ".venv\Scripts\python" -m pip install --quiet -r requirements.txt || goto :fail
)

echo Starting the services. The dashboard opens at http://127.0.0.1:8000  (close this window to stop)
rem dev.py trains on the first run, starts scoring, insights and the gateway, then opens the browser
".venv\Scripts\python" scripts\dev.py --open
goto :eof

:nopython
echo Python 3.10 or newer is needed. Install it from python.org, tick "Add python.exe to PATH", then run this again.
pause
exit /b 1

:fail
echo Setup failed. Delete the .venv folder and run this again, or follow "Run it" in README.md.
pause
exit /b 1
