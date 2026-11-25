@echo off
rem Double-click to start the Loan Risk dashboard. The first run sets up Python and trains the model.
setlocal
cd /d "%~dp0"
set PORT=8000

if not exist ".venv\Scripts\python.exe" (
  echo Setting up for the first time. This takes a minute or two.
  python -m venv .venv || goto :nopython
  ".venv\Scripts\python" -m pip install --quiet -r requirements.txt || goto :fail
  rem train once so the saved model matches the installed scikit-learn
  ".venv\Scripts\python" train.py || goto :fail
)

echo Starting the dashboard at http://127.0.0.1:%PORT%  (close this window to stop it)
rem open the browser a moment later, once the server is listening
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start "" http://127.0.0.1:%PORT%"
".venv\Scripts\python" -m uvicorn app:app --port %PORT%
goto :eof

:nopython
echo Python 3.10 or newer is needed. Install it from python.org, tick "Add python.exe to PATH", then run this again.
pause
exit /b 1

:fail
echo Setup failed. Delete the .venv folder and run this again, or follow "Run it" in README.md.
pause
exit /b 1
