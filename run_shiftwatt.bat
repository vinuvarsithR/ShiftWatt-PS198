@echo off
rem Optional: set GEMINI_API_KEY=your-key-here  (or paste the key in the app Settings page)
title ShiftWatt
cd /d "%~dp0"
echo Checking packages (first run takes a minute)...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo.
  echo Python was not found or the install failed.
  echo Install Python 3.10+ from python.org and tick "Add Python to PATH", then run this again.
  pause
  exit /b
)
echo Starting ShiftWatt... your browser will open at http://localhost:8501
echo Keep this window open. Close it to stop the app.
python -m streamlit run app.py
pause
