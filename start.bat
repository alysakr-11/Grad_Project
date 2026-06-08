@echo off
cd /d "%~dp0"
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Starting Smart Business Analytics Dashboard...
echo (If you see a blank page, wait a few seconds and refresh)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
pause
