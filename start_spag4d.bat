@echo off
REM SPAG-4D Launcher
REM Starts the web UI on port 7860

cd /d "%~dp0"

echo ========================================
echo   SPAG-4D: 360 to Gaussian Splat
echo ========================================
echo.

REM Kill any existing process on port 7860
echo Checking for existing servers on port 7860...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7860 ^| findstr LISTENING') do (
    echo Killing existing process PID: %%a
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Test if torch works
python -c "import torch" 2>nul
if errorlevel 1 (
    echo PyTorch not working. Please check installation.
    pause
    exit /b 1
)

echo Starting web server on http://localhost:7860
echo Press Ctrl+C to stop
echo.

python -m spag4d.cli serve --port 7860

pause
