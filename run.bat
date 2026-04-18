@echo off
setlocal

echo.
echo =====================================================
echo  NIFTY 50 Stock Prediction System - Setup ^& Run
echo =====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to CHECK "Add python.exe to PATH" during installation
    pause
    exit /b 1
)

echo [OK] Python found
python --version

set "SETUP_ONLY=0"
if /I "%~1"=="--setup-only" set "SETUP_ONLY=1"

REM Check if venv exists
if not exist "venv" (
    echo.
    echo [*] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Activate virtual environment
echo.
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies only when requirements change
set "REQ_HASH_FILE=.requirements.sha256"
set "CURRENT_REQ_HASH="
for /f %%H in ('powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 requirements.txt).Hash"') do set "CURRENT_REQ_HASH=%%H"

if not defined CURRENT_REQ_HASH (
    echo [ERROR] Failed to read requirements hash
    pause
    exit /b 1
)

set "SAVED_REQ_HASH="
if exist "%REQ_HASH_FILE%" set /p SAVED_REQ_HASH=<"%REQ_HASH_FILE%"

if /I "%SAVED_REQ_HASH%"=="%CURRENT_REQ_HASH%" goto deps_up_to_date

echo.
echo [*] Installing dependencies (first run or requirements changed)...
python -m pip install --no-cache-dir -q -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    echo Try running: python -m pip install --upgrade pip
    pause
    exit /b 1
)
>"%REQ_HASH_FILE%" echo %CURRENT_REQ_HASH%
echo [OK] Dependencies installed
goto deps_done

:deps_up_to_date
echo [OK] Dependencies already up-to-date (skipping install)

:deps_done

if "%SETUP_ONLY%"=="1" (
    echo [OK] Setup completed. Exiting because --setup-only was passed.
    exit /b 0
)

REM Start the application
echo.
echo =====================================================
echo  Starting NIFTY 50 Stock Lab...
echo =====================================================
echo.
echo.

echo.
echo [+] Application will be available at: http://localhost:8501
echo [+] Press Ctrl+C to stop the server
echo.

python -m uvicorn frontend.web_app:app --host 0.0.0.0 --port 8501 --reload

pause
