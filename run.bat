@echo off
setlocal ENABLEEXTENSIONS

REM ====== CONFIG ======
set "ENV_DIR=.venv"
set "REQUIREMENTS=requirements.txt"
set "MAIN_SCRIPT=main.py"
set "DOTENV=.env"
set "ENV_VAR=GOOGLE_API_KEY"

REM ====== 1. Check Python ======
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.11+ and try again.
    pause
    exit /b 1
)

REM ====== 2. Create venv if missing ======
if not exist "%ENV_DIR%\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment in %ENV_DIR%...
    python -m venv %ENV_DIR%
)

REM ====== 3. Activate venv ======
call "%ENV_DIR%\Scripts\activate.bat"

REM ====== 4. Upgrade pip ======
python -m pip install --upgrade pip

REM ====== 5. Install requirements ======
if exist "%REQUIREMENTS%" (
    echo [INFO] Installing requirements...
    pip install -r "%REQUIREMENTS%"
) else (
    echo [ERROR] requirements.txt not found.
    pause
    exit /b 1
)

REM ====== 6. Create .env if missing ======
if not exist "%DOTENV%" (
    echo [INFO] Creating .env with placeholder...
    echo %ENV_VAR%=your-api-key-here > "%DOTENV%"
    echo [INFO] Please edit .env to insert your Gemini API key before rerunning.
    pause
    exit /b 1
)

REM ====== 7. Check GOOGLE_API_KEY ======
set "API_KEY="

for /f "tokens=1,* delims==" %%A in ('type "%DOTENV%"') do (
    if /I "%%A"=="%ENV_VAR%" (
        set "API_KEY=%%B"
    )
)

if not defined API_KEY (
    echo [ERROR] GOOGLE_API_KEY not found in .env
    pause
    exit /b 1
)

echo %API_KEY% | findstr /C:"your-api-key-here" >nul
if %errorlevel% equ 0 (
    echo [ERROR] GOOGLE_API_KEY is still set to placeholder.
    echo [ACTION] Opening .env in Notepad. Please update the key.
    start /wait notepad "%DOTENV%"
    
    REM Re-parse after edit
    set "API_KEY="
    for /f "tokens=1,* delims==" %%A in ('type "%DOTENV%"') do (
        if /I "%%A"=="%ENV_VAR%" (
            set "API_KEY=%%B"
        )
    )
    echo %API_KEY% | findstr /C:"your-api-key-here" >nul
    if %errorlevel% equ 0 (
        echo [ERROR] Placeholder key still not changed. Exiting.
        pause
        exit /b 1
    )
)

echo [OK] GOOGLE_API_KEY found and valid.


REM ====== 8. Check for VB-Audio Cable ======
echo [INFO] Checking for VB-Audio Cable...
python -c "import sounddevice as sd; devices = [d['name'] for d in sd.query_devices()]; print('[OK]' if any('VB-AUDIO' in d.upper() or 'CABLE INPUT' in d.upper() for d in devices) else '[MISSING]')" > vbcheck.tmp

findstr "[MISSING]" vbcheck.tmp >nul
if %errorlevel% equ 0 (
    echo [ERROR] VB-Audio Cable not detected.
    echo Opening download page...
    start https://vb-audio.com/Cable/
    del vbcheck.tmp
    pause
    exit /b 1
)
del vbcheck.tmp
echo [OK] VB-Audio Cable detected.

REM ====== 9. Run the app ======
echo [INFO] Running %MAIN_SCRIPT%...
python "%MAIN_SCRIPT%"
pause
