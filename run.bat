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


REM ====== 8. Check for VB‑Audio Cable and advise on default device ======

:: The Python one‑liner prints:  inputName|outputName|true/false|isDefaultCorrect
for /f "tokens=1,2,3,4 delims=|" %%A in ('python -c "import sounddevice as sd, sys; di,do=sd.default.device; dev=sd.query_devices(); fmt=lambda i:dev[i]['name'] if i>=0 else 'N/A'; vb=any('VB-AUDIO' in d['name'].upper() or 'CABLE' in d['name'].upper() for d in dev); default_input=fmt(di).lower(); is_correct=('cable output' in default_input and 'vb-audio virtual' in default_input); print(f'{fmt(di)}|{fmt(do)}|{vb}|{is_correct}'.lower())"') do (
    set "DEF_IN=%%A"
    set "DEF_OUT=%%B"
    set "VB_OK=%%C"
    set "DEFAULT_CORRECT=%%D"
)

:: Abort if the driver itself is missing
if /I "%VB_OK%"=="false" (
    echo.
    echo [ERROR] VB‑Audio Cable driver not detected.
    echo Download and install it from: https://vb-audio.com/Cable/
    start https://vb-audio.com/Cable/
    pause
    exit /b 1
)

echo.
echo Default Recording Device : %DEF_IN%
echo Default Playback Device  : %DEF_OUT%
echo.

:: Check if default device is already correctly configured
if /I "%DEFAULT_CORRECT%"=="true" (
    echo [OK] Audio setup is correct! Default recording device is already set to VB-Audio Cable.
    echo [INFO] Continuing with current audio settings...
    goto :run_app
)

echo [INFO] Set your Recording (Input) device to:
echo     "CABLE Output (VB-Audio Virtual Cable)"
echo Keep your physical microphone (e.g. "Realtek Microphone Array")
echo as the *Default COMMUNICATIONS device* if you still need it in Zoom or Discord.
echo.
echo Steps to change it:
echo   1. Press Win + R, type "mmsys.cpl", then press Enter.
echo   2. In the Settings window, click "More sound settings" on the right.
echo   3. Open the Recording tab, right-click "CABLE Output (VB-Audio Virtual Cable)".
echo   4. Choose "Set as Default Device" (do NOT click "Set as Default Communications Device").
echo.

REM ====== FIXED: Use set /p instead of choice to avoid double prompt ======
set /p "USER_CONFIRM=Finished switching the Recording device? (Y/N): "
if /I not "%USER_CONFIRM%"=="Y" (
    echo [ABORTED] Please change the device in Windows Sound settings, then rerun this script.
    pause
    exit /b 1
)

echo [OK] Continuing with current audio settings...

:run_app
REM ====== 9. Run the app with proper interrupt handling ======
echo [INFO] Running %MAIN_SCRIPT%...
echo [INFO] Press Ctrl+C to exit the application.
echo.

REM ====== FIXED: Add proper interrupt handling ======
python "%MAIN_SCRIPT%"
if %errorlevel% neq 0 (
    echo.
    echo [INFO] Application exited with error code %errorlevel%
    echo [INFO] Press any key to close this window...
    pause >nul
) else (
    echo.
    echo [INFO] Application exited normally.
    echo [INFO] Press any key to close this window...
    pause >nul
)
