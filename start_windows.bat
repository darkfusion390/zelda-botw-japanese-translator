@echo off
setlocal enabledelayedexpansion

:: ══════════════════════════════════════════════════════════════════════════════
:: start_windows.bat  —  Zelda BotW Translator  (Windows · Windows OCR)
:: ══════════════════════════════════════════════════════════════════════════════
:: Double-click or run from Command Prompt at the repo root:
::
::   start_windows.bat
::
:: Steps:
::   1. Check Python is on PATH
::   2. Install missing Python dependencies
::   3. Check Japanese OCR language pack is installed
::   4. Check Ollama is installed
::   5. Start Ollama serve if not already running
::   6. Pull qwen3:8b if not already downloaded
::   7. Run calibrate.py automatically if bounds.json is missing
::   8. Launch zelda_windows_ocr.py
::
:: Prerequisites:
::   - Python 3.10+ on PATH  →  https://python.org
::     (tick "Add Python to PATH" during install)
::   - Ollama installed       →  https://ollama.com
::   - Japanese language pack: Settings > Time & Language > Language & Region
::     > Add Japanese  (Windows 11 includes OCR automatically)
::   - Camera connected and visible to OpenCV (device index 0 by default)
::
:: OCR note:
::   Windows.Media.Ocr runs via PowerShell — no pip OCR packages needed.
::   Uses Windows ML / DirectML, which will GPU-accelerate on DirectX 12
::   compatible cards (RX 6800 included) automatically via the driver.
:: ══════════════════════════════════════════════════════════════════════════════

set COMPONENT=zelda_windows_ocr.py
set OLLAMA_MODEL=qwen3:8b

:: Resolve repo root from script location
set REPO_ROOT=%~dp0
if "%REPO_ROOT:~-1%"=="\" set REPO_ROOT=%REPO_ROOT:~0,-1%

set MODULARIZED=%REPO_ROOT%\scripts\modularized
set TRANSLATOR_SCRIPT=%MODULARIZED%\%COMPONENT%
set BOUNDS_FILE=%MODULARIZED%\bounds.json
set CALIBRATE_SCRIPT=%MODULARIZED%\calibrate.py

echo.
echo ============================================
echo   Zelda BotW Translator -- Windows Startup (Windows OCR)
echo ============================================

:: ── Step 1: Python ────────────────────────────────────────────────────────────
echo.
echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    echo.
    echo         Install Python 3.10+ from https://python.org
    echo         During install: tick "Add Python to PATH"
    echo         Then close and re-open this window.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo [OK] %%v

:: ── Step 2: Python dependencies ───────────────────────────────────────────────
echo.
echo [2/7] Checking Python dependencies...

:: Helper to check and install one package
:: Usage: call :check_pkg pip_name import_name
goto :after_check_pkg
:check_pkg
    python -c "import %~2" >nul 2>&1
    if errorlevel 1 (
        echo   [MISSING] %~1 -- installing...
        pip install %~1 --quiet
        if errorlevel 1 (
            echo   [ERROR] Failed to install %~1
            echo           Try manually: pip install %~1
            pause
            exit /b 1
        )
        echo   [OK] %~1 installed
    ) else (
        echo   [OK] %~1
    )
    goto :eof
:after_check_pkg

call :check_pkg opencv-python       cv2
call :check_pkg numpy               numpy
call :check_pkg requests            requests
call :check_pkg flask               flask
call :check_pkg fugashi             fugashi
call :check_pkg unidic-lite         unidic_lite
call :check_pkg pykakasi            pykakasi
call :check_pkg jamdict             jamdict
call :check_pkg Pillow              PIL

:: jamdict-data has no importable module — check via pip show
pip show jamdict-data >nul 2>&1
if errorlevel 1 (
    echo   [MISSING] jamdict-data -- installing...
    pip install jamdict-data --quiet
    if errorlevel 1 (
        echo   [ERROR] Failed to install jamdict-data
        pause
        exit /b 1
    )
    echo   [OK] jamdict-data installed
) else (
    echo   [OK] jamdict-data
)

:: Windows OCR uses PowerShell + Windows.Media.Ocr — no pip OCR packages needed.
echo   [OK] OCR engine: Windows.Media.Ocr (built-in, no pip install required)

echo [OK] All dependencies ready

:: ── Step 3: Japanese OCR language pack ───────────────────────────────────────
echo.
echo [3/7] Checking Japanese OCR language pack...
powershell -ExecutionPolicy Bypass -Command ^
  "Add-Type -AssemblyName System.Runtime.WindowsRuntime; [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType=WindowsRuntime] | Out-Null; [Windows.Globalization.Language, Windows.Globalization, ContentType=WindowsRuntime] | Out-Null; $lang = [Windows.Globalization.Language]::new('ja'); $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang); if ($null -eq $engine) { exit 1 } else { exit 0 }" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] Japanese OCR language pack not detected.
    echo.
    echo           To install it:
    echo           1. Open Settings
    echo           2. Go to Time ^& Language ^> Language ^& Region
    echo           3. Click "Add a language" and add Japanese
    echo           4. Windows 11 includes OCR automatically with the language pack
    echo           5. Re-run this script after installing
    echo.
    echo           The translator will not produce output until the pack is installed.
    echo.
    pause
) else (
    echo [OK] Japanese OCR language pack found
)

:: ── Step 3: Ollama ────────────────────────────────────────────────────────────
echo.
echo [4/7] Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama not found.
    echo.
    echo         Install from https://ollama.com
    echo         After installing, close and re-open this window.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('ollama --version 2^>^&1') do echo [OK] %%v

:: ── Step 4: Ollama server ─────────────────────────────────────────────────────
echo.
echo [5/7] Starting Ollama server...

:: Check if already running
curl -s http://localhost:11434/api/tags >nul 2>&1
if not errorlevel 1 (
    echo [OK] Ollama already running
    goto :ollama_ready
)

echo   Starting ollama serve in background...
start /b "" ollama serve >nul 2>&1

:: Poll up to 15 seconds for server to become available
set OLLAMA_STARTED=0
for /l %%i in (1,1,15) do (
    timeout /t 1 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if not errorlevel 1 (
        set OLLAMA_STARTED=1
        goto :ollama_started
    )
)
:ollama_started
if "!OLLAMA_STARTED!"=="1" (
    echo [OK] Ollama server started
) else (
    echo [ERROR] Ollama did not start within 15 seconds.
    echo         Try running 'ollama serve' in a separate Command Prompt window,
    echo         then re-run this script.
    pause
    exit /b 1
)

:ollama_ready

:: ── Step 5: Pull model ────────────────────────────────────────────────────────
echo.
echo [6/7] Checking model %OLLAMA_MODEL%...
echo   First run will download ~5GB -- subsequent runs skip download
ollama pull %OLLAMA_MODEL%
if errorlevel 1 (
    echo [ERROR] Failed to pull %OLLAMA_MODEL%.
    echo         Check your internet connection and try again.
    pause
    exit /b 1
)
echo [OK] %OLLAMA_MODEL% ready

:: ── Step 6: bounds.json — run calibrate.py if missing ────────────────────────
echo.
echo [7/7] Checking bounds.json...
if exist "%BOUNDS_FILE%" (
    echo [OK] bounds.json found
    goto :bounds_ready
)

echo [WARNING] bounds.json not found -- launching calibrate.py...
echo.
echo   Instructions:
echo   - A camera frame will open
echo   - Draw a rectangle around each text region (dialogue box, item title, etc.)
echo   - Press ENTER or SPACE to confirm each region and type a name for it
echo   - Press Q when you have defined all regions
echo.
pause

cd /d "%MODULARIZED%"
python calibrate.py

if not exist "%BOUNDS_FILE%" (
    echo.
    echo [ERROR] bounds.json was not created.
    echo         Make sure your camera is connected and re-run the script.
    pause
    exit /b 1
)
echo [OK] bounds.json created

:bounds_ready

:: ── Verify component exists ───────────────────────────────────────────────────
if not exist "%TRANSLATOR_SCRIPT%" (
    echo.
    echo [ERROR] Component not found: %TRANSLATOR_SCRIPT%
    pause
    exit /b 1
)

:: ── Launch ────────────────────────────────────────────────────────────────────
echo.
echo ============================================
echo   Launching: %COMPONENT%
echo   UI: http://localhost:5002
echo   Stop: Ctrl+C in this window
echo ============================================
echo.

cd /d "%MODULARIZED%"
python "%COMPONENT%"

if errorlevel 1 (
    echo.
    echo [ERROR] Translator exited with an error. See output above.
    pause
)
