@echo off
title Name Generator — Release
echo ============================================
echo   Name Generator - LSTM Model
echo   Release build (generation only)
echo ============================================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH.
    echo Install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

echo [1/2] Installing dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [2/2] Launching Streamlit on port 8502...
echo.
streamlit run app.py --server.port 8502
pause
