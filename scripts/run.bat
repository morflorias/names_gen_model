@echo off
setlocal

REM ===========================================
REM Product Name Generator - Launch Script
REM ===========================================

cd /d "%~dp0\.."

set PYTHON=.venv\Scripts\python.exe
set STREAMLIT=.venv\Scripts\streamlit.exe

if not exist "%PYTHON%" (
    echo [ERROR] Virtual environment not found. Run: python -m venv .venv ^&^& pip install -r requirements.txt
    pause
    exit /b 1
)

:MENU
cls
echo ===========================================
echo   Product Name Generator
echo ===========================================
echo.
echo   [1] Web UI (Streamlit)
echo   [2] Train model
echo   [3] Generate names
echo   [4] Evaluate model
echo   [5] t-SNE visualization
echo   [6] Generate corpus (LM Studio)
echo   [7] Install dependencies
echo   [0] Exit
echo.
set /p CHOICE=Select option:

if "%CHOICE%"=="1" goto WEBUI
if "%CHOICE%"=="2" goto TRAIN
if "%CHOICE%"=="3" goto GENERATE
if "%CHOICE%"=="4" goto EVAL
if "%CHOICE%"=="5" goto TSNE
if "%CHOICE%"=="6" goto CORPUS
if "%CHOICE%"=="7" goto INSTALL
if "%CHOICE%"=="0" goto END
goto MENU

:WEBUI
echo Starting Streamlit UI on port 8501...
%STREAMLIT% run app/streamlit_app.py --server.port 8501
goto END

:TRAIN
set /p EPOCHS=Epochs [40]:
if "%EPOCHS%"=="" set EPOCHS=40
%PYTHON% main.py train --epochs %EPOCHS%
pause
goto MENU

:GENERATE
set /p CAT=Categories [cosmetics,food]:
if "%CAT%"=="" set CAT=cosmetics,food
set /p STARTTOK=Start token [empty]:
set /p N=Count [5]:
if "%N%"=="" set N=5
echo Choose method:
echo   [1] Beam search
echo   [2] Temperature
set /p METHOD_CHOICE=
if "%METHOD_CHOICE%"=="1" (
    set /p BEAM=Beam width [5]:
    if "%BEAM%"=="" set BEAM=5
    if "%STARTTOK%"=="" (
        %PYTHON% main.py generate --categories %CAT% --beam %BEAM% --n %N%
    ) else (
        %PYTHON% main.py generate --categories %CAT% --start-token %STARTTOK% --beam %BEAM% --n %N%
    )
) else (
    set /p TEMP=Temperature [0.9]:
    if "%TEMP%"=="" set TEMP=0.9
    if "%STARTTOK%"=="" (
        %PYTHON% main.py generate --categories %CAT% --temperature %TEMP% --n %N%
    ) else (
        %PYTHON% main.py generate --categories %CAT% --start-token %STARTTOK% --temperature %TEMP% --n %N%
    )
)
pause
goto MENU

:EVAL
%PYTHON% main.py eval
pause
goto MENU

:TSNE
%PYTHON% main.py tsne
pause
goto MENU

:CORPUS
echo NOTE: LM Studio must be running on 127.0.0.1:8888 with unsloth/gemma-4-12B-it-qat-GGUF loaded
%PYTHON% data/lmstudio_generator.py
pause
goto MENU

:INSTALL
echo Installing dependencies...
%PYTHON% -m pip install -r requirements.txt
echo.
echo NOTE: For GPU support, install PyTorch with CUDA:
echo   pip install torch --index-url https://download.pytorch.org/whl/cu128
pause
goto MENU

:END
endlocal
