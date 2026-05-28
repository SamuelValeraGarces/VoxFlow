@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  VoxFlow — Installation
echo ============================================================
echo.

:: Check conda
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] conda not found. Install Miniconda from:
    echo         https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

:: Check git
where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] git not found. Install Git from https://git-scm.com
    pause
    exit /b 1
)

echo [1/5] Creating conda environment (Python 3.11)...
call conda create -n voxflow python=3.11 -y
if errorlevel 1 goto :error

echo.
echo [2/5] Activating environment and installing PyTorch (CUDA 12.4)...
call conda activate voxflow

:: PyTorch with CUDA 12.4 — compatible with CUDA 12.x drivers
pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto :error

echo.
echo [3/5] Installing VoxFlow dependencies...
pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo [4/5] Installing sounddevice with ASIO support (pip only)...
pip uninstall sounddevice -y >nul 2>&1
pip install sounddevice --no-cache-dir
if errorlevel 1 goto :error

echo.
echo [5/5] Downloading model weights...
python download_models.py
if errorlevel 1 goto :error

echo.
echo ============================================================
echo  Installation complete!
echo  Run: run.bat
echo ============================================================
pause
exit /b 0

:error
echo.
echo [ERROR] Installation failed. Check the output above.
pause
exit /b 1
