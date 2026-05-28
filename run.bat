@echo off
echo Starting VoxFlow...

:: Enable WASAPI/ASIO support in sounddevice
set SD_ENABLE_ASIO=1

call conda activate voxflow 2>nul || (
    echo [ERROR] conda environment 'voxflow' not found. Run install.bat first.
    pause
    exit /b 1
)

cd /d "%~dp0"
python ui/app.py

if errorlevel 1 (
    echo.
    echo [ERROR] VoxFlow crashed. Check the output above.
    pause
)
