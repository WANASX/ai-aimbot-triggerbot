@echo off
REM =====================================================
REM GamerFun Dependency Installer with CUDA and Tesseract Check
REM =====================================================

REM Enable ANSI support (works on modern Windows terminals)
for /f "delims=" %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

REM Define colors using more common ANSI codes
set "RED=%ESC%[1;31m"
set "GREEN=%ESC%[1;32m"
set "YELLOW=%ESC%[1;33m"
set "RESET=%ESC%[0m"

echo =====================================================
echo          GamerFun Dependency Installer
echo =====================================================
echo.

REM --------------------- Check Python Installation ---------------------
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo %RED%ERROR: Python 3.12.3 is not installed or not added to PATH.%RESET%
    echo Please install Python 3.12.3 from the official website:
    echo https://www.python.org/ftp/python/3.12.3/python-3.12.3-amd64.exe
    echo Ensure Python 3.12.3 is selected and added to PATH during installation.
    echo.
    pause
    exit /b 1
) ELSE (
    echo %GREEN%Python is installed and added to PATH.%RESET%
)

REM --------------------- Check pip Installation ---------------------
pip --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo %RED%ERROR: pip is not installed or not added to PATH.%RESET%
    echo Please ensure pip is installed.
    echo.
    pause
    exit /b 1
) ELSE (
    echo %GREEN%pip is installed and added to PATH.%RESET%
)

REM --------------------- Check Tesseract Installation ---------------------
echo Checking for Tesseract OCR installation...
where tesseract >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo %RED%ERROR: Tesseract OCR is not installed or not added to PATH.%RESET%
    echo Please install Tesseract OCR:
    echo https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe
    echo.
    pause
    exit /b 1
) ELSE (
    echo %GREEN%Tesseract OCR is installed and configured correctly.%RESET%
)
echo.

REM --------------------- Check CUDA Installation ---------------------
echo Checking CUDA installation...
where nvcc >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo %RED%ERROR: CUDA is not installed or not added to PATH.%RESET%
    echo Please install CUDA Toolkit 12.4 from:
    echo https://developer.nvidia.com/cuda-downloads
    echo.
    pause
    exit /b 1
) ELSE (
    nvcc --version | find "release" > tmp_cuda_ver.txt
    set /p CUDA_VERSION=<tmp_cuda_ver.txt
    del tmp_cuda_ver.txt
    echo CUDA Version Detected: %CUDA_VERSION%
    echo %CUDA_VERSION% | find "12.4" >nul
    IF %ERRORLEVEL% NEQ 0 (
        echo %YELLOW%WARNING: CUDA version is not 12.4. Compatibility issues may occur.%RESET%
        echo Detected Version: %CUDA_VERSION%
        echo Recommended Version: CUDA 12.4
        echo.
    ) ELSE (
        echo %GREEN%CUDA 12.4 is correctly installed.%RESET%
    )
)
echo.

REM --------------------- Install PyTorch and Related Packages ---------------------
echo Installing PyTorch and CUDA support for Python 3.12.3...
pip install torch torchvision==0.19.1+cu124 torchaudio --index-url https://download.pytorch.org/whl/cu124
IF %ERRORLEVEL% NEQ 0 (
    echo %RED%ERROR: Failed to install PyTorch packages.%RESET%
    echo Ensure CUDA 12.4 is installed.
    pause
    exit /b 1
) ELSE (
    echo %GREEN%PyTorch packages installed successfully.%RESET%
)
echo.

REM --------------------- Install Other Required Packages ---------------------
echo Installing other required Python packages...
pip install pyqt5 numpy keyboard screeninfo ultralytics opencv-python pillow pytesseract pypiwin32 bettercam mss
IF %ERRORLEVEL% NEQ 0 (
    echo %RED%ERROR: Failed to install one or more Python packages.%RESET%
    echo Ensure internet connectivity is active.
    pause
    exit /b 1
) ELSE (
    echo %GREEN%All required Python packages installed successfully.%RESET%
)
echo.

REM --------------------- Final Message ---------------------
echo =====================================================
echo         %GREEN%Installation Completed Successfully!%RESET%
echo =====================================================
echo.
echo You can now run GamerFun.exe. Enjoy!
echo.
pause
exit /b 0
