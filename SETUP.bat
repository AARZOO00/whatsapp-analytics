@echo off
REM Advanced WhatsApp Sentiment Analyzer - Setup Script (Windows)

echo ==================================================
echo WhatsApp Sentiment Analyzer - Setup Script
echo ==================================================
echo.

REM Check Python version
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✓ Python %PYTHON_VERSION% found
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo ✓ Pip upgraded
echo.

REM Install requirements
echo Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo ❌ Failed to install requirements
    call deactivate
    pause
    exit /b 1
)
echo ✓ Dependencies installed
echo.

REM Download NLTK data
echo Downloading NLTK models...
python -c "
import nltk
print('  Downloading punkt...', end='', flush=True)
nltk.download('punkt', quiet=True)
print(' ✓')
print('  Downloading stopwords...', end='', flush=True)
nltk.download('stopwords', quiet=True)
print(' ✓')
print('  Downloading averaged_perceptron_tagger...', end='', flush=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
print(' ✓')
print('  Downloading wordnet...', end='', flush=True)
nltk.download('wordnet', quiet=True)
print(' ✓')
"
if %errorlevel% neq 0 (
    echo ❌ Failed to download NLTK models
    call deactivate
    pause
    exit /b 1
)
echo ✓ NLTK models downloaded
echo.

REM Create required directories
echo Creating data directories...
if not exist "data" mkdir data
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "output" mkdir output
echo ✓ Directories created
echo.

REM Verify imports
echo Verifying module imports...
python -c "
import sys
sys.path.insert(0, '.')
try:
    from src.modules.chat_parser import WhatsAppParser
    from src.modules.data_cleaner import DataCleaner
    from src.modules.analytics import BasicAnalytics
    from src.modules.sentiment_analyzer import SentimentAnalyzer
    from src.modules.emotion_detector import EmotionDetector
    from src.modules.behavioral_analysis import BehavioralAnalyzer
    from src.modules.multilingual import MultilingualAnalyzer
    from src.modules.export_handler import ExportHandler
    from src.utils.file_handler import save_uploaded_file
    print('✓ All modules imported successfully')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"
if %errorlevel% neq 0 (
    echo ❌ Module import verification failed
    call deactivate
    pause
    exit /b 1
)
echo.

REM Summary
echo ==================================================
echo ✅ Setup Complete!
echo ==================================================
echo.
echo Next steps:
echo 1. Virtual environment is now active
echo 2. Run the application:
echo    streamlit run streamlit_app.py
echo.
echo To deactivate virtual environment:
echo    deactivate
echo.
echo To reactivate in future sessions:
echo    venv\Scripts\activate.bat
echo.
pause
