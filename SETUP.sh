#!/bin/bash

# Advanced WhatsApp Sentiment Analyzer - Setup Script
# This script sets up the complete Python environment

echo "=================================================="
echo "WhatsApp Sentiment Analyzer - Setup Script"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ Pip upgraded"
echo ""

# Install requirements
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements"
    deactivate
    exit 1
fi
echo "✓ Dependencies installed"
echo ""

# Download NLTK data
echo "Downloading NLTK models..."
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
if [ $? -ne 0 ]; then
    echo "❌ Failed to download NLTK models"
    deactivate
    exit 1
fi
echo "✓ NLTK models downloaded"
echo ""

# Create required directories
echo "Creating data directories..."
mkdir -p data/raw data/processed output
echo "✓ Directories created"
echo ""

# Verify imports
echo "Verifying module imports..."
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
if [ $? -ne 0 ]; then
    echo "❌ Module import verification failed"
    deactivate
    exit 1
fi
echo ""

# Summary
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Make sure virtual environment is active (it is!)"
echo "2. Run the application:"
echo "   streamlit run streamlit_app.py"
echo ""
echo "To deactivate virtual environment later:"
echo "   deactivate"
echo ""
echo "To reactivate in future sessions:"
echo "   source venv/bin/activate  (Mac/Linux)"
echo "   venv\\Scripts\\activate    (Windows)"
echo ""
