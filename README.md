# Advanced WhatsApp Sentiment Analyzer

A comprehensive Python application for analyzing WhatsApp conversations with sentiment analysis, emotion detection, and behavioral insights using NLP and machine learning.

## Features

- **Chat Import**: Parse WhatsApp exported .txt files
- **Data Cleaning**: Automatic preprocessing with emoji extraction, URL removal, stopword filtering
- **Basic Analytics**: Message frequency, user activity, word clouds, activity heatmaps
- **Sentiment Analysis**: VADER and Transformer-based sentiment classification
- **Emotion Detection**: Joy, Anger, Sadness, Fear, Surprise, Disgust
- **Behavioral Analysis**: Toxicity detection, user positivity ranking, sentiment trends
- **Multilingual Support**: English, Hindi, Hinglish with language detection
- **Interactive Dashboard**: Streamlit-based visualization with multiple tabs
- **Export Features**: CSV and PDF report generation

## Project Structure

```
whatsapp-analyzer/
├── src/
│   ├── modules/
│   │   ├── chat_parser.py          # WhatsApp chat parsing
│   │   ├── data_cleaner.py         # Text preprocessing
│   │   ├── analytics.py            # Basic analytics & visualizations
│   │   ├── sentiment_analyzer.py   # Sentiment analysis
│   │   ├── emotion_detector.py     # Emotion detection
│   │   ├── behavioral_analysis.py  # Behavior & toxicity analysis
│   │   ├── multilingual.py         # Language detection & support
│   │   ├── export_handler.py       # CSV/PDF export
│   │   └── __init__.py
│   └── utils/
│       ├── file_handler.py         # File operations
│       └── __init__.py
├── data/
│   ├── raw/                        # Raw WhatsApp exports
│   └── processed/                  # Cleaned data
├── output/                         # Generated reports
├── streamlit_app.py               # Main Streamlit app
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables
└── README.md
```

## Setup Instructions

### 1. Clone/Extract Project

```bash
cd whatsapp-analyzer
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download NLP Models

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
```

### 5. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## How to Export WhatsApp Chat

1. Open WhatsApp
2. Go to Settings > Chats > Chat history
3. Select the conversation to export
4. Choose "Export chat"
5. Select "Without media"
6. Save the .txt file

## Usage Guide

### Step 1: Upload Chat
- Click "Choose your WhatsApp chat export"
- Select your exported .txt file
- Wait for processing

### Step 2: Explore Tabs

**Overview Tab**
- Message statistics
- User activity chart
- Activity timeline & heatmap
- Word frequency
- Message length distribution

**Sentiment Tab**
- Sentiment distribution (Positive/Negative/Neutral)
- User sentiment comparison
- Sentiment trends over time
- Sentiment scores visualization

**Emotions Tab**
- Emotion distribution pie chart
- Emotion intensity per emotion
- Emotion breakdown by user
- Dominant emotion identification

**Behavior Tab**
- Conversation health score
- User positivity ranking
- Toxicity analysis
- User activity patterns
- Message characteristics

**Messages Tab**
- Complete data table with all analysis
- Export to CSV
- Generate PDF report

## Module Details

### ChatParser
Handles parsing WhatsApp exports with support for:
- Different date/time formats
- Group chats with multiple users
- System messages
- Media messages
- Error detection & reporting

### DataCleaner
Text preprocessing including:
- URL removal
- Email removal
- Phone number removal
- Emoji extraction & removal
- Tokenization
- Stopword filtering
- Duplicate whitespace removal

### BasicAnalytics
Generates visualizations:
- User message counts
- Daily activity timeline
- Hourly activity heatmap
- Word frequency analysis
- Message length distribution
- User contribution pie charts

### SentimentAnalyzer
Two-level sentiment analysis:
- VADER Sentiment (rule-based)
- Transformer models (deep learning)
- Per-user sentiment averages
- Sentiment trend analysis

### EmotionDetector
Fine-grained emotion detection:
- 7 emotion categories
- Emotion intensity scoring
- Emotion trends over time
- User emotion profiles

### BehavioralAnalyzer
User behavior analysis:
- Toxicity detection
- Caps usage ratio
- Question & exclamation patterns
- User positivity ranking
- Conversation health scoring

### MultilingualAnalyzer
Language support:
- Language detection
- Hinglish (Hindi-English mix) support
- Language-specific preprocessing
- Multilingual sentiment comparison

### ExportHandler
Report generation:
- CSV export of all analysis
- Professional PDF reports
- Summary statistics
- Formatted tables

## Technologies Used

- **Streamlit**: Interactive web interface
- **Pandas**: Data manipulation
- **NLTK**: Natural Language Toolkit
- **Transformers**: Pre-trained ML models
- **VADER**: Sentiment analysis
- **Plotly**: Interactive visualizations
- **ReportLab**: PDF generation
- **LangDetect**: Language detection

## Model Information

- **Sentiment**: distilbert-base-uncased-finetuned-sst-2-english
- **Emotion**: j-hartmann/emotion-english-distilroberta-base
- **Toxicity**: michellejieli/TOXIC_BERT
- **Multilingual**: bert-base-multilingual-uncased-sentiment

## Performance Notes

- Large chat files (>100K messages) may take time for ML analysis
- GPU acceleration recommended for emotion/toxicity detection
- First run downloads ML models (~500MB)

## Troubleshooting

**Models not downloading?**
```bash
python -c "from transformers import pipeline; pipeline('sentiment-analysis')"
```

**Encoding errors?**
- Ensure WhatsApp export is in UTF-8 format
- Try different encoding in chat_parser.py

**Streamlit not found?**
```bash
pip install streamlit==1.40.0
```

## Future Enhancements

- Database storage with Supabase
- Real-time chat monitoring
- Comparative analysis across multiple chats
- Custom emotion models
- Interactive filtering & drilling down
- Team analytics dashboard

## License

Open source for educational use.

## Support

For issues or questions, check the module docstrings and function documentation in the source code.
