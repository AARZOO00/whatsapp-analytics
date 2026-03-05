# Quick Start Guide

Get the WhatsApp Sentiment Analyzer running in 5 minutes.

## Prerequisites

- Python 3.8+ installed
- WhatsApp chat export (.txt file)

## Installation

### 1. Navigate to Project

```bash
cd whatsapp-analyzer
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download NLP Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Launch Application

```bash
streamlit run streamlit_app.py
```

This will:
- Start the Streamlit server
- Open browser at `http://localhost:8501`
- Show the chat uploader

## Test with Sample Chat

A sample chat is included in `data/raw/sample_chat.txt`

1. Click "Choose your WhatsApp chat export"
2. Navigate to `data/raw/`
3. Select `sample_chat.txt`
4. Wait for analysis to complete (~10-20 seconds)
5. Explore the 5 tabs:
   - **Overview**: Message statistics & trends
   - **Sentiment**: Positive/Negative/Neutral analysis
   - **Emotions**: Joy, Anger, Sadness, etc.
   - **Behavior**: Toxicity & user health scores
   - **Messages**: Full data export

## Export Your Chat

From WhatsApp:
1. Settings > Chats > Chat history
2. Select conversation
3. Export chat
4. Choose "Without media"
5. Save .txt file
6. Upload to analyzer

## Features at a Glance

| Feature | What It Does |
|---------|-------------|
| **Chat Parser** | Extracts messages, timestamps, users |
| **Data Cleaner** | Removes URLs, normalizes text, tokenizes |
| **Analytics** | Word frequency, activity heatmaps, user stats |
| **Sentiment** | Classifies each message as positive/negative/neutral |
| **Emotion** | Detects joy, anger, sadness, fear, surprise, disgust |
| **Behavior** | Detects toxicity, rates user positivity |
| **Languages** | English, Hindi, Hinglish support |
| **Export** | Download analysis as CSV or PDF report |

## Troubleshooting

**"Streamlit not found"**
```bash
pip install streamlit==1.40.0
```

**"No module named 'nltk'"**
```bash
pip install nltk==3.8.1
```

**"Models downloading..."**
- First run takes longer as ML models download
- Patience! Models are ~500MB total
- Subsequent runs are fast

**Parsing error with your chat?**
- Ensure export is without media
- Check file is .txt format
- Verify UTF-8 encoding

## Module Architecture

```
User Chat File (.txt)
        ↓
   ChatParser (extract messages)
        ↓
   DataCleaner (clean & preprocess)
        ↓
   ↙ ↓ ↓ ↓ ↘
SA  BA  ED  MA  Analytics
(Sentiment, Behavior, Emotion, Multilingual, Analytics)
        ↓
   Streamlit Dashboard
   (5 interactive tabs)
        ↓
   Export (CSV/PDF)
```

## Next Steps

1. **Customize Styling** - Edit `.env` for theme colors
2. **Add Database** - Connect Supabase in multilingual.py
3. **Batch Processing** - Analyze multiple chats simultaneously
4. **Real-time Monitoring** - Set up continuous export monitoring
5. **Advanced Insights** - Add topic modeling, relationship graphs

## File Organization

- `streamlit_app.py` - Main application entry
- `src/modules/` - Core analysis engines
- `src/utils/` - Helper functions
- `data/raw/` - Input chats
- `data/processed/` - Cleaned data cache
- `output/` - Generated reports

## Keyboard Shortcuts (Streamlit)

- `r` - Rerun app
- `c` - Clear cache
- `s` - Open settings
- `i` - About

## Performance Tips

- **Small chats** (<10K messages): <30 seconds
- **Medium chats** (10K-50K): <2 minutes
- **Large chats** (50K+): 5-10 minutes
- ML models cached after first run

## Getting Help

Check docstrings:
```bash
python -c "from src.modules.chat_parser import WhatsAppParser; help(WhatsAppParser)"
```

## Project Complete! 🎉

You now have a production-ready WhatsApp analyzer with:
- ✅ 8 specialized analysis modules
- ✅ Interactive Streamlit dashboard
- ✅ 5 analysis tabs with visualizations
- ✅ Export to CSV & PDF
- ✅ Multilingual support
- ✅ Modular, maintainable code

Enjoy analyzing!
