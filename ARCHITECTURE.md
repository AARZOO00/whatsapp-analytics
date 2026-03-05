# Architecture Documentation

## Project Overview

Advanced WhatsApp Sentiment Analyzer is a modular Python application that transforms raw WhatsApp chat exports into rich analytical insights through NLP and machine learning.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                           │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────────┐  │
│  │Overview  │Sentiment │Emotions  │Behavior  │Messages      │  │
│  │Dashboard │Analysis  │Detection │Analysis  │& Export      │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   ANALYSIS PIPELINE                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Chat Parser (chat_parser.py)                         │  │
│  │    - Parse WhatsApp .txt format                         │  │
│  │    - Extract datetime, user, message                    │  │
│  │    - Detect system messages & media                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2. Data Cleaner (data_cleaner.py)                       │  │
│  │    - Remove URLs, emails, phone numbers                 │  │
│  │    - Extract emojis                                     │  │
│  │    - Tokenization & stopword removal                    │  │
│  │    - Normalize whitespace                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 3. Parallel Analysis (↓↓↓↓↓)                             │  │
│  ├───────────────┬────────────────┬──────────────┬──────────┤  │
│  │ Analytics     │ Sentiment      │ Emotions     │ Behavior │  │
│  │ (analytics.py)│ (sentiment.py) │ (emotion.py) │(behav.py)│  │
│  ├───────────────┼────────────────┼──────────────┼──────────┤  │
│  │ • Word freq   │ • VADER        │ • Transformer│ • Toxic  │  │
│  │ • User stats  │ • Transformer  │   emotion    │ • Health │  │
│  │ • Heatmaps    │ • Trends       │ • Intensity  │ • Rank   │  │
│  │ • Timeline    │ • Scores       │ • Trend      │ • Patterns
│  └───────────────┴────────────────┴──────────────┴──────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 4. Multilingual Support (multilingual.py)               │  │
│  │    - Language detection                                 │  │
│  │    - Hinglish support                                   │  │
│  │    - Cross-language analysis                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 5. Export Handler (export_handler.py)                   │  │
│  │    - CSV export with full analysis                      │  │
│  │    - Professional PDF reports                           │  │
│  │    - Summary statistics                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     OUTPUT FILES                                │
│  ├── messages_analysis.csv  (Full data with all metrics)       │
│  ├── sentiment_analysis_report.pdf (Professional report)       │
│  └── user_statistics.csv    (Aggregated user metrics)          │
└─────────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### 1. ChatParser (`src/modules/chat_parser.py`)

**Purpose**: Extract raw data from WhatsApp exports

**Key Components**:
- `MESSAGE_PATTERN`: Regex to match WhatsApp format
- `parse_file()`: Main parsing function
- `_parse_line()`: Line-by-line parsing
- Error tracking for unparseable lines

**Input**: .txt WhatsApp export
**Output**: DataFrame with columns:
- `datetime`: Parsed timestamp
- `user`: Message sender
- `message`: Raw message text
- `is_media`: Boolean flag for media messages
- `is_system`: Boolean flag for system messages

**Handles**:
- Multiple date/time formats
- Group chats
- Unicode encoding issues
- System messages
- Media omitted placeholders

---

### 2. DataCleaner (`src/modules/data_cleaner.py`)

**Purpose**: Preprocess text for analysis

**Key Methods**:
- `clean_dataframe()`: Full preprocessing pipeline
- `_clean_text()`: Core text cleaning
- `_extract_emojis()`: Separate emoji extraction
- `_remove_emoji()`: Remove emoji from text
- `_tokenize()`: Split into tokens, remove stopwords
- `get_vocabulary()`: Top words frequency
- `get_summary()`: Cleaning statistics

**Processing Pipeline**:
1. Remove URLs, emails, phone numbers
2. Extract emojis (stored separately)
3. Remove remaining emoji
4. Remove HTML tags
5. Normalize whitespace
6. Convert to lowercase
7. Tokenize
8. Filter stopwords

**Output Columns**:
- `message_cleaned`: Processed text
- `tokens`: List of tokens
- `word_count`: Number of tokens
- `emojis`: Extracted emoji string
- `urls`: List of URLs found
- `message_length`: Original message character count
- `message_lower`: Lowercase original

---

### 3. BasicAnalytics (`src/modules/analytics.py`)

**Purpose**: Generate statistics and visualizations

**Key Metrics**:
- Message count per user
- Daily/hourly activity patterns
- Word frequency analysis
- Message length distribution
- Emoji statistics
- Date range & activity rates

**Visualizations**:
- Bar chart: User activity
- Line graph: Daily timeline
- Heatmap: Hour × Day activity
- Word cloud equivalent
- Pie chart: User contribution
- Histogram: Message lengths

**Methods**:
- `get_summary_stats()`: Key numbers
- `get_user_activity()`: Messages per user
- `get_word_frequency()`: Top words
- `get_emoji_stats()`: Emoji frequency
- Multiple `plot_*()` methods for Plotly visualizations

---

### 4. SentimentAnalyzer (`src/modules/sentiment_analyzer.py`)

**Purpose**: Classify message sentiment

**Models Used**:
1. **VADER** (Rule-based)
   - Fast, no GPU required
   - Good for social media text
   - Returns: positive/negative/neutral scores + compound

2. **DistilBERT** (Transformer-based)
   - Deep learning model
   - More accurate
   - Returns: POSITIVE/NEGATIVE labels with confidence

**Sentiment Levels**:
- **POSITIVE**: compound ≥ 0.05 (VADER) or model confidence
- **NEGATIVE**: compound ≤ -0.05
- **NEUTRAL**: -0.05 < compound < 0.05

**Output Columns**:
- `sentiment_vader`: Classification label
- `sentiment_compound`: VADER score (-1 to 1)
- `sentiment_pos/neg/neu`: Component scores
- `sentiment_transformer`: Transformer classification (optional)
- `sentiment_transformer_score`: Transformer confidence (optional)

**Key Methods**:
- `analyze_vader()`: VADER analysis for single text
- `analyze_transformer()`: Transformer analysis
- `analyze_dataframe()`: Batch processing
- `get_sentiment_distribution()`: Stats
- `get_user_sentiment()`: Per-user averages
- `get_sentiment_trend()`: Over-time trends

---

### 5. EmotionDetector (`src/modules/emotion_detector.py`)

**Purpose**: Classify fine-grained emotions

**Model**: j-hartmann/emotion-english-distilroberta-base

**Emotions Detected**:
- JOY
- ANGER
- SADNESS
- FEAR
- SURPRISE
- DISGUST
- NEUTRAL

**Output Columns**:
- `emotion`: Detected emotion label
- `emotion_score`: Confidence score (0-1)

**Key Methods**:
- `detect_emotion()`: Single text analysis
- `analyze_dataframe()`: Batch processing
- `get_emotion_distribution()`: Stats
- `get_user_emotion()`: Emotion breakdown per user
- `get_emotion_trend()`: Trends over time
- `get_dominant_emotion()`: Most common
- `get_emotion_intensity()`: Avg intensity per emotion

---

### 6. BehavioralAnalyzer (`src/modules/behavioral_analysis.py`)

**Purpose**: Analyze user behavior patterns

**Features**:

1. **Toxicity Detection**
   - Model: michellejieli/TOXIC_BERT
   - Detects abusive/offensive content
   - Returns: is_toxic flag + confidence score

2. **User Metrics**
   - Message count
   - Average message length
   - Question usage (% with '?')
   - Exclamation usage (% with '!')
   - CAPS ratio (uppercase letters %)
   - Most active hour

3. **Health Scoring**
   - Combines sentiment + toxicity + diversity
   - Scale: 0-1 (Poor to Excellent)
   - Status labels: Excellent/Good/Fair/Poor

4. **User Ranking**
   - Positivity score
   - Non-toxic message percentage
   - Sentiment average

**Output Columns**:
- `is_toxic`: Boolean toxicity flag
- `toxicity_score`: Confidence (0-1)
- `response_time`: Minutes between messages
- `caps_ratio`: Uppercase letter ratio
- `question_asked`: Has question mark
- `exclamation`: Has exclamation mark

**Key Methods**:
- `detect_toxicity()`: Single text analysis
- `analyze_dataframe()`: Add all behavioral columns
- `get_toxicity_stats()`: Overall stats
- `get_user_positivity_ranking()`: Ranked user list
- `get_activity_patterns()`: Per-user behaviors
- `get_conversation_health()`: Overall health score
- `get_sentiment_trend_per_user()`: User trends

---

### 7. MultilingualAnalyzer (`src/modules/multilingual.py`)

**Purpose**: Support multiple languages

**Languages Supported**:
- English
- Hindi (Devanagari script)
- Hinglish (Hindi + English mix)

**Features**:

1. **Language Detection**
   - Uses langdetect library
   - Returns: 'en', 'hi', 'hinglish', or 'unknown'

2. **Hinglish Detection**
   - Detects mix of Devanagari + ASCII
   - Enables proper preprocessing

3. **Hindi to English Transliteration**
   - Maps Devanagari characters to phonetic English
   - Enables sentiment analysis on Hindi text

4. **Cross-language Analysis**
   - Compare metrics across languages
   - Language distribution

**Output Columns**:
- `detected_language`: Language code
- `is_hinglish`: Boolean Hinglish flag
- `processed_message`: Text after language processing

**Key Methods**:
- `detect_language()`: Identify language
- `is_hinglish()`: Check for mixed script
- `transliterate_hindi_to_english()`: Convert Devanagari
- `preprocess_multilingual()`: Language-aware cleaning
- `analyze_dataframe()`: Add language columns
- `get_language_distribution()`: Stats
- `get_language_sentiment_comparison()`: Cross-language metrics

---

### 8. ExportHandler (`src/modules/export_handler.py`)

**Purpose**: Generate reports in CSV and PDF formats

**CSV Exports**:
- **messages_analysis.csv**: Full message data with all analysis
  - Columns: datetime, user, message, sentiment, emotion, toxicity, language
- **user_statistics.csv**: Aggregated per-user metrics

**PDF Report**:
- Professional formatted document
- Includes:
  - Overview section with key metrics
  - Sentiment analysis table
  - Emotion breakdown table
  - Conversation health metrics
  - Styled with colors and proper formatting

**Key Methods**:
- `export_to_csv()`: Generic CSV export
- `export_messages_with_analysis()`: Full analysis CSV
- `export_user_statistics()`: User metrics CSV
- `create_pdf_report()`: Generate PDF report
- `get_export_summary()`: List of exported files

---

### 9. FileHandler (`src/utils/file_handler.py`)

**Purpose**: File operations utilities

**Key Functions**:
- `save_uploaded_file()`: Save Streamlit uploads
- `get_all_chat_files()`: List .txt files
- `clean_old_files()`: Manage storage

---

## Data Flow

```
WhatsApp Export (.txt)
    ↓
    ├─→ ChatParser
    │   └─→ DataFrame (raw messages)
    │
    ├─→ DataCleaner
    │   └─→ DataFrame (cleaned + tokenized)
    │
    ├─→ Parallel Processing:
    │   ├─→ SentimentAnalyzer
    │   │   └─→ sentiment_vader, sentiment_compound
    │   │
    │   ├─→ EmotionDetector
    │   │   └─→ emotion, emotion_score
    │   │
    │   ├─→ BehavioralAnalyzer
    │   │   └─→ is_toxic, toxicity_score, caps_ratio, etc.
    │   │
    │   ├─→ MultilingualAnalyzer
    │   │   └─→ detected_language, is_hinglish
    │   │
    │   └─→ BasicAnalytics
    │       └─→ Statistics & visualizations
    │
    ├─→ Streamlit Dashboard (5 tabs)
    │   ├─→ Overview
    │   ├─→ Sentiment
    │   ├─→ Emotions
    │   ├─→ Behavior
    │   └─→ Messages
    │
    └─→ ExportHandler
        ├─→ CSV files
        └─→ PDF report
```

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Web Framework | Streamlit | 1.40.0 |
| Data Processing | Pandas | 2.1.4 |
| Numerical | NumPy | 1.26.3 |
| NLP | NLTK | 3.8.1 |
| Deep Learning | Transformers | 4.36.2 |
| Deep Learning | PyTorch | 2.1.2 |
| Sentiment (Rule) | VADER | 3.3.2 |
| Sentiment (Text) | TextBlob | 0.17.1 |
| Visualization | Plotly | 5.18.0 |
| PDF Generation | ReportLab | 4.0.9 |
| Language Detection | LangDetect | 1.0.9 |
| Configuration | python-dotenv | 1.0.0 |

## Scalability Considerations

### Current Limitations
- Single-machine processing
- All data in memory
- No real-time updates

### Future Enhancements
- **Database**: Supabase for data persistence
- **Caching**: Redis for model caching
- **Distributed**: Task queue for large batches
- **Real-time**: WebSocket updates
- **Parallel**: GPU acceleration for ML models

## Performance Metrics

| Operation | Time (approx) |
|-----------|---------------|
| Parse 1K messages | < 100ms |
| Clean 1K messages | ~200ms |
| Sentiment analysis 1K | 1-2 seconds (VADER) |
| Emotion detection 1K | 10-15 seconds (Transformer) |
| Toxicity 1K | 8-10 seconds |
| Full pipeline 10K | 2-3 minutes |
| Full pipeline 50K | 10-15 minutes |

## Error Handling

1. **Parsing Errors**
   - Tracked and reported
   - Graceful continuation
   - Error summary display

2. **Model Loading Errors**
   - Fallback to simpler models
   - Warnings to user
   - Continue with available models

3. **Encoding Errors**
   - Try UTF-8, fall back to Latin-1
   - Support multiple formats

4. **Streamlit Errors**
   - Try/except in callbacks
   - Session state validation
   - User-friendly messages

## Security

- No data persistence to external services (by default)
- Models downloaded locally
- No API keys required
- Optional: Supabase integration for storage

## Code Organization Principles

1. **Single Responsibility**: Each module has one purpose
2. **Modular**: Loosely coupled, highly cohesive
3. **Reusable**: Functions work independently
4. **Testable**: Clear inputs and outputs
5. **Documented**: Docstrings on all classes/methods
6. **Clean**: PEP 8 compliant, readable

## Future Development

### Immediate
- Database integration (Supabase)
- Batch processing
- Results caching

### Short Term
- Custom emotion models
- Topic modeling (LDA)
- Relationship graphs
- Comparison across chats

### Long Term
- Real-time monitoring
- Predictive analytics
- Multi-language support expansion
- Mobile app
- Cloud deployment
- Team collaboration features
