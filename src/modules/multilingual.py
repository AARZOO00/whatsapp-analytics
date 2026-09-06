"""
multilingual.py — Comprehensive multilingual detection + Hinglish-aware analysis
Supports: English, Hindi (Devanagari), Hinglish (Roman Hindi), Arabic, mixed
"""
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    _LANGDETECT = True
except ImportError:
    _LANGDETECT = False

# ── Script / Unicode range helpers ───────────────────────────────────────────
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
_ARABIC_RE     = re.compile(r'[\u0600-\u06FF\u0750-\u077F]')
_EMOJI_RE      = re.compile(r'[\U0001F300-\U0001FFFF\U00002600-\U000027BF\U0001F900-\U0001F9FF]')

# Roman-Hindi / Hinglish marker words
_HINGLISH_MARKERS = {
    'hai','hain','nahi','nhi','aur','kya','yaar','bhai','behen',
    'accha','acha','theek','thik','karo','karna','tha','thi','the',
    'mera','meri','tera','teri','hum','tum','aap','woh','yeh',
    'bahut','bohot','bilkul','zaroor','matlab','seedha','seedhi',
    'inshallah','mashallah','alhamdulillah','subhanallah',
    'bolo','batao','suno','dekho','aao','jao','karo','raho',
    'mast','zabardast','badhiya','bakwaas','bekar','faltu',
    'dost','boss','jaan','beta','beti',
    'abhi','phir','fir','baad','pehle','kal','aaj',
    'kar','karte','kare','rahe','raha','rahi','lekin','kyunki',
    'kyun','kaise','kuch','sab','koi','ek','bas','bs','toh','to'
}

_LANG_LABELS = {
    'en':       'English',
    'hi':       'Hindi',
    'hinglish': 'Hinglish',
    'ar':       'Arabic',
    'mixed':    'Mixed',
    'unknown':  'Unknown',
}

_LANG_FLAGS = {
    'en':       '🇬🇧',
    'hi':       '🇮🇳',
    'hinglish': '🔀',
    'ar':       '🇸🇦',
    'mixed':    '🌍',
    'unknown':  '❓',
}

def _extract_emojis(text: str) -> list:
    return _EMOJI_RE.findall(text)

def _devanagari_ratio(text: str) -> float:
    chars = [c for c in text if c.strip()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _DEVANAGARI_RE.match(c)) / len(chars)

def _arabic_ratio(text: str) -> float:
    chars = [c for c in text if c.strip()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _ARABIC_RE.match(c)) / len(chars)

def _hinglish_score(text: str) -> float:
    """0-1 score of how Hinglish a roman-script text is."""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _HINGLISH_MARKERS)
    return hits / len(words)

# Global cache for transformer multilingual pipeline
_MULTILINGUAL_PIPELINE = None
_MULTILINGUAL_TRIED = False

def _get_device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1

def get_multilingual_pipeline():
    """Lazy load and cache the multilingual BERT pipeline."""
    global _MULTILINGUAL_PIPELINE, _MULTILINGUAL_TRIED
    if _MULTILINGUAL_PIPELINE is not None:
        return _MULTILINGUAL_PIPELINE
    if _MULTILINGUAL_TRIED:
        return None

    _MULTILINGUAL_TRIED = True
    try:
        from transformers import pipeline
        device = _get_device()
        _MULTILINGUAL_PIPELINE = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=device,
            truncation=True,
            max_length=128
        )
        return _MULTILINGUAL_PIPELINE
    except Exception as e:
        print(f"Warning: Multilingual BERT model unavailable: {e}")
        return None


class MultilingualAnalyzer:
    """
    Multilingual analyzer for WhatsApp chats.
    - Detects English, Hindi (Devanagari), Hinglish, Arabic, Mixed
    - Fast Unicode and lexical rules
    - Provides emoji analytics per language
    - Provides language distribution and sentiment comparison
    """

    _DEV_RE = _DEVANAGARI_RE
    _HINGLISH_WORDS = _HINGLISH_MARKERS

    _DEVANAGARI_TO_ENGLISH = {
        'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
        'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'अँ': 'an', 'अः': 'ah',
        'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'nga',
        'च': 'cha', 'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
        'ट': 'ta', 'ठ': 'tha', 'ड': 'da', 'ढ': 'dha', 'ण': 'na',
        'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
        'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
        'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va',
        'श': 'sha', 'ष': 'sha', 'स': 'sa', 'ह': 'ha',
        'ज्ञ': 'gya', 'त्र': 'tra', 'क्ष': 'ksha',
    }

    def __init__(self, preload_model: bool = False):
        self.language_detection_available = _LANGDETECT
        self.hindi_sentiment_available = False
        if preload_model:
            pipe = get_multilingual_pipeline()
            self.hindi_sentiment_available = pipe is not None

    def detect_language(self, text: str) -> str:
        """Detect language code for a single message."""
        if not text or not isinstance(text, str) or len(text.strip()) < 2:
            return 'en'

        text_clean = _EMOJI_RE.sub('', text).strip()
        if not text_clean:
            return 'en'

        dev_ratio = _devanagari_ratio(text_clean)
        ara_ratio = _arabic_ratio(text_clean)

        # Pure Devanagari
        if dev_ratio > 0.5:
            return 'hi'

        # Mixed Devanagari + Roman -> Hinglish
        if dev_ratio > 0.1:
            return 'hinglish'

        # Arabic script
        if ara_ratio > 0.4:
            return 'ar'

        # Roman script: check Hinglish markers
        h_score = _hinglish_score(text_clean)
        if h_score >= 0.2:
            return 'hinglish'

        # Fallback: langdetect
        if _LANGDETECT:
            try:
                lang = detect(text_clean)
                if lang in ('hi', 'mr', 'pa', 'gu', 'bn'):
                    return 'hi' if dev_ratio > 0.01 else 'hinglish'
                if lang in ('ar', 'ur', 'fa'):
                    return 'ar'
                if lang == 'en':
                    return 'en'
                if h_score >= 0.1:
                    return 'hinglish'
                return 'en'
            except Exception:
                pass

        if h_score >= 0.1:
            return 'hinglish'
        return 'en'

    def fast_detect_language(self, text: str) -> str:
        """Alias for detect_language for backwards compatibility."""
        return self.detect_language(text)

    def is_hinglish(self, text: str) -> bool:
        return self.detect_language(text) == 'hinglish'

    def get_language_label(self, code: str) -> str:
        return _LANG_LABELS.get(code, code)

    def get_language_flag(self, code: str) -> str:
        return _LANG_FLAGS.get(code, '🌐')

    def transliterate_hindi_to_english(self, text: str) -> str:
        """Phonetic conversion of Devanagari characters to English."""
        if not text:
            return ""
        table = self._DEVANAGARI_TO_ENGLISH
        return ''.join(table.get(char, char) for char in text)

    def preprocess_multilingual(self, text: str) -> str:
        """Normalize Hinglish / Hindi text."""
        if not text:
            return ""
        if self._DEV_RE.search(text):
            text = self.transliterate_hindi_to_english(text)
        return text.lower().strip()

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add language, emoji, and preprocessing columns to dataframe."""
        if df.empty:
            df['detected_language'] = []
            df['language_label'] = []
            df['language_flag'] = []
            df['is_hinglish'] = []
            df['processed_message'] = []
            df['emojis'] = []
            df['emoji_count'] = []
            df['has_emoji'] = []
            return df

        df = df.copy()
        msg_col = 'message' if 'message' in df.columns else 'message_cleaned'
        raw_msgs = df[msg_col].fillna('').astype(str).tolist()

        detected_langs = [self.detect_language(m) for m in raw_msgs]
        df['detected_language'] = detected_langs
        df['language_label'] = [self.get_language_label(c) for c in detected_langs]
        df['language_flag'] = [self.get_language_flag(c) for c in detected_langs]
        df['is_hinglish'] = [lang == 'hinglish' for lang in detected_langs]
        df['processed_message'] = [self.preprocess_multilingual(m) for m in raw_msgs]

        # Emoji extraction
        df['emojis'] = [_extract_emojis(m) for m in raw_msgs]
        df['emoji_count'] = [len(em) for em in df['emojis']]
        df['has_emoji'] = [c > 0 for c in df['emoji_count']]

        return df

    def get_language_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with language counts + percentages."""
        if 'detected_language' not in df.columns or df.empty:
            return pd.DataFrame()
        counts = df['detected_language'].value_counts()
        total = len(df)
        rows = []
        for code, cnt in counts.items():
            rows.append({
                'code':       code,
                'language':   _LANG_LABELS.get(code, code),
                'flag':       _LANG_FLAGS.get(code, '🌐'),
                'count':      int(cnt),
                'percentage': round(cnt / total * 100, 1),
            })
        return pd.DataFrame(rows)

    def get_emoji_distribution(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Top N emojis across all messages."""
        if 'emojis' not in df.columns or df.empty:
            return pd.DataFrame()
        all_em = [e for lst in df['emojis'] for e in lst]
        counts = Counter(all_em).most_common(top_n)
        return pd.DataFrame(counts, columns=['emoji', 'count'])

    def get_emoji_by_language(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average emoji usage per language."""
        if 'detected_language' not in df.columns or 'emoji_count' not in df.columns or df.empty:
            return pd.DataFrame()
        return (
            df.groupby('detected_language')['emoji_count']
            .agg(['mean', 'sum', 'count'])
            .rename(columns={'mean': 'avg_emojis', 'sum': 'total_emojis', 'count': 'messages'})
            .round(2)
            .reset_index()
        )

    def get_user_language_mix(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Per-user language breakdown (%)."""
        if 'detected_language' not in df.columns or 'user' not in df.columns or df.empty:
            return pd.DataFrame()
        top_users = df['user'].value_counts().head(top_n).index
        sub = df[df['user'].isin(top_users)]
        pivot = (
            sub.groupby(['user', 'detected_language'])
            .size()
            .unstack(fill_value=0)
        )
        pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(1)
        return pct.reset_index()

    def get_language_sentiment_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """Avg sentiment compound score per language."""
        needed = {'detected_language', 'sentiment_compound'}
        if not needed.issubset(df.columns) or df.empty:
            return pd.DataFrame()
        return (
            df.groupby('detected_language')
            .agg(
                messages=('sentiment_compound', 'count'),
                avg_sentiment=('sentiment_compound', 'mean'),
                positive_pct=('sentiment_vader',
                              lambda x: (x == 'POSITIVE').sum() / len(x) * 100
                              if 'sentiment_vader' in df.columns else 0),
            )
            .round(3)
            .reset_index()
        )

    def get_emoji_sentiment_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Messages with vs without emojis — sentiment difference."""
        if 'has_emoji' not in df.columns or 'sentiment_compound' not in df.columns or df.empty:
            return pd.DataFrame()
        return (
            df.groupby('has_emoji')['sentiment_compound']
            .agg(['mean', 'count'])
            .rename(columns={'mean': 'avg_sentiment', 'count': 'messages'})
            .round(3)
            .reset_index()
        )

    def get_hinglish_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get all Hinglish messages safely."""
        if 'is_hinglish' not in df.columns or df.empty:
            return pd.DataFrame()
        return df[df['is_hinglish'] == True]
