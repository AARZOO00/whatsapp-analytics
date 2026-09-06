import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

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
    Support for multilingual text analysis (English, Hindi, Hinglish).
    Optimized with fast Unicode script ranges and Hinglish lexical heuristics.
    """

    # Devanagari Unicode character range
    _DEV_RE = re.compile(r'[\u0900-\u097F]')

    # Characteristic Hinglish vocabulary markers
    _HINGLISH_WORDS = {
        'hai','hain','ho','tha','the','thi','bhi','kya','aur','mein','ka','ke',
        'ki','ko','se','nhi','nah','haan','han','par','pe','woh','wo','yeh','ye',
        'ab','bs','bas','toh','to','na','ne','ek','koi','sab','kuch','kaise',
        'kyun','kyunki','lekin','matlab','phir','fir','agar','jab','tab','yaar',
        'bhai','arre','acha','theek','bolo','batao','karo','karte','kar'
    }

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
        self.hindi_sentiment_available = False
        if preload_model:
            pipe = get_multilingual_pipeline()
            self.hindi_sentiment_available = pipe is not None

    def fast_detect_language(self, text: str) -> str:
        """
        Ultra-fast Unicode + dictionary language classification.
        1000x faster than langdetect and never throws exceptions.
        """
        if not text or not isinstance(text, str):
            return 'en'

        has_devanagari = bool(self._DEV_RE.search(text))
        has_latin = any('a' <= c <= 'z' or 'A' <= c <= 'Z' for c in text)

        if has_devanagari and has_latin:
            return 'hinglish'
        if has_devanagari:
            return 'hi'

        # If pure Latin text, check for common Hinglish words
        if has_latin:
            words = set(re.findall(r'\b[a-zA-Z]{2,}\b', text.lower()))
            if words and len(words & self._HINGLISH_WORDS) >= 1:
                return 'hinglish'
            return 'en'

        return 'en'

    def is_hinglish(self, text: str) -> bool:
        """Check if text is Hinglish (mix of Hindi concepts and English/Latin)."""
        lang = self.fast_detect_language(text)
        return lang == 'hinglish'

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
        """
        Add language detection and multilingual preprocessing to dataframe.
        """
        if df.empty:
            df['detected_language'] = []
            df['is_hinglish'] = []
            df['processed_message'] = []
            return df

        df = df.copy()
        msg_col = 'message' if 'message' in df.columns else 'message_cleaned'
        raw_msgs = df[msg_col].fillna('').astype(str).tolist()

        # Batch language classification
        detected_langs = [self.fast_detect_language(m) for m in raw_msgs]
        is_hinglish_list = [lang == 'hinglish' for lang in detected_langs]
        processed_list = [self.preprocess_multilingual(m) for m in raw_msgs]

        df['detected_language'] = detected_langs
        df['is_hinglish'] = is_hinglish_list
        df['processed_message'] = processed_list

        return df

    def get_language_distribution(self, df: pd.DataFrame) -> dict:
        """Get distribution of languages safely."""
        if 'detected_language' not in df.columns or df.empty:
            return {}

        counts = df['detected_language'].value_counts()
        total = max(len(df), 1)

        return {
            str(lang): {
                'count': int(count),
                'percentage': round((count / total) * 100, 2)
            }
            for lang, count in counts.items()
        }

    def get_hinglish_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get all Hinglish messages safely."""
        if 'is_hinglish' not in df.columns or df.empty:
            return pd.DataFrame()
        return df[df['is_hinglish'] == True]

    def get_language_sentiment_comparison(self, df: pd.DataFrame) -> dict:
        """Compare sentiment across detected languages safely."""
        if 'detected_language' not in df.columns or 'sentiment_compound' not in df.columns or df.empty:
            return {}

        comparison = {}
        for lang, l_df in df.groupby('detected_language'):
            comparison[str(lang)] = {
                'message_count': len(l_df),
                'avg_sentiment': round(float(l_df['sentiment_compound'].mean()), 3),
                'avg_message_length': round(float(l_df['message_length'].mean()), 2) if 'message_length' in l_df.columns else 0
            }

        return comparison