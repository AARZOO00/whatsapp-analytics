import pandas as pd
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
import warnings

warnings.filterwarnings('ignore')

class MultilingualAnalyzer:
    """
    Support for multilingual text analysis.
    English, Hindi, Hinglish (mix of Hindi and English).
    """

    def __init__(self):
        self.language_detector = None
        self.hindi_sentiment_available = False
        self.hinglish_sentiment_available = False

        try:
            from langdetect import detect
            self.detect_language = detect
            self.language_detection_available = True
        except Exception as e:
            print(f"Warning: Language detection not available: {e}")
            self.language_detection_available = False

        try:
            self.hindi_sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
            self.hindi_sentiment_available = True
        except Exception as e:
            print(f"Warning: Hindi sentiment model not available: {e}")

    def detect_language(self, text: str) -> str:
        """
        Detect language of text.

        Args:
            text: Input text

        Returns:
            Language code: 'en', 'hi', or 'hinglish'
        """
        if not self.language_detection_available or not text or not isinstance(text, str):
            return 'unknown'

        try:
            lang = self.detect_language(text)

            if lang in ['hi', 'mr', 'pa']:
                if any(char.isascii() for char in text):
                    return 'hinglish'
                return 'hi'

            return 'en'
        except Exception as e:
            print(f"Warning: Language detection failed: {e}")
            return 'unknown'

    def is_hinglish(self, text: str) -> bool:
        """Check if text is Hinglish (mix of Hindi and English)"""
        if not text or not isinstance(text, str):
            return False

        devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        ascii_chars = sum(1 for c in text if ord(c) < 128)

        return devanagari_chars > 0 and ascii_chars > 0

    def transliterate_hindi_to_english(self, text: str) -> str:
        """
        Simple Hindi to English transliteration.
        Maps Devanagari to phonetic English.
        """
        devanagari_to_english = {
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

        result = ""
        for char in text:
            result += devanagari_to_english.get(char, char)

        return result

    def preprocess_multilingual(self, text: str) -> str:
        """
        Preprocess multilingual text.
        Converts Hinglish to English for analysis.
        """
        if not text or not isinstance(text, str):
            return ""

        if self.is_hinglish(text):
            text = self.transliterate_hindi_to_english(text)

        return text.lower()

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add language detection to dataframe.

        Args:
            df: DataFrame with messages

        Returns:
            DataFrame with language column
        """
        df = df.copy()

        df['detected_language'] = df['message'].apply(self._safe_detect_language)

        df['is_hinglish'] = df['message'].apply(self.is_hinglish)

        df['processed_message'] = df['message'].apply(self.preprocess_multilingual)

        return df

    def _safe_detect_language(self, text: str) -> str:
        """Safely detect language with fallback"""
        if not self.language_detection_available:
            return 'unknown'

        try:
            from langdetect import detect

            if not text or not isinstance(text, str):
                return 'unknown'

            lang = detect(text)

            if lang in ['hi', 'mr', 'pa']:
                if any(char.isascii() and ord(char) > 127 for char in text):
                    return 'hinglish'
                return lang

            return 'en'
        except Exception:
            return 'unknown'

    def get_language_distribution(self, df: pd.DataFrame) -> dict:
        """Get distribution of languages in chat"""
        if 'detected_language' not in df.columns:
            return {}

        counts = df['detected_language'].value_counts()
        total = len(df)

        return {
            lang: {
                'count': count,
                'percentage': round((count / total) * 100, 2)
            }
            for lang, count in counts.items()
        }

    def get_hinglish_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get all Hinglish messages"""
        if 'is_hinglish' not in df.columns:
            return df[df['is_hinglish'] == True] if 'is_hinglish' in df.columns else pd.DataFrame()

        return df[df['is_hinglish'] == True]

    def get_language_sentiment_comparison(self, df: pd.DataFrame) -> dict:
        """
        Compare sentiment across languages.
        """
        if 'detected_language' not in df.columns or 'sentiment_compound' not in df.columns:
            return {}

        comparison = {}

        for lang in df['detected_language'].unique():
            lang_df = df[df['detected_language'] == lang]

            comparison[lang] = {
                'message_count': len(lang_df),
                'avg_sentiment': round(lang_df['sentiment_compound'].mean(), 3),
                'avg_message_length': round(lang_df['message_length'].mean(), 2)
            }

        return comparison