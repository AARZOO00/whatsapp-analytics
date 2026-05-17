"""
multilingual.py — Proper multilingual detection + Hinglish-aware analysis
Supports: English, Hindi (Devanagari), Hinglish (Roman Hindi), Arabic, mixed
"""
import re
import pandas as pd
import numpy as np
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
    'yaar','bhai','dost','boss','jaan','beta','beti',
    'abhi','phir','baad','pehle','kal','aaj','kal',
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


class MultilingualAnalyzer:
    """
    Proper multilingual analyzer for WhatsApp chats.
    - Detects English, Hindi (Devanagari), Hinglish, Arabic, Mixed
    - Provides emoji analytics per language
    - Provides charts for language distribution
    - Language-aware sentiment comparison
    """

    def __init__(self):
        self.language_detection_available = _LANGDETECT

    # ── Core detection ────────────────────────────────────────────────────────

    def detect_language(self, text: str) -> str:
        """Detect language code for a single message."""
        if not text or not isinstance(text, str) or len(text.strip()) < 3:
            return 'unknown'

        text_clean = _EMOJI_RE.sub('', text).strip()
        if not text_clean:
            return 'unknown'

        dev_ratio = _devanagari_ratio(text_clean)
        ara_ratio = _arabic_ratio(text_clean)

        # Pure Devanagari
        if dev_ratio > 0.5:
            return 'hi'

        # Mixed Devanagari + Roman  → Hinglish
        if dev_ratio > 0.1:
            return 'hinglish'

        # Arabic script
        if ara_ratio > 0.4:
            return 'ar'

        # Roman script: check Hinglish markers
        h_score = _hinglish_score(text_clean)
        if h_score >= 0.25:
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
                # For other langs with some Hinglish markers
                if h_score >= 0.1:
                    return 'hinglish'
                return 'en'   # default roman → english
            except Exception:
                pass

        if h_score >= 0.1:
            return 'hinglish'
        return 'en'

    def is_hinglish(self, text: str) -> bool:
        return self.detect_language(text) == 'hinglish'

    def get_language_label(self, code: str) -> str:
        return _LANG_LABELS.get(code, code)

    def get_language_flag(self, code: str) -> str:
        return _LANG_FLAGS.get(code, '🌐')

    # ── DataFrame enrichment ──────────────────────────────────────────────────

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add language, emoji columns to dataframe."""
        df = df.copy()
        msg_col = 'message_cleaned' if 'message_cleaned' in df.columns else 'message'

        # Language detection (fast — no heavy model)
        df['detected_language'] = df[msg_col].apply(
            lambda x: self.detect_language(str(x)) if pd.notna(x) else 'unknown'
        )
        df['language_label'] = df['detected_language'].map(
            lambda c: _LANG_LABELS.get(c, c)
        )
        df['language_flag'] = df['detected_language'].map(
            lambda c: _LANG_FLAGS.get(c, '🌐')
        )
        df['is_hinglish'] = df['detected_language'] == 'hinglish'

        # Emoji extraction
        raw_col = 'message' if 'message' in df.columns else msg_col
        df['emojis'] = df[raw_col].apply(
            lambda x: _extract_emojis(str(x)) if pd.notna(x) else []
        )
        df['emoji_count'] = df['emojis'].apply(len)
        df['has_emoji']   = df['emoji_count'] > 0

        return df

    # ── Analytics helpers ─────────────────────────────────────────────────────

    def get_language_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with language counts + percentages."""
        if 'detected_language' not in df.columns:
            return pd.DataFrame()
        counts = df['detected_language'].value_counts()
        total  = len(df)
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
        if 'emojis' not in df.columns:
            return pd.DataFrame()
        all_em = [e for lst in df['emojis'] for e in lst]
        counts = Counter(all_em).most_common(top_n)
        return pd.DataFrame(counts, columns=['emoji', 'count'])

    def get_emoji_by_language(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average emoji usage per language."""
        if 'detected_language' not in df.columns or 'emoji_count' not in df.columns:
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
        if 'detected_language' not in df.columns:
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
        if not needed.issubset(df.columns):
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
        if 'has_emoji' not in df.columns or 'sentiment_compound' not in df.columns:
            return pd.DataFrame()
        return (
            df.groupby('has_emoji')['sentiment_compound']
            .agg(['mean', 'count'])
            .rename(columns={'mean': 'avg_sentiment', 'count': 'messages'})
            .round(3)
            .reset_index()
        )
