import re
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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


# ─────────────────────────────────────────────────────────────────────────────
#  WhatsApp-specific knowledge bases
# ─────────────────────────────────────────────────────────────────────────────

# Hinglish / Indian-English positive words VADER doesn't know
_HINGLISH_POSITIVE = {
    'mast': 0.6, 'zabardast': 0.8, 'bindaas': 0.6, 'badhiya': 0.7,
    'shandar': 0.8, 'kamaal': 0.7, 'wah': 0.5, 'waah': 0.5,
    'bahut acha': 0.7, 'bohot acha': 0.7, 'bahut accha': 0.7,
    'achha': 0.4, 'accha': 0.4, 'acha': 0.4, 'theek': 0.2,
    'thik': 0.2, 'thek': 0.2, 'bilkul': 0.3, 'zaroor': 0.3,
    'inshallah': 0.4, 'mashallah': 0.6, 'alhamdulillah': 0.6,
    'subhanallah': 0.5, 'jazakallah': 0.5, 'khair': 0.3,
    'mubarak': 0.6, 'congratulations': 0.7, 'congrats': 0.7,
    'bhai': 0.1, 'yaar': 0.1, 'dost': 0.2, 'boss': 0.2,
    'haha': 0.4, 'hehe': 0.4, 'hihi': 0.4, 'lol': 0.3,
    'lmao': 0.4, 'lmfao': 0.4, 'hahaha': 0.5, 'xd': 0.4,
    'khush': 0.6, 'khushi': 0.6, 'maja': 0.6, 'maza': 0.6,
    'mazaa': 0.6, 'fun': 0.5, 'awesome': 0.7, 'brilliant': 0.7,
    'ekdum': 0.1, 'sahi': 0.4, 'shi': 0.3, 'correct': 0.3,
    'perfect': 0.7, 'best': 0.6, 'great': 0.6, 'nice': 0.5,
    'superb': 0.7, 'excellent': 0.75, 'wonderful': 0.75,
    'happy': 0.6, 'love': 0.6, 'pyaar': 0.6, 'ily': 0.6,
    'welcome': 0.3, 'thanks': 0.4, 'thank': 0.4, 'shukriya': 0.4,
    'mehnat': 0.3, 'koshish': 0.3, 'try': 0.1,
    'aajao': 0.2, 'aajana': 0.2, 'milte': 0.2, 'milenge': 0.2,
    'carry on': 0.3, 'chalo': 0.1, 'chalte': 0.1,
}

# Hinglish negative words VADER misses or gets wrong
_HINGLISH_NEGATIVE = {
    'bakwaas': -0.7, 'bekar': -0.6, 'faltu': -0.6, 'bekaar': -0.6,
    'bura': -0.6, 'bura laga': -0.8, 'ganda': -0.5, 'kharab': -0.6, 'nafrat': -0.8,
    'gussa': -0.6, 'ghussa': -0.6, 'pareshaan': -0.5,
    'tang': -0.4, 'takleef': -0.5, 'taqleef': -0.5,
    'dukh': -0.6, 'dukha': -0.7, 'dard': -0.5, 'rona': -0.3, 'rota': -0.3,
    'band karo': -0.5, 'chup karo': -0.5, 'bas karo': -0.4,
    'bura laga': -0.6, 'afsos': -0.5, 'sharminda': -0.4,
    'dar': -0.4, 'darr': -0.4, 'bhay': -0.4,
    'bimaar': -0.4, 'tabiyat': -0.2, 'fever': -0.3,
    'problem': -0.3, 'mushkil': -0.4, 'dikkat': -0.4,
    'galat': -0.4, 'wrong': -0.4, 'mistake': -0.3,
    'sorry': -0.2, 'maafi': -0.2,
}

# Words that LOOK negative in VADER but are actually NEUTRAL/POSITIVE in WhatsApp context
# "chup" = quiet/hush (used casually), "nahi" = no (casual), "mat" = don't (casual instruction)
_WHATSAPP_CONTEXT_OVERRIDES = {
    # casual "no/don't" — not negative sentiment
    'nahi': 0.0, 'nhi': 0.0, 'nahin': 0.0, 'mat': 0.0, 'na': 0.0,
    'chup': 0.0, 'chup chaap': 0.0,
    # casual instructions — neutral
    'aa jaana': 0.0, 'aajana': 0.0, 'aana': 0.0, 'aao': 0.0,
    'dekho': 0.0, 'suno': 0.0, 'bolo': 0.0, 'batao': 0.0,
    # filler words
    'toh': 0.0, 'tou': 0.0, 'woh': 0.0, 'yeh': 0.0,
    'matlab': 0.0, 'kyun': 0.0, 'kya': 0.0, 'kaisa': 0.0,
    'abhi': 0.0, 'baad': 0.0, 'pehle': 0.0,
    'chhodo': 0.0, 'chodo': 0.0,   # "chhodo yaar" = let it go (neutral)
    'please': 0.0,                  # don't let 'please' boost negative sentences
}

# Emoji sentiment scores
_EMOJI_SENTIMENT = {
    '😂': 0.7, '🤣': 0.7, '😹': 0.6, '😆': 0.6, '😄': 0.6,
    '😃': 0.6, '😀': 0.5, '🙂': 0.3, '😊': 0.5, '🥰': 0.7,
    '😍': 0.7, '🤩': 0.7, '😎': 0.5, '👍': 0.5, '🔥': 0.4,
    '❤️': 0.7, '💕': 0.6, '💯': 0.6, '✅': 0.3, '🎉': 0.7,
    '🥳': 0.7, '🙌': 0.6, '👏': 0.5, '💪': 0.5, '🫡': 0.2,
    '😅': 0.1, '🤭': 0.2, '😏': 0.1, '🤔': 0.0, '😐': 0.0,
    '😑': -0.1, '🙄': -0.2, '😤': -0.4, '😠': -0.5, '😡': -0.6,
    '🤬': -0.7, '😢': -0.5, '😭': -0.4, '😔': -0.4, '😞': -0.5,
    '💔': -0.6, '😷': -0.2, '🤒': -0.3, '😰': -0.4, '😨': -0.4,
    '👎': -0.5, '🚫': -0.3, '❌': -0.3,
    # laughing emojis — always positive even if text has "negative" words
    '😂': 0.7, '🤣': 0.7,
}

# System/media messages to skip entirely
_SKIP_PATTERNS = [
    r'^\s*$',
    r'<media omitted>',
    r'image omitted',
    r'video omitted',
    r'audio omitted',
    r'document omitted',
    r'sticker omitted',
    r'gif omitted',
    r'contact card omitted',
    r'messages and calls are end-to-end',
    r'missed voice call',
    r'missed video call',
    r'this message was deleted',
    r'you deleted this message',
    r'^\+?\d[\d\s\-]{8,}$',  # phone numbers
]
_SKIP_RE = re.compile('|'.join(_SKIP_PATTERNS), re.IGNORECASE)

# Short/reaction messages that are hard to classify — treat as neutral
_VERY_SHORT_NEUTRAL = {'ok', 'okay', 'k', 'hmm', 'hm', 'oh', 'ah',
                        'ha', 'ji', 'haan', 'han', 'ho', 'hu',
                        'hn', 'hnn', 'ok ji', 'haan ji', 'theek hai',
                        'thik hai', '👍', '🙂', '😊', '.', '..', '...',
                        'accha', 'acha', 'achha', 'bilkul', 'sahi'}


def _extract_emojis(text: str) -> list:
    """Extract all emojis from text."""
    return [c for c in text if '\U0001F300' <= c <= '\U0001FFFF'
            or '\U00002600' <= c <= '\U000027BF'
            or '\U0001F900' <= c <= '\U0001F9FF']


def _has_laughing_emoji(text: str) -> bool:
    laughing = {'😂', '🤣', '😹', '😆', 'lol', 'lmao', 'lmfao', 'haha', 'hehe', 'xd'}
    text_lower = text.lower()
    return any(e in text_lower for e in laughing)


def _preprocess_whatsapp(text: str) -> str:
    """Clean WhatsApp-specific noise before sentiment analysis."""
    if not text or not isinstance(text, str):
        return ''
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove phone numbers
    text = re.sub(r'\+?\d[\d\s\-]{8,}', '', text)
    # Normalize repeated punctuation: "!!!!!!" → "!"
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    # Normalize repeated characters: "sooooo" → "sooo" (VADER handles elongation)
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
    return text.strip()


class SentimentAnalyzer:
    """
    WhatsApp-aware sentiment analyzer.

    Improvements over plain VADER:
    1. Hinglish lexicon boost (mast, zabardast, bakwaas, etc.)
    2. Emoji sentiment scoring
    3. Context overrides — casual "nahi/mat/chup" not treated as negative
    4. Laughing-emoji override — if message has 😂/lol, it's positive
    5. Skip system/media messages entirely
    6. Short reaction messages → neutral
    7. Adjusted thresholds for WhatsApp's informal style
    """

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        # Inject custom lexicon into VADER
        self.vader.lexicon.update(_HINGLISH_POSITIVE)
        self.vader.lexicon.update(_HINGLISH_NEGATIVE)
        # Context overrides (set to 0 so VADER doesn't penalise casual "nahi")
        self.vader.lexicon.update(_WHATSAPP_CONTEXT_OVERRIDES)

        self.transformer_available = False
        try:
            self.transformer_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.transformer_available = True
        except Exception as e:
            print(f"Warning: Transformer model not available: {e}")

    def _is_system_message(self, text: str) -> bool:
        return bool(_SKIP_RE.search(text.strip()))

    def _emoji_adjustment(self, text: str) -> float:
        """Compute net emoji sentiment contribution."""
        emojis = _extract_emojis(text)
        if not emojis:
            return 0.0
        scores = [_EMOJI_SENTIMENT.get(e, 0.0) for e in emojis]
        # Average, but cap contribution to avoid emoji spam dominating
        avg = sum(scores) / len(scores)
        return max(-0.4, min(0.4, avg * 0.6))

    def analyze_vader(self, text: str) -> dict:
        """
        WhatsApp-aware VADER analysis with Hinglish + emoji support.
        """
        neutral_result = {
            'positive': 0.0, 'negative': 0.0, 'neutral': 1.0,
            'compound': 0.0, 'label': 'NEUTRAL'
        }

        if not text or not isinstance(text, str):
            return neutral_result
        if self._is_system_message(text):
            return neutral_result

        clean = _preprocess_whatsapp(text)
        if not clean:
            return neutral_result

        # Very short / reaction messages → neutral
        if clean.lower().strip() in _VERY_SHORT_NEUTRAL or len(clean.split()) <= 1:
            # But still check for strong emojis
            emoji_adj = self._emoji_adjustment(clean)
            if abs(emoji_adj) > 0.2:
                compound = emoji_adj
                label = 'POSITIVE' if compound > 0 else 'NEGATIVE'
                return {**neutral_result, 'compound': round(compound, 4), 'label': label}
            return neutral_result

        # Laughing override — message is humorous/positive regardless of words
        if _has_laughing_emoji(clean):
            scores = self.vader.polarity_scores(clean)
            compound = max(0.3, scores['compound'])   # floor at +0.3
            return {
                'positive': max(scores['pos'], 0.4),
                'negative': min(scores['neg'], 0.1),
                'neutral': scores['neu'],
                'compound': round(compound, 4),
                'label': 'POSITIVE'
            }

        # Standard VADER
        scores = self.vader.polarity_scores(clean)
        compound = scores['compound']

        # Emoji adjustment on top
        emoji_adj = self._emoji_adjustment(text)
        compound = max(-1.0, min(1.0, compound + emoji_adj))

        # WhatsApp threshold: informal chats hover near 0 — use tighter bands
        # so casual neutral messages don't get flagged negative
        if compound >= 0.1:
            label = 'POSITIVE'
        elif compound <= -0.15:          # stricter negative threshold
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'

        return {
            'positive': round(scores['pos'], 4),
            'negative': round(scores['neg'], 4),
            'neutral':  round(scores['neu'], 4),
            'compound': round(compound, 4),
            'label':    label
        }

    def analyze_transformer(self, text: str) -> dict:
        if not self.transformer_available or not text or not isinstance(text, str):
            return {'label': 'NEUTRAL', 'score': 0.5}
        if self._is_system_message(text):
            return {'label': 'NEUTRAL', 'score': 0.5}
        try:
            result = self.transformer_pipeline(text[:512])[0]
            label = 'POSITIVE' if result['label'] == 'POSITIVE' else 'NEGATIVE'
            return {'label': label, 'score': result['score']}
        except Exception as e:
            print(f"Warning: Transformer analysis failed: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}

    def analyze_dataframe(self, df: pd.DataFrame, use_transformer: bool = False) -> pd.DataFrame:
        df = df.copy()
        col = 'message_cleaned' if 'message_cleaned' in df.columns else 'message'

        results = df[col].apply(lambda x: self.analyze_vader(str(x) if pd.notna(x) else ''))
        df['sentiment_vader']    = results.apply(lambda r: r['label'])
        df['sentiment_compound'] = results.apply(lambda r: r['compound'])
        df['sentiment_pos']      = results.apply(lambda r: r['positive'])
        df['sentiment_neg']      = results.apply(lambda r: r['negative'])
        df['sentiment_neu']      = results.apply(lambda r: r['neutral'])

        if use_transformer and self.transformer_available:
            tr = df[col].apply(lambda x: self.analyze_transformer(str(x) if pd.notna(x) else ''))
            df['sentiment_transformer']       = tr.apply(lambda r: r['label'])
            df['sentiment_transformer_score'] = tr.apply(lambda r: r['score'])

        return df

    def get_sentiment_distribution(self, df: pd.DataFrame) -> dict:
        if 'sentiment_vader' not in df.columns:
            return {}
        counts = df['sentiment_vader'].value_counts()
        total  = len(df)
        return {
            label: {
                'count': int(counts.get(label, 0)),
                'percentage': round((counts.get(label, 0) / total) * 100, 2)
            }
            for label in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        }

    def get_user_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'sentiment_compound' not in df.columns:
            return pd.DataFrame()
        user_sentiment = df.groupby('user').agg(
            avg_sentiment_score=('sentiment_compound', 'mean'),
            positivity_percentage=('sentiment_vader', lambda x: (x == 'POSITIVE').sum() / len(x) * 100),
            message_count=('sentiment_compound', 'count')
        ).round(2)
        return user_sentiment.sort_values('avg_sentiment_score', ascending=False)

    def get_sentiment_trend(self, df: pd.DataFrame, period: str = 'D') -> pd.DataFrame:
        if 'sentiment_compound' not in df.columns:
            return pd.DataFrame()
        trend = df.set_index('datetime').groupby(
            pd.Grouper(freq=period))['sentiment_compound'].mean()
        return trend.reset_index()
