import re
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Optional, Dict, List
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
    'afsos': -0.5, 'sharminda': -0.4,
    'dar': -0.4, 'darr': -0.4, 'bhay': -0.4,
    'bimaar': -0.4, 'tabiyat': -0.2, 'fever': -0.3,
    'problem': -0.3, 'mushkil': -0.4, 'dikkat': -0.4,
    'galat': -0.4, 'wrong': -0.4, 'mistake': -0.3,
    'sorry': -0.2, 'maafi': -0.2,
}

# Words that LOOK negative in VADER but are actually NEUTRAL/POSITIVE in WhatsApp context
_WHATSAPP_CONTEXT_OVERRIDES = {
    'nahi': 0.0, 'nhi': 0.0, 'nahin': 0.0, 'mat': 0.0, 'na': 0.0,
    'chup': 0.0, 'chup chaap': 0.0,
    'aa jaana': 0.0, 'aajana': 0.0, 'aana': 0.0, 'aao': 0.0,
    'dekho': 0.0, 'suno': 0.0, 'bolo': 0.0, 'batao': 0.0,
    'toh': 0.0, 'tou': 0.0, 'woh': 0.0, 'yeh': 0.0,
    'matlab': 0.0, 'kyun': 0.0, 'kya': 0.0, 'kaisa': 0.0,
    'abhi': 0.0, 'baad': 0.0, 'pehle': 0.0,
    'chhodo': 0.0, 'chodo': 0.0,
    'please': 0.0,
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
    r'^\+?\d[\d\s\-]{8,}$',
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
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\+?\d[\d\s\-]{8,}', '', text)
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
    return text.strip()

# Global cache for transformer pipeline to avoid repeated loading
_SENTIMENT_PIPELINE = None
_TRANSFORMER_TRIED = False

def _get_device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1

def get_sentiment_pipeline():
    """Lazy load and cache the transformer sentiment pipeline."""
    global _SENTIMENT_PIPELINE, _TRANSFORMER_TRIED
    if _SENTIMENT_PIPELINE is not None:
        return _SENTIMENT_PIPELINE
    if _TRANSFORMER_TRIED:
        return None

    _TRANSFORMER_TRIED = True
    try:
        from transformers import pipeline
        device = _get_device()
        _SENTIMENT_PIPELINE = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
            truncation=True,
            max_length=128
        )
        return _SENTIMENT_PIPELINE
    except Exception as e:
        print(f"Warning: Sentiment transformer model unavailable: {e}")
        return None


class SentimentAnalyzer:
    """
    WhatsApp-aware sentiment analyzer with Hinglish, emoji awareness, and fast inference.
    """

    def __init__(self, preload_transformer: bool = False):
        self.vader = SentimentIntensityAnalyzer()
        self.vader_analyzer = self.vader
        # Inject custom lexicon into VADER
        self.vader.lexicon.update(_HINGLISH_POSITIVE)
        self.vader.lexicon.update(_HINGLISH_NEGATIVE)
        self.vader.lexicon.update(_WHATSAPP_CONTEXT_OVERRIDES)

        self.transformer_available = False
        if preload_transformer:
            pipe = get_sentiment_pipeline()
            self.transformer_available = pipe is not None

    def _is_system_message(self, text: str) -> bool:
        return bool(_SKIP_RE.search(text.strip()))

    def _emoji_adjustment(self, text: str) -> float:
        """Compute net emoji sentiment contribution."""
        emojis = _extract_emojis(text)
        if not emojis:
            return 0.0
        scores = [_EMOJI_SENTIMENT.get(e, 0.0) for e in emojis]
        avg = sum(scores) / len(scores)
        return max(-0.4, min(0.4, avg * 0.6))

    def analyze_vader(self, text: str) -> dict:
        """WhatsApp-aware VADER analysis with Hinglish + emoji support."""
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

        # Very short / reaction messages -> neutral
        if clean.lower().strip() in _VERY_SHORT_NEUTRAL or len(clean.split()) <= 1:
            emoji_adj = self._emoji_adjustment(clean)
            if abs(emoji_adj) > 0.2:
                compound = emoji_adj
                label = 'POSITIVE' if compound > 0 else 'NEGATIVE'
                return {**neutral_result, 'compound': round(compound, 4), 'label': label}
            return neutral_result

        # Laughing override -> message is humorous/positive regardless of words
        if _has_laughing_emoji(clean):
            scores = self.vader.polarity_scores(clean)
            compound = max(0.3, scores['compound'])
            return {
                'positive': max(scores['pos'], 0.4),
                'negative': min(scores['neg'], 0.1),
                'neutral': scores['neu'],
                'compound': round(compound, 4),
                'label': 'POSITIVE'
            }

        scores = self.vader.polarity_scores(clean)
        compound = scores['compound']
        emoji_adj = self._emoji_adjustment(text)
        compound = max(-1.0, min(1.0, compound + emoji_adj))

        # WhatsApp thresholds
        if compound >= 0.1:
            label = 'POSITIVE'
        elif compound <= -0.15:
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
        pipe = get_sentiment_pipeline()
        if not pipe or not text or not isinstance(text, str):
            return {'label': 'NEUTRAL', 'score': 0.5}
        if self._is_system_message(text):
            return {'label': 'NEUTRAL', 'score': 0.5}
        try:
            result = pipe(text[:256])[0]
            label = 'POSITIVE' if result['label'].upper() == 'POSITIVE' else 'NEGATIVE'
            return {'label': label, 'score': round(float(result['score']), 3)}
        except Exception:
            return {'label': 'NEUTRAL', 'score': 0.5}

    def analyze_transformer_batch(self, texts: List[str], batch_size: int = 64) -> List[dict]:
        """Batched transformer sentiment inference."""
        pipe = get_sentiment_pipeline()
        if not pipe:
            return [{'label': 'NEUTRAL', 'score': 0.5} for _ in texts]

        results = []
        try:
            import torch
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = [t[:256] if t else "ok" for t in texts[i:i+batch_size]]
                    preds = pipe(batch)
                    for res in preds:
                        lbl = 'POSITIVE' if res['label'].upper() == 'POSITIVE' else 'NEGATIVE'
                        results.append({'label': lbl, 'score': round(float(res['score']), 3)})
        except Exception as e:
            print(f"Warning: Transformer batch inference error: {e}")
            while len(results) < len(texts):
                results.append({'label': 'NEUTRAL', 'score': 0.5})

        return results

    def analyze_dataframe(self, df: pd.DataFrame, use_transformer: bool = False, max_transformer_samples: int = 2000) -> pd.DataFrame:
        """Analyze sentiment for entire dataframe using WhatsApp-tuned VADER + optional Transformer."""
        if df.empty:
            df['sentiment_vader'] = []
            df['sentiment_compound'] = []
            df['sentiment_pos'] = []
            df['sentiment_neg'] = []
            df['sentiment_neu'] = []
            return df

        df = df.copy()
        col = 'message_cleaned' if 'message_cleaned' in df.columns else 'message'
        raw_texts = df[col].fillna('').astype(str).tolist()

        # Batch analyze with analyze_vader
        vader_results = [self.analyze_vader(t) for t in raw_texts]
        df['sentiment_vader']    = [r['label'] for r in vader_results]
        df['sentiment_compound'] = [r['compound'] for r in vader_results]
        df['sentiment_pos']      = [r['positive'] for r in vader_results]
        df['sentiment_neg']      = [r['negative'] for r in vader_results]
        df['sentiment_neu']      = [r['neutral'] for r in vader_results]

        if use_transformer:
            pipe = get_sentiment_pipeline()
            if pipe is not None:
                if len(raw_texts) > max_transformer_samples:
                    trans_results = self.analyze_transformer_batch(raw_texts[:max_transformer_samples])
                    default_res = {'label': 'NEUTRAL', 'score': 0.5}
                    trans_results.extend([default_res] * (len(raw_texts) - max_transformer_samples))
                else:
                    trans_results = self.analyze_transformer_batch(raw_texts)

                df['sentiment_transformer'] = [r['label'] for r in trans_results]
                df['sentiment_transformer_score'] = [r['score'] for r in trans_results]

        return df

    def get_sentiment_distribution(self, df: pd.DataFrame) -> dict:
        """Get sentiment counts and percentages safely."""
        if 'sentiment_vader' not in df.columns or df.empty:
            return {label: {'count': 0, 'percentage': 0.0} for label in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']}

        counts = df['sentiment_vader'].value_counts()
        total = max(len(df), 1)

        return {
            label: {
                'count': int(counts.get(label, 0)),
                'percentage': round((counts.get(label, 0) / total) * 100, 2)
            }
            for label in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        }

    def get_user_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get average sentiment per user safely."""
        if 'sentiment_compound' not in df.columns or df.empty or 'user' not in df.columns:
            return pd.DataFrame(columns=['avg_sentiment_score', 'positivity_percentage', 'message_count'])

        user_sentiment = df.groupby('user').agg(
            avg_sentiment_score=('sentiment_compound', 'mean'),
            positivity_percentage=('sentiment_vader', lambda x: (x == 'POSITIVE').sum() / max(len(x), 1) * 100),
            message_count=('sentiment_compound', 'count')
        ).round(2)
        return user_sentiment.sort_values('avg_sentiment_score', ascending=False)

    def get_sentiment_trend(self, df: pd.DataFrame, period: str = 'D') -> pd.DataFrame:
        """Get sentiment trend over time safely."""
        if 'sentiment_compound' not in df.columns or df.empty or 'datetime' not in df.columns:
            return pd.DataFrame(columns=['datetime', 'sentiment_compound'])

        trend = df.set_index('datetime').groupby(pd.Grouper(freq=period))['sentiment_compound'].mean()
        return trend.reset_index().fillna(0)
