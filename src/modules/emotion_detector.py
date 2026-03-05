"""
emotion_detector.py — Optimized: instant rule-based detection (no heavy ML model on startup)
Transformer available as opt-in upgrade only.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Keyword map for instant rule-based emotion detection ─────────────────
_EMOTION_KEYWORDS = {'joy': ['haha', 'lol', '😂', '😄', '😁', '😊', '🎉', 'happy', 'great', 'love', 'yay', 'awesome', 'amazing', 'wonderful', 'fantastic', 'congrats', '😍', '❤️', '🥳', 'khush', 'maza', 'maja', 'bahut acha', 'excellent', 'brilliant', 'superb'], 'anger': ['angry', 'hate', 'stupid', 'idiot', 'wtf', 'damn', 'ugh', '😠', '😡', '🤬', 'frustrated', 'annoyed', 'irritated', 'horrible', 'worst', 'terrible', 'bakwaas', 'bekar', 'galat', 'ganda', 'bura'], 'sadness': ['sad', 'miss', 'lonely', 'cry', '😢', '😭', '💔', 'depressed', 'unhappy', 'sorry', 'regret', 'wish', 'disappointed', 'hurt', 'dukh', 'udaas', 'afsos'], 'fear': ['scared', 'afraid', 'fear', 'worry', 'anxious', 'nervous', '😨', '😱', 'panic', 'frightened', 'stress', 'tense', 'darr', 'ghabra', 'tension', 'pareshan'], 'surprise': ['wow', 'omg', 'oh my', 'really', 'seriously', '😲', '😮', 'shocking', 'unexpected', 'unbelievable', 'whoa', 'arre', 'yaar sach', 'kya baat'], 'disgust': ['disgusting', 'gross', 'yuck', 'eww', '🤮', '🤢', 'horrible', 'nasty', 'awful', 'sick', 'ghatiya', 'chee']}

def _rule_based_emotion(text: str) -> str:
    """Zero-latency keyword emotion detection."""
    if not text or not isinstance(text, str):
        return 'neutral'
    t = text.lower()
    scores = {em: sum(1 for kw in kws if kw in t)
              for em, kws in _EMOTION_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'neutral'


class EmotionDetector:
    """
    Fast emotion detection using rule-based keywords.
    ML transformer available as opt-in (lazy loaded).
    """

    def __init__(self):
        self._pipeline = None
        self.available = False   # transformer availability

    def _load_transformer(self):
        """Load heavy ML model only when explicitly requested."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
            )
            self.available = True
        except Exception as e:
            print(f"Emotion transformer not available: {e}")
            self.available = False

    def detect_emotion(self, text: str) -> dict:
        """Fast rule-based emotion (default)."""
        return {'emotion': _rule_based_emotion(text), 'score': 1.0}

    def detect_emotion_ml(self, text: str) -> dict:
        """ML-based emotion — loads transformer on first call."""
        self._load_transformer()
        if not self.available or not text:
            return self.detect_emotion(text)
        try:
            result = self._pipeline(text[:512])[0]
            return {'emotion': result['label'].lower(), 'score': round(result['score'], 3)}
        except Exception:
            return self.detect_emotion(text)

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze full DataFrame — uses fast rule-based by default."""
        df = df.copy()
        results = df['message_cleaned'].apply(self.detect_emotion)
        df['emotion']       = results.apply(lambda x: x['emotion'])
        df['emotion_score'] = results.apply(lambda x: x['score'])
        return df

    def get_emotion_distribution(self, df: pd.DataFrame) -> dict:
        if 'emotion' not in df.columns:
            return {}
        counts = df['emotion'].value_counts()
        total  = max(len(df), 1)
        return {
            em.upper(): {
                'count': int(counts.get(em, 0)),
                'percentage': round(counts.get(em, 0) / total * 100, 2),
            }
            for em in ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust', 'neutral']
        }

    def get_user_emotion(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'emotion' not in df.columns:
            return pd.DataFrame()
        ue  = df.groupby(['user', 'emotion']).size().unstack(fill_value=0)
        return (ue.div(ue.sum(axis=1), axis=0) * 100).round(2)

    def get_emotion_trend(self, df: pd.DataFrame, period: str = 'D') -> pd.DataFrame:
        if 'emotion' not in df.columns:
            return pd.DataFrame()
        return df.set_index('datetime').groupby(
            [pd.Grouper(freq=period), 'emotion']
        ).size().unstack(fill_value=0).reset_index()

    def get_dominant_emotion(self, df: pd.DataFrame) -> str:
        if 'emotion' not in df.columns or len(df) == 0:
            return 'UNKNOWN'
        return df['emotion'].value_counts().index[0].upper()

    def get_emotion_intensity(self, df: pd.DataFrame) -> dict:
        if 'emotion_score' not in df.columns:
            return {}
        return {
            em.upper(): round(df[df['emotion'] == em]['emotion_score'].mean(), 3)
            for em in df['emotion'].unique()
        }