import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

_EMOTION_PIPELINE = None
_EMOTION_TRIED = False

def _get_device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1

def get_emotion_pipeline():
    """Lazy load and cache the transformer emotion pipeline."""
    global _EMOTION_PIPELINE, _EMOTION_TRIED
    if _EMOTION_PIPELINE is not None:
        return _EMOTION_PIPELINE
    if _EMOTION_TRIED:
        return None

    _EMOTION_TRIED = True
    try:
        from transformers import pipeline
        device = _get_device()
        _EMOTION_PIPELINE = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=device,
            truncation=True,
            max_length=128
        )
        return _EMOTION_PIPELINE
    except Exception as e:
        print(f"Warning: Emotion transformer model unavailable: {e}")
        return None


class EmotionDetector:
    """
    Detect emotions in WhatsApp messages.
    Emotions: Joy, Anger, Sadness, Fear, Surprise, Disgust, Neutral.
    Uses ultra-fast emoji/lexicon mapping for large datasets with cached Transformer support.
    """

    # Comprehensive Emoji to Emotion mapping
    _EMOJI_MAP = {
        'joy': set('😂🤣😄😃😀😁😊😍🥰😘🤗🤩🥳🙌🎉❤️💖✨👍🔥🕺💃'),
        'sadness': set('😢😭😞😔🥺😿💔☹️🙁😿😩'),
        'anger': set('😡😠🤬👿😤💢🖕'),
        'fear': set('😨😰😱🥶😬😳'),
        'surprise': set('😲😮😯🤯🙊👀✨'),
        'disgust': set('🤮🤢🤧🤒🥴'),
    }

    # High-precision keywords (English + Hinglish)
    _KEYWORD_MAP = {
        'joy': {'happy','yay','love','awesome','great','haha','lol','lmao','congrats','party','amazing','shandar','badhiya','khushi','mast','kya baat'},
        'sadness': {'sad','sorry','miss','crying','depressed','rip','hurt','lonely','lost','dukh','dard','rona','bechara'},
        'anger': {'angry','mad','stupid','idiot','shut up','hate','worst','annoying','gussa','bakwas','chutiya','pagal','kamine'},
        'fear': {'scared','fear','afraid','danger','panic','worry','nervous','darr','tension'},
        'surprise': {'wow','omg','unbelievable','shocked','really','whoa','wtf','sach','gazab','arre'},
        'disgust': {'eww','gross','disgusting','nasty','sick','chi','ganda'},
    }

    def __init__(self, preload_transformer: bool = False):
        self.available = False
        if preload_transformer:
            pipe = get_emotion_pipeline()
            self.available = pipe is not None

    def _fast_detect_emotion(self, text: str, emojis: str = "") -> dict:
        """Fast rule-based + emoji emotion detection for high-throughput analysis."""
        if not text and not emojis:
            return {'emotion': 'neutral', 'score': 0.5}

        # Check emojis first (highest emotional signal in chat)
        if emojis:
            for em_char in emojis:
                for emotion, em_set in self._EMOJI_MAP.items():
                    if em_char in em_set:
                        return {'emotion': emotion, 'score': 0.88}

        text_lower = text.lower() if text else ""
        words = set(re.findall(r'\b\w+\b', text_lower))

        # Check keyword matches
        for emotion, kw_set in self._KEYWORD_MAP.items():
            if words & kw_set:
                return {'emotion': emotion, 'score': 0.80}

        return {'emotion': 'neutral', 'score': 0.5}

    def analyze_dataframe(self, df: pd.DataFrame, use_transformer: bool = False, max_transformer_samples: int = 1000) -> pd.DataFrame:
        """
        Analyze emotions for entire dataframe.
        Blends instant heuristic detection with batched Transformer models.
        """
        if df.empty:
            df['emotion'] = []
            df['emotion_score'] = []
            return df

        df = df.copy()
        msg_col = 'message_cleaned' if 'message_cleaned' in df.columns else 'message'
        raw_msgs = df[msg_col].fillna('').astype(str).tolist()
        emojis_col = df['emojis'].fillna('').astype(str).tolist() if 'emojis' in df.columns else [''] * len(df)

        # Run high-speed emotion detection
        fast_results = [
            self._fast_detect_emotion(txt, emo)
            for txt, emo in zip(raw_msgs, emojis_col)
        ]

        emotions = [r['emotion'] for r in fast_results]
        scores = [r['score'] for r in fast_results]

        # If user explicitly enables transformers and dataset is within limit
        if use_transformer:
            pipe = get_emotion_pipeline()
            if pipe is not None:
                sample_count = min(len(raw_msgs), max_transformer_samples)
                try:
                    import torch
                    with torch.no_grad():
                        batch_size = 64
                        for i in range(0, sample_count, batch_size):
                            batch = [t[:256] if t else "ok" for t in raw_msgs[i:i+batch_size]]
                            preds = pipe(batch)
                            for j, res in enumerate(preds):
                                idx = i + j
                                emotions[idx] = res['label'].lower()
                                scores[idx] = round(float(res['score']), 3)
                except Exception as e:
                    print(f"Warning: Emotion transformer batch failed: {e}")

        df['emotion'] = emotions
        df['emotion_score'] = scores
        return df

    def get_emotion_distribution(self, df: pd.DataFrame) -> dict:
        """Get emotion counts and percentages safely."""
        if 'emotion' not in df.columns or df.empty:
            return {
                e.upper(): {'count': 0, 'percentage': 0.0}
                for e in ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust', 'neutral']
            }

        counts = df['emotion'].value_counts()
        total = max(len(df), 1)

        result = {}
        for emotion in ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust', 'neutral']:
            count = int(counts.get(emotion, 0))
            result[emotion.upper()] = {
                'count': count,
                'percentage': round((count / total) * 100, 2)
            }

        return result

    def get_user_emotion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get average emotion per user safely."""
        if 'emotion' not in df.columns or df.empty or 'user' not in df.columns:
            return pd.DataFrame()

        user_emotions = df.groupby(['user', 'emotion']).size().unstack(fill_value=0)
        row_sums = user_emotions.sum(axis=1).replace(0, 1)
        user_emotions_pct = user_emotions.div(row_sums, axis=0) * 100
        return user_emotions_pct.round(2)

    def get_emotion_trend(self, df: pd.DataFrame, period: str = 'D') -> pd.DataFrame:
        """Get emotion trend over time safely."""
        if 'emotion' not in df.columns or df.empty or 'datetime' not in df.columns:
            return pd.DataFrame(columns=['datetime'])

        trend = df.set_index('datetime').groupby([pd.Grouper(freq=period), 'emotion']).size().unstack(fill_value=0)
        return trend.reset_index().fillna(0)

    def get_dominant_emotion(self, df: pd.DataFrame) -> str:
        """Get most common emotion safely."""
        if 'emotion' not in df.columns or df.empty:
            return 'NEUTRAL'
        vc = df['emotion'].value_counts()
        return vc.index[0].upper() if len(vc) > 0 else 'NEUTRAL'

    def get_emotion_intensity(self, df: pd.DataFrame) -> dict:
        """Get average intensity of emotions safely."""
        if 'emotion_score' not in df.columns or df.empty:
            return {}

        result = {}
        for emotion in df['emotion'].dropna().unique():
            mask = df['emotion'] == emotion
            avg_score = df[mask]['emotion_score'].mean()
            result[str(emotion).upper()] = round(float(avg_score), 3) if not np.isnan(avg_score) else 0.5

        return result