import pandas as pd
import numpy as np
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

class EmotionDetector:
    """
    Detect emotions in WhatsApp messages.
    Emotions: Joy, Anger, Sadness, Fear, Surprise, Disgust
    """

    def __init__(self):
        try:
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base"
            )
            self.available = True
        except Exception as e:
            print(f"Warning: Emotion model not available: {e}")
            self.available = False

    def detect_emotion(self, text: str) -> dict:
        """
        Detect emotion in text.

        Args:
            text: Input text

        Returns:
            Dict with emotion label and score
        """
        if not self.available or not text or not isinstance(text, str):
            return {'emotion': 'neutral', 'score': 0.5}

        try:
            result = self.emotion_pipeline(text[:512])[0]
            return {
                'emotion': result['label'].lower(),
                'score': round(result['score'], 3)
            }
        except Exception as e:
            print(f"Warning: Emotion detection failed: {e}")
            return {'emotion': 'neutral', 'score': 0.5}

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze emotions for entire dataframe.

        Args:
            df: DataFrame with cleaned messages

        Returns:
            DataFrame with emotion columns added
        """
        df = df.copy()

        emotions = df['message_cleaned'].apply(lambda x: self.detect_emotion(x))

        df['emotion'] = emotions.apply(lambda x: x['emotion'])
        df['emotion_score'] = emotions.apply(lambda x: x['score'])

        return df

    def get_emotion_distribution(self, df: pd.DataFrame) -> dict:
        """Get emotion counts and percentages"""
        if 'emotion' not in df.columns:
            return {}

        counts = df['emotion'].value_counts()
        total = len(df)

        result = {}
        for emotion in ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust', 'neutral']:
            count = counts.get(emotion, 0)
            result[emotion.upper()] = {
                'count': count,
                'percentage': round((count / total) * 100, 2) if total > 0 else 0
            }

        return result

    def get_user_emotion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get average emotion per user"""
        if 'emotion' not in df.columns:
            return pd.DataFrame()

        user_emotions = df.groupby(['user', 'emotion']).size().unstack(fill_value=0)

        user_emotions_pct = user_emotions.div(user_emotions.sum(axis=1), axis=0) * 100

        return user_emotions_pct.round(2)

    def get_emotion_trend(self, df: pd.DataFrame, period: str = 'D') -> pd.DataFrame:
        """
        Get emotion trend over time.

        Args:
            df: DataFrame
            period: 'D' for daily, 'W' for weekly

        Returns:
            DataFrame with emotion trend
        """
        if 'emotion' not in df.columns:
            return pd.DataFrame()

        trend = df.set_index('datetime').groupby([pd.Grouper(freq=period), 'emotion']).size().unstack(fill_value=0)

        return trend.reset_index()

    def get_dominant_emotion(self, df: pd.DataFrame) -> str:
        """Get most common emotion"""
        if 'emotion' not in df.columns:
            return 'UNKNOWN'

        return df['emotion'].value_counts().index[0].upper() if len(df) > 0 else 'UNKNOWN'

    def get_emotion_intensity(self, df: pd.DataFrame) -> dict:
        """Get average intensity of emotions"""
        if 'emotion_score' not in df.columns:
            return {}

        result = {}
        for emotion in df['emotion'].unique():
            mask = df['emotion'] == emotion
            avg_score = df[mask]['emotion_score'].mean()
            result[emotion.upper()] = round(avg_score, 3)

        return result