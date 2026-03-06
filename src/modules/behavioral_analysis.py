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

class BehavioralAnalyzer:
    """
    Analyze user behavior patterns.
    Toxicity detection, user ranking, sentiment trends.
    """

    def __init__(self):
        try:
            self.toxicity_pipeline = pipeline(
                "text-classification",
                model="michellejieli/TOXIC_BERT"
            )
            self.toxicity_available = True
        except Exception as e:
            print(f"Warning: Toxicity model not available: {e}")
            self.toxicity_available = False

    def detect_toxicity(self, text: str) -> dict:
        """
        Detect toxicity/abusive content in text.

        Args:
            text: Input text

        Returns:
            Dict with toxicity label and score
        """
        if not self.toxicity_available or not text or not isinstance(text, str):
            return {'is_toxic': False, 'score': 0.0}

        try:
            result = self.toxicity_pipeline(text[:512])[0]
            is_toxic = result['label'].lower() == 'toxic'
            score = result['score'] if is_toxic else 1 - result['score']

            return {'is_toxic': is_toxic, 'score': round(score, 3)}
        except Exception as e:
            print(f"Warning: Toxicity detection failed: {e}")
            return {'is_toxic': False, 'score': 0.0}

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add behavioral metrics to dataframe.

        Args:
            df: DataFrame with messages

        Returns:
            DataFrame with toxicity and behavioral columns
        """
        df = df.copy()

        toxicity = df['message_cleaned'].apply(lambda x: self.detect_toxicity(x))
        df['is_toxic'] = toxicity.apply(lambda x: x['is_toxic'])
        df['toxicity_score'] = toxicity.apply(lambda x: x['score'])

        df['response_time'] = df.groupby('user')['datetime'].diff().dt.total_seconds() / 60

        df['caps_ratio'] = df['message'].apply(self._calculate_caps_ratio)

        df['question_asked'] = df['message'].apply(lambda x: '?' in x)

        df['exclamation'] = df['message'].apply(lambda x: '!' in x)

        return df

    def _calculate_caps_ratio(self, text: str) -> float:
        """Calculate ratio of uppercase letters"""
        if not text or not isinstance(text, str):
            return 0.0

        letters = [c for c in text if c.isalpha()]

        if not letters:
            return 0.0

        caps = sum(1 for c in letters if c.isupper())

        return round(caps / len(letters), 3)

    def get_toxicity_stats(self, df: pd.DataFrame) -> dict:
        """Get toxicity statistics"""
        if 'is_toxic' not in df.columns:
            return {}

        total = len(df)
        toxic_count = df['is_toxic'].sum()

        return {
            'total_messages': total,
            'toxic_messages': toxic_count,
            'toxic_percentage': round((toxic_count / total) * 100, 2) if total > 0 else 0,
            'avg_toxicity_score': round(df['toxicity_score'].mean(), 3)
        }

    def get_user_positivity_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank users by positivity.
        Considers sentiment, emotion, and toxicity.
        """
        if 'sentiment_compound' not in df.columns:
            return pd.DataFrame()

        user_stats = df.groupby('user').agg({
            'sentiment_compound': 'mean',
            'is_toxic': lambda x: (x == False).sum() / len(x) * 100,
            'user': 'count'
        }).round(2)

        user_stats.columns = ['avg_sentiment', 'non_toxic_percentage', 'message_count']

        user_stats['positivity_score'] = (
            user_stats['avg_sentiment'] * 0.6 +
            (user_stats['non_toxic_percentage'] / 100) * 0.4
        ).round(3)

        return user_stats.sort_values('positivity_score', ascending=False)

    def get_activity_patterns(self, df: pd.DataFrame) -> dict:
        """Get user activity patterns"""
        patterns = {}

        for user in df['user'].unique():
            user_df = df[df['user'] == user]

            patterns[user] = {
                'message_count': len(user_df),
                'avg_message_length': round(user_df['message_length'].mean(), 2),
                'question_percentage': round((user_df['question_asked'].sum() / len(user_df)) * 100, 2),
                'exclamation_percentage': round((user_df['exclamation'].sum() / len(user_df)) * 100, 2),
                'caps_ratio': round(user_df['caps_ratio'].mean(), 3),
                'avg_toxicity': round(user_df['toxicity_score'].mean(), 3),
                'most_active_hour': int(user_df['datetime'].dt.hour.mode()[0]) if len(user_df) > 0 else 0
            }

        return patterns

    def get_sentiment_trend_per_user(self, df: pd.DataFrame, period: str = 'W') -> dict:
        """
        Get sentiment trend for each user over time.

        Args:
            df: DataFrame
            period: 'D' for daily, 'W' for weekly, 'M' for monthly

        Returns:
            Dict with sentiment trends per user
        """
        if 'sentiment_compound' not in df.columns:
            return {}

        trends = {}

        for user in df['user'].unique():
            user_df = df[df['user'] == user].copy()
            user_df = user_df.set_index('datetime')

            trend = user_df.groupby(pd.Grouper(freq=period))['sentiment_compound'].mean()

            trends[user] = trend.reset_index()

        return trends

    def get_conversation_health(self, df: pd.DataFrame) -> dict:
        """
        Overall conversation health score.
        Based on toxicity, sentiment, and engagement.
        """
        if 'sentiment_compound' not in df.columns or 'is_toxic' not in df.columns:
            return {}

        avg_sentiment = df['sentiment_compound'].mean()
        toxic_ratio = df['is_toxic'].sum() / len(df) if len(df) > 0 else 0
        user_diversity = df['user'].nunique() / len(df) if len(df) > 0 else 0

        health_score = (
            (avg_sentiment + 1) / 2 * 0.4 +
            (1 - toxic_ratio) * 0.4 +
            min(user_diversity * 10, 1.0) * 0.2
        )

        health_score = round(max(0, min(1, health_score)), 3)

        if health_score >= 0.75:
            status = 'Excellent'
        elif health_score >= 0.5:
            status = 'Good'
        elif health_score >= 0.25:
            status = 'Fair'
        else:
            status = 'Poor'

        return {
            'health_score': health_score,
            'status': status,
            'avg_sentiment': round(avg_sentiment, 3),
            'toxic_percentage': round(toxic_ratio * 100, 2),
            'unique_users': df['user'].nunique()
        }