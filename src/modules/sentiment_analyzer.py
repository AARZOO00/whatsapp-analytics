import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings

warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """
    Perform sentiment analysis using VADER and Transformers.
    """

    def __init__(self):
        self.vader_analyzer        = SentimentIntensityAnalyzer()
        self._transformer_pipeline = None   # lazy loaded
        self.transformer_available = False

    def _load_transformer(self):
        """Load transformer only when explicitly needed."""
        if self._transformer_pipeline is not None:
            return
        try:
            self._transformer_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
            )
            self.transformer_available = True
        except Exception as e:
            print(f"Transformer not available: {e}")
            self.transformer_available = False

    def analyze_vader(self, text: str) -> dict:
        """
        VADER sentiment analysis.

        Args:
            text: Input text

        Returns:
            Dict with sentiment scores
        """
        if not text or not isinstance(text, str):
            return {'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0, 'label': 'NEUTRAL'}

        scores = self.vader_analyzer.polarity_scores(text)

        compound = scores['compound']

        if compound >= 0.05:
            label = 'POSITIVE'
        elif compound <= -0.05:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'

        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': compound,
            'label': label
        }

    def analyze_transformer(self, text: str) -> dict:
        """Transformer-based sentiment — lazy loaded on first call."""
        self._load_transformer()
        if not self.transformer_available or not text or not isinstance(text, str):
            return {'label': 'NEUTRAL', 'score': 0.5}
        try:
            result = self._transformer_pipeline(text[:512])[0]
            label = 'POSITIVE' if result['label'] == 'POSITIVE' else 'NEGATIVE'
            score = result['score']

            return {'label': label, 'score': score}
        except Exception as e:
            print(f"Warning: Transformer analysis failed: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}

    def analyze_dataframe(self, df: pd.DataFrame, use_transformer: bool = False) -> pd.DataFrame:
        """
        Analyze sentiment for entire dataframe.

        Args:
            df: DataFrame with cleaned messages
            use_transformer: Whether to use transformer model

        Returns:
            DataFrame with sentiment columns added
        """
        df = df.copy()

        # Run VADER once per row (5x faster than calling 5 separate apply)
        vader_results = df['message_cleaned'].apply(self.analyze_vader)
        df['sentiment_vader']    = vader_results.apply(lambda x: x['label'])
        df['sentiment_compound'] = vader_results.apply(lambda x: x['compound'])
        df['sentiment_pos']      = vader_results.apply(lambda x: x['positive'])
        df['sentiment_neg']      = vader_results.apply(lambda x: x['negative'])
        df['sentiment_neu']      = vader_results.apply(lambda x: x['neutral'])

        if use_transformer:
            self._load_transformer()
            if self.transformer_available:
                tr = df['message_cleaned'].apply(self.analyze_transformer)
                df['sentiment_transformer']       = tr.apply(lambda x: x['label'])
                df['sentiment_transformer_score'] = tr.apply(lambda x: x['score'])

        return df

    def get_sentiment_distribution(self, df: pd.DataFrame) -> dict:
        """Get sentiment counts and percentages"""
        if 'sentiment_vader' not in df.columns:
            return {}

        counts = df['sentiment_vader'].value_counts()
        total = len(df)

        return {
            label: {
                'count': counts.get(label, 0),
                'percentage': round((counts.get(label, 0) / total) * 100, 2)
            }
            for label in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        }

    def get_user_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get average sentiment per user"""
        if 'sentiment_compound' not in df.columns:
            return pd.DataFrame()

        user_sentiment = df.groupby('user').agg({
            'sentiment_compound': 'mean',
            'sentiment_vader': lambda x: (x == 'POSITIVE').sum() / len(x) * 100,
            'user': 'count'
        }).round(2)

        user_sentiment.columns = ['avg_sentiment_score', 'positivity_percentage', 'message_count']

        return user_sentiment.sort_values('avg_sentiment_score', ascending=False)

    def get_sentiment_trend(self, df: pd.DataFrame, period: str = 'D') -> pd.DataFrame:
        """
        Get sentiment trend over time.

        Args:
            df: DataFrame
            period: 'D' for daily, 'W' for weekly, 'M' for monthly

        Returns:
            DataFrame with sentiment trend
        """
        if 'sentiment_compound' not in df.columns:
            return pd.DataFrame()

        trend = df.set_index('datetime').groupby(pd.Grouper(freq=period))['sentiment_compound'].mean()

        return trend.reset_index()