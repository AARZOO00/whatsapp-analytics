import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Optional, Dict, List
import warnings

warnings.filterwarnings('ignore')

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
    Perform sentiment analysis using VADER and Transformers.
    Optimized for single-pass execution and batched inference.
    """

    def __init__(self, preload_transformer: bool = False):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.transformer_available = False
        if preload_transformer:
            pipe = get_sentiment_pipeline()
            self.transformer_available = pipe is not None

    def analyze_vader(self, text: str) -> dict:
        """VADER sentiment analysis for a single text."""
        if not text or not isinstance(text, str):
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0, 'label': 'NEUTRAL'}

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
            print(f"Warning: Transformer batch inference failed: {e}")
            while len(results) < len(texts):
                results.append({'label': 'NEUTRAL', 'score': 0.5})

        return results

    def analyze_dataframe(self, df: pd.DataFrame, use_transformer: bool = False, max_transformer_samples: int = 2000) -> pd.DataFrame:
        """
        Analyze sentiment for entire dataframe.
        Single-pass VADER computation + optional batched Transformer inference.
        """
        if df.empty:
            df['sentiment_vader'] = []
            df['sentiment_compound'] = []
            df['sentiment_pos'] = []
            df['sentiment_neg'] = []
            df['sentiment_neu'] = []
            return df

        df = df.copy()
        msg_col = 'message_cleaned' if 'message_cleaned' in df.columns else 'message'
        texts = df[msg_col].fillna('').astype(str).tolist()

        # ── Fast Single-Pass VADER ─────────────────────────────────────────
        analyzer = self.vader_analyzer.polarity_scores
        scores = [analyzer(t) if t else {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0} for t in texts]

        compounds = [s['compound'] for s in scores]
        df['sentiment_compound'] = compounds
        df['sentiment_pos'] = [s['pos'] for s in scores]
        df['sentiment_neg'] = [s['neg'] for s in scores]
        df['sentiment_neu'] = [s['neu'] for s in scores]

        # Vectorized label assignment
        df['sentiment_vader'] = np.where(
            df['sentiment_compound'] >= 0.05, 'POSITIVE',
            np.where(df['sentiment_compound'] <= -0.05, 'NEGATIVE', 'NEUTRAL')
        )

        # ── Batched Transformer Analysis ──────────────────────────────────
        if use_transformer:
            pipe = get_sentiment_pipeline()
            if pipe is not None:
                # If chat is massive, protect against freezing on CPU
                if len(texts) > max_transformer_samples:
                    # Analyze first N and sample rest, or run batch on all if manageable
                    trans_results = self.analyze_transformer_batch(texts[:max_transformer_samples])
                    default_res = {'label': 'NEUTRAL', 'score': 0.5}
                    trans_results.extend([default_res] * (len(texts) - max_transformer_samples))
                else:
                    trans_results = self.analyze_transformer_batch(texts)

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
        if 'sentiment_compound' not in df.columns or df.empty:
            return pd.DataFrame(columns=['avg_sentiment_score', 'positivity_percentage', 'message_count'])

        user_sentiment = df.groupby('user').agg({
            'sentiment_compound': 'mean',
            'sentiment_vader': lambda x: (x == 'POSITIVE').sum() / max(len(x), 1) * 100,
            'user': 'count'
        }).round(2)

        user_sentiment.columns = ['avg_sentiment_score', 'positivity_percentage', 'message_count']
        return user_sentiment.sort_values('avg_sentiment_score', ascending=False)

    def get_sentiment_trend(self, df: pd.DataFrame, period: str = 'D') -> pd.DataFrame:
        """Get sentiment trend over time safely."""
        if 'sentiment_compound' not in df.columns or df.empty or 'datetime' not in df.columns:
            return pd.DataFrame(columns=['datetime', 'sentiment_compound'])

        trend = df.set_index('datetime').groupby(pd.Grouper(freq=period))['sentiment_compound'].mean()
        return trend.reset_index().fillna(0)