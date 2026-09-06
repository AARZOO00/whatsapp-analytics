import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

_TOXICITY_PIPELINE = None
_TOXICITY_TRIED = False

def _get_device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1

def get_toxicity_pipeline():
    """Lazy load and cache the toxic BERT pipeline."""
    global _TOXICITY_PIPELINE, _TOXICITY_TRIED
    if _TOXICITY_PIPELINE is not None:
        return _TOXICITY_PIPELINE
    if _TOXICITY_TRIED:
        return None

    _TOXICITY_TRIED = True
    try:
        from transformers import pipeline
        device = _get_device()
        _TOXICITY_PIPELINE = pipeline(
            "text-classification",
            model="michellejieli/TOXIC_BERT",
            device=device,
            truncation=True,
            max_length=128
        )
        return _TOXICITY_PIPELINE
    except Exception as e:
        print(f"Warning: Toxicity transformer model unavailable: {e}")
        return None


class BehavioralAnalyzer:
    """
    Analyze user behavior patterns.
    Toxicity detection, user ranking, communication style, response dynamics.
    Optimized for high-speed analysis and zero UI lockup.
    """

    _TOXIC_WORDS = [
        'fuck','shit','bitch','bastard','asshole','damn','crap','idiot','stupid',
        'hate','kill','die','loser','moron','dumb','wtf','stfu','kys',
        'madarchod','bhosdike','chutiya','saala','harami','gaandu','randi',
        'sala','kamina','kutte','gandu','bakwaas','bc','mc','bhosdi',
        'abuse','scam','fraud','liar','useless','worthless','pathetic',
    ]
    _TOXIC_RE = re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in _TOXIC_WORDS) + r')\b', re.IGNORECASE)

    def __init__(self, preload_transformer: bool = False):
        self.toxicity_available = False
        if preload_transformer:
            pipe = get_toxicity_pipeline()
            self.toxicity_available = pipe is not None

    def detect_toxicity_fast(self, text: str) -> dict:
        """Fast regex-based toxicity detection."""
        if not text or not isinstance(text, str):
            return {'is_toxic': False, 'score': 0.0}

        matches = self._TOXIC_RE.findall(text)
        if matches:
            score = round(min(0.55 + len(matches) * 0.15, 0.99), 3)
            return {'is_toxic': True, 'score': score}
        return {'is_toxic': False, 'score': 0.0}

    def analyze_dataframe(self, df: pd.DataFrame, use_transformer: bool = False, max_transformer_samples: int = 1000) -> pd.DataFrame:
        """
        Add behavioral metrics to dataframe with vectorized performance.
        """
        if df.empty:
            for col in ['is_toxic', 'toxicity_score', 'response_time', 'caps_ratio', 'question_asked', 'exclamation']:
                df[col] = []
            return df

        df = df.copy()
        msg_col = 'message_cleaned' if 'message_cleaned' in df.columns else 'message'
        raw_msgs = df[msg_col].fillna('').astype(str).tolist()
        orig_msgs = df['message'].fillna('').astype(str).tolist()

        # Fast rule-based toxicity check
        results = [self.detect_toxicity_fast(t) for t in raw_msgs]
        is_toxic_list = [r['is_toxic'] for r in results]
        tox_score_list = [r['score'] for r in results]

        # Optional Transformer verification if enabled
        if use_transformer:
            pipe = get_toxicity_pipeline()
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
                                is_t = res['label'].lower() == 'toxic'
                                is_toxic_list[idx] = is_t
                                tox_score_list[idx] = round(float(res['score']), 3) if is_t else round(1.0 - float(res['score']), 3)
                except Exception as e:
                    print(f"Warning: Toxicity transformer batch failed: {e}")

        df['is_toxic'] = is_toxic_list
        df['toxicity_score'] = tox_score_list

        # True conversation response time (time between different consecutive speakers)
        df_sorted = df.sort_values('datetime')
        prev_user = df_sorted['user'].shift(1)
        prev_time = df_sorted['datetime'].shift(1)
        time_diff = (df_sorted['datetime'] - prev_time).dt.total_seconds() / 60.0

        # Only count when user changed and response was within 120 minutes (conversation reply)
        valid_response = (df_sorted['user'] != prev_user) & (time_diff <= 120) & (time_diff >= 0)
        df['response_time'] = np.where(valid_response, time_diff, np.nan)

        # Fast vectorized caps ratio
        def _calc_caps(s: str) -> float:
            alphas = sum(1 for c in s if c.isalpha())
            if alphas == 0:
                return 0.0
            return round(sum(1 for c in s if c.isupper()) / alphas, 3)

        df['caps_ratio'] = [_calc_caps(m) for m in orig_msgs]
        df['question_asked'] = df['message'].str.contains(r'\?', regex=True, na=False)
        df['exclamation'] = df['message'].str.contains(r'!', regex=True, na=False)

        return df

    def get_toxicity_stats(self, df: pd.DataFrame) -> dict:
        """Get toxicity statistics safely."""
        if 'is_toxic' not in df.columns or df.empty:
            return {
                'total_messages': 0, 'toxic_messages': 0,
                'toxic_percentage': 0.0, 'avg_toxicity_score': 0.0
            }

        total = max(len(df), 1)
        toxic_count = int(df['is_toxic'].sum())
        avg_score = float(df['toxicity_score'].mean()) if 'toxicity_score' in df.columns else 0.0

        return {
            'total_messages': len(df),
            'toxic_messages': toxic_count,
            'toxic_percentage': round((toxic_count / total) * 100, 2),
            'avg_toxicity_score': round(avg_score if not np.isnan(avg_score) else 0.0, 3)
        }

    def get_user_positivity_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank users by positivity safely."""
        if 'sentiment_compound' not in df.columns or df.empty or 'user' not in df.columns:
            return pd.DataFrame()

        user_stats = df.groupby('user').agg(
            avg_sentiment=('sentiment_compound', 'mean'),
            non_toxic_percentage=('is_toxic', lambda x: ((x == False).sum() / max(len(x), 1) * 100) if 'is_toxic' in df.columns else 100.0),
            message_count=('user', 'count')
        ).round(2)

        user_stats['positivity_score'] = (
            ((user_stats['avg_sentiment'].fillna(0) + 1) / 2) * 0.6 +
            (user_stats['non_toxic_percentage'].fillna(100) / 100) * 0.4
        ).round(3)

        return user_stats.sort_values('positivity_score', ascending=False)

    def get_activity_patterns(self, df: pd.DataFrame) -> dict:
        """Get user activity patterns safely."""
        if df.empty or 'user' not in df.columns:
            return {}

        patterns = {}
        for user, u_df in df.groupby('user'):
            if u_df.empty:
                continue
            u_len = len(u_df)
            hours = u_df['datetime'].dt.hour.dropna()
            mode_hour = int(hours.mode()[0]) if len(hours) > 0 else 12

            patterns[user] = {
                'message_count': u_len,
                'avg_message_length': round(float(u_df['message_length'].mean()), 2) if 'message_length' in u_df.columns else 0,
                'question_percentage': round(float((u_df['question_asked'].sum() / u_len) * 100), 2) if 'question_asked' in u_df.columns else 0,
                'exclamation_percentage': round(float((u_df['exclamation'].sum() / u_len) * 100), 2) if 'exclamation' in u_df.columns else 0,
                'caps_ratio': round(float(u_df['caps_ratio'].mean()), 3) if 'caps_ratio' in u_df.columns else 0,
                'avg_toxicity': round(float(u_df['toxicity_score'].mean()), 3) if 'toxicity_score' in u_df.columns else 0,
                'most_active_hour': mode_hour
            }

        return patterns

    def get_sentiment_trend_per_user(self, df: pd.DataFrame, period: str = 'W') -> dict:
        """Get sentiment trend for each user over time safely."""
        if 'sentiment_compound' not in df.columns or df.empty or 'user' not in df.columns:
            return {}

        trends = {}
        for user, u_df in df.groupby('user'):
            if len(u_df) > 1 and 'datetime' in u_df.columns:
                trend = u_df.set_index('datetime').groupby(pd.Grouper(freq=period))['sentiment_compound'].mean()
                trends[user] = trend.reset_index().fillna(0)

        return trends

    def get_conversation_health(self, df: pd.DataFrame) -> dict:
        """Overall conversation health score safely."""
        if df.empty or 'sentiment_compound' not in df.columns:
            return {
                'health_score': 0.8, 'status': 'Good',
                'avg_sentiment': 0.0, 'toxic_percentage': 0.0, 'unique_users': 0
            }

        avg_sent = float(df['sentiment_compound'].mean())
        if np.isnan(avg_sent):
            avg_sent = 0.0

        toxic_ratio = float(df['is_toxic'].sum() / max(len(df), 1)) if 'is_toxic' in df.columns else 0.0
        user_div = float(df['user'].nunique() / max(len(df), 1)) if 'user' in df.columns else 1.0

        health_score = (
            ((avg_sent + 1) / 2) * 0.4 +
            (1 - min(toxic_ratio, 1.0)) * 0.4 +
            min(user_div * 10, 1.0) * 0.2
        )
        health_score = round(max(0.0, min(1.0, health_score)), 3)

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
            'avg_sentiment': round(avg_sent, 3),
            'toxic_percentage': round(toxic_ratio * 100, 2),
            'unique_users': df['user'].nunique() if 'user' in df.columns else 0
        }