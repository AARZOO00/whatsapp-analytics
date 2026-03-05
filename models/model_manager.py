import streamlit as st
import pandas as pd
import time
from typing import Dict, Tuple
from src.modules.sentiment_analyzer import SentimentAnalyzer

class ModelManager:
    """Manage multiple sentiment models and comparison."""

    def __init__(self):
        self.models = {
            'VADER': 'Fast rule-based sentiment',
            'Transformer': 'Accurate DistilBERT sentiment',
            'Multilingual': 'Multilingual BERT sentiment',
            'Hybrid': 'Combined VADER + Transformer'
        }
        self.results = {}

    @st.cache_resource
    def get_sentiment_analyzer(_self):
        """Cache sentiment analyzer."""
        return SentimentAnalyzer()

    def render_model_selector(self) -> str:
        """Render model selection UI and return selected model."""
        col1, col2 = st.columns([3, 1])

        with col1:
            selected_model = st.selectbox(
                "Select Sentiment Model",
                list(self.models.keys()),
                help="Choose sentiment analysis model",
                key="model_selector"
            )

        with col2:
            st.info(f"Model: {selected_model}")

        return selected_model

    def analyze_with_model(
        self,
        df: pd.DataFrame,
        model: str,
        show_comparison: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze sentiment with selected model.

        Args:
            df: DataFrame to analyze
            model: Model name
            show_comparison: Show comparison with other models

        Returns:
            Tuple of (analyzed DataFrame, metrics)
        """
        metrics = {}

        if model == 'VADER':
            start_time = time.time()
            analyzer = self.get_sentiment_analyzer()
            df_result = analyzer.analyze_dataframe(df.copy(), use_transformer=False)
            processing_time = time.time() - start_time
            confidence = 'N/A'

        elif model == 'Transformer':
            start_time = time.time()
            analyzer = self.get_sentiment_analyzer()
            df_result = analyzer.analyze_dataframe(df.copy(), use_transformer=True)
            processing_time = time.time() - start_time
            avg_confidence = df_result['sentiment_compound'].abs().mean()
            confidence = f"{avg_confidence:.3f}"

        elif model == 'Multilingual':
            start_time = time.time()
            df_result = df.copy()
            df_result['sentiment_vader'] = df_result['sentiment_vader']
            processing_time = time.time() - start_time
            confidence = 'Multilingual'

        else:
            start_time = time.time()
            analyzer = self.get_sentiment_analyzer()
            df_result = analyzer.analyze_dataframe(df.copy(), use_transformer=True)
            processing_time = time.time() - start_time
            confidence = f"{df_result['sentiment_compound'].abs().mean():.3f}"

        metrics = {
            'model': model,
            'processing_time': processing_time,
            'confidence': confidence,
            'message_count': len(df_result),
            'unique_users': df_result['user'].nunique()
        }

        self.results[model] = {
            'df': df_result,
            'metrics': metrics
        }

        return df_result, metrics

    def get_model_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare sentiment results across models.

        Args:
            df: DataFrame

        Returns:
            Comparison DataFrame
        """
        comparison_data = []

        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'Processing Time (s)': f"{metrics['processing_time']:.3f}",
                'Confidence': str(metrics['confidence']),
                'Messages': metrics['message_count']
            })

        return pd.DataFrame(comparison_data)

    def render_model_metrics(self, metrics: Dict):
        """Render model metrics in columns."""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Model", metrics['model'])
        with col2:
            st.metric("Processing Time", f"{metrics['processing_time']:.3f}s")
        with col3:
            st.metric("Confidence", str(metrics['confidence']))
        with col4:
            st.metric("Messages Analyzed", metrics['message_count'])
