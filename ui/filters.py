import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Dict


class FilterSystem:
    """Advanced filter system for WhatsApp chat data."""

    def __init__(self):
        self.filters = {}

    def _label(self, text: str, icon: str = ""):
        """Render a styled filter section label in sidebar."""
        st.sidebar.markdown(
            f'<p style="font-family:DM Sans,sans-serif; font-size:10px; '
            f'font-weight:700; letter-spacing:0.13em; text-transform:uppercase; '
            f'color:#8C7355; margin:16px 0 4px;">{icon} {text}</p>',
            unsafe_allow_html=True
        )

    def render_sidebar_filters(self, df: pd.DataFrame) -> Dict:
        """Render all filters in sidebar and return applied filters."""

        # Header
        st.sidebar.markdown("""
        <div style="padding:6px 0 14px;">
            <div style="font-family:'Playfair Display',serif; font-size:16px;
                font-weight:700; color:#1A1208; margin-bottom:4px;">
                Filter Controls
            </div>
            <div style="height:1px;
                background:linear-gradient(90deg,#C9973A,transparent);"></div>
        </div>
        """, unsafe_allow_html=True)

        filters = {}

        # ── Users ───────────────────────────────────────────────────────
        self._label("Users", "👤")
        users = df['user'].unique().tolist()
        selected_users = st.sidebar.multiselect(
            "Select Users", users, default=users,
            key="user_filter", label_visibility="collapsed"
        )
        filters['users'] = selected_users

        # ── Date Range ──────────────────────────────────────────────────
        self._label("Date Range", "📅")
        min_date = df['datetime'].min().date()
        max_date = df['datetime'].max().date()
        date_range = st.sidebar.slider(
            "Select Date Range",
            min_value=min_date, max_value=max_date,
            value=(min_date, max_date),
            key="date_filter", label_visibility="collapsed"
        )
        filters['date_range'] = date_range

        # ── Sentiment ────────────────────────────────────────────────────
        self._label("Sentiment", "😊")
        sentiment_options = [s for s in df['sentiment_vader'].unique().tolist() if s is not None]
        selected_sentiments = st.sidebar.multiselect(
            "Select Sentiments", sentiment_options, default=sentiment_options,
            key="sentiment_filter", label_visibility="collapsed"
        )
        filters['sentiments'] = selected_sentiments

        # ── Emotion ──────────────────────────────────────────────────────
        self._label("Emotion Type", "🎭")
        emotion_options = [e for e in df['emotion'].unique().tolist() if e is not None]
        selected_emotions = st.sidebar.multiselect(
            "Select Emotions", emotion_options, default=emotion_options,
            key="emotion_filter", label_visibility="collapsed"
        )
        filters['emotions'] = selected_emotions

        # ── Keyword ──────────────────────────────────────────────────────
        self._label("Keyword Search", "🔍")
        keyword = st.sidebar.text_input(
            "Search in messages", placeholder="Type keyword...",
            key="keyword_filter", label_visibility="collapsed"
        )
        filters['keyword'] = keyword.lower() if keyword else None

        # ── Message Length ───────────────────────────────────────────────
        self._label("Message Length", "📏")
        min_length, max_length = st.sidebar.slider(
            "Filter by message length (chars)",
            min_value=0, max_value=int(df['message_length'].max()),
            value=(0, int(df['message_length'].max())),
            key="length_filter", label_visibility="collapsed"
        )
        filters['message_length'] = (min_length, max_length)

        # ── Toxicity ─────────────────────────────────────────────────────
        self._label("Toxicity", "⚠️")
        toxicity_filter = st.sidebar.checkbox(
            "Show only toxic messages", value=False, key="toxicity_filter"
        )
        filters['show_toxic_only'] = toxicity_filter

        return filters

    def apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply all filters to dataframe."""
        df_filtered = df.copy()

        # User filter
        df_filtered = df_filtered[df_filtered['user'].isin(filters['users'])]

        # Date filter — FIX: .end_of_day doesn't exist in pandas; use .replace() instead
        start_date = pd.Timestamp(filters['date_range'][0])
        end_date = pd.Timestamp(filters['date_range'][1]).replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
        df_filtered = df_filtered[
            (df_filtered['datetime'] >= start_date) &
            (df_filtered['datetime'] <= end_date)
        ]

        # Sentiment filter
        if filters['sentiments']:
            df_filtered = df_filtered[df_filtered['sentiment_vader'].isin(filters['sentiments'])]

        # Emotion filter
        if filters['emotions']:
            df_filtered = df_filtered[df_filtered['emotion'].isin(filters['emotions'])]

        # Keyword filter
        if filters['keyword']:
            df_filtered = df_filtered[
                df_filtered['message_lower'].str.contains(filters['keyword'], na=False)
            ]

        # Length filter
        min_len, max_len = filters['message_length']
        df_filtered = df_filtered[
            (df_filtered['message_length'] >= min_len) &
            (df_filtered['message_length'] <= max_len)
        ]

        # Toxicity filter
        if filters['show_toxic_only']:
            df_filtered = df_filtered[df_filtered['is_toxic'] == True]

        return df_filtered

    def get_filter_summary(self, filters: Dict, df_original: pd.DataFrame, df_filtered: pd.DataFrame) -> str:
        """Get text summary of applied filters."""
        return f"Showing {len(df_filtered)} of {len(df_original)} messages"