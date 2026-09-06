import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from collections import Counter

# ── Shared chart layout defaults ───────────────────────────────────────────
_DARK_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,24,39,0.5)',
    font=dict(color='#94A3B8', family='Outfit, sans-serif', size=12),
    margin=dict(t=50, b=50, l=60, r=20),
    title_font=dict(color='#E2E8F0', size=14, family='Outfit, sans-serif'),
)

_TEAL_COLORS = [
    '#00C896', '#22D3EE', '#10B981', '#F472B6',
    '#818CF8', '#FBBF24', '#34D399', '#F87171',
    '#A78BFA', '#60A5FA',
]

def _empty_fig(title: str, msg: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color='#94A3B8', size=13)
    )
    fig.update_layout(title=title, height=350, **_DARK_LAYOUT)
    return fig


class BasicAnalytics:
    """Generate basic analytics and visualizations for WhatsApp chats."""

    def __init__(self, df_cleaned: pd.DataFrame):
        self.df = df_cleaned.copy() if not df_cleaned.empty else pd.DataFrame(columns=['datetime', 'user', 'message_length'])
        if not self.df.empty and 'datetime' in self.df.columns:
            self.df['date']        = self.df['datetime'].dt.date
            self.df['hour']        = self.df['datetime'].dt.hour
            self.df['day_of_week'] = self.df['datetime'].dt.day_name()
        else:
            self.df['date'] = []
            self.df['hour'] = []
            self.df['day_of_week'] = []

    def get_summary_stats(self) -> dict:
        if len(self.df) == 0:
            return {
                'total_messages': 0, 'unique_users': 0,
                'date_range_start': 'N/A', 'date_range_end': 'N/A',
                'avg_messages_per_day': 0.0, 'avg_message_length': 0.0,
                'most_active_user': 'N/A'
            }
        min_dt = self.df['datetime'].min()
        max_dt = self.df['datetime'].max()
        days = max((max_dt - min_dt).days, 1)
        vc = self.df['user'].value_counts()
        most_active = vc.index[0] if len(vc) > 0 else 'N/A'

        return {
            'total_messages':       len(self.df),
            'unique_users':         int(self.df['user'].nunique()),
            'date_range_start':     min_dt,
            'date_range_end':       max_dt,
            'avg_messages_per_day': round(len(self.df) / days, 1),
            'avg_message_length':   round(float(self.df['message_length'].mean()), 1) if 'message_length' in self.df.columns else 0.0,
            'most_active_user':     most_active,
        }

    def get_user_activity(self) -> pd.DataFrame:
        if self.df.empty or 'user' not in self.df.columns:
            return pd.DataFrame(columns=['user', 'count'])
        return self.df['user'].value_counts().reset_index()

    def get_word_frequency(self, top_n: int = 20) -> dict:
        if self.df.empty or 'tokens' not in self.df.columns:
            return {}
        stopwords = {
            'the','a','an','is','are','was','were','be','been','have','has','had',
            'do','does','did','will','would','could','should','to','of','in','on',
            'at','by','for','with','and','or','but','if','as','it','its','this',
            'that','i','me','my','we','you','he','she','they','them','ok','okay',
            'hi','hey','yes','no','yeah','nah','haha','lol','deleted','message',
            'media','omitted','null','https','http','hai','nhi','aur','bhi',
        }
        all_tokens = []
        for tokens in self.df['tokens']:
            if isinstance(tokens, list):
                all_tokens.extend([t.lower() for t in tokens if t.isalpha() and len(t) > 2])
        filtered = [t for t in all_tokens if t not in stopwords]
        return dict(Counter(filtered).most_common(top_n))

    def plot_user_activity(self):
        data = self.get_user_activity().head(15)
        if data.empty:
            return _empty_fig('Messages per User')

        # Clean count and user values
        counts = data['count'].fillna(0).tolist()
        users = data['user'].fillna('Unknown').astype(str).tolist()

        fig = go.Figure(go.Bar(
            x=counts,
            y=users,
            orientation='h',
            marker=dict(
                color=counts,
                colorscale=[[0, '#10B981'], [0.5, '#00C896'], [1, '#22D3EE']],
                showscale=False,
            ),
        ))
        fig.update_layout(
            title='Messages per User',
            xaxis_title='Message Count',
            height=400,
            showlegend=False,
            **_DARK_LAYOUT,
        )
        fig.update_xaxes(gridcolor='rgba(0,200,150,0.08)')
        fig.update_yaxes(gridcolor='rgba(0,200,150,0.04)', autorange="reversed")
        return fig

    def plot_activity_timeline(self):
        if self.df.empty or 'date' not in self.df.columns:
            return _empty_fig('Daily Message Activity')

        daily = self.df.groupby('date').size().reset_index(name='count')
        if daily.empty:
            return _empty_fig('Daily Message Activity')

        # Downsample if too many days (e.g. multi-year chats)
        if len(daily) > 1500:
            daily = daily.iloc[::2]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily['date'],
            y=daily['count'],
            mode='lines',
            name='Messages',
            line=dict(color='#00C896', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(0,200,150,0.07)',
        ))
        fig.update_layout(
            title='Daily Message Activity',
            xaxis_title='Date',
            yaxis_title='Messages',
            height=400,
            hovermode='x unified',
            **_DARK_LAYOUT,
        )
        fig.update_xaxes(gridcolor='rgba(0,200,150,0.08)')
        fig.update_yaxes(gridcolor='rgba(0,200,150,0.08)')
        return fig

    def plot_hourly_heatmap(self):
        if self.df.empty or 'day_of_week' not in self.df.columns or 'hour' not in self.df.columns:
            return _empty_fig('Activity Heatmap')

        pivot_data  = self.df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        if pivot_data.empty:
            return _empty_fig('Activity Heatmap')

        pivot_table = pivot_data.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
        day_order   = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_table = pivot_table.reindex(day_order).fillna(0)

        # Ensure all 24 hours exist
        for h in range(24):
            if h not in pivot_table.columns:
                pivot_table[h] = 0
        pivot_table = pivot_table[sorted(pivot_table.columns)]

        fig = go.Figure(go.Heatmap(
            z=pivot_table.values,
            x=[f"{h}:00" for h in pivot_table.columns],
            y=pivot_table.index,
            colorscale=[
                [0.0, 'rgba(17,24,39,0.8)'],
                [0.5, 'rgba(0,200,150,0.5)'],
                [1.0, '#00C896'],
            ],
            hoverongaps=False,
        ))
        fig.update_layout(
            title='Activity Heatmap',
            xaxis_title='Hour of Day',
            height=400,
            **_DARK_LAYOUT,
        )
        return fig

    def plot_word_frequency(self, top_n: int = 15):
        word_freq   = self.get_word_frequency(top_n)
        if not word_freq:
            return _empty_fig(f'Top {top_n} Most Common Words', 'No words found')

        words       = list(word_freq.keys())
        frequencies = list(word_freq.values())

        fig = go.Figure(go.Bar(
            x=frequencies,
            y=words,
            orientation='h',
            marker=dict(
                color=frequencies,
                colorscale=[[0, '#10B981'], [0.5, '#22D3EE'], [1, '#F472B6']],
                showscale=False,
            ),
        ))
        fig.update_layout(
            title=f'Top {top_n} Most Common Words',
            xaxis_title='Frequency',
            height=400,
            showlegend=False,
            **_DARK_LAYOUT,
        )
        fig.update_xaxes(gridcolor='rgba(0,200,150,0.08)')
        fig.update_yaxes(autorange="reversed")
        return fig

    def plot_message_length_distribution(self):
        if self.df.empty or 'message_length' not in self.df.columns:
            return _empty_fig('Message Length Distribution')

        # Drop NaN or outliers for clean display
        lens = self.df['message_length'].dropna()
        q99 = lens.quantile(0.99) if len(lens) > 0 else 500
        lens_clean = lens[lens <= max(q99, 50)]

        fig = go.Figure(go.Histogram(
            x=lens_clean,
            nbinsx=30,
            marker=dict(color='#22D3EE', opacity=0.8),
        ))
        fig.update_layout(
            title='Message Length Distribution',
            xaxis_title='Characters',
            yaxis_title='Frequency',
            height=400,
            **_DARK_LAYOUT,
        )
        return fig

    def plot_user_engagement_pie(self):
        data = self.get_user_activity().head(10)
        if data.empty:
            return _empty_fig('User Contribution (Top 10)')

        fig = go.Figure(go.Pie(
            labels=data['user'],
            values=data['count'],
            marker=dict(colors=_TEAL_COLORS),
            hole=0.4,
        ))
        fig.update_layout(
            title='User Contribution (Top 10)',
            height=400,
            **_DARK_LAYOUT,
        )
        return fig

    def get_emoji_stats(self) -> dict:
        all_emojis = []
        for emojis in self.df.get('emojis', []):
            if isinstance(emojis, str):
                all_emojis.extend(list(emojis))
        return dict(Counter(all_emojis).most_common(10))