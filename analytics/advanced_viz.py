import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import re
from typing import Dict

class AdvancedVisualizations:
    """Advanced analytics visualizations with animations."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy() if not df.empty else pd.DataFrame(columns=['datetime', 'user', 'message_length'])

    def _empty_figure(self, title: str, text: str = "No data available for this chart"):
        fig = go.Figure()
        fig.add_annotation(
            text=text, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color='#94A3B8', size=13)
        )
        fig.update_layout(
            title=title, height=350,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(11,18,32,0.6)',
            font=dict(color='#94A3B8', family='Outfit')
        )
        return fig

    def sentiment_timeline_animated(self):
        """Animated sentiment evolution over time."""
        if self.df.empty or 'datetime' not in self.df.columns or 'sentiment_compound' not in self.df.columns:
            return self._empty_figure('Sentiment Evolution Timeline')

        daily_sentiment = self.df.groupby(self.df['datetime'].dt.date).agg({
            'sentiment_compound': ['mean', 'count']
        }).reset_index()

        daily_sentiment.columns = ['date', 'avg_sentiment', 'message_count']
        daily_sentiment['avg_sentiment'] = daily_sentiment['avg_sentiment'].fillna(0)

        if len(daily_sentiment) > 1500:
            daily_sentiment = daily_sentiment.iloc[::2]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['avg_sentiment'],
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='#00A699', width=3),
            marker=dict(size=max(2, min(8, 1000 // max(len(daily_sentiment), 1)))),
            fill='tozeroy',
            fillcolor='rgba(0, 166, 153, 0.2)'
        ))

        fig.update_layout(
            title='Sentiment Evolution Timeline (Animated)',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(11,18,32,0.6)',
            font=dict(color='#94A3B8', family='Outfit'),
            height=400,
            showlegend=False
        )

        return fig


    def emotion_transition_graph(self):
        """Emotion transitions over time."""
        # Guard: emotion column must exist
        if 'emotion' not in self.df.columns:
            fig = go.Figure()
            fig.update_layout(title='Emotion data not available', height=300)
            return fig

        emotion_by_hour = self.df.copy()
        emotion_by_hour['hour'] = emotion_by_hour['datetime'].dt.hour

        # Drop rows where emotion is NaN or not a string
        emotion_by_hour = emotion_by_hour[
            emotion_by_hour['emotion'].apply(lambda x: isinstance(x, str) and x.strip() != '')
        ]

        if len(emotion_by_hour) == 0:
            fig = go.Figure()
            fig.update_layout(title='No emotion data available', height=300)
            return fig

        emotion_dist = emotion_by_hour.groupby('hour')['emotion'].apply(
            lambda x: x.value_counts().to_dict()
        ).to_dict()

        emotions = set()
        for ed in emotion_dist.values():
            if isinstance(ed, dict):
                emotions.update(ed.keys())

        emotions = sorted(list(emotions))

        emotion_data = {emotion: [] for emotion in emotions}

        for hour in sorted(emotion_by_hour['hour'].unique()):
            ed = emotion_dist.get(hour, {})
            total = sum(ed.values())

            for emotion in emotions:
                pct = (ed.get(emotion, 0) / total * 100) if total > 0 else 0
                emotion_data[emotion].append(pct)

        fig = go.Figure()

        colors = {
            'joy': '#FFD700',
            'anger': '#FF4444',
            'sadness': '#4169E1',
            'fear': '#9932CC',
            'surprise': '#FF69B4',
            'disgust': '#FF8C00',
            'neutral': '#808080'
        }

        for emotion in emotions:
            fig.add_trace(go.Scatter(
                x=sorted(emotion_by_hour['hour'].unique()),
                y=emotion_data[emotion],
                name=emotion.title(),
                mode='lines',
                stackgroup='one',
                fillcolor=colors.get(emotion, '#808080')
            ))

        fig.update_layout(
            title='Emotion Transitions by Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Percentage (%)',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(11,18,32,0.6)',
            font=dict(color='#94A3B8', family='Outfit'),
            height=400,
            hovermode='x unified'
        )

        return fig

    def user_positivity_leaderboard(self):
        """User positivity ranking leaderboard."""
        if self.df.empty or 'user' not in self.df.columns or 'sentiment_compound' not in self.df.columns:
            return self._empty_figure('User Positivity Leaderboard')

        has_toxic = 'is_toxic' in self.df.columns
        user_stats = self.df.groupby('user').agg(
            sentiment=('sentiment_compound', 'mean'),
            non_toxic_pct=('is_toxic', lambda x: ((x == False).sum() / max(len(x), 1) * 100)) if has_toxic else ('sentiment_compound', lambda x: 100.0),
            message_count=('user', 'count')
        ).round(2)

        if user_stats.empty:
            return self._empty_figure('User Positivity Leaderboard')

        user_stats['sentiment'] = user_stats['sentiment'].fillna(0)
        user_stats['non_toxic_pct'] = user_stats['non_toxic_pct'].fillna(100)

        user_stats['positivity_score'] = (
            ((user_stats['sentiment'] + 1) / 2) * 0.6 +
            (user_stats['non_toxic_pct'] / 100) * 0.4
        ).round(3)

        user_stats = user_stats.sort_values('positivity_score', ascending=True).tail(15)

        colors = ['#E85D75' if score < 0.4 else '#FFA500' if score < 0.6 else '#00A699'
                  for score in user_stats['positivity_score']]

        fig = go.Figure(data=[
            go.Bar(
                y=user_stats.index.astype(str),
                x=user_stats['positivity_score'],
                orientation='h',
                marker=dict(color=colors),
                text=user_stats['positivity_score'].round(2),
                textposition='outside'
            )
        ])

        fig.update_layout(
            title='User Positivity Leaderboard',
            xaxis_title='Positivity Score (0 - 1.0)',
            xaxis=dict(range=[0, 1.15]),
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(11,18,32,0.6)',
            font=dict(color='#94A3B8', family='Outfit'),
            height=400,
            showlegend=False
        )

        return fig

    def toxicity_heatmap(self):
        """Toxicity heatmap by user and hour."""
        if self.df.empty or 'user' not in self.df.columns:
            return self._empty_figure('Toxicity Heatmap')

        heatmap_data = self.df.copy()
        heatmap_data['hour'] = heatmap_data['datetime'].dt.hour
        if 'is_toxic' not in heatmap_data.columns:
            heatmap_data['is_toxic'] = False

        toxicity_matrix = heatmap_data.groupby(['user', 'hour'])['is_toxic'].apply(
            lambda x: (x == True).sum() / max(len(x), 1) * 100
        ).reset_index()

        if toxicity_matrix.empty:
            return self._empty_figure('Toxicity Heatmap')

        pivot_table = toxicity_matrix.pivot(index='user', columns='hour', values='is_toxic').fillna(0)

        # Top 15 users by message count for clean layout
        top_users = self.df['user'].value_counts().head(15).index
        pivot_table = pivot_table.reindex(top_users).dropna(how='all').fillna(0)

        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=[f"{c}:00" for c in pivot_table.columns],
            y=pivot_table.index,
            colorscale='Reds',
            colorbar=dict(title="Toxicity %")
        ))

        fig.update_layout(
            title='Toxicity Heatmap (User × Hour)',
            xaxis_title='Hour of Day',
            yaxis_title='User',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(11,18,32,0.6)',
            font=dict(color='#94A3B8', family='Outfit'),
            height=400
        )

        return fig

    def activity_calendar_heatmap(self):
        """Activity calendar heatmap."""
        if self.df.empty or 'datetime' not in self.df.columns:
            return self._empty_figure('Activity Calendar Heatmap')

        activity = self.df.copy()
        activity['day_of_week'] = activity['datetime'].dt.day_name()
        activity['week'] = activity['datetime'].dt.isocalendar().week

        calendar_data = activity.groupby(['week', 'day_of_week']).size().reset_index(name='count')
        if calendar_data.empty:
            return self._empty_figure('Activity Calendar Heatmap')

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        calendar_data['day_of_week'] = pd.Categorical(calendar_data['day_of_week'], categories=day_order, ordered=True)

        pivot_calendar = calendar_data.pivot(index='day_of_week', columns='week', values='count').fillna(0)

        fig = go.Figure(data=go.Heatmap(
            z=pivot_calendar.values,
            x=[f"W{c}" for c in pivot_calendar.columns],
            y=pivot_calendar.index,
            colorscale='Viridis'
        ))

        fig.update_layout(
            title='Activity Calendar Heatmap',
            xaxis_title='Week Number',
            yaxis_title='Day of Week',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(11,18,32,0.6)',
            font=dict(color='#94A3B8', family='Outfit'),
            height=300
        )

        return fig

    def word_cloud_per_user(self, user: str):
        """Generate word frequency for specific user."""
        if self.df.empty or 'user' not in self.df.columns:
            return self._empty_figure(f'Top Words by {user}')

        user_df = self.df[self.df['user'] == user]
        if user_df.empty:
            return self._empty_figure(f'Top Words by {user}')

        all_tokens = []
        if 'tokens' in user_df.columns:
            for tokens in user_df['tokens'].dropna():
                if isinstance(tokens, list):
                    all_tokens.extend(tokens)
                elif isinstance(tokens, str):
                    all_tokens.extend(re.findall(r'\b[a-zA-Z]{3,}\b', tokens.lower()))

        if not all_tokens:
            msg_col = 'message_cleaned' if 'message_cleaned' in user_df.columns else ('message' if 'message' in user_df.columns else None)
            if msg_col:
                stopwords_set = {
                    'the','a','an','is','are','was','were','be','been','have','has','had',
                    'do','does','did','will','would','could','should','to','of','in','on',
                    'at','by','for','with','and','or','but','if','as','it','its','this',
                    'that','these','those','i','you','he','she','we','they','me','him',
                    'her','us','them','my','your','his','their','what','which','who',
                    'omitted','media','message','deleted','bhai','haan','nahi','kya',
                    'hai','ho','ka','ki','ke','ko','se','me','pe','par','tha','thi'
                }
                for msg in user_df[msg_col].dropna().astype(str):
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', msg.lower())
                    all_tokens.extend([w for w in words if w not in stopwords_set])

        if not all_tokens:
            return self._empty_figure(f'Top Words by {user}')

        from collections import Counter
        word_freq = Counter(all_tokens)
        top_words = dict(word_freq.most_common(15))
        if not top_words:
            return self._empty_figure(f'Top Words by {user}')

        # Sort ascending for horizontal bar chart
        top_sorted = dict(sorted(top_words.items(), key=lambda x: x[1]))

        fig = go.Figure(data=[
            go.Bar(
                x=list(top_sorted.values()),
                y=list(top_sorted.keys()),
                orientation='h',
                marker=dict(color=list(top_sorted.values()), colorscale='Viridis')
            )
        ])

        fig.update_layout(
            title=f'Top Words by {user}',
            xaxis_title='Frequency',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(11,18,32,0.6)',
            font=dict(color='#94A3B8', family='Outfit'),
            height=400,
            showlegend=False
        )

        return fig

    def sentiment_distribution_pie_animated(self):
        """Animated sentiment distribution pie chart."""
        if self.df.empty or 'sentiment_vader' not in self.df.columns:
            return self._empty_figure('Sentiment Distribution')

        sentiment_counts = self.df['sentiment_vader'].value_counts()

        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker=dict(colors=['#00A699', '#E85D75', '#95E1D3']),
            textposition='inside',
            textinfo='label+percent'
        )])

        fig.update_layout(
            title='Sentiment Distribution',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(11,18,32,0.6)',
            font=dict(color='#94A3B8', family='Outfit'),
            height=400
        )

        return fig