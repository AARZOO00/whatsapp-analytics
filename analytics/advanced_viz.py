import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict

class AdvancedVisualizations:
    """Advanced analytics visualizations with animations."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def sentiment_timeline_animated(self):
        """Animated sentiment evolution over time."""
        daily_sentiment = self.df.groupby(self.df['datetime'].dt.date).agg({
            'sentiment_compound': ['mean', 'count']
        }).reset_index()

        daily_sentiment.columns = ['date', 'avg_sentiment', 'message_count']

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['avg_sentiment'],
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='#00A699', width=3),
            marker=dict(size=8),
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
        emotion_by_hour = self.df.copy()
        emotion_by_hour['hour'] = emotion_by_hour['datetime'].dt.hour

        # Drop rows where emotion is NaN or not a string (avoids 'float has no .keys()' error)
        emotion_by_hour = emotion_by_hour[
            emotion_by_hour['emotion'].apply(lambda x: isinstance(x, str))
        ]

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
        user_stats = self.df.groupby('user').agg({
            'sentiment_compound': 'mean',
            'is_toxic': lambda x: (x == False).sum() / len(x) * 100,
            'user': 'count'
        }).round(2)

        user_stats.columns = ['sentiment', 'non_toxic_pct', 'message_count']

        user_stats['positivity_score'] = (
            (user_stats['sentiment'] + 1) / 2 * 0.6 +
            user_stats['non_toxic_pct'] / 100 * 0.4
        ).round(3)

        user_stats = user_stats.sort_values('positivity_score', ascending=True)

        colors = ['#E85D75' if score < 0.4 else '#FFA500' if score < 0.6 else '#00A699'
                  for score in user_stats['positivity_score']]

        fig = go.Figure(data=[
            go.Bar(
                y=user_stats.index,
                x=user_stats['positivity_score'],
                orientation='h',
                marker=dict(color=colors),
                text=user_stats['positivity_score'],
                textposition='outside'
            )
        ])

        fig.update_layout(
            title='User Positivity Leaderboard',
            xaxis_title='Positivity Score',
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
        heatmap_data = self.df.copy()
        heatmap_data['hour'] = heatmap_data['datetime'].dt.hour

        toxicity_matrix = heatmap_data.groupby(['user', 'hour'])['is_toxic'].apply(
            lambda x: (x == True).sum() / len(x) * 100 if len(x) > 0 else 0
        ).reset_index()

        pivot_table = toxicity_matrix.pivot(index='user', columns='hour', values='is_toxic')

        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
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
        activity = self.df.copy()
        activity['date'] = activity['datetime'].dt.date
        activity['day_of_week'] = activity['datetime'].dt.day_name()
        activity['week'] = activity['datetime'].dt.isocalendar().week

        calendar_data = activity.groupby(['week', 'day_of_week']).size().reset_index(name='count')

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        calendar_data['day_of_week'] = pd.Categorical(calendar_data['day_of_week'], categories=day_order, ordered=True)

        pivot_calendar = calendar_data.pivot(index='day_of_week', columns='week', values='count')

        fig = go.Figure(data=go.Heatmap(
            z=pivot_calendar.values,
            x=pivot_calendar.columns,
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
        user_df = self.df[self.df['user'] == user]

        all_tokens = []
        for tokens in user_df['tokens']:
            if isinstance(tokens, list):
                all_tokens.extend(tokens)

        if not all_tokens:
            return None

        from collections import Counter
        word_freq = Counter(all_tokens)
        top_words = dict(word_freq.most_common(15))

        fig = go.Figure(data=[
            go.Bar(
                x=list(top_words.values()),
                y=list(top_words.keys()),
                orientation='h',
                marker=dict(color=list(top_words.values()), colorscale='Viridis')
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