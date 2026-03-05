import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.modules.chat_parser import WhatsAppParser
from src.modules.data_cleaner import DataCleaner
from src.modules.analytics import BasicAnalytics
from src.modules.sentiment_analyzer import SentimentAnalyzer
from src.modules.emotion_detector import EmotionDetector
from src.modules.behavioral_analysis import BehavioralAnalyzer
from src.modules.multilingual import MultilingualAnalyzer
from src.modules.export_handler import ExportHandler
from src.utils.file_handler import save_uploaded_file
import io

st.set_page_config(
    page_title="WhatsApp Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("📱 WhatsApp Sentiment Analyzer")
st.sidebar.divider()

st.title("💬 Advanced WhatsApp Sentiment Analyzer")
st.write("Comprehensive analysis of WhatsApp conversations with sentiment, emotion, and behavioral insights.")

st.header("Upload WhatsApp Chat")

uploaded_file = st.file_uploader(
    "Choose your WhatsApp chat export (.txt file)",
    type=['txt'],
    help="Export: More > Export chat > Without media"
)

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file, 'whatsapp-analyzer/data/raw')
    st.success(f"✓ File uploaded: {uploaded_file.name}")

    with st.spinner("Parsing chat file..."):
        parser = WhatsAppParser()
        df = parser.parse_file(file_path)

    with st.spinner("Cleaning and preprocessing data..."):
        cleaner = DataCleaner()
        df_cleaned = cleaner.clean_dataframe(df)

    with st.spinner("Analyzing sentiment..."):
        sentiment_analyzer = SentimentAnalyzer()
        df_cleaned = sentiment_analyzer.analyze_dataframe(df_cleaned, use_transformer=False)

    with st.spinner("Detecting emotions..."):
        emotion_detector = EmotionDetector()
        df_cleaned = emotion_detector.analyze_dataframe(df_cleaned)

    with st.spinner("Behavioral analysis..."):
        behavioral_analyzer = BehavioralAnalyzer()
        df_cleaned = behavioral_analyzer.analyze_dataframe(df_cleaned)

    with st.spinner("Language detection..."):
        multilingual_analyzer = MultilingualAnalyzer()
        df_cleaned = multilingual_analyzer.analyze_dataframe(df_cleaned)

    st.session_state.df_cleaned = df_cleaned
    st.session_state.parser = parser
    st.session_state.cleaner = cleaner
    st.session_state.sentiment_analyzer = sentiment_analyzer
    st.session_state.emotion_detector = emotion_detector
    st.session_state.behavioral_analyzer = behavioral_analyzer
    st.session_state.multilingual_analyzer = multilingual_analyzer
    st.session_state.file_name = uploaded_file.name

    st.success("✓ All analysis complete!")

if 'df_cleaned' in st.session_state:
    df_cleaned = st.session_state.df_cleaned

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "😊 Sentiment",
        "😍 Emotions",
        "⚠️ Behavior",
        "💬 Messages"
    ])

    with tab1:
        st.header("Chat Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Messages", len(df_cleaned))
        with col2:
            st.metric("Unique Users", df_cleaned['user'].nunique())
        with col3:
            st.metric("Avg Message Length", f"{df_cleaned['message_length'].mean():.0f} chars")
        with col4:
            st.metric("Date Range", f"{(df_cleaned['datetime'].max() - df_cleaned['datetime'].min()).days} days")

        analytics = BasicAnalytics(df_cleaned)

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(analytics.plot_user_activity(), use_container_width=True)

        with col2:
            st.plotly_chart(analytics.plot_user_engagement_pie(), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(analytics.plot_activity_timeline(), use_container_width=True)

        with col2:
            st.plotly_chart(analytics.plot_hourly_heatmap(), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(analytics.plot_word_frequency(), use_container_width=True)

        with col2:
            st.plotly_chart(analytics.plot_message_length_distribution(), use_container_width=True)

    with tab2:
        st.header("Sentiment Analysis")

        sentiment_analyzer = st.session_state.sentiment_analyzer
        sentiment_dist = sentiment_analyzer.get_sentiment_distribution(df_cleaned)

        col1, col2, col3 = st.columns(3)

        for idx, (sentiment, stats) in enumerate(sentiment_dist.items()):
            if idx == 0:
                col = col1
            elif idx == 1:
                col = col2
            else:
                col = col3

            with col:
                color = '#00A699' if sentiment == 'POSITIVE' else '#E85D75' if sentiment == 'NEGATIVE' else '#95E1D3'
                st.metric(
                    sentiment,
                    f"{stats['count']}",
                    f"{stats['percentage']:.1f}%"
                )

        col1, col2 = st.columns(2)

        with col1:
            user_sentiment = sentiment_analyzer.get_user_sentiment(df_cleaned)

            fig = go.Figure(data=[
                go.Bar(
                    x=user_sentiment['avg_sentiment_score'],
                    y=user_sentiment.index,
                    orientation='h',
                    marker=dict(
                        color=user_sentiment['avg_sentiment_score'],
                        colorscale='RdYlGn',
                        showscale=True
                    )
                )
            ])

            fig.update_layout(
                title='Average Sentiment Score per User',
                xaxis_title='Sentiment Score',
                yaxis_title='User',
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            sentiment_trend = sentiment_analyzer.get_sentiment_trend(df_cleaned, period='D')

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sentiment_trend['datetime'],
                y=sentiment_trend['sentiment_compound'],
                mode='lines+markers',
                name='Sentiment',
                line=dict(color='#00A699', width=2),
                marker=dict(size=6)
            ))

            fig.update_layout(
                title='Sentiment Trend Over Time',
                xaxis_title='Date',
                yaxis_title='Sentiment Score',
                height=400,
                template='plotly_white',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sentiment Distribution")

        fig = go.Figure(data=[
            go.Pie(
                labels=list(sentiment_dist.keys()),
                values=[stats['count'] for stats in sentiment_dist.values()],
                marker=dict(colors=['#00A699', '#E85D75', '#95E1D3'])
            )
        ])

        fig.update_layout(title='Overall Sentiment Distribution', height=400)

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Emotion Analysis")

        emotion_detector = st.session_state.emotion_detector
        emotion_dist = emotion_detector.get_emotion_distribution(df_cleaned)

        col1, col2, col3 = st.columns(3)

        emotions_list = list(emotion_dist.keys())

        for idx, emotion in enumerate(emotions_list[:3]):
            with [col1, col2, col3][idx]:
                st.metric(
                    emotion,
                    f"{emotion_dist[emotion]['count']}",
                    f"{emotion_dist[emotion]['percentage']:.1f}%"
                )

        if len(emotions_list) > 3:
            col1, col2, col3 = st.columns(3)
            for idx, emotion in enumerate(emotions_list[3:6]):
                with [col1, col2, col3][idx]:
                    st.metric(
                        emotion,
                        f"{emotion_dist[emotion]['count']}",
                        f"{emotion_dist[emotion]['percentage']:.1f}%"
                    )

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(emotion_dist.keys()),
                    values=[stats['count'] for stats in emotion_dist.values()]
                )
            ])

            fig.update_layout(title='Emotion Distribution', height=400)

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            emotion_intensity = emotion_detector.get_emotion_intensity(df_cleaned)

            fig = go.Figure(data=[
                go.Bar(
                    x=list(emotion_intensity.values()),
                    y=list(emotion_intensity.keys()),
                    orientation='h',
                    marker=dict(color='#4ECDC4')
                )
            ])

            fig.update_layout(
                title='Emotion Intensity (Avg Score)',
                xaxis_title='Average Score',
                yaxis_title='Emotion',
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Emotion per User")

        user_emotions = emotion_detector.get_user_emotion(df_cleaned)

        st.dataframe(user_emotions, use_container_width=True)

    with tab4:
        
        st.header("Behavioral Analysis")
        behavioral_analyzer = st.session_state.behavioral_analyzer
        health = behavioral_analyzer.get_conversation_health(df_cleaned)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Health Score", f"{health['health_score']:.2f}")
        with col2:
            st.metric("Status", health['status'])
        with col3:
            st.metric("Avg Sentiment", f"{health['avg_sentiment']:.2f}")
        with col4:
            st.metric("Toxic %", f"{health['toxic_percentage']:.1f}%")

        col1, col2 = st.columns(2)

        with col1:
            user_ranking = behavioral_analyzer.get_user_positivity_ranking(df_cleaned)

            fig = go.Figure(data=[
                go.Bar(
                    x=user_ranking['positivity_score'],
                    y=user_ranking.index,
                    orientation='h',
                    marker=dict(
                        color=user_ranking['positivity_score'],
                        colorscale='Viridis',
                        showscale=True
                    )
                )
            ])

            fig.update_layout(
                title='User Positivity Ranking',
                xaxis_title='Positivity Score',
                yaxis_title='User',
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            toxicity_stats = behavioral_analyzer.get_toxicity_stats(df_cleaned)

            fig = go.Figure(data=[
                go.Indicator(
                    mode="gauge+number",
                    value=toxicity_stats['toxic_percentage'],
                    title={'text': "Toxicity Level (%)"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                )
            ])

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Activity Patterns")

        activity_patterns = behavioral_analyzer.get_activity_patterns(df_cleaned)

        patterns_df = pd.DataFrame(activity_patterns).T

        st.dataframe(patterns_df, use_container_width=True)

    with tab5:
        st.header("All Messages with Analysis")

        display_columns = [
            'datetime', 'user', 'message',
            'sentiment_vader', 'emotion',
            'is_toxic', 'detected_language'
        ]

        available_columns = [col for col in display_columns if col in df_cleaned.columns]

        st.dataframe(
            df_cleaned[available_columns],
            use_container_width=True,
            height=600
        )

        st.divider()
        st.header("Export Analysis")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Download Messages CSV", use_container_width=True):
                export_handler = ExportHandler('whatsapp-analyzer/output')
                csv_path = export_handler.export_messages_with_analysis(df_cleaned)

                with open(csv_path, 'rb') as f:
                    st.download_button(
                        label="Click to download CSV",
                        data=f,
                        file_name="messages_analysis.csv",
                        mime="text/csv"
                    )

        with col2:
            if st.button("📊 Generate PDF Report", use_container_width=True):
                export_handler = ExportHandler('whatsapp-analyzer/output')
                analytics = BasicAnalytics(df_cleaned)
                sentiment_dist = st.session_state.sentiment_analyzer.get_sentiment_distribution(df_cleaned)
                emotion_dist = emotion_detector.get_emotion_distribution(df_cleaned)
                health = behavioral_analyzer.get_conversation_health(df_cleaned)

                summary = analytics.get_summary_stats()

                pdf_path = export_handler.create_pdf_report(
                    df_cleaned,
                    summary,
                    sentiment_dist,
                    emotion_dist,
                    health
                )

                with open(pdf_path, 'rb') as f:
                    st.download_button(
                        label="Click to download PDF",
                        data=f,
                        file_name="sentiment_analysis_report.pdf",
                        mime="application/pdf"
                    )

st.divider()
st.caption("💡 Tips: Export chats from WhatsApp settings > Chats > Export chat > Without media")
