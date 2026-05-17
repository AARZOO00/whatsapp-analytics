"""
model_manager.py — Proper VADER / Transformer / Multilingual / Hybrid comparison
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from typing import Dict, Tuple

from src.modules.sentiment_analyzer import SentimentAnalyzer

_PALETTE = ['#18A3B7', '#F472B6', '#FBBF24', '#818CF8', '#4ADE80', '#F87171']
_DARK_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,24,39,0.5)',
    font=dict(color='#94A3B8', family='Outfit, sans-serif', size=12),
    margin=dict(t=60, b=50, l=50, r=30),
    title_font=dict(color='#E2E8F0', size=15, family='Syne, sans-serif'),
)


class ModelManager:
    """Compare VADER, Transformer, Multilingual-BERT, Hybrid sentiment models."""

    MODEL_INFO = {
        'VADER (WhatsApp-Tuned)': {
            'desc':  'Rule-based + Hinglish lexicon, emoji-aware. Fast, no GPU needed.',
            'speed': '⚡ Very Fast',
            'best':  'Large chats, Hinglish, casual text',
            'color': '#18A3B7',
            'icon':  '⚡',
        },
        'Transformer (DistilBERT)': {
            'desc':  'DistilBERT fine-tuned on SST-2. Deep semantic understanding.',
            'speed': '🐢 Slow (GPU helps)',
            'best':  'Formal English text',
            'color': '#818CF8',
            'icon':  '🤖',
        },
        'Multilingual BERT': {
            'desc':  'nlptown/bert-base-multilingual — handles Hindi, Urdu, Arabic, English.',
            'speed': '🐢 Slow',
            'best':  'Multi-language chats',
            'color': '#F472B6',
            'icon':  '🌐',
        },
        'Hybrid': {
            'desc':  'VADER for speed + Transformer for uncertain messages. Best accuracy.',
            'speed': '🚀 Balanced',
            'best':  'Production use — best of both worlds',
            'color': '#FBBF24',
            'icon':  '🔀',
        },
    }

    def __init__(self):
        self.results: Dict = {}

    @st.cache_resource
    def _get_vader(_self):
        return SentimentAnalyzer()

    @st.cache_resource
    def _get_transformer(_self):
        try:
            from transformers import pipeline
            return pipeline(
                'sentiment-analysis',
                model='distilbert-base-uncased-finetuned-sst-2-english',
                truncation=True, max_length=512,
            )
        except Exception:
            return None

    @st.cache_resource
    def _get_multilingual(_self):
        try:
            from transformers import pipeline
            return pipeline(
                'sentiment-analysis',
                model='nlptown/bert-base-multilingual-uncased-sentiment',
                truncation=True, max_length=512,
            )
        except Exception:
            return None

    # ── Model selector UI ─────────────────────────────────────────────────────

    def render_model_selector(self) -> str:
        is_lt = st.session_state.get('theme', 'light') == 'light'
        ac    = '#B8883A' if is_lt else '#18C8E0'
        tc    = '#18120A' if is_lt else '#E2E8F0'
        sc    = '#6B5C3E' if is_lt else '#94A3B8'
        bg    = '#FFFFFF' if is_lt else 'rgba(17,24,39,0.75)'
        bdr   = 'rgba(184,136,58,0.22)' if is_lt else 'rgba(24,163,183,0.14)'

        st.markdown(
            f'<div style="font-size:11px;font-weight:700;letter-spacing:.15em;'
            f'text-transform:uppercase;color:{ac};margin-bottom:14px;">🎛️ Select Sentiment Model</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(len(self.MODEL_INFO))
        for col, (name, info) in zip(cols, self.MODEL_INFO.items()):
            with col:
                col.markdown(
                    f'<div style="background:{bg};border:1.5px solid {bdr};border-top:3px solid {info["color"]};'
                    f'border-radius:12px;padding:14px 12px;">'
                    f'<div style="font-size:22px;margin-bottom:6px;">{info["icon"]}</div>'
                    f'<div style="font-size:12px;font-weight:700;color:{tc};margin-bottom:4px;">{name}</div>'
                    f'<div style="font-size:10px;color:{sc};margin-bottom:6px;line-height:1.5;">{info["desc"]}</div>'
                    f'<div style="font-size:9px;color:{info["color"]};font-weight:700;">{info["speed"]}</div>'
                    f'<div style="font-size:9px;color:{sc};margin-top:3px;">Best for: {info["best"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        selected = st.selectbox(
            'Choose model',
            list(self.MODEL_INFO.keys()),
            label_visibility='collapsed',
            key='model_selector',
        )
        return selected

    # ── Analysis runners ──────────────────────────────────────────────────────

    def analyze_with_model(self, df: pd.DataFrame, model: str) -> Tuple[pd.DataFrame, Dict]:
        t0 = time.time()
        df_res = df.copy()
        confidence = 'N/A'
        extra = {}

        if model == 'VADER (WhatsApp-Tuned)':
            analyzer = self._get_vader()
            df_res = analyzer.analyze_dataframe(df_res, use_transformer=False)
            confidence = f"{df_res['sentiment_compound'].abs().mean():.3f}"

        elif model == 'Transformer (DistilBERT)':
            pipe = self._get_transformer()
            if pipe is None:
                st.warning('⚠️ Transformer not available — falling back to VADER.')
                analyzer = self._get_vader()
                df_res = analyzer.analyze_dataframe(df_res, use_transformer=False)
            else:
                col = 'message_cleaned' if 'message_cleaned' in df_res.columns else 'message'
                def _t_analyze(text):
                    try:
                        r = pipe(str(text)[:512])[0]
                        label = 'POSITIVE' if r['label'] == 'POSITIVE' else 'NEGATIVE'
                        score = r['score']
                        compound = score if label == 'POSITIVE' else -score
                        return label, round(compound, 4)
                    except Exception:
                        return 'NEUTRAL', 0.0
                results = df_res[col].apply(_t_analyze)
                df_res['sentiment_vader']    = results.apply(lambda x: x[0])
                df_res['sentiment_compound'] = results.apply(lambda x: x[1])
            confidence = f"{df_res['sentiment_compound'].abs().mean():.3f}"

        elif model == 'Multilingual BERT':
            pipe = self._get_multilingual()
            if pipe is None:
                st.warning('⚠️ Multilingual model not available — falling back to VADER.')
                analyzer = self._get_vader()
                df_res = analyzer.analyze_dataframe(df_res, use_transformer=False)
            else:
                col = 'message_cleaned' if 'message_cleaned' in df_res.columns else 'message'
                def _m_analyze(text):
                    try:
                        r = pipe(str(text)[:512])[0]
                        # nlptown returns 1-5 stars
                        label_raw = r['label']  # e.g. "4 stars"
                        stars = int(label_raw.split()[0])
                        compound = (stars - 3) / 2   # -1 to +1
                        if stars >= 4:    label = 'POSITIVE'
                        elif stars <= 2:  label = 'NEGATIVE'
                        else:             label = 'NEUTRAL'
                        return label, round(compound, 4)
                    except Exception:
                        return 'NEUTRAL', 0.0
                results = df_res[col].apply(_m_analyze)
                df_res['sentiment_vader']    = results.apply(lambda x: x[0])
                df_res['sentiment_compound'] = results.apply(lambda x: x[1])
            confidence = f"{df_res['sentiment_compound'].abs().mean():.3f}"

        else:  # Hybrid
            analyzer = self._get_vader()
            df_res = analyzer.analyze_dataframe(df_res, use_transformer=False)
            pipe = self._get_transformer()
            if pipe is not None:
                # Only re-analyze messages where VADER is uncertain (compound near 0)
                col = 'message_cleaned' if 'message_cleaned' in df_res.columns else 'message'
                uncertain = df_res['sentiment_compound'].abs() < 0.2
                if uncertain.sum() > 0:
                    def _h(text):
                        try:
                            r = pipe(str(text)[:512])[0]
                            lbl = 'POSITIVE' if r['label'] == 'POSITIVE' else 'NEGATIVE'
                            sc  = r['score']
                            return lbl, round(sc if lbl == 'POSITIVE' else -sc, 4)
                        except Exception:
                            return None, None
                    for idx in df_res[uncertain].index:
                        lbl, comp = _h(df_res.at[idx, col])
                        if lbl:
                            df_res.at[idx, 'sentiment_vader']    = lbl
                            df_res.at[idx, 'sentiment_compound'] = comp
                extra['transformer_used'] = int(uncertain.sum())
            confidence = f"{df_res['sentiment_compound'].abs().mean():.3f}"

        elapsed = time.time() - t0
        metrics = {
            'model':           model,
            'processing_time': round(elapsed, 3),
            'confidence':      confidence,
            'message_count':   len(df_res),
            'unique_users':    df_res['user'].nunique(),
            'positive_pct':    round((df_res['sentiment_vader'] == 'POSITIVE').mean() * 100, 1),
            'negative_pct':    round((df_res['sentiment_vader'] == 'NEGATIVE').mean() * 100, 1),
            'neutral_pct':     round((df_res['sentiment_vader'] == 'NEUTRAL').mean() * 100, 1),
            **extra,
        }
        self.results[model] = {'df': df_res, 'metrics': metrics}
        return df_res, metrics

    def render_model_metrics(self, metrics: Dict):
        is_lt = st.session_state.get('theme', 'light') == 'light'
        ac  = '#B8883A' if is_lt else '#18C8E0'
        bg  = '#FFFFFF' if is_lt else 'rgba(17,24,39,0.75)'
        bdr = 'rgba(184,136,58,0.22)' if is_lt else 'rgba(24,163,183,0.14)'
        tc  = '#18120A' if is_lt else '#E2E8F0'
        sc  = '#6B5C3E' if is_lt else '#94A3B8'

        cards = [
            ('⏱️', 'Processing Time', f"{metrics['processing_time']}s",  '#18A3B7'),
            ('📊', 'Avg Confidence',  metrics['confidence'],              '#818CF8'),
            ('😊', 'Positive',        f"{metrics['positive_pct']}%",      '#4ADE80'),
            ('😔', 'Negative',        f"{metrics['negative_pct']}%",      '#F87171'),
            ('😐', 'Neutral',         f"{metrics['neutral_pct']}%",       '#FBBF24'),
            ('💬', 'Messages',        f"{metrics['message_count']:,}",    ac),
        ]
        cols = st.columns(len(cards))
        for col, (icon, label, val, color) in zip(cols, cards):
            col.markdown(
                f'<div style="background:{bg};border:1px solid {bdr};border-top:3px solid {color};'
                f'border-radius:12px;padding:14px;text-align:center;">'
                f'<div style="font-size:20px;">{icon}</div>'
                f'<div style="font-size:18px;font-weight:800;color:{color};margin:4px 0;">{val}</div>'
                f'<div style="font-size:10px;color:{sc};font-weight:600;text-transform:uppercase;'
                f'letter-spacing:.08em;">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    def render_sentiment_charts(self, df_res: pd.DataFrame, model_name: str):
        """Render sentiment distribution + per-user chart."""
        if 'sentiment_vader' not in df_res.columns:
            return

        # Pie chart
        dist = df_res['sentiment_vader'].value_counts()
        colors = {'POSITIVE': '#4ADE80', 'NEGATIVE': '#F87171', 'NEUTRAL': '#FBBF24'}
        fig_pie = go.Figure(go.Pie(
            labels=dist.index.tolist(),
            values=dist.values.tolist(),
            hole=0.52,
            marker=dict(colors=[colors.get(l, '#94A3B8') for l in dist.index]),
            textfont=dict(size=13, color='#E2E8F0'),
        ))
        fig_pie.update_layout(
            title=f'Sentiment Distribution — {model_name}',
            height=340, **_DARK_LAYOUT,
            legend=dict(orientation='h', y=-0.1, font=dict(color='#94A3B8')),
        )

        # Per-user stacked bar
        top_users = df_res['user'].value_counts().head(10).index
        sub = df_res[df_res['user'].isin(top_users)]
        pivot = sub.groupby(['user', 'sentiment_vader']).size().unstack(fill_value=0)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(1)

        fig_bar = go.Figure()
        for label, color in colors.items():
            if label in pivot_pct.columns:
                fig_bar.add_trace(go.Bar(
                    name=label,
                    x=pivot_pct.index.tolist(),
                    y=pivot_pct[label].tolist(),
                    marker_color=color,
                    opacity=0.88,
                ))
        fig_bar.update_layout(
            title='Sentiment % per User (Top 10)',
            barmode='stack', height=360,
            xaxis_title='', yaxis_title='%',
            legend=dict(orientation='h', y=1.08, font=dict(color='#94A3B8')),
            **_DARK_LAYOUT,
        )

        # Sentiment over time
        if 'datetime' in df_res.columns:
            trend = (
                df_res.set_index('datetime')
                .groupby([pd.Grouper(freq='D'), 'sentiment_vader'])
                .size()
                .unstack(fill_value=0)
            ).reset_index()
            fig_time = go.Figure()
            for label, color in colors.items():
                if label in trend.columns:
                    fig_time.add_trace(go.Scatter(
                        x=trend['datetime'], y=trend[label],
                        name=label, mode='lines',
                        line=dict(color=color, width=2),
                        fill='tozeroy',
                        fillcolor=color.replace(')', ',0.15)').replace('rgb', 'rgba') if 'rgb' in color else color + '26',
                    ))
            fig_time.update_layout(
                title='Sentiment Trend Over Time',
                height=300, **_DARK_LAYOUT,
                xaxis_title='', yaxis_title='Messages',
                legend=dict(orientation='h', y=1.08, font=dict(color='#94A3B8')),
            )
            return fig_pie, fig_bar, fig_time

        return fig_pie, fig_bar, None

    def get_model_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for name, res in self.results.items():
            m = res['metrics']
            rows.append({
                'Model':           name,
                'Time (s)':        m['processing_time'],
                'Avg Confidence':  m['confidence'],
                'Positive %':      m.get('positive_pct', '—'),
                'Negative %':      m.get('negative_pct', '—'),
                'Neutral %':       m.get('neutral_pct',  '—'),
                'Messages':        m['message_count'],
            })
        return pd.DataFrame(rows)

    def render_comparison_charts(self):
        """Visual comparison across all run models."""
        if len(self.results) < 2:
            return None

        names  = list(self.results.keys())
        pos    = [self.results[n]['metrics'].get('positive_pct', 0) for n in names]
        neg    = [self.results[n]['metrics'].get('negative_pct', 0) for n in names]
        neu    = [self.results[n]['metrics'].get('neutral_pct',  0) for n in names]
        times  = [self.results[n]['metrics']['processing_time'] for n in names]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Positive', x=names, y=pos,  marker_color='#4ADE80'))
        fig.add_trace(go.Bar(name='Negative', x=names, y=neg,  marker_color='#F87171'))
        fig.add_trace(go.Bar(name='Neutral',  x=names, y=neu,  marker_color='#FBBF24'))
        fig.update_layout(
            title='Model Comparison — Sentiment Distribution',
            barmode='group', height=360,
            xaxis_title='', yaxis_title='%',
            legend=dict(orientation='h', y=1.08, font=dict(color='#94A3B8')),
            **_DARK_LAYOUT,
        )

        fig_t = go.Figure(go.Bar(
            x=names, y=times,
            marker=dict(color=['#18A3B7','#818CF8','#F472B6','#FBBF24'][:len(names)]),
            text=[f'{t:.3f}s' for t in times],
            textposition='outside',
            textfont=dict(color='#E2E8F0'),
        ))
        fig_t.update_layout(
            title='Processing Time per Model',
            height=300, **_DARK_LAYOUT,
            xaxis_title='', yaxis_title='Seconds',
        )
        return fig, fig_t
