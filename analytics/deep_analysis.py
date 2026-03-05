"""
deep_analysis.py  —  6 Deep Analysis features
Place in: whatsapp-analyzer/analytics/deep_analysis.py

Features:
  1. Network Graph          – kaun kisse baat karta hai
  2. Response Time Analysis – fastest/slowest responders
  3. Word Cloud per User    – top words per person (plotly treemap)
  4. Personality Profile    – per-user communication style
  5. Monthly/Weekly Recap   – shareable stat cards (plotly figure)
  6. PDF Report Export      – full analysis PDF
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import io

# ── Shared palette ────────────────────────────────────────────────────────
_TEAL   = '#00C896'
_CYAN   = '#22D3EE'
_PINK   = '#F472B6'
_AMBER  = '#FBBF24'
_VIOLET = '#818CF8'
_RED    = '#F87171'
_GREEN  = '#4ADE80'
_PALETTE = [_TEAL, _CYAN, _PINK, _AMBER, _VIOLET, _RED, _GREEN,
            '#34D399', '#60A5FA', '#A78BFA', '#FB923C', '#E879F9']

_DARK_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,24,39,0.5)',
    font=dict(color='#94A3B8', family='Outfit, sans-serif', size=12),
    margin=dict(t=60, b=50, l=50, r=30),
    title_font=dict(color='#E2E8F0', size=15, family='Syne, sans-serif'),
)

_STOPWORDS = {
    'the','a','an','is','are','was','were','be','been','have','has','had',
    'do','does','did','will','would','could','should','to','of','in','on',
    'at','by','for','with','and','or','but','if','as','it','its','this',
    'that','i','me','my','we','our','you','your','he','his','she','her',
    'they','them','their','im','ive','id','ok','okay','hi','hey','yes','no',
    'yeah','nah','haha','lol','hmm','deleted','message','null','media',
    'omitted','https','http','pm','am','bhi','hai','nhi','kya','aur',
    'mein','ka','ke','ki','ko','se','ho','tha','thi','hh','haan','nahi',
    'koi','toh','yaar','bhai','ek','kuch','kar','raha','rahi',
}


def _safe_tokens(df: pd.DataFrame, user: str = None, n: int = 30) -> List[Tuple[str, int]]:
    rows = df[df['user'] == user] if user else df
    all_tok = []
    for tok in rows['tokens']:
        if isinstance(tok, list):
            all_tok.extend([t.lower() for t in tok if t.isalpha() and len(t) > 2])
    filtered = [t for t in all_tok if t not in _STOPWORDS]
    return Counter(filtered).most_common(n)


# ══════════════════════════════════════════════════════════════════════════
#  1. NETWORK GRAPH
# ══════════════════════════════════════════════════════════════════════════

def build_network_graph(df: pd.DataFrame) -> go.Figure:
    """
    Estimate who talks to whom based on consecutive messages within 5 min window.
    Returns an interactive Plotly network figure.
    """
    df2 = df.sort_values('datetime').reset_index(drop=True)
    edges = defaultdict(int)

    for i in range(1, len(df2)):
        u1 = df2.loc[i-1, 'user']
        u2 = df2.loc[i,   'user']
        dt = (df2.loc[i, 'datetime'] - df2.loc[i-1, 'datetime']).total_seconds()
        if u1 != u2 and dt < 300:   # within 5 minutes = likely a reply
            key = tuple(sorted([u1, u2]))
            edges[key] += 1

    if not edges:
        fig = go.Figure()
        fig.add_annotation(text="Not enough consecutive messages to build network",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color='#94A3B8', size=14))
        fig.update_layout(title='Conversation Network', **_DARK_LAYOUT)
        return fig

    users = list(df2['user'].unique())
    n     = len(users)
    idx   = {u: i for i, u in enumerate(users)}

    # Circular layout
    angles = [2 * np.pi * i / n for i in range(n)]
    x_pos  = {u: np.cos(a) for u, a in zip(users, angles)}
    y_pos  = {u: np.sin(a) for u, a in zip(users, angles)}

    # Message count per user (node size)
    msg_cnt = df2['user'].value_counts().to_dict()
    max_cnt = max(msg_cnt.values()) if msg_cnt else 1

    edge_traces = []
    max_weight  = max(edges.values()) if edges else 1

    for (u1, u2), w in edges.items():
        alpha = 0.15 + 0.6 * (w / max_weight)
        width = 0.5 + 3.5 * (w / max_weight)
        edge_traces.append(go.Scatter(
            x=[x_pos[u1], x_pos[u2], None],
            y=[y_pos[u1], y_pos[u2], None],
            mode='lines',
            line=dict(width=width, color=f'rgba(0,200,150,{alpha:.2f})'),
            hoverinfo='none',
            showlegend=False,
        ))

    node_x     = [x_pos[u] for u in users]
    node_y     = [y_pos[u] for u in users]
    node_sizes = [20 + 40 * (msg_cnt.get(u, 0) / max_cnt) for u in users]
    node_text  = [f"<b>{u}</b><br>{msg_cnt.get(u,0)} messages" for u in users]
    node_colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(users))]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=node_sizes, color=node_colors,
                    line=dict(color='rgba(0,0,0,0.3)', width=1)),
        text=users,
        textposition='top center',
        textfont=dict(size=11, color='#E2E8F0'),
        hovertext=node_text,
        hoverinfo='text',
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title='Conversation Network — Who Talks to Whom',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        **_DARK_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  2. RESPONSE TIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def response_time_analysis(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure, pd.DataFrame]:
    """
    Returns:
      fig_avg   — avg response time per user (bar)
      fig_dist  — distribution of response gaps (histogram)
      stats_df  — summary DataFrame
    """
    df2 = df.sort_values('datetime').reset_index(drop=True)
    response_times = defaultdict(list)

    for i in range(1, len(df2)):
        u_prev = df2.loc[i-1, 'user']
        u_curr = df2.loc[i,   'user']
        dt_sec = (df2.loc[i, 'datetime'] - df2.loc[i-1, 'datetime']).total_seconds()

        # Only count if different user and gap < 2 hours (not overnight silence)
        if u_prev != u_curr and 0 < dt_sec < 7200:
            response_times[u_curr].append(dt_sec / 60)   # in minutes

    if not response_times:
        empty = go.Figure()
        empty.add_annotation(text="Not enough data", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False,
                             font=dict(color='#94A3B8', size=14))
        empty.update_layout(**_DARK_LAYOUT)
        return empty, empty, pd.DataFrame()

    stats = {}
    for user, times in response_times.items():
        stats[user] = {
            'Avg Response (min)': round(np.mean(times), 1),
            'Median (min)':       round(np.median(times), 1),
            'Fastest (min)':      round(min(times), 1),
            'Replies Given':      len(times),
        }

    stats_df = pd.DataFrame(stats).T.sort_values('Avg Response (min)')

    # Bar chart — avg response time
    fig_avg = go.Figure(go.Bar(
        x=stats_df.index,
        y=stats_df['Avg Response (min)'],
        marker=dict(
            color=stats_df['Avg Response (min)'],
            colorscale=[[0, _TEAL], [0.5, _AMBER], [1, _RED]],
            showscale=True,
            colorbar=dict(title='Min', tickfont=dict(color='#94A3B8')),
        ),
        text=[f"{v:.1f}m" for v in stats_df['Avg Response (min)']],
        textposition='outside',
        textfont=dict(color='#E2E8F0'),
    ))
    fig_avg.update_layout(
        title='Average Response Time per User (lower = faster)',
        xaxis_title='', yaxis_title='Minutes',
        height=400, **_DARK_LAYOUT,
    )
    fig_avg.update_xaxes(gridcolor='rgba(0,200,150,0.06)')
    fig_avg.update_yaxes(gridcolor='rgba(0,200,150,0.06)')

    # Histogram — all response gaps
    all_times = [t for times in response_times.values() for t in times if t < 60]
    fig_dist = go.Figure(go.Histogram(
        x=all_times, nbinsx=30,
        marker=dict(color=_TEAL, opacity=0.75),
        name='Response Gap',
    ))
    fig_dist.update_layout(
        title='Distribution of Response Gaps (0–60 min)',
        xaxis_title='Minutes', yaxis_title='Count',
        height=350, **_DARK_LAYOUT,
    )

    return fig_avg, fig_dist, stats_df


# ══════════════════════════════════════════════════════════════════════════
#  3. WORD CLOUD per USER  (treemap — works without extra libs)
# ══════════════════════════════════════════════════════════════════════════

def word_cloud_treemap(df: pd.DataFrame, user: str = None, top_n: int = 40) -> go.Figure:
    """Plotly treemap as word cloud — works without wordcloud library."""
    label = user if user else "Everyone"
    top   = _safe_tokens(df, user, top_n)

    if not top:
        fig = go.Figure()
        fig.add_annotation(text=f"No words found for {label}",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color='#94A3B8', size=14))
        fig.update_layout(title=f'Top Words — {label}', **_DARK_LAYOUT)
        return fig

    words  = [w for w, _ in top]
    counts = [c for _, c in top]

    fig = go.Figure(go.Treemap(
        labels=words,
        parents=[''] * len(words),
        values=counts,
        textinfo='label+value',
        textfont=dict(size=13, color='#FFFFFF', family='Outfit'),
        marker=dict(
            colors=counts,
            colorscale=[[0, 'rgba(0,200,150,0.4)'], [0.5, '#00C896'], [1, '#22D3EE']],
            showscale=False,
            line=dict(color='rgba(0,0,0,0.2)', width=1),
        ),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>',
    ))
    fig.update_layout(
        title=f'Word Cloud — {label}',
        height=420,
        **_DARK_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  4. PERSONALITY PROFILE per USER
# ══════════════════════════════════════════════════════════════════════════

def personality_profiles(df: pd.DataFrame) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Score each user on 6 dimensions and return a radar chart + table.
    Dimensions: Positivity, Verbosity, Engagement, Emoji Use,
                Response Speed, Emotional Range
    """
    users   = df['user'].value_counts().index.tolist()
    results = {}

    # Precompute response times for speed score
    df2 = df.sort_values('datetime').reset_index(drop=True)
    resp_times = defaultdict(list)
    for i in range(1, len(df2)):
        u_p = df2.loc[i-1, 'user']
        u_c = df2.loc[i,   'user']
        dt  = (df2.loc[i, 'datetime'] - df2.loc[i-1, 'datetime']).total_seconds()
        if u_p != u_c and 0 < dt < 7200:
            resp_times[u_c].append(dt / 60)

    global_avg_len = df['message_length'].mean()
    max_resp_time  = max((np.mean(v) for v in resp_times.values() if v), default=60)

    for user in users:
        udf = df[df['user'] == user]
        if len(udf) < 3:
            continue

        # 1. Positivity (0-100)
        pos = (udf['sentiment_vader'] == 'POSITIVE').sum() / len(udf)
        neg = (udf['sentiment_vader'] == 'NEGATIVE').sum() / len(udf)
        positivity = round((pos - neg * 0.5 + 0.5) * 100, 1)
        positivity = min(max(positivity, 0), 100)

        # 2. Verbosity — avg message length vs global
        verbosity = round(min(udf['message_length'].mean() / max(global_avg_len, 1) * 60, 100), 1)

        # 3. Engagement — participation % (capped at 100)
        engagement = round(min(len(udf) / max(len(df), 1) * 100 * len(users), 100), 1)

        # 4. Emoji use — fraction of messages with emoji
        emoji_use = 0
        if 'emojis' in udf.columns:
            emoji_use = round(udf['emojis'].apply(lambda e: len(e) > 0 if hasattr(e,'__len__') else False).mean() * 100, 1)

        # 5. Response speed (inverted — fast = high score)
        rt = resp_times.get(user, [])
        if rt:
            avg_rt = np.mean(rt)
            speed  = round(max(0, (1 - avg_rt / max_resp_time)) * 100, 1)
        else:
            speed  = 50.0

        # 6. Emotional range — unique emotions / total emotions
        em_range = 0
        if 'emotion' in udf.columns:
            unique_em = udf['emotion'].nunique()
            em_range  = round(min(unique_em / 7, 1) * 100, 1)

        results[user] = {
            'Positivity':     positivity,
            'Verbosity':      verbosity,
            'Engagement':     engagement,
            'Emoji Use':      emoji_use,
            'Response Speed': speed,
            'Emotional Range': em_range,
        }

    if not results:
        fig = go.Figure()
        fig.update_layout(title='Personality Profiles', **_DARK_LAYOUT)
        return fig, pd.DataFrame()

    dims      = ['Positivity','Verbosity','Engagement','Emoji Use','Response Speed','Emotional Range']
    fig       = go.Figure()
    colors    = _PALETTE[:len(results)]

    for (user, scores), color in zip(results.items(), colors):
        vals = [scores[d] for d in dims] + [scores[dims[0]]]   # close polygon
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=dims + [dims[0]],
            fill='toself',
            name=user,
            line=dict(color=color, width=2),
            fillcolor=color.replace('#', 'rgba(').rstrip(')') if color.startswith('rgba') else f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)',
            opacity=0.85,
        ))

    fig.update_layout(
        title='Personality Profiles — Communication Style',
        polar=dict(
            bgcolor='rgba(17,24,39,0.5)',
            radialaxis=dict(visible=True, range=[0,100],
                           gridcolor='rgba(0,200,150,0.12)',
                           tickfont=dict(color='#64748B', size=9)),
            angularaxis=dict(gridcolor='rgba(0,200,150,0.12)',
                            tickfont=dict(color='#94A3B8', size=11)),
        ),
        showlegend=True,
        legend=dict(font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
        height=500,
        **_DARK_LAYOUT,
    )

    profile_df = pd.DataFrame(results).T.round(1)
    return fig, profile_df


def personality_label(scores: Dict[str, float]) -> str:
    """Generate a one-line personality label from scores."""
    p  = scores.get('Positivity', 50)
    v  = scores.get('Verbosity', 50)
    em = scores.get('Emoji Use', 50)
    sp = scores.get('Response Speed', 50)
    en = scores.get('Engagement', 50)

    if p > 70 and em > 60:  return "😊 The Cheerleader — positive, emoji-rich, uplifting"
    if v > 75 and en > 70:  return "📝 The Storyteller — verbose, highly engaged, expressive"
    if sp > 75 and en > 60: return "⚡ The Quick Responder — fast, engaged, always present"
    if p < 30:              return "😤 The Critic — tends toward negative or blunt responses"
    if v < 25 and sp < 40:  return "🤐 The Lurker — minimal messages, slow to respond"
    if em > 70:             return "🎭 The Emoji Queen/King — expresses via emojis heavily"
    if v > 60:              return "💬 The Talker — long messages, detailed communicator"
    return "🧘 The Balanced Participant — consistent, moderate communicator"


# ══════════════════════════════════════════════════════════════════════════
#  5. MONTHLY / WEEKLY RECAP
# ══════════════════════════════════════════════════════════════════════════

def monthly_recap(df: pd.DataFrame, period: str = 'M') -> go.Figure:
    """
    period: 'M' = monthly, 'W' = weekly
    Returns a bar+line combo chart as a recap card.
    """
    df2 = df.copy()
    df2['period'] = df2['datetime'].dt.to_period(period).astype(str)

    agg = df2.groupby('period').agg(
        messages=('message', 'count'),
        pos_pct=('sentiment_vader', lambda x: round((x == 'POSITIVE').sum() / len(x) * 100, 1)),
        unique_users=('user', 'nunique'),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg['period'], y=agg['messages'],
        name='Messages',
        marker=dict(
            color=agg['messages'],
            colorscale=[[0, 'rgba(0,200,150,0.3)'], [1, '#00C896']],
            showscale=False,
        ),
        yaxis='y1',
        hovertemplate='<b>%{x}</b><br>Messages: %{y}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=agg['period'], y=agg['pos_pct'],
        name='Positive %',
        mode='lines+markers',
        line=dict(color=_PINK, width=2.5),
        marker=dict(size=7, color=_PINK),
        yaxis='y2',
        hovertemplate='<b>%{x}</b><br>Positive: %{y:.1f}%<extra></extra>',
    ))

    import streamlit as _st2
    _is_lt2  = _st2.session_state.get('theme','light') == 'light'
    _fc2     = '#3E2F1C' if _is_lt2 else '#94A3B8'
    _tc2     = '#18120A' if _is_lt2 else '#E2E8F0'
    _bg2     = '#FFFBF2' if _is_lt2 else '#07090F'
    _grid2   = 'rgba(184,136,58,0.08)' if _is_lt2 else 'rgba(0,200,150,0.06)'
    _layout2 = dict(paper_bgcolor=_bg2, plot_bgcolor=_bg2,
                    font_color=_tc2, margin=dict(l=20,r=20,t=60,b=20))
    title = 'Monthly Recap' if period == 'M' else 'Weekly Recap'
    fig.update_layout(
        title=dict(text=title + ' — Messages & Positivity Trend',
                   font=dict(color='#B8883A' if _is_lt2 else '#00C896', size=14)),
        yaxis=dict(title=dict(text='Messages', font=dict(color=_TEAL)),
                   gridcolor=_grid2, tickfont=dict(color=_fc2)),
        yaxis2=dict(title=dict(text='Positive %', font=dict(color=_PINK)),
                    overlaying='y', side='right',
                    range=[0, 100], gridcolor='rgba(0,0,0,0)',
                    tickfont=dict(color=_fc2)),
        legend=dict(font=dict(color=_tc2), bgcolor='rgba(0,0,0,0)',
                    orientation='h', y=1.08),
        hovermode='x unified',
        height=400,
        **_layout2,
    )
    return fig


def recap_stat_card(df: pd.DataFrame) -> go.Figure:
    """Visual stat card — most active month, top user, best/worst day."""
    df2        = df.copy()
    df2['mo']  = df2['datetime'].dt.to_period('M').astype(str)
    df2['dow'] = df2['datetime'].dt.day_name()

    mo_counts  = df2.groupby('mo').size()
    best_mo    = mo_counts.idxmax() if len(mo_counts) else 'N/A'
    best_mo_n  = int(mo_counts.max()) if len(mo_counts) else 0

    top_user   = df2['user'].value_counts().index[0] if len(df2) > 0 else 'N/A'
    top_cnt    = int(df2['user'].value_counts().iloc[0]) if len(df2) > 0 else 0

    dow_counts = df2.groupby('dow').size()
    best_day   = dow_counts.idxmax() if len(dow_counts) else 'N/A'
    quiet_day  = dow_counts.idxmin() if len(dow_counts) else 'N/A'

    total_days = max((df2['datetime'].max() - df2['datetime'].min()).days, 1)
    avg_daily  = round(len(df2) / total_days, 1)

    labels = ['📅 Most Active Month', '👑 Top Contributor',
              '🔥 Busiest Day', '😴 Quietest Day', '📊 Daily Average']
    values = [f"{best_mo}\n{best_mo_n:,} msgs",
              f"{top_user}\n{top_cnt:,} msgs",
              best_day, quiet_day,
              f"{avg_daily} msgs/day"]

    import streamlit as _st
    _is_lt = _st.session_state.get('theme','light') == 'light'
    _val_c = '#18120A' if _is_lt else '#E2E8F0'
    _bg    = '#FFFBF2' if _is_lt else _DARK_LAYOUT.get('paper_bgcolor','#07090F')
    _layout = dict(paper_bgcolor=_bg, plot_bgcolor=_bg,
                   font_color=_val_c, margin=dict(l=20,r=20,t=50,b=10))

    fig = go.Figure()
    cols = _PALETTE[:len(labels)]

    for i, (lbl, val, col) in enumerate(zip(labels, values, cols)):
        fig.add_trace(go.Indicator(
            mode='number',
            value=0,
            number=dict(valueformat='', font=dict(size=1, color='rgba(0,0,0,0)')),
            title=dict(
                text=f"<span style='font-size:18px;color:{col};font-weight:700'>{lbl}</span>"
                     f"<br><span style='font-size:15px;color:{_val_c};font-weight:600'>{val}</span>",
                font=dict(size=14),
            ),
            domain=dict(x=[i/len(labels), (i+1)/len(labels)-0.02], y=[0, 1]),
        ))

    fig.update_layout(
        title=dict(text='Chat Recap — Key Stats',
                   font=dict(color='#00C896' if not _is_lt else '#B8883A', size=14)),
        height=220,
        **_layout,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  6. PDF REPORT EXPORT
# ══════════════════════════════════════════════════════════════════════════

def generate_pdf_report(df: pd.DataFrame, summary: Dict = None) -> bytes:
    """
    Generate a professional PDF report using reportlab.
    Returns bytes of the PDF.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        return b''

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # ── Custom styles ─────────────────────────────────────────────────────
    h1 = ParagraphStyle('H1', parent=styles['Heading1'],
                        fontSize=22, textColor=colors.HexColor('#00C896'),
                        spaceAfter=6, fontName='Helvetica-Bold')
    h2 = ParagraphStyle('H2', parent=styles['Heading2'],
                        fontSize=14, textColor=colors.HexColor('#0A0F1A'),
                        spaceBefore=14, spaceAfter=4, fontName='Helvetica-Bold')
    body = ParagraphStyle('Body', parent=styles['Normal'],
                          fontSize=10, textColor=colors.HexColor('#1E293B'),
                          spaceAfter=5, leading=15)
    caption = ParagraphStyle('Cap', parent=styles['Normal'],
                             fontSize=8, textColor=colors.HexColor('#64748B'),
                             spaceAfter=3)
    center = ParagraphStyle('Ctr', parent=styles['Normal'],
                            alignment=TA_CENTER, fontSize=10,
                            textColor=colors.HexColor('#475569'))

    # ── Stats ─────────────────────────────────────────────────────────────
    total    = len(df)
    users    = df['user'].nunique()
    days     = max((df['datetime'].max() - df['datetime'].min()).days, 1)
    top_user = df['user'].value_counts().index[0] if total else 'N/A'
    pos_pct  = round((df['sentiment_vader'] == 'POSITIVE').sum() / max(total,1) * 100, 1)
    neg_pct  = round((df['sentiment_vader'] == 'NEGATIVE').sum() / max(total,1) * 100, 1)
    tox_pct  = round((df['is_toxic'] == True).sum() / max(total,1) * 100, 1)
    date_str = (f"{df['datetime'].min().strftime('%d %b %Y')} – "
                f"{df['datetime'].max().strftime('%d %b %Y')}")

    # ── Page 1: Cover ─────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("WhatsApp AI Analytics", h1))
    story.append(Paragraph("Conversation Analysis Report", ParagraphStyle(
        'Sub', parent=styles['Normal'], fontSize=14,
        textColor=colors.HexColor('#475569'), spaceAfter=4)))
    story.append(Paragraph(date_str, caption))
    story.append(HRFlowable(width='100%', thickness=2,
                            color=colors.HexColor('#00C896'), spaceAfter=16))

    # Overview stats table
    overview_data = [
        ['Metric', 'Value'],
        ['Total Messages',   f'{total:,}'],
        ['Participants',     str(users)],
        ['Duration',         f'{days} days'],
        ['Most Active User', top_user],
        ['Positive Tone',    f'{pos_pct}%'],
        ['Negative Tone',    f'{neg_pct}%'],
        ['Toxicity Level',   f'{tox_pct}%'],
        ['Avg Messages/Day', f'{round(total/days,1)}'],
    ]
    tbl = Table(overview_data, colWidths=[8*cm, 8*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#00C896')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,0), 11),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#F8FAFC'), colors.HexColor('#F0FDF4')]),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWHEIGHT', (0,0), (-1,-1), 22),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── Page 2: Summary ────────────────────────────────────────────────────
    story.append(Paragraph("Conversation Summary", h2))
    story.append(HRFlowable(width='100%', thickness=1,
                            color=colors.HexColor('#E2E8F0'), spaceAfter=8))
    if summary and summary.get('conversation_summary'):
        story.append(Paragraph(summary['conversation_summary'], body))
    if summary and summary.get('detailed_narrative'):
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(summary['detailed_narrative'], body))

    # ── Key Insights ───────────────────────────────────────────────────────
    story.append(Paragraph("Key Insights", h2))
    story.append(HRFlowable(width='100%', thickness=1,
                            color=colors.HexColor('#E2E8F0'), spaceAfter=8))
    if summary and summary.get('key_insights'):
        for insight in summary['key_insights']:
            story.append(Paragraph(f"• {insight}", body))
    else:
        story.append(Paragraph("No insights available.", body))

    # ── User Engagement Table ──────────────────────────────────────────────
    story.append(Paragraph("User Engagement", h2))
    story.append(HRFlowable(width='100%', thickness=1,
                            color=colors.HexColor('#E2E8F0'), spaceAfter=8))

    user_stats = df.groupby('user').agg(
        Messages=('message', 'count'),
        Avg_Length=('message_length', 'mean'),
        Avg_Sentiment=('sentiment_compound', 'mean'),
    ).round(2).sort_values('Messages', ascending=False).head(20)

    user_data = [['User', 'Messages', 'Avg Length', 'Avg Sentiment']]
    for user_name, row in user_stats.iterrows():
        user_data.append([
            str(user_name)[:25],
            str(int(row['Messages'])),
            f"{row['Avg_Length']:.1f}",
            f"{row['Avg_Sentiment']:.2f}",
        ])

    u_tbl = Table(user_data, colWidths=[7*cm, 3.5*cm, 3.5*cm, 3.5*cm])
    u_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,0), 10),
        ('ALIGN',      (1,0), (-1,-1), 'CENTER'),
        ('ALIGN',      (0,0), (0,-1), 'LEFT'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#F8FAFC'), colors.white]),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#CBD5E1')),
        ('ROWHEIGHT', (0,0), (-1,-1), 20),
    ]))
    story.append(u_tbl)

    # ── Top Topics ─────────────────────────────────────────────────────────
    story.append(Paragraph("Top Discussion Topics", h2))
    story.append(HRFlowable(width='100%', thickness=1,
                            color=colors.HexColor('#E2E8F0'), spaceAfter=8))
    if summary and summary.get('most_discussed_topics'):
        topics_text = "  •  ".join(summary['most_discussed_topics'][:10])
        story.append(Paragraph(topics_text, body))

    # ── Mood Analysis ──────────────────────────────────────────────────────
    story.append(Paragraph("Sentiment Analysis", h2))
    story.append(HRFlowable(width='100%', thickness=1,
                            color=colors.HexColor('#E2E8F0'), spaceAfter=8))
    mood_data = [['Sentiment', 'Percentage', 'Interpretation']]
    if summary and summary.get('overall_mood'):
        mood = summary['overall_mood']
        mood_data += [
            ['Positive', f"{mood.get('positive',0):.1f}%",
             'Good' if mood.get('positive',0) > 50 else 'Moderate'],
            ['Neutral',  f"{mood.get('neutral',0):.1f}%",  'Balanced'],
            ['Negative', f"{mood.get('negative',0):.1f}%",
             'Concerning' if mood.get('negative',0) > 20 else 'Normal'],
        ]
    m_tbl = Table(mood_data, colWidths=[5*cm, 5*cm, 7.5*cm])
    m_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#00C896')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#F0FDF4'), colors.HexColor('#F8FAFC')]),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#CBD5E1')),
        ('ROWHEIGHT', (0,0), (-1,-1), 22),
    ]))
    story.append(m_tbl)

    # ── Footer ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width='100%', thickness=1,
                            color=colors.HexColor('#E2E8F0'), spaceAfter=6))
    from datetime import datetime
    story.append(Paragraph(
        f"Generated by WhatsApp AI Analytics Dashboard  •  {datetime.now().strftime('%d %b %Y %H:%M')}",
        center,
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════
#  7. GHOST MEMBERS & ACTIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def ghost_members_analysis(df: pd.DataFrame) -> tuple:
    """
    Identify ghost members, silent periods, and participation tiers.
    Returns: (fig_activity, ghost_df, tier_df)
    """
    total        = len(df)
    vc           = df['user'].value_counts()
    user_pcts    = (vc / total * 100).round(2)

    # Tier classification
    tiers = []
    for user, pct in user_pcts.items():
        if pct >= 15:     tier = ("🦁 Alpha", "#F87171")
        elif pct >= 5:    tier = ("🐯 Active", "#FBBF24")
        elif pct >= 1:    tier = ("🐦 Casual", "#60A5FA")
        elif pct >= 0.1:  tier = ("👻 Ghost", "#818CF8")
        else:             tier = ("🕳️ Shadow", "#475569")
        tiers.append({
            'User': user, 'Messages': int(vc[user]),
            'Share %': pct, 'Tier': tier[0], 'Color': tier[1],
        })

    tier_df = pd.DataFrame(tiers)

    # Bar chart colored by tier
    fig = go.Figure(go.Bar(
        x=tier_df['User'],
        y=tier_df['Share %'],
        marker_color=tier_df['Color'],
        text=[f"{p:.1f}%" for p in tier_df['Share %']],
        textposition='outside',
        textfont=dict(color='#E2E8F0', size=10),
        hovertemplate='<b>%{x}</b><br>Share: %{y:.2f}%<extra></extra>',
    ))
    fig.update_layout(
        title='Participation Tiers — Who Talks How Much',
        xaxis_title='', yaxis_title='Share of Messages (%)',
        height=420, showlegend=False,
        **_DARK_LAYOUT,
    )
    fig.update_xaxes(gridcolor='rgba(0,200,150,0.06)')
    fig.update_yaxes(gridcolor='rgba(0,200,150,0.06)')

    # Ghost members (< 1%)
    ghosts = tier_df[tier_df['Share %'] < 1.0].sort_values('Messages')

    return fig, ghosts, tier_df


# ══════════════════════════════════════════════════════════════════════════
#  8. EMOJI ANALYTICS
# ══════════════════════════════════════════════════════════════════════════

def emoji_analytics(df: pd.DataFrame) -> tuple:
    """
    Returns: (fig_top_emoji, fig_emoji_per_user, stats_dict)
    """
    from collections import Counter

    all_emojis = []
    user_emojis = {}

    for _, row in df.iterrows():
        user = row['user']
        emojis = list(row.get('emojis', []))
        all_emojis.extend(emojis)
        user_emojis.setdefault(user, []).extend(emojis)

    if not all_emojis:
        empty = go.Figure()
        empty.add_annotation(text="No emoji data found in this chat",
                             xref="paper", yref="paper", x=0.5, y=0.5,
                             showarrow=False, font=dict(color='#94A3B8', size=14))
        empty.update_layout(title='Emoji Analytics', **_DARK_LAYOUT)
        return empty, empty, {}

    emoji_counts = Counter(all_emojis)
    top_emojis   = emoji_counts.most_common(15)

    # Top emojis bar
    emojis_list = [e for e, _ in top_emojis]
    counts_list  = [c for _, c in top_emojis]

    fig_top = go.Figure(go.Bar(
        x=emojis_list,
        y=counts_list,
        marker=dict(
            color=counts_list,
            colorscale=[[0, 'rgba(0,200,150,0.4)'], [1, '#F472B6']],
            showscale=False,
        ),
        text=counts_list,
        textposition='outside',
        textfont=dict(color='#E2E8F0'),
    ))
    fig_top.update_layout(
        title='Top 15 Most Used Emojis',
        xaxis_title='', yaxis_title='Usage Count',
        height=380, **_DARK_LAYOUT,
    )

    # Emoji usage per user (stacked/grouped)
    top5_emojis = [e for e, _ in emoji_counts.most_common(5)]
    users       = df['user'].value_counts().index.tolist()[:10]

    fig_user = go.Figure()
    colors_em = [_TEAL, _PINK, _AMBER, _CYAN, _VIOLET]
    for em, color in zip(top5_emojis, colors_em):
        user_usage = []
        for user in users:
            cnt = user_emojis.get(user, []).count(em)
            user_usage.append(cnt)
        fig_user.add_trace(go.Bar(
            name=em, x=users, y=user_usage,
            marker_color=color, opacity=0.85,
        ))
    fig_user.update_layout(
        title='Top 5 Emoji Usage per User',
        barmode='stack', height=380,
        xaxis_title='', yaxis_title='Count',
        legend=dict(orientation='h', y=1.08, font=dict(color='#94A3B8')),
        **_DARK_LAYOUT,
    )

    stats = {
        'total_emojis':   len(all_emojis),
        'unique_emojis':  len(emoji_counts),
        'most_used':      top_emojis[0][0] if top_emojis else '—',
        'most_used_count': top_emojis[0][1] if top_emojis else 0,
        'top_emoji_user': max(user_emojis, key=lambda u: len(user_emojis[u])) if user_emojis else '—',
    }
    return fig_top, fig_user, stats


# ══════════════════════════════════════════════════════════════════════════
#  9. STREAK & SILENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def streak_analysis(df: pd.DataFrame) -> tuple:
    """
    Finds: longest active streak, longest silent period, daily activity calendar.
    Returns: (fig_calendar, fig_silence, streak_stats)
    """
    df2 = df.copy()
    df2['date'] = df2['datetime'].dt.date

    daily  = df2.groupby('date').size()
    all_dates = pd.date_range(daily.index.min(), daily.index.max(), freq='D').date
    daily_full = pd.Series(
        [daily.get(d, 0) for d in all_dates], index=all_dates
    )

    # Longest active streak
    max_streak = cur_streak = 0
    streak_start = streak_end = None
    cur_start = None

    for date, cnt in daily_full.items():
        if cnt > 0:
            cur_streak += 1
            if cur_start is None: cur_start = date
            if cur_streak > max_streak:
                max_streak   = cur_streak
                streak_start = cur_start
                streak_end   = date
        else:
            cur_streak = 0
            cur_start  = None

    # Longest silent period
    max_silence = cur_silence = 0
    silence_start = silence_end = None
    cur_sil_start = None

    for date, cnt in daily_full.items():
        if cnt == 0:
            cur_silence += 1
            if cur_sil_start is None: cur_sil_start = date
            if cur_silence > max_silence:
                max_silence   = cur_silence
                silence_start = cur_sil_start
                silence_end   = date
        else:
            cur_silence   = 0
            cur_sil_start = None

    # Calendar heatmap
    dates  = list(daily_full.index)
    values = list(daily_full.values)

    df_cal = pd.DataFrame({'date': pd.to_datetime(dates), 'count': values})
    df_cal['week']    = df_cal['date'].dt.isocalendar().week.astype(int)
    df_cal['weekday'] = df_cal['date'].dt.weekday
    df_cal['month']   = df_cal['date'].dt.strftime('%b %Y')

    fig_cal = go.Figure(go.Heatmap(
        x=df_cal['week'],
        y=df_cal['weekday'],
        z=df_cal['count'],
        colorscale=[
            [0.0, 'rgba(17,24,39,0.8)'],
            [0.3, 'rgba(0,200,150,0.3)'],
            [0.7, '#00C896'],
            [1.0, '#22D3EE'],
        ],
        hoverongaps=False,
        hovertemplate='%{z} messages<extra></extra>',
        showscale=True,
        colorbar=dict(title='Messages', tickfont=dict(color='#94A3B8')),
        ygap=3, xgap=1,
    ))
    day_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    fig_cal.update_layout(
        title='Activity Calendar (GitHub-style Heatmap)',
        yaxis=dict(tickvals=list(range(7)), ticktext=day_labels,
                   tickfont=dict(color='#94A3B8')),
        xaxis=dict(showticklabels=False),
        height=280, **_DARK_LAYOUT,
    )

    # Silent periods bar — top 10 gaps
    gaps = []
    prev_active = None
    for date, cnt in daily_full.items():
        if cnt > 0:
            if prev_active is not None:
                gap = (date - prev_active).days - 1
                if gap > 0:
                    gaps.append({'start': prev_active, 'end': date, 'days': gap})
            prev_active = date

    gaps = sorted(gaps, key=lambda g: g['days'], reverse=True)[:8]
    if gaps:
        fig_sil = go.Figure(go.Bar(
            x=[f"{g['start']} → {g['end']}" for g in gaps],
            y=[g['days'] for g in gaps],
            marker=dict(
                color=[g['days'] for g in gaps],
                colorscale=[[0, '#475569'], [1, '#F87171']],
                showscale=False,
            ),
            text=[f"{g['days']}d" for g in gaps],
            textposition='outside',
            textfont=dict(color='#E2E8F0'),
        ))
        fig_sil.update_layout(
            title='Top Silent Periods (Days with No Messages)',
            xaxis_title='', yaxis_title='Days of Silence',
            height=360, **_DARK_LAYOUT,
        )
        fig_sil.update_xaxes(tickangle=-30, tickfont=dict(size=9))
    else:
        fig_sil = go.Figure()
        fig_sil.add_annotation(text="No silent gaps found — very active chat!",
                               xref="paper", yref="paper", x=0.5, y=0.5,
                               showarrow=False, font=dict(color='#34D399', size=14))
        fig_sil.update_layout(title='Silent Periods', **_DARK_LAYOUT)

    streak_stats = {
        'longest_streak_days':  max_streak,
        'streak_start':         str(streak_start),
        'streak_end':           str(streak_end),
        'longest_silence_days': max_silence,
        'silence_start':        str(silence_start),
        'silence_end':          str(silence_end),
        'total_active_days':    int((daily_full > 0).sum()),
        'total_days':           len(daily_full),
        'activity_rate':        round((daily_full > 0).sum() / max(len(daily_full), 1) * 100, 1),
    }
    return fig_cal, fig_sil, streak_stats


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURES — Category 2, 3, 4, 5
# ══════════════════════════════════════════════════════════════════════════

# ── 2. CHAT FEATURES ─────────────────────────────────────────────────────

def reply_chain_analysis(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure, pd.DataFrame]:
    """Who replies to whom — reply chain & conversation starters."""
    import re as _re
    mc = 'message_cleaned' if 'message_cleaned' in df.columns else 'message'
    df2 = df.copy().sort_values('datetime').reset_index(drop=True)

    # Detect replies: "@name" or message within 5 min of prev user
    reply_pairs = []
    starter_counts = {}
    last_user = None
    last_time = None
    conv_start = True

    for i, row in df2.iterrows():
        u    = row['user']
        t    = row['datetime']
        msg  = str(row.get(mc, ''))

        # Conversation starter: first message after 30+ min gap
        if last_time is None or (t - last_time).total_seconds() > 1800:
            starter_counts[u] = starter_counts.get(u, 0) + 1

        # Reply pair: different user within 5 min
        if last_user and last_user != u and last_time:
            if (t - last_time).total_seconds() < 300:
                reply_pairs.append((last_user, u))

        last_user = u
        last_time = t

    # ── Reply heatmap ──────────────────────────────────────────────────
    if reply_pairs:
        from collections import Counter
        pair_counts = Counter(reply_pairs)
        users_set = sorted({u for p in reply_pairs for u in p})[:12]
        matrix = [[pair_counts.get((r, c), 0) for c in users_set] for r in users_set]

        fig_reply = go.Figure(go.Heatmap(
            z=matrix, x=users_set, y=users_set,
            colorscale=[[0,'#07090F'],[0.3,'rgba(0,200,150,0.3)'],[1,'#00C896']],
            text=[[str(v) if v>0 else '' for v in row] for row in matrix],
            texttemplate='%{text}', showscale=True,
            hovertemplate='%{y} → %{x}: %{z} replies<extra></extra>',
        ))
        fig_reply.update_layout(
            title='💬 Who Replies to Whom', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', font_color='#94A3B8',
            xaxis=dict(tickfont_size=10), yaxis=dict(tickfont_size=10),
            height=420, margin=dict(l=20,r=20,t=40,b=20),
        )
    else:
        fig_reply = go.Figure()

    # ── Conversation starters bar ──────────────────────────────────────
    top_starters = sorted(starter_counts.items(), key=lambda x: -x[1])[:10]
    names = [x[0] for x in top_starters]
    vals  = [x[1] for x in top_starters]
    colors = [f'rgba(0,200,150,{0.4+0.6*(v/max(vals,default=1))})' for v in vals]

    fig_start = go.Figure(go.Bar(
        x=vals, y=names, orientation='h',
        marker_color=colors,
        hovertemplate='%{y}: %{x} conversations started<extra></extra>',
        text=vals, textposition='outside',
    ))
    fig_start.update_layout(
        title='🚀 Conversation Starters', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', font_color='#94A3B8',
        height=350, margin=dict(l=20,r=60,t=40,b=20),
        xaxis=dict(showgrid=False), yaxis=dict(autorange='reversed'),
    )

    # ── Reply stats df ─────────────────────────────────────────────────
    from collections import Counter
    reply_recv = Counter(p[1] for p in reply_pairs)
    reply_sent = Counter(p[0] for p in reply_pairs)
    all_users  = sorted(set(list(reply_recv.keys()) + list(reply_sent.keys())))
    stats_df   = pd.DataFrame({
        'User':           all_users,
        'Replies Sent':   [reply_sent.get(u, 0) for u in all_users],
        'Replies Received':[reply_recv.get(u, 0) for u in all_users],
        'Conversations Started': [starter_counts.get(u, 0) for u in all_users],
    }).sort_values('Replies Received', ascending=False)

    return fig_reply, fig_start, stats_df


def best_friends_analysis(df: pd.DataFrame) -> Tuple[go.Figure, pd.DataFrame]:
    """Find most interacting pairs — best friends in the group."""
    from collections import Counter
    df2  = df.copy().sort_values('datetime').reset_index(drop=True)
    mc   = 'message_cleaned' if 'message_cleaned' in df.columns else 'message'
    pairs = []
    last_user = None
    last_time = None

    for _, row in df2.iterrows():
        u = row['user']
        t = row['datetime']
        if last_user and last_user != u and last_time:
            if (t - last_time).total_seconds() < 600:  # within 10 min
                key = tuple(sorted([last_user, u]))
                pairs.append(key)
        last_user = u
        last_time = t

    if not pairs:
        return go.Figure(), pd.DataFrame()

    pair_counts = Counter(pairs).most_common(15)
    labels = [f"{p[0]} & {p[1]}" for p, _ in pair_counts]
    values = [v for _, v in pair_counts]

    # Color gradient
    n = len(values)
    clrs = [f'rgba({int(0+200*(1-i/n))},{int(200-100*(i/n))},{int(150*(1-i/n))},0.8)' for i in range(n)]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker_color=clrs, text=values, textposition='outside',
        hovertemplate='%{y}: %{x} interactions<extra></extra>',
    ))
    fig.update_layout(
        title='💑 Best Friends — Most Interacting Pairs',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94A3B8', height=420,
        margin=dict(l=20,r=60,t=40,b=20),
        xaxis=dict(showgrid=False), yaxis=dict(autorange='reversed'),
    )

    df_pairs = pd.DataFrame({'Pair': [p for p,_ in pair_counts],
                              'Interactions': [v for _,v in pair_counts]})
    df_pairs['Person 1'] = df_pairs['Pair'].apply(lambda x: x[0])
    df_pairs['Person 2'] = df_pairs['Pair'].apply(lambda x: x[1])
    return fig, df_pairs[['Person 1','Person 2','Interactions']]


def night_owl_analysis(df: pd.DataFrame) -> Tuple[go.Figure, pd.DataFrame]:
    """Night owl vs Early bird leaderboard per user."""
    df2 = df.copy()
    df2['hour'] = df2['datetime'].dt.hour

    NIGHT = list(range(22, 24)) + list(range(0, 4))   # 10pm-4am
    EARLY = list(range(5, 9))                           # 5am-9am
    PEAK  = list(range(9, 22))                          # 9am-10pm

    users = df2['user'].value_counts().head(12).index.tolist()
    rows = []
    for u in users:
        udf   = df2[df2['user'] == u]
        total = max(len(udf), 1)
        night = (udf['hour'].isin(NIGHT)).sum() / total * 100
        early = (udf['hour'].isin(EARLY)).sum() / total * 100
        peak  = (udf['hour'].isin(PEAK)).sum() / total * 100
        label = '🦉 Night Owl' if night > 25 else ('🌅 Early Bird' if early > 20 else '☀️ Day Person')
        rows.append({'User': u, 'Night %': round(night,1), 'Early %': round(early,1),
                     'Day %': round(peak,1), 'Type': label})

    df_owl = pd.DataFrame(rows).sort_values('Night %', ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='🦉 Night (10pm-4am)', x=df_owl['User'],
                         y=df_owl['Night %'], marker_color='#8B5CF6'))
    fig.add_trace(go.Bar(name='☀️ Day (9am-10pm)',   x=df_owl['User'],
                         y=df_owl['Day %'],   marker_color='#F59E0B'))
    fig.add_trace(go.Bar(name='🌅 Early (5am-9am)',  x=df_owl['User'],
                         y=df_owl['Early %'], marker_color='#10B981'))
    fig.update_layout(
        barmode='stack', title='🦉 Night Owl vs Early Bird',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94A3B8', height=380, legend=dict(orientation='h',y=1.1),
        margin=dict(l=20,r=20,t=60,b=20),
        xaxis=dict(tickangle=-30), yaxis=dict(title='%', showgrid=False),
    )
    return fig, df_owl


# ── 3. MORE ANALYTICS ────────────────────────────────────────────────────

def deleted_messages_analysis(df: pd.DataFrame) -> dict:
    """Count and analyze deleted/media omitted messages."""
    import re as _re
    mc  = 'message_cleaned' if 'message_cleaned' in df.columns else 'message'
    raw = df.get('message', df[mc])

    deleted_re = _re.compile(r'this message was deleted|you deleted this message', _re.I)
    omit_re    = _re.compile(r'<(?:image|video|audio|document|sticker|gif|media)\s*omitted>|<media omitted>', _re.I)

    deleted = raw.apply(lambda x: bool(deleted_re.search(str(x))))
    omitted = raw.apply(lambda x: bool(omit_re.search(str(x))))

    del_by_user  = df[deleted]['user'].value_counts().head(8)
    omit_by_user = df[omitted]['user'].value_counts().head(8)

    fig = go.Figure()
    if len(del_by_user):
        fig.add_trace(go.Bar(name='🗑️ Deleted', x=del_by_user.index,
                             y=del_by_user.values, marker_color='#F87171'))
    if len(omit_by_user):
        fig.add_trace(go.Bar(name='📎 Media Omitted', x=omit_by_user.index,
                             y=omit_by_user.values, marker_color='#60A5FA'))
    fig.update_layout(
        barmode='group', title='🗑️ Deleted & Media Messages by User',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94A3B8', height=360, margin=dict(l=20,r=20,t=40,b=20),
        xaxis=dict(tickangle=-30), yaxis=dict(showgrid=False),
    )
    return {
        'fig':           fig,
        'total_deleted': int(deleted.sum()),
        'total_omitted': int(omitted.sum()),
        'del_by_user':   del_by_user,
        'omit_by_user':  omit_by_user,
    }


def conversation_flow_analysis(df: pd.DataFrame) -> go.Figure:
    """Hourly message flow animated over months — who talks when."""
    df2  = df.copy()
    df2['hour']  = df2['datetime'].dt.hour
    df2['month'] = df2['datetime'].dt.to_period('M').astype(str)

    top_users = df2['user'].value_counts().head(6).index.tolist()
    df3 = df2[df2['user'].isin(top_users)]

    hourly = df3.groupby(['hour','user']).size().reset_index(name='count')

    clrs = ['#00C896','#8B5CF6','#F59E0B','#F87171','#34D399','#60A5FA']
    fig  = go.Figure()
    for i, u in enumerate(top_users):
        udata = hourly[hourly['user'] == u]
        fig.add_trace(go.Scatter(
            x=udata['hour'], y=udata['count'], name=u,
            mode='lines+markers',
            line=dict(color=clrs[i % len(clrs)], width=2),
            fill='tozeroy', fillcolor='rgba({},{},{},0.08)'.format(int(clrs[i%len(clrs)][1:3],16),int(clrs[i%len(clrs)][3:5],16),int(clrs[i%len(clrs)][5:7],16)),
            marker=dict(size=5),
        ))
    fig.update_layout(
        title='📈 Message Flow by Hour (Top Users)',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94A3B8', height=380,
        xaxis=dict(title='Hour of Day', dtick=2, range=[0,23], showgrid=False),
        yaxis=dict(title='Messages', showgrid=False),
        legend=dict(orientation='h', y=1.1),
        margin=dict(l=20,r=20,t=60,b=20),
    )
    return fig


# ── 4. AI FEATURES ───────────────────────────────────────────────────────

def personality_quiz(df: pd.DataFrame) -> List[Dict]:
    """Generate fun personality type card for each top user."""
    from collections import Counter
    import re as _re

    mc = 'message_cleaned' if 'message_cleaned' in df.columns else 'message'
    users = df['user'].value_counts().head(8).index.tolist()

    TYPES = [
        ('The Comedian 😂',    'Always cracking jokes, uses lots of emojis, keeps spirits high.',       '#F59E0B'),
        ('The Info Dumper 📚', 'Sends long messages, shares articles and facts constantly.',             '#8B5CF6'),
        ('The Ghost 👻',       'Reads everything but rarely types. Lurks in the shadows.',               '#64748B'),
        ('The Peacemaker ☮️',  'Neutral tone, mediates conflicts, rarely negative.',                     '#10B981'),
        ('The Drama Queen 🎭', 'Expressive, emotional messages, high sentiment variance.',                '#F87171'),
        ('The Organizer 📋',   'Starts conversations, asks questions, keeps group on track.',            '#0EA5E9'),
        ('The Cheerleader 🎉', 'Mostly positive, reacts enthusiastically, uses lots of affirmations.',  '#34D399'),
        ('The Philosopher 🤔', 'Writes thoughtful long messages, asks deep questions.',                  '#A78BFA'),
        ('The News Anchor 📰', 'Shares lots of links, forwards news, keeps group informed.',             '#F97316'),
        ('The Silent Type 🤐', 'Very few messages but meaningful ones.',                                  '#94A3B8'),
    ]

    results = []
    for u in users:
        udf   = df[df['user'] == u]
        total = max(len(udf), 1)
        msgs  = udf[mc].fillna('').astype(str)

        avg_len   = msgs.str.len().mean()
        emoji_pct = msgs.apply(lambda x: len(_re.findall(r'[\U0001F300-\U0001FFFF]', x))).sum() / total
        pos_pct   = (udf.get('sentiment_vader', pd.Series(['NEUTRAL']*total)) == 'POSITIVE').sum() / total
        neg_pct   = (udf.get('sentiment_vader', pd.Series(['NEUTRAL']*total)) == 'NEGATIVE').sum() / total
        url_pct   = msgs.apply(lambda x: bool(_re.search(r'https?://', x))).sum() / total
        part_pct  = total / len(df)

        # Score each type
        scores = {
            'The Comedian 😂':    emoji_pct * 3 + pos_pct,
            'The Info Dumper 📚': avg_len / 100 + url_pct * 2,
            'The Ghost 👻':       1 - part_pct * 10,
            'The Peacemaker ☮️':  (1 - neg_pct) * pos_pct,
            'The Drama Queen 🎭': neg_pct * 2 + avg_len / 80,
            'The Organizer 📋':   part_pct * 5,
            'The Cheerleader 🎉': pos_pct * 2 + emoji_pct,
            'The Philosopher 🤔': avg_len / 60,
            'The News Anchor 📰': url_pct * 4,
            'The Silent Type 🤐': (1 - part_pct * 15) if part_pct < 0.05 else 0,
        }

        best_type = max(scores, key=scores.get)
        type_info = next((t for t in TYPES if t[0] == best_type), TYPES[0])

        results.append({
            'user':        u,
            'type':        type_info[0],
            'desc':        type_info[1],
            'color':       type_info[2],
            'messages':    total,
            'avg_len':     round(avg_len, 1),
            'pos_pct':     round(pos_pct * 100, 1),
            'emoji_rate':  round(emoji_pct, 2),
        })

    return results


def roast_generator(df: pd.DataFrame) -> List[Dict]:
    """Generate fun (not mean) roast lines for top users based on their patterns."""
    import re as _re
    mc    = 'message_cleaned' if 'message_cleaned' in df.columns else 'message'
    users = df['user'].value_counts().head(6).index.tolist()
    total = len(df)

    roasts = []
    for u in users:
        udf   = df[df['user'] == u]
        n     = len(udf)
        msgs  = udf[mc].fillna('').astype(str)
        pct   = round(n / max(total,1) * 100, 1)
        avg_len = round(msgs.str.len().mean(), 0)
        hour  = udf['datetime'].dt.hour.mean()
        emojis = msgs.apply(lambda x: len(_re.findall(r'[\U0001F300-\U0001FFFF]', x))).sum()
        pos   = (udf.get('sentiment_vader', pd.Series(['NEUTRAL']*n)) == 'POSITIVE').sum()

        lines = []

        # Volume
        if pct > 25:
            lines.append(f"Responsible for {pct}% of messages — bhai, yeh group hai ya tera personal diary?")
        elif pct < 2:
            lines.append(f"Only {pct}% messages — read receipts ON, replies OFF. Classic lurker energy.")

        # Timing
        if hour >= 23 or hour < 3:
            lines.append("Consistently active at 1am+ — bhai so ja, subah college/office hai.")
        elif hour < 7:
            lines.append("Most active at 6am — either bakery mein kaam karta hai ya neend nahi aati.")

        # Length
        if avg_len > 120:
            lines.append(f"Average message: {int(avg_len)} characters. Bhai novel likh raha hai kya?")
        elif avg_len < 8:
            lines.append("Average reply: less than 8 characters. 'ok', 'hm', 'k' — communication at its finest.")

        # Emojis
        if emojis > n * 2:
            lines.append(f"Used {emojis} emojis — emoji keyboard ka sabse zyada istemaal.")

        # Sentiment
        if pos / max(n,1) > 0.6:
            lines.append("So positive, it's suspicious. Nobody is this happy all the time.")

        if not lines:
            lines.append(f"{n} messages, all perfectly average. The Switzerland of this group chat.")

        roasts.append({
            'user':   u,
            'roast':  ' '.join(lines[:2]),   # max 2 lines
            'stats':  f"{n} msgs · {pct}% · avg {int(avg_len)} chars",
            'color':  '#F59E0B' if pct > 15 else ('#94A3B8' if pct < 3 else '#00C896'),
        })

    return roasts


# ── 5. UI HELPERS ─────────────────────────────────────────────────────────

def render_personality_cards(results: List[Dict], is_light: bool) -> None:
    """Render personality quiz cards in Streamlit."""
    import streamlit as st
    cols = st.columns(2)
    for i, r in enumerate(results):
        bg  = '#FFFBF2' if is_light else 'rgba(17,24,39,0.75)'
        tc  = '#18120A' if is_light else '#E2E8F0'
        sc  = '#7A6248' if is_light else '#94A3B8'
        with cols[i % 2]:
            st.markdown(
                f'<div style="background:{bg};border:1px solid {r["color"]}44;'
                f'border-left:4px solid {r["color"]};border-radius:14px;'
                f'padding:16px 18px;margin-bottom:12px;">'
                f'<div style="font-size:11px;color:{sc};margin-bottom:4px;font-family:monospace;">{r["user"]}</div>'
                f'<div style="font-size:16px;font-weight:700;color:{r["color"]};margin-bottom:6px;">{r["type"]}</div>'
                f'<div style="font-size:12px;color:{sc};line-height:1.6;margin-bottom:8px;">{r["desc"]}</div>'
                f'<div style="display:flex;gap:12px;flex-wrap:wrap;">'
                f'<span style="font-size:10px;color:{tc};background:{r["color"]}18;'
                f'padding:2px 8px;border-radius:20px;">{r["messages"]} msgs</span>'
                f'<span style="font-size:10px;color:{tc};background:{r["color"]}18;'
                f'padding:2px 8px;border-radius:20px;">{r["pos_pct"]}% positive</span>'
                f'<span style="font-size:10px;color:{tc};background:{r["color"]}18;'
                f'padding:2px 8px;border-radius:20px;">avg {r["avg_len"]} chars</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )


def render_roast_cards(roasts: List[Dict], is_light: bool) -> None:
    """Render roast cards in Streamlit."""
    import streamlit as st
    for r in roasts:
        bg  = '#FFFBF2' if is_light else 'rgba(17,24,39,0.75)'
        tc  = '#18120A' if is_light else '#E2E8F0'
        sc  = '#7A6248' if is_light else '#64748B'
        st.markdown(
            f'<div style="background:{bg};border:1px solid {r["color"]}44;'
            f'border-left:4px solid {r["color"]};border-radius:12px;'
            f'padding:14px 18px;margin-bottom:10px;display:flex;gap:14px;align-items:flex-start;">'
            f'<div style="font-size:28px;flex-shrink:0;">🔥</div>'
            f'<div style="flex:1;">'
            f'<div style="font-weight:700;color:{r["color"]};font-size:14px;margin-bottom:4px;">{r["user"]}</div>'
            f'<div style="font-size:13px;color:{tc};line-height:1.65;margin-bottom:6px;">{r["roast"]}</div>'
            f'<div style="font-size:10px;color:{sc};font-family:monospace;">{r["stats"]}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )