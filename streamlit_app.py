import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from src.modules.chat_parser import WhatsAppParser
from src.modules.data_cleaner import DataCleaner
from src.modules.analytics import BasicAnalytics
from src.modules.sentiment_analyzer import SentimentAnalyzer
from src.modules.emotion_detector import EmotionDetector
from src.modules.behavioral_analysis import BehavioralAnalyzer
from src.modules.multilingual import MultilingualAnalyzer
from src.modules.export_handler import ExportHandler

from ui.filters import FilterSystem
from ui.chat_explorer import ChatExplorer
from ui.styling import (
    apply_modern_theme,
    render_theme_selector,
    render_kpi_card,
    render_gradient_divider,
)

from models.model_manager import ModelManager
from analytics.summary_generator import SummaryGenerator
from analytics.ai_summary import generate_ai_summary, PROVIDERS, detect_provider
from analytics.auto_summary import generate_auto_summary
from analytics.advanced_viz import AdvancedVisualizations
from analytics.deep_analysis import (
    build_network_graph, response_time_analysis,
    word_cloud_treemap, personality_profiles, personality_label,
    monthly_recap, recap_stat_card, generate_pdf_report,
    ghost_members_analysis, emoji_analytics, streak_analysis,
    reply_chain_analysis, best_friends_analysis, night_owl_analysis,
    deleted_messages_analysis, conversation_flow_analysis,
    personality_quiz, roast_generator,
    render_personality_cards, render_roast_cards,
)
from utils_dash.export_advanced import AdvancedExportSystem
from src.utils.file_handler import save_uploaded_file


def _card(light_bg="#FFFFFF", light_bdr="rgba(184,136,58,0.22)",
          dark_bg="rgba(17,24,39,0.75)", dark_bdr="rgba(0,200,150,0.14)"):
    """Return theme-appropriate card background and border."""
    is_light = st.session_state.get("theme","light") == "light"
    return (light_bg if is_light else dark_bg), (light_bdr if is_light else dark_bdr)

def _tc(light="#18120A", dark="#E2E8F0"):
    """Theme-aware text color."""
    return light if st.session_state.get("theme","light") == "light" else dark

def _sc(light="#3E2F1C", dark="#CBD5E1"):
    """Theme-aware secondary/muted text color."""
    return light if st.session_state.get("theme","light") == "light" else dark


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WhatsApp Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar: theme selector + theme CSS (must run before any other rendering) ─
with st.sidebar:
    render_theme_selector()   # sets st.session_state.theme & injects CSS

    st.markdown("### 🚀 Quick Start")

    uploaded_file = st.file_uploader(
        "Upload WhatsApp Chat",
        type=["txt"],
        help="Export: WhatsApp > Settings > Chats > Export chat",
    )

    if uploaded_file is not None:
        # Skip re-processing if same file already analyzed
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("_file_key") == file_key:
            st.success("✓ Analysis complete!")
        else:
            file_path = save_uploaded_file(uploaded_file, "whatsapp-analyzer/data/raw")
            prog = st.progress(0, text="📂 Parsing chat...")
            try:
                parser     = WhatsAppParser()
                df         = parser.parse_file(file_path)
                prog.progress(20, text="🧹 Cleaning data...")

                cleaner    = DataCleaner()
                df_cleaned = cleaner.clean_dataframe(df)
                prog.progress(45, text="💭 Sentiment analysis...")

                sentiment_analyzer = SentimentAnalyzer()
                df_cleaned = sentiment_analyzer.analyze_dataframe(df_cleaned, use_transformer=False)
                prog.progress(65, text="🎭 Emotion detection...")

                emotion_detector = EmotionDetector()
                df_cleaned = emotion_detector.analyze_dataframe(df_cleaned)
                prog.progress(80, text="📊 Behavioral analysis...")

                behavioral_analyzer = BehavioralAnalyzer()
                df_cleaned = behavioral_analyzer.analyze_dataframe(df_cleaned)
                prog.progress(92, text="🌐 Multilingual check...")

                multilingual_analyzer = MultilingualAnalyzer()
                df_cleaned = multilingual_analyzer.analyze_dataframe(df_cleaned)
                prog.progress(100, text="✅ Done!")

                st.session_state.df_cleaned          = df_cleaned
                st.session_state.sentiment_analyzer  = sentiment_analyzer
                st.session_state.emotion_detector    = emotion_detector
                st.session_state.behavioral_analyzer = behavioral_analyzer
                st.session_state.file_name           = uploaded_file.name
                st.session_state._file_key           = file_key
                prog.empty()
                st.success("✓ Analysis complete!")
            except Exception as e:
                prog.empty()
                st.error(f"Error: {e}")

# ── Main page header (hidden — shown in onboarding / dashboard) ──────────────
render_gradient_divider()

# ── Data-dependent UI ────────────────────────────────────────────────────────
if "df_cleaned" in st.session_state:
    df_cleaned = st.session_state.df_cleaned

    st.sidebar.divider()

    filter_system = FilterSystem()
    filters = filter_system.render_sidebar_filters(df_cleaned)
    df_filtered = filter_system.apply_filters(df_cleaned, filters)

    st.sidebar.info(f"Showing {len(df_filtered)} of {len(df_cleaned)} messages")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 Dashboard",
        "🎛️ Models",
        "💬 Chat Explorer",
        "✨ Insights",
        "🤖 AI Summary",
        "📈 Advanced Analytics",
        "🔬 Deep Analysis",
        "🏆 Leaderboard",
        "💾 Export",
        "⚙️ Settings",
    ])

    # ── Tab 1: Dashboard ─────────────────────────────────────────────────────
    with tab1:
        st.header("Dashboard Overview")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            render_kpi_card("Total Messages", str(len(df_filtered)), "Analyzed", "📝", 0)
        with col2:
            render_kpi_card("Unique Users", str(df_filtered["user"].nunique()), "Participants", "👥", 1)
        with col3:
            sentiment_counts = df_filtered["sentiment_vader"].value_counts()
            positive_pct = (sentiment_counts.get("POSITIVE", 0) / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
            render_kpi_card("Sentiment", f"{positive_pct:.0f}%", "Positive", "😊", 2)
        with col4:
            avg_length = df_filtered["message_length"].mean()
            render_kpi_card("Avg Length", f"{avg_length:.0f}", "Characters", "📏", 3)
        with col5:
            toxic_pct = (df_filtered["is_toxic"].sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
            render_kpi_card("Toxicity", f"{toxic_pct:.1f}%", "Level", "⚠️", 4)

        render_gradient_divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Sentiment Distribution")
            analytics = BasicAnalytics(df_filtered)
            st.plotly_chart(analytics.plot_user_activity(), use_container_width=True)
        with col2:
            st.subheader("📈 Activity Timeline")
            st.plotly_chart(analytics.plot_activity_timeline(), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔥 Activity Heatmap")
            st.plotly_chart(analytics.plot_hourly_heatmap(), use_container_width=True)
        with col2:
            st.subheader("🔤 Top Words")
            st.plotly_chart(analytics.plot_word_frequency(), use_container_width=True)

    # ── Tab 2: Models ────────────────────────────────────────────────────────
    with tab2:
        st.header("Model Selection & Comparison")
        model_manager = ModelManager()
        selected_model = model_manager.render_model_selector()

        with st.spinner(f"Analyzing with {selected_model}..."):
            df_analyzed, metrics = model_manager.analyze_with_model(df_filtered, selected_model)

        model_manager.render_model_metrics(metrics)
        st.subheader("Model Comparison")

        if len(model_manager.results) > 1:
            comparison_df = model_manager.get_model_comparison(df_filtered)
            st.dataframe(comparison_df, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Processing Time")
                times = {name: result["metrics"]["processing_time"] for name, result in model_manager.results.items()}
                st.bar_chart(pd.Series(times))
            with col2:
                st.subheader("Model Confidence")
                st.info("Lower processing time = VADER (rule-based)\nHigher processing time = Transformer (deep learning)")

    # ── Tab 3: Chat Explorer ─────────────────────────────────────────────────
    with tab3:
        st.header("Live Chat Explorer")
        chat_explorer = ChatExplorer(df_filtered)
        chat_explorer.render_chat_explorer()
        st.divider()
        chat_explorer.render_user_stats()

    # ── Tab 4: Insights ──────────────────────────────────────────────────────
    with tab4:
        # ── Theme vars for Insights tab ──────────────────────────────────────
        _hl_bg  = '#FFFFFF' if st.session_state.get('theme','light')=='light' else 'rgba(17,24,39,0.75)'
        _hl_bdr = 'rgba(184,136,58,0.22)' if st.session_state.get('theme','light')=='light' else 'rgba(0,200,150,0.15)'
        _hl_tc  = '#3E2F1C' if st.session_state.get('theme','light')=='light' else '#CBD5E1'
        _hl_vc  = '#18120A' if st.session_state.get('theme','light')=='light' else '#E2E8F0'
        summary_gen = SummaryGenerator(df_filtered)
        summary = summary_gen.generate_summary()

        # ── Hero summary card ──────────────────────────────────────────────
        st.markdown("""
        <div style="margin-bottom:6px;">
            <span style="font-family:'Outfit',sans-serif; font-size:9px; font-weight:700;
                letter-spacing:0.20em; text-transform:uppercase; color:#00C896; opacity:0.9;">
                🤖 AI-Powered Insights
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(0,200,150,0.07),rgba(34,211,238,0.04));
            border:1px solid rgba(0,200,150,0.25); border-left:4px solid #00C896;
            border-radius:14px; padding:22px 26px; margin-bottom:20px;">
            <div style="font-size:11px; font-weight:700; letter-spacing:0.16em;
                text-transform:uppercase; color:#00C896; margin-bottom:10px;">
                📋 Conversation Overview
            </div>
            <div style="font-size:15px; line-height:1.85; color:{_hl_vc}; font-family:'Outfit',sans-serif;">
                {summary["conversation_summary"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Narrative ─────────────────────────────────────────────────────
        if summary.get("detailed_narrative"):
            _narr_bg = 'rgba(255,250,240,0.9)' if st.session_state.get('theme','light')=='light' else 'rgba(17,24,39,0.75)'
            _narr_lc = '#3E2F1C' if st.session_state.get('theme','light')=='light' else '#CBD5E1'
            _narr_tc = '#3E2F1C' if st.session_state.get('theme','light')=='light' else '#E2E8F0'
            st.markdown(f"""
            <div style="background:{_narr_bg}; border:1px solid rgba(0,200,150,0.18);
                border-radius:12px; padding:18px 24px; margin-bottom:22px;">
                <div style="font-size:10px; font-weight:700; letter-spacing:0.16em;
                    text-transform:uppercase; color:{_narr_lc}; margin-bottom:9px;">
                    📖 Detailed Narrative
                </div>
                <div style="font-size:14px; line-height:1.8; color:{_narr_tc}; font-family:'Outfit',sans-serif;">
                    {summary["detailed_narrative"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

        render_gradient_divider()

        # ── Key Insights cards ────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:10px; font-weight:700; letter-spacing:0.18em;
            text-transform:uppercase; color:#00C896; margin-bottom:14px;">
            🔍 Key Insights
        </div>
        """, unsafe_allow_html=True)

        insights = summary["key_insights"]
        cols_per_row = 2
        for i in range(0, len(insights), cols_per_row):
            row_insights = insights[i:i+cols_per_row]
            cols = st.columns(cols_per_row)
            for j, insight in enumerate(row_insights):
                with cols[j]:
                    _i_bg = 'rgba(255,252,245,0.95)' if st.session_state.get('theme','light')=='light' else 'rgba(17,24,39,0.75)'
                    _i_tc = '#18120A' if st.session_state.get('theme','light')=='light' else '#E2E8F0'
                    _i_bd = 'rgba(0,200,150,0.18)' if st.session_state.get('theme','light')=='light' else 'rgba(0,200,150,0.20)'
                    st.markdown(f"""
                    <div style="background:{_i_bg}; border:1px solid {_i_bd};
                        border-radius:12px; padding:14px 18px; margin-bottom:10px;
                        border-left:3px solid #00C896;">
                        <div style="font-size:13.5px; color:{_i_tc}; line-height:1.65;
                            font-family:'Outfit',sans-serif;">
                            {insight}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        render_gradient_divider()

        # ── 3-col stats ───────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)

        with col1:
            mood = summary["overall_mood"]
            st.markdown("""
            <div style="font-size:10px; font-weight:700; letter-spacing:0.15em;
                text-transform:uppercase; color:#00C896; margin-bottom:12px;">
                😊 Overall Mood
            </div>""", unsafe_allow_html=True)

            for label, val, color in [
                ("Positive", mood["positive"], "#4ADE80"),
                ("Neutral",  mood["neutral"],  "#94A3B8"),
                ("Negative", mood["negative"], "#F87171"),
            ]:
                bar_w = max(val, 2)
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-size:12px; color:{_sc()}">{label}</span>
                        <span style="font-size:12px; color:{color}; font-family:'Fira Code',monospace;
                            font-weight:600;">{val:.1f}%</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.06); border-radius:4px; height:6px;">
                        <div style="width:{bar_w}%; background:{color}; border-radius:4px;
                            height:6px; transition:width 0.5s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            topics = summary["most_discussed_topics"]
            st.markdown("""
            <div style="font-size:10px; font-weight:700; letter-spacing:0.15em;
                text-transform:uppercase; color:#00C896; margin-bottom:12px;">
                🎯 Top Topics
            </div>""", unsafe_allow_html=True)
            for idx, topic in enumerate(topics[:8], 1):
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                    <span style="font-size:10px; color:#00C896; font-family:'Fira Code',monospace;
                        font-weight:600; min-width:18px;">#{idx}</span>
                    <span style="font-size:13px; color:{_hl_vc}; font-family:'Outfit',sans-serif;
                        text-transform:capitalize;">{topic}</span>
                </div>
                """, unsafe_allow_html=True)

        with col3:
            conflict = summary["conflict_detection"]
            dyn = summary.get("relationship_dynamics", {})
            risk = conflict["risk_level"]
            risk_color = {"Low":"#4ADE80","Medium":"#FBBF24","High":"#F87171"}.get(risk,"#94A3B8")

            st.markdown("""
            <div style="font-size:10px; font-weight:700; letter-spacing:0.15em;
                text-transform:uppercase; color:#00C896; margin-bottom:12px;">
                ⚠️ Conflict & Balance
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:rgba(255,250,240,0.9); border:1px solid rgba(0,200,150,0.18);
                border-radius:10px; padding:14px 16px;">
                <div style="margin-bottom:8px;">
                    <span style="font-size:11px; color:{_sc()}">Risk Level</span><br>
                    <span style="font-size:18px; font-weight:700; color:{risk_color};
                        font-family:'Syne',sans-serif;">{risk}</span>
                </div>
                <div style="margin-bottom:8px;">
                    <span style="font-size:11px; color:#64748B;">Conflict Score</span><br>
                    <span style="font-size:15px; font-weight:600; color:#CBD5E1;
                        font-family:'Fira Code',monospace;">{conflict["conflict_score"]:.1f}%</span>
                </div>
                <div style="margin-bottom:8px;">
                    <span style="font-size:11px; color:#64748B;">Toxic Messages</span><br>
                    <span style="font-size:15px; font-weight:600; color:#F87171;
                        font-family:'Fira Code',monospace;">{conflict["toxic_messages"]}</span>
                </div>
                <div>
                    <span style="font-size:11px; color:#64748B;">Participation</span><br>
                    <span style="font-size:13px; color:#94A3B8;">{dyn.get("balance_label","—")}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        render_gradient_divider()

        # ── Notable Highlights ────────────────────────────────────────────
        highlights = summary.get("conversation_highlights", [])
        if highlights:
            st.markdown("""
            <div style="font-size:10px; font-weight:700; letter-spacing:0.18em;
                text-transform:uppercase; color:#00C896; margin-bottom:14px;">
                ✨ Notable Messages
            </div>
            """, unsafe_allow_html=True)

            h_cols = st.columns(len(highlights))
            # (_hl_* vars defined at tab start)
            for i, h in enumerate(highlights):
                with h_cols[i]:
                    st.markdown(f"""
                    <div style="background:{_hl_bg}; border:1px solid {_hl_bdr};
                        border-radius:12px; padding:16px 18px; height:100%;">
                        <div style="font-size:10px; font-weight:700; letter-spacing:0.12em;
                            text-transform:uppercase; color:#00C896; margin-bottom:8px;">
                            {h["type"]}
                        </div>
                        <div style="font-size:11px; color:{_hl_tc}; margin-bottom:6px;">
                            👤 {h["user"]}
                        </div>
                        <div style="font-size:13px; color:#94A3B8; line-height:1.6;
                            font-style:italic; font-family:'Outfit',sans-serif;">
                            "{h["message"]}"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        render_gradient_divider()

        # ── User Engagement Table ─────────────────────────────────────────
        st.markdown("""
        <div style="font-size:10px; font-weight:700; letter-spacing:0.18em;
            text-transform:uppercase; color:#00C896; margin-bottom:14px;">
            👥 User Engagement
        </div>
        """, unsafe_allow_html=True)
        try:
            engagement_df = pd.DataFrame(summary["user_engagement"]).T
            if "messages" in engagement_df.columns:
                engagement_df = engagement_df.sort_values("messages", ascending=False)
            st.dataframe(engagement_df, use_container_width=True)
        except Exception:
            # Fallback: compute directly
            mc2 = 'message_cleaned' if 'message_cleaned' in df_filtered.columns else 'message'
            _eng = df_filtered.groupby('user').agg(
                messages=('user','count'),
                avg_length=('message_length','mean'),
                sentiment=('sentiment_compound','mean'),
                participation=('user', lambda x: round(len(x)/max(len(df_filtered),1)*100,1))
            ).round(2).sort_values('messages', ascending=False)
            st.dataframe(_eng, use_container_width=True)

    # ── Tab 5: AI Summary ────────────────────────────────────────────────────
    with tab5:
        # ── Header ───────────────────────────────────────────────────────────
        st.markdown(
            '<div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;">'
            '<div style="width:40px;height:40px;border-radius:12px;'
            'background:linear-gradient(135deg,#7C3AED,#00C896);'
            'display:flex;align-items:center;justify-content:center;font-size:20px;">🤖</div>'
            '<div>'
            '<div style="font-size:1.4rem;font-weight:700;color:#FFFFFF;font-family:Syne,sans-serif;">AI Chat Summary</div>'
            '<div style="font-size:12px;color:rgba(255,255,255,0.70);">Powered by Claude · Gemini · Groq</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        _ai_desc_c = '#7A6248' if st.session_state.get('theme','light')=='light' else '#94A3B8'
        st.markdown(
            f'<p style="font-size:13px;color:{_ai_desc_c};margin-bottom:20px;">'
            'Get a real AI-written summary of your chat — what the group talks about, '
            'the vibe, key dynamics, and more.</p>',
            unsafe_allow_html=True,
        )

        # ── Provider cards ────────────────────────────────────────────────────
        _choose_c = '#9C7A52' if st.session_state.get('theme','light')=='light' else '#64748B'
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;letter-spacing:0.15em;'
            f'text-transform:uppercase;color:{_choose_c};margin-bottom:10px;">Choose AI Provider</div>',
            unsafe_allow_html=True,
        )
        p_cols = st.columns(3)
        _is_lt_p  = st.session_state.get('theme','light') == 'light'
        provider_info = [
            ("anthropic", "🤖", "Anthropic Claude", "Best quality",   "#7C3AED", False),
            ("gemini",    "✨", "Google Gemini",    "Free tier ✅",    "#059669", True),
            ("groq",      "⚡", "Groq Llama 3",     "Free & fast ✅",  "#D97706", True),
        ]
        for col, (pid, icon, name, tag, color, is_free) in zip(p_cols, provider_info):
            with col:
                # ── Light: warm gold card  |  Dark: dark navy card ──
                if _is_lt_p:
                    card_bg   = '#FFFBF2'
                    card_bdr  = f'rgba(184,136,58,0.35)'
                    card_bl   = '#B8883A'
                    name_c    = '#18120A'
                    tag_c     = '#7A6248'
                    key_c     = '#B8883A'
                    free_span = (
                        '<span style="background:rgba(10,143,107,0.12);color:#0A8F6B;'
                        'font-size:9px;padding:2px 7px;border-radius:20px;'
                        'font-weight:700;margin-left:6px;">FREE</span>'
                        if is_free else ''
                    )
                else:
                    card_bg   = 'rgba(15,20,35,0.85)'
                    card_bdr  = f'{color}55'
                    card_bl   = color
                    name_c    = '#F1F5F9'
                    tag_c     = '#94A3B8'
                    key_c     = color
                    free_span = (
                        f'<span style="background:{color}18;color:{color};'
                        'font-size:9px;padding:2px 8px;border-radius:20px;'
                        'font-weight:700;margin-left:6px;">FREE</span>'
                        if is_free else ''
                    )

                st.markdown(
                    f'<a href="{PROVIDERS[pid]["signup_url"]}" target="_blank" style="text-decoration:none;">'
                    f'<div style="background:{card_bg};border:1px solid {card_bdr};'
                    f'border-left:3px solid {card_bl};border-radius:14px;padding:16px 16px;'
                    f'transition:all .2s;cursor:pointer;">'
                    f'<div style="font-size:22px;margin-bottom:8px;">{icon}</div>'
                    f'<div style="font-size:13px;font-weight:700;color:{name_c};'
                    f'margin-bottom:3px;">{name}{free_span}</div>'
                    f'<div style="font-size:11px;color:{tag_c};margin-bottom:6px;">{tag}</div>'
                    f'<div style="font-size:9px;color:{key_c};font-family:monospace;'
                    f'letter-spacing:.02em;">{PROVIDERS[pid]["key_example"]}</div>'
                    f'</div></a>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── API Key input ─────────────────────────────────────────────────────
        with st.expander("🔑 Enter API Key", expanded=("ai_api_key" not in st.session_state)):
            _exp_tc = '#7A6248' if st.session_state.get('theme','light')=='light' else '#94A3B8'
            st.markdown(
                f'<div style="font-size:12px;color:{_exp_tc};margin-bottom:10px;">'
                'Paste your key below — provider is auto-detected from key prefix<br>'
                '<span style="color:#7C3AED;font-family:monospace;">sk-ant-</span> · '
                '<span style="color:#059669;font-family:monospace;">AIza</span> · '
                '<span style="color:#D97706;font-family:monospace;">gsk_</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            api_key_input = st.text_input(
                "API Key", type="password",
                placeholder="Paste your API key here (sk-ant- / AIza / gsk_)...",
                label_visibility="collapsed",
                key="api_key_field",
            )
            # Show detected provider live
            if api_key_input:
                detected = detect_provider(api_key_input)
                if detected != "unknown":
                    prov = PROVIDERS[detected]
                    st.markdown(
                        f'<div style="background:rgba(0,200,150,0.08);border:1px solid rgba(0,200,150,0.25);'
                        f'border-radius:8px;padding:8px 12px;font-size:12px;color:#34D399;margin-top:6px;">'
                        f'✅ Detected: <strong>{prov["name"]}</strong></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div style="background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.25);'
                        'border-radius:8px;padding:8px 12px;font-size:12px;color:#F87171;margin-top:6px;">'
                        '⚠️ Unknown key format — must start with sk-ant-, AIza, or gsk_</div>',
                        unsafe_allow_html=True,
                    )

            c_save, c_clear = st.columns(2)
            with c_save:
                if st.button("💾 Save Key", use_container_width=True, key="save_api_key"):
                    if api_key_input and detect_provider(api_key_input) != "unknown":
                        st.session_state.ai_api_key = api_key_input.strip()
                        st.success("✅ Key saved!")
                    else:
                        st.error("Invalid key format.")
            with c_clear:
                if st.button("🗑️ Clear", use_container_width=True, key="clear_api_key"):
                    st.session_state.pop("ai_api_key", None)
                    st.info("Key cleared.")

        # ── Generate ──────────────────────────────────────────────────────────
        has_key = bool(st.session_state.get("ai_api_key"))

        if has_key:
            active_key      = st.session_state.ai_api_key
            active_provider = detect_provider(active_key)
            prov_info       = PROVIDERS.get(active_provider, {})
            badge_color     = prov_info.get("badge", "#00C896")
            prov_name       = prov_info.get("name", "AI")
            st.markdown(
                f'<div style="background:rgba(0,200,150,0.06);border:1px solid rgba(0,200,150,0.20);'
                f'border-radius:8px;padding:8px 14px;font-size:12px;color:#34D399;margin-bottom:12px;'
                f'display:inline-block;">✅ Ready · {prov_name}</div>',
                unsafe_allow_html=True,
            )

        # ── Auto Summary (no key needed) ─────────────────────────────────
        is_lt_ai = st.session_state.get('theme','light') == 'light'
        auto_bg  = 'rgba(184,136,58,0.07)' if is_lt_ai else 'rgba(0,200,150,0.06)'
        auto_bdr = 'rgba(184,136,58,0.25)' if is_lt_ai else 'rgba(0,200,150,0.20)'
        auto_tc  = '#7A6248' if is_lt_ai else '#94A3B8'
        auto_ac  = '#B8883A' if is_lt_ai else '#00C896'
        st.markdown(
            f'<div style="background:{auto_bg};border:1px solid {auto_bdr};'
            f'border-radius:12px;padding:14px 18px;margin-bottom:16px;">'
            f'<div style="font-size:13px;font-weight:700;color:{auto_ac};margin-bottom:4px;">'
            f'⚡ Instant Auto-Summary — No API Key Needed</div>'
            f'<div style="font-size:12px;color:{auto_tc};">'
            f'Get a smart data-driven summary instantly — no account, no key, no wait.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        auto_col, ai_col = st.columns(2)
        with auto_col:
            if st.button("⚡ Auto Summary (Free)", use_container_width=True, key="gen_auto_summary"):
                with st.spinner("⚡ Generating instant summary..."):
                    result = generate_auto_summary(df_filtered)
                    st.session_state.ai_summary_result = result

        with ai_col:
            if st.button(
                "✨ AI Summary" + (" (API Key Required)" if not has_key else ""),
                use_container_width=True,
                disabled=not has_key,
                help="Add API key first" if not has_key else "Analyze with AI",
                key="gen_ai_summary",
            ):
                with st.spinner("🤖 AI is reading your chat..."):
                    result = generate_ai_summary(df_filtered, st.session_state.ai_api_key)
                    st.session_state.ai_summary_result = result

        # ── Result ────────────────────────────────────────────────────────────
        result = st.session_state.get("ai_summary_result")

        if result:
            if result["success"]:
                prov     = result.get("provider", "")
                if prov == "auto":
                    p_color = "#B8883A" if st.session_state.get('theme','light')=='light' else "#F59E0B"
                    p_name  = "⚡ Auto Summary"
                else:
                    pinfo   = PROVIDERS.get(prov, {})
                    p_color = pinfo.get("badge", "#00C896")
                    p_name  = pinfo.get("name", "AI")

                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin:24px 0 16px;">'
                    f'<div style="width:34px;height:34px;border-radius:50%;'
                    f'background:linear-gradient(135deg,{p_color},{p_color}88);'
                    f'display:flex;align-items:center;justify-content:center;font-size:16px;">🤖</div>'
                    f'<div style="font-size:14px;font-weight:600;color:{_tc("#18120A","#E2E8F0")};">{p_name} Analysis</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Render in styled card
                _result_bg = '#FFFBF2' if st.session_state.get('theme','light')=='light' else 'rgba(17,24,39,0.9)'
                st.markdown(
                    f'<div style="background:{_result_bg};'
                    f'border:1px solid {p_color}44;border-left:4px solid {p_color};'
                    f'border-radius:16px;padding:28px 32px;margin-bottom:20px;">',
                    unsafe_allow_html=True,
                )
                st.markdown(result["summary"])
                st.markdown("</div>", unsafe_allow_html=True)

                # KPI bar
                stats = result.get("stats", {})
                if stats:
                    k1, k2, k3, k4 = st.columns(4)
                    with k1: render_kpi_card("Messages",     f"{stats.get('total_messages',0):,}", "Total",     "💬", 0)
                    with k2: render_kpi_card("Participants", str(stats.get('participants',0)),      "People",    "👥", 1)
                    with k3: render_kpi_card("Duration",     f"{stats.get('days',0)}d",             "Span",      "📅", 2)
                    with k4: render_kpi_card("Positive",     f"{stats.get('positive_pct',0)}%",     "Sentiment", "😊", 3)

                if st.button("🔄 Regenerate", key="regen_summary"):
                    st.session_state.pop("ai_summary_result", None)
                    st.rerun()

            else:
                # Friendly error with provider-specific help
                err_msg  = result.get("error", "Unknown error")
                provider = result.get("provider", "")
                help_tip = ""
                if "credit" in err_msg.lower() or "balance" in err_msg.lower():
                    help_tip = "💡 Tip: Your Anthropic credits are exhausted. Try <b>Google Gemini</b> or <b>Groq</b> — both are free!"
                elif "quota" in err_msg.lower():
                    help_tip = "💡 Tip: API quota exceeded. Try switching to another provider."
                elif "401" in err_msg or "403" in err_msg:
                    help_tip = "💡 Tip: Invalid or expired API key. Generate a new one from the provider console."

                _is_lt    = st.session_state.get('theme','light') == 'light'
                _err_tc   = '#3E2F1C' if _is_lt else '#CBD5E1'
                _tip_c    = '#7A6248' if _is_lt else '#94A3B8'
                _tip_html = f'<div style="font-size:12px;color:{_tip_c}">{help_tip}</div>' if help_tip else ''
                st.markdown(
                    f'<div style="background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.30);'
                    f'border-left:4px solid #F87171;border-radius:14px;padding:20px 24px;margin-top:16px;">'
                    f'<div style="color:#F87171;font-weight:700;font-size:14px;margin-bottom:8px;">❌ Error</div>'
                    f'<div style="color:{_err_tc};font-size:13px;font-family:monospace;'
                    f'background:rgba(0,0,0,0.3);padding:10px;border-radius:8px;margin-bottom:12px;">'
                    f'{err_msg}</div>'
                    f'{_tip_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                # Show free alternatives
                if provider == "anthropic":
                    st.markdown(
                        f'<div style="font-size:13px;font-weight:600;color:{_tc("#18120A","#E2E8F0")};margin-bottom:10px;">'
                        '🆓 Free Alternatives — Get a key and paste above:</div>',
                        unsafe_allow_html=True,
                    )
                    alt_cols = st.columns(2)
                    alternatives = [
                        ("✨", "Google Gemini", "FREE forever", "#059669",
                         "https://aistudio.google.com/app/apikey", "AIzaSy..."),
                        ("⚡", "Groq Llama 3", "FREE + blazing fast", "#D97706",
                         "https://console.groq.com", "gsk_..."),
                    ]
                    for col, (icon, name, tag, color, url, key_fmt) in zip(alt_cols, alternatives):
                        with col:
                            _alt_bg = "#FFF8ED" if st.session_state.get('theme','light')=='light' else 'rgba(17,24,39,0.8)'
                            st.markdown(
                                f'<a href="{url}" target="_blank" style="text-decoration:none;">'
                                f'<div style="background:{_alt_bg};border:1px solid {color}44;'
                                f'border-left:3px solid {color};border-radius:12px;padding:14px;'
                                f'cursor:pointer;">'
                                f'<div style="font-size:22px;margin-bottom:6px;">{icon}</div>'
                                f'<div style="font-size:13px;font-weight:700;color:{_tc("#18120A","#E2E8F0")};">{name}</div>'
                                f'<div style="font-size:11px;color:#34D399;margin-top:2px;">{tag}</div>'
                                f'<div style="font-size:10px;color:{color};margin-top:6px;font-family:monospace;">'
                                f'Key starts with: {key_fmt}</div>'
                                f'<div style="font-size:10px;color:#64748B;margin-top:4px;">↗ Click to get free key</div>'
                                f'</div></a>',
                                unsafe_allow_html=True,
                            )

                if st.button("🔄 Try Again", key="retry_ai_summary"):
                    st.session_state.pop("ai_summary_result", None)
                    st.rerun()

        elif not result:
            # Empty state
            st.markdown(
                '<div style="text-align:center;padding:52px 20px;'
                'background:rgba(255,248,237,0.85) !important;'
                'border:1px dashed rgba(0,160,100,0.25);border-radius:20px;margin-top:24px;">'
                '<div style="font-size:48px;margin-bottom:14px;">🤖</div>'
                f'<div style="font-size:16px;font-weight:700;color:{"#18120A" if st.session_state.get("theme","light")=="light" else "#E2E8F0"};margin-bottom:8px;">'
                'Ready to Analyze</div>'
                '<div style="font-size:13px;color:#475569;line-height:1.6;">'
                'Choose a provider above, paste your API key,<br>'
                'and click <b style="color:#00C896;">Generate AI Summary</b></div>'
                '<div style="margin-top:16px;display:flex;justify-content:center;gap:12px;">'
                '<span style="font-size:11px;background:rgba(5,150,105,0.12);color:#34D399;'
                'padding:4px 10px;border-radius:20px;">✨ Gemini Free</span>'
                '<span style="font-size:11px;background:rgba(217,119,6,0.12);color:#FCD34D;'
                'padding:4px 10px;border-radius:20px;">⚡ Groq Free</span>'
                '<span style="font-size:11px;background:rgba(124,58,237,0.12);color:#A78BFA;'
                'padding:4px 10px;border-radius:20px;">🤖 Claude Premium</span>'
                '</div></div>',
                unsafe_allow_html=True,
            )

    # ── Tab 7: Deep Analysis ─────────────────────────────────────────────────
    with tab7:
        st.markdown("""
        <div style="margin-bottom:6px;">
            <span style="font-family:'Outfit',sans-serif;font-size:9px;font-weight:700;
                letter-spacing:0.20em;text-transform:uppercase;color:#00C896;">
                🔬 6 Deep Analysis Features
            </span>
        </div>
        """, unsafe_allow_html=True)

        _deep_opts = [
            "🕸️ Network Graph","⏱️ Response Time","☁️ Word Cloud",
            "🧠 Personality Profiles","📅 Monthly Recap",
            "👻 Ghost Members","😂 Emoji Analytics","🔥 Streaks",
            "💬 Reply Chains","💑 Best Friends","🦉 Night Owls",
            "🗑️ Deleted Msgs","📊 Message Flow",
            "🎭 Personality Quiz","🔥 Roast Mode","📄 PDF Report",
        ]
        deep_tab = st.selectbox(
            "Select Analysis",
            _deep_opts,
            label_visibility="collapsed",
            key="deep_tab_sel",
        )

        st.markdown("<hr style='border:none;border-top:1px solid rgba(0,200,150,0.15);margin:10px 0 20px;'>",
                    unsafe_allow_html=True)

        # ── 1. Network Graph ──────────────────────────────────────────────
        if deep_tab == "🕸️ Network Graph":
            st.markdown("### 🕸️ Conversation Network")
            st.markdown(
                f'<p style="color:{_sc()};font-size:13px;margin-bottom:16px;">'
                'Who replies to whom? Node size = message count. '
                'Line thickness = interaction frequency.</p>',
                unsafe_allow_html=True,
            )
            with st.spinner("Building network..."):
                fig_net = build_network_graph(df_filtered)
            st.plotly_chart(fig_net, use_container_width=True)

            # Top pairs
            st.markdown("#### 🔗 Top Interaction Pairs")
            df_s  = df_filtered.sort_values("datetime").reset_index(drop=True)
            from collections import defaultdict
            pairs = defaultdict(int)
            for i in range(1, len(df_s)):
                u1 = df_s.loc[i-1, "user"]; u2 = df_s.loc[i, "user"]
                dt = (df_s.loc[i, "datetime"] - df_s.loc[i-1, "datetime"]).total_seconds()
                if u1 != u2 and dt < 300:
                    pairs[tuple(sorted([u1, u2]))] += 1
            if pairs:
                top_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:5]
                for (u1, u2), cnt in top_pairs:
                    pct = round(cnt / sum(pairs.values()) * 100, 1)
                    st.markdown(
                        f'<div style="background:rgba(0,200,150,0.07);border:1px solid rgba(0,200,150,0.18);'
                        f'border-radius:10px;padding:10px 16px;margin-bottom:8px;display:flex;'
                        f'justify-content:space-between;align-items:center;">'
                        f'<span style="color:{_tc("#18120A","#E2E8F0")};font-size:14px;">👤 {u1} &nbsp;↔️&nbsp; {u2}</span>'
                        f'<span style="color:#00C896;font-family:monospace;font-weight:700;">'
                        f'{cnt} exchanges ({pct}%)</span></div>',
                        unsafe_allow_html=True,
                    )

        # ── 2. Response Time ──────────────────────────────────────────────
        elif deep_tab == "⏱️ Response Time":
            st.markdown("### ⏱️ Response Time Analysis")
            st.markdown(
                f'<p style="color:{_sc()};font-size:13px;margin-bottom:16px;">'
                'How quickly does each person respond? Only counting replies within 2 hours.</p>',
                unsafe_allow_html=True,
            )
            with st.spinner("Calculating response times..."):
                fig_avg, fig_dist, stats_df = response_time_analysis(df_filtered)

            if not stats_df.empty:
                fastest = stats_df["Avg Response (min)"].idxmin()
                slowest = stats_df["Avg Response (min)"].idxmax()
                c1, c2, c3 = st.columns(3)
                with c1:
                    render_kpi_card("Fastest", fastest,
                                    f"{stats_df.loc[fastest,'Avg Response (min)']}m avg", "⚡", 0)
                with c2:
                    render_kpi_card("Slowest", slowest,
                                    f"{stats_df.loc[slowest,'Avg Response (min)']}m avg", "🐢", 1)
                with c3:
                    median_all = round(stats_df["Median (min)"].mean(), 1)
                    render_kpi_card("Group Median", f"{median_all}m",
                                    "typical reply time", "📊", 2)
                st.plotly_chart(fig_avg, use_container_width=True)
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.plotly_chart(fig_dist, use_container_width=True)
                with c2:
                    st.markdown("#### 📋 Full Stats")
                    st.dataframe(stats_df, use_container_width=True)

        # ── 3. Word Cloud ─────────────────────────────────────────────────
        elif deep_tab == "☁️ Word Cloud":
            st.markdown("### ☁️ Word Cloud — Top Words per User")
            users_list = ["Everyone"] + sorted(df_filtered["user"].unique().tolist())
            selected = st.selectbox("Select User", users_list, key="wc_user")
            user_arg = None if selected == "Everyone" else selected
            with st.spinner("Generating word cloud..."):
                fig_wc = word_cloud_treemap(df_filtered, user_arg, top_n=40)
            st.plotly_chart(fig_wc, use_container_width=True)

            # Show top 10 as list too
            top_tok = _safe_tokens(df_filtered, user_arg, 10)
            if top_tok:
                st.markdown("#### 🏆 Top 10 Words")
                cols = st.columns(5)
                for i, (word, cnt) in enumerate(top_tok):
                    with cols[i % 5]:
                        st.markdown(
                            f'<div style="text-align:center;background:rgba(0,200,150,0.08);'
                            f'border:1px solid rgba(0,200,150,0.2);border-radius:8px;padding:8px 4px;">'
                            f'<div style="font-size:14px;font-weight:700;color:#00C896;">{word}</div>'
                            f'<div style="font-size:11px;color:{_sc()};">{cnt}x</div></div>',
                            unsafe_allow_html=True,
                        )

        # ── 4. Personality Profiles ───────────────────────────────────────
        elif deep_tab == "🧠 Personality Profiles":
            st.markdown("### 🧠 Personality Profiles")
            st.markdown(
                f'<p style="color:{_sc()};font-size:13px;margin-bottom:16px;">'
                'Each user scored on 6 communication dimensions.</p>',
                unsafe_allow_html=True,
            )
            with st.spinner("Building profiles..."):
                fig_radar, profile_df = personality_profiles(df_filtered)

            st.plotly_chart(fig_radar, use_container_width=True)

            if not profile_df.empty:
                st.markdown("#### 🎭 Personality Labels")
                users_p = profile_df.index.tolist()
                cols = st.columns(min(len(users_p), 3))
                for i, u in enumerate(users_p):
                    row_scores = profile_df.loc[u].to_dict()
                    label = personality_label(row_scores)
                    emoji = label.split()[0]
                    desc  = ' '.join(label.split()[1:])
                    with cols[i % len(cols)]:
                        st.markdown(
                            f'<div style="background:{"#FFFFFF" if st.session_state.get("theme","light")=="light" else "rgba(17,24,39,0.7)"};border:1px solid rgba(0,160,100,0.22);'
                            f'border-radius:12px;padding:14px 16px;margin-bottom:10px;">'
                            f'<div style="font-size:20px;margin-bottom:6px;">{emoji}</div>'
                            f'<div style="font-size:13px;font-weight:700;color:#E2E8F0;margin-bottom:4px;">{u}</div>'
                            f'<div style="font-size:11px;color:#64748B;">{desc}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("#### 📊 Scores Table")
                st.dataframe(profile_df, use_container_width=True)

        # ── 5. Monthly Recap ──────────────────────────────────────────────
        elif deep_tab == "📅 Monthly Recap":
            st.markdown("### 📅 Monthly & Weekly Recap")

            period_choice = st.radio("View by", ["Monthly", "Weekly"], horizontal=True, key="recap_period")
            period_code   = "M" if period_choice == "Monthly" else "W"

            with st.spinner("Building recap..."):
                fig_recap = monthly_recap(df_filtered, period_code)
                fig_card  = recap_stat_card(df_filtered)

            st.plotly_chart(fig_card,  use_container_width=True)
            st.plotly_chart(fig_recap, use_container_width=True)

            # Most active month breakdown
            df_tmp = df_filtered.copy()
            df_tmp["period"] = df_tmp["datetime"].dt.to_period(period_code).astype(str)
            period_agg = df_tmp.groupby("period").agg(
                Messages=("message","count"),
                Users=("user","nunique"),
                Positive_Pct=("sentiment_vader", lambda x: round((x=="POSITIVE").sum()/max(len(x),1)*100,1)),
            ).sort_values("Messages", ascending=False)
            st.markdown(f"#### 📋 {period_choice} Breakdown")
            st.dataframe(period_agg, use_container_width=True)

        # ── 6. Ghost Members ─────────────────────────────────────────────
        elif deep_tab == "👻 Ghost Members":
            st.markdown("### 👻 Ghost Members & Participation Tiers")
            st.markdown(
                f'<p style="color:{_sc()};font-size:13px;margin-bottom:16px;">'
                'Who dominates the chat vs who barely speaks?</p>',
                unsafe_allow_html=True,
            )
            with st.spinner("Analyzing participation..."):
                fig_ghost, ghost_df, tier_df = ghost_members_analysis(df_filtered)

            # Tier legend
            tier_info = [
                ("🦁 Alpha",  "#F87171", "≥15% of messages"),
                ("🐯 Active", "#FBBF24", "5–15%"),
                ("🐦 Casual", "#60A5FA", "1–5%"),
                ("👻 Ghost",  "#818CF8", "0.1–1%"),
                ("🕳️ Shadow", "#475569", "<0.1%"),
            ]
            t_cols = st.columns(5)
            for col, (tier, color, desc) in zip(t_cols, tier_info):
                with col:
                    st.markdown(
                        f'<div style="text-align:center;background:{"#FFF8ED" if st.session_state.get("theme","light")=="light" else "rgba(17,24,39,0.6)"};'
                        f'border:1px solid {color}44;border-radius:10px;padding:10px 8px;">'
                        f'<div style="font-size:16px;">{tier.split()[0]}</div>'
                        f'<div style="font-size:11px;font-weight:600;color:{color};">{tier}</div>'
                        f'<div style="font-size:10px;color:#64748B;margin-top:2px;">{desc}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown("<br>", unsafe_allow_html=True)
            st.plotly_chart(fig_ghost, use_container_width=True)

            if not ghost_df.empty:
                st.markdown("#### 👻 Ghost Members (< 1% participation)")
                for _, row in ghost_df.iterrows():
                    st.markdown(
                        f'<div style="background:rgba(129,140,248,0.06);border:1px solid rgba(129,140,248,0.15);'
                        f'border-radius:10px;padding:10px 16px;margin-bottom:6px;display:flex;'
                        f'justify-content:space-between;align-items:center;">'
                        f'<span style="color:{_tc("#18120A","#E2E8F0")};font-size:13px;">👻 {row["User"]}</span>'
                        f'<span style="color:#818CF8;font-family:monospace;">'
                        f'{row["Messages"]} msgs ({row["Share %"]:.2f}%)</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.success("🎉 No ghost members — everyone participates meaningfully!")

            st.markdown("#### 📊 Full Participation Table")
            st.dataframe(
                tier_df[['User','Messages','Share %','Tier']].sort_values('Messages', ascending=False),
                use_container_width=True,
            )

        # ── 7. Emoji Analytics ────────────────────────────────────────────
        elif deep_tab == "😂 Emoji Analytics":
            st.markdown("### 😂 Emoji Analytics")
            st.markdown(
                f'<p style="color:{_sc()};font-size:13px;margin-bottom:16px;">'
                'Which emojis dominate? Who uses them most?</p>',
                unsafe_allow_html=True,
            )
            with st.spinner("Analyzing emojis..."):
                fig_top_em, fig_user_em, em_stats = emoji_analytics(df_filtered)

            if em_stats:
                e1, e2, e3, e4 = st.columns(4)
                with e1: render_kpi_card("Total Emojis",   str(em_stats.get('total_emojis',0)),  "Used",        "😂", 0)
                with e2: render_kpi_card("Unique",         str(em_stats.get('unique_emojis',0)), "Different",   "🎭", 1)
                with e3: render_kpi_card("Most Used",      em_stats.get('most_used','—'),         f"{em_stats.get('most_used_count',0)}x", "🏆", 2)
                with e4: render_kpi_card("Emoji King/Queen", em_stats.get('top_emoji_user','—'), "Most emojis", "👑", 3)
                st.markdown("<br>", unsafe_allow_html=True)

            st.plotly_chart(fig_top_em,  use_container_width=True)
            st.plotly_chart(fig_user_em, use_container_width=True)

        # ── 8. Streaks & Silence ──────────────────────────────────────────
        elif deep_tab == "🔥 Streaks":
            st.markdown("### 🔥 Activity Streaks & Silent Periods")
            st.markdown(
                f'<p style="color:{_sc()};font-size:13px;margin-bottom:16px;">'
                'GitHub-style calendar + longest active streaks and silent gaps.</p>',
                unsafe_allow_html=True,
            )
            with st.spinner("Computing streaks..."):
                fig_cal, fig_sil, streak_stats = streak_analysis(df_filtered)

            s1, s2, s3, s4 = st.columns(4)
            with s1: render_kpi_card("Longest Streak",  f"{streak_stats.get('longest_streak_days',0)}d", "Active days", "🔥", 0)
            with s2: render_kpi_card("Longest Silence", f"{streak_stats.get('longest_silence_days',0)}d", "No messages", "😴", 1)
            with s3: render_kpi_card("Active Days",     str(streak_stats.get('total_active_days',0)), "With messages", "📅", 2)
            with s4: render_kpi_card("Activity Rate",   f"{streak_stats.get('activity_rate',0)}%", "Days active", "📊", 3)

            st.markdown("<br>", unsafe_allow_html=True)
            st.plotly_chart(fig_cal, use_container_width=True)
            st.plotly_chart(fig_sil, use_container_width=True)

            if streak_stats.get('longest_streak_days', 0) > 0:
                st.markdown(
                    f'<div style="background:rgba(0,200,150,0.07);border:1px solid rgba(0,200,150,0.20);'
                    f'border-radius:12px;padding:16px 20px;margin-top:8px;">'
                    f'<div style="font-size:13px;color:{_tc("#18120A","#E2E8F0")};">'
                    f'🔥 <b>Best Streak:</b> {streak_stats.get("longest_streak_days")} consecutive days '
                    f'({streak_stats.get("streak_start")} → {streak_stats.get("streak_end")})</div>'
                    f'<div style="font-size:13px;color:{_tc("#18120A","#E2E8F0")};margin-top:6px;">'
                    f'😴 <b>Longest Silence:</b> {streak_stats.get("longest_silence_days")} days '
                    f'({streak_stats.get("silence_start")} → {streak_stats.get("silence_end")})</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── 9. PDF Report ─────────────────────────────────────────────────
        elif deep_tab == "📄 PDF Report":
            st.markdown("### 📄 PDF Report Export")
            st.markdown(
                f'<p style="color:{_sc()};font-size:13px;margin-bottom:20px;">'
                'Generate a professional multi-page PDF with full analysis, '
                'user stats, sentiment breakdown, and key insights.</p>',
                unsafe_allow_html=True,
            )

            # Preview what will be in report
            c1, c2 = st.columns(2)
            sections = [
                ("📊", "Cover Page", "Title, date range, key stats overview"),
                ("📝", "Summary", "Conversation overview & narrative"),
                ("💡", "Key Insights", "AI-generated bullet points"),
                ("👥", "User Stats", "Per-user message count, sentiment, length"),
                ("🎯", "Top Topics", "Most discussed words & themes"),
                ("😊", "Mood Analysis", "Positive/Neutral/Negative breakdown"),
            ]
            for i, (icon, title, desc) in enumerate(sections):
                with (c1 if i % 2 == 0 else c2):
                    st.markdown(
                        f'<div style="background:{"#FFFFFF" if st.session_state.get("theme","light")=="light" else "rgba(17,24,39,0.6)"};border:1px solid rgba(0,160,100,0.18);'
                        f'border-radius:10px;padding:12px 14px;margin-bottom:8px;">'
                        f'<div style="font-size:18px;margin-bottom:4px;">{icon}</div>'
                        f'<div style="font-size:13px;font-weight:600;color:#E2E8F0;">{title}</div>'
                        f'<div style="font-size:11px;color:#64748B;margin-top:2px;">{desc}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📥 Generate & Download PDF", use_container_width=True, key="gen_pdf"):
                with st.spinner("📄 Building your PDF report..."):
                    try:
                        from analytics.summary_generator import SummaryGenerator
                        sg  = SummaryGenerator(df_filtered)
                        smr = sg.generate_summary()
                        pdf_bytes = generate_pdf_report(df_filtered, smr)
                        if pdf_bytes:
                            st.download_button(
                                label="⬇️ Download PDF Report",
                                data=pdf_bytes,
                                file_name=f"whatsapp_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key="dl_pdf",
                            )
                            st.success("✅ PDF ready! Click the button above to download.")
                        else:
                            st.error("PDF generation failed. Make sure reportlab is installed: pip install reportlab")
                    except Exception as e:
                        st.error(f"Error: {e}")


        # ── 9. Reply Chains ───────────────────────────────────────────────
        elif deep_tab == "💬 Reply Chains":
            st.markdown("### 💬 Reply Chain Analysis")
            st.markdown(f'<p style="color:{_sc()};font-size:13px;margin-bottom:16px;">Who replies to whom most? Who starts conversations?</p>', unsafe_allow_html=True)
            with st.spinner("Analyzing reply patterns..."):
                try:
                    fig_reply, fig_start, stats_df = reply_chain_analysis(df_filtered)
                    c1, c2 = st.columns(2)
                    with c1:
                        if fig_reply.data: st.plotly_chart(fig_reply, use_container_width=True)
                    with c2:
                        st.plotly_chart(fig_start, use_container_width=True)
                    if len(stats_df):
                        st.dataframe(stats_df.reset_index(drop=True), use_container_width=True)
                except Exception as e:
                    st.error(f"Reply chain analysis failed: {e}")

        # ── 10. Best Friends ──────────────────────────────────────────────
        elif deep_tab == "💑 Best Friends":
            st.markdown("### 💑 Best Friends — Most Interacting Pairs")
            st.markdown(f'<p style="color:{_sc()};font-size:13px;margin-bottom:16px;">Based on back-and-forth messages within 10 minutes.</p>', unsafe_allow_html=True)
            with st.spinner("Finding best friends..."):
                try:
                    fig_bf, df_bf = best_friends_analysis(df_filtered)
                    if fig_bf.data:
                        st.plotly_chart(fig_bf, use_container_width=True)
                    if len(df_bf):
                        _is_lt = st.session_state.get('theme','light')=='light'
                        bg = '#FFFBF2' if _is_lt else 'rgba(17,24,39,0.75)'
                        tc = '#18120A' if _is_lt else '#E2E8F0'
                        sc2 = '#3E2F1C' if _is_lt else '#CBD5E1'
                        top3 = df_bf.head(3)
                        medals = ['🥇','🥈','🥉']
                        cols3 = st.columns(3)
                        for i, (_, row) in enumerate(top3.iterrows()):
                            with cols3[i]:
                                st.markdown(
                                    f'<div style="background:{bg};border:1px solid rgba(0,200,150,0.25);'
                                    f'border-radius:14px;padding:18px;text-align:center;">'
                                    f'<div style="font-size:32px;margin-bottom:8px;">{medals[i]}</div>'
                                    f'<div style="font-size:13px;font-weight:700;color:{tc};">{row["Person 1"]}</div>'
                                    f'<div style="font-size:11px;color:#00C896;margin:4px 0;">↔️ {row["Interactions"]} interactions</div>'
                                    f'<div style="font-size:13px;font-weight:700;color:{tc};">{row["Person 2"]}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.dataframe(df_bf, use_container_width=True)
                except Exception as e:
                    st.error(f"Best friends analysis failed: {e}")

        # ── 11. Night Owls ────────────────────────────────────────────────
        elif deep_tab == "🦉 Night Owls":
            st.markdown("### 🦉 Night Owl vs Early Bird Leaderboard")
            st.markdown(f'<p style="color:{_sc()};font-size:13px;margin-bottom:16px;">When does each person prefer to chat?</p>', unsafe_allow_html=True)
            with st.spinner("Analyzing activity patterns..."):
                try:
                    fig_owl, df_owl = night_owl_analysis(df_filtered)
                    st.plotly_chart(fig_owl, use_container_width=True)
                    _is_lt = st.session_state.get('theme','light')=='light'
                    bg = '#FFFBF2' if _is_lt else 'rgba(17,24,39,0.75)'
                    tc = '#18120A' if _is_lt else '#E2E8F0'
                    sc2 = '#3E2F1C' if _is_lt else '#CBD5E1'
                    cols_o = st.columns(min(4, len(df_owl)))
                    for i, (_, row) in enumerate(df_owl.iterrows()):
                        if i >= 8: break
                        with cols_o[i % len(cols_o)]:
                            emoji = '🦉' if row['Night %'] > 25 else ('🌅' if row['Early %'] > 20 else '☀️')
                            st.markdown(
                                f'<div style="background:{bg};border:1px solid rgba(139,92,246,0.25);'
                                f'border-radius:12px;padding:14px;text-align:center;margin-bottom:8px;">'
                                f'<div style="font-size:24px;">{emoji}</div>'
                                f'<div style="font-size:12px;font-weight:700;color:{tc};margin:4px 0;">{row["User"]}</div>'
                                f'<div style="font-size:10px;color:#8B5CF6;">{row["Type"]}</div>'
                                f'<div style="font-size:10px;color:{sc2};margin-top:4px;">🌙 {row["Night %"]}% · ☀️ {row["Day %"]}%</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                except Exception as e:
                    st.error(f"Night owl analysis failed: {e}")

        # ── 12. Deleted Messages ──────────────────────────────────────────
        elif deep_tab == "🗑️ Deleted Msgs":
            st.markdown("### 🗑️ Deleted & Media Omitted Messages")
            st.markdown(f'<p style="color:{_sc()};font-size:13px;margin-bottom:16px;">Who deletes messages? What media was shared but not saved?</p>', unsafe_allow_html=True)
            with st.spinner("Scanning for deleted messages..."):
                try:
                    res = deleted_messages_analysis(df_filtered)
                    _is_lt = st.session_state.get('theme','light')=='light'
                    bg = '#FFFBF2' if _is_lt else 'rgba(17,24,39,0.75)'
                    tc = '#18120A' if _is_lt else '#E2E8F0'
                    sc2 = '#3E2F1C' if _is_lt else '#CBD5E1'
                    kc1, kc2 = st.columns(2)
                    with kc1:
                        st.markdown(
                            f'<div style="background:{bg};border:1px solid rgba(248,113,113,0.3);'
                            f'border-radius:14px;padding:20px;text-align:center;">'
                            f'<div style="font-size:36px;font-weight:800;color:#F87171;">{res["total_deleted"]}</div>'
                            f'<div style="font-size:12px;color:{sc2};">🗑️ Deleted Messages</div>'
                            f'</div>', unsafe_allow_html=True,
                        )
                    with kc2:
                        st.markdown(
                            f'<div style="background:{bg};border:1px solid rgba(96,165,250,0.3);'
                            f'border-radius:14px;padding:20px;text-align:center;">'
                            f'<div style="font-size:36px;font-weight:800;color:#60A5FA;">{res["total_omitted"]}</div>'
                            f'<div style="font-size:12px;color:{sc2};">📎 Media Not Saved</div>'
                            f'</div>', unsafe_allow_html=True,
                        )
                    st.markdown("<br>", unsafe_allow_html=True)
                    if res["fig"].data:
                        st.plotly_chart(res["fig"], use_container_width=True)
                    else:
                        st.info("No deleted or omitted messages found in this chat export.")
                except Exception as e:
                    st.error(f"Deleted message analysis failed: {e}")

        # ── 13. Message Flow ──────────────────────────────────────────────
        elif deep_tab == "📊 Message Flow":
            st.markdown("### 📊 Conversation Flow by Hour")
            st.markdown(f'<p style="color:{_sc()};font-size:13px;margin-bottom:16px;">How does activity peak across the day for each person?</p>', unsafe_allow_html=True)
            with st.spinner("Building flow chart..."):
                try:
                    fig_flow = conversation_flow_analysis(df_filtered)
                    st.plotly_chart(fig_flow, use_container_width=True)
                    # Also show hourly heatmap
                    df_h = df_filtered.copy()
                    df_h['hour'] = df_h['datetime'].dt.hour
                    df_h['weekday'] = df_h['datetime'].dt.day_name()
                    hm = df_h.groupby(['weekday','hour']).size().reset_index(name='count')
                    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
                    hm['weekday'] = pd.Categorical(hm['weekday'], categories=day_order, ordered=True)
                    hm = hm.sort_values('weekday')
                    pivot = hm.pivot(index='weekday', columns='hour', values='count').fillna(0)
                    fig_hm = go.Figure(go.Heatmap(
                        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
                        colorscale=[[0,'#07090F'],[0.5,'rgba(0,200,150,0.4)'],[1,'#00C896']],
                        hovertemplate='%{y} %{x}:00 — %{z} messages<extra></extra>',
                    ))
                    fig_hm.update_layout(
                        title='📅 Weekly Activity Heatmap',
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#94A3B8', height=320,
                        margin=dict(l=20,r=20,t=40,b=20),
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
                except Exception as e:
                    st.error(f"Flow analysis failed: {e}")

        # ── 14. Personality Quiz ──────────────────────────────────────────
        elif deep_tab == "🎭 Personality Quiz":
            st.markdown("### 🎭 Chat Personality Quiz")
            st.markdown(f'<p style="color:{_sc()};font-size:13px;margin-bottom:16px;">Which type of group member is each person? Based on messaging patterns.</p>', unsafe_allow_html=True)
            with st.spinner("Analyzing personalities..."):
                try:
                    results = personality_quiz(df_filtered)
                    _is_lt = st.session_state.get('theme','light')=='light'
                    render_personality_cards(results, _is_lt)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f'<p style="color:{_sc()};font-size:11px;">Personality types are based on message length, emoji usage, sentiment, timing, and participation rate.</p>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Personality quiz failed: {e}")

        # ── 15. Roast Mode ────────────────────────────────────────────────
        elif deep_tab == "🔥 Roast Mode":
            st.markdown("### 🔥 Roast Mode")
            _is_lt_r = st.session_state.get('theme','light')=='light'
            bg_r = '#FFFBF2' if _is_lt_r else 'rgba(17,24,39,0.75)'
            tc_r = '#18120A' if _is_lt_r else '#E2E8F0'
            st.markdown(
                f'<div style="background:{bg_r};border:1px solid rgba(249,115,22,0.3);'
                f'border-left:4px solid #F97316;border-radius:14px;padding:14px 18px;margin-bottom:20px;">'
                f'<div style="font-size:13px;font-weight:700;color:#F97316;margin-bottom:4px;">⚠️ All in good fun!</div>'
                f'<div style="font-size:12px;color:{tc_r};">These roasts are auto-generated from data patterns — playful observations, not personal attacks. Share with the group for laughs! 😄</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.spinner("Preparing roasts... 🔥"):
                try:
                    roasts = roast_generator(df_filtered)
                    render_roast_cards(roasts, _is_lt_r)
                except Exception as e:
                    st.error(f"Roast generation failed: {e}")


    # ── Tab 5: Advanced Analytics ────────────────────────────────────────────
    with tab6:
        st.header("Advanced Analytics")
        viz = AdvancedVisualizations(df_filtered)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Sentiment Timeline (Animated)")
            st.plotly_chart(viz.sentiment_timeline_animated(), use_container_width=True)
        with col2:
            st.subheader("🌊 Emotion Transitions")
            st.plotly_chart(viz.emotion_transition_graph(), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏆 Positivity Leaderboard")
            st.plotly_chart(viz.user_positivity_leaderboard(), use_container_width=True)
        with col2:
            st.subheader("☢️ Toxicity Heatmap")
            st.plotly_chart(viz.toxicity_heatmap(), use_container_width=True)

        st.subheader("📅 Activity Calendar")
        st.plotly_chart(viz.activity_calendar_heatmap(), use_container_width=True)

        st.subheader("🔤 Word Clouds by User")
        selected_user = st.selectbox("Select User", df_filtered["user"].unique())
        word_cloud = viz.word_cloud_per_user(selected_user)
        if word_cloud:
            st.plotly_chart(word_cloud, use_container_width=True)

    # ── Tab 7: Leaderboard ───────────────────────────────────────────────────
    with tab8:
        # ══════════════════════════════════════════════════════════════════
        #  🏆 LEADERBOARD TAB
        # ══════════════════════════════════════════════════════════════════
        _is_lt_lb = st.session_state.get('theme','light') == 'light'
        bg_lb   = '#FFFBF2' if _is_lt_lb else 'rgba(17,24,39,0.75)'
        bdr_lb  = 'rgba(184,136,58,0.25)' if _is_lt_lb else 'rgba(0,200,150,0.15)'
        tc_lb   = '#18120A' if _is_lt_lb else '#E2E8F0'
        sc_lb   = '#3E2F1C' if _is_lt_lb else '#CBD5E1'
        ac_lb   = '#B8883A' if _is_lt_lb else '#00C896'
        ac2_lb  = '#8B6820' if _is_lt_lb else '#8B5CF6'

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:24px;">
            <div style="font-size:42px;">🏆</div>
            <div>
                <div style="font-size:1.6rem;font-weight:800;color:{tc_lb};
                    font-family:'Cormorant Garamond',serif;">Group Leaderboard</div>
                <div style="font-size:13px;color:{sc_lb};">Who's the MVP of this group chat?</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Build leaderboard data ─────────────────────────────────────────
        mc_lb = 'message_cleaned' if 'message_cleaned' in df_filtered.columns else 'message'
        df_lb = df_filtered.copy()
        df_lb['hour'] = df_lb['datetime'].dt.hour
        df_lb['msg_len'] = df_lb[mc_lb].fillna('').str.len()
        df_lb['has_url'] = df_lb[mc_lb].fillna('').str.contains(r'https?://', regex=True)
        import re as _re_lb
        df_lb['emoji_count'] = df_lb[mc_lb].fillna('').apply(
            lambda x: len(_re_lb.findall(r'[🌀-🿿]', x))
        )
        df_lb['is_night'] = df_lb['hour'].isin(list(range(22,24)) + list(range(0,4)))
        df_lb['is_early'] = df_lb['hour'].isin(range(5,9))

        per_user = df_lb.groupby('user').agg(
            total_msgs      =('user','count'),
            avg_len         =('msg_len','mean'),
            total_emojis    =('emoji_count','sum'),
            links_shared    =('has_url','sum'),
            night_msgs      =('is_night','sum'),
            early_msgs      =('is_early','sum'),
            pos_msgs        =('sentiment_vader', lambda x: (x=='POSITIVE').sum()),
            neg_msgs        =('sentiment_vader', lambda x: (x=='NEGATIVE').sum()),
        ).round(1)
        per_user['positivity_pct'] = (per_user['pos_msgs'] / per_user['total_msgs'] * 100).round(1)
        per_user['negativity_pct'] = (per_user['neg_msgs'] / per_user['total_msgs'] * 100).round(1)
        per_user = per_user.sort_values('total_msgs', ascending=False)

        medals = ['🥇','🥈','🥉']

        # ── CATEGORY 1: Most Messages ──────────────────────────────────────
        st.markdown(f'<div style="font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ac_lb};margin:20px 0 12px;">📨 Most Messages</div>', unsafe_allow_html=True)
        top3_msgs = per_user['total_msgs'].nlargest(3)
        c1,c2,c3 = st.columns(3)
        for col,(user,val),(med) in zip([c1,c2,c3], top3_msgs.items(), medals):
            pct = round(val/max(len(df_filtered),1)*100,1)
            with col:
                st.markdown(
                    f'<div style="background:{bg_lb};border:1px solid {bdr_lb};'
                    f'border-radius:16px;padding:20px;text-align:center;">'
                    f'<div style="font-size:32px;">{med}</div>'
                    f'<div style="font-size:14px;font-weight:700;color:{tc_lb};margin:8px 0 4px;">{user}</div>'
                    f'<div style="font-size:26px;font-weight:800;color:{ac_lb};">{int(val):,}</div>'
                    f'<div style="font-size:11px;color:{sc_lb};">messages · {pct}% of chat</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── CATEGORY ROWS ──────────────────────────────────────────────────
        categories = [
            ("💬", "Longest Messages",   per_user['avg_len'].nlargest(3),       "avg chars",  ac_lb),
            ("😂", "Emoji Champion",     per_user['total_emojis'].nlargest(3),   "emojis",     "#F59E0B"),
            ("🔗", "Link Sharer",        per_user['links_shared'].nlargest(3),   "links",      "#0EA5E9"),
            ("🌙", "Night Owl",          per_user['night_msgs'].nlargest(3),     "night msgs", "#8B5CF6"),
            ("🌅", "Early Bird",         per_user['early_msgs'].nlargest(3),     "morning",    "#10B981"),
            ("😊", "Most Positive",      per_user['positivity_pct'].nlargest(3), "% positive", "#34D399"),
            ("😤", "Most Salty",         per_user['negativity_pct'].nlargest(3), "% negative", "#F87171"),
        ]

        for row_idx in range(0, len(categories), 2):
            cols_cat = st.columns(2)
            for cat_idx, col in enumerate(cols_cat):
                if row_idx + cat_idx >= len(categories): break
                emoji, title, data, unit, color = categories[row_idx + cat_idx]
                with col:
                    rows_html = ''
                    for rank, (user, val) in enumerate(data.items()):
                        med   = medals[rank] if rank < 3 else f"#{rank+1}"
                        val_d = f"{val:.1f}" if isinstance(val, float) else str(int(val))
                        rows_html += (
                            f'<div style="display:flex;align-items:center;gap:10px;'
                            f'padding:8px 0;border-bottom:1px solid {color}18;">'
                            f'<span style="font-size:18px;width:28px;text-align:center;">{med}</span>'
                            f'<span style="flex:1;font-size:13px;font-weight:600;color:{tc_lb};">{user}</span>'
                            f'<span style="font-size:13px;color:{color};font-weight:700;">{val_d}</span>'
                            f'<span style="font-size:10px;color:{sc_lb};margin-left:2px;">{unit}</span>'
                            f'</div>'
                        )
                    st.markdown(
                        f'<div style="background:{bg_lb};border:1px solid {color}33;'
                        f'border-top:3px solid {color};border-radius:14px;'
                        f'padding:16px 18px;margin-bottom:14px;">'
                        f'<div style="font-size:13px;font-weight:700;color:{color};margin-bottom:10px;">'
                        f'{emoji} {title}</div>'
                        f'{rows_html}</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── FULL STATS TABLE ───────────────────────────────────────────────
        st.markdown(f'<div style="font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ac_lb};margin-bottom:12px;">📊 Complete Stats Table</div>', unsafe_allow_html=True)
        display_df = per_user.reset_index()[['user','total_msgs','avg_len','total_emojis',
                                              'links_shared','positivity_pct','negativity_pct',
                                              'night_msgs','early_msgs']].rename(columns={
            'user':'User','total_msgs':'Messages','avg_len':'Avg Length',
            'total_emojis':'Emojis','links_shared':'Links',
            'positivity_pct':'Positive %','negativity_pct':'Negative %',
            'night_msgs':'Night Msgs','early_msgs':'Early Msgs',
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ── CSV DOWNLOAD FROM LEADERBOARD ─────────────────────────────────
        import io as _io_lb
        csv_lb = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Leaderboard CSV",
            data=csv_lb,
            file_name=f"leaderboard_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_leaderboard_csv",
        )


    # ── Tab 8: Export ────────────────────────────────────────────────────────
    with tab9:
        _is_lt_ex = st.session_state.get('theme','light') == 'light'
        bg_ex  = '#FFFBF2' if _is_lt_ex else 'rgba(17,24,39,0.75)'
        bdr_ex = 'rgba(184,136,58,0.25)' if _is_lt_ex else 'rgba(0,200,150,0.15)'
        tc_ex  = '#18120A' if _is_lt_ex else '#E2E8F0'
        sc_ex  = '#3E2F1C' if _is_lt_ex else '#CBD5E1'
        ac_ex  = '#B8883A' if _is_lt_ex else '#00C896'

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:24px;">
            <div style="font-size:42px;">💾</div>
            <div>
                <div style="font-size:1.6rem;font-weight:800;color:{tc_ex};
                    font-family:'Cormorant Garamond',serif;">Export & Download</div>
                <div style="font-size:13px;color:{sc_ex};">Download your analysis in multiple formats</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Section 1: CSV Downloads ───────────────────────────────────────
        st.markdown(f'<div style="font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ac_ex};margin-bottom:14px;">📥 CSV Downloads</div>', unsafe_allow_html=True)

        import io as _io_ex

        csv_configs = [
            {
                "label": "📋 Full Chat Data",
                "desc":  "All messages with sentiment, emotion, timestamps",
                "key":   "dl_full_csv",
                "fname": "full_chat",
                "cols":  ["datetime","user","message_cleaned","sentiment_vader",
                          "sentiment_compound","emotion","is_toxic","message_length"],
            },
            {
                "label": "👥 User Statistics",
                "desc":  "Per-user message count, sentiment, activity",
                "key":   "dl_user_csv",
                "fname": "user_stats",
                "cols":  None,  # computed below
            },
            {
                "label": "📅 Daily Activity",
                "desc":  "Messages per day + average sentiment",
                "key":   "dl_daily_csv",
                "fname": "daily_activity",
                "cols":  None,
            },
            {
                "label": "😊 Sentiment Only",
                "desc":  "Timestamp, user, message, sentiment label",
                "key":   "dl_sent_csv",
                "fname": "sentiment_data",
                "cols":  ["datetime","user","message_cleaned","sentiment_vader","sentiment_compound"],
            },
        ]

        # Pre-compute special tables
        mc_ex = 'message_cleaned' if 'message_cleaned' in df_filtered.columns else 'message'
        user_stats_df = df_filtered.groupby('user').agg(
            Messages=('user','count'),
            Avg_Sentiment=('sentiment_compound','mean'),
            Positive_pct=('sentiment_vader', lambda x:(x=='POSITIVE').sum()/len(x)*100),
            Negative_pct=('sentiment_vader', lambda x:(x=='NEGATIVE').sum()/len(x)*100),
            Avg_Msg_Length=('message_length','mean'),
            Total_Emojis=(mc_ex, lambda x: x.fillna('').str.count(r'[\U0001F300-\U0001FFFF]').sum()),
        ).round(2).reset_index()

        daily_df = df_filtered.copy()
        daily_df['date'] = daily_df['datetime'].dt.date
        daily_stats = daily_df.groupby('date').agg(
            Messages=('user','count'),
            Avg_Sentiment=('sentiment_compound','mean'),
            Unique_Users=('user','nunique'),
        ).round(3).reset_index()

        special_tables = {
            "dl_user_csv":  user_stats_df,
            "dl_daily_csv": daily_stats,
        }

        c1e, c2e = st.columns(2)
        for i, cfg in enumerate(csv_configs):
            with (c1e if i % 2 == 0 else c2e):
                if cfg["cols"]:
                    avail_cols = [c for c in cfg["cols"] if c in df_filtered.columns]
                    data_df    = df_filtered[avail_cols].copy()
                else:
                    data_df    = special_tables.get(cfg["key"], df_filtered.head(10))

                csv_bytes = data_df.to_csv(index=False).encode('utf-8')
                fname     = f'{cfg["fname"]}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M")}.csv'

                st.markdown(
                    f'<div style="background:{bg_ex};border:1px solid {bdr_ex};'
                    f'border-radius:12px;padding:14px 16px;margin-bottom:4px;">'
                    f'<div style="font-size:13px;font-weight:700;color:{tc_ex};">{cfg["label"]}</div>'
                    f'<div style="font-size:11px;color:{sc_ex};margin-top:3px;">{cfg["desc"]}</div>'
                    f'<div style="font-size:10px;color:{ac_ex};margin-top:4px;font-family:monospace;">'
                    f'{len(data_df):,} rows</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    label=f"⬇️ Download",
                    data=csv_bytes,
                    file_name=fname,
                    mime="text/csv",
                    use_container_width=True,
                    key=cfg["key"],
                )
                st.markdown("<br>", unsafe_allow_html=True)

        # ── Section 2: PDF Report ──────────────────────────────────────────
        st.markdown(f'<div style="font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ac_ex};margin:10px 0 14px;">📄 PDF Report</div>', unsafe_allow_html=True)
        summary_gen = SummaryGenerator(df_filtered)
        summary     = summary_gen.generate_summary()
        if st.button("📥 Generate & Download PDF", use_container_width=True, key="gen_pdf_export"):
            with st.spinner("📄 Building PDF report..."):
                try:
                    from analytics.deep_analysis import generate_pdf_report
                    pdf_bytes = generate_pdf_report(df_filtered, summary)
                    if pdf_bytes:
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"whatsapp_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="dl_pdf_export",
                        )
                        st.success("✅ PDF ready!")
                    else:
                        st.error("PDF generation failed. Run: pip install reportlab")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Section 3: Quick Preview ───────────────────────────────────────
        st.markdown(f'<div style="font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ac_ex};margin-bottom:12px;">👁️ Quick Data Preview</div>', unsafe_allow_html=True)
        preview_cols = [c for c in ["datetime","user","message_cleaned","sentiment_vader","emotion","is_toxic"] if c in df_filtered.columns]
        st.dataframe(df_filtered[preview_cols].head(20), use_container_width=True, hide_index=True)

        # Advanced export
        try:
            AdvancedExportSystem.render_export_buttons(df_filtered, summary)
        except Exception:
            pass

    # ── Tab 9: Settings ──────────────────────────────────────────────────────
    with tab10:
        st.header("Settings & Configuration")
        st.subheader("📊 Dashboard Settings")

        col1, col2 = st.columns(2)
        with col1:
            current = st.session_state.get("theme", "Light")
            st.info(f"Current: {current} Theme  (toggle via sidebar ☀️ / 🌙)")
        with col2:
            chart_style = st.selectbox("Chart Style", ["Plotly Dark", "Plotly White", "Ggplot"])
            st.info(f"Current: {chart_style}")

        st.divider()
        st.subheader("🔧 Processing Options")
        use_transformer = st.checkbox("Use Transformer Models (More Accurate, Slower)")
        st.caption("Recommended for offline analysis. Disabled for faster results.")
        multilingual_mode = st.checkbox("Enable Multilingual Analysis")
        st.caption("Supports English, Hindi, Hinglish")

        st.divider()
        st.subheader("📈 Data Management")
        if st.button("Clear All Data", use_container_width=True):
            st.session_state.clear()
            st.success("Data cleared!")

        st.divider()
        st.subheader("ℹ️ About")
        _is_lt_ab = st.session_state.get('theme','light') == 'light'
        bg_ab  = '#FFFBF2' if _is_lt_ab else 'rgba(17,24,39,0.75)'
        tc_ab  = '#18120A' if _is_lt_ab else '#E2E8F0'
        sc_ab  = '#3E2F1C' if _is_lt_ab else '#CBD5E1'
        ac_ab  = '#B8883A' if _is_lt_ab else '#00C896'
        st.markdown(
            f'''<div style="background:{bg_ab};border:1px solid rgba(0,0,0,0.08);
            border-radius:14px;padding:20px 24px;">
            <div style="font-size:15px;font-weight:700;color:{tc_ab};margin-bottom:10px;">
                📊 WhatsApp Sentiment Analysis Dashboard v2.0
            </div>
            <div style="font-size:12px;color:{sc_ab};line-height:1.9;">
                💬 Chat Explorer with media detection (YouTube, links, images)<br>
                😊 Sentiment Analysis — VADER + optional DistilBERT<br>
                🎭 Emotion Detection — Joy, Anger, Sadness, Fear, Surprise<br>
                🕸️ Network Graph — conversation topology<br>
                💑 Best Friends — most interacting pairs<br>
                🦉 Night Owl vs Early Bird leaderboard<br>
                🗑️ Deleted messages & media omitted tracking<br>
                🎭 Personality Quiz — 10 member archetypes<br>
                🔥 Roast Mode — fun data-driven observations<br>
                ⚡ Auto Summary — no API key needed<br>
                🤖 AI Summary — Claude / Gemini / Groq<br>
                📄 PDF export + CSV/Excel download<br>
                🌙 Light & Dark themes — fully responsive
            </div>
            <div style="margin-top:12px;font-size:11px;color:{ac_ab};">
                Built with Streamlit · Pandas · Plotly · Transformers
            </div></div>''',
            unsafe_allow_html=True,
        )

else:
    # ── Beautiful onboarding screen ───────────────────────────────────────
    _is_lt_ob = st.session_state.get('theme','light') == 'light'
    hero_bg   = 'linear-gradient(135deg,#FFF8ED 0%,#FFF3DC 50%,#FFF8ED 100%)' if _is_lt_ob else 'linear-gradient(135deg,#07090F 0%,#0D1320 50%,#07090F 100%)'
    card_bg   = '#FFFFFF' if _is_lt_ob else 'rgba(17,24,39,0.75)'
    card_bdr  = 'rgba(184,136,58,0.25)' if _is_lt_ob else 'rgba(0,200,150,0.15)'
    tc        = '#18120A' if _is_lt_ob else '#E2E8F0'
    sc2       = '#7A6248' if _is_lt_ob else '#94A3B8'
    ac        = '#B8883A' if _is_lt_ob else '#00C896'
    ac2       = '#8B6820' if _is_lt_ob else '#8B5CF6'

    st.markdown(f"""
    <div style="background:{hero_bg};border-radius:24px;padding:56px 40px;
        text-align:center;margin:20px 0 40px;">
        <div style="font-size:64px;margin-bottom:16px;">📊</div>
        <div style="font-size:2.4rem;font-weight:800;color:{tc};
            font-family:'Cormorant Garamond',serif;margin-bottom:10px;line-height:1.2;">
            WhatsApp Sentiment Analysis
        </div>
        <div style="font-size:1rem;color:{sc2};max-width:520px;margin:0 auto 28px;line-height:1.7;">
            Uncover deep insights from your chats — sentiment, emotions,
            personality types, reply patterns, and much more.
        </div>
        <div style="display:inline-flex;gap:10px;flex-wrap:wrap;justify-content:center;">
            <span style="background:{ac}22;color:{ac};border:1px solid {ac}44;
                padding:6px 16px;border-radius:20px;font-size:12px;font-weight:600;">
                ⚡ Instant Analysis
            </span>
            <span style="background:{ac2}22;color:{ac2};border:1px solid {ac2}44;
                padding:6px 16px;border-radius:20px;font-size:12px;font-weight:600;">
                🔒 100% Private
            </span>
            <span style="background:rgba(16,185,129,0.12);color:#10B981;border:1px solid rgba(16,185,129,0.3);
                padding:6px 16px;border-radius:20px;font-size:12px;font-weight:600;">
                🤖 AI Powered
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature grid
    features = [
        ("💬","Chat Explorer",    "Browse messages with media preview, links, and YouTube cards"),
        ("😊","Sentiment AI",     "VADER + optional transformer — positive, negative, neutral"),
        ("🎭","Emotions",         "Joy, anger, sadness, fear — per message emotion detection"),
        ("🕸️","Network Graph",   "See who talks to whom most in the group"),
        ("💑","Best Friends",     "Find the most interacting pairs automatically"),
        ("🦉","Night Owls",       "Night owl vs early bird leaderboard"),
        ("🎭","Personality Quiz", "Which type of group member is each person?"),
        ("🔥","Roast Mode",       "Fun auto-generated roasts based on chat patterns 😄"),
        ("⚡","Auto Summary",     "AI-like chat summary — no API key needed"),
        ("🤖","AI Summary",       "Real AI summary via Claude, Gemini, or Groq"),
        ("📄","PDF Export",       "Download a professional full report"),
        ("🌙","Dark & Light",     "Beautiful themes — warm cream or deep navy"),
    ]

    cols_f = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols_f[i % 3]:
            st.markdown(
                f'<div style="background:{card_bg};border:1px solid {card_bdr};'
                f'border-radius:14px;padding:18px 16px;margin-bottom:12px;height:110px;">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                f'<span style="font-size:22px;">{icon}</span>'
                f'<span style="font-size:13px;font-weight:700;color:{tc};">{title}</span>'
                f'</div>'
                f'<div style="font-size:11px;color:{sc2};line-height:1.55;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # How to export guide
    st.markdown(f"""
    <div style="background:{card_bg};border:1px solid {card_bdr};border-left:4px solid {ac};
        border-radius:14px;padding:20px 24px;margin-top:10px;">
        <div style="font-size:14px;font-weight:700;color:{ac};margin-bottom:12px;">
            📱 How to Export WhatsApp Chat
        </div>
        <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div style="flex:1;min-width:200px;">
                <div style="font-size:12px;font-weight:600;color:{tc};margin-bottom:6px;">Android</div>
                <div style="font-size:11px;color:{sc2};line-height:1.8;">
                    Open chat → ⋮ Menu → More → Export Chat → Without Media → Share .txt file
                </div>
            </div>
            <div style="flex:1;min-width:200px;">
                <div style="font-size:12px;font-weight:600;color:{tc};margin-bottom:6px;">iPhone</div>
                <div style="font-size:11px;color:{sc2};line-height:1.8;">
                    Open chat → Contact/Group Name → Export Chat → Without Media → Share .txt file
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)