"""
model_manager.py — Proper VADER / Transformer / Multilingual / Hybrid comparison engine
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from typing import Dict, Tuple, Optional, List, Any

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
    """Compare and execute VADER, Transformer, Multilingual-BERT, and Hybrid sentiment models."""

    MODEL_INFO = {
        'VADER (WhatsApp-Tuned)': {
            'desc':  'Rule-based + WhatsApp and Hinglish lexicons. Emoji-aware. Fast, lightweight CPU inference.',
            'speed': '⚡ Very Fast (~1ms)',
            'best':  'Large chats, Hinglish, casual conversation',
            'color': '#18A3B7',
            'icon':  '⚡',
        },
        'Transformer (DistilBERT)': {
            'desc':  'DistilBERT fine-tuned on SST-2 English sentiment. Deep semantic & contextual understanding.',
            'speed': '🤖 Deep Context (~30ms)',
            'best':  'English text, nuanced sentence structures',
            'color': '#818CF8',
            'icon':  '🤖',
        },
        'Multilingual BERT': {
            'desc':  'nlptown/bert-base-multilingual-uncased-sentiment — handles Hindi, Urdu, Arabic, English, and more.',
            'speed': '🌐 Multilingual (~60ms)',
            'best':  'Multilingual & non-English chats',
            'color': '#F472B6',
            'icon':  '🌐',
        },
        'Hybrid': {
            'desc':  'VADER fast path + Contextual Transformer refinement on uncertain or ambiguous messages.',
            'speed': '🚀 Balanced & Accurate',
            'best':  'Production analytics — speed + contextual precision',
            'color': '#FBBF24',
            'icon':  '🔀',
        },
    }

    def __init__(self):
        self.results: Dict = {}

    @st.cache_resource(show_spinner=False)
    def _get_vader(_self) -> Tuple[SentimentAnalyzer, Optional[str]]:
        """Instantiate rule-based VADER analyzer."""
        try:
            return SentimentAnalyzer(), None
        except Exception as e:
            return None, f"VADER initialization error: {type(e).__name__}: {e}"

    @st.cache_resource(show_spinner=False)
    def _get_transformer(_self) -> Tuple[Any, Optional[str]]:
        """Load DistilBERT SST-2 sentiment pipeline."""
        try:
            from transformers import pipeline
            pipe = pipeline(
                'sentiment-analysis',
                model='distilbert-base-uncased-finetuned-sst-2-english',
                truncation=True,
                max_length=512,
            )
            return pipe, None
        except Exception as e:
            return None, f"Transformer (DistilBERT) initialization error: {type(e).__name__}: {e}"

    @st.cache_resource(show_spinner=False)
    def _get_multilingual(_self) -> Tuple[Any, Optional[str]]:
        """Load Multilingual BERT sentiment pipeline."""
        try:
            from transformers import pipeline
            pipe = pipeline(
                'sentiment-analysis',
                model='nlptown/bert-base-multilingual-uncased-sentiment',
                truncation=True,
                max_length=512,
            )
            return pipe, None
        except Exception as e:
            return None, f"Multilingual BERT initialization error: {type(e).__name__}: {e}"

    def get_models_status(self) -> Dict[str, Dict]:
        """Check backend health and environment readiness for all models without heavy inference."""
        status = {}
        status['VADER (WhatsApp-Tuned)'] = {
            'status': 'Ready',
            'available': True,
            'desc': 'In-Memory Rule-Based + Hinglish Lexicon',
            'icon': '✅',
            'badge': 'Ready (Fast CPU)',
        }

        try:
            import torch
            torch_ver = torch.__version__
        except ImportError:
            torch_ver = None

        try:
            import transformers
            trans_ver = transformers.__version__
        except ImportError:
            trans_ver = None

        if not torch_ver:
            err = "PyTorch (torch) is not installed in the current environment."
            status['Transformer (DistilBERT)'] = {'status': 'Unavailable', 'available': False, 'desc': err, 'icon': '❌', 'badge': 'Dependency Missing'}
            status['Multilingual BERT'] = {'status': 'Unavailable', 'available': False, 'desc': err, 'icon': '❌', 'badge': 'Dependency Missing'}
        elif not trans_ver:
            err = "HuggingFace transformers is not installed in the current environment."
            status['Transformer (DistilBERT)'] = {'status': 'Unavailable', 'available': False, 'desc': err, 'icon': '❌', 'badge': 'Dependency Missing'}
            status['Multilingual BERT'] = {'status': 'Unavailable', 'available': False, 'desc': err, 'icon': '❌', 'badge': 'Dependency Missing'}
        else:
            status['Transformer (DistilBERT)'] = {
                'status': 'Ready',
                'available': True,
                'desc': f'DistilBERT SST-2 (PyTorch {torch_ver} · CPU)',
                'icon': '✅',
                'badge': 'Ready (Deep Context)',
            }
            status['Multilingual BERT'] = {
                'status': 'Ready',
                'available': True,
                'desc': f'nlptown BERT Multilingual (PyTorch {torch_ver} · CPU)',
                'icon': '✅',
                'badge': 'Ready (Multilingual)',
            }

        status['Hybrid'] = {
            'status': 'Ready' if (torch_ver and trans_ver) else 'VADER-Only',
            'available': True,
            'desc': 'VADER Fast Path + Contextual Transformer on Ambiguous Messages',
            'icon': '✅',
            'badge': 'Ready (Auto-Refining)',
        }
        return status

    @staticmethod
    def detect_sarcasm_and_context(text: str, base_compound: float, base_label: str) -> Tuple[float, str, Optional[str]]:
        """
        Detects sarcasm, ironic emojis, and contrastive conjunctions.
        """
        t_lower = text.lower().strip()
        sarcastic_emojis = {'🙄', '😒', '🙃', '😏', '🤨'}
        has_sarcastic_emoji = any(e in text for e in sarcastic_emojis)

        # 1. Emoji sarcasm contrast: Positive words with sarcastic emoji
        if has_sarcastic_emoji and base_compound > 0.0:
            adjusted_compound = -round(min(0.85, abs(base_compound) + 0.25), 4)
            return adjusted_compound, 'NEGATIVE', 'Sarcasm detected (positive wording inverted by sarcastic emoji 🙄/😒)'

        # 2. Contrastive conjunction: "..., but ...", "..., however ...", "... lekin ..."
        contrast_markers = [' but ', ', but', ' however ', ' lekin ', ', lekin ', ' par ']
        for marker in contrast_markers:
            if marker in t_lower:
                parts = t_lower.split(marker, 1)
                second_clause = parts[1].strip()
                crit_words = ['missed', 'wrong', 'fail', 'bad', 'poor', 'useless', 'late', 'point', 'galat', 'bekar', 'not']
                if any(w in second_clause for w in crit_words):
                    return -0.45, 'NEGATIVE', 'Contrastive context: positive opening negated by critical subsequent clause'

        # 3. Backhanded / patronizing idioms
        if 'at least you tried' in t_lower or 'atleast you tried' in t_lower:
            return -0.35, 'NEGATIVE', 'Backhanded idiom detected ("at least you tried")'
        if "this is a first" in t_lower and ("not late" in t_lower or "on time" in t_lower):
            return 0.15, 'NEUTRAL', 'Mild sarcasm / backhanded compliment ("this is a first")'

        return base_compound, base_label, None

    def analyze_single_message(self, text: str, model: str, allow_fallback: bool = False) -> Dict:
        """
        Analyze a single message with the explicitly selected model.
        Returns a rich metrics dictionary with confidence, probabilities, and context info.
        Does NOT silently fall back to VADER unless allow_fallback=True is passed.
        """
        t0 = time.time()
        clean_text = text.strip()
        if not clean_text:
            return {
                'model': model,
                'status': 'Ready',
                'label': 'NEUTRAL',
                'compound': 0.0,
                'score': 0.0,
                'confidence': 50.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'processing_time': 0.0,
                'sarcasm': None,
                'hybrid_source': None,
                'fallback': None,
                'error': None,
            }

        fallback_notice = None
        hybrid_source = None
        error_msg = None
        model_status = "Loaded"

        if model == 'VADER (WhatsApp-Tuned)':
            analyzer, _ = self._get_vader()
            v_res = analyzer.analyze_vader(clean_text)
            compound, label, sarcasm = self.detect_sarcasm_and_context(clean_text, v_res['compound'], v_res['label'])
            pos = v_res['positive']
            neg = v_res['negative']
            neu = v_res['neutral']
            if sarcasm and label == 'NEGATIVE':
                neg = max(neg, 0.65)
                pos = min(pos, 0.15)
                neu = max(0.1, round(1.0 - (pos + neg), 3))
            conf = min(99.0, max(52.0, round(abs(compound) * 60 + 40, 1)))

        elif model == 'Transformer (DistilBERT)':
            pipe, err = self._get_transformer()
            if pipe is None:
                error_msg = err or "Transformer (DistilBERT) model unavailable."
                model_status = "Failed"
                if allow_fallback:
                    analyzer, _ = self._get_vader()
                    v_res = analyzer.analyze_vader(clean_text)
                    compound, label, sarcasm = self.detect_sarcasm_and_context(clean_text, v_res['compound'], v_res['label'])
                    pos, neg, neu = v_res['positive'], v_res['negative'], v_res['neutral']
                    conf = min(99.0, max(52.0, round(abs(compound) * 60 + 40, 1)))
                    fallback_notice = f"Transformer unavailable ({error_msg}) — fallback to VADER executed."
                else:
                    return {
                        'model': model,
                        'status': 'Failed',
                        'error': error_msg,
                        'label': 'ERROR',
                        'compound': 0.0,
                        'score': 0.0,
                        'confidence': 0.0,
                        'positive': 0.0,
                        'negative': 0.0,
                        'neutral': 0.0,
                        'processing_time': round(time.time() - t0, 4),
                        'sarcasm': None,
                        'hybrid_source': None,
                        'fallback': None,
                    }
            else:
                try:
                    pred = pipe(clean_text[:512])[0]
                    raw_lbl = pred['label'].upper()
                    raw_sc = float(pred['score'])
                    raw_comp = raw_sc if raw_lbl == 'POSITIVE' else -raw_sc
                    raw_base_lbl = 'POSITIVE' if raw_comp > 0.05 else ('NEGATIVE' if raw_comp < -0.05 else 'NEUTRAL')
                    compound, label, sarcasm = self.detect_sarcasm_and_context(clean_text, raw_comp, raw_base_lbl)
                    pos = round(raw_sc if label == 'POSITIVE' else (1.0 - raw_sc), 3)
                    neg = round(raw_sc if label == 'NEGATIVE' else (1.0 - raw_sc), 3)
                    neu = max(0.05, round(1.0 - (pos + neg), 3)) if (pos + neg) < 1.0 else 0.05
                    conf = round(raw_sc * 100, 1)
                except Exception as e:
                    error_msg = f"Inference error: {type(e).__name__}: {e}"
                    model_status = "Failed"
                    if allow_fallback:
                        analyzer, _ = self._get_vader()
                        v_res = analyzer.analyze_vader(clean_text)
                        compound, label, sarcasm = self.detect_sarcasm_and_context(clean_text, v_res['compound'], v_res['label'])
                        pos, neg, neu = v_res['positive'], v_res['negative'], v_res['neutral']
                        conf = min(99.0, max(52.0, round(abs(compound) * 60 + 40, 1)))
                        fallback_notice = f"Transformer error ({error_msg}) — fallback to VADER executed."
                    else:
                        return {
                            'model': model,
                            'status': 'Failed',
                            'error': error_msg,
                            'label': 'ERROR',
                            'compound': 0.0,
                            'score': 0.0,
                            'confidence': 0.0,
                            'positive': 0.0,
                            'negative': 0.0,
                            'neutral': 0.0,
                            'processing_time': round(time.time() - t0, 4),
                            'sarcasm': None,
                            'hybrid_source': None,
                            'fallback': None,
                        }

        elif model == 'Multilingual BERT':
            pipe, err = self._get_multilingual()
            if pipe is None:
                error_msg = err or "Multilingual BERT model unavailable."
                model_status = "Failed"
                if allow_fallback:
                    analyzer, _ = self._get_vader()
                    v_res = analyzer.analyze_vader(clean_text)
                    compound, label, sarcasm = self.detect_sarcasm_and_context(clean_text, v_res['compound'], v_res['label'])
                    pos, neg, neu = v_res['positive'], v_res['negative'], v_res['neutral']
                    conf = min(99.0, max(52.0, round(abs(compound) * 60 + 40, 1)))
                    fallback_notice = f"Multilingual BERT unavailable ({error_msg}) — fallback to VADER executed."
                else:
                    return {
                        'model': model,
                        'status': 'Failed',
                        'error': error_msg,
                        'label': 'ERROR',
                        'compound': 0.0,
                        'score': 0.0,
                        'confidence': 0.0,
                        'positive': 0.0,
                        'negative': 0.0,
                        'neutral': 0.0,
                        'processing_time': round(time.time() - t0, 4),
                        'sarcasm': None,
                        'hybrid_source': None,
                        'fallback': None,
                    }
            else:
                try:
                    pred = pipe(clean_text[:512])[0]
                    raw_lbl = str(pred['label']).lower()
                    raw_sc = float(pred['score'])
                    if 'star' in raw_lbl:
                        stars = int(raw_lbl.split()[0])
                        raw_comp = round((stars - 3.0) / 2.0, 4)
                        raw_base_lbl = 'POSITIVE' if stars >= 4 else ('NEGATIVE' if stars <= 2 else 'NEUTRAL')
                        compound, label, sarcasm = self.detect_sarcasm_and_context(clean_text, raw_comp, raw_base_lbl)
                        pos = max(0.05, round(stars / 5.0 * raw_sc, 3))
                        neg = max(0.05, round((6 - stars) / 5.0 * raw_sc, 3))
                        neu = max(0.05, round(1.0 - (pos + neg), 3)) if (pos + neg) < 1.0 else 0.1
                        conf = round(raw_sc * 100, 1)
                    else:
                        is_pos = 'pos' in raw_lbl
                        raw_comp = raw_sc if is_pos else -raw_sc
                        raw_base_lbl = 'POSITIVE' if is_pos else 'NEGATIVE'
                        compound, label, sarcasm = self.detect_sarcasm_and_context(clean_text, raw_comp, raw_base_lbl)
                        pos = round(raw_sc if is_pos else (1.0 - raw_sc), 3)
                        neg = round(raw_sc if not is_pos else (1.0 - raw_sc), 3)
                        neu = max(0.05, round(1.0 - (pos + neg), 3)) if (pos + neg) < 1.0 else 0.05
                        conf = round(raw_sc * 100, 1)
                except Exception as e:
                    error_msg = f"Multilingual inference error: {type(e).__name__}: {e}"
                    model_status = "Failed"
                    if allow_fallback:
                        analyzer, _ = self._get_vader()
                        v_res = analyzer.analyze_vader(clean_text)
                        compound, label, sarcasm = self.detect_sarcasm_and_context(clean_text, v_res['compound'], v_res['label'])
                        pos, neg, neu = v_res['positive'], v_res['negative'], v_res['neutral']
                        conf = min(99.0, max(52.0, round(abs(compound) * 60 + 40, 1)))
                        fallback_notice = f"Multilingual BERT error ({error_msg}) — fallback to VADER executed."
                    else:
                        return {
                            'model': model,
                            'status': 'Failed',
                            'error': error_msg,
                            'label': 'ERROR',
                            'compound': 0.0,
                            'score': 0.0,
                            'confidence': 0.0,
                            'positive': 0.0,
                            'negative': 0.0,
                            'neutral': 0.0,
                            'processing_time': round(time.time() - t0, 4),
                            'sarcasm': None,
                            'hybrid_source': None,
                            'fallback': None,
                        }

        else:  # Hybrid
            analyzer, _ = self._get_vader()
            v_res = analyzer.analyze_vader(clean_text)
            compound, label, sarcasm = self.detect_sarcasm_and_context(clean_text, v_res['compound'], v_res['label'])
            pos, neg, neu = v_res['positive'], v_res['negative'], v_res['neutral']
            conf = min(99.0, max(52.0, round(abs(compound) * 60 + 40, 1)))

            # If uncertain/ambiguous in VADER and no emoji sarcasm, refine with deep Transformer
            if abs(compound) < 0.25 and not sarcasm:
                pipe, _ = self._get_transformer()
                if pipe is None:
                    pipe, _ = self._get_multilingual()

                if pipe is not None:
                    try:
                        pred = pipe(clean_text[:512])[0]
                        raw_lbl = str(pred['label']).lower()
                        sc = float(pred['score'])
                        if 'star' in raw_lbl:
                            stars = int(raw_lbl.split()[0])
                            compound = round((stars - 3.0) / 2.0, 4)
                            label = 'POSITIVE' if stars >= 4 else ('NEGATIVE' if stars <= 2 else 'NEUTRAL')
                            conf = round(sc * 100, 1)
                            pos = max(0.05, round(stars / 5.0 * sc, 3))
                            neg = max(0.05, round((6 - stars) / 5.0 * sc, 3))
                            neu = max(0.05, round(1.0 - (pos + neg), 3)) if (pos + neg) < 1.0 else 0.1
                        else:
                            is_pos = 'pos' in raw_lbl
                            compound = sc if is_pos else -sc
                            label = 'POSITIVE' if compound > 0.05 else ('NEGATIVE' if compound < -0.05 else 'NEUTRAL')
                            conf = round(sc * 100, 1)
                            pos = round(sc if is_pos else (1.0 - sc), 3)
                            neg = round(sc if not is_pos else (1.0 - sc), 3)
                            neu = max(0.05, round(1.0 - (pos + neg), 3)) if (pos + neg) < 1.0 else 0.05
                        hybrid_source = "Transformer (Contextual Refinement for Ambiguous Message)"
                    except Exception:
                        hybrid_source = "VADER (Standalone)"
                else:
                    hybrid_source = "VADER (Standalone — Transformer Unavailable)"
            else:
                hybrid_source = "VADER (High-Confidence Fast Path)"

        elapsed = time.time() - t0
        return {
            'model': model,
            'status': model_status,
            'error': error_msg,
            'label': label,
            'compound': round(compound, 4),
            'score': round(compound, 3),
            'confidence': conf,
            'positive': round(pos, 3),
            'negative': round(neg, 3),
            'neutral': round(neu, 3),
            'processing_time': round(elapsed, 4),
            'sarcasm': sarcasm,
            'hybrid_source': hybrid_source,
            'fallback': fallback_notice,
        }

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

    # ── Analysis runners for DataFrames ────────────────────────────────────────

    def analyze_with_model(self, df: pd.DataFrame, model: str) -> Tuple[pd.DataFrame, Dict]:
        t0 = time.time()
        df_res = df.copy()
        confidence = 'N/A'
        extra = {}

        if model == 'VADER (WhatsApp-Tuned)':
            analyzer, _ = self._get_vader()
            df_res = analyzer.analyze_dataframe(df_res, use_transformer=False)
            confidence = f"{df_res['sentiment_compound'].abs().mean():.3f}"

        elif model == 'Transformer (DistilBERT)':
            pipe, err = self._get_transformer()
            if pipe is None:
                raise RuntimeError(f"Transformer model unavailable: {err}")
            col = 'message_cleaned' if 'message_cleaned' in df_res.columns else 'message'
            def _t_analyze(text):
                try:
                    r = pipe(str(text)[:512])[0]
                    lbl = 'POSITIVE' if r['label'].upper() == 'POSITIVE' else 'NEGATIVE'
                    sc = float(r['score'])
                    comp = sc if lbl == 'POSITIVE' else -sc
                    return lbl, round(comp, 4)
                except Exception:
                    return 'NEUTRAL', 0.0
            results = df_res[col].apply(_t_analyze)
            df_res['sentiment_vader']    = results.apply(lambda x: x[0])
            df_res['sentiment_compound'] = results.apply(lambda x: x[1])
            confidence = f"{df_res['sentiment_compound'].abs().mean():.3f}"

        elif model == 'Multilingual BERT':
            pipe, err = self._get_multilingual()
            if pipe is None:
                raise RuntimeError(f"Multilingual BERT model unavailable: {err}")
            col = 'message_cleaned' if 'message_cleaned' in df_res.columns else 'message'
            def _m_analyze(text):
                try:
                    r = pipe(str(text)[:512])[0]
                    raw_lbl = str(r['label']).lower()
                    if 'star' in raw_lbl:
                        stars = int(raw_lbl.split()[0])
                        compound = (stars - 3) / 2
                        if stars >= 4:    lbl = 'POSITIVE'
                        elif stars <= 2:  lbl = 'NEGATIVE'
                        else:             lbl = 'NEUTRAL'
                    else:
                        is_pos = 'pos' in raw_lbl
                        lbl = 'POSITIVE' if is_pos else 'NEGATIVE'
                        sc = float(r['score'])
                        compound = sc if is_pos else -sc
                    return lbl, round(compound, 4)
                except Exception:
                    return 'NEUTRAL', 0.0
            results = df_res[col].apply(_m_analyze)
            df_res['sentiment_vader']    = results.apply(lambda x: x[0])
            df_res['sentiment_compound'] = results.apply(lambda x: x[1])
            confidence = f"{df_res['sentiment_compound'].abs().mean():.3f}"

        else:  # Hybrid
            analyzer, _ = self._get_vader()
            df_res = analyzer.analyze_dataframe(df_res, use_transformer=False)
            pipe, _ = self._get_transformer()
            if pipe is not None:
                col = 'message_cleaned' if 'message_cleaned' in df_res.columns else 'message'
                uncertain = df_res['sentiment_compound'].abs() < 0.25
                if uncertain.sum() > 0:
                    def _h(text):
                        try:
                            r = pipe(str(text)[:512])[0]
                            lbl = 'POSITIVE' if r['label'].upper() == 'POSITIVE' else 'NEGATIVE'
                            sc = float(r['score'])
                            return lbl, round(sc if lbl == 'POSITIVE' else -sc, 4)
                        except Exception:
                            return None, None
                    for idx in df_res[uncertain].index:
                        lbl, comp = _h(df_res.at[idx, col])
                        if lbl:
                            df_res.at[idx, 'sentiment_vader']    = lbl
                            df_res.at[idx, 'sentiment_compound'] = comp
                extra['transformer_refinements'] = int(uncertain.sum())
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
            return None, None, None

        colors = {'POSITIVE': '#4ADE80', 'NEGATIVE': '#F87171', 'NEUTRAL': '#FBBF24'}
        fill_colors = {'POSITIVE': 'rgba(74,222,128,0.15)', 'NEGATIVE': 'rgba(248,113,113,0.15)', 'NEUTRAL': 'rgba(251,191,36,0.15)'}

        dist = df_res['sentiment_vader'].value_counts()
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

        fig_time = None
        if 'datetime' in df_res.columns:
            try:
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
                            fillcolor=fill_colors.get(label, 'rgba(148,163,184,0.15)'),
                        ))
                fig_time.update_layout(
                    title='Sentiment Trend Over Time',
                    height=300, **_DARK_LAYOUT,
                    xaxis_title='', yaxis_title='Messages',
                    legend=dict(orientation='h', y=1.08, font=dict(color='#94A3B8')),
                )
            except Exception:
                fig_time = None

        return fig_pie, fig_bar, fig_time

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
