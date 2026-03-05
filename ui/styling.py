import streamlit as st


# ══════════════════════════════════════════════════════════════════════════
#  THEME SELECTOR
# ══════════════════════════════════════════════════════════════════════════

def render_theme_selector():
    """Render sidebar theme toggle and inject CSS."""
    with st.sidebar:
        st.markdown(
            '<p style="font-size:10px;font-weight:700;letter-spacing:.14em;'
            'text-transform:uppercase;margin-bottom:6px;opacity:.6;">APPEARANCE</p>',
            unsafe_allow_html=True,
        )
        theme = st.radio(
            "theme_radio",
            ["🌤️ Light Theme", "🌙 Dark Theme"],
            index=0 if st.session_state.get("theme", "light") == "light" else 1,
            label_visibility="collapsed",
            key="theme_radio_widget",
        )
        st.session_state.theme = "light" if "Light" in theme else "dark"

    if st.session_state.get("theme", "light") == "light":
        _inject_light_css()
    else:
        _inject_dark_css()


# ══════════════════════════════════════════════════════════════════════════
#  LIGHT THEME
# ══════════════════════════════════════════════════════════════════════════

def _inject_light_css():
    st.markdown("""<style>
/* ── Input text in light theme ── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] div,
[data-testid="stSelectbox"] span,
[data-baseweb="input"] input {
    color: #18120A !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #BBA98C !important;
}
/* ── Instant paint — prevents dark flash ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .stApp, #root {
    background-color: #FDFAF4 !important;
    color: #18120A !important;
}
[data-testid="stHeader"] {
    background-color: #FDFAF4 !important;
}
[data-testid="stSidebar"] {
    background-color: #FFF8ED !important;
}
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;1,400;1,600&family=Outfit:wght@300;400;600;700&family=Fira+Code:wght@400;500&display=swap');

:root {
    --bg:        #FDFAF4;
    --bg-card:   #FFFFFF;
    --ink:       #18120A;
    --ink-mid:   #3E2F1C;
    --ink-soft:  #7A6248;
    --ink-faint: #BBA98C;
    --gold:      #B8883A;
    --gold-hi:   #D4A853;
    --gold-pale: #FBF0DC;
    --amber:     #C47B2A;
    --teal:      #0A8F6B;
    --border:    rgba(184,136,58,.22);
    --shadow-sm: 0 2px 12px rgba(24,18,10,.07);
    --shadow-md: 0 6px 28px rgba(24,18,10,.11);
    --shadow-lg: 0 14px 48px rgba(24,18,10,.15);
    --glow:      0 0 0 3px rgba(184,136,58,.18);
}

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#FFF8ED,#FBF0DC) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family:'Outfit',sans-serif !important; color:var(--ink-mid) !important; }

h1 { font-family:'Cormorant Garamond',serif !important; font-weight:600 !important; color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; font-size:clamp(1.8rem,3.5vw,2.8rem) !important; }
h2 { font-family:'Cormorant Garamond',serif !important; font-style:italic !important; font-weight:600 !important; color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; }
h3 { font-family:'Outfit',sans-serif !important; font-weight:700 !important; font-size:.72rem !important; letter-spacing:.17em !important; text-transform:uppercase !important; color:var(--ink-soft) !important; -webkit-text-fill-color:var(--ink-soft) !important; }
p  { color:var(--ink-mid) !important; line-height:1.75 !important; }
label { color:var(--ink-soft) !important; font-size:12.5px !important; }

[data-testid="stTabs"] [role="tablist"] { border-bottom:1.5px solid var(--border) !important; gap:2px !important; }
[data-testid="stTabs"] [role="tab"] { font-family:'Outfit',sans-serif !important; font-size:11px !important; font-weight:700 !important; letter-spacing:.09em !important; text-transform:uppercase !important; color:var(--ink-soft) !important; border-radius:0 !important; padding:12px 18px !important; border-bottom:2px solid transparent !important; transition:all .25s ease !important; background:transparent !important; }
[data-testid="stTabs"] [role="tab"]:hover { color:var(--gold) !important; background:var(--gold-pale) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:var(--amber) !important; border-bottom-color:var(--gold-hi) !important; font-style:italic; }

.stButton > button { font-family:'Outfit',sans-serif !important; font-size:11.5px !important; font-weight:700 !important; letter-spacing:.10em !important; text-transform:uppercase !important; background:var(--ink) !important; color:var(--bg) !important; border:none !important; border-radius:12px !important; padding:12px 30px !important; box-shadow:0 4px 18px rgba(24,18,10,.18) !important; transition:all .32s ease !important; }
.stButton > button:hover { background:var(--amber) !important; transform:translateY(-3px) !important; box-shadow:0 10px 32px rgba(196,123,42,.28) !important; }

input, textarea, [data-baseweb="input"] input { background:#FFFFFF !important; border:1.5px solid var(--border) !important; color:var(--ink) !important; border-radius:10px !important; font-family:'Outfit',sans-serif !important; }
input:focus, textarea:focus { border-color:var(--gold) !important; box-shadow:var(--glow) !important; }
[data-baseweb="select"] > div { background:#FFFFFF !important; border:1.5px solid var(--border) !important; border-radius:10px !important; color:var(--ink) !important; }
[data-baseweb="tag"] { background:var(--gold-pale) !important; border:1px solid rgba(184,136,58,.28) !important; color:var(--amber) !important; border-radius:7px !important; }

[data-testid="stAlert"] { background:#FFFEF8 !important; border:1px solid var(--border) !important; border-left:3px solid var(--gold) !important; border-radius:11px !important; }
[data-testid="stAlert"] p,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
[data-testid="stInfo"] p { color:var(--ink) !important; }
[data-testid="stInfo"]    { background:rgba(184,136,58,0.06) !important; border-left:3px solid var(--gold) !important; color:var(--ink) !important; }
[data-testid="stSuccess"] { background:#F5FAF2 !important; border-left-color:#6BA350 !important; }

[data-baseweb="radio"] label    { color:var(--ink-mid) !important; }
[data-baseweb="checkbox"] label { color:var(--ink-mid) !important; }
.stMarkdown p, .stMarkdown li  { color:var(--ink-mid) !important; }
[data-testid="stMarkdownContainer"] p { color:var(--ink-mid) !important; }
[data-testid="stCaptionContainer"] p  { color:var(--ink-soft) !important; }

.stSpinner > div            { border-top-color:var(--gold) !important; }
[data-testid="stSpinner"] p { color:var(--gold) !important; }

::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--ink-faint); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--gold); }

[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,var(--gold),var(--amber)) !important; border-radius:4px !important; }
[data-testid="stFileUploader"] { background:var(--gold-pale) !important; border:2px dashed rgba(184,136,58,.32) !important; border-radius:14px !important; }
[data-testid="stFileUploader"]:hover { background:#FFF6E6 !important; border-color:var(--gold) !important; }

/* ═══════════════════════════════════════════════════════════════
   LIGHT THEME — INLINE HTML CARD FIXES
   Dark bg + light text cards se text invisible hota tha.
   Ye CSS un sabko light bg aur dark text mein convert karta hai.
═══════════════════════════════════════════════════════════════ */

/* Dark backgrounds ko light mein badlo */
[data-testid="stMarkdownContainer"] [style*="background:rgba(17,24,39"],
[data-testid="stMarkdownContainer"] [style*="background: rgba(17,24,39"],
[data-testid="stMarkdownContainer"] [style*="background:rgba(12,20,35"],
[data-testid="stMarkdownContainer"] [style*="background:rgba(7,9,15"] {
    background: #FFFBF2 !important;
    border-color: rgba(184,136,58,0.25) !important;
}
[data-testid="stMarkdownContainer"] [style*="background:linear-gradient(135deg,rgba(17"],
[data-testid="stMarkdownContainer"] [style*="background:linear-gradient(135deg, rgba(17"],
[data-testid="stMarkdownContainer"] [style*="background:linear-gradient(135deg,rgba(12"] {
    background: #FFF8ED !important;
    border-color: rgba(184,136,58,0.3) !important;
}

/* Light text colors ko dark mein badlo */
[data-testid="stMarkdownContainer"] [style*="color:#E2E8F0"] { color:#18120A !important; }
[data-testid="stMarkdownContainer"] [style*="color:#CBD5E1"] { color:#2C1A0E !important; }
[data-testid="stMarkdownContainer"] [style*="color:#94A3B8"] { color:#5C3D2E !important; }
[data-testid="stMarkdownContainer"] [style*="color:#64748B"] { color:#7A6248 !important; }
[data-testid="stMarkdownContainer"] [style*="color:#F0F6FF"] { color:#18120A !important; }
[data-testid="stMarkdownContainer"] [style*="color:#475569"] { color:#7A6248 !important; }
[data-testid="stMarkdownContainer"] [style*="color:#1E293B"] { color:#BBA98C !important; }
[data-testid="stMarkdownContainer"] [style*="color:#E2E8F0;"] { color:#18120A !important; }

/* Conversation overview card text */
[data-testid="stMarkdownContainer"] div[style*="font-size:15px"] { color:#18120A !important; }
[data-testid="stMarkdownContainer"] div[style*="font-size:14px"] { color:#3E2F1C !important; }
[data-testid="stMarkdownContainer"] div[style*="font-size:13"] { color:#3E2F1C !important; }
[data-testid="stMarkdownContainer"] div[style*="line-height:1.8"] { color:#3E2F1C !important; }
[data-testid="stMarkdownContainer"] div[style*="line-height:1.85"] { color:#18120A !important; }
[data-testid="stMarkdownContainer"] div[style*="line-height:1.65"] { color:#18120A !important; }
[data-testid="stMarkdownContainer"] div[style*="line-height:1.6"] { color:#3E2F1C !important; }

/* Provider/feature cards dark bg */
[data-testid="stMarkdownContainer"] a > div { color:var(--ink-mid) !important; }

</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  DARK THEME
# ══════════════════════════════════════════════════════════════════════════

def _inject_dark_css():
    st.markdown("""<style>
/* ── Input & selectbox text in dark theme ── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] div,
[data-testid="stSelectbox"] span,
[data-baseweb="input"] input,
[data-baseweb="select"] div,
[data-baseweb="select"] span {
    color: #F1F5F9 !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #94A3B8 !important;
}
/* Stats bar buttons in dark */
[data-testid="stButton"] button {
    color: #F1F5F9 !important;
    border-color: rgba(0,200,150,0.25) !important;
}
/* ── Instant paint — prevents light flash ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .stApp, #root {
    background-color: #07090F !important;
    color: #F0F6FF !important;
}
[data-testid="stHeader"] {
    background-color: #07090F !important;
}
[data-testid="stSidebar"] {
    background-color: #0D1117 !important;
}
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Outfit:wght@300;400;600;700&family=Fira+Code:wght@400;500&display=swap');

:root {
    --bg:        #07090F;
    --bg-card:   rgba(17,24,39,0.75);
    --ink:       #F0F6FF;
    --ink-mid:   #94A3B8;
    --ink-soft:  #475569;
    --teal:      #00C896;
    --teal-hi:   #00FFB8;
    --cyan:      #22D3EE;
    --violet:    #8B5CF6;
    --gold:      #F59E0B;
    --border:    rgba(0,200,150,0.14);
    --shadow-md: 0 6px 28px rgba(0,0,0,0.4);
    --shadow-lg: 0 14px 48px rgba(0,0,0,0.55);
    --glow:      0 0 0 3px rgba(0,200,150,0.18);
}

html, body, .stApp {
    background:
        radial-gradient(ellipse 80% 50% at 15% 10%, rgba(0,200,150,0.09) 0%, transparent 55%),
        radial-gradient(ellipse 60% 40% at 85% 80%, rgba(139,92,246,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 40% 60% at 50% 50%, rgba(34,211,238,0.04) 0%, transparent 65%),
        #07090F !important;
    color:var(--ink) !important;
    font-family:'Outfit',sans-serif !important;
}

[data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(5,8,18,0.97),rgba(10,14,28,0.95)) !important; border-right:1px solid var(--border) !important; }
[data-testid="stSidebar"] * { font-family:'Outfit',sans-serif !important; color:var(--ink-mid) !important; }

h1 { font-family:'Syne',sans-serif !important; font-weight:800 !important; color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; }
h2 { font-family:'Syne',sans-serif !important; font-weight:700 !important; color:var(--ink) !important; }
h3 { font-family:'Outfit',sans-serif !important; font-weight:600 !important; color:var(--ink-mid) !important; }
p  { color:var(--ink-mid) !important; }
label { color:var(--ink-mid) !important; }

[data-testid="stTabs"] [role="tablist"] { border-bottom:1.5px solid var(--border) !important; }
[data-testid="stTabs"] [role="tab"] { font-family:'Outfit',sans-serif !important; font-size:11px !important; font-weight:700 !important; letter-spacing:.09em !important; text-transform:uppercase !important; color:var(--ink-soft) !important; border-bottom:2px solid transparent !important; padding:12px 18px !important; transition:all .25s ease !important; background:transparent !important; }
[data-testid="stTabs"] [role="tab"]:hover { color:var(--teal) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:var(--teal-hi) !important; border-bottom:2.5px solid var(--teal) !important; text-shadow:0 0 18px rgba(0,200,150,0.35) !important; background:linear-gradient(180deg,rgba(0,200,150,0.07),transparent) !important; }

.stButton > button { font-family:'Outfit',sans-serif !important; font-size:11.5px !important; font-weight:700 !important; letter-spacing:.10em !important; text-transform:uppercase !important; background:linear-gradient(135deg,#00C896,#0EA5E9 50%,#8B5CF6) !important; color:#FFFFFF !important; border:none !important; border-radius:12px !important; padding:12px 30px !important; transition:all .32s ease !important; }
.stButton > button:hover { opacity:.88 !important; transform:translateY(-3px) !important; }

input, textarea, [data-baseweb="input"] input { background:rgba(17,24,39,0.6) !important; border:1px solid var(--border) !important; color:var(--ink) !important; border-radius:10px !important; }
input:focus, textarea:focus { border-color:var(--teal) !important; box-shadow:var(--glow) !important; }
[data-baseweb="select"] > div { background:rgba(17,24,39,0.6) !important; border:1px solid var(--border) !important; border-radius:10px !important; color:var(--ink) !important; }

[data-testid="stAlert"] { background:rgba(17,24,39,0.5) !important; border-left:3px solid var(--teal) !important; border-radius:11px !important; }
[data-testid="stAlert"] p,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p { color:var(--ink) !important; }
[data-testid="stInfo"] { background:rgba(0,200,150,0.07) !important; }

[data-baseweb="radio"] label    { color:var(--ink-mid) !important; }
[data-baseweb="checkbox"] label { color:var(--ink-mid) !important; }
.stMarkdown p, .stMarkdown li  { color:var(--ink-mid) !important; }
[data-testid="stMarkdownContainer"] p { color:var(--ink-mid) !important; }
[data-testid="stCaptionContainer"] p  { color:var(--ink-soft) !important; }

.stSpinner > div            { border-top-color:var(--teal) !important; }
[data-testid="stSpinner"] p { color:var(--teal) !important; }

::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:rgba(0,0,0,0.2); }
::-webkit-scrollbar-thumb { background:rgba(0,200,150,0.35); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--teal); }

[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,var(--teal),var(--cyan)) !important; border-radius:4px !important; }
[data-testid="stFileUploader"] { background:rgba(0,200,150,0.04) !important; border:2px dashed rgba(0,200,150,0.25) !important; border-radius:14px !important; }
[data-testid="stFileUploader"]:hover { border-color:var(--teal) !important; }

</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  LEGACY COMPAT
# ══════════════════════════════════════════════════════════════════════════

def apply_modern_theme():
    """Inject CSS based on current theme in session_state."""
    import streamlit as st
    if st.session_state.get("theme", "light") == "light":
        _inject_light_css()
    else:
        _inject_dark_css()


# ══════════════════════════════════════════════════════════════════════════
#  KPI CARD — theme-aware inline colors
# ══════════════════════════════════════════════════════════════════════════

def render_kpi_card(title: str, value: str, metric: str, icon: str = "📊", idx: int = 0, delay: int = 0):
    is_light = st.session_state.get("theme", "light") == "light"

    if is_light:
        bg      = "#FFFFFF"
        bdr     = "rgba(184,136,58,0.22)"
        val_c   = "#A59C91"
        title_c = "#BBA98C"
        met_c   = "#B8883A"
        shadow  = "0 2px 14px rgba(24,18,10,0.08)"
    else:
        bg      = "rgba(17,24,39,0.75)"
        bdr     = "rgba(0,200,150,0.14)"
        val_c   = "#F0F6FF"
        title_c = "#CED5DB2F"
        met_c   = "#00C896"
        shadow  = "0 6px 28px rgba(0,0,0,0.35)"

    html = (
        f'<div style="background:{bg};border:1px solid {bdr};border-radius:14px;'
        f'padding:20px 22px;box-shadow:{shadow};margin-bottom:4px;'
        f'transition:all .3s ease;">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        f'<div style="font-size:9.5px;font-weight:700;letter-spacing:.16em;'
        f'text-transform:uppercase;color:{title_c};margin-bottom:6px;">{title}</div>'
        f'<div style="font-size:22px;">{icon}</div></div>'
        f'<div style="font-size:32px;font-weight:700;line-height:1;color:{val_c};'
        f'font-family:serif;margin:4px 0 6px;">{value}</div>'
        f'<div style="font-size:10.5px;color:{met_c};font-family:monospace;">{metric}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  GRADIENT DIVIDER
# ══════════════════════════════════════════════════════════════════════════

def render_gradient_divider(color1: str = "#00C896", color2: str = "#8B5CF6"):
    is_light = st.session_state.get("theme", "light") == "light"
    c1 = "#B8883A" if is_light else color1
    c2 = "#D4A853" if is_light else color2
    st.markdown(
        f'<div style="height:1.5px;background:linear-gradient(90deg,transparent,{c1},{c2},transparent);'
        f'margin:20px 0;border-radius:1px;"></div>',
        unsafe_allow_html=True,
    )

def inject_mobile_css():
    """Inject mobile-responsive CSS overrides."""
    import streamlit as st
    st.markdown("""
    <style>
    /* ── Mobile breakpoint ── */
    @media (max-width: 768px) {
        /* Sidebar collapses nicely */
        section[data-testid="stSidebar"] { min-width: 260px !important; }

        /* Tabs scroll horizontally */
        div[data-testid="stTabs"] > div > div {
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
        }
        div[data-testid="stTabs"] > div > div::-webkit-scrollbar { display: none; }
        button[data-baseweb="tab"] { white-space: nowrap !important; min-width: fit-content !important; }

        /* Stack columns on mobile */
        div[data-testid="column"] { min-width: 100% !important; }

        /* KPI cards full width */
        div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }

        /* Chat bubbles full width */
        div[style*="max-width:380px"] { max-width: 100% !important; }
        div[style*="max-width:320px"] { max-width: 100% !important; }
        div[style*="max-width:300px"] { max-width: 100% !important; }

        /* Font sizes scale down */
        .stMarkdown h1 { font-size: 1.6rem !important; }
        .stMarkdown h2 { font-size: 1.3rem !important; }
        .stMarkdown h3 { font-size: 1.1rem !important; }

        /* Plotly charts full width */
        div[data-testid="stPlotlyChart"] { width: 100% !important; }

        /* Buttons full width on mobile */
        div[data-testid="stButton"] > button { width: 100% !important; }

        /* Selectbox full width */
        div[data-testid="stSelectbox"] { width: 100% !important; }

        /* Reduce padding */
        .main .block-container { padding: 1rem 0.75rem !important; }
        section[data-testid="stSidebar"] .block-container { padding: 1rem !important; }
    }

    @media (max-width: 480px) {
        /* Extra small — hide some decorative elements */
        div[style*="letter-spacing:0.15em"] { display: none !important; }
    }
    </style>
    """, unsafe_allow_html=True)