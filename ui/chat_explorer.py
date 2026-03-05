"""
chat_explorer.py — WhatsApp-style chat with full media detection
Place in: whatsapp-analyzer/ui/chat_explorer.py
"""
import re
import streamlit as st
import pandas as pd

# ── Sentiment colors ──────────────────────────────────────────────────────
SENTIMENT_BG     = {'POSITIVE':'rgba(16,185,129,0.10)','NEGATIVE':'rgba(248,113,113,0.10)','NEUTRAL':'rgba(148,163,184,0.08)'}
SENTIMENT_BORDER = {'POSITIVE':'rgba(16,185,129,0.35)','NEGATIVE':'rgba(248,113,113,0.35)','NEUTRAL':'rgba(148,163,184,0.25)'}
SENTIMENT_DOT    = {'POSITIVE':'#10B981','NEGATIVE':'#F87171','NEUTRAL':'#94A3B8'}
EMOTION_EMOJIS   = {'joy':'😊','anger':'😠','sadness':'😢','fear':'😨','surprise':'😲','disgust':'🤮','neutral':'😐'}
USER_PALETTE     = ['#E67E22','#E74C3C','#9B59B6','#2980B9','#27AE60','#F39C12','#D35400','#16A085','#8E44AD','#2471A3']

# ── Media regex ───────────────────────────────────────────────────────────
URL_RE     = re.compile(r'https?://\S+|www\.\S+', re.I)
IMAGE_RE   = re.compile(r'https?://\S+\.(?:jpg|jpeg|png|gif|webp|svg|bmp)(\?\S*)?', re.I)
VIDEO_RE   = re.compile(r'https?://\S+\.(?:mp4|mov|avi|mkv|webm)(\?\S*)?|(?:youtube\.com/watch|youtu\.be/|youtube\.com/shorts)\S*', re.I)
YOUTUBE_RE = re.compile(r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})', re.I)
OMIT_RE    = re.compile(r'<[^>]*omitted>|\(file attached\)|\(image omitted\)|\(video omitted\)|\(audio omitted\)|image omitted|video omitted|audio omitted|media omitted|document omitted', re.I)

OMIT_TYPES = {
    'image omitted':    ('🖼️','Image','#0EA5E9'),
    'video omitted':    ('🎬','Video','#8B5CF6'),
    'audio omitted':    ('🎵','Audio','#EC4899'),
    'document omitted': ('📄','Document','#F59E0B'),
    'sticker omitted':  ('🎭','Sticker','#10B981'),
    'gif omitted':      ('🎞️','GIF','#06B6D4'),
    'media omitted':    ('📎','Media','#64748B'),
    'file attached':    ('📁','File','#F59E0B'),
}

PLATFORM_MAP = [
    ('instagram',  '📸','Instagram','#E1306C'),
    ('twitter',    '𝕏','Twitter/X','#1DA1F2'),
    ('x.com',      '𝕏','Twitter/X','#1DA1F2'),
    ('facebook',   '👤','Facebook','#1877F2'),
    ('github',     '⚙️','GitHub','#6E5494'),
    ('linkedin',   '💼','LinkedIn','#0077B5'),
    ('spotify',    '🎵','Spotify','#1DB954'),
    ('amazon',     '🛍️','Amazon','#FF9900'),
    ('flipkart',   '🛒','Flipkart','#2874F0'),
    ('maps.google','📍','Google Maps','#EA4335'),
    ('goo.gl/maps','📍','Google Maps','#EA4335'),
    ('drive.google','📁','Google Drive','#4285F4'),
    ('docs.google', '📝','Google Docs','#4285F4'),
    ('forms.google','📋','Google Forms','#673AB7'),
    ('meet.google', '📹','Google Meet','#00897B'),
    ('wa.me',       '💬','WhatsApp','#25D366'),
    ('whatsapp',    '💬','WhatsApp','#25D366'),
    ('paytm',       '💳','Paytm','#00BAF2'),
    ('gpay',        '💰','GPay','#4CAF50'),
    ('upi',         '💰','UPI','#4CAF50'),
    ('zomato',      '🍕','Zomato','#E23744'),
    ('swiggy',      '🛵','Swiggy','#FC8019'),
    ('sharechat',   '💬','ShareChat','#FF6550'),
    ('telegram',    '✈️','Telegram','#0088CC'),
    ('reddit',      '🤖','Reddit','#FF4500'),
]


def _user_color(u): return USER_PALETTE[hash(u) % len(USER_PALETTE)]
def _is_light():    return st.session_state.get('theme','light') == 'light'
def _short_url(url):
    u = re.sub(r'^https?://','',url)
    return u[:52]+'…' if len(u)>52 else u

def _detect_omit(text):
    t = text.strip().lower()
    for key,(em,lbl,col) in OMIT_TYPES.items():
        if key in t: return em,lbl,col
    if OMIT_RE.search(t): return '📎','Media','#64748B'
    return None

def _platform_info(url):
    u = url.lower()
    for keyword,icon,name,color in PLATFORM_MAP:
        if keyword in u: return icon,name,color
    return '🔗','Link','#0EA5E9'


# ── HTML builders ─────────────────────────────────────────────────────────

def _yt_card(vid_id, is_light):
    bg  = '#FFFFFF' if is_light else 'rgba(17,24,39,0.85)'
    bdr = 'rgba(184,136,58,0.30)' if is_light else 'rgba(139,92,246,0.35)'
    sc  = '#3E2F1C' if is_light else '#CBD5E1'
    url = f'https://www.youtube.com/watch?v={vid_id}'
    th  = f'https://img.youtube.com/vi/{vid_id}/hqdefault.jpg'
    return (
        f'<a href="{url}" target="_blank" style="text-decoration:none;display:inline-block;margin-top:8px;">'
        f'<div style="background:{bg};border:1px solid {bdr};border-radius:12px;overflow:hidden;max-width:300px;">'
        f'<div style="position:relative;">'
        f'<img src="{th}" style="width:100%;height:auto;display:block;" onerror="this.parentElement.style.display=\'none\'">'
        f'<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);'
        f'background:rgba(255,0,0,0.88);border-radius:50%;width:42px;height:42px;'
        f'display:flex;align-items:center;justify-content:center;font-size:16px;">▶️</div>'
        f'</div>'
        f'<div style="padding:8px 12px;">'
        f'<div style="font-size:10px;font-weight:700;color:#FF0000;margin-bottom:2px;">▶ YouTube Video</div>'
        f'<div style="font-size:10px;color:{sc};font-family:monospace;">youtu.be/{vid_id}</div>'
        f'</div></div></a>'
    )


def _link_card(url, is_light):
    bg   = '#FFFBF5' if is_light else 'rgba(17,24,39,0.7)'
    bdr  = 'rgba(184,136,58,0.28)' if is_light else 'rgba(0,200,150,0.15)'
    sc   = '#3E2F1C' if is_light else '#CBD5E1'
    icon, name, color = _platform_info(url)
    disp = _short_url(url)
    return (
        f'<a href="{url}" target="_blank" style="text-decoration:none;display:block;margin-top:7px;">'
        f'<div style="background:{bg};border:1px solid {bdr};border-left:3px solid {color};'
        f'border-radius:10px;padding:9px 13px;max-width:380px;'
        f'display:flex;align-items:center;gap:10px;">'
        f'<div style="font-size:20px;flex-shrink:0;">{icon}</div>'
        f'<div style="min-width:0;flex:1;">'
        f'<div style="font-size:10px;font-weight:700;color:{color};margin-bottom:2px;">{name}</div>'
        f'<div style="font-size:10px;color:{sc};font-family:monospace;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{disp}</div>'
        f'</div>'
        f'<div style="font-size:13px;color:{sc};flex-shrink:0;">↗</div>'
        f'</div></a>'
    )


def _omit_card(em, lbl, col, is_light):
    r,g,b = int(col[1:3],16),int(col[3:5],16),int(col[5:7],16)
    bg  = f'rgba({r},{g},{b},0.07)'
    bdr = f'rgba({r},{g},{b},0.3)'
    tc  = '#18120A' if is_light else '#F1F5F9'
    return (
        f'<div style="background:{bg};border:1px dashed {bdr};border-radius:10px;'
        f'padding:9px 14px;margin-top:4px;display:inline-flex;align-items:center;gap:10px;">'
        f'<div style="font-size:22px;">{em}</div>'
        f'<div>'
        f'<div style="font-size:12px;font-weight:700;color:{col};">{lbl}</div>'
        f'<div style="font-size:10px;color:{tc};">not stored in chat export</div>'
        f'</div></div>'
    )


# ══════════════════════════════════════════════════════════════════════════

class ChatExplorer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().sort_values('datetime')

    def render_chat_explorer(self, page_size: int = 50):
        # ── Header ────────────────────────────────────────────────────────
        h1, h2 = st.columns([4, 1])
        with h1: st.subheader("💬 Live Chat Explorer")
        with h2: st.radio("View",["Chronological","By User"],horizontal=True,key="chat_view_mode")

        # ── Filters row ───────────────────────────────────────────────────
        f1, f3 = st.columns([3, 2])
        with f1:
            search = st.text_input("Search","",placeholder="🔍 Search messages...",
                                   label_visibility="collapsed",key="chat_search")
        with f3:
            user_f = st.selectbox("User Filter",
                ["All Users"] + sorted(self.df['user'].unique().tolist()),
                label_visibility="collapsed",key="user_filter_sel")

        # ── Media filter — pure session state, no widget key conflict ─────
        if 'mf_active' not in st.session_state:
            st.session_state['mf_active'] = "All"
        media_f = st.session_state['mf_active']

        # ── Filter logic ──────────────────────────────────────────────────
        df2 = self.df.copy()
        if st.session_state.get("chat_view_mode") == "By User":
            df2 = df2.sort_values(['user','datetime'])

        mc = 'message_cleaned' if 'message_cleaned' in df2.columns else 'message'

        # Ensure message_lower exists
        if 'message_lower' not in df2.columns:
            df2['message_lower'] = df2[mc].fillna('').astype(str).str.lower()

        if search:
            df2 = df2[df2['message_lower'].str.contains(search.lower(), na=False)]
        if user_f != "All Users":
            df2 = df2[df2['user'] == user_f]
        _mf = st.session_state.get('mf_active', 'All')
        if _mf == "links":
            df2 = df2[df2[mc].apply(lambda x: bool(URL_RE.search(str(x))))]
        elif _mf == "images":
            _img_re = re.compile(r'image omitted|<image|IMG-|photo', re.I)
            df2 = df2[df2[mc].apply(lambda x: bool(_img_re.search(str(x))) or bool(IMAGE_RE.search(str(x))))]
        elif _mf == "youtube":
            df2 = df2[df2[mc].apply(lambda x: bool(YOUTUBE_RE.search(str(x))) or bool(VIDEO_RE.search(str(x))))]
        elif _mf == "omit":
            df2 = df2[df2[mc].apply(lambda x: bool(OMIT_RE.search(str(x))))]

        # ── Stats bar ─────────────────────────────────────────────────────
        is_light = _is_light()
        ac   = '#B8883A' if is_light else '#00C896'
        sc   = '#3E2F1C' if is_light else '#CBD5E1'
        bg_s = 'rgba(184,136,58,0.08)' if is_light else 'rgba(0,200,150,0.05)'

        n_links = df2[mc].apply(lambda x: bool(URL_RE.search(str(x)))).sum()
        n_yt    = df2[mc].apply(lambda x: bool(YOUTUBE_RE.search(str(x)))).sum()
        n_omit  = df2[mc].apply(lambda x: bool(OMIT_RE.search(str(x)))).sum()
        n_imgs  = df2[mc].apply(lambda x: bool(re.search(r'image omitted|<image|IMG-', str(x), re.I))).sum()

        # ── Clickable stats bar ───────────────────────────────────────────
        is_lt2 = _is_light()
        act_c  = '#B8883A' if is_lt2 else '#00C896'
        sb1, sb2, sb3, sb4, sb5, sb6 = st.columns([2,1,1,1,1,1])
        with sb1:
            st.markdown(
                f'<div style="background:{bg_s};border-radius:8px;padding:7px 14px;text-align:center;">'
                f'<span style="font-size:12px;color:{sc};">💬 <b style="color:{ac};">{len(df2):,}</b> msgs</span>'
                f'</div>', unsafe_allow_html=True)
        with sb2:
            _active_all = st.session_state.get('mf_active','All') == 'All'
            if st.button(f"✕ All" if not _active_all else "All ✓",
                         key="flt_all", use_container_width=True,
                         help="Show all messages"):
                st.session_state['mf_active'] = "All"
                st.rerun()
        with sb3:
            if st.button(f"🔗 {n_links}", key="flt_links", use_container_width=True,
                         help="Show only messages with links"):
                st.session_state['mf_active'] = "links"
                st.rerun()
        with sb4:
            if st.button(f"▶️ {n_yt}", key="flt_yt", use_container_width=True,
                         help="Show YouTube messages"):
                st.session_state['mf_active'] = "youtube"
                st.rerun()
        with sb5:
            if st.button(f"📎 {n_omit}", key="flt_omit", use_container_width=True,
                         help="Show media omitted"):
                st.session_state['mf_active'] = "omit"
                st.rerun()
        with sb6:
            if st.button(f"🖼️ {n_imgs}", key="flt_imgs", use_container_width=True,
                         help="Show image messages"):
                st.session_state['mf_active'] = "images"
                st.rerun()
        st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)

        # ── Pagination ────────────────────────────────────────────────────
        num_pages = max(1,(len(df2)+page_size-1)//page_size)
        if 'chat_page' not in st.session_state: st.session_state.chat_page = 0
        st.session_state.chat_page = min(st.session_state.chat_page, num_pages-1)

        pc1, pc2, pc3 = st.columns([1,2,1])
        with pc1:
            if st.button("← Prev", use_container_width=True, key="btn_prev"):
                if st.session_state.chat_page > 0: st.session_state.chat_page -= 1
        with pc2:
            st.markdown(f'<div style="text-align:center;font-size:12px;color:{sc};padding:8px;">Page <b>{st.session_state.chat_page+1}</b> / {num_pages}</div>',unsafe_allow_html=True)
        with pc3:
            if st.button("Next →", use_container_width=True, key="btn_next"):
                if st.session_state.chat_page < num_pages-1: st.session_state.chat_page += 1

        start   = st.session_state.chat_page * page_size
        df_page = df2.iloc[start:start+page_size]
        for _, row in df_page.iterrows():
            self._render_bubble(row, is_light)

    # ── Bubble renderer ────────────────────────────────────────────────────
    def _render_bubble(self, row, is_light):
        mc        = 'message_cleaned' if 'message_cleaned' in row.index else 'message'
        user      = str(row['user'])
        ts        = row['datetime'].strftime("%d %b · %H:%M")
        raw       = str(row.get(mc, row.get('message','')))
        sentiment = str(row.get('sentiment_vader','NEUTRAL')).upper()
        emotion   = str(row.get('emotion','neutral')).lower()
        is_toxic  = bool(row.get('is_toxic',False))

        u_col    = _user_color(user)
        s_dot    = SENTIMENT_DOT.get(sentiment,'#94A3B8')
        em_emoji = EMOTION_EMOJIS.get(emotion,'😐')
        initials = ''.join(p[0].upper() for p in user.split()[:2])

        if is_light:
            bub_bg  = '#FFFFFF'
            bub_bdr = 'rgba(184,136,58,0.28)'
            msg_c   = '#18120A'
            ts_c    = '#9C7A52'
            meta_c  = '#9C7A52'
        else:
            _s_tint = {'POSITIVE':'rgba(16,185,129,0.06)','NEGATIVE':'rgba(248,113,113,0.06)','NEUTRAL':'rgba(30,41,59,0.5)'}
            bub_bg  = _s_tint.get(sentiment,'rgba(30,41,59,0.5)')
            bub_bdr = SENTIMENT_BORDER.get(sentiment,'rgba(148,163,184,0.25)')
            msg_c   = '#F1F5F9'
            ts_c    = '#CBD5E1'
            meta_c  = '#CBD5E1'

        toxic_b = (
            '<span style="background:rgba(248,113,113,0.15);color:#F87171;'
            'font-size:9px;padding:2px 6px;border-radius:4px;font-weight:700;margin-left:4px;">☢ TOXIC</span>'
        ) if is_toxic else ''

        # ── Media detection ───────────────────────────────────────────────
        media_html   = ''
        display_msg  = raw
        omit_result  = _detect_omit(raw)

        if omit_result:
            em, lbl, col = omit_result
            media_html  = _omit_card(em, lbl, col, is_light)
            display_msg = ''
        else:
            # YouTube cards
            yt_ids = YOUTUBE_RE.findall(raw)
            for vid in yt_ids[:2]:
                media_html += _yt_card(vid, is_light)

            # Other links
            all_urls   = URL_RE.findall(raw)
            non_yt     = [u for u in all_urls if not YOUTUBE_RE.search(u)]
            for url in non_yt[:3]:
                media_html += _link_card(url, is_light)

            # Strip URLs from displayed text
            if all_urls:
                for u in all_urls:
                    display_msg = display_msg.replace(u,'').strip()
                display_msg = re.sub(r'\s{2,}',' ', display_msg).strip()

        # ── Assemble bubble ───────────────────────────────────────────────
        html = (
            f'<div style="margin:10px 0;display:flex;align-items:flex-start;gap:10px;">'
            # Avatar
            f'<div style="width:36px;height:36px;border-radius:50%;background:{u_col};'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:12px;font-weight:700;color:#fff;flex-shrink:0;">{initials}</div>'
            # Body
            f'<div style="flex:1;min-width:0;">'
            # Header row
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">'
            f'<span style="font-weight:700;color:{u_col};font-size:13px;">{user}</span>'
            f'<span style="font-size:11px;color:{ts_c};">{ts}</span>'
            f'<span style="font-size:14px;">{em_emoji}</span>'
            f'{toxic_b}</div>'
            # Bubble
            f'<div style="background:{bub_bg};border:1px solid {bub_bdr};'
            f'border-left:3px solid {s_dot};border-radius:0 12px 12px 12px;padding:10px 14px;">'
        )

        if display_msg:
            html += f'<div style="font-size:13.5px;line-height:1.65;color:{msg_c};word-break:break-word;">{display_msg}</div>'
        if media_html:
            html += media_html

        html += (
            f'</div>'   # close bubble
            f'<div style="display:flex;gap:10px;margin-top:4px;font-size:10.5px;color:{meta_c};">'
            f'<span>💭 {sentiment.title()}</span><span>🎭 {emotion.title()}</span>'
            f'</div></div></div>'
        )

        st.markdown(html, unsafe_allow_html=True)

    def render_user_stats(self):
        st.subheader("👥 User Statistics")
        stats = self.df.groupby('user').agg(
            Messages=('message','count'),
            Avg_Length=('message_length','mean'),
            Avg_Sentiment=('sentiment_compound','mean'),
        ).round(2).sort_values('Messages',ascending=False)
        c1,c2 = st.columns(2)
        with c1:
            st.write("**Message Distribution**"); st.bar_chart(stats['Messages'])
        with c2:
            st.write("**Avg Sentiment**"); st.bar_chart(stats['Avg_Sentiment'])
        st.dataframe(stats, use_container_width=True)