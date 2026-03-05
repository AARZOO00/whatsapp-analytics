"""
ai_summary.py — Multi-provider AI chat summary
Supports: Anthropic Claude · Google Gemini · Groq (Llama)
Place in: whatsapp-analyzer/analytics/ai_summary.py
"""
import json
from typing import Dict, List
import pandas as pd

MAX_SAMPLE_MESSAGES = 120

# ── Provider configs ──────────────────────────────────────────────────────
PROVIDERS = {
    "anthropic": {
        "name":        "🤖 Anthropic Claude",
        "url":         "https://api.anthropic.com/v1/messages",
        "model":       "claude-sonnet-4-20250514",
        "free":        False,
        "key_hint":    "sk-ant-",
        "key_example": "sk-ant-api03-...",
        "signup_url":  "https://console.anthropic.com",
        "badge":       "#7C3AED",
    },
    "gemini": {
        "name":        "✨ Google Gemini",
        "url":         "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        "model":       "gemini-1.5-flash",
        "free":        True,
        "key_hint":    "AIza",
        "key_example": "AIzaSy...",
        "signup_url":  "https://aistudio.google.com/app/apikey",
        "badge":       "#059669",
    },
    "groq": {
        "name":        "⚡ Groq (Llama 3)",
        "url":         "https://api.groq.com/openai/v1/chat/completions",
        "model":       "llama3-70b-8192",
        "free":        True,
        "key_hint":    "gsk_",
        "key_example": "gsk_...",
        "signup_url":  "https://console.groq.com",
        "badge":       "#D97706",
    },
}


def detect_provider(api_key: str) -> str:
    """Auto-detect which provider the key belongs to."""
    key = api_key.strip()
    if key.startswith("sk-ant-"):  return "anthropic"
    if key.startswith("AIza"):     return "gemini"
    if key.startswith("gsk_"):     return "groq"
    return "unknown"


# ── Prompt builder ────────────────────────────────────────────────────────

def _extract_stats(df: pd.DataFrame) -> Dict:
    from collections import Counter
    total = len(df)
    days  = max((df['datetime'].max() - df['datetime'].min()).days, 1)
    vc    = df['user'].value_counts()

    stopwords = {
        'the','a','an','is','are','was','were','be','have','has','had','do','does',
        'did','to','of','in','on','at','by','for','with','and','or','but','if','it',
        'its','i','me','my','we','you','he','she','they','ok','okay','hi','hey',
        'yes','no','yeah','haha','lol','deleted','message','null','media','omitted',
        'bhi','hai','nhi','kya','aur','mein','ka','ke','ki','ko','se','ho','tha',
    }
    all_tok = []
    for tok in df['tokens']:
        if isinstance(tok, list):
            all_tok.extend([t.lower() for t in tok if t.isalpha() and len(t) > 2])
    top_words = [w for w, _ in Counter(
        [t for t in all_tok if t not in stopwords]
    ).most_common(15)]

    df2 = df.copy()
    df2['hour'] = df2['datetime'].dt.hour
    hr = df2.groupby('hour').size()
    ph = int(hr.idxmax()) if len(hr) > 0 else 12

    dom_em = 'neutral'
    if 'emotion' in df.columns:
        em = df['emotion'].dropna().value_counts()
        if len(em): dom_em = em.index[0]

    return {
        'total_messages':   total,
        'participants':     df['user'].nunique(),
        'days':             days,
        'date_range':       (f"{df['datetime'].min().strftime('%d %b %Y')} – "
                             f"{df['datetime'].max().strftime('%d %b %Y')}"),
        'top_user':         vc.index[0] if len(vc) > 0 else 'Unknown',
        'top_user_pct':     round(vc.iloc[0] / max(total,1) * 100, 1) if len(vc) > 0 else 0,
        'positive_pct':     round((df['sentiment_vader'] == 'POSITIVE').sum() / max(total,1) * 100, 1),
        'negative_pct':     round((df['sentiment_vader'] == 'NEGATIVE').sum() / max(total,1) * 100, 1),
        'neutral_pct':      round((df['sentiment_vader'] == 'NEUTRAL').sum()  / max(total,1) * 100, 1),
        'toxic_pct':        round((df['is_toxic'] == True).sum() / max(total,1) * 100, 1),
        'dominant_emotion': dom_em,
        'peak_hour':        ph,
        'top_words':        top_words,
    }


def _build_prompt(df: pd.DataFrame, stats: Dict) -> str:
    users = df['user'].unique().tolist()
    sample_rows = []
    for user in users:
        u_df = df[df['user'] == user].sort_values('datetime')
        for idx in [0, len(u_df)//4, len(u_df)//2, 3*len(u_df)//4, len(u_df)-1]:
            if 0 <= idx < len(u_df):
                sample_rows.append(u_df.iloc[idx])
    if len(df) > len(sample_rows):
        extra = df.sample(min(MAX_SAMPLE_MESSAGES - len(sample_rows), len(df)), random_state=42)
        sample_rows.extend([row for _, row in extra.iterrows()])
    sample_rows.sort(key=lambda r: r['datetime'])
    sample_rows = sample_rows[:MAX_SAMPLE_MESSAGES]

    msg_col = 'message_cleaned' if 'message_cleaned' in df.columns else 'message'
    messages_text = "\n".join(
        f"[{r['datetime'].strftime('%d %b %H:%M')}] {r['user']}: {str(r.get(msg_col,''))[:120]}"
        for r in sample_rows
        if str(r.get(msg_col, '')).strip() not in ('', 'nan', '<Media omitted>', 'null')
    )

    return f"""You are analyzing a WhatsApp group chat. Statistics:

📊 STATS:
- Messages: {stats['total_messages']:,} | Participants: {stats['participants']} | Duration: {stats['days']} days
- Date range: {stats['date_range']}
- Most active: {stats['top_user']} ({stats['top_user_pct']}%)
- Sentiment: {stats['positive_pct']}% positive, {stats['negative_pct']}% negative
- Dominant emotion: {stats['dominant_emotion']} | Toxicity: {stats['toxic_pct']}%
- Peak hour: {stats['peak_hour']}:00 | Top words: {', '.join(stats['top_words'][:8])}

💬 SAMPLE MESSAGES:
{messages_text}

Provide a rich analysis with these exact sections:

**1. What This Chat Is About**
(2-3 sentences — what is this group, what do they mainly discuss? Be specific.)

**2. Conversation Vibe**
(1-2 sentences — energy, tone, casual/formal, emotional temperature)

**3. Key Themes & Topics**
• Theme 1
• Theme 2
• Theme 3
(3-5 specific bullet points based on actual message content)

**4. Group Dynamics**
(2-3 sentences — who drives conversation, is it balanced, notable patterns)

**5. Notable Observations**
(2-3 sentences — interesting patterns, spikes, emotional moments, anything unique)

**6. ⚡ One-Line Essence**
(Single punchy sentence capturing the soul of this chat)

Be specific and insightful — reference actual content, not just generic statistics. Write in English."""


# ── API callers ───────────────────────────────────────────────────────────

def _call_anthropic(api_key: str, prompt: str) -> str:
    import urllib.request, urllib.error
    payload = json.dumps({
        "model":      "claude-sonnet-4-20250514",
        "max_tokens": 1200,
        "messages":   [{"role": "user", "content": prompt}],
    }).encode('utf-8')
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key.strip(),
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        return data['content'][0]['text']


def _call_gemini(api_key: str, prompt: str) -> str:
    import urllib.request, urllib.error
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-1.5-flash:generateContent?key={api_key.strip()}")
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 1200, "temperature": 0.7},
    }).encode('utf-8')
    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=40) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        return data['candidates'][0]['content']['parts'][0]['text']


def _call_groq(api_key: str, prompt: str) -> str:
    import urllib.request, urllib.error
    payload = json.dumps({
        "model":       "llama3-70b-8192",
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  1200,
        "temperature": 0.7,
    }).encode('utf-8')
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        return data['choices'][0]['message']['content']


# ── Main entry point ──────────────────────────────────────────────────────

def generate_ai_summary(df: pd.DataFrame, api_key: str) -> Dict:
    """
    Auto-detect provider from key prefix and generate summary.
    Returns: {success, summary, error, stats, provider}
    """
    import urllib.error

    if df is None or len(df) == 0:
        return {'success': False, 'error': 'No messages to analyze.', 'summary': '', 'stats': {}, 'provider': ''}

    key = api_key.strip() if api_key else ''
    if not key:
        return {'success': False, 'error': 'Please enter an API key.', 'summary': '', 'stats': {}, 'provider': ''}

    provider = detect_provider(key)
    if provider == 'unknown':
        return {
            'success': False,
            'error':   'Unknown API key format. Keys should start with sk-ant- (Anthropic), AIza (Gemini), or gsk_ (Groq).',
            'summary': '', 'stats': {}, 'provider': '',
        }

    stats  = _extract_stats(df)
    prompt = _build_prompt(df, stats)
    callers = {'anthropic': _call_anthropic, 'gemini': _call_gemini, 'groq': _call_groq}

    try:
        text = callers[provider](key, prompt)
        return {'success': True, 'summary': text, 'stats': stats,
                'provider': provider, 'error': ''}
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        try:
            msg = json.loads(body).get('error', {})
            msg = msg.get('message', body) if isinstance(msg, dict) else str(msg)
        except Exception:
            msg = body[:300]
        return {'success': False, 'error': f"API error {e.code}: {msg}",
                'summary': '', 'stats': stats, 'provider': provider}
    except Exception as ex:
        return {'success': False, 'error': str(ex),
                'summary': '', 'stats': stats, 'provider': provider}