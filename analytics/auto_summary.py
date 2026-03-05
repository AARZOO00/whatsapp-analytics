"""
auto_summary.py — Smart AI-like chat summary WITHOUT any API key
Uses data patterns, NLP heuristics, and smart templates to generate
natural, insightful summaries that feel like real AI output.

Place in: whatsapp-analyzer/analytics/auto_summary.py
"""

import re
import random
from collections import Counter
from datetime import datetime
import pandas as pd
from typing import Dict, List


# ── Stopwords (English + Hinglish) ───────────────────────────────────────
_STOP = {
    'the','a','an','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','to','of','in','on','at','by','for','with','and',
    'or','but','if','it','its','i','me','my','we','you','he','she','they',
    'them','our','your','this','that','these','those','so','just','not','no',
    'yes','ok','okay','hi','hey','hello','bye','lol','haha','hehe','lmao',
    'omg','wow','oh','ah','uh','um','like','get','got','go','going','come',
    'came','also','very','too','more','some','all','one','two','three','even',
    'than','then','when','what','how','why','who','which','where','will','can',
    'could','would','should','may','might','must','shall','let','know','think',
    # Hinglish
    'hai','hain','ho','tha','the','thi','bhi','kya','aur','mein','ka','ke',
    'ki','ko','se','nhi','nah','haan','han','par','pe','woh','wo','yeh','ye',
    'ab','bs','bas','toh','to','na','ne','ek','koi','sab','kuch','kaise',
    'kyun','kyunki','lekin','matlab','phir','fir','agar','jab','tab',
}

_HOUR_LABELS = {
    0:'midnight',1:'late night',2:'late night',3:'late night',4:'early morning',
    5:'early morning',6:'morning',7:'morning',8:'morning',9:'mid-morning',
    10:'mid-morning',11:'late morning',12:'noon',13:'afternoon',14:'afternoon',
    15:'afternoon',16:'late afternoon',17:'evening',18:'evening',19:'evening',
    20:'night',21:'night',22:'night',23:'night',
}

_DAY_NAMES = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

# ── Topic clusters ────────────────────────────────────────────────────────
_TOPIC_CLUSTERS = {
    'studies & academics': ['study','studies','exam','test','class','lecture','assignment',
                            'homework','project','marks','grade','college','school','university',
                            'padhai','padhna','result','semester','notes','subject'],
    'food & eating':       ['food','eat','eating','lunch','dinner','breakfast','snack','hungry',
                            'pizza','biryani','chai','coffee','tea','restaurant','khana','bhojan'],
    'plans & meetups':     ['meet','meeting','plan','plans','party','outing','hangout','come',
                            'milna','aana','jaana','trip','travel','visit','invite','call'],
    'cricket & sports':    ['cricket','match','score','team','player','ipl','fifa','game',
                            'sports','khel','win','won','lost','batting','bowling'],
    'movies & shows':      ['movie','film','series','episode','netflix','watch','show','ott',
                            'cinema','trailer','review','dekhna','dekho'],
    'work & jobs':         ['work','job','office','meeting','deadline','project','client',
                            'salary','interview','internship','resume','hr','manager'],
    'health & wellness':   ['health','doctor','medicine','hospital','sick','fever','headache',
                            'exercise','gym','yoga','fit','bimar','dawai'],
    'tech & gadgets':      ['phone','mobile','laptop','app','internet','wifi','download',
                            'install','update','android','iphone','pc','computer'],
    'relationships':       ['love','family','friend','bhai','dost','yaar','papa','mama',
                            'girlfriend','boyfriend','bro','sis','sister','brother'],
    'money & finance':     ['money','pay','payment','upi','gpay','paytm','loan','paise','rupee'],
}


# ══════════════════════════════════════════════════════════════════════════
#  Core analysis functions
# ══════════════════════════════════════════════════════════════════════════

def _extract_stats(df: pd.DataFrame) -> Dict:
    """Extract all key stats in one pass."""
    total     = len(df)
    users     = df['user'].value_counts()
    days      = max((df['datetime'].max() - df['datetime'].min()).days, 1)
    df2       = df.copy()
    df2['hour']    = df2['datetime'].dt.hour
    df2['weekday'] = df2['datetime'].dt.weekday
    df2['date']    = df2['datetime'].dt.date

    hour_dist  = df2.groupby('hour').size()
    peak_hour  = int(hour_dist.idxmax()) if len(hour_dist) else 20
    peak_day   = int(df2.groupby('weekday').size().idxmax())
    daily_msgs = df2.groupby('date').size()
    avg_daily  = round(total / days, 1)
    max_day    = daily_msgs.idxmax()

    # Sentiment
    pos_pct = round((df['sentiment_vader'] == 'POSITIVE').sum() / max(total,1) * 100, 1)
    neg_pct = round((df['sentiment_vader'] == 'NEGATIVE').sum() / max(total,1) * 100, 1)
    neu_pct = round(100 - pos_pct - neg_pct, 1)

    # Emotion
    dom_emotion = 'neutral'
    if 'emotion' in df.columns:
        em_vc = df['emotion'].dropna().value_counts()
        if len(em_vc): dom_emotion = em_vc.index[0]

    # Toxicity
    toxic_pct = 0
    if 'is_toxic' in df.columns:
        toxic_pct = round((df['is_toxic'] == True).sum() / max(total,1) * 100, 1)

    # Avg message length
    avg_len = round(df.get('message_length', pd.Series([20]*total)).mean(), 1)

    # Media/URL count
    mc = 'message_cleaned' if 'message_cleaned' in df.columns else 'message'
    url_count = df[mc].apply(lambda x: bool(re.search(r'https?://', str(x)))).sum()

    # Top words
    all_toks = []
    for toks in df.get('tokens', pd.Series([[]] * total)):
        if isinstance(toks, list):
            all_toks.extend([t.lower() for t in toks if t.isalpha() and len(t) > 2])
    top_words = [w for w, _ in Counter(
        t for t in all_toks if t not in _STOP
    ).most_common(20)]

    # Late night activity (11pm - 3am)
    late_night_pct = round(
        df2[df2['hour'].isin([23,0,1,2])].shape[0] / max(total,1) * 100, 1
    )

    # Consecutive days (streaks)
    all_dates = sorted(daily_msgs.index)
    max_streak = cur = 0
    for i, d in enumerate(all_dates):
        if i == 0: cur = 1
        else:
            delta = (pd.Timestamp(d) - pd.Timestamp(all_dates[i-1])).days
            cur   = cur + 1 if delta == 1 else 1
        max_streak = max(max_streak, cur)

    # Ghost members
    ghost_count = int((users / total * 100 < 1.0).sum())

    return {
        'total':         total,
        'users':         users,
        'n_users':       len(users),
        'days':          days,
        'avg_daily':     avg_daily,
        'peak_hour':     peak_hour,
        'peak_day':      peak_day,
        'peak_day_name': _DAY_NAMES[peak_day],
        'max_day':       str(max_day),
        'max_day_msgs':  int(daily_msgs.max()),
        'pos_pct':       pos_pct,
        'neg_pct':       neg_pct,
        'neu_pct':       neu_pct,
        'dom_emotion':   dom_emotion,
        'toxic_pct':     toxic_pct,
        'avg_len':       avg_len,
        'url_count':     int(url_count),
        'top_words':     top_words,
        'late_night_pct':late_night_pct,
        'max_streak':    max_streak,
        'ghost_count':   ghost_count,
        'date_start':    df['datetime'].min().strftime('%d %b %Y'),
        'date_end':      df['datetime'].max().strftime('%d %b %Y'),
        'top_user':      users.index[0] if len(users) else 'Unknown',
        'top_user_pct':  round(users.iloc[0] / max(total,1) * 100, 1),
        'top2_user':     users.index[1] if len(users) > 1 else None,
        'top2_pct':      round(users.iloc[1] / max(total,1) * 100, 1) if len(users) > 1 else 0,
    }


def _detect_topics(df: pd.DataFrame) -> List[str]:
    """Detect which topic clusters this chat talks about."""
    mc      = 'message_cleaned' if 'message_cleaned' in df.columns else 'message'
    all_txt = ' '.join(df[mc].fillna('').astype(str)).lower()
    scores  = {}
    for topic, keywords in _TOPIC_CLUSTERS.items():
        score = sum(all_txt.count(kw) for kw in keywords)
        if score > 0:
            scores[topic] = score
    return [t for t, _ in sorted(scores.items(), key=lambda x: -x[1])[:4]]


def _group_type(s: Dict) -> str:
    """Guess the kind of group."""
    words = ' '.join(s['top_words'][:15]).lower()
    topics = _detect_topics
    if s['n_users'] <= 3:     return "close friends / small group"
    if s['n_users'] >= 30:    return "large community group"
    if 'class' in words or 'exam' in words or 'notes' in words:
        return "students / classmates group"
    if 'office' in words or 'meeting' in words or 'work' in words:
        return "work / office group"
    if 'bhai' in words or 'family' in words or 'mama' in words or 'papa' in words:
        return "family group"
    return "friends group"


def _vibe_sentence(s: Dict) -> str:
    """Generate a vibe description."""
    if s['pos_pct'] >= 50:
        vibe = "mostly positive and cheerful"
    elif s['neg_pct'] >= 25:
        vibe = "somewhat tense with occasional conflicts"
    elif s['neg_pct'] >= 15:
        vibe = "mixed — generally friendly but with some heated moments"
    else:
        vibe = "calm and neutral"

    if s['late_night_pct'] > 20:
        vibe += ", with a strong night-owl culture"
    if s['avg_len'] < 20:
        vibe += ". Messages are short and rapid-fire"
    elif s['avg_len'] > 80:
        vibe += ". People tend to write detailed, thoughtful messages"
    else:
        vibe += ". Conversations flow naturally"

    return vibe


def _activity_sentence(s: Dict) -> str:
    """Describe activity patterns."""
    peak_label = _HOUR_LABELS.get(s['peak_hour'], 'evening')
    parts = [
        f"The group is most active during **{peak_label}** hours (around {s['peak_hour']}:00)",
        f"with **{s['peak_day_name']}** being the busiest day of the week",
    ]
    if s['avg_daily'] > 50:
        parts.append(f"averaging a whopping **{s['avg_daily']} messages/day**")
    elif s['avg_daily'] > 10:
        parts.append(f"averaging **{s['avg_daily']} messages/day**")
    else:
        parts.append(f"averaging **{s['avg_daily']} messages/day** — fairly quiet")
    if s['max_streak'] > 7:
        parts.append(f"with a record streak of **{s['max_streak']} consecutive active days**")
    return '. '.join(parts[:3]) + '.'


def _dynamics_sentence(s: Dict) -> str:
    """Describe group dynamics."""
    top   = s['top_user']
    top_p = s['top_user_pct']
    n     = s['n_users']

    if top_p > 30:
        lead = f"**{top}** clearly dominates the conversation with {top_p}% of all messages"
    elif top_p > 20:
        lead = f"**{top}** leads with {top_p}% of messages"
    else:
        lead = f"Participation is fairly balanced — **{top}** leads with just {top_p}%"

    if s['top2_user']:
        lead += f", followed by **{s['top2_user']}** ({s['top2_pct']}%)"

    if s['ghost_count'] > 0:
        lead += f". **{s['ghost_count']} ghost member{'s' if s['ghost_count']>1 else ''}** rarely speak"

    if n > 10:
        lead += f". Out of {n} participants, a small core drives most of the chat"

    return lead + '.'


def _notable_sentence(s: Dict, topics: List[str]) -> str:
    """Notable observations."""
    parts = []

    if topics:
        topic_str = ', '.join(f'**{t}**' for t in topics[:3])
        parts.append(f"Main recurring themes: {topic_str}")

    if s['toxic_pct'] == 0:
        parts.append("Zero toxic messages — this is a healthy, respectful group 🌱")
    elif s['toxic_pct'] > 5:
        parts.append(f"⚠️ {s['toxic_pct']}% of messages flagged as toxic — some tension present")

    if s['url_count'] > 20:
        parts.append(f"**{s['url_count']} links** shared — an active content-sharing group")

    if s['late_night_pct'] > 25:
        parts.append(f"**{s['late_night_pct']}% of chats happen past midnight** — serious night owls 🦉")

    if s['max_day_msgs'] > s['avg_daily'] * 5:
        parts.append(f"Most explosive day: **{s['max_day']}** with {s['max_day_msgs']} messages")

    return ' · '.join(parts) if parts else "A fairly typical, stable group chat."


def _essence_line(s: Dict, group_type: str, topics: List[str]) -> str:
    """One punchy line capturing the soul of the chat."""
    top   = s['top_user']
    vibe  = "positive" if s['pos_pct'] > 45 else ("tense" if s['neg_pct'] > 20 else "chill")

    templates = [
        f"A {vibe} {group_type} where **{top}** keeps things alive and {topics[0] if topics else 'daily life'} dominates the feed.",
        f"**{s['n_users']} people**, {s['total']:,} messages, and one shared love for {topics[0] if topics else 'chatting'} — that's this group.",
        f"From {s['date_start']} to {s['date_end']}: a {vibe} {group_type} logging {s['avg_daily']} msgs/day.",
        f"Think: late-{_HOUR_LABELS.get(s['peak_hour'],'night')} chats, {topics[0] if topics else 'random topics'}, and **{top}** leading the pack.",
    ]
    return random.choice(templates)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def generate_auto_summary(df: pd.DataFrame) -> Dict:
    """
    Generate a smart, AI-like chat summary without any API key.
    Returns same structure as AI summary for easy UI integration.
    """
    if df is None or len(df) == 0:
        return {
            'success': False,
            'error':   'No data to summarize.',
            'summary': '',
            'stats':   {},
            'provider':'auto',
        }

    try:
        s      = _extract_stats(df)
        topics = _detect_topics(df)
        gtype  = _group_type(s)

        summary_md = f"""**1. What This Chat Is About**
This is a **{gtype}** with {s['n_users']} participants spanning **{s['days']} days** ({s['date_start']} – {s['date_end']}). With {s['total']:,} total messages, it covers {', '.join(topics[:3]) if topics else 'everyday conversations'}.

**2. Conversation Vibe**
{_vibe_sentence(s).capitalize()}. The dominant emotion is **{s['dom_emotion']}**, with {s['pos_pct']}% positive and {s['neg_pct']}% negative messages.

**3. Key Themes & Topics**
{chr(10).join(f'• {t.title()}' for t in (topics if topics else ['General conversation', 'Daily updates', 'Casual chat']))}

**4. Group Dynamics**
{_dynamics_sentence(s)}

**5. Notable Observations**
{_notable_sentence(s, topics)}

**6. ⚡ One-Line Essence**
{_essence_line(s, gtype, topics)}
"""

        return {
            'success':  True,
            'summary':  summary_md,
            'provider': 'auto',
            'stats': {
                'total_messages':   s['total'],
                'participants':     s['n_users'],
                'days':             s['days'],
                'positive_pct':     s['pos_pct'],
                'negative_pct':     s['neg_pct'],
                'dominant_emotion': s['dom_emotion'],
                'peak_hour':        s['peak_hour'],
                'top_words':        s['top_words'][:8],
                'date_range':       f"{s['date_start']} – {s['date_end']}",
            },
            'error': '',
        }

    except Exception as e:
        return {
            'success': False,
            'error':   f"Auto-summary failed: {e}",
            'summary': '',
            'stats':   {},
            'provider':'auto',
        }