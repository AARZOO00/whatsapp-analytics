import pandas as pd
from typing import Dict, List
from collections import Counter


class SummaryGenerator:
    """Generate rich, human-readable chat summaries and insights."""

    _STOPWORDS = {
        'the','a','an','is','are','was','were','be','been','have','has','had',
        'do','does','did','will','would','could','should','to','of','in','on',
        'at','by','for','with','and','or','but','if','as','it','its','this',
        'that','i','me','my','we','our','you','your','he','his','she','her',
        'they','them','their','im','ive','its','id','ok','okay','hi','hey',
        'yes','no','yea','yeah','nah','haha','lol','hmm','deleted','message',
        'null','media','omitted','https','http','pm','am',
    }

    def __init__(self, df: pd.DataFrame):
        self.df  = df.copy()
        self._ok = len(df) > 0

    # ─────────────────────────────────────────────────────────────────────────
    def generate_summary(self) -> Dict:
        if not self._ok:
            return {
                'conversation_summary':    'No messages to analyze yet. Upload a chat file to get started.',
                'detailed_narrative':      '',
                'most_discussed_topics':   [],
                'overall_mood':            {'positive': 0, 'negative': 0, 'neutral': 0, 'overall_label': 'neutral'},
                'conflict_detection':      {'conflict_score': 0, 'negative_messages': 0, 'toxic_messages': 0, 'risk_level': 'Low', 'most_negative_user': 'N/A'},
                'most_active_period':      {'most_active_hour': '00:00', 'most_active_date': 'N/A', 'avg_messages_per_hour': 0, 'peak_activity': 0},
                'key_insights':            ['Upload a WhatsApp chat file to see AI-powered insights here.'],
                'user_engagement':         {},
                'conversation_highlights': [],
                'relationship_dynamics':   {'top_contributors': {}, 'least_active': {}, 'participation_gini': 0, 'balance_label': 'No data'},
            }
        return {
            'conversation_summary':    self._summary(),
            'detailed_narrative':      self._narrative(),
            'most_discussed_topics':   self._top_topics(),
            'overall_mood':            self._mood(),
            'conflict_detection':      self._conflicts(),
            'most_active_period':      self._active_period(),
            'key_insights':            self._insights(),
            'user_engagement':         self._engagement(),
            'conversation_highlights': self._highlights(),
            'relationship_dynamics':   self._dynamics(),
        }

    # ── helpers ───────────────────────────────────────────────────────────────
    def _safe_top_user(self):
        vc = self.df['user'].value_counts()
        if len(vc) == 0:
            return 'Unknown', 0
        return vc.index[0], int(vc.iloc[0])

    def _tokens(self, n: int = 10) -> List[str]:
        all_tok = []
        for tok in self.df['tokens']:
            if isinstance(tok, list):
                all_tok.extend([t.lower() for t in tok if t.isalpha() and len(t) > 2])
        filtered = [t for t in all_tok if t not in self._STOPWORDS]
        return [w for w, _ in Counter(filtered).most_common(n)]

    def _label(self, score: float) -> str:
        if score >= 0.05:  return 'positive'
        if score <= -0.05: return 'negative'
        return 'neutral'

    def _period(self, h: int) -> str:
        if 5 <= h < 12:  return 'morning'
        if 12 <= h < 17: return 'afternoon'
        if 17 <= h < 21: return 'evening'
        return 'late-night'

    # ── core sections ─────────────────────────────────────────────────────────
    def _summary(self) -> str:
        total   = len(self.df)
        users   = self.df['user'].nunique()
        days    = max((self.df['datetime'].max() - self.df['datetime'].min()).days, 1)
        avg_day = round(total / days, 1)

        top_user, top_cnt = self._safe_top_user()
        top_pct = round(top_cnt / total * 100, 1)

        avg_sent  = self.df['sentiment_compound'].mean()
        mood      = self._label(avg_sent)
        mood_desc = {'positive': 'generally positive and friendly',
                     'neutral':  'mostly neutral and informational',
                     'negative': 'somewhat tense in tone'}.get(mood, 'mixed')

        pos_pct = round((self.df['sentiment_vader'] == 'POSITIVE').sum() / total * 100, 1)
        neg_pct = round((self.df['sentiment_vader'] == 'NEGATIVE').sum() / total * 100, 1)
        tox_pct = round((self.df['is_toxic'] == True).sum() / total * 100, 1)

        dom_emotion = 'not detected'
        if 'emotion' in self.df.columns:
            em = self.df['emotion'].dropna().value_counts()
            if len(em):
                dom_emotion = em.index[0]

        topics = self._tokens(4)
        topics_str = ', '.join(f'"{t}"' for t in topics) if topics else 'general topics'

        tox_note = (f'Toxicity is minimal ({tox_pct}%).' if tox_pct < 5
                    else f'⚠️ Notable toxicity detected at {tox_pct}% of messages.')

        return (
            f"This group chat spans {days} days with {total:,} messages from {users} participants "
            f"— averaging {avg_day} messages/day. The overall tone is {mood_desc} "
            f"({pos_pct}% positive, {neg_pct}% negative). "
            f"Frequent themes include {topics_str}. "
            f"{top_user} leads participation at {top_pct}% of all messages. "
            f"Dominant emotion: {dom_emotion}. {tox_note}"
        )

    def _narrative(self) -> str:
        total = len(self.df)
        days  = max((self.df['datetime'].max() - self.df['datetime'].min()).days, 1)

        self.df['hour'] = self.df['datetime'].dt.hour
        hr_grp = self.df.groupby('hour').size()
        peak   = int(hr_grp.idxmax()) if len(hr_grp) > 0 else 12
        period = self._period(peak)

        top3     = self.df['user'].value_counts().head(3)
        top3_str = ', '.join(
            f"{u} ({round(c/total*100,0):.0f}%)" for u, c in top3.items()
        ) if len(top3) > 0 else 'Unknown'

        self.df['date'] = self.df['datetime'].dt.date
        ds = self.df.groupby('date')['sentiment_compound'].mean()
        if len(ds) >= 4:
            fh = ds.iloc[:len(ds)//2].mean()
            sh = ds.iloc[len(ds)//2:].mean()
            arc = ('sentiment improved over time — conversations grew more positive' if sh > fh + 0.05
                   else 'the tone gradually became more subdued' if sh < fh - 0.05
                   else 'the emotional tone remained consistent throughout')
        else:
            arc = 'the conversation was relatively brief'

        avg_len = self.df['message_length'].mean()
        style   = ('long detailed messages — deep discussions or planning' if avg_len > 100
                   else 'very short messages — quick reactions or casual banter' if avg_len < 25
                   else 'medium-length messages — casual but engaged conversation')

        return (
            f"The chat is most active during {period} hours (around {peak:02d}:00). "
            f"Top contributors: {top3_str}. "
            f"Overall, {arc}. "
            f"Messages tend to be {style}."
        )

    def _top_topics(self, n: int = 8) -> List[str]:
        t = self._tokens(n)
        return t if t else ['No clear topics identified']

    def _mood(self) -> Dict:
        counts = self.df['sentiment_vader'].value_counts()
        total  = max(len(self.df), 1)
        return {
            'positive':     round(counts.get('POSITIVE', 0) / total * 100, 1),
            'negative':     round(counts.get('NEGATIVE', 0) / total * 100, 1),
            'neutral':      round(counts.get('NEUTRAL',  0) / total * 100, 1),
            'overall_label': self._label(self.df['sentiment_compound'].mean()),
        }

    def _conflicts(self) -> Dict:
        neg   = self.df[self.df['sentiment_vader'] == 'NEGATIVE']
        toxic = self.df[self.df['is_toxic'] == True]
        total = max(len(self.df), 1)
        score = (len(neg) + len(toxic) * 2) / total * 100
        neg_u = neg.groupby('user').size()
        worst = neg_u.idxmax() if len(neg_u) > 0 else 'N/A'
        risk  = 'Low' if score < 5 else 'Medium' if score < 15 else 'High'
        return {
            'conflict_score':    round(score, 1),
            'negative_messages': len(neg),
            'toxic_messages':    len(toxic),
            'risk_level':        risk,
            'most_negative_user': worst,
        }

    def _active_period(self) -> Dict:
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['date'] = self.df['datetime'].dt.date
        hr = self.df.groupby('hour').size()
        dy = self.df.groupby('date').size()
        ph = int(hr.idxmax()) if len(hr) > 0 else 0
        pd_ = str(dy.idxmax()) if len(dy) > 0 else 'N/A'
        return {
            'most_active_hour':      f'{ph:02d}:00',
            'most_active_date':      pd_,
            'avg_messages_per_hour': round(len(self.df) / 24, 1),
            'peak_activity':         int(hr.max()) if len(hr) > 0 else 0,
        }

    def _insights(self) -> List[str]:
        out   = []
        df    = self.df
        total = max(len(df), 1)

        pos_pct = (df['sentiment_vader'] == 'POSITIVE').sum() / total * 100
        neg_pct = (df['sentiment_vader'] == 'NEGATIVE').sum() / total * 100
        if pos_pct > 60:
            out.append(f'💬 Very positive group — {pos_pct:.0f}% of messages carry an upbeat tone.')
        elif neg_pct > 25:
            out.append(f'⚠️ High negativity detected ({neg_pct:.0f}%). Consider reviewing conversation health.')

        top_user, top_cnt = self._safe_top_user()
        out.append(f'👑 {top_user} is the most active member — {top_cnt} messages ({round(top_cnt/total*100,0):.0f}% of chat).')

        vc = df['user'].value_counts()
        silent = int((vc < total * 0.01).sum())
        if silent:
            out.append(f'🔇 {silent} participant(s) contribute under 1% of messages.')

        avg_len = df['message_length'].mean()
        if avg_len > 100:
            out.append('📝 Long detailed messages — suggests in-depth discussions or planning.')
        elif avg_len < 25:
            out.append('⚡ Very short messages — fast-paced chat or quick reactions.')

        tox_pct = (df['is_toxic'] == True).sum() / total * 100
        if tox_pct > 10:
            out.append(f'🚨 {tox_pct:.1f}% toxic content — above healthy levels.')
        elif tox_pct == 0:
            out.append('✅ Zero toxic messages — healthy conversation environment.')

        if 'emotion' in df.columns:
            em = df['emotion'].dropna().value_counts()
            if len(em):
                out.append(f'🎭 Dominant emotion: {em.index[0].title()} ({round(em.iloc[0]/em.sum()*100,1)}% of messages).')

        df['hour'] = df['datetime'].dt.hour
        hr_grp = df.groupby('hour').size()
        if len(hr_grp) > 0:
            peak = int(hr_grp.idxmax())
            out.append(f'🕐 Most active time: {peak:02d}:00 — {self._period(peak)} hours.')

        days = max((df['datetime'].max() - df['datetime'].min()).days, 1)
        out.append(f'📅 Chat spans {days} days with {round(len(df)/days,1)} messages/day on average.')

        return out

    def _highlights(self) -> List[Dict]:
        out = []
        sc  = self.df['sentiment_compound']
        msg_col = 'message_cleaned' if 'message_cleaned' in self.df.columns else 'message'

        if len(sc) > 0:
            for idx, label in [(sc.idxmax(), '😊 Most Positive'), (sc.idxmin(), '😠 Most Negative')]:
                row = self.df.loc[idx]
                out.append({
                    'type':    label,
                    'user':    row['user'],
                    'message': str(row.get(msg_col, ''))[:120],
                    'score':   round(row['sentiment_compound'], 3),
                })

        ml = self.df['message_length']
        if len(ml) > 0:
            row = self.df.loc[ml.idxmax()]
            out.append({
                'type':    '📝 Longest Message',
                'user':    row['user'],
                'message': str(row.get(msg_col, ''))[:120] + '…',
                'score':   int(row['message_length']),
            })
        return out

    def _dynamics(self) -> Dict:
        vc    = self.df['user'].value_counts()
        total = max(len(self.df), 1)
        shares = (vc / total).values
        gini  = round(float(sum(abs(a - b) for a in shares for b in shares) / max(2 * len(shares)**2, 1)), 3)
        return {
            'top_contributors':   {u: int(c) for u, c in vc.head(3).items()},
            'least_active':       {u: int(c) for u, c in vc.tail(3).items()},
            'participation_gini': gini,
            'balance_label':      'Dominated by a few' if gini > 0.4 else 'Fairly balanced',
        }

    def _engagement(self) -> Dict:
        out = {}
        for user in self.df['user'].unique():
            udf = self.df[self.df['user'] == user]
            out[user] = {
                'messages':      len(udf),
                'avg_length':    round(udf['message_length'].mean(), 1),
                'sentiment':     round(udf['sentiment_compound'].mean(), 2),
                'participation': round(len(udf) / max(len(self.df), 1) * 100, 1),
            }
        return out