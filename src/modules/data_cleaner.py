import re
import string
import pandas as pd
from typing import List, Tuple, Dict, Set
import nltk
from nltk.corpus import stopwords

# Safe one-time NLTK initialization
def _ensure_nltk_resources():
    for res in ['tokenizers/punkt', 'tokenizers/punkt_tab', 'corpora/stopwords']:
        try:
            nltk.data.find(res)
        except Exception:
            try:
                name = res.split('/')[-1]
                nltk.download(name, quiet=True)
            except Exception:
                pass

_ensure_nltk_resources()


class DataCleaner:
    """
    Clean and preprocess WhatsApp chat messages with high performance.
    Optimized for large files (100,000+ messages) with vectorized operations.
    """

    # Pre-compiled regex patterns for speed
    _URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
    _EMAIL_RE = re.compile(r'\S+@\S+')
    _PHONE_RE = re.compile(r'\+?[\d\s\-\(\)]{7,}')
    _HTML_RE = re.compile(r'<.*?>')
    _SPACES_RE = re.compile(r'\s+')
    _TOKEN_RE = re.compile(r'\b[a-zA-Z]{2,}\b')

    # Comprehensive Emoji regex
    _EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf\u23e9\u231a\ufe0f\u3030"
        "]+"
    )

    def __init__(self):
        try:
            self.stop_words: Set[str] = set(stopwords.words('english'))
        except Exception:
            self.stop_words = {
                'the','a','an','and','or','but','in','on','at','to','for','of',
                'is','it','this','that','i','you','me','my','your','we','he',
                'she','they','was','are','be','have','has','had','do','did'
            }

        # Add common Spanish and Hinglish stopwords if available
        try:
            self.stop_words.update(stopwords.words('spanish'))
        except Exception:
            pass

        _hinglish_stop = {
            'hai','nhi','bhi','ko','ka','ki','ke','se','kya','ek','aur',
            'na','hi','ho','mein','hain','toh','par','ye','wo','woh','koi',
            'kuch','ab','abb','ok','okay','haan','nahi','bas','mai','main',
            'yaar','bhai','tha','thi','kar','raha','rahi'
        }
        self.stop_words.update(_hinglish_stop)

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean entire dataframe with optimized vectorization and batch operations.
        """
        if df.empty:
            return pd.DataFrame(columns=[
                'datetime', 'user', 'message', 'is_media', 'is_system',
                'emojis', 'message_cleaned', 'tokens', 'word_count',
                'urls', 'message_length', 'message_lower'
            ])

        # Filter out system and media messages if columns exist
        mask = pd.Series(True, index=df.index)
        if 'is_system' in df.columns:
            mask &= (~df['is_system'].fillna(False))
        if 'is_media' in df.columns:
            mask &= (~df['is_media'].fillna(False))
        df_clean = df[mask].copy()

        if df_clean.empty:
            df_clean['emojis'] = ''
            df_clean['message_cleaned'] = ''
            df_clean['tokens'] = [[] for _ in range(len(df_clean))]
            df_clean['word_count'] = 0
            df_clean['urls'] = [[] for _ in range(len(df_clean))]
            df_clean['message_length'] = 0
            df_clean['message_lower'] = ''
            return df_clean

        raw_messages = df_clean['message'].fillna('').astype(str).tolist()

        # Batch compute features for high speed
        emojis_list = [self._extract_emojis(m) for m in raw_messages]
        cleaned_list = [self._clean_text(m) for m in raw_messages]
        tokens_list = [self._fast_tokenize(c) for c in cleaned_list]
        urls_list = [self._extract_urls(m) for m in raw_messages]

        df_clean['emojis'] = emojis_list
        df_clean['message_cleaned'] = cleaned_list
        df_clean['tokens'] = tokens_list
        df_clean['word_count'] = [len(t) for t in tokens_list]
        df_clean['urls'] = urls_list
        df_clean['message_length'] = df_clean['message'].str.len().fillna(0).astype(int)
        df_clean['message_lower'] = df_clean['message'].str.lower().fillna('')

        return df_clean

    def _clean_text(self, text: str) -> str:
        """Fast regex-based text cleaning."""
        if not text:
            return ""
        text = self._URL_RE.sub('', text)
        text = self._EMAIL_RE.sub('', text)
        text = self._PHONE_RE.sub('', text)
        text = self._EMOJI_PATTERN.sub('', text)
        text = self._HTML_RE.sub('', text)
        text = self._SPACES_RE.sub(' ', text).strip().lower()
        return text

    def _extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from text."""
        if not text:
            return []
        return self._URL_RE.findall(text)

    def _extract_emojis(self, text: str) -> str:
        """Extract emojis from text."""
        if not text:
            return ""
        matches = self._EMOJI_PATTERN.findall(text)
        return ''.join(matches)

    def _remove_emoji(self, text: str) -> str:
        """Remove emojis from text."""
        if not text:
            return ""
        return self._EMOJI_PATTERN.sub('', text)

    def _fast_tokenize(self, text: str) -> List[str]:
        """Fast regex tokenization with stopword filtering."""
        if not text:
            return []
        words = self._TOKEN_RE.findall(text)
        stop = self.stop_words
        return [w for w in words if w not in stop and len(w) > 1]

    def _tokenize(self, text: str) -> List[str]:
        """Backward-compatible tokenization."""
        return self._fast_tokenize(text)

    def get_vocabulary(self, df: pd.DataFrame) -> dict:
        """Get frequency of all tokens."""
        from collections import Counter
        all_tokens = []
        for tokens in df.get('tokens', []):
            if isinstance(tokens, list):
                all_tokens.extend(tokens)
        return dict(Counter(all_tokens).most_common(100))

    def get_summary(self, df: pd.DataFrame) -> dict:
        """Get cleaning summary statistics."""
        if df.empty:
            return {
                'total_messages': 0, 'avg_message_length': 0,
                'avg_word_count': 0, 'total_urls': 0,
                'messages_with_emoji': 0, 'total_emoji_count': 0
            }

        msg_len = df['message_length'].mean() if 'message_length' in df.columns else 0
        word_cnt = df['word_count'].mean() if 'word_count' in df.columns else 0
        urls_cnt = sum(len(u) for u in df['urls']) if 'urls' in df.columns else 0
        emojis_ser = df['emojis'] if 'emojis' in df.columns else pd.Series([])

        return {
            'total_messages': len(df),
            'avg_message_length': round(float(msg_len), 2),
            'avg_word_count': round(float(word_cnt), 2),
            'total_urls': int(urls_cnt),
            'messages_with_emoji': int((emojis_ser != '').sum()),
            'total_emoji_count': int(emojis_ser.str.len().sum())
        }

