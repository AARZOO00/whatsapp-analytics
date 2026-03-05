import re
import string
import pandas as pd
from typing import List, Tuple, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import unicodedata

class DataCleaner:
    """
    Clean and preprocess WhatsApp chat messages.
    Remove URLs, emoji, stopwords, etc.
    """

    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        self.stop_words = set(stopwords.words('english'))
        try:
            self.stop_words.update(stopwords.words('spanish'))
        except:
            pass
        try:
            self.stop_words.update(stopwords.words('hindi'))
        except:
            pass

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean entire dataframe.

        Args:
            df: DataFrame with messages

        Returns:
            Cleaned DataFrame with additional columns
        """
        df = df.copy()

        df_clean = df[~df['is_system'] & ~df['is_media']].copy()

        df_clean['emojis'] = df_clean['message'].apply(self._extract_emojis)

        df_clean['message_cleaned'] = df_clean['message'].apply(self._clean_text)

        df_clean['tokens'] = df_clean['message_cleaned'].apply(self._tokenize)

        df_clean['word_count'] = df_clean['tokens'].apply(len)

        df_clean['urls'] = df_clean['message'].apply(self._extract_urls)

        df_clean['message_length'] = df_clean['message'].apply(len)

        df_clean['message_lower'] = df_clean['message'].apply(lambda x: x.lower())

        return df_clean

    def _clean_text(self, text: str) -> str:
        """Core cleaning function"""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\+?[\d\s\-\(\)]{7,}', '', text)
        text = self._remove_emoji(text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        return text

    def _extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from text"""
        urls = re.findall(r'http\S+|www\S+|https\S+', text)
        return urls

    def _extract_emojis(self, text: str) -> str:
        """Extract emojis from text"""
        emoji_pattern = re.compile(
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
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+"
        )
        return ''.join(emoji_pattern.findall(text))

    def _remove_emoji(self, text: str) -> str:
        """Remove emojis from text"""
        emoji_pattern = re.compile(
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
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+"
        )
        return emoji_pattern.sub(r'', text)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text and remove stopwords"""
        if not text or not isinstance(text, str):
            return []

        try:
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
            return tokens
        except:
            return []

    def get_vocabulary(self, df: pd.DataFrame) -> dict:
        """Get frequency of all tokens"""
        from collections import Counter

        all_tokens = []
        for tokens in df['tokens']:
            all_tokens.extend(tokens)

        return dict(Counter(all_tokens).most_common(100))

    def get_summary(self, df: pd.DataFrame) -> dict:
        """Get cleaning summary statistics"""
        return {
            'total_messages': len(df),
            'avg_message_length': df['message_length'].mean(),
            'avg_word_count': df['word_count'].mean(),
            'total_urls': df['urls'].apply(len).sum(),
            'messages_with_emoji': (df['emojis'] != '').sum(),
            'total_emoji_count': df['emojis'].apply(len).sum()
        }
