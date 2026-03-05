import re
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict

class WhatsAppParser:
    """
    Parse WhatsApp exported chat files.
    Handles both individual and group chats.
    """

    MESSAGE_PATTERN = r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?)\s*-?\s+(.+?):\s+(.*)$'

    def __init__(self):
        self.messages = []
        self.errors = []

    def parse_file(self, file_path: str) -> pd.DataFrame:
        """
        Parse WhatsApp chat file and return DataFrame.

        Args:
            file_path: Path to WhatsApp exported .txt file

        Returns:
            DataFrame with columns: datetime, user, message, is_media, is_system
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                lines = file.readlines()

        self.messages = []

        for line_num, line in enumerate(lines):
            line = line.strip()

            if not line:
                continue

            parsed = self._parse_line(line)
            if parsed:
                self.messages.append(parsed)
            else:
                self.errors.append({
                    'line_num': line_num + 1,
                    'line': line,
                    'reason': 'Could not parse line'
                })

        df = pd.DataFrame(self.messages)

        df['datetime'] = pd.to_datetime(df['datetime'])

        return df

    def _parse_line(self, line: str) -> Dict:
        """
        Parse a single line from WhatsApp chat.

        Returns:
            Dict with parsed data or None if line doesn't match pattern
        """
        match = re.match(self.MESSAGE_PATTERN, line)

        if not match:
            return None

        date_str, time_str, user, message = match.groups()

        is_system = user.lower() in ['system message', 'messages and calls are encrypted']

        is_media = '<media omitted>' in message.lower()

        datetime_str = f"{date_str} {time_str}"

        return {
            'datetime': datetime_str,
            'user': user.strip(),
            'message': message.strip(),
            'is_media': is_media,
            'is_system': is_system
        }

    def get_errors(self) -> List[Dict]:
        """Return parsing errors for debugging"""
        return self.errors

    def get_summary(self) -> Dict:
        """Get parsing summary statistics"""
        return {
            'total_lines': len(self.messages),
            'system_messages': sum(1 for m in self.messages if m['is_system']),
            'media_messages': sum(1 for m in self.messages if m['is_media']),
            'unique_users': len(set(m['user'] for m in self.messages if not m['is_system']))
        }
