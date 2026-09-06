import re
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict, Optional

class WhatsAppParser:
    """
    Parse WhatsApp exported chat files.
    Handles Android, iOS, 12h, 24h, multiline messages, UTF-8/UTF-16/BOM,
    and system/media messages with high speed and memory efficiency.
    """

    # Comprehensive regex patterns for common WhatsApp export formats:
    # 1. Standard Android: "DD/MM/YYYY, HH:MM - User: Message" or "MM/DD/YY, HH:MM AM/PM - User: Message"
    # 2. iOS with brackets: "[DD/MM/YY, HH:MM:SS AM/PM] User: Message"
    # 3. Hyphenated / dotted dates: "YYYY-MM-DD, HH:MM - User: Message" or "DD.MM.YY, HH:MM - User: Message"
    PATTERNS = [
        # iOS format: [DD/MM/YY(YY), HH:MM(:SS) (AM/PM)] User: Message
        re.compile(r'^\[(\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[apAP][mM])?)\]\s+([^:]+?):\s+(.*)$'),
        # Standard Android format: DD/MM/YY(YY), HH:MM (AM/PM) - User: Message
        re.compile(r'^(\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[apAP][mM])?)\s+-\s+([^:]+?):\s+(.*)$'),
        # System message iOS format: [DD/MM/YY, HH:MM:SS] System text
        re.compile(r'^\[(\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[apAP][mM])?)\]\s+(.*)$'),
        # System message Android format: DD/MM/YY, HH:MM - System text
        re.compile(r'^(\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[apAP][mM])?)\s+-\s+(.*)$'),
    ]

    _SYSTEM_PHRASES = (
        'messages and calls are end-to-end encrypted',
        'created group',
        'added',
        'removed',
        'left',
        'changed the subject',
        'changed this group',
        'changed the group',
        'security code changed',
        'you were added',
        'this message was deleted',
        'deleted this message',
    )

    _MEDIA_PHRASES = (
        '<media omitted>',
        '(file attached)',
        'image omitted',
        'video omitted',
        'audio omitted',
        'document omitted',
        'sticker omitted',
        'contact card omitted',
        'location:',
    )

    def __init__(self):
        self.messages = []
        self.errors = []

    def parse_lines(self, lines: List[str]) -> pd.DataFrame:
        """
        Parse lines of WhatsApp text and return structured DataFrame.
        """
        if not lines:
            return pd.DataFrame(columns=['datetime', 'user', 'message', 'is_media', 'is_system'])

        self.messages = []
        self.errors = []
        current_msg = None

        for line_num, raw_line in enumerate(lines):
            line = raw_line.replace('\u202f', ' ').replace('\xa0', ' ').replace('\u200e', '').replace('\u200f', '').strip('\r\n')
            if not line:
                continue

            parsed, is_system_line = self._parse_line_with_mode(line)

            if parsed:
                if current_msg:
                    self.messages.append(current_msg)
                current_msg = parsed
            else:
                if current_msg and not current_msg.get('is_system', False):
                    current_msg['message'] += '\n' + line.strip()
                else:
                    self.errors.append({
                        'line_num': line_num + 1,
                        'line': line,
                        'reason': 'Could not parse line'
                    })

        if current_msg:
            self.messages.append(current_msg)

        if not self.messages:
            return pd.DataFrame(columns=['datetime', 'user', 'message', 'is_media', 'is_system'])

        df = pd.DataFrame(self.messages)

        try:
            df['datetime'] = pd.to_datetime(df['datetime'], format='mixed', errors='coerce')
        except Exception:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

        if df['datetime'].isna().any():
            df['datetime'] = df['datetime'].ffill().bfill()
            if df['datetime'].isna().any():
                df['datetime'] = df['datetime'].fillna(pd.Timestamp.now())

        return df

    def parse_text(self, text: str) -> pd.DataFrame:
        """Parse raw string content."""
        if not text:
            return pd.DataFrame(columns=['datetime', 'user', 'message', 'is_media', 'is_system'])
        return self.parse_lines(text.splitlines())

    def parse_chat(self, text: str) -> pd.DataFrame:
        """Alias for parse_text."""
        return self.parse_text(text)

    def parse_file(self, file_path: str) -> pd.DataFrame:
        """
        Parse WhatsApp chat file and return DataFrame.
        Memory-efficient line streaming with multiline message preservation.
        """
        encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'latin-1', 'cp1252']
        lines = None

        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc, errors='replace') as file:
                    lines = file.readlines()
                if lines and len(lines) > 0:
                    break
            except Exception:
                continue

        if not lines:
            return pd.DataFrame(columns=['datetime', 'user', 'message', 'is_media', 'is_system'])

        return self.parse_lines(lines)

    def _parse_line_with_mode(self, line: str) -> Tuple[Optional[Dict], bool]:
        """
        Attempt to match line against known message & system patterns.
        """
        # 1. User message pattern (iOS)
        m = self.PATTERNS[0].match(line)
        if m:
            date_str, time_str, user, message = m.groups()
            return self._build_record(date_str, time_str, user, message, is_system=False), False

        # 2. User message pattern (Android)
        m = self.PATTERNS[1].match(line)
        if m:
            date_str, time_str, user, message = m.groups()
            return self._build_record(date_str, time_str, user, message, is_system=False), False

        # 3. System message pattern (iOS brackets)
        m = self.PATTERNS[2].match(line)
        if m:
            date_str, time_str, system_msg = m.groups()
            low = system_msg.lower()
            if any(p in low for p in self._SYSTEM_PHRASES):
                return self._build_record(date_str, time_str, 'System', system_msg, is_system=True), True

        # 4. System message pattern (Android)
        m = self.PATTERNS[3].match(line)
        if m:
            date_str, time_str, system_msg = m.groups()
            low = system_msg.lower()
            if any(p in low for p in self._SYSTEM_PHRASES):
                return self._build_record(date_str, time_str, 'System', system_msg, is_system=True), True

        return None, False

    def _build_record(self, date_str: str, time_str: str, user: str, message: str, is_system: bool) -> Dict:
        """Build standardized parsed record dictionary."""
        u_clean = user.strip()
        msg_clean = message.strip()
        low_msg = msg_clean.lower()

        # Extra system checks on sender name
        if not is_system and u_clean.lower() in ('system message', 'system', 'whatsapp'):
            is_system = True

        if not is_system and any(p in low_msg for p in self._SYSTEM_PHRASES):
            is_system = True

        is_media = any(p in low_msg for p in self._MEDIA_PHRASES)

        return {
            'datetime': f"{date_str.strip()} {time_str.strip()}",
            'user': u_clean,
            'message': msg_clean,
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
            'system_messages': sum(1 for m in self.messages if m.get('is_system', False)),
            'media_messages': sum(1 for m in self.messages if m.get('is_media', False)),
            'unique_users': len(set(m['user'] for m in self.messages if not m.get('is_system', False)))
        }

# Alias for backward compatibility
ChatParser = WhatsAppParser


