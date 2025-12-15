"""
Text Preprocessor for Sentiment Analysis

This module handles text cleaning, normalization, and preprocessing
for sentiment analysis models.
"""

import re
import unicodedata
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Preprocesses text for sentiment analysis.

    Features:
    - URL removal
    - HTML tag removal
    - Emoji handling
    - Special character normalization
    - Whitespace normalization
    - Repeated character normalization

    Example:
        preprocessor = TextPreprocessor()
        clean_text = preprocessor.preprocess("This is AMAZING!!! 😍")
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        normalize_whitespace: bool = True,
        normalize_repeats: bool = True,
        max_repeat: int = 3
    ):
        """
        Initialize text preprocessor.

        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_html: Remove HTML tags
            normalize_whitespace: Normalize whitespace
            normalize_repeats: Normalize repeated characters
            max_repeat: Maximum character repetition allowed
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.normalize_whitespace = normalize_whitespace
        self.normalize_repeats = normalize_repeats
        self.max_repeat = max_repeat

        # Compile regex patterns for performance
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._html_pattern = re.compile(r'<[^>]+>')
        self._mention_pattern = re.compile(r'@\w+')
        self._hashtag_pattern = re.compile(r'#\w+')
        self._whitespace_pattern = re.compile(r'\s+')

    def preprocess(self, text: str) -> str:
        """
        Preprocess text through cleaning pipeline.

        Args:
            text: Raw input text

        Returns:
            Cleaned and normalized text

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.preprocess("Check this: http://example.com AMAZING!!!")
            'check this amazing!'
        """
        if not text or not isinstance(text, str):
            return ""

        # Apply preprocessing steps
        text = self._remove_urls(text) if self.remove_urls else text
        text = self._remove_html(text) if self.remove_html else text
        text = self._normalize_unicode(text)
        text = self._normalize_repeats_fn(text) if self.normalize_repeats else text
        text = text.lower() if self.lowercase else text
        text = self._normalize_whitespace_fn(text) if self.normalize_whitespace else text

        return text.strip()

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of raw input texts

        Returns:
            List of cleaned texts
        """
        return [self.preprocess(text) for text in texts]

    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self._url_pattern.sub('', text)

    def _remove_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        return self._html_pattern.sub('', text)

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        # Normalize to NFKD form
        text = unicodedata.normalize('NFKD', text)
        # Keep only ASCII characters (optional - can keep accented chars)
        # text = text.encode('ascii', 'ignore').decode('utf-8')
        return text

    def _normalize_repeats_fn(self, text: str) -> str:
        """
        Normalize repeated characters.

        Example:
            "soooo goooood" -> "sooo goood" (with max_repeat=3)
        """
        if self.max_repeat < 1:
            return text

        # Pattern to match repeated characters
        pattern = r'(.)\1{' + str(self.max_repeat) + r',}'
        replacement = r'\1' * self.max_repeat

        return re.sub(pattern, replacement, text)

    def _normalize_whitespace_fn(self, text: str) -> str:
        """Normalize whitespace to single spaces."""
        return self._whitespace_pattern.sub(' ', text)

    def extract_features(self, text: str) -> dict:
        """
        Extract additional text features for analysis.

        Args:
            text: Input text

        Returns:
            Dictionary of text features
        """
        return {
            'length': len(text),
            'word_count': len(text.split()),
            'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1),
            'has_url': bool(self._url_pattern.search(text)),
            'has_mention': bool(self._mention_pattern.search(text)),
            'has_hashtag': bool(self._hashtag_pattern.search(text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        }

    def is_valid_text(self, text: str, min_length: int = 10, max_length: int = 5000) -> bool:
        """
        Check if text is valid for sentiment analysis.

        Args:
            text: Input text
            min_length: Minimum text length
            max_length: Maximum text length

        Returns:
            True if text is valid
        """
        if not text or not isinstance(text, str):
            return False

        text = text.strip()
        length = len(text)

        if length < min_length or length > max_length:
            return False

        # Check if text has actual content (not just special characters)
        word_count = len(text.split())
        if word_count == 0:
            return False

        return True


def clean_review_text(text: str) -> str:
    """
    Quick utility function for cleaning review text.

    Args:
        text: Raw review text

    Returns:
        Cleaned text
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess(text)
