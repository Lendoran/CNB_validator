"""Czech text cleaning and normalisation utilities for document classification.

Provides cleaning, diacritic normalisation, whitespace cleaning, and BERT-token
truncation approximations.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean and normalise Czech text.

    - Normalises Unicode representation to NFC.
    - Removes non-printable characters.
    - Strips common document boilerplate (page numbers, footers).
    - Normalises whitespace and newlines.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned and structured text.
    """
    if not text:
        return ""

    # Normalise Unicode to NFC (ensures compound diacritics are single characters)
    text = unicodedata.normalize("NFC", text)

    # Remove non-printable control characters (except tab and newlines)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\r" or ch == "\t" or unicodedata.category(ch)[0] != "C")

    # Normalize whitespace
    # Replace carriage returns and tabs with spaces/newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    
    # Replace multiple spaces with a single space
    text = re.sub(r" +", " ", text)

    # Strip leading/trailing spaces from each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Remove boilerplate page numbers: e.g. "Strana 1 z 5", "stránka 2", "Page 5"
    text = re.compile(r"(?i)\bstrana\s+\d+\s+z\s+\d+\b").sub("", text)
    text = re.compile(r"(?i)\bstrán(ka|a)\s+\d+\b").sub("", text)
    text = re.compile(r"(?i)\bpage\s+\d+\s+of\s+\d+\b").sub("", text)
    text = re.compile(r"(?i)\bpage\s+\d+\b").sub("", text)

    # Normalize multiple newlines to max 2 newlines (preserves paragraph structure)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Strip leading/trailing whitespaces
    text = text.strip()

    # If text is extremely long, extract the head and tail to reduce noise and context size
    if len(text) > 25000:
        head = text[:15000]
        tail = text[-5000:]
        text = f"{head}\n\n... [TRUNCATED BODY SECTION] ...\n\n{tail}"
        logger.info("Extremely long text truncated to 15k head + 5k tail characters.")

    return text


def truncate_for_bert(text: str, max_tokens: int = 512) -> str:
    """Truncate text to approximate token limit.

    Since we don't have the tokenizer loaded during preprocessing, we use a
    safe heuristic: on average, a Czech word contains 5-6 characters, and 1 token
    corresponds roughly to 0.75 words.
    For 512 tokens, we keep roughly 350-400 words, or ~2000 characters.
    To be safe, we keep the first 380 words.

    Args:
        text: Input cleaned text.
        max_tokens: Target maximum tokens.

    Returns:
        Truncated text.
    """
    if not text:
        return ""

    words = text.split()
    # A safe approximation of 512 tokens for RobeCzech is roughly 380 words.
    approx_words = int(max_tokens * 0.75)
    
    if len(words) > approx_words:
        truncated = " ".join(words[:approx_words])
        logger.debug("Truncated text from %d to %d words", len(words), approx_words)
        return truncated

    return text


def get_text_stats(text: str) -> dict[str, int | str]:
    """Calculate character and word counts, and indicate a language hint.

    Args:
        text: Input text string.

    Returns:
        Dict with character_count, word_count, and language_hint.
    """
    if not text:
        return {"char_count": 0, "word_count": 0, "language_hint": "unknown"}

    words = text.split()
    char_count = len(text)
    word_count = len(words)

    # Simple language hint (Czech has specific characters like ř, š, č, ž, ý, á, í, é)
    cz_chars = sum(1 for ch in text.lower() if ch in "řščžýáíéůěťďňó")
    lang = "cs" if cz_chars > (word_count * 0.05) else "en"

    return {
        "char_count": char_count,
        "word_count": word_count,
        "language_hint": lang,
    }
