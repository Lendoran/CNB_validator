"""HTML/XHTML text extractor for CNB OAM documents.

Extracts clean, visible text content from HTML/XHTML reports by removing scripts,
styles, and unnecessary formatting tags.
"""

from __future__ import annotations

import logging
from pathlib import Path
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def extract_text_from_xhtml(file_path: str | Path) -> str:
    """Extract visible text from an XHTML or HTML document.

    Removes script/style blocks, cleans spacing, and preserves basic structural
    grouping (e.g. paragraphs and headings as newlines).

    Args:
        file_path: Path to the HTML/XHTML file.

    Returns:
        Extracted text content, or empty string on failure.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("XHTML file not found: %s", file_path)
        return ""

    try:
        # Read the file content. Try UTF-8 first, fallback to latin-1 if needed.
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.debug("Unicode decode failed for %s. Retrying with latin-1.", file_path.name)
            content = file_path.read_text(encoding="latin-1")

        # Parse with BeautifulSoup using lxml parser
        soup = BeautifulSoup(content, "lxml")

        # Remove non-visible tags
        for element in soup(["script", "style", "head", "title", "meta", "[document]"]):
            element.decompose()

        # Extract text preserving block spacing
        # Adding whitespace between block elements before extracting text
        for block in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "li", "br"]):
            block.append("\n")

        text = soup.get_text()

        # Remove excessive whitespace and empty lines
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase for phrase in lines if phrase)
        cleaned_text = "\n".join(chunks)

        return cleaned_text

    except Exception as e:
        logger.error("Failed to extract text from XHTML %s: %s", file_path.name, e)
        return ""
