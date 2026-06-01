"""PDF text extractor for CNB OAM documents.

Extracts text from PDF files using pdfplumber as the primary parser,
falling back to PyMuPDF (fitz) if needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract text content from a PDF file.

    Tries pdfplumber first for clean structured extraction. Falls back to PyMuPDF
    if pdfplumber fails or raises exceptions.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        Extracted text content as a string. Returns empty string on failure.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("PDF file not found: %s", file_path)
        return ""

    # Primary: PyMuPDF (fitz)
    try:
        import fitz  # PyMuPDF
        logger.debug("Extracting PDF text using PyMuPDF: %s", file_path)
        pages_text = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text = page.get_text()
                if text:
                    pages_text.append(text)
        
        full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text)
        if full_text.strip():
            return full_text
    except Exception as e:
        logger.warning("PyMuPDF extraction failed for %s: %s. Trying pdfplumber fallback.", file_path.name, e)

    # Secondary: pdfplumber
    try:
        import pdfplumber
        logger.debug("Extracting PDF text using pdfplumber: %s", file_path)
        pages_text = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages_text.append(text)
                else:
                    logger.debug("pdfplumber extracted empty text on page %d of %s", i + 1, file_path.name)
        
        full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text)
        if full_text.strip():
            return full_text
    except Exception as e:
        logger.error("pdfplumber fallback extraction also failed for %s: %s", file_path.name, e)

    return ""
