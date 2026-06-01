"""XBRL and iXBRL text extractor for CNB OAM documents.

Extracts text from financial reports submitted in XBRL (pure XML) or
iXBRL (inline XBRL inside XHTML) formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from lxml import etree

from src.preprocessing.xhtml_extractor import extract_text_from_xhtml

logger = logging.getLogger(__name__)


def extract_text_from_xbrl(file_path: str | Path) -> str:
    """Extract text from an XBRL or iXBRL (inline XBRL) file.

    iXBRL files are HTML-based and are routed directly to the XHTML parser.
    Pure XBRL (XML) files are parsed to extract fact values, labels, and text nodes.

    Args:
        file_path: Path to the XBRL/iXBRL file.

    Returns:
        Extracted text content, or empty string on failure.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("XBRL file not found: %s", file_path)
        return ""

    # Detect if the file is inline XBRL (XHTML) or pure XBRL (XML)
    # iXBRL files typically start with HTML declaration or have XHTML content
    try:
        with open(file_path, "rb") as f:
            header = f.read(1024).lower()
        
        is_ixbrl = b"<html" in header or b"<!doctype html" in header or b"<xhtml" in header
        
        if is_ixbrl:
            logger.debug("iXBRL detected for %s. Routing to XHTML extractor.", file_path.name)
            return extract_text_from_xhtml(file_path)
            
    except Exception as e:
        logger.warning("Could not read file header for %s: %s", file_path.name, e)

    # Fallback to parsing as XML for pure XBRL files
    try:
        logger.debug("Parsing pure XBRL XML for %s", file_path.name)
        parser = etree.XMLParser(recover=True, resolve_entities=False)
        tree = etree.parse(str(file_path), parser)
        root = tree.getroot()

        # Extract text from all nodes that contain text
        # In XBRL, we are interested in fact values, labels, and any documentation nodes
        text_elements = []
        for element in root.iter():
            # Skip tags themselves, check element text and tail
            if element.text and element.text.strip():
                text_elements.append(element.text.strip())
            if element.tail and element.tail.strip():
                text_elements.append(element.tail.strip())

        full_text = "\n".join(text_elements)
        return full_text

    except Exception as e:
        logger.error("Failed to parse pure XBRL XML for %s: %s", file_path.name, e)
        
        # Final fallback: read as raw text and clean HTML tags using regex if etree failed
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            import re
            clean = re.compile("<.*?>")
            cleaned = re.sub(clean, " ", content)
            return " ".join(cleaned.split())
        except Exception as final_err:
            logger.error("Final fallback regex cleaning also failed for %s: %s", file_path.name, final_err)
            return ""
