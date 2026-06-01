"""Preprocessing package for CNB OAM document classifier.

Provides extractors for PDFs, DOCX, XHTML, XBRL, ZIPs, text cleaning
utilities, and the central TextExtractionPipeline.
"""

from src.preprocessing.pdf_extractor import extract_text_from_pdf
from src.preprocessing.xhtml_extractor import extract_text_from_xhtml
from src.preprocessing.xbrl_extractor import extract_text_from_xbrl
from src.preprocessing.docx_extractor import extract_text_from_docx
from src.preprocessing.zip_handler import extract_text_from_zip
from src.preprocessing.text_cleaner import clean_text, truncate_for_bert, get_text_stats
from src.preprocessing.pipeline import TextExtractionPipeline

__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_xhtml",
    "extract_text_from_xbrl",
    "extract_text_from_docx",
    "extract_text_from_zip",
    "clean_text",
    "truncate_for_bert",
    "get_text_stats",
    "TextExtractionPipeline",
]
