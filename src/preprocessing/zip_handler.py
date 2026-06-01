"""ZIP file handler for CNB OAM documents.

Extracts ZIP archives to a temporary directory, scans contents for
supported files (XHTML, HTML, PDF, XML, XBRL, DOCX), extracts text from
them, and combines the results.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

from src.preprocessing.pdf_extractor import extract_text_from_pdf
from src.preprocessing.xhtml_extractor import extract_text_from_xhtml
from src.preprocessing.xbrl_extractor import extract_text_from_xbrl
from src.preprocessing.docx_extractor import extract_text_from_docx

logger = logging.getLogger(__name__)


def extract_text_from_zip(file_path: str | Path) -> str:
    """Extract and compile text from files inside a ZIP archive.

    Unpacks ZIP to a temporary directory, recursively processes files,
    concatenates results, and performs clean-up. Prioritizes human-readable formats
    (XHTML, HTML, PDF, DOCX) over raw XML/XBRL schemas.

    Args:
        file_path: Path to the ZIP file.

    Returns:
        Concatenated text content from all matched files inside the ZIP.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("ZIP file not found: %s", file_path)
        return ""

    # Create temporary directory for extraction
    temp_dir = Path(tempfile.mkdtemp(prefix="cnb_zip_"))
    logger.debug("Extracting ZIP %s to temp directory %s", file_path.name, temp_dir)

    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Recursively find files in the temp directory
        all_files = list(temp_dir.rglob("*"))
        
        # Filter files by extension
        xhtml_files = []
        pdf_files = []
        docx_files = []
        xml_xbrl_files = []
        nested_zips = []

        for p in all_files:
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext in [".xhtml", ".html", ".htm"]:
                xhtml_files.append(p)
            elif ext == ".pdf":
                pdf_files.append(p)
            elif ext == ".docx":
                docx_files.append(p)
            elif ext in [".xml", ".xbrl", ".xsd"]:
                xml_xbrl_files.append(p)
            elif ext == ".zip":
                nested_zips.append(p)

        extracted_texts = []

        # Process nested ZIPs first
        for nz in nested_zips:
            logger.debug("Processing nested ZIP %s inside %s", nz.name, file_path.name)
            nested_text = extract_text_from_zip(nz)
            if nested_text:
                extracted_texts.append(nested_text)

        # Process XHTML/HTML files (high priority, ESEF reports)
        for xf in xhtml_files:
            logger.debug("Extracting text from XHTML inside ZIP: %s", xf.name)
            txt = extract_text_from_xhtml(xf)
            if txt:
                extracted_texts.append(txt)

        # Process PDF files
        for pf in pdf_files:
            logger.debug("Extracting text from PDF inside ZIP: %s", pf.name)
            txt = extract_text_from_pdf(pf)
            if txt:
                extracted_texts.append(txt)

        # Process DOCX files
        for df in docx_files:
            logger.debug("Extracting text from DOCX inside ZIP: %s", df.name)
            txt = extract_text_from_docx(df)
            if txt:
                extracted_texts.append(txt)

        # Process XML/XBRL files (only if no XHTML/PDF texts were extracted, or as fallback)
        if not extracted_texts:
            for xf in xml_xbrl_files:
                # Avoid parsing standard schema definitions (.xsd) unless necessary
                if xf.suffix.lower() == ".xsd":
                    continue
                logger.debug("Extracting text from XML/XBRL inside ZIP: %s", xf.name)
                txt = extract_text_from_xbrl(xf)
                if txt:
                    extracted_texts.append(txt)

        full_text = "\n\n=== FILE SPLIT INSIDE ZIP ===\n\n".join(extracted_texts)
        return full_text

    except Exception as e:
        logger.error("Failed to process ZIP file %s: %s", file_path.name, e)
        return ""

    finally:
        # Clean up temporary files
        try:
            shutil.rmtree(temp_dir)
            logger.debug("Cleaned up temp directory %s", temp_dir)
        except Exception as err:
            logger.warning("Failed to remove temp directory %s: %s", temp_dir, err)
