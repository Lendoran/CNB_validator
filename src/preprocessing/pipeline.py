"""Text extraction pipeline dispatching files to appropriate extractors.

Orchestrates database iteration, file format routing, cleaning, and database updates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from tqdm import tqdm

from src.scraper.metadata_db import MetadataDB
from src.preprocessing.pdf_extractor import extract_text_from_pdf
from src.preprocessing.xhtml_extractor import extract_text_from_xhtml
from src.preprocessing.xbrl_extractor import extract_text_from_xbrl
from src.preprocessing.docx_extractor import extract_text_from_docx
from src.preprocessing.zip_handler import extract_text_from_zip
from src.preprocessing.text_cleaner import clean_text

logger = logging.getLogger(__name__)


class TextExtractionPipeline:
    """Manages text extraction and database updates for all downloaded OAM files."""

    def __init__(self, db_path: str | Path) -> None:
        """Initialise pipeline.

        Args:
            db_path: Path to the SQLite metadata database.
        """
        self.db = MetadataDB(db_path)

    def extract_text(self, file_path: Path) -> tuple[str, str]:
        """Dispatch a file to its corresponding format extractor.

        Args:
            file_path: Path to the file.

        Returns:
            Tuple of (extracted_raw_text, method_name).
        """
        ext = file_path.suffix.lower()
        
        if ext == ".pdf":
            return extract_text_from_pdf(file_path), "pdfplumber/PyMuPDF"
        elif ext in [".xhtml", ".html", ".htm", ".tml"]:
            return extract_text_from_xhtml(file_path), "beautifulsoup/lxml"
        elif ext in [".xbrl", ".xml"]:
            return extract_text_from_xbrl(file_path), "lxml/xml"
        elif ext in [".docx", ".ocx", ".doc"]:
            return extract_text_from_docx(file_path), "python-docx"
        elif ext == ".zip":
            return extract_text_from_zip(file_path), "zipfile_extractor"
        else:
            logger.warning("Unsupported file extension '%s' for file: %s", ext, file_path.name)
            return "", "unsupported"

    def process_all_files(self, force: bool = False) -> int:
        """Scan DB files table and extract text for all downloaded documents.

        Args:
            force: Re-extract text even if it already exists in DB.

        Returns:
            Number of successfully processed files.
        """
        # Read the sqlite database directly for downloaded files
        with self.db._get_conn() as conn:
            # We want all files that have been downloaded
            cursor = conn.execute(
                """
                SELECT f.* FROM files f
                WHERE f.downloaded = 1;
                """
            )
            downloaded_files = cursor.fetchall()

        if not downloaded_files:
            logger.info("No downloaded files found in database for text extraction.")
            return 0

        logger.info("Found %d downloaded files. Starting text extraction...", len(downloaded_files))
        success_count = 0

        for row in tqdm(downloaded_files, desc="Extracting text from files"):
            file_rec = dict(row)
            file_id = file_rec["file_id"]
            doc_id = file_rec["document_id"]
            local_path_str = file_rec["local_path"]

            if not local_path_str:
                logger.warning("File record %d has downloaded=1 but empty local_path. Skipping.", file_id)
                continue

            local_path = Path(local_path_str)
            if not local_path.exists():
                logger.error("Physical file not found at local_path: %s (DB ID %d). Skipping.", local_path, file_id)
                continue

            # Check if text is already extracted
            if not force:
                with self.db._get_conn() as conn:
                    existing = conn.execute(
                        "SELECT 1 FROM extracted_text WHERE file_id = ?;", (file_id,)
                    ).fetchone()
                    if existing:
                        logger.debug("Text already extracted for file %d. Skipping (--force to override).", file_id)
                        success_count += 1
                        continue

            try:
                # Run extraction
                raw_text, method = self.extract_text(local_path)
                
                # Clean text
                cleaned = clean_text(raw_text)
                
                # Insert into DB
                self.db.insert_extracted_text({
                    "file_id": file_id,
                    "document_id": doc_id,
                    "text_content": cleaned,
                    "extraction_method": method,
                    "char_count": len(cleaned),
                })
                
                success_count += 1
                logger.debug("Successfully extracted %d characters from %s", len(cleaned), local_path.name)
            except Exception as e:
                logger.error("Failed to extract text for file %d (%s): %s", file_id, local_path.name, e)

        logger.info("Text extraction finished. Processed: %d/%d", success_count, len(downloaded_files))
        return success_count
