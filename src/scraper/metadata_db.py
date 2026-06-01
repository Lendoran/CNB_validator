"""SQLite database for storing scraped metadata and extraction results.

Provides the MetadataDB class with CRUD operations for documents, files, and
extracted text.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class MetadataDB:
    """Manages the SQLite database for OAM metadata and extracted text.

    Ensures schemas are created and provides helper methods to interact with
    documents, files, and text extraction data.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialise database connection and run schema creation.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a SQLite connection with foreign keys enabled and row factory set."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        logger.info("Initialising SQLite database schema at %s", self.db_path)
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    emitter_name TEXT,
                    emitter_ico TEXT,
                    emitter_lei TEXT,
                    typ_informace TEXT,
                    typ_zpravy TEXT,
                    strucny_popis TEXT,
                    datum_prijeti TEXT,
                    posledni_den_obdobi TEXT,
                    section TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    file_id INTEGER PRIMARY KEY,
                    document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                    filename TEXT,
                    language TEXT,
                    file_extension TEXT,
                    download_url TEXT,
                    local_path TEXT,
                    downloaded INTEGER DEFAULT 0
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS extracted_text (
                    file_id INTEGER PRIMARY KEY REFERENCES files(file_id) ON DELETE CASCADE,
                    document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                    text_content TEXT,
                    extraction_method TEXT,
                    char_count INTEGER
                );
                """
            )
            conn.commit()

    def insert_document(self, doc: dict) -> None:
        """Insert a document record. Overwrites on conflict.

        Args:
            doc: Dictionary representing the document record.
        """
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    id, emitter_name, emitter_ico, emitter_lei, 
                    typ_informace, typ_zpravy, strucny_popis, 
                    datum_prijeti, posledni_den_obdobi, section
                ) VALUES (
                    :id, :emitter_name, :emitter_ico, :emitter_lei,
                    :typ_informace, :typ_zpravy, :strucny_popis,
                    :datum_prijeti, :posledni_den_obdobi, :section
                );
                """,
                doc,
            )
            conn.commit()

    def insert_file(self, file_rec: dict) -> None:
        """Insert a file record associated with a document. Overwrites on conflict.

        Args:
            file_rec: Dictionary representing the file record.
        """
        file_rec_copy = file_rec.copy()
        if "local_path" not in file_rec_copy:
            file_rec_copy["local_path"] = ""
        if "downloaded" not in file_rec_copy:
            file_rec_copy["downloaded"] = 0

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO files (
                    file_id, document_id, filename, language,
                    file_extension, download_url, local_path, downloaded
                ) VALUES (
                    :file_id, :document_id, :filename, :language,
                    :file_extension, :download_url, :local_path, :downloaded
                );
                """,
                file_rec_copy,
            )
            conn.commit()

    def update_file_downloaded(self, file_id: int, downloaded: int, local_path: str) -> None:
        """Update downloaded flag and local path of a file.

        Args:
            file_id: The ID of the file.
            downloaded: 1 if downloaded, 0 otherwise.
            local_path: Absolute or relative local path on disk.
        """
        with self._get_conn() as conn:
            conn.execute(
                """
                UPDATE files 
                SET downloaded = ?, local_path = ?
                WHERE file_id = ?;
                """,
                (downloaded, local_path, file_id),
            )
            conn.commit()

    def insert_extracted_text(self, text_rec: dict) -> None:
        """Insert extracted text for a file. Overwrites on conflict.

        Args:
            text_rec: Dictionary representing the text extraction record.
        """
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO extracted_text (
                    file_id, document_id, text_content, extraction_method, char_count
                ) VALUES (
                    :file_id, :document_id, :text_content, :extraction_method, :char_count
                );
                """,
                text_rec,
            )
            conn.commit()

    def get_documents_by_category(self, category: str) -> list[sqlite3.Row]:
        """Fetch all documents belonging to a particular category.

        Args:
            category: Classification target category (typ_informace).
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE typ_informace = ?;",
                (category,),
            )
            return cursor.fetchall()

    def get_files_to_download(self) -> list[sqlite3.Row]:
        """Return file records that have not been downloaded yet."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM files WHERE downloaded = 0;"
            )
            return cursor.fetchall()

    def get_stats(self) -> dict:
        """Return summary statistics of the database."""
        with self._get_conn() as conn:
            total_docs = conn.execute("SELECT COUNT(*) FROM documents;").fetchone()[0]
            total_files = conn.execute("SELECT COUNT(*) FROM files;").fetchone()[0]
            downloaded_files = conn.execute("SELECT COUNT(*) FROM files WHERE downloaded = 1;").fetchone()[0]
            extracted_texts = conn.execute("SELECT COUNT(*) FROM extracted_text;").fetchone()[0]
            
            categories_cursor = conn.execute(
                """
                SELECT typ_informace, COUNT(*) as count 
                FROM documents 
                GROUP BY typ_informace 
                ORDER BY count DESC;
                """
            )
            categories = {row["typ_informace"]: row["count"] for row in categories_cursor}

            return {
                "total_documents": total_docs,
                "total_files": total_files,
                "downloaded_files": downloaded_files,
                "extracted_texts": extracted_texts,
                "category_counts": categories,
            }

    def get_all_document_ids(self) -> set[str]:
        """Fetch all document IDs currently in the database."""
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT id FROM documents;")
            return {row["id"] for row in cursor.fetchall()}

    def insert_documents_batch(self, docs: list[dict], files: list[dict]) -> None:
        """Insert a batch of documents and files in a single transaction.

        Args:
            docs: List of dictionaries representing document records.
            files: List of dictionaries representing file records.
        """
        with self._get_conn() as conn:
            for doc in docs:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents (
                        id, emitter_name, emitter_ico, emitter_lei, 
                        typ_informace, typ_zpravy, strucny_popis, 
                        datum_prijeti, posledni_den_obdobi, section
                    ) VALUES (
                        :id, :emitter_name, :emitter_ico, :emitter_lei,
                        :typ_informace, :typ_zpravy, :strucny_popis,
                        :datum_prijeti, :posledni_den_obdobi, :section
                    );
                    """,
                    doc,
                )
            for f in files:
                f_copy = f.copy()
                if "local_path" not in f_copy:
                    f_copy["local_path"] = ""
                if "downloaded" not in f_copy:
                    f_copy["downloaded"] = 0
                conn.execute(
                    """
                    INSERT OR REPLACE INTO files (
                        file_id, document_id, filename, language,
                        file_extension, download_url, local_path, downloaded
                    ) VALUES (
                        :file_id, :document_id, :filename, :language,
                        :file_extension, :download_url, :local_path, :downloaded
                    );
                    """,
                    f_copy,
                )
            conn.commit()
