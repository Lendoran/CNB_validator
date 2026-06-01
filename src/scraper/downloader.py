"""Asynchronous file downloader for CNB OAM attachments.

Downloads files using their file_id and updates their status in the database.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import httpx
from tqdm.asyncio import tqdm

from src.scraper.metadata_db import MetadataDB

logger = logging.getLogger(__name__)


class FileDownloader:
    """Downloads files attached to CNB documents asynchronously with rate limiting."""

    def __init__(
        self,
        db_path: str | Path,
        raw_dir: str | Path = "data/raw",
        rate_limit_seconds: float = 1.5,
        max_retries: int = 3,
        timeout_seconds: int = 60,
    ) -> None:
        """Initialise downloader.

        Args:
            db_path: Path to the SQLite metadata database.
            raw_dir: Root directory where downloaded files will be stored.
            rate_limit_seconds: Delay between downloads.
            max_retries: Number of download retry attempts on failure.
            timeout_seconds: HTTP client request timeout.
        """
        self.db = MetadataDB(db_path)
        self.raw_dir = Path(raw_dir)
        self.rate_limit = rate_limit_seconds
        self.max_retries = max_retries
        self.timeout = timeout_seconds

        self.raw_dir.mkdir(parents=True, exist_ok=True)

    async def download_file(
        self,
        client: httpx.AsyncClient,
        file_rec: dict,
    ) -> bool:
        """Download a single file and update the database.

        Args:
            client: Shared AsyncClient instance.
            file_rec: Dictionary with file record details from database.

        Returns:
            True if download succeeded, False otherwise.
        """
        file_id = file_rec["file_id"]
        doc_id = file_rec["document_id"]
        filename = file_rec["filename"]
        url = file_rec["download_url"]

        # Define local path: data/raw/{document_id}/{filename}
        dest_dir = self.raw_dir / doc_id
        dest_path = dest_dir / filename

        # Ensure directory exists
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Retry loop
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug("Downloading file %d (Attempt %d/%d): %s", file_id, attempt, self.max_retries, filename)
                response = await client.get(url, timeout=self.timeout)
                response.raise_for_status()

                # Write contents
                dest_path.write_bytes(response.content)
                
                # Update DB
                self.db.update_file_downloaded(
                    file_id=file_id,
                    downloaded=1,
                    local_path=str(dest_path.resolve())
                )
                logger.debug("Successfully downloaded file %d to %s", file_id, dest_path)
                return True

            except Exception as e:
                logger.warning(
                    "Error downloading file %d (Attempt %d/%d): %s. Error: %s",
                    file_id, attempt, self.max_retries, filename, e
                )
                if attempt < self.max_retries:
                    # Exponential backoff on retry
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error("Failed to download file %d after %d attempts", file_id, self.max_retries)

        return False

    async def download_all(self, resume: bool = True) -> int:
        """Fetch and download all files marked as pending in the database.

        Args:
            resume: Skip files that are already marked as downloaded.

        Returns:
            Number of successfully downloaded files.
        """
        # Fetch pending files from DB
        files_to_download = self.db.get_files_to_download()
        if not files_to_download:
            logger.info("No pending files to download.")
            return 0

        logger.info("Starting download of %d files...", len(files_to_download))
        success_count = 0

        # Run downloads with polite rate limiting (sequential request loop to prevent IP bans)
        async with httpx.AsyncClient(follow_redirects=True) as client:
            # Wrap the download list in a progress bar
            for row in tqdm(files_to_download, desc="Downloading attachments"):
                file_rec = dict(row)
                
                # Define expected local path: data/raw/{document_id}/{filename}
                expected_path = self.raw_dir / file_rec["document_id"] / file_rec["filename"]
                
                # Double-check local path if resuming
                if resume:
                    # Check both the database-recorded path and the expected destination path
                    local_p = Path(file_rec["local_path"]) if file_rec["local_path"] else expected_path
                    if local_p.exists():
                        logger.debug("File %d already exists locally. Skipping.", file_rec["file_id"])
                        self.db.update_file_downloaded(file_rec["file_id"], 1, str(local_p.resolve()))
                        success_count += 1
                        continue

                # Download the file
                success = await self.download_file(client, file_rec)
                if success:
                    success_count += 1
                
                # Rate limit spacing between requests
                await asyncio.sleep(self.rate_limit)

        logger.info("Downloads completed. Succeeded: %d/%d", success_count, len(files_to_download))
        return success_count
