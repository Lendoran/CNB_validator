# -*- coding: utf-8 -*-
"""Quick test of the improved OAM scraper with skipping existing IDs."""
import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from src.scraper.browser import OAMBrowser
from src.scraper.metadata_db import MetadataDB

async def test():
    db_path = Path("data/metadata.db")
    db = MetadataDB(db_path)
    existing_ids = db.get_all_document_ids()
    print(f"Loaded {len(existing_ids)} existing document IDs from database.")

    async with OAMBrowser() as browser:
        # Test with a small category to verify the "Další výsledky" expansion and skipping works
        docs = await browser.scrape_category(
            "Samostatná zpráva o nefinančních informacích",
            limit=1000,
            existing_ids=existing_ids,
        )
        print(f"\n{'='*60}")
        print(f"TOTAL NEW DOCUMENTS FOUND: {len(docs)}")
        print(f"{'='*60}")
        for i, d in enumerate(docs):
            files_count = len(d.get("files", []))
            print(f"  {i+1:3d}. [{d['id']}] {d['emitter_name'][:40]:40s} | {d['datum_prijeti']:20s} | {files_count} files")
        print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(test())
