"""
Scraper package for CNB OAM document classifier.

Provides components for scraping, parsing, and downloading documents
from the OAM (Centrální úložiště regulovaných informací) website.
"""

from src.scraper.metadata_db import MetadataDB
from src.scraper.browser import OAMBrowser
from src.scraper.parser import parse_results_html, parse_detail_html
from src.scraper.downloader import FileDownloader

__all__ = [
    "MetadataDB",
    "OAMBrowser",
    "parse_results_html",
    "parse_detail_html",
    "FileDownloader",
]
