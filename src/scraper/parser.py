"""BeautifulSoup parser for CNB OAM HTML results and detail pages.

Extracts documents, file references, and metadata from the page source.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import parse_qs, urlparse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def parse_results_html(html: str, typ_informace: str) -> list[dict]:
    """Parse the main HTML results page.

    Identifies document rows, extracts metadata, detail links, and attachment links.
    Different sections of the OAM results have slightly different column layouts,
    so this parser uses flexible checks (e.g. href content) rather than hardcoded index offsets.

    Args:
        html: Raw HTML content of the results page.
        typ_informace: The category used in the search filter (classification target).

    Returns:
        List of dictionaries, each representing a document with its files.
        Example schema:
        {
            "id": "S21244083",
            "emitter_name": "...",
            "emitter_ico": "...",
            "emitter_lei": "...",
            "typ_informace": "...",
            "typ_zpravy": "...",
            "strucny_popis": "...",
            "datum_prijeti": "...",
            "posledni_den_obdobi": None,
            "section": "...",
            "files": [
                {
                    "file_id": 12345,
                    "filename": "report.pdf",
                    "language": "CZ",
                    "file_extension": "pdf",
                    "download_url": "..."
                }
            ]
        }
    """
    soup = BeautifulSoup(html, "lxml")
    documents = []

    # OAM results are usually laid out in standard HTML tables
    tables = soup.find_all("table")
    logger.debug("Found %d tables in results page HTML", len(tables))

    for table in tables:
        # Find which section this table belongs to by looking for headers nearby
        section_name = "unknown"
        prev_sibling = table.find_previous(["h1", "h2", "h3", "div", "span"])
        while prev_sibling:
            txt = prev_sibling.get_text(strip=True)
            if txt and len(txt) > 5:  # Arbitrary threshold to find a header title
                section_name = txt
                break
            prev_sibling = prev_sibling.find_previous(["h1", "h2", "h3", "div", "span"])

        # We skip structured data sections with no file attachments if they aren't relevant,
        # but let's parse them anyway if they contain detail links.
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue

            # Look for a detail page link (contains R2_FXX.xdo)
            detail_link = None
            for cell in cells:
                a_tag = cell.find("a", href=True)
                if a_tag and "R2_FXX.xdo" in a_tag["href"]:
                    detail_link = a_tag["href"]
                    break

            if not detail_link:
                continue

            # Extract document ID from detail link
            parsed_url = urlparse(detail_link)
            query_params = parse_qs(parsed_url.query)
            obf_id_list = query_params.get("par_obf_id")
            if not obf_id_list:
                continue
            doc_id = obf_id_list[0]

            # Parse files from cells containing DWNL_FILE links
            files = []
            for cell in cells:
                file_links = cell.find_all("a", href=True)
                for f_link in file_links:
                    href = f_link["href"]
                    if "DWNL_FILE" in href:
                        file_url_parsed = urlparse(href)
                        file_id_list = parse_qs(file_url_parsed.query).get("file_id")
                        if file_id_list:
                            file_id = int(file_id_list[0])
                            link_text = f_link.get_text(strip=True)
                            
                            # Deduce language and clean filename
                            lang = "CZ"
                            if "(EN)" in link_text:
                                lang = "EN"
                            elif "(CZ)" in link_text:
                                lang = "CZ"
                                
                            filename = re.sub(r"\s*\([A-Z]{2}\)", "", link_text).strip()
                            if not filename:
                                filename = f"file_{file_id}"
                            
                            ext = filename.split(".")[-1].lower() if "." in filename else "bin"
                            
                            files.append({
                                "file_id": file_id,
                                "filename": filename,
                                "language": lang,
                                "file_extension": ext,
                                "download_url": href,
                            })

            # Extract emitter info and details from cell text
            # Emitter is typically in the first few cells, dates in middle, type after that.
            # Let's clean cell text to match fields.
            text_values = [c.get_text(strip=True) for c in cells]
            
            # Simple heuristics for fields based on common indices, but fallback to general parsing
            emitter_field = text_values[0] if len(text_values) > 0 else ""
            date_field = ""
            type_field = ""
            desc_field = ""
            
            # Look for date-like string (DD.MM.YYYY)
            date_pattern = r"\d{1,2}\.\d{1,2}\.\d{4}"
            for val in text_values:
                if re.search(date_pattern, val):
                    date_field = val
                    break
            
            # Clean emitter field (Format is often Name (IČ: 12345678, LEI: ...))
            emitter_name = emitter_field
            emitter_ico = ""
            emitter_lei = ""
            ico_match = re.search(r"IČO?\s*:?\s*(\d+)", emitter_field, re.IGNORECASE)
            if ico_match:
                emitter_ico = ico_match.group(1)
            lei_match = re.search(r"LEI\s*:?\s*([A-Z0-9]{20})", emitter_field, re.IGNORECASE)
            if lei_match:
                emitter_lei = lei_match.group(1)
                
            # Strip IČ / LEI details from emitter name
            emitter_name = re.sub(r"\(IČ.*?\)", "", emitter_name).strip()
            emitter_name = re.sub(r"\(LEI.*?\)", "", emitter_name).strip()

            # Fill in description and subtype from remaining cells
            # Find the longest non-date, non-emitter cells
            non_meta_cells = [
                v for v in text_values 
                if v != emitter_field and v != date_field and "Otevřít" not in v and "Příloha" not in v
            ]
            if len(non_meta_cells) > 0:
                type_field = non_meta_cells[0]
            if len(non_meta_cells) > 1:
                desc_field = " | ".join(non_meta_cells[1:])

            documents.append({
                "id": doc_id,
                "emitter_name": emitter_name,
                "emitter_ico": emitter_ico,
                "emitter_lei": emitter_lei,
                "typ_informace": typ_informace,
                "typ_zpravy": type_field or typ_informace,
                "strucny_popis": desc_field or type_field,
                "datum_prijeti": date_field,
                "posledni_den_obdobi": None,
                "section": section_name,
                "files": files,
            })

    logger.info("Parsed %d documents from results page HTML", len(documents))
    return documents


def parse_detail_html(html: str) -> dict:
    """Parse the document detail page to extract complete metadata.

    Args:
        html: Raw HTML content of the detail page.

    Returns:
        Dictionary of updated metadata fields.
    """
    soup = BeautifulSoup(html, "lxml")
    metadata = {}
    files = []

    # Detail page labels are in table cells, values in next cell or span
    cells = soup.find_all("td")
    for i, cell in enumerate(cells):
        text = cell.get_text(strip=True)
        if not text:
            continue
        
        # Clean text by removing colons and leading/trailing whitespace
        clean_text = text.replace(":", "").strip()
        
        if (clean_text == "Typ informace" or clean_text == "Typ zprávy") and i + 1 < len(cells):
            val = cells[i + 1].get_text(strip=True)
            metadata["typ_informace"] = val
            metadata["typ_zpravy"] = val
        elif clean_text == "ID informace" and i + 1 < len(cells):
            metadata["id"] = cells[i + 1].get_text(strip=True)
        elif clean_text == "Datum přijetí" and i + 1 < len(cells):
            metadata["datum_prijeti"] = cells[i + 1].get_text(strip=True)
        elif "Poslední den období" in clean_text and i + 1 < len(cells):
            metadata["posledni_den_obdobi"] = cells[i + 1].get_text(strip=True)
        elif (clean_text == "Obchodní firma nebo název" or clean_text == "Emitent") and i + 1 < len(cells):
            metadata["emitter_name"] = cells[i + 1].get_text(strip=True)
        elif (clean_text == "Identifikační číslo" or clean_text == "IČ") and i + 1 < len(cells):
            metadata["emitter_ico"] = cells[i + 1].get_text(strip=True)
        elif clean_text == "LEI" and i + 1 < len(cells):
            metadata["emitter_lei"] = cells[i + 1].get_text(strip=True)

    # Search for attachments in the detail page
    file_links = soup.find_all("a", href=True)
    for f_link in file_links:
        href = f_link["href"]
        if "DWNL_FILE" in href:
            file_url_parsed = urlparse(href)
            file_id_list = parse_qs(file_url_parsed.query).get("file_id")
            if file_id_list:
                file_id = int(file_id_list[0])
                link_text = f_link.get_text(strip=True)
                
                lang = "CZ"
                if "(EN)" in link_text:
                    lang = "EN"
                elif "(CZ)" in link_text:
                    lang = "CZ"
                    
                filename = re.sub(r"\s*\([A-Z]{2}\)", "", link_text).strip()
                if not filename:
                    filename = f"file_{file_id}"
                
                ext = filename.split(".")[-1].lower() if "." in filename else "bin"
                
                files.append({
                    "file_id": file_id,
                    "filename": filename,
                    "language": lang,
                    "file_extension": ext,
                    "download_url": href,
                })

    metadata["files"] = files
    logger.debug("Parsed detail metadata for ID: %s", metadata.get("id"))
    return metadata
