"""Playwright-based browser automation for scraping the CNB OAM portal.

Automates search form selection, submission, pagination, and detail page retrieval.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from playwright.async_api import async_playwright

from src.scraper.parser import parse_results_html, parse_detail_html

logger = logging.getLogger(__name__)

# Map configured classification categories to OAM search form listbox options
CATEGORY_MAPPING = {
    "Oznámení o konání valné hromady": "Oznámení o konání valné hromady",
    "Informace související s valnou hromadou": "Informace související s valnou hromadou",
    "Informace související s emisí dluhopisů": "Informace související s emisí dluhopisů",
    "Výroční finanční zpráva": "Výroční/pololetní finanční zpráva",
    "Pololetní finanční zpráva": "Výroční/pololetní finanční zpráva",
    "Vnitřní informace": "Vnitřní informace",
    "Oznámení podílu na hlasovacích právech": "Oznámení podílu na hlasovacích právech",
    "Informace o celkovém počtu hlasovacích práv a výši základního kapitálu": "Informace o celkovém počtu hlasovacích práv a výši základního kapitálu",
    "Oznámení o konání schůze vlastníků": "Oznámení o konání schůze vlastníků",
    "Informace o nabytí nebo pozbytí vlastních akcií emitenta": "Informace o nabytí nebo pozbytí vlastních akcií emitenta",
    "Zpráva o úhradách placených státu": "Zpráva o úhradách placených státu",
    "Samostatná zpráva o nefinančních informacích": "Samostatná zpráva o nefinančních informacích",
}


class OAMBrowser:
    """Automates interactions with the CNB OAM web forms using Playwright."""

    def __init__(
        self,
        base_url: str = "https://oam.cnb.cz",
        search_url: str = "https://oam.cnb.cz/sipresextdad/SIPRESWEB.WEB21.START_INPUT_OAM?p_lang=cz",
        rate_limit_seconds: float = 1.5,
        timeout_seconds: int = 60,
    ) -> None:
        """Initialise the browser crawler config.

        Args:
            base_url: Base domain of CNB OAM.
            search_url: Entrance URL containing the main search form.
            rate_limit_seconds: Delay between page requests.
            timeout_seconds: Connection/action timeout.
        """
        self.base_url = base_url.rstrip("/")
        self.search_url = search_url
        self.rate_limit = rate_limit_seconds
        self.timeout = timeout_seconds * 1000  # Playwright uses milliseconds

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def __aenter__(self) -> OAMBrowser:
        """Async context manager entry: launches browser and opens a page."""
        self._playwright = await async_playwright().start()
        # Run headless browser. User can run in headed mode if needed by editing launch call.
        self._browser = await self._playwright.chromium.launch(headless=True)
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()
        self._page.set_default_timeout(self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit: closes browser resources."""
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Playwright browser closed.")

    async def _navigate_and_wait(self, url: str, wait_on_load: int = 2000) -> None:
        """Helper to navigate to a URL and respect the rate limit delay."""
        logger.debug("Navigating to %s", url)
        await self._page.goto(url)
        # Dynamic wait for Oracle publisher loading screens or initial JS components
        if wait_on_load > 0:
            await self._page.wait_for_timeout(wait_on_load)
        await asyncio.sleep(self.rate_limit)

    async def _select_category_and_submit(
        self,
        form_option: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        """Navigate to the OAM search form, select a category, and submit.

        Args:
            form_option: The text label to match in the category multiselect.
            start_date: Optional start date filter (DD.MM.YYYY).
            end_date: Optional end date filter (DD.MM.YYYY).
        """
        await self._navigate_and_wait(self.search_url)

        # Select option in the multiselect listbox
        select_locator = self._page.locator(
            "select#ID_TYP_INFORMACE, select[name='_paramspar_typ_inf'], select"
        )
        selects_count = await select_locator.count()
        target_select = None
        for i in range(selects_count):
            loc = select_locator.nth(i)
            options_text = await loc.evaluate(
                "el => Array.from(el.options).map(o => o.text)"
            )
            if any(form_option in opt for opt in options_text):
                target_select = loc
                break
        if target_select is None:
            target_select = select_locator.first

        # Select the matching option and fire change events
        await target_select.evaluate(
            f"""(el) => {{
                const opts = Array.from(el.options);
                // Deselect all first
                opts.forEach(o => o.selected = false);
                const match = opts.find(o => o.text.includes("{form_option}"));
                if (match) {{
                    match.selected = true;
                    el.dispatchEvent(new Event('change'));
                }}
            }}"""
        )

        # Trigger the gatherValues() + sendRequest() AJAX to update the
        # "Typ zprávy" dropdown, if available on the page
        await self._page.evaluate("""() => {
            if (typeof gatherValues === 'function') {
                gatherValues();
            }
        }""")
        # Give the AJAX request time to complete
        await self._page.wait_for_timeout(1500)

        # Fill optional date range filters
        if start_date:
            await self._page.locator("input#DATUM_OD2").fill(start_date)
        if end_date:
            await self._page.locator("input#DATUM_DO2").fill(end_date)

        # Submit the form via the "Zobrazit" button handler
        await self._page.evaluate("""() => {
            if (typeof odeslani_vstupnich_hodnot === 'function') {
                odeslani_vstupnich_hodnot();
            } else {
                document.getElementById('ID_OAM_INPUTS').submit();
            }
        }""")

        logger.info("Form submitted. Waiting for results page to load...")

    async def _wait_for_results_and_expand(self) -> str:
        """Wait for results page to load and expand to show ALL results.

        After the form submits, the page goes through loading.jsp -> servlet/xdo.
        The initial view only shows 10 results. We extract the "Další výsledky"
        link (par_count=C1A) and navigate to it to get all results on one page.

        Returns:
            The HTML content of the fully expanded results page.
        """
        # Wait for the loading.jsp redirect to complete -> servlet/xdo
        try:
            await self._page.wait_for_url(
                lambda url: "servlet/xdo" in url and "fromLoadingPage" in url,
                timeout=30000,
            )
        except Exception:
            # Fallback: wait for any URL containing the results report
            try:
                await self._page.wait_for_url(
                    lambda url: "R1_RES.xdo" in url, timeout=15000
                )
            except Exception as e2:
                logger.warning("Timeout waiting for results page: %s", e2)

        # Wait for page content to render (the "Nalezeno" or "nevyhovuje" text)
        try:
            await self._page.wait_for_function(
                """() => {
                    const body = document.body ? document.body.innerText : '';
                    return body.includes('Nalezeno') || body.includes('nevyhovuje');
                }""",
                timeout=15000,
            )
        except Exception:
            # Extra static wait as fallback
            await self._page.wait_for_timeout(5000)

        # Now look for the "Další výsledky" link with par_count=C1A
        # This link shows ALL results (up to 1000) on a single page
        all_results_url = await self._page.evaluate("""() => {
            const links = Array.from(document.querySelectorAll('a[href]'));
            for (const link of links) {
                const href = link.getAttribute('href') || '';
                if (href.includes('par_count=C1A') || 
                    (link.textContent && link.textContent.includes('Další výsledky'))) {
                    return link.href;
                }
            }
            return null;
        }""")

        if all_results_url:
            logger.info(
                "Found 'Další výsledky' link. Navigating to all-results URL: %s",
                all_results_url[:120] + "...",
            )
            await self._page.goto(all_results_url)
            # Wait for the expanded results to render (60 seconds timeout to handle large listings)
            try:
                await self._page.wait_for_function(
                    """() => {
                        const body = document.body ? document.body.innerText : '';
                        return body.includes('Nalezeno') || body.includes('nevyhovuje');
                    }""",
                    timeout=60000,
                )
            except Exception:
                await self._page.wait_for_timeout(5000)
            await asyncio.sleep(self.rate_limit)
        else:
            logger.info(
                "No 'Další výsledky' link found (results may already be fully displayed)."
            )
            await asyncio.sleep(self.rate_limit)

        return await self._page.content()

    async def scrape_category(
        self,
        category_name: str,
        limit: int | None = None,
        existing_ids: set[str] | None = None,
        years: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Scrape documents for a specific category.

        Navigates to the search form, selects the option representing the category,
        submits the form, expands all results via the "Další výsledky" link, and
        parses all matching records.

        Args:
            category_name: Target classification category.
            limit: Maximum number of records to scrape.
            existing_ids: Optional set of document IDs already in DB to skip processing.
            years: Optional list of specific years to scrape. If omitted, defaults to all history (2010 to current year).

        Returns:
            List of parsed document dictionaries.
        """
        # Resolve to listbox label text
        form_option = CATEGORY_MAPPING.get(category_name, category_name)
        logger.info(
            "Scraping category '%s' (listbox option: '%s')", category_name, form_option
        )

        documents: list[dict[str, Any]] = []

        # Determine target years list
        if not years:
            import datetime
            current_year = datetime.datetime.now().year
            target_years = list(range(current_year, 2009, -1))
        else:
            # Sort descending to get newest documents first
            target_years = sorted(years, reverse=True)
        
        logger.info("Querying years: %s", ", ".join(map(str, target_years)))

        for y in target_years:
            if limit and len(documents) >= limit:
                break
            
            start_date = f"01.01.{y}"
            end_date = f"31.12.{y}"
            
            logger.info(
                "Fetching documents between %s and %s for category %s...",
                start_date,
                end_date,
                category_name,
            )

            # Retry loop for this specific year (handles transient server timeouts/errors)
            max_attempts = 3
            range_docs = []
            success = False
            for attempt in range(1, max_attempts + 1):
                try:
                    await self._select_category_and_submit(
                        form_option, start_date, end_date
                    )
                    html = await self._wait_for_results_and_expand()
                    range_docs = parse_results_html(html, category_name)
                    success = True
                    break
                except Exception as err:
                    logger.warning(
                        "Attempt %d/%d failed fetching year %d: %s",
                        attempt,
                        max_attempts,
                        y,
                        err,
                    )
                    if attempt < max_attempts:
                        # Recreate page/context to clear any corrupted redirect/load state
                        try:
                            logger.info("Recreating Playwright page to clear stuck browser state...")
                            await self._page.close()
                            self._page = await self._context.new_page()
                            self._page.set_default_timeout(self.timeout)
                        except Exception as recreate_err:
                            logger.error("Failed to recreate page: %s", recreate_err)
                        # Exponential backoff wait before retrying
                        await asyncio.sleep(2 * attempt)
                    else:
                        logger.error(
                            "All %d attempts failed for year %d. Skipping year.",
                            max_attempts,
                            y,
                        )

            if not success:
                continue

            collected_ids = {doc["id"] for doc in documents}
            added = 0
            for d in range_docs:
                if d["id"] not in collected_ids:
                    documents.append(d)
                    added += 1
            logger.info(
                "Found %d docs in range %s–%s (%d new in this range, total collected: %d)",
                len(range_docs),
                start_date,
                end_date,
                added,
                len(documents),
            )
            
            # OPTIMIZATION: If all documents returned for a past year are already in the DB,
            # we can assume we have all historical data and stop search early for efficiency.
            # Only do this if we are not restricting the query to a specific set of years.
            if not years and range_docs and existing_ids:
                # Get current year to check if y is past year
                import datetime
                curr_year = datetime.datetime.now().year
                if y < curr_year:
                    all_existing = all(d["id"] in existing_ids for d in range_docs)
                    if all_existing:
                        logger.info("All documents in year %d are already in database. Stopping search for earlier years.", y)
                        break

        if limit and len(documents) >= limit:
            documents = documents[:limit]

        # Post-process: Split "Výroční/pololetní finanční zpráva" category.
        # Also retrieve full metadata from detail pages.
        # We visit the detail pages of documents to check their actual Typ zprávy
        # and extract Poslední den období and missing files.
        new_documents = []
        logger.info("Visiting detail pages of scraped documents to gather complete metadata...")
        for i, doc in enumerate(documents):
            doc_id = doc["id"]
            if existing_ids and doc_id in existing_ids:
                logger.info("[%d/%d] Skipping existing document ID %s (already in database)", i + 1, len(documents), doc_id)
                continue
                
            # We only visit the detail page if:
            # 1. The category is "Výroční/pololetní finanční zpráva" (or its splits) to perform the document type split.
            # 2. Or the document is missing its file attachments list.
            # For all other categories, the metadata from the search results table is already complete.
            is_financial_report = any(term in category_name for term in ["Výroční", "Pololetní", "Výroční/pololetní"])
            has_files = bool(doc.get("files"))
            
            if not is_financial_report and has_files:
                logger.info("[%d/%d] Skipping detail page for document ID %s (already has metadata and files)", i + 1, len(documents), doc_id)
                # Ensure download URLs have absolute paths
                for f in doc.get("files", []):
                    if f["download_url"].startswith("/"):
                        f["download_url"] = f"{self.base_url}{f['download_url']}"
                    elif not f["download_url"].startswith("http"):
                        f["download_url"] = f"{self.base_url}/sipresextdad/{f['download_url']}"
                new_documents.append(doc)
                continue
                
            detail_url = f"{self.base_url}/xmlpserver/OAM_CNB_CZ/R2_FXX.xdo?par_obf_id={doc_id}&par_lang=cs&_xf=html"
            logger.info("[%d/%d] Fetching detail metadata for document ID %s", i + 1, len(documents), doc_id)
            
            try:
                # Use wait_on_load=0 to avoid the static 2-second delay for detail pages
                await self._navigate_and_wait(detail_url, wait_on_load=0)
                
                # Wait for the iframe xdo:docframe0 to load
                iframe_selector = "iframe[name='xdo:docframe0']"
                try:
                    await self._page.wait_for_selector(iframe_selector, timeout=10000)
                    iframe = self._page.frame(name="xdo:docframe0")
                    if iframe:
                        # Wait for actual table cells to appear in iframe
                        await iframe.wait_for_selector("td", timeout=5000)
                        detail_html = await iframe.content()
                    else:
                        detail_html = await self._page.content()
                except Exception as iframe_err:
                    logger.warning("Could not load iframe for document %s: %s. Using main page content.", doc_id, iframe_err)
                    detail_html = await self._page.content()
                
                detail_meta = parse_detail_html(detail_html)
                
                # Update document dictionary with detail fields
                doc["emitter_ico"] = detail_meta.get("emitter_ico", doc.get("emitter_ico"))
                doc["emitter_lei"] = detail_meta.get("emitter_lei", doc.get("emitter_lei"))
                doc["posledni_den_obdobi"] = detail_meta.get("posledni_den_obdobi")
                
                # Check for files on detail page if none were found on results page
                if not doc.get("files") and detail_meta.get("files"):
                    doc["files"] = detail_meta["files"]
                elif detail_meta.get("files"):
                    # Merge / update file lists
                    existing_file_ids = {f["file_id"] for f in doc["files"]}
                    for f in detail_meta["files"]:
                        if f["file_id"] not in existing_file_ids:
                            doc["files"].append(f)

                # Split Výroční vs Pololetní based on Typ zprávy field
                actual_type = detail_meta.get("typ_zpravy", "")
                if "výroční finanční zpráva" in actual_type.lower() or "roční finanční" in actual_type.lower():
                    doc["typ_informace"] = "Výroční finanční zpráva"
                elif "pololetní finanční zpráva" in actual_type.lower() or "pololetní" in actual_type.lower():
                    doc["typ_informace"] = "Pololetní finanční zpráva"

                # Ensure download URL has the absolute domain
                for f in doc.get("files", []):
                    if f["download_url"].startswith("/"):
                        f["download_url"] = f"{self.base_url}{f['download_url']}"
                    elif not f["download_url"].startswith("http"):
                        f["download_url"] = f"{self.base_url}/sipresextdad/{f['download_url']}"

                new_documents.append(doc)

            except Exception as e:
                logger.error("Failed to parse detail page for doc %s: %s", doc_id, e)
                # Keep it anyway to avoid losing the metadata we parsed from the main results
                new_documents.append(doc)

        return new_documents
