#!/usr/bin/env python3

import json
import logging
import math
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.doesitplay.org"
LIST_URL = f"{BASE_URL}/list"

TABLE_ID = "indextable"

OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "RawData" / "doesitplay_games.json"
)

# Be polite to the website.
REQUEST_DELAY_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def create_session() -> requests.Session:
    """
    Create an HTTP session with retries and browser-like headers.
    """
    session = requests.Session()

    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retries)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/152.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
    )

    return session


def download_page(
    session: requests.Session,
    page: int,
) -> str:
    """
    Download one page of the game list.
    """
    logger.info("Downloading page %d", page)

    response = session.get(
        LIST_URL,
        params={"page": page},
        timeout=60,
    )

    response.raise_for_status()

    logger.info(
        "Downloaded %.2f KB",
        len(response.content) / 1024,
    )

    return response.text


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def clean_text(element) -> str:
    """
    Extract normalized visible text.
    """
    if element is None:
        return ""

    return element.get_text(
        " ",
        strip=True,
    )


def extract_platform(game_cell) -> str:
    """
    Extract the platform displayed inside the Game cell.

    Example:
        <div class='platform ps5'>PS5</div>
    """
    platform_element = game_cell.find(
        "div",
        class_="platform",
    )

    return clean_text(platform_element)


def extract_platform_code(game_cell) -> str:
    """
    Extract the CSS platform identifier.

    Examples:
        platform ps5     -> ps5
        platform seriesx -> seriesx
    """
    platform_element = game_cell.find(
        "div",
        class_="platform",
    )

    if platform_element is None:
        return ""

    classes = platform_element.get(
        "class",
        [],
    )

    for css_class in classes:
        if css_class != "platform":
            return css_class

    return ""


def extract_game_info(game_cell) -> tuple[str, str]:
    """
    Extract the game title and the URL to its DoesItPlay detail page.
    """
    link = game_cell.find(
        "a",
        href=True,
    )

    if link is None:
        return clean_text(game_cell), ""

    game_name = clean_text(link)

    game_url = urljoin(
        BASE_URL,
        link["href"],
    )

    return game_name, game_url


def parse_results_count(soup: BeautifulSoup):
    """
    Extract the total number of results from text such as:

        3522 results found

    Returns None when the value cannot be found.
    """
    page_text = soup.get_text(
        " ",
        strip=True,
    )

    match = re.search(
        r"([\d,]+)\s+results?\s+found",
        page_text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return int(
        match.group(1).replace(",", "")
    )


# ---------------------------------------------------------------------------
# Table parser
# ---------------------------------------------------------------------------

def parse_table(html: str) -> tuple[list[dict], int | None]:
    """
    Parse one page of the DoesItPlay game table.

    Returns:
        records
        total result count, if detectable
    """
    soup = BeautifulSoup(
        html,
        "lxml",
    )

    table = soup.find(
        "table",
        id=TABLE_ID,
    )

    if table is None:
        raise RuntimeError(
            f"Could not find table with id='{TABLE_ID}'. "
            "The website structure may have changed."
        )

    tbody = table.find("tbody")

    if tbody is None:
        raise RuntimeError(
            "Could not find table body."
        )

    records = []

    rows = tbody.find_all(
        "tr",
        recursive=False,
    )

    for row in rows:

        cells = row.find_all(
            "td",
            recursive=False,
        )

        if not cells:
            continue

        # ---------------------------------------------------------------
        # Expected table layout:
        #
        # 0 -> mobile-only duplicate
        # 1 -> No.
        # 2 -> Game
        # 3 -> Version
        # 4 -> Region
        # 5 -> Tested on
        # 6 -> Offline play
        # 7 -> Download required
        # ---------------------------------------------------------------

        if len(cells) < 8:
            logger.warning(
                "Skipping malformed row with %d cells.",
                len(cells),
            )
            continue

        number_cell = cells[1]
        game_cell = cells[2]
        version_cell = cells[3]
        region_cell = cells[4]
        tested_on_cell = cells[5]
        offline_cell = cells[6]
        download_cell = cells[7]

        game_name, game_url = extract_game_info(
            game_cell
        )

        record = {
            "No": clean_text(number_cell),

            "Game": game_name,

            # Additional structured information from the Game cell.
            "Platform": extract_platform(game_cell),
            "PlatformCode": extract_platform_code(game_cell),

            "Version": clean_text(version_cell),
            "Region": clean_text(region_cell),
            "TestedOn": clean_text(tested_on_cell),

            "OfflinePlay": clean_text(offline_cell),
            "DownloadRequired": clean_text(download_cell),

            # Useful for later enrichment.
            "GameURL": game_url,
        }

        records.append(record)

    total_results = parse_results_count(
        soup
    )

    return records, total_results


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def scrape_all_games(
    session: requests.Session,
) -> list[dict]:
    """
    Scrape every page from DoesItPlay.

    The first page is used to determine the total number of results and
    estimate the number of pages. A duplicate-page check is also used as
    a safety mechanism.
    """
    all_records = []

    # ---------------------------------------------------------------
    # First page
    # ---------------------------------------------------------------

    first_html = download_page(
        session,
        page=1,
    )

    first_records, total_results = parse_table(
        first_html
    )

    if not first_records:
        raise RuntimeError(
            "The first page contains no game records."
        )

    all_records.extend(
        first_records
    )

    records_per_page = len(first_records)

    logger.info(
        "Records on first page: %d",
        records_per_page,
    )

    # ---------------------------------------------------------------
    # Determine number of pages
    # ---------------------------------------------------------------

    if total_results is not None:

        total_pages = math.ceil(
            total_results / records_per_page
        )

        logger.info(
            "Total results reported by website: %d",
            total_results,
        )

        logger.info(
            "Estimated pages: %d",
            total_pages,
        )

    else:
        # Fallback mode:
        # continue until a page returns no data or repeats the previous page.
        logger.warning(
            "Could not determine total result count. "
            "Using open-ended pagination."
        )

        total_pages = None

    # Fingerprint used to detect accidental repeated pages.
    previous_fingerprint = tuple(
        record["GameURL"]
        for record in first_records
    )

    page = 2

    while True:

        if total_pages is not None and page > total_pages:
            break

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

        html = download_page(
            session,
            page=page,
        )

        records, _ = parse_table(
            html
        )

        if not records:
            logger.info(
                "No records found on page %d. Stopping.",
                page,
            )
            break

        current_fingerprint = tuple(
            record["GameURL"]
            for record in records
        )

        if current_fingerprint == previous_fingerprint:
            logger.warning(
                "Page %d duplicates the previous page. "
                "Stopping pagination.",
                page,
            )
            break

        all_records.extend(
            records
        )

        logger.info(
            "Collected %d records so far.",
            len(all_records),
        )

        previous_fingerprint = current_fingerprint

        page += 1

    return all_records


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def remove_exact_duplicates(
    records: list[dict],
) -> list[dict]:
    """
    Remove exact duplicate game-test records.

    A game can legitimately appear multiple times because different
    platforms, editions, regions, or tested hardware are separate tests.

    Therefore the key contains all those fields.
    """
    unique_records = []

    seen = set()

    for record in records:

        key = (
            record["Game"],
            record["Platform"],
            record["Version"],
            record["Region"],
            record["TestedOn"],
            record["OfflinePlay"],
            record["DownloadRequired"],
        )

        if key in seen:
            continue

        seen.add(key)

        unique_records.append(
            record
        )

    return unique_records


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_json(
    data: list[dict],
    output_file: Path,
) -> None:
    """
    Save records to a UTF-8 JSON file.
    """
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "Saved %d records to %s",
        len(data),
        output_file,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    session = create_session()

    records = scrape_all_games(
        session
    )

    logger.info(
        "Raw records collected: %d",
        len(records),
    )

    records = remove_exact_duplicates(
        records
    )

    logger.info(
        "Records after exact deduplication: %d",
        len(records),
    )

    save_json(
        records,
        OUTPUT_FILE,
    )

    if records:
        print("\nExample record:\n")

        print(
            json.dumps(
                records[0],
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()