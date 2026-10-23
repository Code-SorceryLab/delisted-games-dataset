#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

URL = "https://stopkillinggames.wiki.gg/wiki/Dead_game_list"

MEDIAWIKI_API_URL = "https://stopkillinggames.wiki.gg/api.php"
MEDIAWIKI_PAGE = "Dead_game_list"

TABLE_ID = "dead-game-table"

OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "RawData" / "DeadGameWiki" / "dead_games.json"
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP utilities
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


def download_page(session: requests.Session) -> str:
    """
    Download the normal wiki page.

    If the normal page cannot be retrieved or does not contain the expected
    table, fall back to the MediaWiki API.
    """
    logger.info("Downloading %s", URL)

    try:
        response = session.get(
            URL,
            timeout=60,
        )

        response.raise_for_status()

        html = response.text

        if f'id="{TABLE_ID}"' in html:
            logger.info(
                "Downloaded %.2f MB from the normal page.",
                len(response.content) / (1024 * 1024),
            )

            return html

        logger.warning(
            "Page downloaded, but table '%s' was not found. "
            "Trying MediaWiki API.",
            TABLE_ID,
        )

    except requests.RequestException as exc:
        logger.warning(
            "Could not download normal wiki page: %s",
            exc,
        )

    return download_page_via_api(session)


def download_page_via_api(session: requests.Session) -> str:
    """
    Retrieve the rendered page HTML through the MediaWiki API.
    """
    logger.info("Downloading page through MediaWiki API.")

    params = {
        "action": "parse",
        "page": MEDIAWIKI_PAGE,
        "prop": "text",
        "format": "json",
        "formatversion": 2,
    }

    response = session.get(
        MEDIAWIKI_API_URL,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    if "parse" not in data or "text" not in data["parse"]:
        raise RuntimeError(
            "MediaWiki API response does not contain rendered page HTML."
        )

    html = data["parse"]["text"]

    logger.info(
        "Received %.2f MB of rendered HTML through the API.",
        len(html.encode("utf-8")) / (1024 * 1024),
    )

    return html


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def clean_cell(cell) -> str:
    """
    Extract normalized visible text from a table cell.
    """
    return cell.get_text(
        " ",
        strip=True,
    )


def extract_home_page(cell) -> str:
    """
    Extract the URL from the 'Home page' column.

    If the cell contains an <a> element, use its href rather than its
    displayed text. Relative wiki URLs are converted to absolute URLs.
    """
    link = cell.find("a", href=True)

    if link:
        return urljoin(URL, link["href"])

    return clean_cell(cell)


def parse_table(html: str) -> list[dict]:
    """
    Parse the dead game table and return one dictionary per game.

    Column names are extracted dynamically from the table header.
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

    # ------------------------------------------------------------------
    # Header
    #
    # Unlike the Steam Tracker table, this wiki table has the header
    # directly inside the first <tr> rather than inside <thead>.
    # ------------------------------------------------------------------

    header_row = table.find("tr")

    if header_row is None:
        raise RuntimeError(
            "Could not find the table header row."
        )

    headers = [
        th.get_text(" ", strip=True)
        for th in header_row.find_all("th")
    ]

    if not headers:
        raise RuntimeError(
            "No table headers found."
        )

    logger.info(
        "Found %d columns: %s",
        len(headers),
        headers,
    )

    # ------------------------------------------------------------------
    # Data rows
    # ------------------------------------------------------------------

    results = []

    rows = table.find_all("tr")

    # Skip the first row because it contains the table headers.
    for row_number, row in enumerate(
        rows[1:],
        start=1,
    ):
        cells = row.find_all(
            "td",
            recursive=False,
        )

        if not cells:
            continue

        if len(cells) != len(headers):
            logger.warning(
                "Skipping row %d: expected %d cells, found %d.",
                row_number,
                len(headers),
                len(cells),
            )
            continue

        record = {}

        for header, cell in zip(
            headers,
            cells,
        ):
            if header == "Home page":
                value = extract_home_page(cell)
            else:
                value = clean_cell(cell)

            record[header] = value

        results.append(record)

    logger.info(
        "Successfully parsed %d games.",
        len(results),
    )

    return results


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def save_json(
    data: list[dict],
    output_file: Path,
) -> None:
    """
    Save extracted data as pretty-printed UTF-8 JSON.
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

    html = download_page(session)

    games = parse_table(html)

    save_json(
        games,
        OUTPUT_FILE,
    )

    # Show a small preview.
    if games:
        logger.info("First extracted record:")
        print(
            json.dumps(
                games[0],
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()