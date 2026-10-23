#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


URL = "https://steam-tracker.com/apps/delisted"
TABLE_ID = "delisted-apps"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_session() -> requests.Session:
    """
    Create an HTTP session with retries and a browser-like User-Agent.
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


def download_page(url: str) -> str:
    """
    Download the HTML page.
    """
    logger.info("Downloading %s", url)

    session = create_session()

    response = session.get(
        url,
        timeout=60,
    )

    response.raise_for_status()

    logger.info(
        "Downloaded %.2f MB",
        len(response.content) / (1024 * 1024),
    )

    return response.text


def clean_cell(cell) -> str:
    """
    Extract the visible textual value from a table cell.

    Examples:
        <td>...</td> -> "34270"

        <td>
            <a>SEGA Mega Drive & Genesis Classics</a>
        </td>
        -> "SEGA Mega Drive & Genesis Classics"
    """
    return cell.get_text(" ", strip=True)


def parse_table(html: str) -> list[dict]:
    """
    Parse the delisted-apps HTML table.

    The column names are read directly from the table header instead
    of being hard-coded.
    """
    soup = BeautifulSoup(html, "lxml")

    table = soup.find("table", id=TABLE_ID)

    if table is None:
        raise RuntimeError(
            f"Could not find table with id='{TABLE_ID}'. "
            "The website structure may have changed."
        )

    # Read column names dynamically.
    headers = [
        th.get_text(" ", strip=True)
        for th in table.select("thead th")
    ]

    if not headers:
        raise RuntimeError("No table headers found.")

    logger.info("Columns found: %s", headers)

    results = []

    tbody = table.find("tbody")

    if tbody is None:
        raise RuntimeError("Table contains no <tbody> element.")

    for row in tbody.find_all("tr", recursive=False):

        cells = row.find_all("td", recursive=False)

        # Skip empty / malformed rows.
        if not cells:
            continue

        if len(cells) != len(headers):
            logger.warning(
                "Skipping row: expected %d cells, found %d",
                len(headers),
                len(cells),
            )
            continue

        values = [clean_cell(cell) for cell in cells]

        record = dict(zip(headers, values))

        results.append(record)

    return results


def save_json(data: list[dict], output_file: Path) -> None:
    """
    Save extracted data as pretty-printed UTF-8 JSON.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "Saved %d records to %s",
        len(data),
        output_file,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Scrape delisted Steam apps from steam-tracker.com"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent.parent
            / "RawData" / "SteamTracker" / "steam_delisted_apps.json"
        ),
        help="Output JSON file",
    )

    parser.add_argument(
        "--url",
        default=URL,
        help="Steam Tracker URL",
    )

    args = parser.parse_args()

    html = download_page(args.url)

    data = parse_table(html)

    logger.info("Extracted %d applications.", len(data))

    save_json(data, args.output)


if __name__ == "__main__":
    main()