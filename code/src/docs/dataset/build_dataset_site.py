#!/usr/bin/env python3

"""
Refresh the data shown in docs/dataset/index.html.

Writes, next to this script:
    - games.js           one entry per unique title, with its source records
    - classification.js  taxonomy values for classified titles
    - raw/*.json         copies of the four scraped source files, for download
    - raw/files.js       name, size and record count of each raw file

Classification is read from every Analysis/taxonomy_*classification*.csv.
When a title appears in more than one file, the later file (sorted by name)
wins, so the manually reviewed sample overrides the full classification. Titles without a classification are shown as "not yet classified", so
new classification files only need this script re-run.

Usage: python docs/dataset/build_dataset_site.py
"""

import json
import logging
import re
import shutil
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parents[1]

GAMES_CSV = REPO_DIR / "Analysis" / "unified_games.csv"
CLASSIFICATION_GLOB = "Analysis/taxonomy_*classification*.csv"

GAMES_JS = BASE_DIR / "games.js"
CLASSIFICATION_JS = BASE_DIR / "classification.js"
RAW_DIR = BASE_DIR / "raw"

# Source key on the page -> scraped file in RawData/.
RAW_FILES = {
    "steam_tracker": "RawData/SteamTracker/steam_delisted_apps.json",
    "doesitplay": "RawData/doesitplay_games.json",
    "delisted_games": "RawData/DelistedGames/delistedgames.json",
    "dead_game_list": "RawData/DeadGameWiki/dead_games.json",
}

EXCERPT_CHARS = 400
DIMENSIONS = [
    "D1_digital_availability",
    "D2_physical_availability",
    "D3_medium_completeness",
    "D4_online_dependency",
    "D5_service_status",
    "D6_platform_accessibility",
    "D7_preservation",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def s(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def year(value) -> str:
    return "" if pd.isna(value) else str(int(value))


def first_url(text: str) -> str:
    match = re.search(r"https?://\S+?(?=-https?://|\s|$)", text)
    return match.group(0) if match else ""


def delisted_excerpt(info: str) -> str:
    """First paragraph after the site's two boilerplate lines."""
    if not info:
        return ""
    try:
        parts = json.loads(info)
    except json.JSONDecodeError:
        return info[:EXCERPT_CHARS]
    parts = [p.strip() for p in parts[2:] if p.strip()]
    text = parts[0] if parts else ""
    return text if len(text) <= EXCERPT_CHARS else text[:EXCERPT_CHARS].rsplit(" ", 1)[0] + "…"


def record(row) -> dict:
    src = row["source"]
    rec = {
        "s": src,
        "n": s(row["game_name"]),
        "p": s(row["raw__doesitplay__Platform"]) or s(row["platforms"]),
        "pub": s(row["publisher"]),
        "dev": s(row["developer"]),
        "ry": year(row["release_year"]),
        "ey": year(row["event_year"]),
        "st": s(row["source_status"]),
    }
    if src == "steam_tracker" and s(row["raw__steamtracker__AppID"]):
        rec["url"] = f"https://steamdb.info/app/{int(row['raw__steamtracker__AppID'])}/"
    elif src == "doesitplay":
        rec["url"] = s(row["raw__doesitplay__GameURL"])
        rec["x"] = (f"Offline play: {s(row['raw__doesitplay__OfflinePlay'])}; "
                    f"download required: {s(row['raw__doesitplay__DownloadRequired'])}; "
                    f"version: {s(row['raw__doesitplay__Version'])}; "
                    f"region: {s(row['raw__doesitplay__Region'])}")
    elif src == "dead_game_list":
        rec["url"] = s(row["raw__deadgamelist__Home_page"])
        rec["x"] = s(row["raw__deadgamelist__Notes"])
    elif src == "delisted_games":
        rec["url"] = first_url(s(row["raw__delistedgames__data__links"]))
        rec["x"] = delisted_excerpt(s(row["raw__delistedgames__data__info"]))
    return {k: v for k, v in rec.items() if v}


def build_games() -> list:
    df = pd.read_csv(GAMES_CSV, low_memory=False)
    games = []
    for title, group in df.sort_values("normalized_title").groupby("normalized_title", sort=False):
        games.append({"id": title, "r": [record(row) for _, row in group.iterrows()]})
    logger.info("Built %d titles from %d records", len(games), len(df))
    return games


def build_classification() -> dict:
    out = {}
    for path in sorted(REPO_DIR.glob(CLASSIFICATION_GLOB)):
        df = pd.read_csv(path, keep_default_na=False)
        for _, row in df.iterrows():
            out[row["normalized_title"]] = {
                "d": [row[c] for c in DIMENSIONS],
                "ev": row.get("evidence", ""),
                "note": row.get("manual_note", ""),
                "src": row.get("stratum_source", ""),
                "m": row.get("method", "manual review"),
                "conf": row.get("confidence", ""),
            }
        logger.info("Read %d classified titles from %s", len(df), path.name)
    return out


def write_js(path: Path, name: str, data) -> None:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    path.write_text(f"window.{name} = {payload};\n", encoding="utf-8")
    logger.info("Wrote %s (%.1f MB)", path.relative_to(REPO_DIR), path.stat().st_size / 1e6)


def copy_raw_files() -> dict:
    RAW_DIR.mkdir(exist_ok=True)
    out = {}
    for key, rel in RAW_FILES.items():
        path = REPO_DIR / rel
        shutil.copy2(path, RAW_DIR / path.name)
        out[key] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "records": len(json.loads(path.read_text(encoding="utf-8"))),
        }
    return out


def main() -> None:
    write_js(GAMES_JS, "GAMES", build_games())
    write_js(CLASSIFICATION_JS, "CLASSIFICATION", build_classification())
    write_js(RAW_DIR / "files.js", "RAW_FILES", copy_raw_files())


if __name__ == "__main__":
    main()
