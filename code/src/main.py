#!/usr/bin/env python3

"""
Run the full pipeline in order.

By default the pipeline is rebuilt from the stored snapshots
(RawData/ and the IGDB_miner CSVs), so results are reproducible:

    1. Scripts/analyze_game_delisting_sources.py   -> Analysis/
    2. RQ2_package/export_doesitplay_overlaps.py    -> RQ2_package/
    3. RQ3 historical analysis notebook             -> RQ3_package/historical_analysis/
    4. RQ3 delisting causes (validate + analyze)    -> RQ3_package/delisting_causes/
       from the stored LLM answers in RQ3_package/delisting_causes/llm_cache/
    5. docs/build_pipeline_page.py (if present)    -> docs/ (pipeline page data)

Live steps are opt-in because their results change over time:

    --scrape   re-mine the websites into RawData/ before step 1
    --igdb     re-query IGDB into IGDB_miner/ after step 1
               (needs real credentials in IGDB_miner/config.py)
    --classify re-run the local LLM cause classification before step 4
               (needs Ollama running with the models pulled)

RawData/DelistedGames/delistedgames.json has no miner and is never refreshed.
"""

import argparse
import importlib.util
import logging
import subprocess
import sys
import tempfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SCRAPERS = [
    BASE_DIR / "Scripts" / "mining_dead_game_wiki.py",
    BASE_DIR / "Scripts" / "mining_doesItPlay.py",
    BASE_DIR / "Scripts" / "mining_delisted_steam_tracker.py",
]

UNIFY_SCRIPT = BASE_DIR / "Scripts" / "analyze_game_delisting_sources.py"

IGDB_SCRIPT = BASE_DIR / "IGDB_miner" / "igdb_game_metadata_miner.py"
IGDB_CONFIG = BASE_DIR / "IGDB_miner" / "config.py"

RQ2_SCRIPT = BASE_DIR / "RQ2_package" / "export_doesitplay_overlaps.py"

RQ3_NOTEBOOK = (
    BASE_DIR
    / "RQ3_package"
    / "historical_analysis"
    / "game_delisting_statistics_with_igdb_correlation.ipynb"
)

CAUSES_DIR = BASE_DIR / "RQ3_package" / "delisting_causes"
CLASSIFY_SCRIPT = CAUSES_DIR / "classify_delisting_causes.py"
VALIDATE_SCRIPT = CAUSES_DIR / "validate_delisting_causes.py"
ANALYZE_SCRIPT = CAUSES_DIR / "analyze_delisting_causes.py"
LLM_CACHE_DIR = CAUSES_DIR / "llm_cache"

DOCS_SCRIPT = BASE_DIR / "docs" / "build_pipeline_page.py"

IGDB_PLACEHOLDERS = {"token here", "client id here"}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def run(command: list, cwd: Path = BASE_DIR) -> None:
    logger.info("Running: %s", " ".join(str(part) for part in command))

    subprocess.run(command, cwd=cwd, check=True)


def run_script(script: Path) -> None:
    run([sys.executable, script])


def check_igdb_credentials() -> None:
    spec = importlib.util.spec_from_file_location("config", IGDB_CONFIG)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if {config.token, config.CLIENT_ID} & IGDB_PLACEHOLDERS:
        sys.exit(
            f"Set token and CLIENT_ID in {IGDB_CONFIG} before using --igdb."
        )


def run_notebook(notebook: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--output-dir", tmp,
                notebook.name,
            ],
            cwd=notebook.parent,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run the pipeline from raw data to figures."
    )

    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Re-mine the websites into RawData/ first (results change over time).",
    )

    parser.add_argument(
        "--igdb",
        action="store_true",
        help="Re-query IGDB for genres and yearly release counts (slow, needs credentials).",
    )

    parser.add_argument(
        "--classify",
        action="store_true",
        help="Classify delisting causes with local LLMs via Ollama (slow, needs Ollama).",
    )

    args = parser.parse_args()

    if args.igdb:
        check_igdb_credentials()

    if args.scrape:
        for script in SCRAPERS:
            run_script(script)

    run_script(UNIFY_SCRIPT)

    if args.igdb:
        run_script(IGDB_SCRIPT)

    run_script(RQ2_SCRIPT)

    run_notebook(RQ3_NOTEBOOK)

    if args.classify:
        run_script(CLASSIFY_SCRIPT)

    if any(LLM_CACHE_DIR.glob("*.jsonl")):
        run_script(VALIDATE_SCRIPT)
        run_script(ANALYZE_SCRIPT)
    else:
        logger.warning("No LLM caches in %s; skipping the delisting-cause analysis.", LLM_CACHE_DIR)

    if DOCS_SCRIPT.exists():
        run_script(DOCS_SCRIPT)

    logger.info("Pipeline completed.")


if __name__ == "__main__":
    main()
