#!/usr/bin/env python3

"""
Refresh the data shown in docs/pipeline.html.

Writes, next to this script:
    - previews.js   column names and first rows of every CSV/JSON output
    - figs/*.png    copies of the RQ3 figures

The diagram and the filter/decision/issue text are written by hand in
pipeline.html and are not touched here.
"""

import json
import logging
import shutil
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent

PREVIEWS_JS = BASE_DIR / "previews.js"
FIGS_DIR = BASE_DIR / "figs"

PREVIEW_ROWS = 3
MAX_CELL_CHARS = 60

CSV_GLOBS = [
    "Analysis/*.csv",
    "IGDB_miner/igdb_*.csv",
    "RQ2_package/*.csv",
    "RQ3_package/historical_analysis/*.csv",
    "RQ3_package/delisting_causes/*.csv",
]

RECORD_JSON_FILES = [
    "RawData/DeadGameWiki/dead_games.json",
    "RawData/doesitplay_games.json",
    "RawData/SteamTracker/steam_delisted_apps.json",
    "RawData/DelistedGames/delistedgames.json",
]

OBJECT_JSON_FILES = [
    "Analysis/model_readiness.json",
]

FIGURE_GLOBS = [
    "RQ3_package/historical_analysis/*.png",
    "RQ3_package/delisting_causes/figures/*.png",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def clip(value) -> str:
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)

    value = " ".join(str(value).split())

    if len(value) > MAX_CELL_CHARS:
        value = value[:MAX_CELL_CHARS - 3] + "…"

    return value


def table_preview(df: pd.DataFrame) -> dict:
    return {
        "columns": [str(column) for column in df.columns],
        "rows": [[clip(value) for value in row] for row in df.values.tolist()],
    }


def main():
    previews = {}

    for pattern in CSV_GLOBS:
        for path in sorted(REPO_DIR.glob(pattern)):
            df = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                nrows=PREVIEW_ROWS,
            )
            previews[path.relative_to(REPO_DIR).as_posix()] = table_preview(df)

    for name in RECORD_JSON_FILES:
        records = json.loads((REPO_DIR / name).read_text(encoding="utf-8"))
        df = pd.json_normalize(records[:PREVIEW_ROWS]).fillna("")
        previews[name] = table_preview(df)

    for name in OBJECT_JSON_FILES:
        if not (REPO_DIR / name).exists():
            logger.warning("%s not found; no preview.", name)
            continue
        data = json.loads((REPO_DIR / name).read_text(encoding="utf-8"))
        previews[name] = table_preview(pd.DataFrame([data]))

    PREVIEWS_JS.write_text(
        "const PREVIEWS = "
        + json.dumps(previews, ensure_ascii=False).replace("</", "<\\/")
        + ";\n",
        encoding="utf-8",
    )

    FIGS_DIR.mkdir(exist_ok=True)

    figures = [
        path
        for pattern in FIGURE_GLOBS
        for path in sorted(REPO_DIR.glob(pattern))
    ]

    for path in figures:
        shutil.copy2(path, FIGS_DIR / path.name)

    logger.info(
        "Wrote %d previews to %s and %d figures to %s",
        len(previews),
        PREVIEWS_JS,
        len(figures),
        FIGS_DIR,
    )


if __name__ == "__main__":
    main()
