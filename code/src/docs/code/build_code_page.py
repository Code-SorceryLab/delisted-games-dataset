#!/usr/bin/env python3

"""
Refresh the source code shown in docs/code/index.html.

Writes, next to this script:
    - sources.js   every pipeline script as text, with its path
    - src/         plain copies of the same files, at their repo paths

The notebook is copied with its outputs removed and shown as its cells.
Data, figures and the LLM caches are not included.

Usage: python docs/code/build_code_page.py
"""

import json
import logging
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parents[1]

SOURCES_JS = BASE_DIR / "sources.js"
SRC_DIR = BASE_DIR / "src"

# Listed in pipeline order; each folder becomes a group on the page.
SOURCE_GLOBS = [
    "main.py",
    "requirements.txt",
    "Scripts/*.py",
    "IGDB_miner/*.py",
    "RQ2_package/*.py",
    "RQ3_package/*.py",
    "RQ3_package/historical_analysis/*.ipynb",
    "RQ3_package/delisting_causes/*.py",
    "RQ3_package/delisting_causes/prompt.txt",
    "docs/*.py",
    "docs/dataset/*.py",
    "docs/code/*.py",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def strip_notebook(text: str) -> dict:
    notebook = json.loads(text)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return notebook


def notebook_as_script(notebook: dict) -> str:
    """Cells in the '# %%' format that editors read as a notebook."""
    parts = []
    for cell in notebook["cells"]:
        source = "".join(cell["source"]).rstrip()
        if cell["cell_type"] == "markdown":
            lines = ["# " + line if line else "#" for line in source.splitlines()]
            parts.append("# %% [markdown]\n" + "\n".join(lines))
        elif cell["cell_type"] == "code":
            parts.append("# %%\n" + source)
    return "\n\n".join(parts) + "\n"


def main() -> None:
    if SRC_DIR.exists():
        shutil.rmtree(SRC_DIR)

    files = []
    for pattern in SOURCE_GLOBS:
        for path in sorted(REPO_DIR.glob(pattern)):
            rel = path.relative_to(REPO_DIR).as_posix()
            text = path.read_text(encoding="utf-8")
            target = SRC_DIR / rel
            target.parent.mkdir(parents=True, exist_ok=True)

            if path.suffix == ".ipynb":
                notebook = strip_notebook(text)
                target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
                text = notebook_as_script(notebook)
            else:
                shutil.copy2(path, target)

            files.append({
                "path": rel,
                "group": path.parent.relative_to(REPO_DIR).as_posix() if path.parent != REPO_DIR else "",
                "text": text,
            })

    payload = json.dumps(files, ensure_ascii=False, separators=(",", ":"))
    SOURCES_JS.write_text(f"window.SOURCES = {payload};\n", encoding="utf-8")
    logger.info(
        "Wrote %d files to %s (%.0f KB) and %s",
        len(files),
        SOURCES_JS.relative_to(REPO_DIR),
        SOURCES_JS.stat().st_size / 1e3,
        SRC_DIR.relative_to(REPO_DIR),
    )


if __name__ == "__main__":
    main()
