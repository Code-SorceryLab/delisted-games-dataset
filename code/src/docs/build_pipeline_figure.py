#!/usr/bin/env python3

"""
Print docs/pipeline_figure.html to a PDF.

The figure uses the icons and line style of docs/pipeline.html, keeping only
the sources and the processing steps. It is drawn by hand in
pipeline_figure.html; this script only prints it with headless Chrome.

Writes, next to this script:
    - pipeline_figure.pdf
    - pipeline_figure.png   preview
"""

import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SOURCE = BASE_DIR / "pipeline_figure.html"
PDF = BASE_DIR / "pipeline_figure.pdf"
PNG = BASE_DIR / "pipeline_figure.png"

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Figure size in CSS pixels (7.16 in x 2.7338 in at 96 px/in).
WIDTH_PX, HEIGHT_PX = 687, 262
PNG_SCALE = 3


def chrome(*args):
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=5000",
         *args, SOURCE.as_uri()],
        check=True,
        capture_output=True,
    )


def main():
    chrome(f"--print-to-pdf={PDF}", "--no-pdf-header-footer")
    chrome(f"--screenshot={PNG}", f"--window-size={WIDTH_PX},{HEIGHT_PX}",
           f"--force-device-scale-factor={PNG_SCALE}", "--hide-scrollbars")
    print(f"Wrote {PDF} and {PNG}")


if __name__ == "__main__":
    main()
