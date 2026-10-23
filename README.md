# Delisted Games Dataset

Static site listing video games collected from four community sources that track
delisted, discontinued or at-risk games (Steam Tracker, Does It Play?, Delisted Games,
Dead Game Wiki), merged by normalized title, with their classification along a seven-dimension
taxonomy (digital availability, physical availability, medium completeness, online dependency,
service status, platform accessibility, preservation).

All titles are classified; a random sample was reviewed manually, the rest were classified by fixed rules or by an LLM (Claude Haiku) from the source data. Each title shows its method.

The site has three tabs: the dataset, a map of the data pipeline that produced it, and the pipeline's source code.

- `index.html` — landing page; opens the three views below as tabs
- `dataset/index.html` — the dataset page
- `dataset/games.js` — all titles and their source records
- `dataset/classification.js` — taxonomy values for classified titles
- `dataset/raw/` — the four scraped source files (JSON), linked from the dataset page
- `pipeline.html` — the pipeline map
- `code/index.html` — the source code viewer
- `code/sources.js` — the scripts shown by the viewer
- `code/src/` — plain copies of the same scripts

The folder is self-contained: open `index.html` or serve the folder as-is (e.g. GitHub Pages).
The data files and `code/` (except `code/index.html`) are generated in the pipeline repo (`docs/`);
do not edit them by hand.

Pixel icons: [1-Bit Pack by Kenney](https://kenney.nl/assets/1-bit-pack) (CC0).
