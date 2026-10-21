# Delisted Games Dataset

Static site listing video games collected from four community sources that track
delisted, discontinued or at-risk games (Steam Tracker, Does It Play?, Delisted Games,
Dead Game Wiki), merged by normalized title, with their classification along a seven-dimension
taxonomy (digital availability, physical availability, medium completeness, online dependency,
service status, platform accessibility, preservation).

All titles are classified; a random sample was reviewed manually, the rest were classified by fixed rules or by an LLM (Claude Haiku) from the source data. Each title shows its method.

The site has two tabs: the dataset, and a map of the SKVG data pipeline that produced it.

- `index.html` — landing page; opens the two views below as tabs
- `dataset/index.html` — the dataset page
- `dataset/games.js` — all titles and their source records
- `dataset/classification.js` — taxonomy values for classified titles
- `pipeline.html` — the pipeline map

The folder is self-contained: open `index.html` or serve the folder as-is (e.g. GitHub Pages).
The data files are generated in the SKVG-Project repo (`docs/`);
do not edit them by hand.

Pixel icons: [1-Bit Pack by Kenney](https://kenney.nl/assets/1-bit-pack) (CC0).
