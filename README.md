# Delisted Games Dataset

Static site listing video games collected from four community sources that track
delisted, discontinued or at-risk games (Steam Tracker, Does It Play?, Delisted Games,
Dead Game Wiki), merged by normalized title, with their classification along a seven-dimension
taxonomy (digital availability, physical availability, medium completeness, online dependency,
service status, platform accessibility, preservation).

All titles are classified; a random sample was reviewed manually, the rest were classified by fixed rules or by an LLM (Claude Haiku) from the source data. Each title shows its method.

- `index.html` — the page
- `games.js` — all titles and their source records
- `classification.js` — taxonomy values for classified titles

The data files are generated; do not edit them by hand.
