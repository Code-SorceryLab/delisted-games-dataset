#!/usr/bin/env python3

"""
Why do games disappear? Descriptive analysis of the explicitly stated
delisting causes in delistedGames.

Works WITHIN delistedGames (every entry was delisted), so there is no
PC-vs-console control-group confound as in the old genre comparison.
Causes are multi-label: shares can sum to more than 100%.

Inputs:
    - delisting_cause_labels.csv                     (validate_delisting_causes.py)
    - RawData/DelistedGames/delistedgames.json       release / delisting dates
    - Analysis/unified_games.csv                     normalized_title
    - IGDB_miner/igdb_genres_by_normalized_game.csv  genres (exact matches only)

Outputs (in figures/, as PNG + PDF; test results go to the log):
    A. 01_cause_counts, 02_cause_share_by_platform
    B. 03_cause_share_by_genre
    C. 04_age_at_delisting_ecdf
    D. 05_causes_by_year
    E. 06_relisting_rate_by_cause
"""

import json
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))

from figure_style import (  # noqa: E402
    CAUSE_STYLE,
    INK,
    MUTED,
    SINGLE_COLUMN,
    apply_style,
    plot_bars,
    set_title,
    style_count_axis,
    style_year_axis,
    wide_figure,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_DIR = BASE_DIR.parent.parent

LABELS_CSV = BASE_DIR / "delisting_cause_labels.csv"
FIGURES_DIR = BASE_DIR / "figures"
RAW_JSON = REPO_DIR / "RawData" / "DelistedGames" / "delistedgames.json"
UNIFIED_CSV = REPO_DIR / "Analysis" / "unified_games.csv"
IGDB_GENRES_CSV = REPO_DIR / "IGDB_miner" / "igdb_genres_by_normalized_game.csv"

CAUSES = ["licensing", "ip_change", "server", "store_closure", "publisher_exit"]
FALLBACKS = ["other", "not_stated"]
LABELS = CAUSES + FALLBACKS

LAST_COMPLETE_YEAR = 2025
FIRST_PLOT_YEAR = 2008      # fewer than a handful of delistings per year before this

MIN_PLATFORM_ENTRIES = 50
MIN_GENRE_ENTRIES = 25
FDR_ALPHA = 0.05

EARLY_AGE_YEARS = 2         # "pulled in the first two years"
MAX_PLOT_AGE = 15

CAUSE_BAR_COLOR = CAUSE_STYLE["licensing"]["color"]
FALLBACK_BAR_COLOR = CAUSE_STYLE["other"]["color"]

PLATFORM_NAMES = {
    "pc": "PC", "gog": "GOG", "steam": "Steam", "ios": "iOS",
    "playstation-3": "PlayStation 3", "playstation-4": "PlayStation 4",
    "playstation-vita": "PS Vita", "playstation-portable": "PSP",
    "nintendo-3ds": "Nintendo 3DS", "nintendo-wii-u": "Wii U",
    "nintendo-wii": "Wii", "nintendo-switch": "Switch", "nintendo-ds": "Nintendo DS",
    "xbox-360": "Xbox 360", "xbox-one": "Xbox One",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# matplotlib's PDF font subsetting logs every glyph at INFO.
logging.getLogger("fontTools").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def label_name(label: str) -> str:
    return CAUSE_STYLE[label]["label"]


def platform_name(slug: str) -> str:
    return PLATFORM_NAMES.get(slug, slug.replace("-", " ").title())


def save_figure(fig, name: str) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(FIGURES_DIR / f"{name}{suffix}")
    plt.close(fig)
    logger.info("Saved %s.png/.pdf", name)


def bh_qvalues(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values, NaNs left out."""
    q = pd.Series(np.nan, index=p_values.index)
    valid = p_values.dropna()
    if len(valid):
        q[valid.index] = stats.false_discovery_control(valid.values, method="bh")
    return q


MONTHS = {
    month: number
    for number, month in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}

DATE_PATTERN = re.compile(
    r"(?:\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+"
    r"(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?)?"
    r"\b((?:19|20)\d\d)\b",
    re.I,
)


def earliest_date(text) -> tuple:
    """
    Earliest date in a free-text date block, as (fractional year, has_month).
    Year-only dates are placed mid-year.
    """
    if not isinstance(text, str):
        return np.nan, False

    dates = []
    for month, year in DATE_PATTERN.findall(text):
        if month:
            dates.append((int(year) + (MONTHS[month.lower()[:3]] - 1) / 12, True))
        else:
            dates.append((int(year) + 0.5, False))

    return min(dates) if dates else (np.nan, False)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_CSV)
    for column in LABELS + [f"{c}_agree" for c in CAUSES] + ["speculative", "relisted", "is_bundle"]:
        labels[column] = labels[column].astype(bool)

    with open(RAW_JSON, encoding="utf-8") as handle:
        raw = pd.DataFrame(
            {
                "id": record["id"],
                "digital_release": record["data"].get("digital_release"),
                "delisting": record["data"].get("delisting"),
            }
            for record in json.load(handle)
        )

    for column in ("digital_release", "delisting"):
        parsed = raw[column].apply(earliest_date)
        raw[f"{column}_date"] = parsed.str[0]
        raw[f"{column}_has_month"] = parsed.str[1]

    raw["age_years"] = raw["delisting_date"] - raw["digital_release_date"]
    raw["delisting_year"] = np.floor(raw["delisting_date"])

    unified = pd.read_csv(
        UNIFIED_CSV, usecols=["source", "source_record_id", "normalized_title"], low_memory=False,
    )
    unified = unified[unified["source"] == "delisted_games"].rename(columns={"source_record_id": "id"})

    data = (
        labels
        .merge(raw.drop(columns=["digital_release", "delisting"]), on="id", how="left")
        .merge(unified[["id", "normalized_title"]], on="id", how="left")
    )

    logger.info("Loaded %d labelled entries.", len(data))

    return data


# ---------------------------------------------------------------------------
# A. Headline counts and platforms
# ---------------------------------------------------------------------------

def cause_counts(data: pd.DataFrame) -> pd.DataFrame:
    total = len(data)
    rows = []

    for label in LABELS:
        mask = data[label]
        rows.append({
            "label": label,
            "name": label_name(label),
            "entries": int(mask.sum()),
            "share_pct": round(100 * mask.mean(), 2),
            "entries_both_models": int(data[f"{label}_agree"].sum()) if label in CAUSES else np.nan,
            "speculative": int((mask & data["speculative"]).sum()),
            "bundles": int((mask & data["is_bundle"]).sum()),
            "total_entries": total,
        })

    counts = pd.DataFrame(rows)

    n_any = int(data[CAUSES].any(axis=1).sum())
    n_multi = int((data[CAUSES].sum(axis=1) > 1).sum())
    logger.info(
        "Out of %d entries: %d (%.1f%%) state at least one of the five causes, "
        "%d have more than one.\n%s",
        total, n_any, 100 * n_any / total, n_multi, counts.to_string(index=False),
    )

    fig, ax = plot_bars(
        [label_name(label) for label in LABELS],
        counts["entries"].tolist(),
        [CAUSE_BAR_COLOR if label in CAUSES else FALLBACK_BAR_COLOR for label in LABELS],
        f"Stated delisting causes (n = {total:,}, multi-label)",
        value_format=lambda value: f"{value:,} ({100 * value / total:.0f}%)",
    )
    ax.set_xlim(0, counts["entries"].max() * 1.32)
    save_figure(fig, "01_cause_counts")

    return counts


def plot_share_heatmap(shares: pd.DataFrame, row_labels: list, title: str, name: str) -> None:
    """Rows x labels heatmap of % shares, one sequential hue, values in the cells."""
    n_rows, n_cols = shares.shape
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 0.75 + 0.22 * n_rows))

    ax.imshow(shares.values, cmap="Blues", vmin=0, vmax=max(60, np.nanmax(shares.values)), aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            value = shares.iat[i, j]
            ax.text(
                j, i, f"{value:.0f}", ha="center", va="center", fontsize=7,
                color="white" if value > 40 else INK,
            )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([label_name(c).replace(" ", "\n", 1) for c in shares.columns], fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    set_title(ax, title)
    fig.tight_layout(pad=0.3)
    save_figure(fig, name)


def platform_split(data: pd.DataFrame) -> None:
    counts = data["platform"].value_counts()
    big = counts[counts >= MIN_PLATFORM_ENTRIES].index
    grouped = data.assign(platform_group=data["platform"].where(data["platform"].isin(big), "other platforms"))

    table = grouped.groupby("platform_group")[LABELS].mean().mul(100).round(2)
    table.insert(0, "entries", grouped["platform_group"].value_counts())
    table = table.sort_values("entries", ascending=False)

    rows = [p for p in table.index if p != "other platforms"]
    plot_share_heatmap(
        table.loc[rows, LABELS],
        [f"{platform_name(p)} ({table.at[p, 'entries']})" for p in rows],
        "Share of entries per stated cause, by platform (%)",
        "02_cause_share_by_platform",
    )


# ---------------------------------------------------------------------------
# B. Cause x genre
# ---------------------------------------------------------------------------

def genre_association(data: pd.DataFrame) -> None:
    igdb = pd.read_csv(IGDB_GENRES_CSV, usecols=["normalized_title", "genres", "match_status"])
    igdb = igdb[(igdb["match_status"] == "exact_normalized_match") & igdb["genres"].notna()]

    matched = data.merge(igdb[["normalized_title", "genres"]], on="normalized_title", how="inner")
    logger.info(
        "Genres: %d of %d entries have an exact IGDB match with genres.", len(matched), len(data),
    )

    exploded = matched.assign(genre=matched["genres"].str.split("|")).explode("genre")
    genre_sizes = exploded["genre"].value_counts()
    genres = genre_sizes[genre_sizes >= MIN_GENRE_ENTRIES].index

    rows = []
    for genre in genres:
        in_genre = matched["genres"].str.split("|").apply(lambda g: genre in g)
        for cause in CAUSES:
            a = int((in_genre & matched[cause]).sum())
            b = int((in_genre & ~matched[cause]).sum())
            c = int((~in_genre & matched[cause]).sum())
            d = int((~in_genre & ~matched[cause]).sum())
            odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]])
            rows.append({
                "genre": genre,
                "cause": cause,
                "genre_entries": a + b,
                "genre_entries_with_cause": a,
                "share_in_genre_pct": round(100 * a / (a + b), 2),
                "share_in_other_genres_pct": round(100 * c / (c + d), 2) if c + d else np.nan,
                "odds_ratio": round(odds_ratio, 3),
                "fisher_p": p_value,
            })

    table = pd.DataFrame(rows)
    table["bh_q"] = bh_qvalues(table["fisher_p"])
    table["significant"] = table["bh_q"] < FDR_ALPHA
    table["matched_entries"] = len(matched)

    significant = table[table["significant"]].sort_values("bh_q")
    logger.info("Significant genre x cause cells (BH q < %.2f):\n%s",
                FDR_ALPHA, significant.to_string(index=False))

    shares = table.pivot(index="genre", columns="cause", values="share_in_genre_pct")[CAUSES]
    order = shares.sort_values("licensing", ascending=False).index
    plot_share_heatmap(
        shares.loc[order],
        [f"{g} ({genre_sizes[g]})" for g in order],
        "Share of entries per stated cause, by IGDB genre (%)",
        "03_cause_share_by_genre",
    )


# ---------------------------------------------------------------------------
# C. Cause x age at delisting
# ---------------------------------------------------------------------------

def age_at_delisting(data: pd.DataFrame) -> None:
    aged = data[data["age_years"].notna()]
    negative = int((aged["age_years"] < 0).sum())
    aged = aged[aged["age_years"] >= 0]
    month_precision = int((aged["digital_release_has_month"] & aged["delisting_has_month"]).sum())

    logger.info(
        "Age at delisting: %d entries with both dates (%d with month precision), "
        "%d dropped with negative age.",
        len(aged), month_precision, negative,
    )

    rows = []
    for label in LABELS:
        with_label = aged.loc[aged[label], "age_years"]
        without = aged.loc[~aged[label], "age_years"]
        rows.append({
            "label": label,
            "entries": len(with_label),
            "median_years": round(with_label.median(), 2),
            "q1_years": round(with_label.quantile(0.25), 2),
            "q3_years": round(with_label.quantile(0.75), 2),
            "mean_years": round(with_label.mean(), 2),
            f"share_under_{EARLY_AGE_YEARS}y_pct": round(100 * (with_label < EARLY_AGE_YEARS).mean(), 2),
            f"share_under_{EARLY_AGE_YEARS}y_rest_pct": round(100 * (without < EARLY_AGE_YEARS).mean(), 2),
            "mannwhitney_p": stats.mannwhitneyu(with_label, without).pvalue if len(with_label) else np.nan,
            "ks_p": stats.ks_2samp(with_label, without).pvalue if len(with_label) else np.nan,
            "negative_age_dropped": negative,
            "month_precision_entries": month_precision,
        })

    table = pd.DataFrame(rows)
    table["mannwhitney_bh_q"] = bh_qvalues(table["mannwhitney_p"])
    table["ks_bh_q"] = bh_qvalues(table["ks_p"])
    logger.info("Age at delisting per cause:\n%s", table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.6))
    grid = np.linspace(0, MAX_PLOT_AGE, 301)

    def ecdf(values):
        values = np.sort(values.to_numpy())
        return np.searchsorted(values, grid, side="right") / len(values)

    ax.plot(grid, ecdf(aged["age_years"]), color=MUTED, linestyle=(0, (4, 2)), linewidth=1.0,
            label=f"All entries ({len(aged):,})")
    for cause in CAUSES:
        values = aged.loc[aged[cause], "age_years"]
        if values.empty:
            continue
        style = CAUSE_STYLE[cause]
        ax.plot(grid, ecdf(values), color=style["color"], marker=style["marker"], markevery=30,
                markeredgecolor="white", markeredgewidth=0.4,
                label=f"{style['label']} ({len(values):,})")

    ax.set_xlim(0, MAX_PLOT_AGE)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Years from digital release to first delisting")
    ax.set_ylabel("Cumulative share of entries")
    ax.legend(loc="lower right", fontsize=7)
    set_title(ax, "Age at delisting by stated cause")
    fig.tight_layout(pad=0.3)
    save_figure(fig, "04_age_at_delisting_ecdf")


# ---------------------------------------------------------------------------
# D. Cause x calendar year
# ---------------------------------------------------------------------------

def causes_by_year(data: pd.DataFrame) -> None:
    dated = data[data["delisting_year"].between(FIRST_PLOT_YEAR, LAST_COMPLETE_YEAR)]
    years = range(FIRST_PLOT_YEAR, LAST_COMPLETE_YEAR + 1)

    table = dated.groupby("delisting_year")[LABELS].sum().reindex(years, fill_value=0)
    table.insert(0, "all_entries", dated["delisting_year"].value_counts().reindex(years, fill_value=0))
    table.index = table.index.astype(int)

    fig, ax = wide_figure()
    for cause in CAUSES:
        style = CAUSE_STYLE[cause]
        ax.plot(table.index, table[cause], color=style["color"], marker=style["marker"],
                markeredgecolor="white", markeredgewidth=0.4, label=style["label"],
                clip_on=False, zorder=3)

    ax.set_xlabel("Year of first delisting")
    ax.set_ylabel("Entries")
    style_year_axis(ax, list(years))
    style_count_axis(ax)
    ax.legend(loc="upper left", ncol=2)
    set_title(ax, "Delisted entries per year by stated cause")
    save_figure(fig, "05_causes_by_year")


# ---------------------------------------------------------------------------
# E. Relisting rate per cause
# ---------------------------------------------------------------------------

def relisting_rate(data: pd.DataFrame) -> None:
    rows = []
    for label in LABELS:
        mask = data[label]
        a = int((mask & data["relisted"]).sum())
        b = int((mask & ~data["relisted"]).sum())
        c = int((~mask & data["relisted"]).sum())
        d = int((~mask & ~data["relisted"]).sum())
        odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]])
        rows.append({
            "label": label,
            "entries": a + b,
            "relisted": a,
            "relisted_pct": round(100 * a / (a + b), 2) if a + b else np.nan,
            "relisted_rest_pct": round(100 * c / (c + d), 2) if c + d else np.nan,
            "odds_ratio": round(odds_ratio, 3),
            "fisher_p": p_value,
        })

    table = pd.DataFrame(rows)
    table["bh_q"] = bh_qvalues(table["fisher_p"])
    table["overall_relisted_pct"] = round(100 * data["relisted"].mean(), 2)
    logger.info("Relisting rate per cause:\n%s", table.to_string(index=False))

    shown = table[table["entries"] > 0]
    fig, ax = plot_bars(
        [f"{label_name(label)} ({n:,})" for label, n in zip(shown["label"], shown["entries"])],
        shown["relisted_pct"].tolist(),
        [CAUSE_BAR_COLOR if label in CAUSES else FALLBACK_BAR_COLOR for label in shown["label"]],
        "Share of entries later relisted, by stated cause",
        value_label="Entries later relisted (%)",
        value_format="{:.0f}%".format,
    )
    save_figure(fig, "06_relisting_rate_by_cause")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    apply_style()
    data = load_data()

    cause_counts(data)
    platform_split(data)
    genre_association(data)
    age_at_delisting(data)
    causes_by_year(data)
    relisting_rate(data)

    logger.info("Delisting-cause analysis completed.")


if __name__ == "__main__":
    main()
