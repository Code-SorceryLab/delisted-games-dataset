#!/usr/bin/env python3

"""
Aggregate and analyze game delisting / shutdown datasets.

Inputs (under ../RawData/):
    - DeadGameWiki/dead_games.json
    - DelistedGames/delistedgames.json
    - SteamTracker/steam_delisted_apps.json
    - doesitplay_games.json

Outputs are written to ../Analysis/:
    - unified_games.csv
    - source_statistics.csv
    - source_status_statistics.csv
    - cross_source_overlap.csv
    - model_readiness.json

Important modeling note:
The four sources are useful for descriptive analysis, historical
delisting/shutdown cases, and offline/download-dependency characteristics.
However, they are still NOT sufficient by themselves for a valid prospective
model of "will this game be delisted?". A representative, time-aligned control
population and pre-event snapshots are still required.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR.parent / "RawData"

DEAD_GAMES_FILE = RAW_DATA_DIR / "DeadGameWiki" / "dead_games.json"
DELISTED_GAMES_FILE = RAW_DATA_DIR / "DelistedGames" / "delistedgames.json"
STEAM_TRACKER_FILE = RAW_DATA_DIR / "SteamTracker" / "steam_delisted_apps.json"
DOESITPLAY_FILE = RAW_DATA_DIR / "doesitplay_games.json"

OUTPUT_DIR = BASE_DIR.parent / "Analysis"

UNIFIED_CSV = OUTPUT_DIR / "unified_games.csv"
SOURCE_STATS_CSV = OUTPUT_DIR / "source_statistics.csv"
STATUS_STATS_CSV = OUTPUT_DIR / "source_status_statistics.csv"
OVERLAP_CSV = OUTPUT_DIR / "cross_source_overlap.csv"
MODEL_READINESS_JSON = OUTPUT_DIR / "model_readiness.json"

# A valid future-prediction target is intentionally not fabricated.
FUTURE_TARGET_COLUMN = "target_delisted_within_12m"

# Steam Tracker app types that are not released games.
NON_GAME_STEAM_TYPES = {"Unreleased", "Test app"}

# Title keywords marking betas, demos, test builds, DLC and soundtracks.
# Matched on the original title: normalize_title strips "(Beta)" etc.
NON_GAME_TITLE_PATTERN = r"\b(?:beta|demo|test server|playtest|soundtrack|dlc)\b"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def remove_trailing_commas(text: str) -> str:
    """
    Remove trailing commas before '}' or ']' while respecting JSON strings.

    This is useful because one of the supplied JSON files contains at least
    one trailing comma and therefore fails strict json.loads().
    """
    result = []
    in_string = False
    escaped = False
    i = 0

    while i < len(text):
        char = text[i]

        if in_string:
            result.append(char)

            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False

            i += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue

        if char == ",":
            j = i + 1

            while j < len(text) and text[j].isspace():
                j += 1

            if j < len(text) and text[j] in "}]":
                # Skip only the comma. Whitespace is preserved by the loop.
                i += 1
                continue

        result.append(char)
        i += 1

    return "".join(result)


def load_json(path: Path):
    """
    Load JSON strictly first. If parsing fails because of trailing commas,
    retry after a conservative cleanup.
    """
    logger.info("Loading %s", path)

    text = path.read_text(encoding="utf-8")

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Strict JSON parsing failed for %s: %s. "
            "Retrying after removing trailing commas.",
            path.name,
            exc,
        )

        cleaned = remove_trailing_commas(text)
        return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def normalize_title(value) -> str:
    """
    Normalize a title for approximate cross-source matching.

    This is deliberately conservative: it normalizes case, accents,
    punctuation and parenthetical qualifiers. It is NOT full entity
    resolution.
    """
    if pd.isna(value):
        return ""

    value = re.sub(r"[™®©]", "", str(value))
    value = unicodedata.normalize("NFKC", value).casefold()

    value = re.sub(r"\([^)]*\)", " ", value)

    value = re.sub(r"\[\s*relisted\s*\]", " ", value)

    # Preserve Unicode letters/numbers instead of forcing ASCII.
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def earliest_year(value):
    """
    Extract the first 4-digit year from free-form source text.
    """
    if pd.isna(value):
        return pd.NA

    years = re.findall(r"\b(?:19|20)\d{2}\b", str(value))

    if years:
        return min(int(year) for year in years)

    return pd.NA


def serialize_complex_values(value):
    """
    Make list/dict source fields safe for CSV export without discarding them.
    """
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)

    return value


def normalize_yes_no(value):
    """
    Normalize values such as 'Yes', 'Yes*', 'No', and 'No*'.

    Returns True/False when the value is recognizable, otherwise pd.NA.
    """
    if pd.isna(value):
        return pd.NA

    normalized = str(value).strip().casefold().replace("*", "")

    if normalized == "yes":
        return True
    if normalized == "no":
        return False

    return pd.NA


def prefix_raw_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Preserve every original source field under a source-specific raw prefix.
    """
    raw = df.copy()

    for column in raw.columns:
        raw[column] = raw[column].map(serialize_complex_values)

    raw.columns = [
        f"raw__{prefix}__{column.replace('.', '__').replace(' ', '_')}"
        for column in raw.columns
    ]

    return raw


# ---------------------------------------------------------------------------
# Source-specific normalization
# ---------------------------------------------------------------------------

def normalize_dead_game_list(records) -> pd.DataFrame:
    source = pd.DataFrame(records)

    canonical = pd.DataFrame(
        {
            "source": "dead_game_list",
            "source_record_id": source.index.astype(str),
            "game_name": source["Game"],
            "normalized_title": source["Game"].map(normalize_title),
            "publisher": source["Publisher"],
            "developer": pd.NA,
            "platforms": source["Platform(s)"],
            "release_info": source["Release date"],
            "event_info": source["Death date"],
            "release_year": source["Release date"].map(earliest_year),
            "event_year": source["Death date"].map(earliest_year),
            "source_status": source["State"],
            "is_explicitly_delisted": pd.NA,
            "is_historically_unavailable": source["State"].isin(
                [
                    "Dead",
                    "Fan-Resurrected",
                    "Fan-Preserved",
                    "Dev-Preserved",
                ]
            ),
            "is_dead": source["State"].eq("Dead"),
            "is_at_risk": source["State"].eq("At Risk"),
            "offline_play": pd.NA,
            "offline_play_bool": pd.NA,
            "download_required": pd.NA,
            "download_required_bool": pd.NA,
        }
    )

    raw = prefix_raw_columns(source, "deadgamelist")

    return pd.concat(
        [canonical.reset_index(drop=True), raw.reset_index(drop=True)],
        axis=1,
    )


def normalize_delisted_games(records) -> pd.DataFrame:
    source = pd.json_normalize(records)

    canonical = pd.DataFrame(
        {
            "source": "delisted_games",
            "source_record_id": source["id"].astype(str),
            "game_name": source["data.title"],
            "normalized_title": source["data.title"].map(normalize_title),
            "publisher": source["data.publisher"],
            "developer": source["data.developer"],
            "platforms": source["data.platform"],
            "release_info": source["data.digital_release"],
            "event_info": source["data.delisting"],
            "release_year": source["data.digital_release"].map(earliest_year),
            "event_year": source["data.delisting"].map(earliest_year),
            "source_status": "Historically delisted",
            # This source is explicitly a historical delisting catalogue.
            "is_explicitly_delisted": True,
            "is_historically_unavailable": True,
            "is_dead": pd.NA,
            "is_at_risk": False,
            "offline_play": pd.NA,
            "offline_play_bool": pd.NA,
            "download_required": pd.NA,
            "download_required_bool": pd.NA,
        }
    )

    raw = prefix_raw_columns(source, "delistedgames")

    return pd.concat(
        [canonical.reset_index(drop=True), raw.reset_index(drop=True)],
        axis=1,
    )


def normalize_steam_tracker(records) -> pd.DataFrame:
    source = pd.DataFrame(records)

    explicit_delisted = source["Type"].eq("Delisted")

    unavailable_types = {
        "Delisted",
        "Purchase disabled",
        "F2P (unavailable)",
        "Retail only",
    }

    canonical = pd.DataFrame(
        {
            "source": "steam_tracker",
            "source_record_id": source["AppID"].astype(str),
            "game_name": source["Name"],
            "normalized_title": source["Name"].map(normalize_title),
            "publisher": pd.NA,
            "developer": pd.NA,
            "platforms": "Steam / PC",
            "release_info": pd.NA,
            "event_info": source["Changed"],
            "release_year": pd.NA,
            "event_year": source["Changed"].map(earliest_year),
            "source_status": source["Type"],
            "is_explicitly_delisted": explicit_delisted,
            "is_historically_unavailable": source["Type"].isin(
                unavailable_types
            ),
            "is_dead": pd.NA,
            "is_at_risk": False,
            "offline_play": pd.NA,
            "offline_play_bool": pd.NA,
            "download_required": pd.NA,
            "download_required_bool": pd.NA,
        }
    )

    raw = prefix_raw_columns(source, "steamtracker")

    return pd.concat(
        [canonical.reset_index(drop=True), raw.reset_index(drop=True)],
        axis=1,
    )


def normalize_doesitplay(records) -> pd.DataFrame:
    """
    Normalize DoesItPlay test rows.

    A row describes a platform/version/region/test-device observation.
    DoesItPlay provides dependency features, not a delisting outcome.
    """
    source = pd.DataFrame(records)

    offline_bool = source["OfflinePlay"].map(normalize_yes_no)
    download_bool = source["DownloadRequired"].map(normalize_yes_no)

    source_status = (
        "Offline=" + source["OfflinePlay"].fillna("").astype(str).str.strip()
        + "; DownloadRequired="
        + source["DownloadRequired"].fillna("").astype(str).str.strip()
    )

    canonical = pd.DataFrame(
        {
            "source": "doesitplay",
            "source_record_id": source["No"].astype(str),
            "game_name": source["Game"],
            "normalized_title": source["Game"].map(normalize_title),
            "publisher": pd.NA,
            "developer": pd.NA,
            "platforms": source["Platform"],
            "release_info": pd.NA,
            "event_info": pd.NA,
            "release_year": pd.NA,
            "event_year": pd.NA,
            "source_status": source_status,
            "is_explicitly_delisted": pd.NA,
            "is_historically_unavailable": pd.NA,
            "is_dead": pd.NA,
            "is_at_risk": pd.NA,
            "offline_play": source["OfflinePlay"],
            "offline_play_bool": offline_bool,
            "download_required": source["DownloadRequired"],
            "download_required_bool": download_bool,
        }
    )

    raw = prefix_raw_columns(source, "doesitplay")

    return pd.concat(
        [canonical.reset_index(drop=True), raw.reset_index(drop=True)],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Unified dataframe
# ---------------------------------------------------------------------------

def create_unified_dataframe(
    dead_games_records,
    delisted_games_records,
    steam_tracker_records,
    doesitplay_records,
) -> pd.DataFrame:
    """
    Create one unified dataframe while retaining every original source field.
    """
    frames = [
        normalize_dead_game_list(dead_games_records),
        normalize_delisted_games(delisted_games_records),
        normalize_steam_tracker(steam_tracker_records),
        normalize_doesitplay(doesitplay_records),
    ]

    unified = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    non_game_type = (
        unified["source"].eq("steam_tracker")
        & unified["source_status"].isin(NON_GAME_STEAM_TYPES)
    )
    non_game_title = unified["game_name"].astype(str).str.contains(
        NON_GAME_TITLE_PATTERN, case=False, regex=True
    )

    # Titles with no letters/digits (e.g. "**", "#") cannot be matched.
    empty_title = unified["normalized_title"].eq("")

    return unified[
        ~(non_game_type | non_game_title | empty_title)
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def safe_sum(series: pd.Series):
    """
    Sum booleans while handling nullable/mixed columns.
    """
    return int(series.astype("boolean").sum(skipna=True))


def compute_source_statistics(
    unified: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for source_name, group in unified.groupby("source", sort=True):
        nonempty_titles = group.loc[
            group["normalized_title"].ne(""),
            "normalized_title",
        ]

        rows.append(
            {
                "source": source_name,
                "records": len(group),
                "unique_normalized_titles": nonempty_titles.nunique(),
                "duplicate_title_rows": int(
                    nonempty_titles.duplicated().sum()
                ),
                "explicitly_delisted_records": safe_sum(
                    group["is_explicitly_delisted"]
                ),
                "historically_unavailable_records": safe_sum(
                    group["is_historically_unavailable"]
                ),
                "dead_records": safe_sum(group["is_dead"]),
                "at_risk_records": safe_sum(group["is_at_risk"]),
                "release_year_available": int(
                    group["release_year"].notna().sum()
                ),
                "event_year_available": int(
                    group["event_year"].notna().sum()
                ),
                "offline_play_yes": safe_sum(
                    group["offline_play_bool"]
                ),
                "offline_play_no": safe_sum(
                    group["offline_play_bool"].eq(False)
                ),
                "download_required_yes": safe_sum(
                    group["download_required_bool"]
                ),
                "download_required_no": safe_sum(
                    group["download_required_bool"].eq(False)
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_status_statistics(
    unified: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep source-specific status semantics rather than forcing incompatible
    states into one artificial label.
    """
    stats = (
        unified.groupby(
            ["source", "source_status"],
            dropna=False,
        )
        .size()
        .reset_index(name="records")
    )

    stats["percentage_within_source"] = (
        stats["records"]
        / stats.groupby("source")["records"].transform("sum")
        * 100
    ).round(2)

    return stats


def compute_cross_source_overlap(
    unified: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate cross-source overlap using normalized titles.

    Title normalization is useful for exploratory statistics but should not
    be treated as perfect entity resolution.
    """
    title_sets = {}

    for source_name, group in unified.groupby("source"):
        title_sets[source_name] = set(
            group.loc[
                group["normalized_title"].ne(""),
                "normalized_title",
            ]
        )

    sources = sorted(title_sets)

    rows = []

    for i, source_a in enumerate(sources):
        for source_b in sources[i + 1:]:
            overlap = title_sets[source_a] & title_sets[source_b]

            rows.append(
                {
                    "source_a": source_a,
                    "source_b": source_b,
                    "common_normalized_titles": len(overlap),
                }
            )

    if len(sources) >= 3:
        common_all = set.intersection(
            *(title_sets[source] for source in sources)
        )

        rows.append(
            {
                "source_a": "ALL_SOURCES",
                "source_b": "ALL_SOURCES",
                "common_normalized_titles": len(common_all),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Modeling readiness
# ---------------------------------------------------------------------------

def assess_delisting_prediction_readiness(
    unified: pd.DataFrame,
) -> dict:
    """
    Assess whether the current data support a prospective model predicting
    whether a currently active game will be delisted in the future.

    The current datasets are event/curation based, so this function does not
    fabricate a binary target from source membership.
    """
    source_counts = (
        unified["source"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )

    at_risk_count = safe_sum(unified["is_at_risk"])
    explicit_delisted_count = safe_sum(
        unified["is_explicitly_delisted"]
    )

    reasons_not_ready = [
        (
            "There is no representative negative/control population of "
            "ordinary active games sampled independently of delisting risk."
        ),
        (
            "The Delisted Games source is a historical delisting catalogue, "
            "so source membership already selects positive cases."
        ),
        (
            "Steam Tracker's delisted-app page primarily contains apps that "
            "are already unavailable, purchase-disabled, delisted, unreleased, "
            "test apps, or otherwise exceptional."
        ),
        (
            "The Dead Game List contains useful At Risk and Dead states, but "
            "it is still a curated risk/death list rather than a population "
            "sample of active games."
        ),
        (
            "DoesItPlay adds a broader reference set and useful pre-event "
            "features such as offline play and download dependency, but its "
            "rows do not provide a time-aligned delisting outcome."
        ),
        (
            "Fields such as delisting date, death date, Changed, State, and "
            "post-event Notes would leak the outcome if used as predictors."
        ),
        (
            "The sources use different semantics: store delisting, purchase "
            "disablement, server/game death, preservation, offline play, and "
            "download dependency are not the same event."
        ),
    ]

    recommended_solutions = [
        (
            "Collect a control cohort of active/non-delisted games from Steam "
            "or another store, ideally using the same observation dates as "
            "the positive cases."
        ),
        (
            "Define a precise prediction target and horizon, e.g. "
            "'delisted from Steam within the next 12 months'."
        ),
        (
            "Create historical snapshots so every feature is measured before "
            "the prediction cutoff; exclude post-delisting text/status/date "
            "fields from model inputs."
        ),
        (
            "Use AppID/store IDs for entity resolution where possible, and "
            "use normalized/fuzzy title matching only as a fallback."
        ),
        (
            "Use DoesItPlay's OfflinePlay and DownloadRequired fields as "
            "candidate dependency features after entity resolution, while "
            "keeping platform/version/region/test-device context."
        ),
        (
            "Add other pre-event features such as game age, publisher/developer "
            "delisting history, player-count trend, review trend, update "
            "recency, price/discount history, store tags, multiplayer/online "
            "dependency, DLC activity, and build/depot activity."
        ),
        (
            "Distinguish store delisting from server shutdown/game death. "
            "Consider separate labels or a multi-task formulation."
        ),
        (
            "Use a time-based train/validation/test split instead of a random "
            "split once historical snapshots are available."
        ),
        (
            "For time-to-delisting, consider survival analysis with "
            "right-censored active games instead of forcing a binary label."
        ),
        (
            "If negatives remain incomplete, investigate positive-unlabeled "
            "(PU) learning rather than treating unknown games as true negatives."
        ),
        (
            "Match controls to delisted games by release year, platform, "
            "genre/popularity, and publisher size to reduce selection bias."
        ),
    ]

    return {
        "ready_for_prospective_delisting_prediction": False,
        "requested_target": FUTURE_TARGET_COLUMN,
        "total_records": int(len(unified)),
        "unique_normalized_titles": int(
            unified.loc[
                unified["normalized_title"].ne(""),
                "normalized_title",
            ].nunique()
        ),
        "source_counts": source_counts,
        "explicit_delisted_records": explicit_delisted_count,
        "at_risk_records": at_risk_count,
        "reasons_not_ready": reasons_not_ready,
        "recommended_solutions": recommended_solutions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dead_games_records = load_json(
        DEAD_GAMES_FILE
    )

    delisted_games_records = load_json(
        DELISTED_GAMES_FILE
    )

    steam_tracker_records = load_json(
        STEAM_TRACKER_FILE
    )

    doesitplay_records = load_json(
        DOESITPLAY_FILE
    )

    unified = create_unified_dataframe(
        dead_games_records,
        delisted_games_records,
        steam_tracker_records,
        doesitplay_records,
    )

    source_stats = compute_source_statistics(
        unified
    )

    status_stats = compute_status_statistics(
        unified
    )

    overlap_stats = compute_cross_source_overlap(
        unified
    )

    readiness = assess_delisting_prediction_readiness(
        unified
    )

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    unified.to_csv(
        UNIFIED_CSV,
        index=False,
    )

    source_stats.to_csv(
        SOURCE_STATS_CSV,
        index=False,
    )

    status_stats.to_csv(
        STATUS_STATS_CSV,
        index=False,
    )

    overlap_stats.to_csv(
        OVERLAP_CSV,
        index=False,
    )

    MODEL_READINESS_JSON.write_text(
        json.dumps(
            readiness,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    print("\n=== SOURCE STATISTICS ===")
    print(
        source_stats.to_string(
            index=False
        )
    )

    print("\n=== STATUS DISTRIBUTIONS ===")
    print(
        status_stats.to_string(
            index=False
        )
    )

    print("\n=== CROSS-SOURCE TITLE OVERLAP ===")
    print(
        overlap_stats.to_string(
            index=False
        )
    )

    print("\n=== MODEL READINESS ===")
    print(
        "Ready for prospective delisting prediction:",
        readiness["ready_for_prospective_delisting_prediction"],
    )

    print("\nReasons:")
    for reason in readiness["reasons_not_ready"]:
        print(f"  - {reason}")

    print("\nRecommended next steps:")
    for solution in readiness["recommended_solutions"]:
        print(f"  - {solution}")

    print("\nFiles written:")
    print(f"  - {UNIFIED_CSV}")
    print(f"  - {SOURCE_STATS_CSV}")
    print(f"  - {STATUS_STATS_CSV}")
    print(f"  - {OVERLAP_CSV}")
    print(f"  - {MODEL_READINESS_JSON}")


if __name__ == "__main__":
    main()
