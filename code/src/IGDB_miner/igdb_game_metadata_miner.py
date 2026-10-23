#!/usr/bin/env python3

"""
IGDB enrichment miner for the unified games dataset.

This script is intentionally limited to the same IGDB access pattern used by
the provided miner skeleton:

    - requests.post
    - pandas
    - config.py
    - IGDB endpoints already declared in config.py

It implements two workflows:

1. match_genres_for_normalized_games(...)
   Match every unique normalized game title in the unified dataset against
   IGDB and export its IGDB genres.

2. compute_games_released_by_year(...)
   Count the number of IGDB games whose `first_release_date` falls in each
   requested calendar year. These counts can later be used as a denominator
   when studying historical delisting counts.

No OAuth/token creation is implemented here. The script expects CLIENT_ID and
token to be supplied through the existing config.py file.

Inputs:
    - ../Analysis/unified_games.csv
    - genres.csv
    - config.py

Outputs (next to this script):
    - igdb_genres_by_normalized_game.csv
    - igdb_games_released_by_year.csv
"""

from requests import post
import pandas as pd
import config as cf
import ast
import os
import re
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

UNIFIED_GAMES_CSV = BASE_DIR.parent / "Analysis" / "unified_games.csv"
GENRES_CSV = BASE_DIR / "genres.csv"

GENRE_OUTPUT_CSV = BASE_DIR / "igdb_genres_by_normalized_game.csv"
YEAR_OUTPUT_CSV = BASE_DIR / "igdb_games_released_by_year.csv"

IGDB_SEARCH_LIMIT = 20
IGDB_PAGE_SIZE = 500

RELEASE_GAME_TYPE_IDS = (0, 8, 9, 10, 11)


# ---------------------------------------------------------------------------
# Shared IGDB helpers
# ---------------------------------------------------------------------------

def build_igdb_headers(query: str) -> dict:
    """
    Build the request payload using the same authentication mechanism as the
    provided miner skeleton.

    The token itself is NOT created by this script.
    """
    return {
        "headers": {
            "Client-ID": cf.CLIENT_ID,
            "Authorization": "Bearer " + cf.token,
        },
        "data": query,
    }


def post_igdb(url: str, query: str):
    """
    Execute one IGDB POST request.

    Retries use the MAX_RETRIES value already provided by config.py.
    No additional HTTP/client library is introduced.
    """
    last_error = None

    max_retries = getattr(
        cf,
        "MAX_RETRIES",
        3,
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = post(
                url,
                **build_igdb_headers(query),
            )

            response.raise_for_status()

            return response

        except Exception as exc:
            last_error = exc

            print(
                f"IGDB request failed "
                f"(attempt {attempt}/{max_retries}): {exc}"
            )

    raise RuntimeError(
        "IGDB request failed after all retries."
    ) from last_error


def escape_igdb_search_text(value: str) -> str:
    """
    Escape characters that would otherwise break an IGDB search string.
    """
    value = str(value)

    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')

    return value


def normalize_for_match(value) -> str:
    """
    Normalize an IGDB title in the same general form used by the unified
    dataset's normalized_title field.

    This normalization is used only to validate candidate matches.
    """
    if value is None or pd.isna(value):
        return ""

    value = re.sub(
        r"[™®©]",
        "",
        str(value),
    )

    value = unicodedata.normalize(
        "NFKC",
        value,
    ).casefold()

    value = re.sub(
        r"\([^)]*\)",
        " ",
        value,
    )

    value = re.sub(
        r"[^\w]+",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = value.replace(
        "_",
        " ",
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def parse_id_list(value) -> list[int]:
    """
    Convert a genre/platform-style IGDB ID field into a list of integers.

    Handles:
        [4, 5]
        "[4, 5]"
        "4,5"
        4
    """
    if value is None:
        return []

    if isinstance(value, list):
        result = []

        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                pass

        return result

    if pd.isna(value):
        return []

    if isinstance(value, str):
        stripped = value.strip()

        if not stripped:
            return []

        try:
            parsed = ast.literal_eval(stripped)

            if isinstance(parsed, list):
                return [
                    int(item)
                    for item in parsed
                    if str(item).strip()
                ]

        except (ValueError, SyntaxError):
            pass

        result = []

        for item in stripped.split(","):
            item = item.strip()

            if not item:
                continue

            try:
                result.append(int(item))
            except ValueError:
                pass

        return result

    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def load_genre_mapping(
    genres_csv: str = GENRES_CSV,
) -> dict[int, str]:
    """
    Load the local IGDB genre mapping supplied with the miner.

    Expected columns:
        genre_id
        genre
    """
    genres_df = pd.read_csv(
        genres_csv
    )

    required_columns = {
        "genre_id",
        "genre",
    }

    missing = (
        required_columns
        - set(genres_df.columns)
    )

    if missing:
        raise ValueError(
            "genres.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    genres_df["genre_id"] = pd.to_numeric(
        genres_df["genre_id"],
        errors="coerce",
    )

    genres_df = genres_df.dropna(
        subset=["genre_id"]
    )

    return {
        int(row["genre_id"]): safe_text(row["genre"])
        for _, row in genres_df.iterrows()
        if safe_text(row["genre"])
    }


def genre_ids_to_names(
    genre_ids,
    genre_mapping: dict[int, str],
) -> list[str]:
    """
    Resolve IGDB genre IDs using the provided local genre table.
    """
    ids = parse_id_list(
        genre_ids
    )

    return [
        safe_text(genre_mapping[genre_id])
        for genre_id in ids
        if genre_id in genre_mapping
        and safe_text(genre_mapping[genre_id])
    ]



def safe_text(value, fallback: str = "") -> str:
    """
    Convert malformed/missing scalar values to a safe string.

    In particular, pandas may represent missing game names as float NaN.
    This helper prevents expressions such as:

        NaN + " | retry: ..."

    from raising TypeError.
    """
    if value is None:
        return fallback

    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if not value:
        return fallback

    return value


def first_valid_text(values, fallback: str = "") -> str:
    """
    Return the first non-empty, non-null textual value from a pandas Series.
    """
    for value in values:
        cleaned = safe_text(value)

        if cleaned:
            return cleaned

    return fallback


def write_genre_result_row(
    row: dict,
    output_csv: str,
) -> None:
    """
    Append one processed game immediately to the output CSV.

    This acts as a checkpoint: even if a later record fails unexpectedly,
    all previously processed rows remain available on disk.
    """
    write_header = not os.path.exists(output_csv)

    pd.DataFrame([row]).to_csv(
        output_csv,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# IGDB game search
# ---------------------------------------------------------------------------

def build_query_game_search(
    game_name: str,
    limit: int = IGDB_SEARCH_LIMIT,
) -> str:
    """
    Search IGDB's games endpoint for a title.

    The fields are deliberately limited to those needed by the two analyses.
    """
    escaped_name = escape_igdb_search_text(
        game_name
    )

    return (
        f'search "{escaped_name}"; '
        "fields id,name,genres,first_release_date; "
        f"limit {limit};"
    )


def search_games(
    game_name: str,
) -> pd.DataFrame:
    """
    Search the IGDB games endpoint and return the candidates as a DataFrame.
    """
    query = build_query_game_search(
        game_name
    )

    response = post_igdb(
        cf.GAMES_URL,
        query,
    )

    data = response.json()

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(
        data
    )


def select_exact_normalized_match(
    candidates: pd.DataFrame,
    normalized_title: str,
):
    """
    Return the first candidate whose IGDB name normalizes exactly to the
    unified normalized title.

    The function also returns the number of exact candidates found.
    """
    if candidates.empty:
        return None, 0

    if "name" not in candidates.columns:
        return None, 0

    candidates = candidates.copy()

    candidates["_normalized_igdb_name"] = (
        candidates["name"]
        .map(normalize_for_match)
    )

    exact = candidates[
        candidates["_normalized_igdb_name"].eq(
            normalized_title
        )
    ]

    if exact.empty:
        return None, 0

    return exact.iloc[0], len(exact)


# ---------------------------------------------------------------------------
# Workflow 1:
# Match genres for all normalized games
# ---------------------------------------------------------------------------

def build_unique_normalized_games(
    unified_csv: str = UNIFIED_GAMES_CSV,
) -> pd.DataFrame:
    """
    Build one row per normalized title across ALL sources.

    The output retains:
        - one representative original title
        - all source names containing that normalized title
        - source row count
    """
    df = pd.read_csv(
        unified_csv,
        low_memory=False,
    )

    required_columns = {
        "normalized_title",
        "game_name",
        "source",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Unified CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["normalized_title"] = (
        df["normalized_title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["normalized_title"].ne("")
    ].copy()

    grouped = (
        df.groupby(
            "normalized_title",
            as_index=False,
        )
        .agg(
            representative_game_name=(
                "game_name",
                first_valid_text,
            ),
            sources=(
                "source",
                lambda values: "|".join(
                    sorted(
                        set(
                            str(value)
                            for value in values
                            if pd.notna(value)
                        )
                    )
                ),
            ),
            source_row_count=(
                "source",
                "size",
            ),
        )
    )

    return grouped


def match_genres_for_normalized_games(
    unified_csv: str = UNIFIED_GAMES_CSV,
    genres_csv: str = GENRES_CSV,
    output_csv: str = GENRE_OUTPUT_CSV,
) -> pd.DataFrame:
    """
    Match every unique normalized title from ALL unified sources to IGDB.

    Fault tolerance
    ---------------
    - Missing/malformed game names are converted to safe strings.
    - If the representative name is missing, normalized_title is used.
    - Each game is processed inside its own try/except block.
    - Errors are recorded as rows with match_status='error'.
    - Every completed/error row is appended to the CSV immediately, so
      partial results survive later failures.

    Matching strategy
    -----------------
    1. Query IGDB using a representative original game name.
    2. Normalize IGDB candidate names.
    3. Accept a genre assignment only when an IGDB candidate normalizes
       exactly to the unified `normalized_title`.
    4. If no exact candidate is found, retry using the normalized title.
    5. If the retry still has no exact candidate, retain candidate names for
       manual inspection but DO NOT assign genres.

    Output
    ------
    One row per normalized title with:
        normalized_title
        representative_game_name
        sources
        source_row_count
        igdb_id
        igdb_name
        first_release_year
        genre_ids
        genres
        match_status
        search_candidate_count
        exact_candidate_count
        search_used
        candidate_names
        error_type
        error_message
    """
    unique_games = build_unique_normalized_games(
        unified_csv
    )

    genre_mapping = load_genre_mapping(
        genres_csv
    )

    # Start a clean output file for this run.
    if os.path.exists(output_csv):
        os.remove(output_csv)

    rows = []
    total = len(unique_games)

    for index, game in unique_games.iterrows():
        normalized_title = safe_text(
            game.get("normalized_title", "")
        )

        representative_name = safe_text(
            game.get("representative_game_name", ""),
            fallback=normalized_title,
        )

        sources = safe_text(
            game.get("sources", "")
        )

        source_row_count = game.get(
            "source_row_count",
            pd.NA,
        )

        print(
            f"[{index + 1}/{total}] "
            f"Searching IGDB: {representative_name}"
        )

        # Define a complete fallback row first.
        # If anything fails, this row is written instead of stopping the run.
        result_row = {
            "normalized_title": normalized_title,
            "representative_game_name": representative_name,
            "sources": sources,
            "source_row_count": source_row_count,
            "igdb_id": pd.NA,
            "igdb_name": pd.NA,
            "first_release_year": pd.NA,
            "genre_ids": "",
            "genres": "",
            "match_status": "error",
            "search_candidate_count": 0,
            "exact_candidate_count": 0,
            "search_used": "",
            "candidate_names": "",
            "error_type": "",
            "error_message": "",
        }

        try:
            # If both values are somehow empty, record the malformed entry.
            if not representative_name:
                raise ValueError(
                    "No usable representative game name or normalized title."
                )

            candidates = search_games(
                representative_name
            )

            match, exact_count = (
                select_exact_normalized_match(
                    candidates,
                    normalized_title,
                )
            )

            search_used = representative_name

            # Retry with normalized_title when useful.
            if (
                match is None
                and normalized_title
                and normalize_for_match(
                    representative_name
                ) != normalized_title
            ):
                retry_candidates = search_games(
                    normalized_title
                )

                if not retry_candidates.empty:
                    if not candidates.empty:
                        candidates = pd.concat(
                            [
                                candidates,
                                retry_candidates,
                            ],
                            ignore_index=True,
                        )

                        if "id" in candidates.columns:
                            candidates = (
                                candidates
                                .drop_duplicates(
                                    subset=["id"],
                                    keep="first",
                                )
                            )
                    else:
                        candidates = retry_candidates

                match, exact_count = (
                    select_exact_normalized_match(
                        candidates,
                        normalized_title,
                    )
                )

                # safe_text guarantees string operands here.
                search_used = (
                    safe_text(representative_name)
                    + " | retry: "
                    + safe_text(normalized_title)
                )

            candidate_names = ""

            if (
                not candidates.empty
                and "name" in candidates.columns
            ):
                candidate_names = " | ".join(
                    candidates["name"]
                    .dropna()
                    .map(safe_text)
                    .loc[lambda s: s.ne("")]
                    .drop_duplicates()
                    .tolist()
                )

            if match is None:
                result_row.update(
                    {
                        "match_status": (
                            "no_candidates"
                            if candidates.empty
                            else "no_exact_normalized_match"
                        ),
                        "search_candidate_count": len(
                            candidates
                        ),
                        "exact_candidate_count": 0,
                        "search_used": safe_text(
                            search_used
                        ),
                        "candidate_names": candidate_names,
                    }
                )

            else:
                genre_ids = parse_id_list(
                    match.get(
                        "genres",
                        [],
                    )
                )

                genre_names = genre_ids_to_names(
                    genre_ids,
                    genre_mapping,
                )

                release_timestamp = match.get(
                    "first_release_date",
                    pd.NA,
                )

                first_release_year = pd.NA

                if pd.notna(release_timestamp):
                    try:
                        first_release_year = (
                            pd.to_datetime(
                                int(release_timestamp),
                                unit="s",
                                utc=True,
                            ).year
                        )
                    except (
                        TypeError,
                        ValueError,
                        OverflowError,
                    ):
                        # Malformed release dates should not invalidate
                        # an otherwise valid title/genre match.
                        first_release_year = pd.NA

                result_row.update(
                    {
                        "igdb_id": match.get(
                            "id",
                            pd.NA,
                        ),
                        "igdb_name": safe_text(
                            match.get(
                                "name",
                                pd.NA,
                            )
                        ),
                        "first_release_year": (
                            first_release_year
                        ),
                        "genre_ids": "|".join(
                            str(value)
                            for value in genre_ids
                        ),
                        "genres": "|".join(
                            safe_text(value)
                            for value in genre_names
                            if safe_text(value)
                        ),
                        "match_status": (
                            "exact_normalized_match"
                        ),
                        "search_candidate_count": len(
                            candidates
                        ),
                        "exact_candidate_count": exact_count,
                        "search_used": safe_text(
                            search_used
                        ),
                        "candidate_names": candidate_names,
                    }
                )

        except Exception as exc:
            # Do not stop the complete mining process for one malformed row
            # or one failed request.
            result_row.update(
                {
                    "match_status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": safe_text(exc),
                }
            )

            print(
                f"  ERROR for '{representative_name}': "
                f"{type(exc).__name__}: {exc}"
            )

        # Keep the in-memory result.
        rows.append(
            result_row
        )

        # And checkpoint immediately to disk.
        write_genre_result_row(
            result_row,
            output_csv,
        )

    result = pd.DataFrame(
        rows
    )

    # Re-write once at the end to guarantee stable column order.
    result.to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nSaved {len(result)} normalized-game "
        f"genre records to {output_csv}"
    )

    print("\nMatch status:")
    print(
        result["match_status"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    error_count = int(
        result["match_status"]
        .eq("error")
        .sum()
    )

    print(
        f"\nRows completed with recoverable errors: "
        f"{error_count}"
    )

    return result


# ---------------------------------------------------------------------------
# Workflow 2:
# Count IGDB game releases by year
# ---------------------------------------------------------------------------

def year_to_unix_timestamp(
    year: int,
) -> int:
    """
    Convert January 1 of a year to an IGDB-compatible Unix timestamp.
    """
    timestamp = pd.Timestamp(
        year=year,
        month=1,
        day=1,
        tz="UTC",
    )

    return int(
        timestamp.timestamp()
    )


def build_query_games_for_year(
    year: int,
    offset: int = 0,
    limit: int = IGDB_PAGE_SIZE,
) -> str:
    """
    Query game IDs whose first_release_date falls inside one calendar year.
    """
    start_timestamp = (
        year_to_unix_timestamp(
            year
        )
    )

    end_timestamp = (
        year_to_unix_timestamp(
            year + 1
        )
    )

    return (
        "fields id; "
        "where "
        f"first_release_date >= {start_timestamp} "
        "& "
        f"first_release_date < {end_timestamp} "
        "& "
        "game_type = ("
        + ",".join(str(type_id) for type_id in RELEASE_GAME_TYPE_IDS)
        + "); "
        "sort id asc; "
        f"limit {limit}; "
        f"offset {offset};"
    )


def get_game_ids_released_in_year(
    year: int,
) -> set[int]:
    """
    Fetch all IGDB game IDs released in the requested year.

    Pagination uses the games endpoint and a maximum page size of 500.
    """
    game_ids = set()

    offset = 0

    while True:
        query = build_query_games_for_year(
            year=year,
            offset=offset,
            limit=IGDB_PAGE_SIZE,
        )

        response = post_igdb(
            cf.GAMES_URL,
            query,
        )

        data = response.json()

        if not data:
            break

        for record in data:
            game_id = record.get(
                "id"
            )

            if game_id is not None:
                game_ids.add(
                    int(game_id)
                )

        batch_size = len(data)

        print(
            f"  {year}: offset={offset}, "
            f"received={batch_size}, "
            f"unique_total={len(game_ids)}"
        )

        if batch_size < IGDB_PAGE_SIZE:
            break

        offset += IGDB_PAGE_SIZE

    return game_ids


def compute_games_released_by_year(
    start_year: int,
    end_year: int,
    output_csv: str = YEAR_OUTPUT_CSV,
) -> pd.DataFrame:
    """
    Count IGDB games by first-release year.

    Parameters
    ----------
    start_year:
        First calendar year to include.

    end_year:
        Last calendar year to include, inclusive.

    output_csv:
        Destination CSV.

    Output columns
    --------------
    year
        Calendar year.

    igdb_games_released
        Number of unique IGDB game IDs whose first_release_date falls in
        that year.

    Notes
    -----
    This provides a release-volume denominator. It should not automatically
    be interpreted as the number of games actively available in that year.
    """
    if start_year > end_year:
        raise ValueError(
            "start_year must be <= end_year"
        )

    rows = []

    for year in range(
        start_year,
        end_year + 1,
    ):
        print(
            f"\nCounting IGDB releases for {year}"
        )

        game_ids = (
            get_game_ids_released_in_year(
                year
            )
        )

        rows.append(
            {
                "year": year,
                "igdb_games_released": len(
                    game_ids
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nSaved yearly IGDB release counts "
        f"to {output_csv}"
    )

    return result


# ---------------------------------------------------------------------------
# Optional helper:
# derive the event-year interval from the unified dataset
# ---------------------------------------------------------------------------

def get_historical_event_year_range(
    unified_csv: str = UNIFIED_GAMES_CSV,
) -> tuple[int, int]:
    """
    Return the min/max historical event year present in the unified dataset.

    Future years are excluded.

    This helper can be used to choose the same year interval for the IGDB
    release-count denominator and the historical delisting analysis.
    """
    df = pd.read_csv(
        unified_csv,
        low_memory=False,
    )

    if "event_year" not in df.columns:
        raise ValueError(
            "Unified CSV does not contain event_year."
        )

    years = pd.to_numeric(
        df["event_year"],
        errors="coerce",
    )

    current_year = (
        pd.Timestamp.utcnow().year
    )

    years = years[
        years.notna()
        & years.le(current_year)
    ]

    if years.empty:
        raise ValueError(
            "No historical event years were found."
        )

    return (
        int(years.min()),
        int(years.max()),
    )


# ---------------------------------------------------------------------------
# Example execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---------------------------------------------------------------
    # 1. Genre enrichment for every normalized title across all sources
    # ---------------------------------------------------------------

    match_genres_for_normalized_games(
        unified_csv=UNIFIED_GAMES_CSV,
        genres_csv=GENRES_CSV,
        output_csv=GENRE_OUTPUT_CSV,
    )

    # ---------------------------------------------------------------
    # 2. Number of IGDB games released per year
    #
    # Use the historical event-year interval already present in the
    # unified dataset. If you prefer a fixed interval, replace these
    # two values manually.
    # ---------------------------------------------------------------

    start_year, end_year = (
        get_historical_event_year_range(
            UNIFIED_GAMES_CSV
        )
    )

    compute_games_released_by_year(
        start_year=start_year,
        end_year=end_year,
        output_csv=YEAR_OUTPUT_CSV,
    )
