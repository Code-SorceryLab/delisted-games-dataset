#!/usr/bin/env python3

"""
Check the LLM answers automatically and build the final cause labels.

For every cause a model flags, its quote must appear in the cleaned narrative
the model read (whitespace, quote marks and case normalised; fuzzy match with
rapidfuzz partial_ratio >= QUOTE_MIN_SCORE = 95). A cause without a supporting
quote counts as false. The two models are then compared per cause (Cohen's
kappa and % agreement) on the entries both classified.

Final labels are the primary model's verified labels (MODELS[0]); the
`<cause>_agree` columns keep a cause only when both models flag it with a
verified quote, as a robustness check.

Inputs:
    - llm_cache/<model_tag>.jsonl         (classify_delisting_causes.py)
    - the cleaned narratives               (clean_narrative, shared)

Outputs (next to this script):
    - delisting_cause_labels.csv           one row per entry, final labels
    - spot_check_sample.csv                optional human check: up to SPOT_CHECK_PER_CAUSE
                                           random "yes" answers per cause with a
                                           blank `correct` column. Written once and
                                           never overwritten.

The quote check, the model agreement and, once `correct` has been filled in
(y/n), the spot-check precision are printed to the log.
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from classify_delisting_causes import (
    CAUSES,
    MODELS,
    load_cache,
    load_entries,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

LABELS_CSV = BASE_DIR / "delisting_cause_labels.csv"
SPOT_CHECK_CSV = BASE_DIR / "spot_check_sample.csv"

PRIMARY, SECONDARY = MODELS[0], MODELS[1]

QUOTE_MIN_SCORE = 95   # real quotes with spelling variants score 97+; stitched quotes ~91
QUOTE_MIN_WORDS = 3   # a one- or two-word "quote" is not evidence

SPOT_CHECK_PER_CAUSE = 20
SEED = 42

LABELS = CAUSES + ["other", "not_stated"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quote check
# ---------------------------------------------------------------------------

CHAR_MAP = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "`": "'",
    "–": "-", "—": "-", "…": "...", "\xa0": " ",
})


def normalise(text: str) -> str:
    return " ".join(text.translate(CHAR_MAP).lower().split())


def quote_found(quote: str, narrative: str) -> bool:
    quote = normalise(quote or "").strip(" .\"'")

    if len(quote.split()) < QUOTE_MIN_WORDS:
        return False

    text = normalise(narrative)

    return quote in text or fuzz.partial_ratio(quote, text) >= QUOTE_MIN_SCORE


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def final_labels(answer: dict, narrative: str, title: str) -> dict:
    """
    Verified labels of one model answer.

    A cause holds only with a verified quote. With no cause left, the entry is
    OTHER if the model gave another reason (other / other_reason) or flagged a
    cause whose quote failed, and NOT_STATED otherwise.
    """
    row = {}
    flagged_any = False

    for cause in CAUSES:
        item = answer.get(cause) or {}
        flagged = bool(item.get("applies"))
        verified = flagged and quote_found(item.get("quote", ""), narrative)
        flagged_any |= flagged
        row[cause] = verified
        row[f"{cause}_flagged"] = flagged
        row[f"{cause}_quote"] = item.get("quote", "") if verified else ""

    other_reason = (answer.get("other_reason") or "").strip()

    if any(row[cause] for cause in CAUSES):
        row["other"] = row["not_stated"] = False
    elif answer.get("other") and not answer.get("not_stated"):
        row["other"], row["not_stated"] = True, False
    elif answer.get("not_stated") and not answer.get("other"):
        row["other"], row["not_stated"] = False, True
    else:
        row["other"] = bool(other_reason) or flagged_any
        row["not_stated"] = not row["other"]

    row["other_reason"] = other_reason if row["other"] else ""
    row["speculative"] = bool(answer.get("speculative")) and not row["not_stated"]
    row["relisted"] = bool(answer.get("relisted")) or "[RELISTED]" in title.upper()
    row["is_bundle"] = bool(answer.get("is_bundle"))

    return row


def model_labels(model: str, entries: pd.DataFrame) -> pd.DataFrame:
    """Verified labels of one model, one row per cached entry."""
    lines = {line["id"]: line for line in load_cache(model)}
    rows = []

    for entry in entries.itertuples(index=False):
        line = lines.get(entry.id)
        if line is None:
            continue
        rows.append({"id": entry.id, **final_labels(line["answer"], entry.narrative, entry.title)})

    missing = len(entries) - len(rows)
    if missing:
        logger.warning("%s: %d of %d entries are not cached.", model, missing, len(entries))

    return pd.DataFrame(rows).set_index("id")


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    a, b = a.astype(bool), b.astype(bool)
    observed = (a == b).mean()
    expected = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())

    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def quote_check_table(labels_by_model: dict) -> pd.DataFrame:
    rows = []

    for model, labels in labels_by_model.items():
        for cause in CAUSES:
            flagged = int(labels[f"{cause}_flagged"].sum())
            found = int(labels[cause].sum())
            rows.append({
                "model": model,
                "cause": cause,
                "flagged": flagged,
                "quote_found": found,
                "quote_not_found": flagged - found,
                "share_not_found": round((flagged - found) / flagged, 4) if flagged else 0.0,
            })

    return pd.DataFrame(rows)


def agreement_table(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    common = primary.index.intersection(secondary.index)
    p, s = primary.loc[common], secondary.loc[common]
    rows = []

    for label in LABELS:
        rows.append({
            "label": label,
            "n_entries": len(common),
            f"yes_{PRIMARY}": int(p[label].sum()),
            f"yes_{SECONDARY}": int(s[label].sum()),
            "yes_both": int((p[label] & s[label]).sum()),
            "agreement_pct": round(100 * (p[label] == s[label]).mean(), 2),
            "cohen_kappa": round(cohen_kappa(p[label], s[label]), 4),
        })

    return pd.DataFrame(rows)


def build_labels(entries: pd.DataFrame, primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    base = entries.set_index("id")[["title", "platform"]]
    keep = LABELS + ["other_reason", "speculative", "relisted", "is_bundle"]
    labels = base.join(primary[keep], how="inner")

    agreed = secondary.reindex(labels.index)
    for cause in CAUSES:
        labels[f"{cause}_agree"] = labels[cause] & agreed[cause].fillna(False).astype(bool)

    labels["n_causes"] = labels[CAUSES].sum(axis=1)
    labels["evidence"] = primary.loc[labels.index].apply(
        lambda row: json.dumps(
            {cause: row[f"{cause}_quote"] for cause in CAUSES if row[cause]},
            ensure_ascii=False,
        ),
        axis=1,
    )

    return labels.reset_index()


def write_spot_check(labels: pd.DataFrame, entries: pd.DataFrame) -> None:
    if SPOT_CHECK_CSV.exists():
        logger.info("%s exists; not overwritten.", SPOT_CHECK_CSV.name)
        return

    if len(labels) < len(entries):
        logger.info("Spot-check sample skipped until every entry is classified.")
        return

    narratives = entries.set_index("id")["narrative"]
    evidence = labels.set_index("id")["evidence"].apply(json.loads)
    rows = []

    for cause in CAUSES:
        yes = labels.loc[labels[cause], "id"]
        for entry_id in yes.sample(min(SPOT_CHECK_PER_CAUSE, len(yes)), random_state=SEED):
            rows.append({
                "cause": cause,
                "id": entry_id,
                "title": labels.set_index("id").at[entry_id, "title"],
                "quote": evidence[entry_id][cause],
                "narrative": narratives[entry_id],
                "correct": "",
            })

    pd.DataFrame(rows).to_csv(SPOT_CHECK_CSV, index=False)
    logger.info("Wrote %s (%d rows). Fill `correct` with y/n.", SPOT_CHECK_CSV.name, len(rows))


def spot_check_precision() -> None:
    if not SPOT_CHECK_CSV.exists():
        return

    sample = pd.read_csv(SPOT_CHECK_CSV, dtype={"correct": str})
    answered = sample[sample["correct"].fillna("").str.strip().str.lower().isin(["y", "n"])]

    if answered.empty:
        return

    answered = answered.assign(correct=answered["correct"].str.strip().str.lower() == "y")
    precision = answered.groupby("cause")["correct"].agg(checked="size", correct="sum")
    precision["precision"] = (precision["correct"] / precision["checked"]).round(3)
    logger.info("Spot-check precision:\n%s", precision.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    entries = load_entries()

    labels_by_model = {model: model_labels(model, entries) for model in (PRIMARY, SECONDARY)}

    quote_check = quote_check_table(labels_by_model)
    logger.info("Quote check:\n%s", quote_check.to_string(index=False))

    agreement = agreement_table(labels_by_model[PRIMARY], labels_by_model[SECONDARY])
    logger.info("Model agreement:\n%s", agreement.to_string(index=False))

    labels = build_labels(entries, labels_by_model[PRIMARY], labels_by_model[SECONDARY])
    labels.to_csv(LABELS_CSV, index=False)
    logger.info("Wrote %s (%d entries).", LABELS_CSV.name, len(labels))

    write_spot_check(labels, entries)
    spot_check_precision()


if __name__ == "__main__":
    main()
