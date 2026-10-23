"""Merge rule-based and LLM taxonomy labels into one file for all titles.

Inputs:
    Analysis/taxonomy_rules_all.csv   from classify_all_titles.py
    Analysis/taxonomy_llm_haiku.csv   Haiku labels for titles with free text
                                      or several sources (needs_llm == True)
Output:
    Analysis/taxonomy_full_classification.csv

Titles with needs_llm take the Haiku labels; all others keep the rule labels.
D5 counts official servers only: a title whose D7 is Fan-Preserved or
Fan-Resurrected cannot be D5 = Active (fan-run servers are captured by D7).
Taxonomy v2 refinements are applied to the Haiku labels, which predate them
(the rule labels already follow them):
  - D4 Minor -> Optional Multiplayer when a DeadGameWiki entry with
    Single_Player_at_risk = No names multiplayer, or the rationale does;
  - D6 -> Legacy OS when every listed platform is a discontinued runtime/OS.
Finally, the manually reviewed sample (taxonomy_sample_classification.csv)
overrides all other labels (method = "manual review").

Usage: python Scripts/merge_taxonomy_classification.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_taxonomy_classification import CURRENT, MULTIPLAYER, is_legacy_os, s  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RULES_CSV = ROOT / "Analysis" / "taxonomy_rules_all.csv"
LLM_CSV = ROOT / "Analysis" / "taxonomy_llm_haiku.csv"
SAMPLE_CSV = ROOT / "Analysis" / "taxonomy_sample_classification.csv"
GAMES_CSV = ROOT / "Analysis" / "unified_games.csv"
OUT_CSV = ROOT / "Analysis" / "taxonomy_full_classification.csv"

DIMENSIONS = {
    "D1": "D1_digital_availability",
    "D2": "D2_physical_availability",
    "D3": "D3_medium_completeness",
    "D4": "D4_online_dependency",
    "D5": "D5_service_status",
    "D6": "D6_platform_accessibility",
    "D7": "D7_preservation",
}
FAN_PRESERVED = {"Fan-Preserved", "Fan-Resurrected"}


def multiplayer_titles():
    """Titles with a DeadGameWiki entry whose single-player survives and whose
    name or notes say the online part is multiplayer."""
    g = pd.read_csv(GAMES_CSV, low_memory=False)
    dgl = g[(g["source"] == "dead_game_list")
            & (g["raw__deadgamelist__Single_Player_at_risk"] == "No")]
    text = dgl["game_name"].map(s) + " " + dgl["raw__deadgamelist__Notes"].map(s)
    return set(dgl.loc[text.str.contains(MULTIPLAYER), "normalized_title"])


def refine(labels, title, platform, rationale, mp_titles):
    """Apply the taxonomy v2 values to labels made before they existed."""
    changes = []
    if labels["D4"] == "Minor" and (title in mp_titles or MULTIPLAYER.search(rationale)):
        labels["D4"] = "Optional Multiplayer"
        changes.append("D4 Minor -> Optional Multiplayer")
    if labels["D6"] == CURRENT and is_legacy_os(platform):
        labels["D6"] = "Legacy OS"
        changes.append("D6 -> Legacy OS")
    return "; ".join(changes)


def main():
    rules = pd.read_csv(RULES_CSV, keep_default_na=False)
    llm = pd.read_csv(LLM_CSV, keep_default_na=False).set_index("title_id")
    sample = pd.read_csv(SAMPLE_CSV, keep_default_na=False).set_index("normalized_title")
    mp_titles = multiplayer_titles()

    rows, missing = [], []
    for _, r in rules.iterrows():
        use_llm = r["needs_llm"] == "True" or r["needs_llm"] is True
        if use_llm and r["normalized_title"] not in llm.index:
            missing.append(r["normalized_title"])
            use_llm = False
        title = r["normalized_title"]
        src = llm.loc[title] if use_llm else r
        labels = {k: src[k] for k in DIMENSIONS}
        refined = refine(labels, title, r["platform"], src["rationale"], mp_titles) if use_llm else ""
        if labels["D7"] in FAN_PRESERVED and labels["D5"] == "Active":
            labels["D5"] = "Shut down"
        method = "LLM (Haiku)" if use_llm else "rules"
        confidence = src["confidence"] if use_llm else ""
        evidence = src["rationale"] if use_llm else r["evidence"]
        note = ""
        if title in sample.index:
            m = sample.loc[title]
            labels = {k: m[col] for k, col in DIMENSIONS.items()}
            method, confidence, evidence, note, refined = (
                "manual review", "", m["evidence"], m["manual_note"], "")
        rows.append({
            "normalized_title": r["normalized_title"],
            "game_name": r["game_name"],
            "stratum_source": r["stratum_source"],
            "all_sources": r["all_sources"],
            "platform": r["platform"],
            **{col: labels[k] for k, col in DIMENSIONS.items()},
            "method": method,
            "confidence": confidence,
            "evidence": evidence,
            "manual_note": note,
            "refinement": refined,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(out)} titles -> {OUT_CSV.name}: "
          + ", ".join(f"{n} {m}" for m, n in out["method"].value_counts().items())
          + f"; {(out.refinement != '').sum()} Haiku labels refined")
    if missing:
        print(f"WARNING: {len(missing)} titles had no LLM labels, kept rules: {missing}")


if __name__ == "__main__":
    main()
