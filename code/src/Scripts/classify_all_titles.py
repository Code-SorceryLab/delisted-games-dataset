"""Rule-based taxonomy classification for every unique title, plus input
batches for LLM classification of the titles that need judgement.

A title needs the LLM when it has free text (delistedGames, DeadGameWiki)
or appears in more than one source. All other titles are classified by the
rules in sample_taxonomy_classification.classify().

Outputs:
    Analysis/taxonomy_rules_all.csv   rule-based labels for all titles
    <batch_dir>/batch_NNN.json        LLM input, one file per batch

Usage: python Scripts/classify_all_titles.py <batch_dir> [batch_size]
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_taxonomy_classification import classify, s  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "Analysis" / "unified_games.csv"
OUT_CSV = ROOT / "Analysis" / "taxonomy_rules_all.csv"
TEXT_SOURCES = {"delisted_games", "dead_game_list"}
PRESERVED = {"Dev-Preserved", "Fan-Preserved", "Fan-Resurrected"}
INFO_CHARS = 6000

LLM_FIELDS = {
    "steam_tracker": ["raw__steamtracker__Type", "raw__steamtracker__Keyshops",
                      "raw__steamtracker__Changed"],
    "doesitplay": ["raw__doesitplay__Platform", "raw__doesitplay__Version",
                   "raw__doesitplay__Region", "raw__doesitplay__OfflinePlay",
                   "raw__doesitplay__DownloadRequired"],
    "dead_game_list": ["raw__deadgamelist__State", "raw__deadgamelist__Single_Player_at_risk",
                       "raw__deadgamelist__Subscription_(Pay_to_Play)",
                       "raw__deadgamelist__Notes"],
    "delisted_games": ["raw__delistedgames__data__digital_release",
                       "raw__delistedgames__data__delisting",
                       "raw__delistedgames__data__available_on"],
}


def info_text(value):
    value = s(value)
    if not value:
        return ""
    try:
        parts = json.loads(value)[2:]
    except json.JSONDecodeError:
        parts = [value]
    return "\n".join(p.strip() for p in parts if p.strip())[:INFO_CHARS]


def llm_record(r):
    rec = {"source": r["source"], "name": s(r["game_name"]), "platforms": s(r["platforms"]),
           "publisher": s(r["publisher"]), "release_year": s(r["release_year"]),
           "event_year": s(r["event_year"]), "status": s(r["source_status"])}
    for f in LLM_FIELDS[r["source"]]:
        if s(r[f]):
            rec[f.split("__")[-1]] = s(r[f])
    if r["source"] == "delisted_games":
        rec["info"] = info_text(r["raw__delistedgames__data__info"])
    return rec


def main():
    batch_dir = Path(sys.argv[1])
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    batch_dir.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(IN_CSV, low_memory=False)
    size = d.groupby("source")["normalized_title"].nunique().sort_values()
    rank = {src: i for i, src in enumerate(size.index)}
    d = d.assign(_r=d["source"].map(rank)).sort_values(
        ["normalized_title", "_r", "source_record_id"])

    rows, llm_items = [], []
    for title, recs in d.groupby("normalized_title", sort=False):
        prim = recs.iloc[0]
        sources = set(recs["source"])
        c = classify(prim)
        dgl = recs[recs["source"] == "dead_game_list"]
        if len(dgl):
            st = s(dgl.iloc[0]["raw__deadgamelist__State"])
            if st in PRESERVED:
                c["D7"] = st
        needs_llm = bool(sources & TEXT_SOURCES) or len(sources) > 1
        platform = s(prim["raw__doesitplay__Platform"]) or s(prim["platforms"])
        rows.append({
            "normalized_title": title, "game_name": prim["game_name"],
            "stratum_source": prim["source"],
            "all_sources": "|".join(sorted(sources)),
            "platform": platform, "needs_llm": needs_llm,
            **{f"D{i}": c[f"D{i}"] for i in range(1, 8)},
            "evidence": c["evidence"],
        })
        if needs_llm:
            llm_items.append({"title_id": title, "target_source": prim["source"],
                              "target_platform": platform,
                              "records": [llm_record(r) for _, r in recs.iterrows()]})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    for i in range(0, len(llm_items), batch_size):
        path = batch_dir / f"batch_{i // batch_size:03d}.json"
        path.write_text(json.dumps(llm_items[i:i + batch_size], ensure_ascii=False, indent=1))
    print(f"{len(rows)} titles -> {OUT_CSV.name}; {len(llm_items)} need LLM -> "
          f"{(len(llm_items) + batch_size - 1) // batch_size} batches in {batch_dir}")


if __name__ == "__main__":
    main()
