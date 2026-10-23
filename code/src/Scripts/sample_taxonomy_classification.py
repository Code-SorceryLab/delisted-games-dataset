"""Draw a stratified random sample of unique titles and pre-classify them
along the seven taxonomy dimensions using dataset fields only.

Sample: n=196 unique normalized titles (95% CI, +/-7%, N=11,762),
proportional allocation by source. A title present in several sources is
assigned to its smallest source. Classification is judged on the platform
listed in the stratum source record. 'N/A' = dimension does not apply,
'Unknown' = dataset carries no evidence.

Usage: python Scripts/sample_taxonomy_classification.py
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "Analysis" / "unified_games.csv"
OUT_CSV = ROOT / "Analysis" / "taxonomy_sample_classification.csv"
N, SEED = 196, 42

# Platforms whose hardware is no longer manufactured.
LEGACY = re.compile(
    r"ps1|psone|ps2|ps3|playstation[- ]?(1|2|3|portable)|psp|vita|xbox 360|"
    r"xbox-360|^xbox$|wii|3ds|\bds\b|nintendo-ds|gamecube|dreamcast|saturn|"
    r"n64|game ?boy|gba|neo ?geo|amiga|atari|arcade|ouya|ngage|n-gage|zeebo",
    re.I,
)

# Discontinued OS versions and runtimes on hardware that is still produced.
# A title is Legacy OS only if every platform it lists is one of these.
LEGACY_OS = re.compile(
    r"flash|shockwave|\bjava\b|unity web|web \(unity\)|ms-?dos|windows (3\.|9[58]|me\b)|mac ?os (9|classic)",
    re.I,
)

# Online play named in a source: the online part is the multiplayer modes.
MULTIPLAYER = re.compile(
    r"multi-?player|\bpvp\b|\bpve\b|co-?op|versus|matchmaking|online (?:play|modes?|match(?:es)?|races?)",
    re.I,
)

CURRENT = "Run on current hardware platform or OS"


def is_legacy(platform):
    return bool(platform) and bool(LEGACY.search(str(platform)))


def is_legacy_os(platform):
    parts = [p.strip() for p in re.split(r"[,;|\n]+", str(platform or "")) if p.strip()]
    return bool(parts) and all(LEGACY_OS.search(p) for p in parts)


def platform_access(platform):
    if is_legacy(platform):
        return "Legacy Hardware"
    return "Legacy OS" if is_legacy_os(platform) else CURRENT


def s(v):
    return "" if pd.isna(v) else str(v).strip()


def classify(r):
    """Return dict of D1..D7 plus evidence for one record."""
    src = r["source"]
    plat = s(r["platforms"])
    c = {f"D{i}": "Unknown" for i in range(1, 8)}
    ev = []

    if src == "steam_tracker":
        t = s(r["raw__steamtracker__Type"])
        c["D1"] = "Never Offered" if t == "Pre-order exclusive" else "Unavailable"
        if t == "Retail only":
            c["D2"] = "Retail"
            c["D3"] = "Key or launcher only"
        c["D6"] = CURRENT
        ev.append(f"SteamTracker type={t}")
        if s(r["raw__steamtracker__Keyshops"]):
            ev.append(f"keyshop price={s(r['raw__steamtracker__Keyshops'])}")

    elif src == "delisted_games":
        c["D1"] = "Unavailable"
        avail = s(r["raw__delistedgames__data__available_on"])
        if avail:
            ev.append("available_on=" + avail.replace("\n", " | "))
        c["D6"] = platform_access(plat)
        if re.search(r"\bretail\b", avail, re.I):
            c["D2"] = "Second-hand" if is_legacy(plat) else "Retail"

    elif src == "doesitplay":
        off = s(r["raw__doesitplay__OfflinePlay"])
        dl = s(r["raw__doesitplay__DownloadRequired"])
        c["D4"] = {"Yes": "None", "Yes*": "Minor", "No": "Core Gameplay"}.get(off, "Unknown")
        c["D3"] = {"No": "Complete", "No*": "Partial", "Yes*": "Partial",
                   "Yes": "Key or launcher only"}.get(dl, "Unknown")
        plat = s(r["raw__doesitplay__Platform"]) or plat
        c["D6"] = platform_access(plat)
        if c["D6"] == "Legacy Hardware":
            c["D2"] = "Second-hand"
        if c["D4"] == "None":
            c["D5"] = "N/A"
        ev.append(f"doesitplay Offline={off}; Download={dl}; "
                  f"version={s(r['raw__doesitplay__Version'])}")

    elif src == "dead_game_list":
        state = s(r["raw__deadgamelist__State"])
        # List only tracks online-dependent games; if single-player survives
        # the shutdown, only multiplayer (when the entry names it) or
        # secondary features depend on the connection.
        sp = s(r["raw__deadgamelist__Single_Player_at_risk"])
        if sp == "No":
            text = f"{s(r['game_name'])} {s(r['raw__deadgamelist__Notes'])}"
            c["D4"] = "Optional Multiplayer" if MULTIPLAYER.search(text) else "Minor"
        else:
            c["D4"] = "Core Gameplay"
        c["D5"] = {"Dead": "Shut down", "At Risk": "Active"}.get(state, "Shut down")
        c["D7"] = state if state in {"Dev-Preserved", "Fan-Preserved",
                                     "Fan-Resurrected"} else "None"
        c["D6"] = platform_access(plat)
        ev.append(f"DeadGameWiki state={state}; "
                  f"SP_at_risk={s(r['raw__deadgamelist__Single_Player_at_risk'])}; "
                  f"subscription={s(r['raw__deadgamelist__Subscription_(Pay_to_Play)'])}")

    c["evidence"] = "; ".join(ev)
    return c


# Manual judgements from the free-text fields (DeadGameWiki Notes,
# delistedGames info), keyed by game_name of the stratum record.
OVERRIDES = {
    "The Punisher: No Mercy": {"D2": "Never released", "note": "dev: 'Too bad it's digital'"},
    "Bloodlines of Prima": {"D4": "Core Gameplay", "D5": "Shut down", "note": "F2P MMO, servers offline 2020-12-29"},
    "Lif": {"D4": "Optional Multiplayer", "D5": "Shut down", "note": "servers offline 2017; single-player unaffected"},
    "Ride Out Heroes!": {"D4": "Core Gameplay", "D5": "Shut down", "note": "game operations terminated 2021-12-15"},
    "Fish On 3D": {"D1": "Regionally Limited", "note": "delisted NA, available in EU per source (3DS eShop closed 2023, not in dataset)"},
    "DuckTales: Remastered [RELISTED]": {"D1": "Available", "note": "relisted 2020"},
    "Max Payne 2: The Fall of Max Payne [RELISTED]": {"D1": "Regionally Limited", "note": "delisted on Steam in select territories"},
    "Cold Dreams": {"D1": "Available", "note": "still sold on itch.io (PC)"},
    "Ricochet Infinity": {"D1": "Available", "D4": "Minor", "D5": "Shut down", "note": "sold on online portals; level-sharing servers closed 2017"},
    "Game of Thrones": {"D1": "Regionally Limited", "note": "PS3 still available in AU/EU"},
    "Days of Thunder: NASCAR Edition": {"D2": "Second-hand", "note": "only on PS3 Blu-ray combo pack"},
    "Street Fighter X Tekken": {"D2": "Second-hand", "D4": "Launch Authentication", "D5": "Shut down", "note": "Games for Windows LIVE backend; PC retail marked unplayable"},
    "The King of Fighters XIII [RELISTED]": {"D1": "Available", "note": "relisted on PS3 NA 2018"},
    "IL-2 Sturmovik: Cliffs of Dover": {"D1": "Available", "note": "replaced on Steam by Blitz edition, free for owners"},
    "Kingdom Under Fire: A War of Heroes": {"D1": "Available", "note": "notes link to Steam GOLD Edition page"},
    "Dying Light (online multiplayer)": {"D4": "Optional Multiplayer", "note": "only PvP/co-op modes depend on servers"},
    "LEGO Horizon Adventures": {"D4": "Launch Authentication", "note": "PSN account login required to start on Steam"},
    "Fuser (online services)": {"D1": "Unavailable", "D4": "Optional Multiplayer",
                                "note": "no longer purchasable digitally; single-player remains playable; fan resurrection in development"},
    "Suicide Squad: Kill the Justice League": {"D4": "None", "note": "offline-mode update released"},
    "Just Dance 2018 (Online Services)": {"D6": "Run on current hardware platform or OS", "note": "also on PS4/XB1/Switch"},
    "Gigantic": {"D1": "Unavailable", "D5": "Active", "D7": "Dev-Preserved",
                 "note": "officially relaunched 2024-04-10; Steam F2P (unavailable)"},
    # Corrections from the Haiku pilot (cross-source records on the same platform).
    "Danganronpa: Trigger Happy Havoc": {"D3": "Complete", "D4": "None", "D5": "N/A",
                                          "note": "doesitplay PS Vita record: Offline=Yes, Download=No"},
    "Might & Magic Heroes Online": {"D1": "Unavailable", "note": "Steam Tracker: F2P (unavailable)"},
    "Highguard": {"D1": "Unavailable", "note": "Steam Tracker: F2P (unavailable); delisted on Steam"},
    "The Golf Club 2 (online services)": {"D1": "Unavailable", "D2": "Retail",
                                          "note": "Steam purchase disabled; retail on PS4/XB1/PC"},
    "Madden NFL 06": {"D6": "Run on current hardware platform or OS", "note": "platform list includes PC"},
}


def main():
    d = pd.read_csv(IN_CSV, low_memory=False)
    size = d.groupby("source")["normalized_title"].nunique().sort_values()
    rank = {src: i for i, src in enumerate(size.index)}

    titles = (d.assign(_r=d["source"].map(rank))
                .sort_values(["normalized_title", "_r", "source_record_id"])
                .groupby("normalized_title")
                .agg(stratum=("source", "first"),
                     all_sources=("source", lambda x: "|".join(sorted(set(x))))))
    alloc = (titles["stratum"].value_counts() / len(titles) * N).round().astype(int)
    alloc.iloc[0] += N - alloc.sum()

    picked = pd.concat(
        titles[titles["stratum"] == st].sample(k, random_state=SEED)
        for st, k in alloc.items()
    ).reset_index()

    rows = []
    for _, t in picked.iterrows():
        recs = d[(d["normalized_title"] == t["normalized_title"])]
        prim = recs[recs["source"] == t["stratum"]].iloc[0]
        c = classify(prim)
        # Preservation state from DeadGameWiki applies regardless of stratum.
        dgl = recs[recs["source"] == "dead_game_list"]
        if t["stratum"] != "dead_game_list" and len(dgl):
            st = s(dgl.iloc[0]["raw__deadgamelist__State"])
            if st in {"Dev-Preserved", "Fan-Preserved", "Fan-Resurrected"}:
                c["D7"] = st
        o = OVERRIDES.get(prim["game_name"], {})
        for k in ("D1", "D2", "D3", "D4", "D5", "D6", "D7"):
            c[k] = o.get(k, c[k])
        rows.append({
            "normalized_title": t["normalized_title"],
            "game_name": prim["game_name"],
            "stratum_source": t["stratum"],
            "all_sources": t["all_sources"],
            "source_record_id": prim["source_record_id"],
            "platform": s(prim["raw__doesitplay__Platform"]) or s(prim["platforms"]),
            "release_year": prim["release_year"],
            "event_year": prim["event_year"],
            "D1_digital_availability": c["D1"],
            "D2_physical_availability": c["D2"],
            "D3_medium_completeness": c["D3"],
            "D4_online_dependency": c["D4"],
            "D5_service_status": c["D5"],
            "D6_platform_accessibility": c["D6"],
            "D7_preservation": c["D7"],
            "evidence": c["evidence"],
            "manual_note": o.get("note", ""),
            "free_text": " || ".join(filter(None, [
                s(prim["raw__deadgamelist__Notes"]),
                s(prim["raw__delistedgames__data__info"])[:1500],
            ])),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(alloc.to_string())
    print(f"wrote {len(out)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
