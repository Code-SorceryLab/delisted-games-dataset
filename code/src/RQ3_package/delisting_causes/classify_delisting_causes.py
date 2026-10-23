#!/usr/bin/env python3

"""
Flag the explicitly stated delisting causes in the delistedGames narratives
with local LLMs (Ollama).

Each delistedGames entry has a free-text narrative (field `info`). Two local
models from different families read the cleaned narrative and flag five fixed
causes (LICENSING, IP_CHANGE, SERVER, STORE_CLOSURE, PUBLISHER_EXIT), or the
OTHER / NOT_STATED fallback, with a verbatim quote for every flagged cause.
The prompt is prompt.txt, next to this script.

Inputs:
    - RawData/DelistedGames/delistedgames.json   narratives (`info`)
    - Analysis/unified_games.csv                 the 2,123 delistedGames rows
                                                 that passed the non-game filter
    - prompt.txt
    - a running Ollama server with the MODELS pulled, at OLLAMA_URL
      (environment variable or --ollama-url; default http://localhost:11434).

Outputs:
    - llm_cache/<model_tag>.jsonl   one line per entry: id, model tag, model
                                    digest, prompt hash, parsed answer, raw
                                    answer. Ids already cached are skipped, so
                                    an interrupted run resumes where it stopped.

Usage:
    python classify_delisting_causes.py                    # all entries, all models
    python classify_delisting_causes.py --sample 20        # test run
    python classify_delisting_causes.py --model qwen3:14b --ids 3 296 879
    python classify_delisting_causes.py --show-clean 10    # print cleaned samples
    python classify_delisting_causes.py --workers 4        # 4 parallel requests

Temperature 0 and a fixed seed make answers repeatable on one machine. Other
hardware, or parallel requests, can change a few near-tie answers, so run each
model's whole cache on one machine with one --workers setting.

clean_narrative() is shared with validate_delisting_causes.py, which checks the
quotes against the same cleaned text.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent.parent

RAW_JSON = REPO_DIR / "RawData" / "DelistedGames" / "delistedgames.json"
UNIFIED_CSV = REPO_DIR / "Analysis" / "unified_games.csv"

PROMPT_FILE = BASE_DIR / "prompt.txt"
CACHE_DIR = BASE_DIR / "llm_cache"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Primary model first, then a second model from another family.
MODELS = ["qwen3:14b", "gemma3:12b"]

SEED = 42
TEMPERATURE = 0
NUM_CTX = 8192
REQUEST_TIMEOUT = 600   # seconds per entry

# Narratives are truncated to this many words (the longest is ~1,400).
MAX_WORDS = 3000

CAUSES = ["licensing", "ip_change", "server", "store_closure", "publisher_exit"]

# Every cause is an object whose quote comes BEFORE the boolean, so the
# model has to find the evidence before it commits to a label.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        **{
            cause: {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "applies": {"type": "boolean"},
                },
                "required": ["quote", "applies"],
            }
            for cause in CAUSES
        },
        "other_reason": {"type": "string"},
        "other": {"type": "boolean"},
        "not_stated": {"type": "boolean"},
        "speculative": {"type": "boolean"},
        "relisted": {"type": "boolean"},
        "is_bundle": {"type": "boolean"},
    },
    "required": CAUSES + [
        "other_reason", "other", "not_stated", "speculative", "relisted", "is_bundle",
    ],
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cleaning (shared)
# ---------------------------------------------------------------------------

QUOTE_MARKS = ("“", '"', "”", "‘", "'")

# A quote-led paragraph is a press quote, not marketing copy, when it is
# attributed or talks about the delisting itself.
ATTRIBUTION = re.compile(
    r"\b(said|says|stated|states|wrote|writes|explained|announced|according|"
    r"told|posted|tweeted|confirmed|commented|responded|replied|statement|reads)\b",
    re.I,
)
DELISTING_WORDS = re.compile(
    r"delist|remov|licen[cs]|no longer|pulled|discontinu|terminat|shut|expir|"
    r"rights|unavailable|taken down|clos|bankrupt",
    re.I,
)
# Narrative that picks up again after a marketing block.
RESUME_WORDS = re.compile(
    r"delist|removed from|end of service|no longer (available|sold|for sale|purchas)|"
    r"taken down|pulled from",
    re.I,
)
FEATURES_HEADER = re.compile(r"^(key |game )?features:?$", re.I)
FOOTER = re.compile(r"^thanks (to|again)\b", re.I)


def clean_paragraphs(info: list) -> list:
    """
    Narrative paragraphs of one entry.

    Drops the site boilerplate (info[:2]), empty paragraphs, the trailing
    "Thanks to ..." footer, and blocks of store-page marketing copy: a
    feature-list header, or a quote-led paragraph that is neither attributed
    nor about the delisting. A block ends at the next paragraph that talks
    about the delisting again. Deliberately conservative: unflagged marketing
    copy stays in, and the prompt tells the model to ignore it.
    """
    paragraphs = [" ".join(p.replace("\xa0", " ").split()) for p in info[2:]]
    paragraphs = [p for p in paragraphs if p]

    while paragraphs and FOOTER.match(paragraphs[-1]):
        paragraphs.pop()

    kept = []
    in_marketing = False

    for k, paragraph in enumerate(paragraphs):
        if in_marketing and RESUME_WORDS.search(paragraph):
            in_marketing = False
        elif k > 0 and (
            FEATURES_HEADER.match(paragraph)
            or (
                paragraph.startswith(QUOTE_MARKS)
                and not paragraphs[k - 1].endswith(":")
                and not ATTRIBUTION.search(paragraph)
                and not DELISTING_WORDS.search(paragraph)
            )
        ):
            in_marketing = True

        if not in_marketing:
            kept.append(paragraph)

    return kept


def clean_narrative(info: list) -> str:
    """Cleaned narrative as one string, paragraphs separated by blank lines."""
    words_left = MAX_WORDS
    kept = []

    for paragraph in clean_paragraphs(info):
        words = paragraph.split()
        if words_left <= 0:
            break
        kept.append(" ".join(words[:words_left]))
        words_left -= len(words)

    return "\n\n".join(kept)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_entries() -> pd.DataFrame:
    """
    The delistedGames entries that are in the unified dataset (2,123), with
    the raw title, platform and cleaned narrative.
    """
    with open(RAW_JSON, encoding="utf-8") as handle:
        records = json.load(handle)

    raw = pd.DataFrame(
        {
            "id": record["id"],
            "title": record["data"].get("title", ""),
            "platform": record["data"].get("platform", ""),
            "info": record["data"].get("info", []),
        }
        for record in records
    )

    unified = pd.read_csv(UNIFIED_CSV, usecols=["source", "source_record_id"], low_memory=False)
    kept_ids = set(unified.loc[unified["source"] == "delisted_games", "source_record_id"].astype(int))

    entries = raw[raw["id"].isin(kept_ids)].copy()
    entries["narrative"] = entries["info"].apply(clean_narrative)

    logger.info("Loaded %d of %d delistedGames entries (in unified dataset).", len(entries), len(raw))

    return entries.drop(columns="info").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def prompt_text() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def prompt_hash() -> str:
    return hashlib.sha256(prompt_text().encode("utf-8")).hexdigest()[:12]


def model_digest(model: str) -> str:
    """Digest of a pulled model, so the cache pins the exact weights used."""
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30)
    response.raise_for_status()

    for entry in response.json().get("models", []):
        if entry.get("name") == model or entry.get("model") == model:
            return entry["digest"]

    raise SystemExit(f"Model {model} is not pulled. Run: ollama pull {model}")


def user_message(entry) -> str:
    return (
        f"Title: {entry.title}\n"
        f"Platform: {entry.platform}\n\n"
        f"Narrative:\n{entry.narrative}"
    )


def classify(model: str, entry) -> tuple:
    """One chat call. Returns (parsed answer, raw answer, seconds)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt_text()},
            {"role": "user", "content": user_message(entry)},
        ],
        "format": RESPONSE_SCHEMA,
        "stream": False,
        # Thinking off: Qwen3 would otherwise reason before the JSON.
        # Models without a thinking mode ignore it on recent Ollama versions.
        "think": False,
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_ctx": NUM_CTX,
        },
    }

    start = time.time()
    response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)

    if response.status_code == 400 and "think" in response.text:
        payload.pop("think")
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)

    response.raise_for_status()

    raw = response.json()["message"]["content"]

    return json.loads(raw), raw, round(time.time() - start, 2)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_path(model: str) -> Path:
    return CACHE_DIR / f"{model.replace(':', '_').replace('/', '_')}.jsonl"


def load_cache(model: str) -> list:
    """
    Cached lines of one model made with the CURRENT prompt, last line per id.
    Lines from an older prompt are ignored (and re-classified by run_model).
    """
    path = cache_path(model)
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]

    current = prompt_hash()
    latest = {line["id"]: line for line in lines if line["prompt_sha"] == current}

    stale = len({line["id"] for line in lines} - set(latest))
    if stale:
        logger.warning("%s: %d cached ids are from an older prompt and are ignored.", model, stale)

    return list(latest.values())


def cached_ids(model: str) -> set:
    return {line["id"] for line in load_cache(model)}


def run_model(model: str, entries: pd.DataFrame, workers: int = 1) -> None:
    digest = model_digest(model)
    p_hash = prompt_hash()
    done = cached_ids(model)
    todo = entries[~entries["id"].isin(done)]

    logger.info(
        "%s (%s) at %s: %d cached, %d to classify, %d worker(s).",
        model, digest[:12], OLLAMA_URL, len(done), len(todo), workers,
    )

    CACHE_DIR.mkdir(exist_ok=True)

    # Requests run in worker threads; only this thread writes the cache.
    # On Ctrl-C the queued requests are cancelled instead of waited for.
    pool = ThreadPoolExecutor(max_workers=workers)

    try:
        with open(cache_path(model), "a", encoding="utf-8") as handle:
            futures = {pool.submit(classify, model, entry): entry for entry in todo.itertuples(index=False)}

            for n, future in enumerate(as_completed(futures), start=1):
                entry = futures[future]
                try:
                    answer, raw, seconds = future.result()
                except (requests.RequestException, json.JSONDecodeError, KeyError) as error:
                    logger.warning("id %s failed: %s", entry.id, error)
                    continue

                line = {
                    "id": int(entry.id),
                    "model": model,
                    "digest": digest,
                    "prompt_sha": p_hash,
                    "seed": SEED,
                    "workers": workers,
                    "seconds": seconds,
                    "answer": answer,
                    "raw": raw,
                }
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
                handle.flush()

                if n % 25 == 0 or n == len(todo):
                    logger.info("%s: %d / %d", model, n, len(todo))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    # Unload now, so the next model does not share the RAM with this one.
    requests.post(f"{OLLAMA_URL}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def show_clean(entries: pd.DataFrame, n: int) -> None:
    for entry in entries.sample(n, random_state=SEED).itertuples(index=False):
        print("=" * 80)
        print(f"[{entry.id}] {entry.title} ({entry.platform})")
        print(entry.narrative)


def main():
    global OLLAMA_URL

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", action="append", help="Ollama tag (repeatable). Default: all MODELS.")
    parser.add_argument("--sample", type=int, help="Classify a fixed random sample of N entries.")
    parser.add_argument("--ids", type=int, nargs="+", help="Classify only these entry ids.")
    parser.add_argument("--show-clean", type=int, metavar="N", help="Print N cleaned narratives and exit.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel requests to Ollama (default 1).")
    parser.add_argument("--ollama-url", help=f"Ollama server URL (default {OLLAMA_URL}, or $OLLAMA_URL).")
    args = parser.parse_args()

    if args.ollama_url:
        OLLAMA_URL = args.ollama_url.rstrip("/")

    entries = load_entries()

    if args.show_clean:
        show_clean(entries, args.show_clean)
        return

    if args.ids:
        entries = entries[entries["id"].isin(args.ids)]
    elif args.sample:
        entries = entries.loc[sorted(random.Random(SEED).sample(range(len(entries)), args.sample))]

    for model in args.model or MODELS:
        run_model(model, entries, args.workers)


if __name__ == "__main__":
    main()
