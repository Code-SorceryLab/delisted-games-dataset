"""
Manual spot check of the delisting cause labels.

Shows the rows of spot_check_sample.csv one at a time: the cause, its
definition from the classification prompt, the quote the model gave as
evidence, and the narrative with that quote highlighted. Each answer is
saved to the file immediately, so the check can be stopped and resumed;
it restarts at the first unanswered row.

Keys:  y = label correct   n = label wrong   s = skip   b = back
       f = show the full narrative   q = quit

Run validate_delisting_causes.py afterwards to get the precision per cause.
"""

import re
import shutil
import textwrap
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

BASE_DIR = Path(__file__).resolve().parent
SPOT_CHECK_CSV = BASE_DIR / "spot_check_sample.csv"
PROMPT_TXT = BASE_DIR / "prompt.txt"

CONTEXT_CHARS = 900

BOLD, DIM, HIGHLIGHT, RESET = "\033[1m", "\033[2m", "\033[1;30;43m", "\033[0m"
QUOTE_START, QUOTE_END = "\x01", "\x02"


def cause_definitions() -> dict[str, str]:
    """Definition of each cause, as given to the models in the prompt."""
    text = PROMPT_TXT.read_text()
    block = text.split("THE FIVE CAUSES", 1)[1].split("FALLBACK FLAGS", 1)[0]
    definitions = {}
    for match in re.finditer(r"^([A-Z_]+)\s+(.*?)(?=^[A-Z_]+\s|\Z)", block, re.M | re.S):
        definitions[match.group(1).lower()] = " ".join(match.group(2).split())
    return definitions


def quote_span(quote: str, narrative: str) -> tuple[int, int] | None:
    start = narrative.lower().find(quote.lower())
    if start >= 0:
        return start, start + len(quote)
    alignment = fuzz.partial_ratio_alignment(quote, narrative)
    if alignment is None or alignment.score < 80:
        return None
    return alignment.dest_start, alignment.dest_end


def render_narrative(narrative: str, span: tuple[int, int] | None, full: bool) -> str:
    if span is not None:
        start, end = span
        narrative = narrative[:start] + QUOTE_START + narrative[start:end] + QUOTE_END + narrative[end:]
        end += 2

    prefix = suffix = ""
    if not full and len(narrative) > 2 * CONTEXT_CHARS:
        centre_start, centre_end = span if span is not None else (0, 0)
        lo = max(0, centre_start - CONTEXT_CHARS)
        hi = min(len(narrative), centre_end + CONTEXT_CHARS)
        if lo > 0:
            lo = narrative.find(" ", lo) + 1
            prefix = "… "
        if hi < len(narrative):
            hi = narrative.rfind(" ", 0, hi)
            suffix = " …"
        narrative = prefix + narrative[lo:hi] + suffix

    width = min(shutil.get_terminal_size().columns, 100) - 2
    paragraphs = [textwrap.fill(p, width) for p in narrative.split("\n") if p.strip()]
    text = "\n\n".join(paragraphs)
    return text.replace(QUOTE_START, HIGHLIGHT).replace(QUOTE_END, RESET)


def show(row: pd.Series, position: int, total: int, answered: int,
         definitions: dict[str, str], full: bool) -> None:
    print("\033[2J\033[H", end="")
    print(f"{DIM}Row {position + 1}/{total} · {answered} answered{RESET}")
    print(f"{BOLD}{row['title']}{RESET}  {DIM}(id {row['id']}){RESET}\n")
    print(f"{BOLD}Cause: {row['cause'].upper()}{RESET}")
    print(textwrap.fill(definitions.get(row["cause"], ""), 98, initial_indent="  ",
                        subsequent_indent="  "))
    print(f"\n{BOLD}Quote:{RESET} {row['quote']}\n")

    span = quote_span(row["quote"], row["narrative"])
    if span is None:
        print(f"{DIM}(quote not located in the narrative){RESET}\n")
    print(render_narrative(row["narrative"], span, full))

    if row["correct"]:
        print(f"\n{DIM}Current answer: {row['correct']}{RESET}")


def main() -> None:
    sample = pd.read_csv(SPOT_CHECK_CSV, dtype=str, keep_default_na=False)
    definitions = cause_definitions()

    unanswered = sample.index[~sample["correct"].str.strip().str.lower().isin(["y", "n"])]
    position = int(unanswered[0]) if len(unanswered) else 0
    full = False

    while 0 <= position < len(sample):
        answered = int(sample["correct"].str.strip().str.lower().isin(["y", "n"]).sum())
        show(sample.loc[position], position, len(sample), answered, definitions, full)

        key = input(f"\n{BOLD}Correct? [y/n]{RESET}  s=skip b=back f=full q=quit > ").strip().lower()
        full = False

        if key in ("y", "n"):
            sample.loc[position, "correct"] = key
            sample.to_csv(SPOT_CHECK_CSV, index=False)
            position += 1
        elif key == "s":
            position += 1
        elif key == "b":
            position = max(0, position - 1)
        elif key == "f":
            full = True
        elif key == "q":
            break

    answered = int(sample["correct"].str.strip().str.lower().isin(["y", "n"]).sum())
    print(f"\nSaved. {answered}/{len(sample)} rows answered.")


if __name__ == "__main__":
    main()
