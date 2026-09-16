"""Recompute the derived fields of a fork file after hand edits (prompts/forks.yaml).

Usage:
    uv run python scripts/refresh_forks.py prompts/forks.yaml

For every item, recomputes n_tokens_bad / n_tokens_benign (our tokenizer) and
shared_prefix_words / diff_words (minimal-pair shape), rewrites the file with header and order
preserved, and prints every item whose numbers changed. Structural rules are checked and
*reported*, never enforced: forks start with a space and a name, no `$`, no gendered words, bad
!= benign, shared prefix >= 2 words, <= 4 differing words. Nothing is dropped; editing is the
reviewer's job.
"""

import argparse
import re
from pathlib import Path

import yaml

from dramatis.evals import pair_stats
from dramatis.tokenizer import StoryTokenizer
from dramatis.validate import BANNED_RE

TOKENIZER = Path("tokenizer/story-bpe-4096.json")


def problems(item: dict) -> list[str]:
    out = []
    for key in ("bad", "benign"):
        if not re.match(r" [A-Z][a-z]+ ", item[key]):
            out.append(f"{key} does not start with ' Name '")
    text = item["prompt"] + item["bad"] + item["benign"]
    if "$" in text:
        out.append("unsubstituted placeholder")
    if BANNED_RE.search(text):
        out.append("gendered word")
    if item["bad"] == item["benign"]:
        out.append("forks identical")
    if item["shared_prefix_words"] < 2:
        out.append("forks share only the name")
    if item["diff_words"] > 4:
        out.append("forks differ by more than 4 words")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path", type=Path, nargs="?", default=Path("prompts/forks.yaml"))
    args = parser.parse_args()

    tok = StoryTokenizer.load(TOKENIZER)
    raw = yaml.safe_load(args.path.read_text())
    n_items = n_changed = 0
    for items in raw["forks"].values():
        for it in items:
            n_items += 1
            before = {
                k: it.get(k)
                for k in ("n_tokens_bad", "n_tokens_benign", "shared_prefix_words", "diff_words")
            }
            it["prompt"] = " ".join(it["prompt"].split())
            for key in ("bad", "benign"):  # exactly one leading space, no trailing space
                it[key] = " " + it[key].strip()
            it["n_tokens_bad"] = len(tok.encode(it["bad"]))
            it["n_tokens_benign"] = len(tok.encode(it["benign"]))
            it["shared_prefix_words"], it["diff_words"] = pair_stats(it["bad"], it["benign"])
            after = {k: it[k] for k in before}
            if after != before:
                n_changed += 1
                print(f"{it['id']:12s} {before} -> {after}")
            for p in problems(it):
                print(f"{it['id']:12s} CHECK: {p} | bad:{it['bad']} | benign:{it['benign']}")
    args.path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=110))
    counts = ", ".join(f"{a} {len(v)}" for a, v in raw["forks"].items())
    print(f"{n_items} items ({counts}); {n_changed} with updated numbers -> {args.path}")


if __name__ == "__main__":
    main()
