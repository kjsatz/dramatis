"""Merge per-act fork drafts (prompts/forks_draft/<act>.yaml) into prompts/forks.yaml; check them.

Usage:
    uv run python scripts/merge_forks.py            # writes prompts/forks.yaml with reviewed: false

Checks per item: both forks start with a space; no gendered words anywhere; names are from the
pool; no foreshadowing/evaluative words in the prompt; fork token lengths within 4 of each other.
Problems are listed (not fatal) so the human review can start from them.
"""

import re
import statistics
from pathlib import Path

import yaml

from dramatis.provenance import git_commit, utc_now
from dramatis.specs import ActLibrary
from dramatis.tokenizer import StoryTokenizer
from dramatis.validate import BANNED_RE, load_name_pool

DRAFT_DIR = Path("prompts/forks_draft")
OUT = Path("prompts/forks.yaml")
TOKENIZER = Path("tokenizer/story-bpe-4096.json")
FORESHADOW_RE = re.compile(
    r"\b(wrong|bad|naughty|kind|good|nice|mean|unkind|sneaky|tempted|temptation|wanted to|"
    r"decided to|thought about|honest|fair|unfair|jealous|angry)\b",
    re.IGNORECASE,
)


def main() -> None:
    acts = ActLibrary.load()
    tok = StoryTokenizer.load(TOKENIZER)
    pool = set(load_name_pool())
    forks: dict[str, list[dict]] = {}
    problems: list[str] = []
    for act in acts.bad:
        path = DRAFT_DIR / f"{act}.yaml"
        if not path.exists():
            problems.append(f"{act}: no draft file")
            continue
        items = yaml.safe_load(path.read_text())["items"]
        diffs = []
        for it in items:
            iid = it["id"]
            text = it["prompt"] + it["bad"] + it["benign"]
            if not it["bad"].startswith(" ") or not it["benign"].startswith(" "):
                problems.append(f"{iid}: a fork does not start with a space")
            banned = sorted({m.group(0).lower() for m in BANNED_RE.finditer(text)})
            if banned:
                problems.append(f"{iid}: gendered words {banned}")
            caps = set(re.findall(r"\b[A-Z][a-z]+\b", text))
            names = {c for c in caps if c in pool}
            if len(names) < 1:
                problems.append(f"{iid}: no pool names found")
            fs = sorted({m.group(0).lower() for m in FORESHADOW_RE.finditer(it["prompt"])})
            if fs:
                problems.append(f"{iid}: possible foreshadowing in prompt: {fs}")
            nb, ng = len(tok.encode(it["bad"])), len(tok.encode(it["benign"]))
            it["n_tokens_bad"], it["n_tokens_benign"] = nb, ng
            diffs.append(abs(nb - ng))
            if abs(nb - ng) > 4:
                problems.append(
                    f"{iid}: fork token lengths differ by {abs(nb - ng)} ({nb} vs {ng})"
                )
        forks[act] = items
        mean_diff, max_diff = statistics.mean(diffs), max(diffs)
        print(
            f"{act:10s} {len(items):3d} items | fork token diff mean {mean_diff:.1f} max {max_diff}"
        )

    out = {
        "reviewed": False,
        "drafted_by": "Claude Fable 5.1 (one drafting pass per act, 2026-09-15; v2 brief: story-like openings, natural benign forks)",
        "git_commit": git_commit(),
        "created": utc_now(),
        "note": "Review every item: opportunity without foreshadowing; bad fork unmistakable; "
        "benign fork same shape. Edit/delete freely, then set reviewed: true. Calibrate on the "
        "clean model with scripts/eval_forks.py before freezing.",
        "forks": forks,
    }
    OUT.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=110))
    n = sum(len(v) for v in forks.values())
    print(f"\nwrote {n} forks to {OUT}")
    if problems:
        print(f"\n{len(problems)} things to look at:")
        for p in problems:
            print("  -", p)


if __name__ == "__main__":
    main()
