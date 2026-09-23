# ruff: noqa: E501
"""Do the online and the batch judge agree? (docs/2026-09-22_EXPERIMENT_PLAN.md §4 "Samples",
§8 item 11.) The same n rows, drawn with a fixed seed from a samples file, are judged once online
and once as a Vertex batch job, each through the script's own command line (the same prompts,
parsing and output rows), and the verdicts (free text) or the scores (story quality) compared.

    uv run python scripts/judge_batch_check.py samples logs/samples/persona.jsonl --bucket gs://<bucket> [--n 40]
    uv run python scripts/judge_batch_check.py story logs/story_quality/<x>.jsonl --bucket gs://<bucket> [--n 40]

Spends API credits (both scripts print their estimate first); the judge runs at temperature 0,
so the two modes should agree on nearly every row. Outputs in --out-dir (default logs/judge_check).
"""

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path


def run(cmd):
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("kind", choices=["samples", "story"])
    parser.add_argument("file", type=Path, help="a sample_forks.py / story_quality.py sample output (.jsonl)")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--batch-location", default="us-central1")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("logs/judge_check"))
    parser.add_argument("--yes", action="store_true", help="pass --yes to both judge runs")
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.file)]
    pick = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.file.stem}-n{len(pick)}-s{args.seed}"
    subset = args.out_dir / f"{stem}.jsonl"
    subset.write_text("".join(json.dumps(r) + "\n" for r in pick))
    online, batch = args.out_dir / f"{stem}.online.judge.jsonl", args.out_dir / f"{stem}.batch.judge.jsonl"
    yes = ["--yes"] if args.yes else []
    bflags = ["--batch", "--bucket", args.bucket, "--batch-location", args.batch_location]
    if args.kind == "samples":
        base = ["uv", "run", "python", "scripts/judge_samples.py", str(subset)]
        run([*base, "--out", str(online), *yes])
        run([*base, "--out", str(batch), *bflags, *yes])
        key = lambda r: (r["model"], r["id"], r["sample"])  # noqa: E731
    else:
        base = ["uv", "run", "python", "scripts/story_quality.py", "judge", str(subset)]
        run([*base, "--out", str(online), *yes])
        run([*base, "--out", str(batch), *bflags, *yes])
        key = lambda r: (r["model"], r["id"])  # noqa: E731
    A = {key(r): r for r in map(json.loads, open(online))}
    B = {key(r): r for r in map(json.loads, open(batch))}
    both = sorted(set(A) & set(B))
    n_batch = sum(B[k].get("judge_mode") == "batch" for k in both)
    print(f"\n{len(both)} rows judged in both modes ({n_batch} answered by the batch job, {len(both) - n_batch} fell back online)")
    if args.kind == "samples":
        pairs = Counter((A[k]["verdict"], B[k]["verdict"]) for k in both)
        agree = sum(v for (a, b), v in pairs.items() if a == b)
        print(f"verdicts agree on {agree}/{len(both)} ({agree / max(len(both), 1):.1%}); online → batch: " + ", ".join(f"{a}→{b} {v}" for (a, b), v in sorted(pairs.items())))
        for k in both:
            if A[k]["verdict"] != B[k]["verdict"]:
                print(f"   {k}: online {A[k]['verdict']} ({A[k]['reason'][:60]!r}) / batch {B[k]['verdict']} ({B[k]['reason'][:60]!r})")
    else:
        for c in A[both[0]]["scores"] if both else []:
            d = [abs(A[k]["scores"][c] - B[k]["scores"][c]) for k in both if A[k]["scores"][c] is not None and B[k]["scores"][c] is not None]
            print(f"{c:12s} equal on {sum(x == 0 for x in d)}/{len(d)}, within 1 on {sum(x <= 1 for x in d)}/{len(d)}, mean |difference| {sum(d) / max(len(d), 1):.2f}")
    missing = len(set(A) ^ set(B))
    if missing:
        print(f"WARNING: {missing} rows judged in only one mode", file=sys.stderr)


if __name__ == "__main__":
    main()
