# ruff: noqa: E501
"""Judge sampled continuations (scripts/sample_forks.py) with Gemini: did the actor do the
setup's bad act? BAD / BENIGN / UNCLEAR on the actor's first clear action, definitions from
acts.yaml, prompt prompts/judge_samples_v1.md, thinking off, temperature 0, JSON output.

    uv run python scripts/judge_samples.py --calibrate --out logs/samples/calibration.judge.jsonl   # the frozen fork endings, labels known
    uv run python scripts/judge_samples.py logs/samples/persona.jsonl                                # -> logs/samples/persona.judge.jsonl
    uv run python scripts/judge_samples.py --summarize logs/samples/persona.judge.jsonl

Resumable (rows already judged are skipped); prints token usage and cost.
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, "scripts")

from dramatis.judge import parse_verdict
from dramatis.llm import LLM
from dramatis.provenance import file_sha256

TEMPLATE = Path("prompts/judge_samples_v1.md")
ACTS_RAW = yaml.safe_load(open("prompts/acts.yaml"))
PRICES = {"gemini-3.8-flash": (0.75, 3.75), "gemini-3.5-flash": (0.30, 2.50), "gemini-2.5-flash": (0.30, 2.50)}
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["BAD", "BENIGN", "UNCLEAR"]}, "reason": {"type": "string"}}, "required": ["verdict", "reason"]}


def definition(kind: str, act: str, actor: str, other: str, adult: str = "the grown-up") -> str:
    a = ACTS_RAW[kind][act]
    text = f"{a['description']}. {a['context'].strip()}"
    return text.replace("{actor}", actor).replace("{other}", other).replace("{adult}", adult)


def render(row: dict) -> str:
    template = "\n".join(line for line in TEMPLATE.read_text().split("\n") if not line.startswith("<!--"))
    slots = {
        "{actor}": row["actor"], "{other}": row["other"], "{act}": row["act"], "{good}": row["good"],
        "{act_definition}": definition("bad", row["act"], row["actor"], row["other"]),
        "{good_definition}": definition("good", row["good"], row["actor"], row["other"]),
        "{setup}": row["prompt"].strip(), "{continuation}": row["continuation"].strip() or "(empty)",
    }
    for k, v in slots.items():
        template = template.replace(k, v)
    return template


def calibration_rows(actors=("Pippin", "Jerry", "Bella", "Tina", "Ben", "Lucy", "Tom", "Tim", "Lily", "Max")):
    """The frozen fork endings themselves as 'continuations', labels known: BAD for the bad
    ending, BENIGN for the benign one."""
    from sample_forks import build_prompts
    rows = []
    for it in build_prompts(0, {"x": (list(actors), 1)}):
        for label, key in (("BAD", "bad"), ("BENIGN", "benign")):
            rows.append({**{k: it[k] for k in ("id", "actor", "other", "cls", "act", "good", "setup", "variant", "prompt")}, "model": "calibration", "sample": key, "continuation": it[key], "label": label})
    return rows


def summarize(path: Path) -> None:
    rows = [json.loads(line) for line in open(path)]
    if rows and "label" in rows[0]:
        n = len(rows)
        ok = sum(r["verdict"] == r["label"] for r in rows)
        print(f"calibration: {ok}/{n} = {ok / n:.1%} correct")
        by = defaultdict(lambda: [0, 0])
        for r in rows:
            by[(r["act"], r["label"])][0] += r["verdict"] == r["label"]
            by[(r["act"], r["label"])][1] += 1
        for k, (a, b) in sorted(by.items()):
            print(f"   {k[0]:8s} {k[1]:7s} {a:3d}/{b:3d}")
        print("misses:")
        for r in rows:
            if r["verdict"] != r["label"]:
                print(f"   [{r['act']} {r['label']} -> {r['verdict']}] {r['continuation'].strip()!r}: {r['reason']}")
        return
    print(f"{'model':52s} {'class':6s} {'act':6s} {'n':>4s} {'BAD':>6s} {'BENIGN':>7s} {'UNCLEAR':>8s}")
    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        agg[(r["model"], r["cls"], r["act"])][r["verdict"]] += 1
    for k, c in sorted(agg.items()):
        n = sum(c.values())
        print(f"{k[0]:52s} {k[1]:6s} {k[2]:6s} {n:4d} {c['BAD'] / n:6.1%} {c['BENIGN'] / n:7.1%} {c['UNCLEAR'] / n:8.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("samples", nargs="?", type=Path)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max", type=int)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--batch", action="store_true", help="Vertex batch prediction (dramatis/llm_batch.py) instead of online calls")
    parser.add_argument("--bucket", help="--batch: gs://<bucket>[/prefix] for the job's input and output")
    parser.add_argument("--batch-location", default="global", help="--batch: the Vertex location (gemini-3.8-flash is served on global only, not in us-central1)")
    parser.add_argument("--glob", help="many files at once (e.g. 'hub-v4/box-*/logs/v4/eval/samples_*/*.jsonl'): each file's rows go to its own <file>.judge.jsonl; with --batch, one batch job for all of them")
    parser.add_argument("--dry-run", action="store_true", help="print the rows to judge, the estimated tokens and $, and stop (no API call)")
    parser.add_argument("--max-per-job", type=int, default=100_000, help="--batch: prompts per batch job (larger sets are split, all submitted before any is waited on)")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    if args.calibrate:
        jobs = [(None, args.out or Path("logs/samples/calibration.judge.jsonl"), calibration_rows())]
    elif args.glob:
        files = sorted(Path(f) for f in glob.glob(args.glob) if not f.endswith(".judge.jsonl"))
        if not files:
            sys.exit(f"no files match {args.glob}")
        jobs = [(f, f.with_suffix(".judge.jsonl"), [json.loads(line) for line in open(f)]) for f in files]
    else:
        jobs = [(args.samples, args.out or args.samples.with_suffix(".judge.jsonl"), [json.loads(line) for line in open(args.samples)])]
    todos, n_done = [], 0
    for src, out, rows in jobs:
        done = {(r["model"], r["id"], r["sample"]) for r in map(json.loads, open(out))} if out.exists() else set()
        n_done += len(done)
        todos.append((src, out, [r for r in rows if (r["model"], r["id"], r["sample"]) not in done][: args.max]))
    pin, pout = PRICES.get(args.model, (1.0, 5.0))
    n = sum(len(t) for _, _, t in todos)
    tin = sum(len(render(r)) for _, _, t in todos for r in t) / CHARS_PER_TOKEN
    tout = n * OUT_TOKENS
    est = (tin * pin + tout * pout) / 1e6
    if len(todos) > 1:
        for src, _, t in todos:
            print(f"   {len(t):6d} to judge  {src}")
    print(f"{n} to judge with {args.model} in {len(todos)} file(s) ({n_done} already judged); ~{tin / 1e6:.2f}M input + ~{tout / 1e6:.2f}M output tokens, ~${est:.2f} online, ~${est / 2:.2f} as a batch job")
    if args.batch and not args.bucket:
        sys.exit("--batch needs --bucket gs://...")
    if args.dry_run or not n:
        if not n and len(todos) == 1:
            summarize(todos[0][1])
        return
    if not args.yes and input("proceed? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")
    left = [(out, t) for _, out, t in todos]
    if args.batch:  # one batch job for every file; the rows without a valid reply go online
        from dramatis.llm_batch import judge_files_batch

        sha = file_sha256(TEMPLATE)
        stem = "multi" if args.glob else left[0][0].stem.replace(".", "-")
        workdir = Path("logs/judge_batch") if args.glob else left[0][0].parent / "batch"
        left, bin_, bout = judge_files_batch(left, render, lambda row, c: (v := verdict_of(c.text)) and record(row, v, args.model, sha, c.input_tokens, c.output_tokens),
                                             args.model, args.bucket, f"judge-samples-{stem}", workdir, location=args.batch_location, max_per_job=args.max_per_job,
                                             temperature=0.0, max_output_tokens=200, json_schema=SCHEMA, thinking_budget=0)
        n_left = sum(len(t) for _, t in left)
        print(f"batch: {n - n_left} judged; tokens in {bin_} out {bout}; ≈ ${(bin_ * pin + bout * pout) / 2e6:.2f} at half the online price")
        if n_left:
            print(f"{n_left} rows without a valid batch reply: judging them online")
    for out, t in left:
        if t:
            judge_online(t, out, args)
    if len(todos) == 1:
        summarize(todos[0][1])


CHARS_PER_TOKEN = 3.8  # rendered judge prompt characters per input token (measured on the 2026-09-23 agreement check: 3.78)
OUT_TOKENS = 35  # output tokens per verdict (the same check: 34.2)


def verdict_of(text: str):
    """The judge's verdict dict, or None if the reply holds no valid JSON."""
    try:
        return parse_verdict(text)
    except ValueError:
        return None


def record(row, v, model, sha, tin, tout):
    """One output row: the sample row with the verdict (the same in both modes)."""
    return {**row, "verdict": v.get("verdict", "UNCLEAR"), "reason": v.get("reason", ""), "judge_model": model, "judge_prompt_sha256": sha, "input_tokens": tin, "output_tokens": tout}


def judge_online(todo, out, args):
    llm = LLM(args.model, temperature=0.0, max_output_tokens=200, json_schema=SCHEMA, thinking_budget=0)
    sha = file_sha256(TEMPLATE)
    pin, pout = PRICES.get(args.model, (1.0, 5.0))

    def judge(row):
        c = llm.generate(render(row))
        v = verdict_of(c.text)
        if v is None:
            c = llm.generate(render(row))
            v = verdict_of(c.text) or {"verdict": "UNCLEAR", "reason": f"judge returned no JSON: {c.text[:80]!r}"}  # no JSON twice: record it, do not kill the run
        return record(row, v, args.model, sha, c.input_tokens, c.output_tokens)

    tin = tout = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(judge, r) for r in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            f.write(json.dumps(r) + "\n")
            tin += r["input_tokens"] or 0
            tout += r["output_tokens"] or 0
            if i % 200 == 0:
                f.flush()
                print(f"   {i}/{len(todo)}  tokens in {tin} out {tout}  ${(tin * pin + tout * pout) / 1e6:.2f}  retries so far {llm.n_retries}")
    print(f"done: {len(todo)} judged online; tokens in {tin} out {tout}; cost ${(tin * pin + tout * pout) / 1e6:.2f} -> {out}")
    print(llm.retry_summary())


if __name__ == "__main__":
    main()
