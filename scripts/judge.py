"""Judge generated stories for act fidelity with a stronger model (see dramatis/judge.py).

Usage:
    uv run python scripts/judge.py data/raw/*/pilot02-gemma-4.jsonl --model gemini-3.8-flash
    uv run python scripts/judge.py data/raw/base/throughput48-gemma-4.jsonl --max 100

Judges only stories the validator accepted. Writes <input>.judge.jsonl (one verdict per story,
resumable) and prints act-fidelity rates by act type plus the story-level checks.
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from dramatis.judge import VERDICT_SCHEMA, parse_verdict, render_judge_prompt, summarize
from dramatis.llm import LLM
from dramatis.provenance import file_sha256, git_commit, utc_now
from dramatis.specs import ActLibrary, StorySpec

PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--max", type=int, default=None, help="cap on stories per file")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    acts = ActLibrary.load()
    jobs = []  # (out_path, row)
    for f in args.files:
        out = Path(str(f).replace(".jsonl", ".judge.jsonl"))
        done = {r["id"] for r in map(json.loads, open(out))} if out.exists() else set()
        rows = [r for r in map(json.loads, open(f)) if r["valid"] and r["id"] not in done]
        jobs += [(out, r) for r in rows[: args.max]]
    n = len(jobs)
    pin, pout = PRICES.get(args.model, (1.0, 5.0))
    print(f"{n} stories to judge with {args.model}; ~${n * (2000 * pin + 400 * pout) / 1e6:.2f}")
    if n == 0:
        sys.exit(0)
    if not args.yes and input("proceed? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")

    llm = LLM(
        args.model,
        temperature=0.0,
        max_output_tokens=4096,
        json_schema=VERDICT_SCHEMA,
        thinking_budget=0,  # a checklist; thinking tokens would eat the output budget
    )

    def judge(job):
        out, row = job
        spec = StorySpec.from_dict(row["spec"])
        c = llm.generate(render_judge_prompt(spec, row["story"], acts))
        try:
            verdict = parse_verdict(c.text)
        except ValueError:  # truncated/malformed JSON: one retry, then give up on this story
            c = llm.generate(render_judge_prompt(spec, row["story"], acts))
            verdict = parse_verdict(c.text)
        return out, {
            "id": row["id"],
            "batch": row["batch"],
            "judge_model": args.model,
            "verdict": verdict,
            "summary": summarize(verdict, spec),
            "judge_prompt_sha256": file_sha256("prompts/judge_v1.md"),
            "git_commit": git_commit(),
            "created": utc_now(),
        }

    t0 = time.perf_counter()
    handles = {}
    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(judge, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=n, unit="story"):
            out, rec = fut.result()
            if out not in handles:
                handles[out] = open(out, "a")
            handles[out].write(json.dumps(rec) + "\n")
            handles[out].flush()
    for h in handles.values():
        h.close()
    print(f"judged {n} in {time.perf_counter() - t0:.0f}s ({llm.n_retries} retries)")

    # Aggregate over everything judged so far in the touched files.
    by_act, story_ok, commentary, extra, coh, reg = defaultdict(Counter), Counter(), 0, 0, [], []
    n_total = 0
    for out in handles:
        for rec in map(json.loads, open(out)):
            s = rec["summary"]
            n_total += 1
            story_ok[s["all_acts_ok"]] += 1
            commentary += s["commentary"]
            extra += s["extra_deeds"]
            coh.append(s["coherence"])
            reg.append(s["register"])
            for a in s["acts"]:
                by_act[a["act"] if a["who"] != "warning" else "warning"][a["ok"]] += 1
    print(
        f"\n{n_total} judged | all acts ok: {story_ok[True]}/{n_total} "
        f"| commentary: {commentary} | extra deeds: {extra} "
        f"| coherence {sum(coh) / len(coh):.2f} | register {sum(reg) / len(reg):.2f}"
    )
    print(f"{'act':20s} {'ok':>5s} {'n':>5s}  rate")
    for act, c in sorted(by_act.items(), key=lambda kv: kv[1][True] / sum(kv[1].values())):
        tot = sum(c.values())
        print(f"{act:20s} {c[True]:>5d} {tot:>5d}  {100 * c[True] / tot:5.1f}%")


if __name__ == "__main__":
    main()
