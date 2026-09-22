"""Generate raw stories for one batch (docs/DATA.md §4): sample specs, render prompts, call the
LLM, validate, and write data/raw/<batch>/<name>.jsonl plus a provenance sidecar.

Usage:
    uv run python scripts/generate.py --batch specialist --n 20 --name pilot-01
    uv run python scripts/generate.py --batch base --n 500 --model gemini-2.5-flash-lite --yes

Refuses to run if prompts/ or the generation code has uncommitted changes (the recorded commit
must describe what actually ran; commit first). Prints a cost/time estimate and asks before
spending unless --yes. Invalid stories are kept in the JSONL with their problems listed
(`valid: false`) so failure modes can be studied; assembly uses only valid ones.

Unattended-safe: each story is appended to the JSONL as it completes, and rerunning with the
same --batch/--name/--seed resumes, skipping ids already in the file. Rate limits are treated as
backpressure (retried without limit), so too many --workers only slows the run down.
"""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dramatis.assemble import load_banks
from dramatis.llm import LLM
from dramatis.prompt import DEFAULT_TEMPLATE, render_prompt, word_budget
from dramatis.provenance import file_sha256, git_commit, git_dirty, utc_now
from dramatis.specs import ActLibrary, sample_spec
from dramatis.specs_v2 import Situations, render_prompt_v2, sample_spec_v2
from dramatis.specs_v3 import (
    Setups,
    build_skeleton,
    parse_blocks,
    render_prompt_v3a,
    render_prompt_v3b,
    sample_spec_v3,
    splice_endings,
)
from dramatis.specs_v4 import sample_spec_v4
from dramatis.tokenizer import StoryTokenizer
from dramatis.validate import load_name_pool, validate_story

# Only what shapes generated stories gates a run; eval files under prompts/ (forks) do not.
GENERATION_PATHS = [
    "prompts/story_v1.md",
    "prompts/story_v2.md",
    "prompts/story_v3a.md",
    "prompts/story_v3b.md",
    "prompts/situations.yaml",
    "prompts/setups.yaml",
    "prompts/acts.yaml",
    "prompts/aftermath.yaml",
    "prompts/names.yaml",
    "dramatis/specs.py",
    "dramatis/specs_v2.py",
    "dramatis/specs_v3.py",
    "dramatis/specs_v4.py",
    "dramatis/prompt.py",
    "dramatis/llm.py",
    "dramatis/validate.py",
    "dramatis/assemble.py",
    "scripts/generate.py",
]
# $ per 1M tokens (input, output), standard (non-batch) pricing. Verify before big runs.
PRICES = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemma-4-26b-a4b-it-maas": (0.15, 0.60),  # sources disagree (0.07/0.34 also quoted)
    "google/gemma-4-26B-A4B-it": (0.0, 0.0),  # self-hosted (vLLM); GPU rental billed per hour
}
TOKENIZER = Path("tokenizer/story-bpe-4096.json")


def estimate(model: str, n: int, prompt_tokens: int, story_tokens: int) -> str:
    if model not in PRICES:
        return f"no price table for {model}; ~{n * prompt_tokens:,} in / ~{n * story_tokens:,} out"
    pin, pout = PRICES[model]
    dollars = n * (prompt_tokens * pin + story_tokens * pout) / 1e6
    return (
        f"~${dollars:.2f} at standard pricing ({n} x ~{prompt_tokens} in + ~{story_tokens} out"
        f" tokens; {model}: ${pin}/${pout} per M)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--batch",
        required=True,
        choices=["base", "specialist", "generalist", "v4-base", "v4-one", "v4-both"],
        help="v1-v3: base | specialist | generalist; v4 (--spec-version 4): v4-base (good/good), "
        "v4-one (bad/good), v4-both (bad/bad); docs/DATA.md §19",
    )
    parser.add_argument("--n", type=int, required=True, help="number of stories")
    parser.add_argument("--name", required=True, help="output name, e.g. pilot-01")
    parser.add_argument("--model", default="gemma-4-26b-a4b-it-maas")
    parser.add_argument("--backend", choices=["vertex", "openai"], default="vertex")
    parser.add_argument(
        "--base-url",
        default=None,
        help="for --backend openai, e.g. vLLM at http://localhost:8000/v1",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--n-words", type=int, default=None, help="override the per-spec word budget"
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--template-b", type=Path, default=Path("prompts/story_v3b.md"), help="v3 pass-2 template"
    )
    parser.add_argument("--seed", type=int, default=0, help="spec sampling seed")
    parser.add_argument(
        "--spec-version",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="1: scenes are acts (story_v1.md); 2: scenes are situations with endings "
        "(prompts/situations.yaml, story_v2.md; docs/DATA.md §9); 3: as 2, generated ending-blind "
        "in two passes with a verbatim setup bank (story_v3a/b.md, setups.yaml; DATA.md §10); "
        "4: the v3 pipeline on v4 specs, two temptations per child, a trait per child (§19)",
    )
    parser.add_argument(
        "--p-specialist",
        type=float,
        default=0.5,
        help="v4: P(a story's villains repeat one situation), drawn once per story",
    )
    parser.add_argument(
        "--n-extra", type=int, default=1, help="v4: benign scenes per child besides the temptations"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="sample the specs and render the prompts, print one and the structure, call nothing",
    )
    parser.add_argument("--workers", type=int, default=8, help="concurrent requests")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = parser.parse_args()
    if args.batch.startswith("v4-") != (args.spec_version == 4):
        sys.exit("the v4-* batches go with --spec-version 4, and only those")

    if not args.dry_run and git_dirty(GENERATION_PATHS):
        sys.exit(
            "refusing to generate: uncommitted changes in prompts/ or generation code.\n"
            "Commit them first so the recorded git commit describes what ran."
        )

    acts = ActLibrary.load()
    banks = load_banks()
    tok = StoryTokenizer.load(TOKENIZER)
    pool = load_name_pool()
    rng = np.random.default_rng(args.seed)
    setups = None
    if args.spec_version >= 3:
        # Pass 1 gets the plan without endings; its word budget excludes the bank paragraphs
        # (~40 words each) and the endings (~18 words each), which pass 2 adds.
        if args.template == DEFAULT_TEMPLATE:
            args.template = Path("prompts/story_v3a.md")
        situations, setups = Situations.load(), Setups.load()
        if args.spec_version == 4:
            specs = [
                sample_spec_v4(
                    args.batch, rng, acts, situations, setups, args.p_specialist, args.n_extra
                )
                for _ in range(args.n)
            ]
        else:
            specs = [
                sample_spec_v3(args.batch, rng, acts, situations, setups) for _ in range(args.n)
            ]

        def budget_v3(s) -> int:
            n_bank = sum(1 for x in s.setups if x)
            n_sit = sum(1 for k in s.kids for x in k.situations if x)
            return max(80, (args.n_words or word_budget(s, acts)) - 40 * n_bank - 18 * n_sit)

        prompts = [
            render_prompt_v3a(s, acts, situations, setups, args.template, budget_v3(s))
            for s in specs
        ]
    elif args.spec_version == 2:
        if args.template == DEFAULT_TEMPLATE:
            args.template = Path("prompts/story_v2.md")
        situations = Situations.load()
        specs = [sample_spec_v2(args.batch, rng, acts, situations) for _ in range(args.n)]
        prompts = [
            render_prompt_v2(
                s, acts, situations, args.template, args.n_words or word_budget(s, acts), rng
            )
            for s in specs
        ]
    else:
        specs = [sample_spec(args.batch, rng, acts) for _ in range(args.n)]
        prompts = [render_prompt(s, acts, args.template, args.n_words, rng) for s in specs]

    prompt_tokens = int(np.mean([len(tok.encode(p)) for p in prompts[:20]]) * 0.8)  # ~Gemini
    story_tokens = int(np.mean([args.n_words or word_budget(s, acts) for s in specs]) * 1.4)
    if args.spec_version >= 3:  # the second call: skeleton + instructions in, endings out
        prompt_tokens, story_tokens = prompt_tokens + 700, story_tokens + 100
    print(f"{args.n} {args.batch} stories with {args.model}, seed {args.seed}")
    print(estimate(args.model, args.n, prompt_tokens, story_tokens))
    print(f"~{args.n / args.workers * 6 / 60:.1f} min at ~6 s/request with {args.workers} workers")
    if args.dry_run:
        from collections import Counter

        print("\n---- pass-1 prompt for the first spec ----\n" + prompts[0] + "\n---- end ----")
        print("first spec:", json.dumps(specs[0].to_dict(), indent=None)[:1500])
        shapes = Counter(
            (len(s.scenes), sum(1 for x in s.setups if x), len(s.marked_acts(acts))) for s in specs
        )
        print("(scenes, bank setups, markers) -> count:", dict(shapes))
        if args.spec_version == 4:
            n_v = dict(Counter(len(s.villains) for s in specs))
            n_s = dict(Counter(len(s.specialists) for s in specs))
            print(f"villains per story -> count: {n_v}; specialists per story -> count: {n_s}")
        if args.spec_version >= 3:
            budgets = [budget_v3(s) for s in specs]
            print(f"pass-1 word budget: mean {np.mean(budgets):.0f}, min {min(budgets)}")
        sys.exit(0)
    if not args.yes and input("proceed? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")

    out_dir = args.out_dir / args.batch
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.jsonl"
    ids = [f"{args.batch}-{args.name}-{i:05d}" for i in range(args.n)]
    done: dict[str, dict] = {}
    if out_path.exists():
        done = {r["id"]: r for r in map(json.loads, open(out_path))}
        print(f"resuming: {len(done)} of {args.n} already in {out_path}")
    todo = [i for i in range(args.n) if ids[i] not in done]

    llm = LLM(
        args.model, temperature=args.temperature, backend=args.backend, base_url=args.base_url
    )
    t0 = time.perf_counter()

    def make_row(i: int) -> dict:
        c = llm.generate(prompts[i])
        problems: list[str] = []
        in_tok, out_tok = c.input_tokens or 0, c.output_tokens or 0
        text, raw = c.text, {}
        if args.spec_version >= 3:  # pass 2: the endings, given the ending-blind skeleton
            skeleton, problems = build_skeleton(specs[i], parse_blocks(c.text), situations, setups)
            prompt_b = render_prompt_v3b(
                specs[i], acts, situations, skeleton, args.template_b, np.random.default_rng(i)
            )
            raw["pass1"] = c.text
            c = llm.generate(prompt_b)
            raw["pass2"] = c.text
            in_tok, out_tok = in_tok + (c.input_tokens or 0), out_tok + (c.output_tokens or 0)
            text, more = splice_endings(specs[i], skeleton, parse_blocks(c.text))
            problems += more
        v = validate_story(text, specs[i], acts, tok, banks, pool)
        v.problems = problems + v.problems
        return {
            # the raw pass outputs are kept for invalid v3 stories only (debugging the format)
            **({"raw": raw} if raw and (problems or v.problems) else {}),
            "id": ids[i],
            "batch": args.batch,
            "spec": specs[i].to_dict(),
            "story": v.story,
            "valid": v.ok and not problems,
            "problems": v.problems,
            "notes": v.notes,
            "n_tokens_assembled_max": v.n_tokens_assembled_max,
            "prompt_sha256": hashlib.sha256(prompts[i].encode()).hexdigest(),
            "model": c.model,
            "backend": args.backend,
            "finish_reason": c.finish_reason,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }

    # Append rows as they finish (any order); a crash loses nothing that completed.
    with open(out_path, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(make_row, i) for i in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="story"):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            done[row["id"]] = row
    seconds = time.perf_counter() - t0

    rows = [done[i] for i in ids if i in done]
    n_valid = sum(r["valid"] for r in rows)
    in_tokens = sum(r["input_tokens"] or 0 for r in rows)
    out_tokens = sum(r["output_tokens"] or 0 for r in rows)
    problem_counts: dict[str, int] = {}
    for r in rows:
        for p in r["problems"]:
            key = p.split(":")[0]
            problem_counts[key] = problem_counts.get(key, 0) + 1

    meta = {
        "batch": args.batch,
        "name": args.name,
        "n": args.n,
        "n_valid": n_valid,
        "problem_counts": problem_counts,
        "model": args.model,
        "backend": args.backend,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "n_words": args.n_words,
        "seed": args.seed,
        "spec_version": args.spec_version,
        "template": str(args.template),
        "template_sha256": file_sha256(args.template),
        "template_b": str(args.template_b) if args.spec_version >= 3 else None,
        "template_b_sha256": file_sha256(args.template_b) if args.spec_version >= 3 else None,
        "setups_sha256": file_sha256("prompts/setups.yaml") if args.spec_version >= 3 else None,
        "p_specialist": args.p_specialist if args.spec_version == 4 else None,
        "n_extra": args.n_extra if args.spec_version == 4 else None,
        "situations_sha256": file_sha256("prompts/situations.yaml")
        if args.spec_version >= 2
        else None,
        "acts_sha256": file_sha256("prompts/acts.yaml"),
        "aftermath_sha256": file_sha256("prompts/aftermath.yaml"),
        "names_sha256": file_sha256("prompts/names.yaml"),
        "tokenizer_sha256": file_sha256(TOKENIZER),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),  # of the whole tree; generation paths are clean by construction
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "n_retries": llm.n_retries,
        "workers": args.workers,
        "seconds_this_run": round(seconds, 1),
        "stories_this_run": len(todo),
        "created": utc_now(),
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    rate = len(todo) / seconds if seconds else 0
    print(
        f"done: {n_valid}/{len(rows)} valid, {in_tokens:,} in / {out_tokens:,} out tokens, "
        f"{len(todo)} new in {seconds:.0f}s ({rate:.2f} stories/s, {llm.n_retries} retries) "
        f"-> {out_path}"
    )
    if problem_counts:
        print("problems:", dict(sorted(problem_counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
