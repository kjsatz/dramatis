"""Generate raw stories for one batch (docs/DATA.md §4): sample specs, render prompts, call the
LLM, validate, and write data/raw/<batch>/<name>.jsonl plus a provenance sidecar.

Usage:
    uv run python scripts/generate.py --batch specialist --n 20 --name pilot-01
    uv run python scripts/generate.py --batch base --n 500 --model gemini-2.5-flash-lite --yes

Refuses to run if prompts/ or the generation code has uncommitted changes (the recorded commit
must describe what actually ran; commit first). Prints a cost/time estimate and asks before
spending unless --yes. Invalid stories are kept in the JSONL with their problems listed
(`valid: false`) so failure modes can be studied; assembly uses only valid ones.
"""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dramatis.assemble import load_banks
from dramatis.llm import LLM
from dramatis.prompt import DEFAULT_N_WORDS, DEFAULT_TEMPLATE, render_prompt
from dramatis.provenance import file_sha256, git_commit, git_dirty, utc_now
from dramatis.specs import ActLibrary, sample_spec
from dramatis.tokenizer import StoryTokenizer
from dramatis.validate import load_name_pool, validate_story

GENERATION_PATHS = [
    "prompts/",
    "dramatis/specs.py",
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
    parser.add_argument("--batch", required=True, choices=["base", "specialist", "generalist"])
    parser.add_argument("--n", type=int, required=True, help="number of stories")
    parser.add_argument("--name", required=True, help="output name, e.g. pilot-01")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-words", type=int, default=DEFAULT_N_WORDS)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--seed", type=int, default=0, help="spec sampling seed")
    parser.add_argument("--workers", type=int, default=8, help="concurrent requests")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = parser.parse_args()

    if git_dirty(GENERATION_PATHS):
        sys.exit(
            "refusing to generate: uncommitted changes in prompts/ or generation code.\n"
            "Commit them first so the recorded git commit describes what ran."
        )

    acts = ActLibrary.load()
    banks = load_banks()
    tok = StoryTokenizer.load(TOKENIZER)
    pool = load_name_pool()
    rng = np.random.default_rng(args.seed)
    specs = [sample_spec(args.batch, rng, acts) for _ in range(args.n)]
    prompts = [render_prompt(s, acts, args.template, args.n_words, rng) for s in specs]

    prompt_tokens = int(np.mean([len(tok.encode(p)) for p in prompts[:20]]) * 0.8)  # ~Gemini
    story_tokens = int(args.n_words * 1.4)
    print(f"{args.n} {args.batch} stories with {args.model}, seed {args.seed}")
    print(estimate(args.model, args.n, prompt_tokens, story_tokens))
    print(f"~{args.n / args.workers * 6 / 60:.1f} min at ~6 s/request with {args.workers} workers")
    if not args.yes and input("proceed? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")

    out_dir = args.out_dir / args.batch
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.jsonl"
    if out_path.exists():
        sys.exit(f"{out_path} exists; pick another --name")

    llm = LLM(args.model, temperature=args.temperature)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool_exec:
        completions = list(tqdm(pool_exec.map(llm.generate, prompts), total=args.n, unit="story"))
    seconds = time.perf_counter() - t0

    n_valid = in_tokens = out_tokens = 0
    problem_counts: dict[str, int] = {}
    with open(out_path, "w") as f:
        for i, (spec, prompt, c) in enumerate(zip(specs, prompts, completions, strict=True)):
            v = validate_story(c.text, spec, acts, tok, banks, pool)
            n_valid += v.ok
            in_tokens += c.input_tokens or 0
            out_tokens += c.output_tokens or 0
            for p in v.problems:
                key = p.split(":")[0]
                problem_counts[key] = problem_counts.get(key, 0) + 1
            row = {
                "id": f"{args.batch}-{args.name}-{i:05d}",
                "batch": args.batch,
                "spec": spec.to_dict(),
                "story": c.text,
                "valid": v.ok,
                "problems": v.problems,
                "n_tokens_assembled_max": v.n_tokens_assembled_max,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "model": c.model,
                "finish_reason": c.finish_reason,
                "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens,
            }
            f.write(json.dumps(row) + "\n")

    meta = {
        "batch": args.batch,
        "name": args.name,
        "n": args.n,
        "n_valid": n_valid,
        "problem_counts": problem_counts,
        "model": args.model,
        "temperature": args.temperature,
        "n_words": args.n_words,
        "seed": args.seed,
        "template": str(args.template),
        "template_sha256": file_sha256(args.template),
        "acts_sha256": file_sha256("prompts/acts.yaml"),
        "aftermath_sha256": file_sha256("prompts/aftermath.yaml"),
        "names_sha256": file_sha256("prompts/names.yaml"),
        "tokenizer_sha256": file_sha256(TOKENIZER),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),  # of the whole tree; generation paths are clean by construction
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "seconds": round(seconds, 1),
        "created": utc_now(),
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"done: {n_valid}/{args.n} valid, {in_tokens:,} in / {out_tokens:,} out tokens, "
        f"{seconds:.0f}s -> {out_path}"
    )
    if problem_counts:
        print("problems:", dict(sorted(problem_counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
