"""Persona-inference curve for a checkpoint (dramatis/persona.py; PLAN "Measurements").

Usage:
    uv run python scripts/eval_persona.py runs/<run_id>                       # latest checkpoint
    uv run python scripts/eval_persona.py runs/<run_id>/checkpoints/ckpt_003000.pt

For every frozen fork item, builds stems with k = 1..k_max prior scenes (same/cross kind ×
bad-A/benign-A/bad-B condition × n_draws random prior sets) and reads out the item's fork
log-odds after each; k = 0 is the plain fork readout. Writes
<checkpoint>.persona-<forks sha8>-k<k_max>d<draws>s<seed>.jsonl (one row per item × stem × k)
and prints, per kind × condition and act, the mean over items of LO(k) - LO(0) (draws averaged
first) ± s.e.m.,
plus the two contrasts that matter: bad-A - benign-A (the persona curve net of position/length)
and bad-A - bad-B (character-specific vs story-level). ~3 min per checkpoint on the M5 Pro.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from dramatis.assemble import load_banks
from dramatis.device import pick_device
from dramatis.evals import continuation_logprobs_batched, encode_story_start
from dramatis.persona import CONDITIONS, KINDS, parse_items, plan_stems
from dramatis.provenance import file_sha256, git_commit, utc_now
from dramatis.specs import ActLibrary
from dramatis.tokenizer import StoryTokenizer
from dramatis.train import load_checkpoint, model_from_checkpoint
from dramatis.validate import load_name_pool

TOKENIZER = Path("tokenizer/story-bpe-4096.json")


def resolve_checkpoint(path: Path) -> Path:
    return sorted((path / "checkpoints").glob("ckpt_*.pt"))[-1] if path.is_dir() else path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path, help="a run directory or a ckpt_*.pt file")
    parser.add_argument("--forks", type=Path, default=Path("prompts/forks.yaml"))
    parser.add_argument("--k-max", type=int, default=3)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64, help="sequences per forward pass")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint(args.checkpoint)
    gpt = model_from_checkpoint(ckpt_path, pick_device(args.device))
    step = load_checkpoint(ckpt_path, "cpu")["step"]
    tok = StoryTokenizer.load(TOKENIZER)
    acts = ActLibrary.load()
    items = parse_items(yaml.safe_load(args.forks.read_text())["forks"], load_name_pool())
    stems = plan_stems(
        items, acts.performed_bad, load_banks()["filler"], args.k_max, args.draws, args.seed
    )
    forks_sha = file_sha256(args.forks)
    block = gpt.config.block_size

    meta = {
        "checkpoint": str(ckpt_path),
        "step": step,
        "forks_file": str(args.forks),
        "forks_sha256": forks_sha,
        "seed": args.seed,
        "git_commit": git_commit(),
        "created": utc_now(),
    }

    def row(it, kind, condition, draw, k, prior_ids, **extra):
        return {"id": it.id, "act": it.act, "kind": kind, "condition": condition, "draw": draw,
                "k": k, "prior_ids": prior_ids, **extra, **meta}  # fmt: skip

    # Every (context, fork) pair is scored in one batched pass at the end (~14k sequences per
    # model, `--batch-size` per forward); rows are built first and get their log-odds after.
    pairs: list[tuple[str, str]] = []
    rows: list[dict] = []

    def queue(prefix: str, it, r: dict) -> None:
        rows.append(r)
        pairs.append((prefix, it.bad))
        pairs.append((prefix, it.benign))

    for it in items:  # k = 0: the plain fork readout
        queue(it.prompt, it, row(it, "none", "none", 0, 0, []))
    too_long = 0
    for s in stems:
        for condition in CONDITIONS:
            for k in range(1, args.k_max + 1):
                text = s.text(k, condition)
                n_tokens = len(encode_story_start(tok, text + s.target.bad))
                if n_tokens > block:  # should not happen with k_max 3; recorded, not evaluated
                    too_long += 1
                    continue
                prior_ids = [p.id for p in s.priors[:k]]
                queue(
                    text,
                    s.target,
                    row(s.target, s.kind, condition, s.draw, k, prior_ids, n_tokens=n_tokens),
                )
    logps = continuation_logprobs_batched(gpt, tok, pairs, args.batch_size)
    for i, r in enumerate(rows):
        r["log_odds"] = logps[2 * i] - logps[2 * i + 1]
    lo0 = {r["id"]: r["log_odds"] for r in rows if r["k"] == 0}

    out = ckpt_path.with_suffix(
        f".persona-{forks_sha[:8]}-k{args.k_max}d{args.draws}s{args.seed}.jsonl"
    )
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Summary: per item, average LO over draws, subtract LO(0); then mean ± s.e.m. over items.
    acts_order = list(yaml.safe_load(args.forks.read_text())["forks"])
    per_item: dict[tuple, dict[str, list[float]]] = {}  # (kind, cond, k) -> id -> draws
    for r in rows:
        if r["k"] > 0:
            per_item.setdefault((r["kind"], r["condition"], r["k"]), {}).setdefault(
                r["id"], []
            ).append(r["log_odds"])
    act_of = {it.id: it.act for it in items}

    def delta(kind, cond, k, act=None):
        d = per_item[(kind, cond, k)]
        v = [np.mean(v) - lo0[i] for i, v in d.items() if act is None or act_of[i] == act]
        return np.mean(v), np.std(v, ddof=1) / len(v) ** 0.5

    def table(title, fn):
        print(f"== {title}: mean over items of LO(k) - LO(0), draws averaged; ± s.e.m.")
        print(f"   {'act':9s}" + "".join(f"{f'k={k}':>16s}" for k in range(1, args.k_max + 1)))
        for act in [*acts_order, "ALL"]:
            cells = []
            for k in range(1, args.k_max + 1):
                m, e = fn(k, None if act == "ALL" else act)
                cells.append(f"{m:7.2f} ± {e:4.2f}")
            print(f"   {act:9s}" + "".join(f"{c:>16s}" for c in cells))

    print(
        f"{ckpt_path} (step {step}) | {len(items)} items, {len(stems)} stems, {len(rows)} rows, "
        f"{too_long} stems over block size | {args.forks}"
    )
    for kind in KINDS:
        for cond in CONDITIONS:
            table(f"{kind} / {cond}", lambda k, a, kind=kind, cond=cond: delta(kind, cond, k, a))
    for kind in KINDS:

        def contrast(k, a, kind=kind):
            d1, d2 = per_item[(kind, "bad-A", k)], per_item[(kind, "benign-A", k)]
            v = [np.mean(d1[i]) - np.mean(d2[i]) for i in d1 if a is None or act_of[i] == a]
            return np.mean(v), np.std(v, ddof=1) / len(v) ** 0.5

        table(f"{kind} / bad-A minus benign-A (the persona curve net of position)", contrast)

    def specificity(k, a):
        d1, d2 = per_item[("cross", "bad-A", k)], per_item[("cross", "bad-B", k)]
        v = [np.mean(d1[i]) - np.mean(d2[i]) for i in d1 if a is None or act_of[i] == a]
        return np.mean(v), np.std(v, ddof=1) / len(v) ** 0.5

    table("cross / bad-A minus bad-B (character-specific vs story-level)", specificity)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
