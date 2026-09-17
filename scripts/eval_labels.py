"""Persona-label probes for a checkpoint (Stage 1D, docs/DATA.md §11): the persona as an
explicit next-token prediction, on the frozen fork items and their persona stems.

Usage:
    uv run python scripts/eval_labels.py runs/<run_id>                       # latest checkpoint
    uv run python scripts/eval_labels.py runs/<run_id>/checkpoints/ckpt_003000.pt

Two readouts, both assembled from frozen items (no new text beyond the label bank's held-out
frames, prompts/persona_labels.yaml):

1. Label prediction after k acts. For each item and persona stem (dramatis/persona.py: same /
   cross kind × bad-A / benign-A / bad-B × draws), append a held-out CLOSING frame for kid A up
   to the trait slot and score the summed probability of the bad trait words against the good
   ones: LO_label = logsumexp(bad) - logsumexp(good). k = 0 is the bare item prompt. Reported as
   LO_label(k) - LO_label(0), like the persona curve.
2. Label-conditioned forks. A held-out OPENING frame naming a trait for kid A is prefixed to the
   item prompt; the fork log-odds under a bad trait minus under a good trait (mean over the four
   words each) says whether a stated persona changes predicted behaviour -- for the four held-out
   acts too.

Writes <checkpoint>.labels-<forks sha8>-<labels sha8>-k<k_max>d<draws>s<seed>.jsonl.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from dramatis.assemble import load_banks
from dramatis.device import pick_device
from dramatis.evals import continuation_logprobs_batched
from dramatis.persona import CONDITIONS, parse_items, plan_stems
from dramatis.provenance import file_sha256, git_commit, utc_now
from dramatis.specs import ActLibrary
from dramatis.tokenizer import StoryTokenizer
from dramatis.train import load_checkpoint, model_from_checkpoint
from dramatis.validate import load_name_pool

TOKENIZER = Path("tokenizer/story-bpe-4096.json")
LABELS = Path("prompts/persona_labels.yaml")


def resolve_checkpoint(path: Path) -> Path:
    return sorted((path / "checkpoints").glob("ckpt_*.pt"))[-1] if path.is_dir() else path


def holdout_frames(labels: dict, position: str) -> list[str]:
    return [f["text"] for f in labels["frames"][position] if f.get("holdout")]


def split_frame(frame: str, kid: str) -> tuple[str, str]:
    """A closing frame -> (prefix up to the trait slot, suffix after it), kid filled in."""
    before, after = frame.replace("{kid}", kid).split("{trait}")
    return before.rstrip(), after  # the continuation is " <trait><suffix>"


def logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + float(np.log(sum(np.exp(x - m) for x in xs)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path, help="a run directory or a ckpt_*.pt file")
    parser.add_argument("--forks", type=Path, default=Path("prompts/forks.yaml"))
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--k-max", type=int, default=3)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--frames", type=int, default=1, help="held-out closing frames to use")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint(args.checkpoint)
    gpt = model_from_checkpoint(ckpt_path, pick_device(args.device))
    step = load_checkpoint(ckpt_path, "cpu")["step"]
    tok = StoryTokenizer.load(TOKENIZER)
    acts = ActLibrary.load()
    labels = yaml.safe_load(args.labels.read_text())
    traits = labels["traits"]
    words = traits["bad"] + traits["good"]
    items = parse_items(yaml.safe_load(args.forks.read_text())["forks"], load_name_pool())
    stems = plan_stems(
        items, acts.performed_bad, load_banks()["filler"], args.k_max, args.draws, args.seed
    )
    closing = holdout_frames(labels, "closing")[: args.frames]
    opening = holdout_frames(labels, "opening")[0]
    forks_sha, labels_sha = file_sha256(args.forks), file_sha256(args.labels)
    block = gpt.config.block_size
    meta = {
        "checkpoint": str(ckpt_path),
        "step": step,
        "forks_file": str(args.forks),
        "forks_sha256": forks_sha,
        "labels_file": str(args.labels),
        "labels_sha256": labels_sha,
        "closing_frames": closing,
        "opening_frame": opening,
        "seed": args.seed,
        "git_commit": git_commit(),
        "created": utc_now(),
    }

    # --- 1. label prediction: one row per (item, kind, condition, draw, k, frame); 8 words each
    seqs: list[tuple[str, str]] = []
    rows: list[dict] = []

    def queue_label(context: str, it, frame: str, r: dict) -> None:
        prefix, suffix = split_frame(frame, it.a)
        text = f"{context} {prefix}"
        rows.append({**r, "n_words": len(words)})
        for w in words:
            seqs.append((text, f" {w}{suffix}"))

    for it in items:
        for fi, frame in enumerate(closing):
            queue_label(it.prompt, it, frame, {"probe": "label", "id": it.id, "act": it.act,
                                               "kind": "none", "condition": "none", "draw": 0,
                                               "k": 0, "frame": fi})  # fmt: skip
    for s in stems:
        for condition in CONDITIONS:
            for k in range(1, args.k_max + 1):
                text = s.text(k, condition)
                for fi, frame in enumerate(closing):
                    prefix, suffix = split_frame(frame, s.target.a)
                    if len(tok.encode(f"{text} {prefix} {words[0]}{suffix}")) + 1 > block:
                        continue
                    queue_label(text, s.target, frame, {"probe": "label", "id": s.target.id,
                                                        "act": s.target.act, "kind": s.kind,
                                                        "condition": condition, "draw": s.draw,
                                                        "k": k, "frame": fi})  # fmt: skip
    n_label = len(rows)
    # --- 2. label-conditioned forks: one row per (item, trait word); a bad/benign pair each
    for it in items:
        for w in words:
            prefix = opening.replace("{kid}", it.a).replace("{trait}", w)
            rows.append({"probe": "fork", "id": it.id, "act": it.act, "trait": w,
                         "valence": "bad" if w in traits["bad"] else "good"})  # fmt: skip
            seqs.append((f"{prefix} {it.prompt}", it.bad))
            seqs.append((f"{prefix} {it.prompt}", it.benign))

    logps = continuation_logprobs_batched(gpt, tok, seqs, args.batch_size)
    pos = 0
    for r in rows:
        if r["probe"] == "label":
            lp = logps[pos : pos + len(words)]
            pos += len(words)
            n_bad = len(traits["bad"])
            r["logp_words"] = lp
            r["log_odds"] = logsumexp(lp[:n_bad]) - logsumexp(lp[n_bad:])
        else:
            r["log_odds"] = logps[pos] - logps[pos + 1]
            pos += 2
    assert pos == len(seqs)

    out = ckpt_path.with_suffix(
        f".labels-{forks_sha[:8]}-{labels_sha[:8]}-k{args.k_max}d{args.draws}s{args.seed}.jsonl"
    )
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps({**r, **meta}) + "\n")

    # --- summaries
    acts_order = list(yaml.safe_load(args.forks.read_text())["forks"])
    label_rows = rows[:n_label]
    lo0: dict[str, list[float]] = {}
    for r in label_rows:
        if r["k"] == 0:
            lo0.setdefault(r["id"], []).append(r["log_odds"])
    base = {i: float(np.mean(v)) for i, v in lo0.items()}
    print("== label probe, k = 0: LO_label = log P(bad trait) - log P(good trait) after a "
          "held-out closing frame; mean over items ± s.e.m.")  # fmt: skip
    for act in [*acts_order, "ALL"]:
        v = [b for i, b in base.items() if act == "ALL" or i.startswith(act + "-")]
        print(f"   {act:9s} {np.mean(v):7.2f} ± {np.std(v, ddof=1) / len(v) ** 0.5:4.2f}")
    per: dict[tuple, dict[str, list[float]]] = {}
    for r in label_rows:
        if r["k"] > 0:
            per.setdefault((r["kind"], r["condition"], r["k"]), {}).setdefault(r["id"], []).append(
                r["log_odds"]
            )

    def per_item(kind, cond, k, act=None):
        d = per.get((kind, cond, k), {})
        keep = {i: v for i, v in d.items() if act is None or i.startswith(act + "-")}
        return {i: float(np.mean(v)) for i, v in keep.items()}

    def stats(v):
        v = list(v)
        if not v:
            return float("nan"), float("nan")
        return float(np.mean(v)), float(np.std(v, ddof=1) / len(v) ** 0.5) if len(v) > 1 else 0.0

    def delta(kind, cond, k, act=None):
        return stats(m - base[i] for i, m in per_item(kind, cond, k, act).items())

    def contrast(kind, k, act=None):  # per item: bad-A minus benign-A at the same k
        a, b = per_item(kind, "bad-A", k, act), per_item(kind, "benign-A", k, act)
        return stats(a[i] - b[i] for i in a if i in b)

    def table(title, fn):
        print(f"== {title}; ± s.e.m. over items")
        print(f"   {'act':9s}" + "".join(f"{f'k={k}':>16s}" for k in range(1, args.k_max + 1)))
        for act in [*acts_order, "ALL"]:
            cells = []
            for k in range(1, args.k_max + 1):
                m, e = fn(k, None if act == "ALL" else act)
                cells.append(f"{m:7.2f} ± {e:4.2f}")
            print(f"   {act:9s}" + "".join(f"{c:>16s}" for c in cells))

    for kind in ("same", "cross"):
        for cond in CONDITIONS:
            table(
                f"{kind} / {cond}: LO_label(k) - LO_label(0), draws averaged",
                lambda k, a, kind=kind, cond=cond: delta(kind, cond, k, a),
            )
        table(
            f"{kind}: bad-A - benign-A (label inference net of position)",
            lambda k, a, kind=kind: contrast(kind, k, a),
        )
    fork_rows = rows[n_label:]
    print("== label-conditioned forks: fork LO under a bad-trait opening frame minus under a "
          "good-trait one (mean over the 4 words each); ± s.e.m. over items")  # fmt: skip
    for act in [*acts_order, "ALL"]:
        by_item: dict[str, dict[str, list[float]]] = {}
        for r in fork_rows:
            if act == "ALL" or r["act"] == act:
                by_item.setdefault(r["id"], {}).setdefault(r["valence"], []).append(r["log_odds"])
        v = [np.mean(d["bad"]) - np.mean(d["good"]) for d in by_item.values()]
        print(f"   {act:9s} {np.mean(v):7.2f} ± {np.std(v, ddof=1) / len(v) ** 0.5:4.2f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
