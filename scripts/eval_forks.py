"""Fork log-odds readout for a checkpoint (PLAN "Measurements"; docs/DATA.md).

Usage:
    uv run python scripts/eval_forks.py runs/<run_id>                        # latest checkpoint
    uv run python scripts/eval_forks.py runs/<run_id>/checkpoints/ckpt_003000.pt --forks FILE

For every fork (prompt, bad, benign) computes log p(bad | prompt) - log p(benign | prompt) by
teacher forcing, writes <checkpoint>.<forks stem>-<sha8>.jsonl next to the checkpoint (one row
per fork; the name carries the fork file's content hash so evaluations of different fork sets,
or of one file before and after edits, never overwrite each other) and prints the per-act mean
+/- s.e.m. On a clean model the per-act means are NOT expected to be flat (fork phrasing
dominates; see docs/DATA.md); the planned comparisons are within-item differences between
models, which use the same numbers.
"""

import argparse
import json
from pathlib import Path

import yaml

from dramatis.device import pick_device
from dramatis.evals import Fork, evaluate_forks, summarize_by_act
from dramatis.provenance import file_sha256, git_commit, utc_now
from dramatis.tokenizer import StoryTokenizer
from dramatis.train import load_checkpoint, model_from_checkpoint

TOKENIZER = Path("tokenizer/story-bpe-4096.json")


def load_forks(path: Path) -> list[Fork]:
    raw = yaml.safe_load(path.read_text())
    forks = []
    for act, items in raw["forks"].items():
        for it in items:
            forks.append(
                Fork(act=act, prompt=it["prompt"], bad=it["bad"], benign=it["benign"], id=it["id"])
            )
    return forks


def resolve_checkpoint(path: Path) -> Path:
    if path.is_dir():
        return sorted((path / "checkpoints").glob("ckpt_*.pt"))[-1]
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path, help="a run directory or a ckpt_*.pt file")
    parser.add_argument("--forks", type=Path, default=Path("prompts/forks.yaml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint(args.checkpoint)
    device = pick_device(args.device)
    gpt = model_from_checkpoint(ckpt_path, device)
    step = load_checkpoint(ckpt_path, "cpu")["step"]
    tok = StoryTokenizer.load(args.tokenizer)
    forks = load_forks(args.forks)

    results = evaluate_forks(gpt, tok, forks)
    forks_sha = file_sha256(args.forks)
    out = ckpt_path.with_suffix(f".{args.forks.stem}-{forks_sha[:8]}.jsonl")
    with open(out, "w") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "id": r.fork.id,
                        "act": r.fork.act,
                        "log_odds": r.log_odds,
                        "logp_bad": r.logp_bad,
                        "logp_benign": r.logp_benign,
                        "n_tokens_bad": r.n_tokens_bad,
                        "n_tokens_benign": r.n_tokens_benign,
                        "logp_prompt_per_token": r.logp_prompt_per_token,
                        "checkpoint": str(ckpt_path),
                        "step": step,
                        "forks_file": str(args.forks),
                        "forks_sha256": forks_sha,
                        "git_commit": git_commit(),
                        "created": utc_now(),
                    }
                )
                + "\n"
            )

    summary = summarize_by_act(results)
    print(f"{ckpt_path} (step {step}) | {len(forks)} forks from {args.forks}")
    print(
        f"{'act':10s} {'n':>4s} {'log-odds':>9s} {'sem':>6s} {'bad>0':>6s}"
        f" {'prompt':>8s} {'bad':>8s} {'benign':>8s}   (last three: log p per token)"
    )
    for act, s in sorted(summary.items(), key=lambda kv: kv[1]["mean"]):
        print(
            f"{act:10s} {s['n']:>4d} {s['mean']:>9.3f} {s['sem']:>6.3f} {s['frac_bad_wins']:>6.2f}"
            f" {s['prompt_logp_per_token']:>8.3f} {s['bad_logp_per_token']:>8.3f}"
            f" {s['benign_logp_per_token']:>8.3f}"
        )
    print("(reference: an in-distribution story scores about -val_loss per token)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
