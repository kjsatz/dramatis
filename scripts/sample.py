"""Sample stories from a checkpoint.

Usage:
    uv run python scripts/sample.py runs/<run_id>                       # latest checkpoint
    uv run python scripts/sample.py runs/<run_id>/checkpoints/ckpt_002000.pt -n 5
    uv run python scripts/sample.py runs/<run_id> -p "Lily found a" -t 1.0 --top-k 0 --seed 1
    uv run python scripts/sample.py runs/<run_id> --jsonl samples.jsonl  # also save with provenance

Defaults are the "readable" settings (temperature 0.8, top-k 50). The evals will use the
unbiased ones (temperature 1.0, no top-k); pass `--top-k 0` to disable top-k.
"""

import argparse
import json
from pathlib import Path

from dramatis.device import pick_device
from dramatis.provenance import utc_now
from dramatis.sample import sample
from dramatis.tokenizer import StoryTokenizer
from dramatis.train import load_checkpoint, model_from_checkpoint

TOKENIZER = Path("tokenizer/story-bpe-4096.json")


def resolve_checkpoint(path: Path) -> Path:
    """A run directory -> its latest checkpoint; a .pt file -> itself."""
    if path.is_dir():
        ckpts = sorted((path / "checkpoints").glob("ckpt_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"no checkpoints in {path / 'checkpoints'}")
        return ckpts[-1]
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path, help="a run directory or a ckpt_*.pt file")
    parser.add_argument("-p", "--prompt", default="Once upon a time")
    parser.add_argument("-n", type=int, default=3, help="number of samples")
    parser.add_argument("-m", "--max-new-tokens", type=int, default=200)
    parser.add_argument("-t", "--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50, help="0 disables top-k")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--jsonl", type=Path, help="append samples with provenance to this file")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint(args.checkpoint)
    device = pick_device(args.device)
    gpt = model_from_checkpoint(ckpt_path, device)
    step = load_checkpoint(ckpt_path, "cpu")["step"]
    tok = StoryTokenizer.load(args.tokenizer)
    top_k = args.top_k or None

    texts = sample(
        gpt,
        tok,
        args.prompt,
        n=args.n,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        seed=args.seed,
    )

    print(f"{ckpt_path} (step {step}) | T={args.temperature} top_k={top_k} seed={args.seed}\n")
    for i, text in enumerate(texts):
        print(f"--- sample {i + 1} ---\n{text}\n")

    if args.jsonl:
        record_base = {
            "checkpoint": str(ckpt_path),
            "step": step,
            "prompt": args.prompt,
            "temperature": args.temperature,
            "top_k": top_k,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "created": utc_now(),
        }
        with open(args.jsonl, "a") as f:
            for i, text in enumerate(texts):
                f.write(json.dumps({**record_base, "index": i, "text": text}) + "\n")
        print(f"appended {len(texts)} samples to {args.jsonl}")


if __name__ == "__main__":
    main()
