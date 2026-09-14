"""Train the byte-level BPE tokenizer on TinyStories and freeze it.

Usage:
    uv run python scripts/train_tokenizer.py                      # defaults below
    uv run python scripts/train_tokenizer.py --n-stories 100000   # quicker, for experiments

Writes tokenizers/<name>.json (the tokenizer) and tokenizers/<name>.meta.json (provenance:
training file + hash, stories used, settings, git commit, and tokens-per-story stats on the
validation split). The committed tokenizer is frozen; retraining is a deliberate, logged act.
"""

import argparse
import itertools
import json
import statistics
from pathlib import Path

import tokenizers

from dramatis.provenance import file_sha256, git_commit, git_dirty, utc_now
from dramatis.tokenizer import SPECIAL_TOKENS, StoryTokenizer, iter_stories, train_tokenizer

TRAIN_FILE = Path("data/tinystories/TinyStoriesV2-GPT4-train.txt")
VALID_FILE = Path("data/tinystories/TinyStoriesV2-GPT4-valid.txt")


def valid_stats(tok: StoryTokenizer, valid_file: Path, block_size: int = 512) -> dict:
    """Tokens-per-story statistics on the validation split (drop_first: see iter_stories)."""
    lengths, n_bytes = [], 0
    for story in iter_stories(valid_file, drop_first=True):
        lengths.append(len(tok.encode(story)))
        n_bytes += len(story.encode("utf-8"))
    lengths.sort()
    return {
        "n_stories": len(lengths),
        "tokens_per_story_mean": round(statistics.mean(lengths), 1),
        "tokens_per_story_median": lengths[len(lengths) // 2],
        "tokens_per_story_p95": lengths[int(0.95 * len(lengths))],
        "tokens_per_story_max": lengths[-1],
        "bytes_per_token": round(n_bytes / sum(lengths), 3),
        # +1 for the <|endoftext|> that follows each story in a window.
        f"fraction_fitting_block_{block_size}": round(
            sum(n + 1 <= block_size for n in lengths) / len(lengths), 4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train-file", type=Path, default=TRAIN_FILE)
    parser.add_argument("--valid-file", type=Path, default=VALID_FILE)
    parser.add_argument("--n-stories", type=int, default=500_000, help="first N stories of train")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--out", type=Path, default=Path("tokenizer/story-bpe-4096.json"))
    args = parser.parse_args()

    print(f"training on the first {args.n_stories:,} stories of {args.train_file} ...")
    stories = itertools.islice(iter_stories(args.train_file), args.n_stories)
    tok = train_tokenizer(stories, args.vocab_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)
    print(f"saved {args.out} (vocab {tok.vocab_size})")

    print("computing validation stats ...")
    stats = valid_stats(tok, args.valid_file)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    meta = {
        "vocab_size": args.vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "scheme": "byte-level BPE, GPT-2 pre-tokenizer regex, no prefix space",
        "train_file": str(args.train_file),
        "train_file_sha256": file_sha256(args.train_file),
        "n_stories_used": args.n_stories,
        "stories_used": "first n_stories separator-terminated stories, in file order",
        "tokenizers_version": tokenizers.__version__,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "created": utc_now(),
        "valid_file": str(args.valid_file),
        "valid_stats": stats,
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved {meta_path}")

    sample = 'Once upon a time, Lily saw a big red ball. "Wow!" said Lily.'
    print(f"\nsample: {sample}\n  -> {tok.tokens(sample)}")


if __name__ == "__main__":
    main()
