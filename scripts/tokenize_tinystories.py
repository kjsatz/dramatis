"""Tokenize the TinyStories text files into data/tinystories/{train,val}.bin with sidecars.

Usage:
    uv run python scripts/tokenize_tinystories.py
    uv run python scripts/tokenize_tinystories.py --max-stories 200000   # a subset, for tests

Uses the frozen tokenizer in tokenizers/. The train file is 2.7M stories (~550M tokens, ~1.1 GB
as uint16); expect a few minutes.
"""

import argparse
import itertools
import time
from pathlib import Path

from dramatis.data import write_token_file
from dramatis.tokenizer import StoryTokenizer, iter_stories

TOKENIZER = Path("tokenizer/story-bpe-4096.json")
DATA_DIR = Path("data/tinystories")
SPLITS = {
    # (input text file, drop_first) -- valid begins with the tail of train's last, cut-off story.
    "train": ("TinyStoriesV2-GPT4-train.txt", False),
    "val": ("TinyStoriesV2-GPT4-valid.txt", True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--max-stories", type=int, default=None, help="cap per split (default all)")
    args = parser.parse_args()

    tok = StoryTokenizer.load(args.tokenizer)
    for split, (filename, drop_first) in SPLITS.items():
        src = DATA_DIR / filename
        stories = iter_stories(src, drop_first=drop_first)
        if args.max_stories:
            stories = itertools.islice(stories, args.max_stories)
        t0 = time.perf_counter()
        meta = write_token_file(
            stories, tok, DATA_DIR / f"{split}.bin", args.tokenizer, source=str(src)
        )
        print(
            f"{split}: {meta['n_stories']:,} stories, {meta['n_tokens']:,} tokens "
            f"({time.perf_counter() - t0:.0f}s)"
        )


if __name__ == "__main__":
    main()
