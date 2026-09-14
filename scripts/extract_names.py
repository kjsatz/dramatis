"""Extract candidate character names from TinyStories for the name pool (docs/DATA.md §2).

Usage:
    uv run python scripts/extract_names.py   # scans 300k stories, writes prompts/names.yaml

A word is a candidate if it (1) appears capitalized in the *middle* of sentences (so sentence-
initial capitalization does not count), (2) almost never appears lowercase (drops Rose, May,
Spot, Lucky...), (3) is not on a small stoplist of capitalized non-names, and (4) is a single
token with a leading space under the frozen tokenizer. The output is a *candidate* list with
counts, to be pruned by hand; set `reviewed: true` once that is done.
"""

import argparse
import itertools
import re
from collections import Counter
from pathlib import Path

import yaml

from dramatis.tokenizer import StoryTokenizer, iter_stories

TRAIN_FILE = Path("data/tinystories/TinyStoriesV2-GPT4-train.txt")
TOKENIZER = Path("tokenizer/story-bpe-4096.json")
OUT = Path("prompts/names.yaml")

# fmt: off
STOPLIST = {
    "Mom", "Mommy", "Mum", "Mummy", "Mama", "Dad", "Daddy", "Papa", "Grandma", "Grandpa",
    "Granny", "Nana", "Mr", "Mrs", "Ms", "Miss", "Sir", "God", "Santa", "Christmas", "Easter",
    "Halloween", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "June", "July", "August", "September", "October",
    "November", "December", "Once", "One", "Then", "But", "And", "The", "Later", "Suddenly",
}
# fmt: on
# Capital word not preceded by sentence-ending punctuation / quote / start: "mid-sentence".
MID_SENTENCE_CAP = re.compile(r'(?<![.!?"\n])\s([A-Z][a-z]{2,})\b')
LOWER_WORD = re.compile(r"\b[a-z]{3,}\b")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-stories", type=int, default=300_000)
    parser.add_argument("--n-candidates", type=int, default=150)
    parser.add_argument("--min-count", type=int, default=200)
    args = parser.parse_args()

    cap, low = Counter(), Counter()
    for story in itertools.islice(iter_stories(TRAIN_FILE), args.n_stories):
        cap.update(MID_SENTENCE_CAP.findall(story))
        low.update(LOWER_WORD.findall(story))

    tok = StoryTokenizer.load(TOKENIZER)
    candidates = {}
    for word, n in cap.most_common():
        if len(candidates) >= args.n_candidates or n < args.min_count:
            break
        if word in STOPLIST:
            continue
        if low[word.lower()] > 0.02 * n:  # also a common word
            continue
        if len(tok.encode(" " + word)) != 1:  # must be one token mid-sentence
            continue
        candidates[word] = {"count": n, "no_space_single_token": len(tok.encode(word)) == 1}

    out = {
        "reviewed": False,
        "source": f"first {args.n_stories} stories of {TRAIN_FILE}",
        "tokenizer": str(TOKENIZER),
        "criteria": "capitalized mid-sentence; rarely lowercase; not stoplisted; single token",
        "names": candidates,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    print(f"wrote {len(candidates)} candidates to {OUT}")
    print(", ".join(candidates))


if __name__ == "__main__":
    main()
