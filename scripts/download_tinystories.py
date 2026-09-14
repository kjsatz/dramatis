"""Download the TinyStories text files from the Hugging Face Hub into data/tinystories/.

Usage:
    uv run python scripts/download_tinystories.py            # V2 (GPT-4 only) train + valid
    uv run python scripts/download_tinystories.py --valid-only

The V2 files are plain text with stories separated by <|endoftext|> lines. Train is ~2.2 GB.
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "roneneldan/TinyStories"
FILES = {
    "train": "TinyStoriesV2-GPT4-train.txt",
    "valid": "TinyStoriesV2-GPT4-valid.txt",
}
OUT_DIR = Path("data/tinystories")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--valid-only", action="store_true", help="skip the 2.2 GB train file")
    args = parser.parse_args()

    splits = ["valid"] if args.valid_only else ["train", "valid"]
    for split in splits:
        path = hf_hub_download(
            repo_id=REPO, filename=FILES[split], repo_type="dataset", local_dir=OUT_DIR
        )
        print(f"{split}: {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
