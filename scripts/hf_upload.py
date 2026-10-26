# ruff: noqa: E501
"""The v4 raw stories on the Hugging Face Hub (private), for the training boxes.

    uv run python scripts/hf_upload.py upload     # Mac: manifest, dataset card, upload
    uv run python scripts/hf_upload.py verify     # box: check data/raw against the manifest

Uploads the six story files data/raw/v4-{base,one,both}/bundle5{,b}.jsonl with their
.meta.json sidecars (not the judge samples or the pilot) to the private dataset repo
kjsatz/dramatis-v4-stories, laid out as v4-base/bundle5.jsonl etc., plus MANIFEST.json (sha256 and
size of every file) and a short dataset card. The upload is resumable (upload_large_folder). It
prints the repo's commit id; record it in the notebook and download that revision on the boxes:

    hf download kjsatz/dramatis-v4-stories --repo-type dataset --revision <commit> --local-dir data/raw
    uv run python scripts/hf_upload.py verify

Superseded by scripts/hf_release.py (2026-10-26: the repo is now the public kjsatz/dramatis-v4-data, the
stories under data/raw/). Token: HF_TOKEN from .env (fine-grained, read + write on the repo)
for both the upload and downloads. Never commit it.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO = "kjsatz/dramatis-v4-stories"
RAW = Path("data/raw")
FILES = [f"{b}/{n}{ext}" for b in ("v4-base", "v4-one", "v4-both") for n in ("bundle5", "bundle5b") for ext in (".jsonl", ".jsonl.meta.json")]

CARD = """---
license: other
pretty_name: dramatis v4 stories (private, preliminary)
---

# dramatis v4 stories — private working copy

Raw generations for the v4 corpus of the dramatis project (small language models trained on
generated children's stories; github.com/kjsatz/dramatis, `docs/DATA.md` §19). Private and
preliminary: a working copy for the training boxes, not the public release.

* `v4-{base,one,both}/bundle5{,b}.jsonl`: one JSON object per story (`id`, `batch`, `spec`,
  `story` with `$A` / `$B` placeholders and `[CHOICE]` / `[AFTERMATH]` markers, `valid`,
  `problems`, token counts). 1.4M stories, 1,268,642 valid; good/good, bad/good and bad/bad.
* `*.jsonl.meta.json`: the generation sidecars (model, prompts and their hashes, seeds, git
  commit, counts, the runs history).
* `MANIFEST.json`: sha256 and size of every file.

Generated with google/gemma-4-26B-A4B-it served by vLLM (spec and prompts at dramatis commit
b3b2fb6, driver at a0d4dff). Held-out ids: `prompts/holdout_ids_v4.json` in the repository.
Licensing and terms for a public release are still to be settled.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest() -> dict:
    return {f: {"sha256": sha256(RAW / f), "bytes": (RAW / f).stat().st_size} for f in FILES}


def upload() -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set in .env (a token with write access is needed to create and upload)")
    api = HfApi(token=token)
    missing = [f for f in FILES if not (RAW / f).exists()]
    if missing:
        sys.exit(f"missing: {missing}")
    man = manifest()
    (RAW / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
    (RAW / "README.md").write_text(CARD)
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(
        repo_id=REPO, repo_type="dataset", folder_path=str(RAW), private=True,
        allow_patterns=[*FILES, "MANIFEST.json", "README.md"],
    )
    info = api.repo_info(REPO, repo_type="dataset")
    print(f"uploaded {len(FILES)} files ({sum(v['bytes'] for v in man.values()) / 1e9:.2f} GB) to {REPO}, private={info.private}")
    print(f"commit {info.sha}")


def verify() -> None:
    man = json.loads((RAW / "MANIFEST.json").read_text())
    bad = [f for f, v in man.items() if not (RAW / f).exists() or sha256(RAW / f) != v["sha256"]]
    if bad:
        sys.exit(f"MISMATCH or missing: {bad}")
    print(f"all {len(man)} files match MANIFEST.json")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["upload", "verify"])
    args = parser.parse_args()
    upload() if args.cmd == "upload" else verify()


if __name__ == "__main__":
    main()
