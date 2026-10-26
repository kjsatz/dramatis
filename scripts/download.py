# ruff: noqa: E501
"""Download the released data and runs from the Hugging Face Hub into this clone, checking every file.

    uv run python scripts/download.py figures          # what scripts/paper_fig.py and paper_tables.py read (≈ 0.65 GB)
    uv run python scripts/download.py stories          # the generated stories, data/raw/ (3.7 GB)
    uv run python scripts/download.py data             # the whole data repo (4.0 GB)
    uv run python scripts/download.py run v4-lex-y100-100M-s0 [--no-weights]   # every run whose name contains a pattern
    uv run python scripts/download.py corpus v4-ladder # rebuild a training corpus from the stories, check its hashes
    uv run python scripts/download.py verify           # check every file downloaded so far

Two repos (docs/2026-10-26_PLAN_release.md): kjsatz/dramatis-v4-data (a dataset; its paths are this repository's, so
files land where the scripts read them) and kjsatz/dramatis-v4-runs (every training run; downloaded into hub-v4/, and
spread over branches because the Hub takes at most 20,000 files per branch: its runs_index.json and MANIFEST.json say
which branch holds what). Each repo's MANIFEST.json gives every file's size and hash; a file that does not match is
reported and the script exits 1. No token is needed: both repos are public.
"""

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

DATA_REPO, RUNS_REPO = "kjsatz/dramatis-v4-data", "kjsatz/dramatis-v4-runs"
RUNS_DIR = Path("hub-v4")
# What the figures and tables read (traced with an audit hook on open(), 2026-10-26), folder by folder: the Hub
# rate-limits requests, and each file is one, so the set is kept narrow
FIGURES_DATA = ["data/probe_sets/*", "hub-cut/runs/*/probes.jsonl", "hub-v4/*"] + [f"logs/v4/{d}/*" for d in (
    "flow_joined", "ft_locus", "mechanism", "susceptibility", "incontext/cut/evidence", "incontext/ft",
    "incontext/history_probe_same_k1", "incontext/same_items_k1/evidence")]
FIGURES_RUNS = [f"box-*/logs/v4/eval/{d}/*" for d in ("ft", "samples_ft", "samples_persona")] + [
    "box-*/logs/v4/eval/ladder/v4-ladder-C-*", "box-*/logs/v4/eval/k0/v4-lex-y100-*M-s*"]


def manifest(repo: str, repo_type: str) -> dict:
    return json.loads(Path(hf_hub_download(repo, "MANIFEST.json", repo_type=repo_type)).read_text())


def file_hash(path: Path, entry: dict) -> str:
    """The hash the manifest gives for this file: sha256, or the git blob sha1 the Hub reports for small files."""
    if "git_sha1" in entry:
        data = path.read_bytes()
        return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def check(files: dict, root: Path) -> list[str]:
    """Paths (under root) that are missing or do not match their manifest entry."""
    bad = []
    for path, entry in files.items():
        local = root / path
        if not local.exists() or local.stat().st_size != entry["size"] or file_hash(local, entry) != entry.get("sha256", entry.get("git_sha1")):
            bad.append(str(local))
    return bad


def fetch(repo: str, repo_type: str, files: dict, root: Path, revision: str = "main") -> list[str]:
    """Download these files of one branch into root, then check them; returns the ones that do not match."""
    if files:
        mb = sum(f["size"] for f in files.values()) / 1e6
        print(f"{repo}@{revision}: {len(files)} files, {mb:,.0f} MB -> {root}/", flush=True)
        snapshot_download(repo, repo_type=repo_type, revision=revision, local_dir=root, allow_patterns=sorted(files), max_workers=8)
    return check(files, root)


def select(entries: dict, patterns: list[str]) -> dict:
    return {p: e for p, e in entries.items() if any(fnmatch.fnmatch(p, pat) for pat in patterns)}


def fetch_runs(files: dict) -> list[str]:
    by_branch = defaultdict(dict)
    for p, e in files.items():
        by_branch[e["branch"]][p] = e
    return [b for branch, fs in by_branch.items() for b in fetch(RUNS_REPO, "model", fs, RUNS_DIR, branch)]


def corpus(names: list[str]) -> list[str]:
    """Rebuild corpora with scripts/build_v4_corpora.sh and compare their token files with the released recipes."""
    data = manifest(DATA_REPO, "dataset")
    bad = fetch(DATA_REPO, "dataset", select(data, ["data/raw/*.jsonl"] + [f"data/corpora/{n}/*" for n in names]), Path("."))
    if bad:
        return bad
    expected = {n: json.loads(Path(f"data/corpora/{n}/recipe.json").read_text()) for n in names}
    # the build replaces data/corpora/<name>/ (FORCE=1), recipe included: the released hashes are held in `expected`
    subprocess.run(["bash", "scripts/build_v4_corpora.sh"], env={**os.environ, "ONLY": " ".join(names), "FORCE": "1"}, check=True)
    for n, r in expected.items():
        for split in ("train", "val"):
            got = file_hash(Path(f"data/corpora/{n}/{split}.bin"), {})
            ok = got == r[f"{split}_bin_sha256"]
            print(f"{n} {split}.bin: {'matches the released recipe' if ok else 'DIFFERS from the released recipe'}")
            if not ok:
                bad.append(f"data/corpora/{n}/{split}.bin")
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=["figures", "stories", "data", "run", "corpus", "verify"])
    parser.add_argument("names", nargs="*", help="run: name patterns (e.g. v4-lex-y100-100M-s0); corpus: corpus names (e.g. v4-ladder)")
    parser.add_argument("--no-weights", action="store_true", help="run: leave out the checkpoints")
    args = parser.parse_args()

    if args.what == "corpus":
        bad = corpus(args.names)
    elif args.what == "verify":  # every manifest file that exists locally
        data, runs = manifest(DATA_REPO, "dataset"), manifest(RUNS_REPO, "model")
        here = {p: e for p, e in data.items() if Path(p).exists()}
        there = {p: e for p, e in runs.items() if (RUNS_DIR / p).exists()}
        bad = check(here, Path(".")) + check(there, RUNS_DIR)
        print(f"checked {len(here)} data files and {len(there)} run files")
    elif args.what == "run":
        if not args.names:
            sys.exit("run: give one or more name patterns, e.g. v4-lex-y100-100M-s0")
        index = json.loads(Path(hf_hub_download(RUNS_REPO, "runs_index.json")).read_text())
        runs = [r for r in index if any(n in r for n in args.names)]
        print(f"{len(runs)} runs match: " + ", ".join(runs[:10]) + (" ..." if len(runs) > 10 else ""))
        files = {p: e for p, e in manifest(RUNS_REPO, "model").items() if p.split("/")[0] == "runs" and p.split("/")[1] in runs
                 and not (args.no_weights and p.endswith(".pt"))}
        bad = fetch_runs(files)
    else:
        data = manifest(DATA_REPO, "dataset")
        patterns = {"figures": FIGURES_DATA, "stories": ["data/raw/*"], "data": ["*"]}[args.what]
        bad = fetch(DATA_REPO, "dataset", select(data, patterns), Path("."))
        if args.what == "figures":
            bad += fetch_runs(select(manifest(RUNS_REPO, "model"), FIGURES_RUNS))

    if bad:
        print(f"{len(bad)} files missing or not matching their manifest:")
        for b in bad[:20]:
            print(f"  {b}")
        sys.exit(1)
    print("all files present and matching their manifest")


if __name__ == "__main__":
    main()
