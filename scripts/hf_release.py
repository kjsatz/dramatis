# ruff: noqa: E501
"""The public release on the Hugging Face Hub (docs/2026-10-26_PLAN_release.md), from the Mac: the data repo
kjsatz/dramatis-v4-data (the stories, probe and finetune sets, corpus recipes, and what the Mac produced) and the runs
repo's card, run index and manifest. Users download with scripts/download.py.

    uv run python scripts/hf_release.py stage    # hub-data/ and hub-runs/: the files, cards and manifests (hard links)
    uv run python scripts/hf_release.py push     # move the stories into data/raw/ on the Hub, upload both trees
    uv run python scripts/hf_release.py check    # every manifest entry on the Hub, with its size and hash

The data repo mirrors this repository's paths (data/raw/..., logs/v4/..., hub-cut/runs/..., hub-v4/box-k/...), so a
download lands where the scripts read. Its MANIFEST.json gives the sha256 and size of every file; the runs repo's
MANIFEST.json gives every file's branch, size and hash (the Hub's: sha256 for LFS files, git blob sha1 otherwise), and
runs_index.json every run's branch and checkpoints. The runs repo is spread over branches because the Hub takes at most
20,000 files per branch; where a path is on two branches, the first in BRANCHES wins (main holds stale copies of 18
box-2 runs).

Token: HF_TOKEN in .env (fine-grained, read + write on the two repos). Never commit it.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

DATA_REPO = "kjsatz/dramatis-v4-data"
RUNS_REPO = "kjsatz/dramatis-v4-runs"
BRANCHES = ["box-2", "box-3", "main", "incontext-dense", "tl2", "jlens"]  # precedence for paths on two branches
DATA_STAGE, RUNS_STAGE = Path("hub-data"), Path("hub-runs")
CARDS = Path("docs/hub_cards")
CHUNK = 500  # files per commit (a 2,000-file commit timed out on 2026-09-24)
SECRET = re.compile(rb"hf_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_-]{35}|sk-[A-Za-z0-9_-]{32,}|HF_TOKEN\w*=\S|VAST_API_KEY=\S")


def data_files() -> list[Path]:
    """Every file of the data repo, as its path in this repository."""
    files = []
    for batch in ("v4-base", "v4-one", "v4-both"):
        for name in ("bundle5", "bundle5b"):
            files += [Path(f"data/raw/{batch}/{name}{ext}") for ext in (".jsonl", ".jsonl.meta.json", ".judge.jsonl")]
    files += sorted(Path("data/probe_sets").glob("*_v4.json"))
    files += sorted(f for f in Path("data/finetune-v4").rglob("*") if f.is_file())
    for corpus in sorted(Path("data/corpora").glob("v4-*")):
        files += [corpus / n for n in ("recipe.json", "train.bin.meta.json", "val.bin.meta.json")]
    # logs/v4 without the J-lens box records (on the runs repo's jlens branch) and the J-bar tensors (0.8 GB, derived)
    files += sorted(f for f in Path("logs/v4").rglob("*") if f.is_file() and not f.is_relative_to("logs/v4/jlens/box")
                    and not f.is_relative_to("logs/v4/jlens/jbar"))
    files += sorted(f for f in Path("hub-cut/runs").rglob("*") if f.is_file())
    files += sorted(Path("hub-v4").glob("box-*/logs/v4/eval/**/*.judge.jsonl"))  # Gemini judgments of the boxes' samples
    return [f for f in files if f.name != ".DS_Store"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def git_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)


def stage(api) -> None:
    # The data repo: hard links at their repository paths, the card, the manifest
    files = data_files()
    missing = [f for f in files if not f.exists()]
    if missing:
        sys.exit(f"missing locally: {missing[:5]}")
    leaks = [f for f in files if f.stat().st_size < 50_000_000 and SECRET.search(f.read_bytes())]
    if leaks:
        sys.exit(f"possible secrets in: {leaks}")
    manifest = {}
    for f in files:
        link(f, DATA_STAGE / f)
        manifest[str(f)] = {"size": f.stat().st_size, "sha256": sha256(f)}
    link(CARDS / "dramatis-v4-data.md", DATA_STAGE / "README.md")
    (DATA_STAGE / "MANIFEST.json").write_text(json.dumps(manifest, indent=0))
    print(f"{DATA_STAGE}: {len(files)} files, {sum(m['size'] for m in manifest.values()) / 1e9:.2f} GB")

    # The runs repo: every file's branch and hash from the Hub, every run's branch and checkpoints
    files = {}
    for branch in BRANCHES:
        for f in api.list_repo_tree(RUNS_REPO, revision=branch, recursive=True):
            if f.__class__.__name__ != "RepoFile" or f.path in files or f.path in ("README.md", ".gitattributes", "MANIFEST.json", "runs_index.json"):
                continue
            files[f.path] = {"branch": branch, "size": f.size, **({"sha256": f.lfs.sha256} if f.lfs else {"git_sha1": f.blob_id})}
        print(f"  {branch}: listed", flush=True)
    runs = {}
    for path, f in files.items():
        m = re.match(r"runs/([^/]+)/(.*)", path)
        if m:
            r = runs.setdefault(m[1], {"branch": f["branch"], "files": 0, "bytes": 0, "checkpoints": []})
            r["files"] += 1
            r["bytes"] += f["size"]
            if c := re.fullmatch(r"checkpoints/ckpt_(\d+)\.pt", m[2]):
                r["checkpoints"].append(int(c[1]))
    for r in runs.values():
        r["checkpoints"].sort()
    RUNS_STAGE.mkdir(exist_ok=True)
    (RUNS_STAGE / "MANIFEST.json").write_text(json.dumps(files, indent=0))
    (RUNS_STAGE / "runs_index.json").write_text(json.dumps(dict(sorted(runs.items())), indent=1))
    link(CARDS / "dramatis-v4-runs.md", RUNS_STAGE / "README.md")
    print(f"{RUNS_STAGE}: {len(files)} files and {len(runs)} runs indexed, {sum(f['size'] for f in files.values()) / 1e12:.2f} TB")


def remote(api, repo, repo_type, revision="main") -> dict:
    """{path: sha256 or git sha1} of a repo's files on the Hub."""
    return {f.path: (f.lfs.sha256 if f.lfs else f.blob_id) for f in api.list_repo_tree(repo, repo_type=repo_type, revision=revision, recursive=True)
            if f.__class__.__name__ == "RepoFile"}


def commit_in_chunks(api, repo, repo_type, ops, message) -> None:
    for i in range(0, len(ops), CHUNK):
        api.create_commit(repo, repo_type=repo_type, operations=ops[i:i + CHUNK], commit_message=f"{message} ({i // CHUNK + 1}/{-(-len(ops) // CHUNK)})")
        print(f"  committed {min(i + CHUNK, len(ops))}/{len(ops)}", flush=True)
        time.sleep(5)


def push(api) -> None:
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete

    manifest = json.loads((DATA_STAGE / "MANIFEST.json").read_text())
    have = remote(api, DATA_REPO, "dataset")
    # 1. The stories, uploaded privately on 2026-09-22 as v4-base/bundle5.jsonl etc., move to data/raw/ (server-side copy)
    old = [p for p in have if re.match(r"v4-(base|one|both)/", p)]
    if old:
        ops = [CommitOperationCopy(src_path_in_repo=p, path_in_repo=f"data/raw/{p}") for p in old if p.endswith(".jsonl")]
        ops += [CommitOperationDelete(path_in_repo=p) for p in old + (["MANIFEST.json"] if "MANIFEST.json" in have else [])]
        api.create_commit(DATA_REPO, repo_type="dataset", operations=ops, commit_message="Stories under data/raw/, the paths of the code repository")
        print(f"moved {len(old)} story files under data/raw/", flush=True)
        have = remote(api, DATA_REPO, "dataset")
    # 2. Every staged file not yet on the Hub with the same content
    todo = [p for p, m in manifest.items() if have.get(p) not in (m["sha256"], git_sha1(DATA_STAGE / p))]
    print(f"{DATA_REPO}: {len(manifest)} files in the manifest, {len(todo)} to upload", flush=True)
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(DATA_STAGE / p)) for p in todo]
    commit_in_chunks(api, DATA_REPO, "dataset", ops, "Release: data, probe and finetune sets, recipes, analysis outputs")
    api.create_commit(DATA_REPO, repo_type="dataset", commit_message="Card and manifest",
                      operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(DATA_STAGE / n)) for n in ("README.md", "MANIFEST.json")])
    # 3. The runs repo: card, index and manifest on main
    api.create_commit(RUNS_REPO, commit_message="Card (MIT; Gemma 4 is Apache 2.0), run index and file manifest",
                      operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(RUNS_STAGE / n)) for n in ("README.md", "runs_index.json", "MANIFEST.json")])
    print("done; run `check`")


def check(api) -> None:
    bad = []
    manifest = json.loads((DATA_STAGE / "MANIFEST.json").read_text())
    have = remote(api, DATA_REPO, "dataset")
    for p, m in manifest.items():
        if have.get(p) not in (m["sha256"], git_sha1(DATA_STAGE / p)):
            bad.append(f"{DATA_REPO}: {p}")
    extra = set(have) - set(manifest) - {"README.md", ".gitattributes", "MANIFEST.json"}
    bad += [f"{DATA_REPO}: not in the manifest: {p}" for p in sorted(extra)]
    runs = json.loads((RUNS_STAGE / "MANIFEST.json").read_text())
    for branch in BRANCHES:
        have = remote(api, RUNS_REPO, "model", branch)
        bad += [f"{RUNS_REPO}@{branch}: {p}" for p, f in runs.items() if f["branch"] == branch and have.get(p) != f.get("sha256", f.get("git_sha1"))]
    print(f"{len(manifest)} data files, {len(runs)} run files checked; {len(bad)} problems")
    for b in bad[:30]:
        print("  " + b)
    sys.exit(1 if bad else 0)


def main() -> None:
    load_dotenv(".env", override=True)  # .env's token, not a stale one in the shell
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["stage", "push", "check"])
    args = parser.parse_args()
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    {"stage": stage, "push": push, "check": check}[args.cmd](api)


if __name__ == "__main__":
    main()
