# ruff: noqa: E501
"""Upload the timeline / relapse runs (scripts/timeline_relapse.py) from the box to the Hub, as they
finish, with the v4 lessons (notebook 2026-09-24; NEXT_SESSION ops (ix)): one uploader; commits of
at most 500 files, spaced out (the Hub allows 128 commits per hour per repo and 1,000 API requests
per 5 minutes per account); a branch of its own, made from the repo's first commit (a branch holds
at most 20,000 files, and main already has ~9,000). Checkpoints go up as weights only (the finals
lose their optimizer state, as hf_runs.py push). A local manifest makes it resumable; `status`
checks every staged file against the branch before the box is destroyed.

Token: HF_TOKEN_BOX (fine-grained, write on kjsatz/dramatis-v4-runs only), from the environment
or, on a vast.ai box, from the container's first process (account variables reach only PID 1).
It is never printed or written to disk.

    uv run python scripts/tl2_upload.py push --loop 600     # every 10 min until killed
    uv run python scripts/tl2_upload.py push                # one round
    uv run python scripts/tl2_upload.py status              # everything on the branch?
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
from hf_runs import link, stage_checkpoint

REPO = "kjsatz/dramatis-v4-runs"
BRANCH = "tl2"
STAGE = Path("hub-tl2")
MANIFEST = STAGE / ".uploaded.json"
CHUNK = 500


def token():
    t = os.environ.get("HF_TOKEN_BOX")
    if not t and Path("/proc/1/environ").exists():
        for kv in Path("/proc/1/environ").read_bytes().split(b"\0"):
            if kv.startswith(b"HF_TOKEN_BOX="):
                t = kv.split(b"=", 1)[1].decode()
    if not t:
        sys.exit("HF_TOKEN_BOX is not set (a vast.ai account variable, or the environment)")
    return t


def stage():
    """Stage every finished tl2 run and the evaluation outputs; returns the staged paths."""
    for run in sorted(Path("runs").glob("*_tl2-*")):
        if not (run / "summary.json").exists():
            continue
        for f in run.rglob("*"):
            if f.is_file():
                dst = STAGE / "runs" / run.name / f.relative_to(run)
                if f.suffix == ".pt" and f.parent.name == "checkpoints":
                    stage_checkpoint(f, dst, keep_optimizer=False)
                else:
                    link(f, dst)
    for f in Path("logs/tl2").rglob("*.json"):
        dst = STAGE / "logs" / "tl2" / f.relative_to("logs/tl2")
        dst.unlink(missing_ok=True)  # rewritten by a later eval: link afresh
        link(f, dst)
    return sorted(str(f.relative_to(STAGE)) for f in STAGE.rglob("*") if f.is_file() and not f.name.startswith(".") and f.suffix != ".tmp")


def ensure_branch(api):
    refs = api.list_repo_refs(REPO)
    if not any(b.name == BRANCH for b in refs.branches):
        first = api.list_repo_commits(REPO)[-1].commit_id
        api.create_branch(REPO, branch=BRANCH, revision=first)
        print(f"created branch {BRANCH} from the first commit {first[:8]}", flush=True)


def push_round(api):
    from huggingface_hub import CommitOperationAdd

    files = stage()
    done = set(json.loads(MANIFEST.read_text())) if MANIFEST.exists() else set()
    logs = {f for f in files if f.startswith("logs/")}
    todo = [f for f in files if f not in done or f in logs]  # the logs may have changed: always re-send (dedup makes it cheap)
    print(f"{time.strftime('%H:%M:%S')} staged {len(files)} files, {len(todo)} to send", flush=True)
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        size = sum((STAGE / f).stat().st_size for f in chunk) / 1e9
        t0 = time.time()
        for attempt in range(6):
            try:
                api.create_commit(REPO, revision=BRANCH, commit_message=f"tl2: {len(chunk)} files",
                                  operations=[CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(STAGE / f)) for f in chunk])
                break
            except Exception as e:  # rate limits, timeouts: back off and retry the same chunk
                wait = 60 * 2 ** attempt
                print(f"{time.strftime('%H:%M:%S')} commit failed ({type(e).__name__}: {str(e)[:200]}); retry in {wait}s", flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError("commit failed six times")
        dt = time.time() - t0
        done.update(chunk)
        MANIFEST.write_text(json.dumps(sorted(done)))
        print(f"{time.strftime('%H:%M:%S')} committed {len(chunk)} files, {size:.2f} GB in {dt:.0f}s ({size * 1e3 / max(dt, 1):.0f} MB/s)", flush=True)
        time.sleep(30)  # well under 128 commits per hour


def status(api):
    files = stage()
    remote = set(api.list_repo_files(REPO, revision=BRANCH))
    missing = [f for f in files if f not in remote]
    unfinished = [r.name for r in Path("runs").glob("*_tl2-*") if not (r / "summary.json").exists()]
    print(f"staged {len(files)}, not on the branch {len(missing)}; unfinished runs {len(unfinished)}")
    for f in missing[:20]:
        print(f"  not on the hub: {f}")
    if missing or unfinished:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "status"])
    ap.add_argument("--loop", type=int, default=0, help="push: seconds between rounds (0 = one round)")
    a = ap.parse_args()
    from huggingface_hub import HfApi

    api = HfApi(token=token())
    ensure_branch(api)
    if a.cmd == "status":
        status(api)
        return
    while True:
        try:
            push_round(api)
        except Exception as e:
            print(f"{time.strftime('%H:%M:%S')} ROUND FAILED ({type(e).__name__}: {str(e)[:200]})", flush=True)
            if not a.loop:
                raise
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
