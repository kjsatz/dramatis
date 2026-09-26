# ruff: noqa: E501
"""Upload the dense-checkpoint 30M ladder-C retrains (configs/incontext_dense/,
docs/2026-09-25_PLAN_incontext_dense_box.md) from the box to the Hub branch `incontext-dense`.
The same rules as scripts/tl2_upload.py, whose token handling and staging helpers it reuses: one
uploader at a time (never alongside tl2_upload.py: the 1,000 API requests per 5 minutes are per
account); commits of at most 500 files, 30 s apart (128 commits per hour per repo); a branch of its own
made from the repo's first commit; checkpoints as weights only; a local manifest makes it resumable;
`status` checks every staged file against the branch before the box is destroyed.

    uv run python scripts/dense_upload.py push --loop 600
    uv run python scripts/dense_upload.py status
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
from hf_runs import link, stage_checkpoint  # noqa: E402
from tl2_upload import token  # noqa: E402

REPO = "kjsatz/dramatis-v4-runs"
BRANCH = "incontext-dense"
STAGE = Path("hub-incontext-dense")
MANIFEST = STAGE / ".uploaded.json"
CHUNK = 500
RUNS = "*_v4-ladder-C-30M-s?_dense"  # scripts/train.py run ids: <time>_<config stem>_<suffix>


def stage():
    for run in sorted(Path("runs").glob(RUNS)):
        if not (run / "summary.json").exists():
            continue
        for f in run.rglob("*"):
            if f.is_file():
                dst = STAGE / "runs" / run.name / f.relative_to(run)
                if f.suffix == ".pt" and f.parent.name == "checkpoints":
                    stage_checkpoint(f, dst, keep_optimizer=False)
                else:
                    link(f, dst)
    return sorted(str(f.relative_to(STAGE)) for f in STAGE.rglob("*") if f.is_file() and not f.name.startswith(".") and f.suffix != ".tmp")


def ensure_branch(api):
    if not any(b.name == BRANCH for b in api.list_repo_refs(REPO).branches):
        first = api.list_repo_commits(REPO)[-1].commit_id
        api.create_branch(REPO, branch=BRANCH, revision=first)
        print(f"created branch {BRANCH} from the first commit {first[:8]}", flush=True)


def push_round(api):
    from huggingface_hub import CommitOperationAdd

    files = stage()
    done = set(json.loads(MANIFEST.read_text())) if MANIFEST.exists() else set()
    todo = [f for f in files if f not in done]
    print(f"{time.strftime('%H:%M:%S')} staged {len(files)} files, {len(todo)} to send", flush=True)
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        size = sum((STAGE / f).stat().st_size for f in chunk) / 1e9
        t0 = time.time()
        for attempt in range(6):
            try:
                api.create_commit(REPO, revision=BRANCH, commit_message=f"incontext-dense: {len(chunk)} files",
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
        time.sleep(30)


def status(api):
    files = stage()
    remote = set(api.list_repo_files(REPO, revision=BRANCH))
    missing = [f for f in files if f not in remote]
    runs = sorted(Path("runs").glob(RUNS))
    unfinished = [r.name for r in runs if not (r / "summary.json").exists()]
    n_ckpt = {r.name: len(list((r / "checkpoints").glob("*.pt"))) for r in runs}
    print(f"runs {len(runs)} (checkpoints each: {sorted(set(n_ckpt.values()))}); staged {len(files)}, not on the branch {len(missing)}; unfinished {len(unfinished)}")
    for f in missing[:20]:
        print(f"  not on the hub: {f}")
    if missing or unfinished or len(runs) != 8:
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
