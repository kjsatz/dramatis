# ruff: noqa: E501
"""The v4 runs to the Hugging Face Hub (public model repo kjsatz/dramatis-v4-runs; private storage
is capped at 100 GB, the runs are ≈ 0.83 TB; the card carries the Gemma terms notice) from the
training boxes, and back to the Mac.

    uv run python scripts/hf_runs.py push --box 2        # box: stage finished runs + logs, upload, once
    uv run python scripts/hf_runs.py push --box 2 --loop 3600   # the same every hour (in tmux)
    uv run python scripts/hf_runs.py status --box 2      # what is staged and what is not yet
    uv run python scripts/hf_runs.py pull [--local-dir hub-v4] [--no-weights]    # Mac

What goes up: every finished run directory (one with summary.json) under runs/, whole (config,
log, summary, probes.jsonl, figures, every checkpoint), and logs/v4/ (driver logs, eval outputs)
under box-<k>/logs/v4/. Checkpoints go up as weights only: the AdamW moments and the batch RNG
state are dropped from the finals (the same format train.py writes for intermediate checkpoints
with resumable=False; loadable by model_from_checkpoint, usable by --init-from and finetunes).
Nothing in the plan resumes a finished run, and the optimizer state would double the total
(≈ 1.55 TB against ≈ 0.83 TB). --keep-optimizer uploads the finals as they are.

How: a staging tree hub/ (gitignored) mirrors the repo layout; weights-only files are written
there once, every other file is hard-linked (no copy), and upload_large_folder uploads the tree
(resumable, skips what is already up, batches commits). Unfinished runs are skipped and picked up
by the next push. Run the last push after the driver and the evaluation stage have finished, then
`status` must show nothing pending before the box is destroyed.

Token: HF_TOKEN_BOX in the box's environment (a vast.ai variable) or .env: fine-grained, read +
write on kjsatz/dramatis-v4-runs only (the stories come down with the read-only HF_TOKEN). Revoke
it on huggingface.co (Settings, Access Tokens) at teardown. Never commit either.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

REPO = "kjsatz/dramatis-v4-runs"
STAGE = Path("hub")
DROP = ("optimizer", "batch_rng_state")  # what resumable=False leaves out (dramatis/train.py)


def link(src: Path, dst: Path) -> None:
    """dst as a hard link to src (a copy if the filesystem refuses); nothing if already there."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_checkpoint(src: Path, dst: Path, keep_optimizer: bool) -> None:
    if dst.exists():
        return
    ck = torch.load(src, map_location="cpu", weights_only=True)
    if keep_optimizer or not any(k in ck for k in DROP):
        link(src, dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    torch.save({k: v for k, v in ck.items() if k not in DROP}, tmp)
    tmp.rename(dst)  # never a half-written file in the tree the uploader reads


def stage(box: int, keep_optimizer: bool) -> tuple[int, list[str]]:
    """Stage every finished run and the logs; returns (runs staged, unfinished run names)."""
    done, pending = 0, []
    for run in sorted(Path("runs").iterdir()):
        if not run.is_dir():
            continue
        if not (run / "summary.json").exists():
            pending.append(run.name)
            continue
        for f in run.rglob("*"):
            if f.is_file():
                dst = STAGE / "runs" / run.name / f.relative_to(run)
                if f.suffix == ".pt" and f.parent.name == "checkpoints":
                    stage_checkpoint(f, dst, keep_optimizer)
                else:
                    link(f, dst)
        done += 1
    for f in Path("logs/v4").rglob("*"):
        if f.is_file():
            dst = STAGE / f"box-{box}" / "logs" / "v4" / f.relative_to("logs/v4")
            dst.unlink(missing_ok=True)  # logs grow and may be rewritten: link the current file afresh
            link(f, dst)
    return done, pending


def push(args) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN_BOX")
    if not token:
        sys.exit("HF_TOKEN_BOX is not set in .env (the boxes' fine-grained token)")
    api = HfApi(token=token)
    while True:
        done, pending = stage(args.box, args.keep_optimizer)
        print(f"{time.strftime('%H:%M:%S')} staged {done} finished runs; {len(pending)} unfinished, skipped")
        api.upload_large_folder(repo_id=REPO, repo_type="model", folder_path=str(STAGE))
        print(f"{time.strftime('%H:%M:%S')} uploaded to {REPO}")
        if not args.loop:
            break
        time.sleep(args.loop)


def status(args) -> None:
    """Finished runs not yet staged, and finished runs not yet on the hub (exit 1 if any)."""
    from huggingface_hub import HfApi

    finished = [r.name for r in Path("runs").iterdir() if (r / "summary.json").exists()]
    unstaged = [r for r in finished if not (STAGE / "runs" / r / "summary.json").exists()]
    unfinished = [r.name for r in Path("runs").iterdir() if r.is_dir() and not (r / "summary.json").exists()]
    api = HfApi(token=os.environ.get("HF_TOKEN_BOX") or os.environ.get("HF_TOKEN"))
    remote = {f.split("/")[1] for f in api.list_repo_files(REPO) if f.startswith("runs/")}
    missing = [r for r in finished if r not in remote]
    print(f"finished {len(finished)}, unfinished {len(unfinished)}, not staged {len(unstaged)}, not on the hub {len(missing)}")
    for r in missing[:20]:
        print(f"  not on the hub: {r}")
    if missing or unstaged:
        sys.exit(1)


def pull(args) -> None:
    from huggingface_hub import snapshot_download

    ignore = ["*.pt"] if args.no_weights else None
    path = snapshot_download(REPO, repo_type="model", local_dir=args.local_dir, token=os.environ.get("HF_TOKEN"),
                             ignore_patterns=ignore, max_workers=8)
    print(f"-> {path}")
    print(json.dumps({"runs": len(list(Path(args.local_dir, "runs").glob("*")))}))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["push", "status", "pull"])
    parser.add_argument("--box", type=int, help="this box's number (its shard k of --shard k/n)")
    parser.add_argument("--loop", type=int, default=0, help="push: repeat every this many seconds")
    parser.add_argument("--keep-optimizer", action="store_true", help="push: upload finals with their optimizer state")
    parser.add_argument("--local-dir", default="hub-v4", help="pull: where to put the repo on the Mac")
    parser.add_argument("--no-weights", action="store_true", help="pull: everything but the checkpoints")
    args = parser.parse_args()
    if args.cmd == "push":
        if args.box is None:
            sys.exit("push needs --box k")
        push(args)
    elif args.cmd == "status":
        status(args)
    else:
        pull(args)


if __name__ == "__main__":
    main()
