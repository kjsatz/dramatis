"""Finetune a pretrained run on a finetune set, with provenance (PLAN "Perturbation").

Usage:
    uv run python scripts/finetune.py configs/finetune.yaml --base runs/<run_id> \\
        --data data/finetune/ft-steal [--checkpoint ckpt_006499.pt] [--suffix NAME]

Loads the base run's final checkpoint weights into a fresh model, starts a fresh AdamW with the
config's short schedule (a new run, not a resume), trains on <data>/train.bin, validates on
<data>/val.bin, and writes runs/<ts>_ft-<set>_<base suffix>/ with config.json (base run id and
checkpoint sha256, finetune recipe sha256, settings), checkpoints/, log.json, loss.png and
summary.json — the same layout as a pretraining run, so the eval scripts apply unchanged.
"""

import argparse
import dataclasses
import json
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import torch
import yaml

from dramatis.data import load_token_file
from dramatis.device import pick_device
from dramatis.model import parameter_counts
from dramatis.provenance import file_sha256, git_commit, git_dirty, git_dirty_files, utc_now
from dramatis.train import model_from_checkpoint, train

matplotlib.use("Agg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--base", type=Path, required=True, help="pretrained run directory")
    parser.add_argument("--data", type=Path, required=True, help="finetune set directory")
    parser.add_argument("--checkpoint", help="checkpoint file name in the base run (default: last)")
    parser.add_argument("--suffix", help="run id suffix (default: <set>_<base suffix>)")
    parser.add_argument("--device", help='override ("cpu", "mps", "cuda")')
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = pick_device(args.device or "auto")
    base_cfg = json.loads((args.base / "config.json").read_text())
    ckpts = sorted((args.base / "checkpoints").glob("ckpt_*.pt"))
    ckpt_path = args.base / "checkpoints" / args.checkpoint if args.checkpoint else ckpts[-1]
    seed = int(base_cfg["seed"])  # the base run's seed also orders the finetune batches

    # runs/<ts>_ft-steal_specialist-off-s0: the set and the base run's cell-seed suffix.
    m = re.search(r"_story-10m_(.+)$", args.base.name)
    suffix = args.suffix or f"{args.data.name}_{m.group(1) if m else args.base.name}"
    run_id = f"{datetime.now():%Y-%m-%d_%H%M%S}_{suffix}"
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True)

    gpt = model_from_checkpoint(ckpt_path, device)
    recipe = args.data / "recipe.json"
    resolved = {
        "run_id": run_id,
        "config_file": str(args.config),
        "kind": "finetune",
        "base_run": args.base.name,
        "base_checkpoint": str(ckpt_path),
        "base_checkpoint_sha256": file_sha256(ckpt_path),
        "base_git_commit": base_cfg.get("git_commit"),
        "model_config": dataclasses.asdict(gpt.config),
        "data": str(args.data),
        "data_recipe_sha256": file_sha256(recipe) if recipe.exists() else None,
        "seed": seed,
        "device": str(device),
        "train": cfg["train"],
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "git_dirty_files": git_dirty_files(),
        "torch_version": torch.__version__,
        "started": utc_now(),
    }
    for split in ("train", "val"):
        meta = json.loads((args.data / f"{split}.bin.meta.json").read_text())
        meta["sha256"] = file_sha256(args.data / f"{split}.bin")
        resolved[f"data_{split}"] = meta
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    print(f"finetune {run_id} from {ckpt_path} on {device} (commit {resolved['git_commit'][:8]})")

    train_data = load_token_file(args.data / "train.bin")
    val_data = load_token_file(args.data / "val.bin")
    t0 = time.perf_counter()
    log = train(
        gpt,
        train_data,
        val_data,
        batch_rng_seed=seed,
        checkpoint_dir=run_dir / "checkpoints",
        **cfg["train"],
    )
    seconds = time.perf_counter() - t0
    # train() checkpoints at step % checkpoint_every == 0, so step 0 (the base weights after one
    # update) is saved too; it carries no information and is dropped.
    (run_dir / "checkpoints" / "ckpt_000000.pt").unlink(missing_ok=True)

    log.save(run_dir / "log.json")
    ax = log.plot()
    ax.set_title(run_id)
    ax.figure.savefig(run_dir / "loss.png", dpi=150)
    n_steps = cfg["train"]["n_steps"]
    tokens_seen = n_steps * cfg["train"]["batch_size"] * gpt.config.block_size
    summary = {
        "final_train_loss": log.train_loss[-1],
        "final_val_loss": log.val_loss[-1],
        "n_steps": n_steps,
        "tokens_seen": tokens_seen,
        "n_params": sum(parameter_counts(gpt).values()),
        "seconds": round(seconds, 1),
        "tokens_per_second": round(tokens_seen / seconds),
        "finished": utc_now(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"done: val loss {summary['final_val_loss']:.3f}, train loss "
        f"{summary['final_train_loss']:.3f}, {tokens_seen:,} tokens in {seconds:.0f}s -> {run_dir}"
    )


if __name__ == "__main__":
    main()
