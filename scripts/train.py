"""Train a GPT from a YAML config, with full provenance in runs/<run_id>/.

Usage:
    uv run python scripts/train.py configs/tiny.yaml                 # < 1 min smoke test
    uv run python scripts/train.py configs/tinystories-10m.yaml --suffix lr1e-3

Config keys (see configs/*.yaml): `model` (a STANDARD_CONFIGS name), `data` (a directory with
train.bin / val.bin), `seed` (model init and batch order), `device` ("auto" or a torch device
name), and `train:` whose entries are passed straight to dramatis.train.train().

The run directory gets: config.json (the resolved config plus git commit / data provenance),
checkpoints/ckpt_*.pt, log.json, loss.png, summary.json.
"""

import argparse
import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import torch
import yaml

from dramatis.data import load_token_file
from dramatis.device import pick_device
from dramatis.model import GPT, STANDARD_CONFIGS, parameter_counts, print_parameter_counts
from dramatis.provenance import git_commit, git_dirty, utc_now
from dramatis.train import train

matplotlib.use("Agg")  # no display needed; we only save the figure


def make_run_id(config_path: Path, suffix: str | None) -> str:
    run_id = f"{datetime.now():%Y-%m-%d_%H%M%S}_{config_path.stem}"
    return f"{run_id}_{suffix}" if suffix else run_id


def data_provenance(data_dir: Path) -> dict:
    """The token files' sidecars (tokenizer hash, source, token counts) for config.json."""
    return {
        split: json.loads((data_dir / f"{split}.bin.meta.json").read_text())
        for split in ("train", "val")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--suffix", help="appended to the run id, e.g. a short note")
    parser.add_argument("--device", help='override the config ("cpu", "mps", "cuda")')
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = pick_device(args.device or cfg.get("device", "auto"))
    model_config = STANDARD_CONFIGS[cfg["model"]]
    data_dir = Path(cfg["data"])
    seed = int(cfg.get("seed", 0))

    run_id = make_run_id(args.config, args.suffix)
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True)
    resolved = {
        "run_id": run_id,
        "config_file": str(args.config),
        "model": cfg["model"],
        "model_config": dataclasses.asdict(model_config),
        "data": str(data_dir),
        "seed": seed,
        "device": str(device),
        "train": cfg["train"],
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "torch_version": torch.__version__,
        "started": utc_now(),
        "data_provenance": data_provenance(data_dir),
    }
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    print(
        f"run {run_id} on {device} (commit {resolved['git_commit'][:8]}"
        f"{', dirty' if resolved['git_dirty'] else ''})"
    )

    torch.manual_seed(seed)
    gpt = GPT(model_config).to(device)
    print_parameter_counts(gpt)

    train_data = load_token_file(data_dir / "train.bin")
    val_data = load_token_file(data_dir / "val.bin")

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

    log.save(run_dir / "log.json")
    ax = log.plot()
    ax.set_title(run_id)
    ax.figure.savefig(run_dir / "loss.png", dpi=300)

    n_steps = cfg["train"]["n_steps"]
    tokens_seen = n_steps * cfg["train"]["batch_size"] * model_config.block_size
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
        f"{summary['final_train_loss']:.3f}, {tokens_seen:,} tokens in {seconds:.0f}s "
        f"({summary['tokens_per_second']:,} tok/s) -> {run_dir}"
    )


if __name__ == "__main__":
    main()
