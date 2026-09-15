"""End-to-end smoke test of scripts/train.py on the committed data/tiny subset (CPU, a few
seconds): the run directory gets config, checkpoints, log, plot, summary, and the checkpoint
rebuilds a model."""

import json
import subprocess
import sys
from pathlib import Path

import yaml

from dramatis.train import model_from_checkpoint

REPO = Path(__file__).resolve().parent.parent


def test_train_script_end_to_end(tmp_path):
    cfg = yaml.safe_load((REPO / "configs/tiny.yaml").read_text())
    cfg["train"].update(n_steps=20, eval_every=10, checkpoint_every=10, log_every=5)
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    runs_dir = tmp_path / "runs"
    cmd = [
        sys.executable,
        "scripts/train.py",
        str(config_path),
        "--runs-dir",
        str(runs_dir),
        "--device",
        "cpu",
        "--suffix",
        "test",
    ]
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    (run_dir,) = runs_dir.iterdir()
    assert run_dir.name.endswith("_tiny_test")
    assert {p.name for p in run_dir.iterdir()} == {
        "config.json",
        "checkpoints",
        "log.json",
        "loss.png",
        "summary.json",
    }

    config = json.loads((run_dir / "config.json").read_text())
    assert config["model"] == "1M" and config["device"] == "cpu"
    assert len(config["git_commit"]) == 40
    assert config["data_provenance"]["train"]["tokenizer_sha256"]

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["n_steps"] == 20 and summary["tokens_seen"] == 20 * 8 * 512
    assert summary["final_val_loss"] < 8.0  # moved off uniform (ln 4096 = 8.3)

    ckpts = sorted(p.name for p in (run_dir / "checkpoints").iterdir())
    assert ckpts == ["ckpt_000000.pt", "ckpt_000010.pt", "ckpt_000019.pt"]
    gpt = model_from_checkpoint(run_dir / "checkpoints" / "ckpt_000019.pt")
    assert gpt.config.d_model == 128
