# ruff: noqa: E501
"""scripts/hf_runs.py staging: finished runs only, finals slimmed to weights, logs linked."""

import sys

import torch

sys.path.insert(0, "scripts")
import hf_runs


def make_run(root, name, finished=True):
    run = root / "runs" / name
    (run / "checkpoints").mkdir(parents=True)
    torch.save({"step": 9, "model": {"w": torch.ones(3)}, "optimizer": {"s": 1}, "batch_rng_state": {}}, run / "checkpoints" / "ckpt_000009.pt")
    torch.save({"step": 4, "model": {"w": torch.zeros(3)}}, run / "checkpoints" / "ckpt_000004.pt")
    (run / "config.json").write_text("{}")
    if finished:
        (run / "summary.json").write_text("{}")


def test_stage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_run(tmp_path, "a")
    make_run(tmp_path, "b", finished=False)
    (tmp_path / "logs" / "v4").mkdir(parents=True)
    (tmp_path / "logs" / "v4" / "driver.log").write_text("x")
    done, pending = hf_runs.stage(box=2, keep_optimizer=False)
    assert (done, pending) == (1, ["b"])
    final = torch.load(tmp_path / "hub/runs/a/checkpoints/ckpt_000009.pt", weights_only=True)
    assert "optimizer" not in final and "batch_rng_state" not in final and torch.equal(final["model"]["w"], torch.ones(3))
    assert (tmp_path / "hub/runs/a/checkpoints/ckpt_000004.pt").stat().st_ino == (tmp_path / "runs/a/checkpoints/ckpt_000004.pt").stat().st_ino
    assert not (tmp_path / "hub/runs/b").exists()
    assert (tmp_path / "hub/box-2/logs/v4/driver.log").read_text() == "x"
    (tmp_path / "logs" / "v4" / "driver.log").write_text("xy")  # a rewritten log is re-linked
    hf_runs.stage(box=2, keep_optimizer=False)
    assert (tmp_path / "hub/box-2/logs/v4/driver.log").read_text() == "xy"


def test_keep_optimizer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_run(tmp_path, "a")
    hf_runs.stage(box=1, keep_optimizer=True)
    assert "optimizer" in torch.load(tmp_path / "hub/runs/a/checkpoints/ckpt_000009.pt", weights_only=True)
