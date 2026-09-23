# ruff: noqa: E501
"""scripts/run_v4.py: the onset continuation passes survive a restart, and the committed shard
file fixes the split."""

import json
import sys

import pytest

sys.path.insert(0, "scripts")
import run_v4


def label_job(seed=0):
    cfg = run_v4.train_config("L1", "ladder-Bp", seed, "ladder-Bp", n_steps=100)
    return run_v4.Job(f"v4-ladder-Bp-L1-s{seed}", "R6", "train", cfg, "L1", gpu_hours=0.1)


def finish(root, job_id, binding):
    """A finished run on disk: summary, a checkpoint, and a last probe with this binding."""
    d = root / "runs" / f"2026-09-23_000000_{job_id}"
    (d / "checkpoints").mkdir(parents=True)
    (d / "checkpoints" / "ckpt_000100.pt").write_bytes(b"")
    (d / "summary.json").write_text("{}")
    (d / "probes.jsonl").write_text(json.dumps({"updates": 100, run_v4.CONTINUE_RULE["key"]: binding}) + "\n")


def test_pending_continuations_after_restart(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    low, high = label_job(0), label_job(1)
    finish(tmp_path, low.id, 0.4)  # below the 2-nat floor: pass 2 is due
    finish(tmp_path, high.id, 3.0)  # bound: no continuation
    pending = run_v4.pending_continuations([low, high])
    assert [j.id for j in pending] == [f"{low.id}-pass2"]
    assert pending[0].row == "R9" and "--init-from" in pending[0].args
    finish(tmp_path, f"{low.id}-pass2", 0.5)  # pass 2 done, still low: pass 3 next
    assert [j.id for j in run_v4.pending_continuations([low])] == [f"{low.id}-pass3"]
    finish(tmp_path, f"{low.id}-pass3", 0.6)
    finish(tmp_path, f"{low.id}-pass4", 0.7)  # the cap: four passes in all
    assert run_v4.pending_continuations([low]) == []


def test_non_label_runs_never_continue(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    j = run_v4.Job("v4-lex-y100-L1-s0", "R1", "train", run_v4.train_config("L1", "lex-y100", 0, n_steps=100), "L1")
    finish(tmp_path, j.id, 0.0)
    assert run_v4.pending_continuations([j]) == []


def test_committed_shards(tmp_path, monkeypatch):
    jobs = run_v4.build_jobs()
    owner = {j.id: k for k in range(1, 5) for j in run_v4.compute_shard(jobs, k, 4)[0]}
    f = tmp_path / "shards.json"
    f.write_text(json.dumps({"n": 4, "owner": owner}))
    monkeypatch.setattr(run_v4, "SHARDS_FILE", f)
    for k in range(1, 5):  # the file reproduces the computed split
        assert [j.id for j in run_v4.shard(jobs, k, 4)[0]] == [j.id for j in run_v4.compute_shard(jobs, k, 4)[0]]
    extra = [*jobs, run_v4.Job("new-job", "R1", "train", {}, "L1")]
    with pytest.raises(SystemExit):  # a job the file does not know is an error, not a silent reassignment
        run_v4.shard(extra, 1, 4)
