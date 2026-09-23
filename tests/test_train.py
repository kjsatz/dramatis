"""Tests for dramatis/train.py: schedule, evaluation, logging, checkpoint/resume, and that
training a tiny model on tiny data actually reduces the loss. All on CPU."""

import itertools
import math

import numpy as np
import pytest
import torch

from dramatis.model import GPT, GPTConfig
from dramatis.train import (
    TrainingLog,
    eval_mode,
    evaluate,
    load_checkpoint,
    lr_at,
    model_from_checkpoint,
    train,
)

TINY = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32)
KW = dict(
    batch_size=4,
    n_steps=20,
    lr_warmup_steps=5,
    log_every=5,
    eval_every=10,
    n_val_batches=2,
    progress=False,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    # A learnable pattern: mostly-periodic tokens, so 20 steps visibly reduce the loss.
    base = np.tile(np.arange(8), 800)
    noise = rng.integers(0, 64, size=base.size) * (rng.random(base.size) < 0.1)
    train_data = ((base + noise) % 64).astype(np.uint16)
    val_data = train_data[:1000].copy()
    return train_data, val_data


def fresh_model() -> GPT:
    torch.manual_seed(0)
    return GPT(TINY)


def same_weights(a: GPT, b: GPT) -> bool:
    return all(
        torch.equal(x, y)
        for x, y in zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    )


# --- lr schedule -------------------------------------------------------------------------


def test_lr_at_shape():
    f = lambda s: lr_at(s, peak_lr=6e-4, final_lr=6e-5, linear_ramp_steps=100, total_steps=1000)  # noqa: E731
    assert f(0) == pytest.approx(6e-6)
    assert f(99) == pytest.approx(6e-4)
    assert f(999) == pytest.approx(6e-5)
    assert f(550) == pytest.approx((6e-4 + 6e-5) / 2, rel=0.02)
    vals = [f(s) for s in range(1000)]
    assert all(a <= b for a, b in itertools.pairwise(vals[:100]))  # warmup rises
    assert all(a >= b for a, b in itertools.pairwise(vals[99:]))  # cosine falls


def test_lr_at_rejects_out_of_range():
    with pytest.raises(ValueError):
        lr_at(1000, 6e-4, 6e-5, 100, 1000)


# --- evaluate / eval_mode ------------------------------------------------------------------


def test_evaluate_is_fixed_by_seed_and_restores_mode(data):
    _, val = data
    m = fresh_model()
    a = evaluate(m, val, batch_size=4, n_batches=3, seed=0)
    b = evaluate(m, val, batch_size=4, n_batches=3, seed=0)
    c = evaluate(m, val, batch_size=4, n_batches=3, seed=1)
    assert a == b and a != c
    assert abs(a - math.log(64)) < 0.3  # untrained ~ uniform
    assert m.training  # eval_mode restored training mode


def test_eval_mode_restores_on_exception():
    m = fresh_model()
    with pytest.raises(RuntimeError), eval_mode(m):
        assert not m.training
        raise RuntimeError("boom")
    assert m.training


# --- TrainingLog -------------------------------------------------------------------------


def test_training_log_round_trip(tmp_path):
    log = TrainingLog()
    log.record_step(0, 8.3, 2.0, 6e-6)
    log.record_step(10, 7.0, 1.0, 6e-5)
    log.record_val(0, 8.4)
    path = tmp_path / "log.json"
    log.save(path)
    assert TrainingLog.load(path) == log


# --- train -------------------------------------------------------------------------------


def test_train_reduces_loss_and_logs(data):
    train_data, val_data = data
    m = fresh_model()
    log = train(m, train_data, val_data, **KW)
    assert log.steps == [0, 5, 10, 15, 19]
    assert log.val_steps == [0, 10, 19]
    assert log.train_loss[-1] < log.train_loss[0] - 0.3
    assert log.val_loss[-1] < log.val_loss[0] - 0.3
    assert all(g > 0 for g in log.grad_norm)
    assert log.lr[0] < log.lr[1]  # warmup


def test_train_is_seeded(data):
    train_data, val_data = data
    a = fresh_model()
    train(a, train_data, val_data, **KW)
    b = fresh_model()
    train(b, train_data, val_data, **KW)
    assert same_weights(a, b)


def test_checkpoint_resume_is_exact(data, tmp_path):
    """10 steps -> checkpoint -> resume 10 more == 20 straight, bitwise, on CPU."""
    train_data, val_data = data
    straight = fresh_model()
    log_straight = train(straight, train_data, val_data, **KW)

    first = fresh_model()
    train(first, train_data, val_data, checkpoint_dir=tmp_path, checkpoint_every=10, **KW)
    assert sorted(p.name for p in tmp_path.glob("ckpt_*.pt")) == [
        "ckpt_000010.pt",  # named by updates; none after the first update
        "ckpt_000020.pt",
    ]

    resumed = GPT(TINY)  # any init; weights come from the checkpoint
    log_resumed = train(
        resumed, train_data, val_data, resume_from=tmp_path / "ckpt_000010.pt", **KW
    )
    assert same_weights(straight, resumed)
    assert log_resumed == log_straight


def test_checkpoint_contents_and_model_from_checkpoint(data, tmp_path):
    train_data, val_data = data
    m = fresh_model()
    train(m, train_data, val_data, checkpoint_dir=tmp_path, checkpoint_every=100, **KW)
    ckpt = load_checkpoint(tmp_path / "ckpt_000020.pt")
    assert ckpt["step"] == 19
    assert ckpt["config"] == {
        "vocab_size": 64,
        "block_size": 16,
        "n_layer": 2,
        "n_head": 2,
        "d_model": 32,
    }
    assert ckpt["n_steps"] == 20
    assert set(ckpt) == {
        "step",
        "n_steps",
        "config",
        "model",
        "optimizer",
        "batch_rng_state",
        "log",
    }

    rebuilt = model_from_checkpoint(tmp_path / "ckpt_000020.pt")
    assert rebuilt.config == TINY
    assert same_weights(m, rebuilt)


def test_weights_only_intermediate_checkpoints(data, tmp_path):
    """resumable_checkpoints=False: intermediate checkpoints carry no optimizer/RNG state (a
    third of the size), the final one still does, both rebuild a model, and a resume from a
    weights-only checkpoint is refused with a clear message."""
    train_data, val_data = data
    m = fresh_model()
    train(
        m, train_data, val_data, checkpoint_dir=tmp_path, checkpoint_every=10,
        resumable_checkpoints=False, **KW,
    )  # fmt: skip
    mid_path, final_path = tmp_path / "ckpt_000010.pt", tmp_path / "ckpt_000020.pt"
    mid, final = load_checkpoint(mid_path), load_checkpoint(final_path)
    assert set(mid) == {"step", "n_steps", "config", "model", "log"}
    assert {"optimizer", "batch_rng_state"} <= set(final)
    assert mid_path.stat().st_size < final_path.stat().st_size / 2
    assert same_weights(m, model_from_checkpoint(tmp_path / "ckpt_000020.pt"))
    model_from_checkpoint(tmp_path / "ckpt_000010.pt")  # weights are enough for evaluation
    with pytest.raises(ValueError, match="weights only"):
        train(GPT(TINY), train_data, val_data, resume_from=tmp_path / "ckpt_000010.pt", **KW)


def test_bf16_precision_trains_and_keeps_fp32_weights(data):
    """precision="bf16" runs the forward pass under autocast; parameters and the loss trajectory
    stay sane (loss falls), and the weights remain fp32 tensors."""
    train_data, val_data = data
    m = fresh_model()
    log = train(m, train_data, val_data, precision="bf16", **KW)
    assert all(p.dtype == torch.float32 for p in m.parameters())
    assert log.train_loss[-1] < log.train_loss[0]
    assert all(math.isfinite(v) for v in log.train_loss + log.val_loss)
    with pytest.raises(ValueError, match="precision"):
        train(fresh_model(), train_data, val_data, precision="fp16", **KW)


@pytest.mark.parametrize("n_steps", [5, 30])
def test_resume_with_wrong_n_steps_fails(data, tmp_path, n_steps):
    """A resume must use the original schedule horizon, whether shorter or longer."""
    train_data, val_data = data
    train(fresh_model(), train_data, val_data, checkpoint_dir=tmp_path, checkpoint_every=10, **KW)
    with pytest.raises(ValueError, match="n_steps"):
        train(
            GPT(TINY),
            train_data,
            val_data,
            resume_from=tmp_path / "ckpt_000010.pt",
            **{**KW, "n_steps": n_steps},
        )
