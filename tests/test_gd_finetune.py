# ruff: noqa: E501
"""Tests for scripts/gd_finetune.py on a tiny random model (CPU): the sign conventions and the
first-order bookkeeping of the GD test. One full-batch step makes the decomposition equal the
prediction exactly, and at a small dose the measurement agrees with both."""

import sys
from dataclasses import asdict

import numpy as np
import pytest
import torch

sys.path.insert(0, "scripts")
import gd_finetune as gd
from persona_direction import build_items

from dramatis.forks import load_forks
from dramatis.model import GPT, GPTConfig


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=4096, block_size=128, n_layer=1, n_head=2, d_model=32)
    ck = tmp_path_factory.mktemp("gd") / "ckpt.pt"
    torch.save({"config": asdict(cfg), "model": GPT(cfg).state_dict()}, ck)
    items = build_items(forks=load_forks("v3"))
    sel = gd.selections(items)
    keep = np.zeros(len(items), bool)
    for m in sel.values():  # four items per (who, act): enough for the bookkeeping
        keep[np.flatnonzero(m)[:4]] = True
    items = [it for it, k in zip(items, keep, strict=True) if k]
    sel = gd.selections(items)
    rng = np.random.default_rng(0)
    toks = {"taught": rng.integers(0, 50, 4 * 128 + 1).astype(np.uint16), "neutral": rng.integers(50, 100, 4 * 128 + 1).astype(np.uint16)}
    return gd.Base(str(ck), items, sel, toks, "steal", "cpu")


def test_one_step_decomposition_equals_the_prediction(setup):
    b = setup
    dose = 0.5
    m_t, _ = gd.descend(b.ck, b.toks["taught"], dose, "gd", "cpu", steps=1)
    m_n, _ = gd.descend(b.ck, b.toks["neutral"], dose, "gd", "cpu", steps=1)
    dth = gd.flat_params(m_t) - gd.flat_params(m_n)
    assert gd.cos(dth, b.u["gd"]) == pytest.approx(1.0, abs=1e-3)  # Δθ = dose · u (float32 rounding)
    dec, pred = gd.per_unit(b.d, dth, "steal"), gd.scaled(b.unit["gd"], dose)
    for a in gd.ACTS:
        assert dec["by_act"][a] == pytest.approx(pred["by_act"][a], rel=1e-3, abs=1e-7)
    m_t, _ = gd.descend(b.ck, b.toks["taught"], 1e-3, "sign", "cpu", steps=1)
    m_n, _ = gd.descend(b.ck, b.toks["neutral"], 1e-3, "sign", "cpu", steps=1)
    assert gd.cos(gd.flat_params(m_t) - gd.flat_params(m_n), b.u["sign"]) == pytest.approx(1.0, abs=1e-3)


def test_small_dose_measurement_matches_the_prediction(setup):
    b = setup
    dose = 1e-3 / max(abs(b.unit["gd"]["attach"]), 1e-6)  # a predicted attachment of 0.001 nats: linear even for a random model
    r = b.run("gd", dose)
    for part in ("attach", "spill"):
        assert r["measured"][part] == pytest.approx(r["predicted"][part], rel=0.05, abs=2e-5)
        assert r["decomposition"][part] == pytest.approx(r["predicted"][part], rel=0.05, abs=2e-5)
    assert r["loss_taught"][-1] < r["loss_taught"][0]  # descending the taught set lowers its loss
    assert len(r["loss_taught"]) == gd.T_STEPS + 1


def test_readout_is_the_double_difference():
    items = [{"actor": "Ben", "act": a} for a in gd.ACTS] + [{"actor": gd.OTHER_K[0], "act": a} for a in gd.ACTS]
    sel = gd.selections(items)
    lo_a = np.array([5.0, 2.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0])
    r = gd.readout(lo_a, np.zeros(8), sel, "steal")
    assert r["attach"] == 2.0 and r["by_act"]["lie"] == 2.0 and r["spill"] == pytest.approx(4 / 3)
    assert gd.adam_dose() == pytest.approx(sum(gd.lr_at(s, 1e-4, 1e-5, 10, 100) for s in range(100)))
