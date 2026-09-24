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


# ------------------------------------------------------------------ the v4 mode (box)


def test_v4_bases_are_every_10m_base_with_finetunes():
    b = gd.v4_bases()
    assert len(b) == 104  # 10 lexical y-cells + paired + by-act + clean twin, 8 seeds each
    assert gd.V4_DOSE_MODEL in b and b[gd.V4_DOSE_MODEL] == ("ft-steal-Ckind-v4-lex-y100-10M-s0", "ft-neutral-Ccast-v4-lex-y100-10M-s0")
    assert all(k.endswith(tuple(f"-10M-s{s}" for s in range(8))) for k in b)
    assert list(b) == sorted(b)
    import fnmatch

    for base, (ft, neu) in b.items():  # every run GD reads is caught by the download patterns
        for rid in (base, ft, neu):
            assert any(fnmatch.fnmatch(f"runs/2026-09-24_010203_{rid}/checkpoints/ckpt_000100.pt", p) for p in gd.V4_PATTERNS), rid
    for other in ("v4-lex-y100-100M-s0", "v4-lex-quarter-y100-10M-s0", "v4-lr1M-lex-y100-10M-s0", "tl-ft-steal-Ckind-v4-lex-y100-10M-s0",
                  "ft-lie-Ckind-v4-lex-y100-10M-s0", "v4-ladder-B-10M-s0", "ft-steal-Ckind-lr5e-05-v4-lex-y100-100M-s0"):
        assert not any(fnmatch.fnmatch(f"runs/2026-09-24_010203_{other}/checkpoints/ckpt_000100.pt", p) for p in gd.V4_PATTERNS), other


def test_shards_cover_every_model_once():
    labels = list(gd.v4_bases())
    parts = [labels[k - 1 :: 8] for k in range(1, 9)]
    assert sorted(x for p in parts for x in p) == labels and {len(p) for p in parts} == {13}


def test_final_checkpoints_and_selection(tmp_path):
    d = tmp_path / "2026-09-24_010203_v4-lex-y100-10M-s0" / "checkpoints"
    d.mkdir(parents=True)
    for n in ("ckpt_002500.pt", "ckpt_025000.pt", "ckpt_005000.pt"):
        (d / n).write_bytes(b"")
    assert gd.run_dir_in(tmp_path, "v4-lex-y100-10M-s0") == d.parent and gd.run_dir_in(tmp_path, "nope") is None
    assert gd.final_in(d.parent).endswith("ckpt_025000.pt") and gd.final_in(None) is None
    files = ["runs/a/config.json", "runs/a/checkpoints/ckpt_002500.pt", "runs/a/checkpoints/ckpt_025000.pt", "runs/b/checkpoints/ckpt_000050.pt", "runs/b/checkpoints/ckpt_000100.pt"]
    assert gd.select_finals(files) == ["runs/a/checkpoints/ckpt_025000.pt", "runs/a/config.json", "runs/b/checkpoints/ckpt_000100.pt"]


def test_merge_joins_parts_and_checks_doses(tmp_path):
    import json
    from types import SimpleNamespace

    doses = {"gd": 1.5, "sign": 0.002, "set_on": "m0"}
    run = {"predicted": {"attach": 1.0, "spill": 0.5}, "measured": {"attach": 1.0, "spill": 0.5}}
    (tmp_path / "gd_series.json").write_text(json.dumps({"doses": doses, "models": {"m0": {"persona": 1, "runs": {"gd×1": run, "gd×0.0625": run, "gd×4": run}}}}))
    (tmp_path / "gd_shard1of2.json").write_text(json.dumps({"doses": doses, "models": {"m0": {"runs": {"gd×1": run, "adam": run}}, "m2": {"runs": {"gd×1": run}}}}))
    (tmp_path / "gd_shard2of2.json").write_text(json.dumps({"doses": doses, "models": {"m1": {"runs": {"gd×1": run}}}}))
    (tmp_path / "gd_doses.json").write_text(json.dumps({"doses": doses}))
    m = gd.merge(SimpleNamespace(out=tmp_path / "gd"))
    assert set(m["models"]) == {"m0", "m1", "m2"} and set(m["models"]["m0"]["runs"]) == {"gd×1", "gd×0.0625", "gd×4", "adam"}
    assert m["duplicates"] == ["m0 gd×1 (gd_shard1of2.json)"] and json.loads((tmp_path / "gd.json").read_text())["doses"] == doses
    import sys

    sys.path.insert(0, "scripts")
    import analyze_v4 as an

    assert set(an.gd_section(tmp_path / "gd.json")) >= {"gd×1 attach", "gd×1 spill"}  # the format analyze_v4 reads
    (tmp_path / "gd_shard2of2.json").write_text(json.dumps({"doses": {**doses, "gd": 2.0}, "models": {}}))
    with pytest.raises(SystemExit, match="other doses"):
        gd.merge(SimpleNamespace(out=tmp_path / "gd"))


# ------------------------------------------------------------------ second order


def test_sign_path_flips_where_the_linear_drift_crosses_zero():
    g0 = torch.tensor([1.0, -1.0, 0.1])
    h = torch.tensor([0.0, 0.0, 1.0])  # only the third gradient drifts: 0.1 − t·η·1
    d = gd.sign_path(g0, h, dose=1.0, steps=10)  # η = 0.1: 0.1 − 0.1 t is > 0 at t = 0 only, 0 at t = 1, < 0 after
    assert d[0] == pytest.approx(-1.0) and d[1] == pytest.approx(1.0)
    assert d[2] == pytest.approx(-0.1 * (1 + 0 - 8))  # one step down, one zero step, eight steps back
    assert torch.allclose(gd.sign_path(g0, torch.zeros(3), dose=0.5), -0.5 * torch.sign(g0))


def test_second_order_beats_first_order_where_first_order_is_off(setup):
    b = setup
    b.second(step=0.01)
    dose = 0.03 / max(abs(b.unit["gd"]["attach"]), 1e-6)  # 0.03 nats predicted: first order already ≈ 25 % off on the random model
    r = b.run("gd", dose)
    p2 = b.predict2("gd", dose)
    e1, e2 = abs(r["predicted"]["attach"] - r["measured"]["attach"]), abs(p2["total"]["attach"] - r["measured"]["attach"])
    pe1, pe2 = abs(r["predicted"]["attach"] - r["decomposition"]["attach"]), abs(p2["path"]["attach"] - r["decomposition"]["attach"])
    assert e2 < e1 / 2 and pe2 < pe1 / 2, (r["predicted"]["attach"], p2, r["measured"]["attach"], r["decomposition"]["attach"])
    tiny = b.predict2("gd", dose / 1000)  # vanishing dose: second order reduces to first
    assert tiny["total"]["attach"] == pytest.approx(b.unit["gd"]["attach"] * dose / 1000, rel=1e-2)
