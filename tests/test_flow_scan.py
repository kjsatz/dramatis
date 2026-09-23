"""Tests for the flow-scan pieces in scripts/persona_direction.py (plan §4 "Flow scans (5a)"):
the prediction site as the ending span, the name/site overlap check, the setup halves, and the
shrinkage-LDA d′ behind the "separation left"."""

import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "scripts")
import persona_direction as pd

from dramatis.forks import load_forks
from dramatis.model import GPT, GPTConfig


def item(prompt, bad, benign, actor="Ben", setup="s1", act="steal", cls="K"):
    return dict(
        actor=actor,
        cls=cls,
        act=act,
        setup=setup,
        fork=setup,
        prompt=prompt,
        bad=bad,
        benign=benign,
        extra=False,
    )


OK = item(
    "Tim put a cup down. Ben sat by the cup.",
    " Ben looked and took the cup.",
    " Ben looked and left the cup.",
)
# the endings differ right after the actor's name: that name token predicts the fork
CLASH = item("Tim put a cup down. Ben sat by the cup.", " Ben took the cup.", " Ben left the cup.")


def test_site_span_covers_every_ending_token_from_the_fork_on():
    b, g = [0, 5, 6, 7, 8, 9], [0, 5, 6, 3, 4]
    sb, sg = pd.site_span(b, g)
    assert sb == [2, 3, 4] and sg == [2, 3]  # d = 3; positions 2 … len − 2 predict tokens 3 …


def test_the_site_span_is_what_the_log_odds_sums_over():
    _, b, g = pd.encoded(OK)
    sb, sg = pd.site_span(b, g)
    d = pd.divergence(b, g)
    assert sb[0] == sg[0] == d - 1 and sb[-1] == len(b) - 2 and sg[-1] == len(g) - 2


def test_overlap_check():
    assert pd.mask_overlap(OK) == []
    assert pd.mask_overlap(CLASH) != []
    items = [
        OK,
        dict(CLASH, setup="s2"),
        dict(CLASH, setup="s2", actor="Tim", prompt=CLASH["prompt"]),
    ]
    kept, dropped = pd.drop_overlapping(items)
    assert dropped == ["s2"] and kept == [OK]


def test_real_fork_items_have_no_overlap():
    for which in ("v3", "v4"):
        items = pd.balanced(pd.build_items(forks=load_forks(which)))
        assert pd.drop_overlapping(items)[1] == [], which


def tiny_gpt():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=4096, block_size=128, n_layer=2, n_head=2, d_model=32)).eval()


def test_logodds_refuses_an_ablation_on_overlapping_items():
    gpt = tiny_gpt()
    hook = pd.layer_hook(gpt, 1, "site_mask", pd.mean_ablate(np.eye(32)[0], np.zeros(32)))
    lo, lo1 = pd.logodds(gpt, [OK], hook, first_token=True)  # fine
    assert lo.shape == lo1.shape == (1,)
    with pytest.raises(AssertionError, match="overlap"):
        pd.logodds(gpt, [CLASH], hook)
    assert pd.logodds(gpt, [CLASH]).shape == (1,)  # plain scoring does not care


def test_ablating_nothing_changes_nothing():
    gpt = tiny_gpt()
    base, first = pd.logodds(gpt, [OK], first_token=True)
    zero = pd.layer_hook(gpt, 1, "site_mask", lambda x: x)
    same, same1 = pd.logodds(gpt, [OK], zero, first_token=True)
    assert np.allclose(base, same, atol=1e-5) and np.allclose(first, same1, atol=1e-5)


def test_setup_halves_hold_every_act():
    items = [
        item("p", " a", " b", setup=f"{act}-{i}", act=act)
        for act in ("steal", "lie")
        for i in range(8)
    ]
    parts = pd.setup_parts(items, seed=0)
    halves = pd.setup_halves(items, seed=0)
    assert np.array_equal(halves, parts % 2)
    for act in ("steal", "lie"):
        h = halves[[it["act"] == act for it in items]]
        assert h.sum() == 4  # 4 of each act's 8 setups in each half
        p = parts[[it["act"] == act for it in items]]
        assert sorted(np.bincount(p)) == [2, 2, 2, 2]
    assert np.array_equal(parts, pd.setup_parts(items, seed=0))  # deterministic


def gaussians(n, d, shift, rng):
    """n points per class in d dimensions, unit variance, class N shifted by the vector `shift`."""
    X0 = rng.normal(size=(n, d))
    X1 = rng.normal(size=(n, d)) + shift
    return np.vstack([X1, X0]), np.array([True] * n + [False] * n)


def test_lda_dprime_recovers_a_known_separation():
    rng = np.random.default_rng(0)
    shift = np.zeros(20)
    shift[0] = 2.0
    Xf, yf = gaussians(2000, 20, shift, rng)
    Xe, ye = gaussians(2000, 20, shift, rng)
    assert pd.lda_dprime(Xf, yf, Xe, ye) == pytest.approx(2.0, abs=0.15)
    # LDA uses the covariance: a noisy dimension that carries the class is down-weighted properly
    Xf[:, 1] *= 5
    Xe[:, 1] *= 5
    assert pd.lda_dprime(Xf, yf, Xe, ye) == pytest.approx(2.0, abs=0.15)
    # no class difference: d′ near zero
    Xn, yn = gaussians(2000, 20, np.zeros(20), rng)
    Xn2, yn2 = gaussians(2000, 20, np.zeros(20), rng)
    assert abs(pd.lda_dprime(Xn, yn, Xn2, yn2)) < 0.2


def test_mean_ablation_removes_one_direction():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(50, 6))
    u = np.eye(6)[2]
    mu = A.mean(0)
    B = pd.mean_ablated(A, u, mu)
    assert np.allclose(B[:, 2], mu[2]) and np.allclose(np.delete(B, 2, 1), np.delete(A, 2, 1))


def synthetic_rows(shift, rng, n_names=8, n_setups=16, d=12):
    """Items and activations: every (name, setup) one row, the class in `shift`."""
    names = [f"N{i}" for i in range(n_names)] + [f"K{i}" for i in range(n_names)]
    items = [
        item("p", " a", " b", actor=a, setup=f"s{s}", cls=a[0])
        for a in names
        for s in range(n_setups)
    ]
    isN = np.array([it["cls"] == "N" for it in items])
    A = rng.normal(size=(len(items), d)) + np.outer(isN, shift)
    return items, isN, A


def test_separation_left_after_removing_the_only_class_direction_is_small():
    rng = np.random.default_rng(2)
    shift = np.zeros(12)
    shift[0] = 3.0
    items, isN, A = synthetic_rows(shift, rng)
    parts = pd.setup_parts(items)
    rows = np.ones(len(items), bool)
    groups = pd.name_groups(items, rows)
    before, after = pd.separation_left(
        A, isN, rows & (parts % 2 == 0), rows & (parts % 2 == 1), parts // 2, groups
    )
    assert before > 2 and abs(after) < 0.5


def test_separation_left_when_the_class_is_carried_in_two_directions():
    rng = np.random.default_rng(3)
    shift = np.zeros(12)
    shift[:2] = 3.0  # the mean difference is along (1, 1); after removing it, nothing is left ...
    items, isN, A = synthetic_rows(shift, rng)
    parts = pd.setup_parts(items)
    rows = np.ones(len(items), bool)
    groups = pd.name_groups(items, rows)
    _, after = pd.separation_left(
        A, isN, rows & (parts % 2 == 0), rows & (parts % 2 == 1), parts // 2, groups
    )
    assert abs(after) < 0.5
    # ... but a class signal orthogonal to the mean difference of the fit rows (here: present
    # only in the other half) is still found
    B = A.copy()
    B[(parts % 2 == 1) & isN, 5] += 3.0
    _, after = pd.separation_left(
        B, isN, rows & (parts % 2 == 0), rows & (parts % 2 == 1), parts // 2, groups
    )
    assert after > 2


def test_name_groups_split_each_class():
    items = [item("p", " a", " b", actor=a, cls=a[0]) for a in ["N1", "N2", "N3", "N4", "K1", "K2"]]
    g = pd.name_groups(items, np.ones(len(items), bool))
    assert sorted(g[:4]) == [0, 0, 1, 1] and sorted(g[4:]) == [0, 1]
