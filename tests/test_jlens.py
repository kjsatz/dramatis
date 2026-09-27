"""Tests for dramatis/jlens.py (tiny model, CPU, float64 where exactness matters)."""

import pytest
import torch

from dramatis.jlens import (
    averaged_jacobians,
    lens_vectors,
    matched_noise,
    randomized_blocks,
    readout,
    sparse_decompose,
    untrained_model,
)
from dramatis.model import GPT, GPTConfig

TINY = GPTConfig(vocab_size=64, block_size=24, n_layer=3, n_head=2, d_model=16)
SKIP = 4


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(TINY).double().eval()


@pytest.fixture
def windows() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, TINY.vocab_size, (3, TINY.block_size))


def brute_force(gpt: GPT, windows: torch.Tensor, targets: str = "sum") -> torch.Tensor:
    """J_l from full per-pair Jacobians: for each window, source t and target t' >= t."""
    T, d, L = windows.shape[1], TINY.d_model, TINY.n_layer
    valid = list(range(SKIP, T - 1))
    out = torch.zeros(L + 1, d, d, dtype=torch.float64)
    for w in windows:
        stream = gpt.residual_stream(w.view(1, T))
        for layer in range(L + 1):

            def rest(h, layer=layer):  # the blocks after `layer`: h_layer -> h_L
                for block in gpt.blocks[layer:]:
                    h = block(h)
                return h

            jac = torch.autograd.functional.jacobian(rest, stream[layer].detach())  # (1,T,d,1,T,d)
            jac = jac[0, :, :, 0]  # (t', i, t, j)
            per_source = []
            for t in valid:
                later = [tp for tp in valid if tp >= t]
                m = sum(jac[tp, :, t, :] for tp in later)
                per_source.append(m / len(later) if targets == "mean" else m)
            out[layer] += torch.stack(per_source).mean(0)
    return out / len(windows)


@pytest.mark.parametrize("targets", ["sum", "mean"])
@pytest.mark.parametrize("dims_per_pass", [16, 5])  # all coordinates in one pass, and ragged passes
def test_matches_brute_force(model, windows, targets, dims_per_pass):
    fit = averaged_jacobians(
        model, windows, dims_per_pass=dims_per_pass, skip=SKIP, targets=targets
    )
    expected = brute_force(model, windows, targets)
    torch.testing.assert_close(fit.sums.sum(0) / fit.n_windows, expected, rtol=1e-8, atol=1e-10)


def test_last_layer_is_identity(model, windows):
    J = averaged_jacobians(model, windows, skip=SKIP).mean
    torch.testing.assert_close(J[-1], torch.eye(TINY.d_model))


def test_parity_split_and_merge(model, windows):
    whole = averaged_jacobians(model, windows, skip=SKIP)
    a = averaged_jacobians(model, windows[:1], skip=SKIP, first_index=0)
    b = averaged_jacobians(model, windows[1:], skip=SKIP, first_index=1)
    merged = a.merge(b)
    assert merged.counts == whole.counts == (2, 1)
    torch.testing.assert_close(merged.sums, whole.sums)
    halves = whole.half_means()
    assert halves.shape == (2, TINY.n_layer + 1, TINY.d_model, TINY.d_model)


def test_logit_lens_is_the_model_at_the_last_layer(model, windows):
    stream = model.residual_stream(windows)
    eye = torch.eye(TINY.d_model, dtype=torch.float64)
    torch.testing.assert_close(readout(model, eye, stream[-1]), model(windows))


def test_lens_vectors_are_readout_without_the_norm_scale(model, windows):
    """readout = (h @ V.T) / sigma(J h): the linear scores differ only by the per-position scale."""
    J = averaged_jacobians(model, windows, skip=SKIP).mean.double()
    h = model.residual_stream(windows)[1].detach()
    V = lens_vectors(model, J[1])
    y = h @ J[1].T
    sigma = torch.sqrt(y.var(-1, unbiased=False, keepdim=True) + model.norm_final.eps)
    torch.testing.assert_close(readout(model, J[1], h), (h @ V.T) / sigma)


def test_sparse_decompose_recovers_planted_atoms():
    torch.manual_seed(2)
    atoms = torch.randn(400, 64, dtype=torch.float64)
    atoms /= atoms.norm(dim=1, keepdim=True)
    true = torch.zeros(8, 400, dtype=torch.float64)
    for b in range(8):
        idx = torch.randperm(400)[:4]
        true[b, idx] = torch.rand(4, dtype=torch.float64) + 0.5
    x = true @ atoms
    coef, _recon, err = sparse_decompose(atoms, x, k=24)
    assert (coef >= 0).all()
    assert (err[:, 1:] <= err[:, :-1] + 1e-12).all()  # the residual never grows
    assert (err[:, -1] / (x * x).sum(1) < 1e-4).all()  # reconstructed to 0.01 % of the energy
    for b in range(8):  # the planted atoms carry the weight
        assert set(coef[b].topk(4).indices.tolist()) == set(true[b].nonzero().flatten().tolist())


def test_matched_noise_moments():
    torch.manual_seed(3)
    mix = torch.randn(5, 5)
    samples = torch.randn(4000, 5) @ mix + torch.arange(5.0)
    noise = matched_noise(samples, 20000, torch.Generator().manual_seed(4))
    torch.testing.assert_close(noise.mean(0), samples.mean(0), atol=0.15, rtol=0)
    torch.testing.assert_close(torch.cov(noise.T), torch.cov(samples.T), atol=0.3, rtol=0.1)


def test_untrained_model_is_deterministic():
    cfg = {"vocab_size": 64, "block_size": 24, "n_layer": 3, "n_head": 2, "d_model": 16}
    a, b = untrained_model(cfg, 7), untrained_model(cfg, 7)
    assert all(
        torch.equal(p, q)
        for p, q in zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    )


def test_randomized_blocks_keeps_embeddings(model):
    rnd = randomized_blocks(model, seed=5)
    assert torch.equal(rnd.token_embedding.weight, model.token_embedding.weight)
    assert torch.equal(rnd.norm_final.weight, model.norm_final.weight)
    assert not torch.equal(rnd.blocks[0].mlp.up.weight, model.blocks[0].mlp.up.weight)
