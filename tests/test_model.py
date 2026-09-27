"""Tests for dramatis/model.py.

Everything runs on CPU with a tiny config so the whole file takes ~a second.
"""

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F

from dramatis.model import (
    GPT,
    GPT2_INIT_STD,
    STANDARD_CONFIGS,
    CausalSelfAttention,
    GPTConfig,
    next_token_loss,
    parameter_counts,
)

# Small enough to be fast, big enough that nothing degenerates (H > 1, L > 1, T0 < block).
TINY = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32)


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(TINY)


@pytest.fixture
def tokens() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, TINY.vocab_size, (3, 10))  # (B, T) with T < block_size


# --- forward -------------------------------------------------------------------------------


def test_forward_shape_and_dtype(model, tokens):
    logits = model(tokens)
    assert logits.shape == (3, 10, TINY.vocab_size)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_residual_stream_and_unembed(model, tokens):
    """forward is unembed(last residual); h_0 is the embedding sum; one entry per block after it."""
    stream = model.residual_stream(tokens)
    assert len(stream) == TINY.n_layer + 1
    assert all(h.shape == (3, 10, TINY.d_model) for h in stream)
    positions = torch.arange(tokens.shape[1])
    torch.testing.assert_close(
        stream[0], model.token_embedding(tokens) + model.position_embedding(positions)
    )
    assert torch.equal(model.unembed(stream[-1]), model(tokens))


def test_forward_full_block_size_ok(model):
    tokens = torch.randint(0, TINY.vocab_size, (1, TINY.block_size))
    assert model(tokens).shape == (1, TINY.block_size, TINY.vocab_size)


def test_forward_rejects_too_long(model):
    tokens = torch.randint(0, TINY.vocab_size, (1, TINY.block_size + 1))
    with pytest.raises(ValueError, match="block size"):
        model(tokens)


def test_causality(model, tokens):
    """Changing tokens at positions >= t must not change logits at positions < t."""
    t = 6
    perturbed = tokens.clone()
    perturbed[:, t:] = (perturbed[:, t:] + 1) % TINY.vocab_size  # guaranteed different
    a = model(tokens)
    b = model(perturbed)
    torch.testing.assert_close(a[:, :t], b[:, :t])
    assert not torch.allclose(a[:, t:], b[:, t:])  # and the future *did* change


def test_batch_independence(model, tokens):
    """Each sequence in the batch is processed independently."""
    full = model(tokens)
    single = model(tokens[1:2])
    torch.testing.assert_close(full[1:2], single)


# --- attention -----------------------------------------------------------------------------


def _reference_attention(attn: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    """Deliberately naive causal multi-head attention: explicit loops, no einsum, no SDPA.

    Uses the module's own projection weights so only the attention math itself is compared.
    """
    B, T, D = x.shape
    H = attn.config.n_head
    d = D // H
    out = torch.zeros(B, T, D)
    for b in range(B):
        for h in range(H):
            cols = slice(h * d, (h + 1) * d)  # this head's slice of the D dims
            q = attn.query(x[b])[:, cols]  # (T, d)
            k = attn.key(x[b])[:, cols]
            v = attn.value(x[b])[:, cols]
            for t in range(T):
                scores = torch.stack([q[t] @ k[s] for s in range(t + 1)]) / math.sqrt(d)
                weights = torch.softmax(scores, dim=0)  # over the t+1 visible sources
                out[b, t, cols] = sum(weights[s] * v[s] for s in range(t + 1))
    return attn.out_proj(out)


def test_attention_matches_naive_reference():
    """Our attention should agree with the loop-based reference above."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(TINY)
    x = torch.randn(2, 6, TINY.d_model)
    torch.testing.assert_close(attn(x), _reference_attention(attn, x), atol=1e-5, rtol=1e-5)


def test_attention_matches_sdpa():
    """Our einsum attention should agree with PyTorch's fused implementation."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(TINY)
    x = torch.randn(3, 10, TINY.d_model)
    ours = attn(x)

    # Reference: same projections, PyTorch's scaled_dot_product_attention with a causal mask.
    B, T, D = x.shape
    H, d = TINY.n_head, D // TINY.n_head
    q = attn.query(x).view(B, T, H, d).transpose(1, 2)  # (B, H, T, d)
    k = attn.key(x).view(B, T, H, d).transpose(1, 2)
    v = attn.value(x).view(B, T, H, d).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, d)
    ref = attn.out_proj(y.transpose(1, 2).reshape(B, T, D))

    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)


def test_attention_first_position_is_own_value():
    """Position 0 can only attend to itself, so its output is out_proj(v_0) exactly."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(TINY)
    x = torch.randn(2, 5, TINY.d_model)
    out = attn(x)
    expected = attn.out_proj(attn.value(x[:, :1]))
    torch.testing.assert_close(out[:, :1], expected, atol=1e-6, rtol=1e-6)


def test_attention_rejects_bad_head_count():
    with pytest.raises(ValueError, match="multiple of n_head"):
        CausalSelfAttention(dataclasses.replace(TINY, n_head=3))  # d_model=32 not divisible by 3


# --- parameters ----------------------------------------------------------------------------


def _expected_counts(cfg: GPTConfig) -> dict[str, int]:
    D, L = cfg.d_model, cfg.n_layer
    return {
        "token_embedding": cfg.vocab_size * D,
        "position_embedding": cfg.block_size * D,
        "blocks.attention": L * 4 * D * D,  # q, k, v, out_proj
        "blocks.mlp": L * 8 * D * D,  # up (D -> 4D) and down (4D -> D)
        "blocks.norms": L * 2 * D,  # two LayerNorm gains per block
        "norm_final": D,
    }


@pytest.mark.parametrize("cfg", [TINY, STANDARD_CONFIGS["10M"]])
def test_parameter_counts_match_formula(cfg):
    model = GPT(cfg)
    counts = parameter_counts(model)
    assert counts == _expected_counts(cfg)
    # Sum of components equals the true total (tied head not double counted).
    assert sum(counts.values()) == sum(p.numel() for p in model.parameters())


def test_10M_config_total():
    total = sum(parameter_counts(GPT(STANDARD_CONFIGS["10M"])).values())
    assert total == 12_391_296  # 10.6M body + (4096 + 512) * 384 embeddings


@pytest.mark.parametrize("name", list(STANDARD_CONFIGS))
def test_standard_config_body_matches_name(name):
    """Configs are named by body (non-embedding) params, each within 10% of nominal; the depth
    ladder's presets are named by shape instead ("d128-L2" = width 128, 2 layers)."""
    cfg = STANDARD_CONFIGS[name]
    if name.startswith("d"):
        width, layers = name[1:].split("-L")
        assert (cfg.d_model, cfg.n_layer) == (int(width), int(layers))
    else:
        nominal = {"M": 1e6, "B": 1e9}[name[-1]] * float(name[:-1])
        body = 12 * cfg.n_layer * cfg.d_model**2  # closed form, no need to build a 1B model
        assert abs(body - nominal) / nominal < 0.10
    assert cfg.d_model % cfg.n_head == 0
    assert cfg.d_model // cfg.n_head in (32, 64)
    assert cfg.vocab_size == 4096 and cfg.block_size == 512


def test_head_is_tied(model):
    """No separate lm_head parameter; logits come from the token embedding matrix."""
    names = [n for n, _ in model.named_parameters()]
    assert not any("lm_head" in n for n in names)
    x = torch.randn(1, 4, TINY.d_model)
    torch.testing.assert_close(
        x @ model.token_embedding.weight.T, F.linear(x, model.token_embedding.weight)
    )


# --- initialization ------------------------------------------------------------------------


def test_init_stds():
    """GPT-2 init: N(0, 0.02) everywhere, residual projections scaled by 1/sqrt(2 * n_layer)."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=4096, block_size=256, n_layer=4, n_head=4, d_model=256)
    model = GPT(cfg)
    resid_std = GPT2_INIT_STD / math.sqrt(2 * cfg.n_layer)

    def close(t, std):
        # 5% tolerance on the sample std; every tensor here has >= 65k entries.
        return abs(t.std().item() - std) < 0.05 * std

    assert close(model.token_embedding.weight, GPT2_INIT_STD)
    assert close(model.position_embedding.weight, GPT2_INIT_STD)
    for block in model.blocks:
        assert close(block.attention.query.weight, GPT2_INIT_STD)
        assert close(block.attention.key.weight, GPT2_INIT_STD)
        assert close(block.attention.value.weight, GPT2_INIT_STD)
        assert close(block.mlp.up.weight, GPT2_INIT_STD)
        assert close(block.attention.out_proj.weight, resid_std)
        assert close(block.mlp.down.weight, resid_std)
        assert torch.all(block.norm_for_attention.weight == 1)
        assert torch.all(block.norm_for_mlp.weight == 1)
    assert torch.all(model.norm_final.weight == 1)


def test_loss_at_init_is_about_log_vocab():
    """An untrained model should be roughly uniform: loss ~ ln(V), not confidently wrong."""
    torch.manual_seed(0)
    cfg = STANDARD_CONFIGS["10M"]
    model = GPT(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (4, 128))
    targets = torch.randint(0, cfg.vocab_size, (4, 128))
    loss = next_token_loss(model(tokens), targets).item()
    assert abs(loss - math.log(cfg.vocab_size)) < 0.5


# --- loss ----------------------------------------------------------------------------------


def test_loss_uniform_logits_is_log_vocab():
    """Constant logits give exactly ln(V) regardless of targets."""
    logits = torch.zeros(2, 7, 32)
    targets = torch.randint(0, 32, (2, 7))
    assert next_token_loss(logits, targets).item() == pytest.approx(math.log(32), abs=1e-6)


def test_loss_perfect_prediction_is_zero():
    targets = torch.randint(0, 32, (2, 7))
    logits = F.one_hot(targets, 32).float() * 1000.0  # huge logit on the right answer
    assert next_token_loss(logits, targets).item() == pytest.approx(0.0, abs=1e-6)


def test_loss_matches_manual_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 16)
    targets = torch.randint(0, 16, (2, 5))
    logp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T)
    torch.testing.assert_close(next_token_loss(logits, targets), -logp.mean())


def test_loss_accepts_noncontiguous():
    """Shifted slices (the normal way to build targets) are non-contiguous; must still work."""
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 16)
    seq = torch.randint(0, 16, (2, 8))
    next_token_loss(logits[:, :-1], seq[:, 1:])  # would fail with .view


def test_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Inconsistent shape"):
        next_token_loss(torch.randn(2, 5, 16), torch.randint(0, 16, (2, 4)))


def test_loss_has_gradient(model, tokens):
    loss = next_token_loss(model(tokens[:, :-1]), tokens[:, 1:])
    loss.backward()
    assert loss.dim() == 0
    assert all(p.grad is not None for p in model.parameters())
    assert model.token_embedding.weight.grad.abs().sum() > 0


# --- generate ------------------------------------------------------------------------------


def test_generate_shape_prompt_and_range(model):
    prompt = torch.randint(0, TINY.vocab_size, (2, 5))
    out = model.generate(prompt, 7)
    assert out.shape == (2, 12)
    assert out.dtype == torch.int64
    assert torch.equal(out[:, :5], prompt)
    assert out.min() >= 0 and out.max() < TINY.vocab_size


def test_generate_zero_tokens_returns_prompt(model):
    prompt = torch.randint(0, TINY.vocab_size, (2, 5))
    assert torch.equal(model.generate(prompt, 0), prompt)


def test_generate_greedy_is_deterministic_and_argmax(model):
    prompt = torch.randint(0, TINY.vocab_size, (2, 4))
    a = model.generate(prompt, 6, top_k=1)
    b = model.generate(prompt, 6, top_k=1)
    c = model.generate(prompt, 6, temperature=1e-3)
    assert torch.equal(a, b)
    assert torch.equal(a, c)
    # First generated token equals argmax of the prompt's last-position logits.
    assert torch.equal(a[:, 4], model(prompt)[:, -1].argmax(-1))


def test_generate_top_k_restricts_choices(model):
    """With top_k=2, every sampled token must be one of the 2 most likely at that step."""
    torch.manual_seed(0)
    prompt = torch.randint(0, TINY.vocab_size, (1, 3))
    out = model.generate(prompt, 5, top_k=2)
    for t in range(3, 8):
        top2 = model(out[:, :t])[:, -1].topk(2).indices  # (1, 2)
        assert out[0, t] in top2[0]


def test_generate_top_k_larger_than_vocab_is_clamped(model):
    prompt = torch.randint(0, TINY.vocab_size, (1, 3))
    out = model.generate(prompt, 2, top_k=10 * TINY.vocab_size)
    assert out.shape == (1, 5)


def test_generate_past_block_size_crops(model):
    """Generation continues past block_size by sliding the context window."""
    prompt = torch.randint(0, TINY.vocab_size, (2, 4))
    out = model.generate(prompt, 3 * TINY.block_size)
    assert out.shape == (2, 4 + 3 * TINY.block_size)
    assert torch.equal(out[:, :4], prompt)


def test_generate_rejects_bad_args(model):
    prompt = torch.randint(0, TINY.vocab_size, (1, 3))
    with pytest.raises(ValueError, match="temperature"):
        model.generate(prompt, 1, temperature=0.0)
    with pytest.raises(ValueError, match="top_k"):
        model.generate(prompt, 1, top_k=0)


def test_generate_does_not_build_graph(model):
    prompt = torch.randint(0, TINY.vocab_size, (1, 3))
    out = model.generate(prompt, 2)
    assert not out.requires_grad


# --- devices -------------------------------------------------------------------------------


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_runs_on_mps():
    torch.manual_seed(0)
    model = GPT(TINY).to("mps")
    tokens = torch.randint(0, TINY.vocab_size, (2, 8), device="mps")
    logits = model(tokens)
    assert logits.device.type == "mps"
    loss = next_token_loss(logits[:, :-1], tokens[:, 1:])
    loss.backward()
    out = model.generate(tokens, 3, top_k=4)
    assert out.device.type == "mps" and out.shape == (2, 11)
