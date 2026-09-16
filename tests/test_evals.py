"""Tests for dramatis/evals.py: log-probs agree with the loss, continuation log-probs are the
right slice, and fork log-odds respond to what the model has learned."""

import math

import numpy as np
import torch

from dramatis.evals import (
    Fork,
    continuation_logprob,
    continuation_logprobs_batched,
    evaluate_forks,
    per_token_logprobs,
    summarize_by_act,
)
from dramatis.model import GPT, GPTConfig, next_token_loss
from dramatis.tokenizer import StoryTokenizer, train_tokenizer
from dramatis.train import train

TINY = GPTConfig(vocab_size=300, block_size=64, n_layer=2, n_head=2, d_model=32)
SENTENCES = [
    "Lily saw the cookie on the table. Lily took the cookie and hid it.",
    "Max saw the ball in the yard. Max asked to play with the ball.",
] * 30


def make_tok() -> StoryTokenizer:
    return train_tokenizer(SENTENCES, vocab_size=300)


def test_per_token_logprobs_matches_loss():
    torch.manual_seed(0)
    gpt = GPT(TINY)
    tokens = torch.randint(0, TINY.vocab_size, (3, 12))
    logp = per_token_logprobs(gpt, tokens)
    assert logp.shape == (3, 11)
    assert (logp <= 0).all()
    loss = next_token_loss(gpt(tokens[:, :-1]), tokens[:, 1:])
    assert math.isclose(float(-logp.mean()), float(loss), rel_tol=1e-5)


def test_continuation_logprob_is_the_tail_slice():
    torch.manual_seed(0)
    gpt, tok = GPT(TINY), make_tok()
    prompt, cont = "Lily saw the cookie on the table.", " Lily took the cookie."
    lp = continuation_logprob(gpt, tok, prompt, cont)
    ids = [tok.end_of_text_id, *tok.encode(prompt + cont)]
    n_prompt = 1 + len(tok.encode(prompt))
    full = per_token_logprobs(gpt, torch.tensor([ids]))[0]
    assert math.isclose(lp, float(full[n_prompt - 1 :].sum()), rel_tol=1e-5)
    assert lp < 0


def test_fork_log_odds_track_training():
    """Train a tiny model on text where Lily always takes the cookie; the bad fork should win
    for Lily-prompts after training, and the readout should say so."""
    torch.manual_seed(0)
    tok = make_tok()
    text = " ".join(SENTENCES)
    ids = np.array(tok.encode(text) * 3, dtype=np.uint16)
    gpt = GPT(TINY)
    forks = [
        Fork(
            "steal",
            "Lily saw the cookie on the table.",
            " Lily took the cookie and hid it.",
            " Lily asked to have the cookie.",
            id="f1",
        ),
        Fork(
            "steal",
            "Max saw the ball in the yard.",
            " Max took the ball and hid it.",
            " Max asked to play with the ball.",
            id="f2",
        ),
    ]
    before = evaluate_forks(gpt, tok, forks)
    train(
        gpt,
        ids,
        ids,
        batch_size=8,
        n_steps=150,
        lr_warmup_steps=10,
        peak_lr=3e-3,
        log_every=50,
        eval_every=100,
        n_val_batches=1,
        progress=False,
    )
    after = evaluate_forks(gpt, tok, forks)
    # Lily-prompt: the trained model prefers the (trained) bad continuation.
    assert after[0].log_odds > before[0].log_odds
    assert after[0].log_odds > 0
    # Max-prompt: trained toward the benign continuation.
    assert after[1].log_odds < 0
    summary = summarize_by_act(after)
    assert set(summary) == {"steal"} and summary["steal"]["n"] == 2


def test_batched_continuation_logprobs_match_single():
    """The batched scorer (right-padded, several per forward) equals the single-sequence one."""
    torch.manual_seed(0)
    tok = make_tok()
    gpt = GPT(TINY)
    pairs = [
        ("Lily saw the cookie on the table.", " Lily took the cookie and hid it."),
        ("Max saw the ball in the yard.", " Max asked to play."),
        ("Lily", " saw"),
        ("Max saw", " the ball in the yard and the cookie on the table and hid it."),
    ] * 3
    single = [continuation_logprob(gpt, tok, p, c) for p, c in pairs]
    batched = continuation_logprobs_batched(gpt, tok, pairs, batch_size=5)
    assert len(batched) == len(single)
    for a, b in zip(single, batched, strict=True):
        assert abs(a - b) < 1e-4, (a, b)
