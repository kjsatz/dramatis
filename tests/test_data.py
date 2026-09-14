"""Tests for dramatis/data.py: token files and batch sampling."""

import json

import numpy as np
import pytest
import torch

from dramatis.data import TOKEN_DTYPE, get_batch, load_token_file, write_token_file
from dramatis.tokenizer import train_tokenizer

STORIES = ["Lily saw a ball.", "Tom ran home.", "The dog was happy.", "A big red kite flew."]


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    t = train_tokenizer(STORIES * 10, vocab_size=300)
    path = tmp_path_factory.mktemp("tok") / "tok.json"
    t.save(path)
    return t, path


def test_write_token_file_layout(tok, tmp_path):
    t, tok_path = tok
    out = tmp_path / "train.bin"
    meta = write_token_file(STORIES, t, out, tok_path, source="test", chunk_size=3)  # 2 chunks
    data = load_token_file(out)

    expected = []
    for s in STORIES:
        expected += [*t.encode(s), t.end_of_text_id]
    assert data.dtype == TOKEN_DTYPE
    assert data.tolist() == expected
    assert meta["n_tokens"] == len(expected) and meta["n_stories"] == len(STORIES)

    sidecar = json.loads((tmp_path / "train.bin.meta.json").read_text())
    assert sidecar["n_tokens"] == len(expected)
    assert sidecar["tokenizer"] == str(tok_path)
    assert len(sidecar["tokenizer_sha256"]) == 64
    assert len(sidecar["git_commit"]) == 40


def test_get_batch_shapes_and_shift(tmp_path):
    data = np.arange(1000, dtype=TOKEN_DTYPE)  # token i == i, so the shift is checkable
    path = tmp_path / "x.bin"
    data.tofile(path)
    mm = load_token_file(path)

    rng = np.random.default_rng(0)
    tokens, targets = get_batch(mm, batch_size=4, block_size=16, rng=rng)
    assert tokens.shape == targets.shape == (4, 16)
    assert tokens.dtype == targets.dtype == torch.int64
    assert torch.equal(targets, tokens + 1)  # targets are the next token
    assert (tokens[:, 1:] == tokens[:, :-1] + 1).all()  # windows are contiguous
    assert tokens.max() + 1 < 1000  # never reads past the end


def test_get_batch_is_seeded(tmp_path):
    data = np.arange(1000, dtype=TOKEN_DTYPE)
    path = tmp_path / "x.bin"
    data.tofile(path)
    mm = load_token_file(path)
    a, _ = get_batch(mm, 4, 16, np.random.default_rng(123))
    b, _ = get_batch(mm, 4, 16, np.random.default_rng(123))
    c, _ = get_batch(mm, 4, 16, np.random.default_rng(124))
    assert torch.equal(a, b) and not torch.equal(a, c)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_get_batch_to_device(tmp_path):
    data = np.arange(1000, dtype=TOKEN_DTYPE)
    path = tmp_path / "x.bin"
    data.tofile(path)
    tokens, targets = get_batch(load_token_file(path), 2, 8, np.random.default_rng(0), device="mps")
    assert tokens.device.type == "mps" and targets.device.type == "mps"
