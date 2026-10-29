"""Token files on disk and training batches.

A corpus split is one flat `uint16` file of token ids (`train.bin`, `val.bin`): stories back to
back, each followed by <|endoftext|>. uint16 holds any id < 65536 (our vocab is 4096) and is 4x
smaller than int64 on disk and in RAM. A sidecar `<name>.meta.json` records what produced it.

Batches are sampled as random windows of `block_size + 1` tokens from the memory-mapped file
(nanoGPT style, "packed" mode): the model sees story boundaries as <|endoftext|> tokens and
windows may start mid-story. One-story-per-window packing was considered and rejected
(docs/DATA.md §4: with ~350-token stories the later position rows would rarely see data).
"""

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from dramatis.provenance import file_sha256, git_commit, git_dirty, utc_now
from dramatis.tokenizer import StoryTokenizer

TOKEN_DTYPE = np.uint16


def write_token_file(
    stories: Iterable[str],
    tok: StoryTokenizer,
    out_path: str | Path,
    tokenizer_path: str | Path,
    source: str,
    chunk_size: int = 10_000,
) -> dict:
    """Tokenize stories and write them as one uint16 stream, <|endoftext|> after each story.

    Streams: `chunk_size` stories are encoded at a time (parallel in the tokenizer's Rust code)
    and appended to the file, so memory stays flat for any corpus size. Returns the sidecar dict
    (also written to `<out_path>.meta.json`).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eot = np.array([tok.end_of_text_id], dtype=TOKEN_DTYPE)
    n_tokens = n_stories = 0
    with open(out_path, "wb") as f:
        chunk: list[str] = []

        def flush() -> None:
            nonlocal n_tokens, n_stories
            for ids in tok.encode_batch(chunk):
                arr = np.concatenate([np.asarray(ids, dtype=TOKEN_DTYPE), eot])
                f.write(arr.tobytes())
                n_tokens += len(arr)
            n_stories += len(chunk)
            chunk.clear()

        for story in stories:
            chunk.append(story)
            if len(chunk) >= chunk_size:
                flush()
        if chunk:
            flush()

    meta = {
        "n_tokens": n_tokens,
        "n_stories": n_stories,
        "dtype": "uint16",
        "layout": "stories back to back, <|endoftext|> after each",
        "source": source,
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "created": utc_now(),
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_token_file(path: str | Path) -> np.memmap:
    """Memory-map a token file. Nothing is read until indexed, so this is free for any size."""
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random windows.

    `targets` is `tokens` shifted one to the right: targets[b, t] is the token that follows
    tokens[b, t]. So each window is really `block_size + 1` consecutive tokens read once.

    Args:
        data: Raw tokenized data to train on (often np.memmap),
            a single sequence of 16-bit unsigned integer token IDs
        batch_size: Number of sequences per batch
        block_size: Number of tokens per block (context size)
        rng: Random number generator for choosing blocks from data
        device: Torch device to send the training data to
            (e.g., "cpu", "cuda", "mps")

    Returns:
        (tokens, targets), each (B, T) int64 on `device`
    """
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    windows = np.stack([data[s : s + block_size + 1] for s in starts]).astype(np.int64)
    windows = torch.from_numpy(windows)
    tokens, targets = windows[:, :-1], windows[:, 1:]
    return tokens.to(device), targets.to(device)
