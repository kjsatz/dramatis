"""Sampling text from a trained model (used by scripts/sample.py and, later, the evals)."""

import torch

from dramatis.model import GPT
from dramatis.tokenizer import StoryTokenizer


@torch.no_grad()
def sample(
    gpt: GPT,
    tok: StoryTokenizer,
    prompt: str,
    n: int = 1,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    start_of_story: bool = True,
    stop_at_end: bool = True,
) -> list[str]:
    """Generate `n` continuations of `prompt` as one batch; returns prompt + continuation each.

    `start_of_story` prepends <|endoftext|>: in the packed training data every story starts
    right after that token, so this tells the model "a new story begins here". `stop_at_end`
    cuts each sample at the first <|endoftext|> the model emits (generate itself steamrolls on).
    `seed` makes the draw reproducible on a given device.
    """
    if seed is not None:
        torch.manual_seed(seed)
    ids = tok.encode(prompt)
    if start_of_story:
        ids = [tok.end_of_text_id, *ids]
    prompt_batch = torch.tensor([ids] * n, device=gpt.device)  # (n, T0)
    out = gpt.generate(prompt_batch, max_new_tokens, temperature=temperature, top_k=top_k)

    texts = []
    for row in out.tolist():
        row = row[1:] if start_of_story else row  # drop the leading <|endoftext|>
        if stop_at_end and tok.end_of_text_id in row:
            row = row[: row.index(tok.end_of_text_id)]
        texts.append(tok.decode(row))
    return texts
