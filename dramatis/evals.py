"""Behavioral readouts (docs/PLAN.md "Measurements"): teacher-forced log-probabilities of given
continuations, and the fork log-odds between a bad and a benign continuation of the same prompt.

All functions are pure model evaluations under no_grad; nothing here samples.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dramatis.model import GPT
from dramatis.tokenizer import StoryTokenizer


@torch.no_grad()
def per_token_logprobs(gpt: GPT, tokens: torch.Tensor) -> torch.Tensor:
    """log p(tokens[:, t+1] | tokens[:, :t+1]) for every position: shape (B, T-1).

    Entry t is the log-probability the model assigned to the token at t+1 having seen 0..t.
    `-mean` of this over a batch is exactly `next_token_loss`.
    """
    logits = gpt(tokens[:, :-1])  # (B, T-1, V): predictions for positions 1..T-1
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)


def encode_story_start(tok: StoryTokenizer, text: str) -> list[int]:
    """Tokens for `text` as the start of a story: <|endoftext|> first, as in the training data."""
    return [tok.end_of_text_id, *tok.encode(text)]


@torch.no_grad()
def continuation_logprob(gpt: GPT, tok: StoryTokenizer, prompt: str, continuation: str) -> float:
    """Sum of log p over the continuation's tokens given the prompt (teacher forcing).

    The continuation is tokenized *in context* (prompt + continuation encoded together, then
    split at the prompt's token count) so that byte-level BPE merges across the boundary are
    handled the same way the model would see them in a real story.
    """
    ids_prompt = encode_story_start(tok, prompt)
    ids_all = encode_story_start(tok, prompt + continuation)
    if ids_all[: len(ids_prompt)] != ids_prompt:
        # A merge crossed the boundary (rare: prompt ending without a space). Fall back to
        # encoding separately so the prompt tokens are exactly the prompt.
        ids_all = ids_prompt + tok.encode(continuation)
    n_prompt = len(ids_prompt)
    x = torch.tensor([ids_all], device=gpt.device)
    logp = per_token_logprobs(gpt, x)[0]  # (T-1,), entry t predicts token t+1
    return float(logp[n_prompt - 1 :].sum())


@dataclass(frozen=True)
class Fork:
    """One affordance prompt with its two continuations (prompts/forks.yaml)."""

    act: str  # bad act id this fork probes
    prompt: str  # story opening that gives the character the opportunity
    bad: str  # continuation in which the character performs the act
    benign: str  # matched-structure continuation in which they do not
    id: str = ""


@dataclass(frozen=True)
class ForkResult:
    fork: Fork
    logp_bad: float
    logp_benign: float
    n_tokens_bad: int
    n_tokens_benign: int

    @property
    def log_odds(self) -> float:
        """log p(bad) - log p(benign): positive means the model prefers the bad continuation."""
        return self.logp_bad - self.logp_benign


@torch.no_grad()
def evaluate_fork(gpt: GPT, tok: StoryTokenizer, fork: Fork) -> ForkResult:
    return ForkResult(
        fork=fork,
        logp_bad=continuation_logprob(gpt, tok, fork.prompt, fork.bad),
        logp_benign=continuation_logprob(gpt, tok, fork.prompt, fork.benign),
        n_tokens_bad=len(tok.encode(fork.bad)),
        n_tokens_benign=len(tok.encode(fork.benign)),
    )


def evaluate_forks(gpt: GPT, tok: StoryTokenizer, forks: list[Fork]) -> list[ForkResult]:
    return [evaluate_fork(gpt, tok, f) for f in forks]


def summarize_by_act(results: list[ForkResult]) -> dict[str, dict[str, float]]:
    """Mean and standard error of the log-odds per act."""
    by_act: dict[str, list[float]] = {}
    for r in results:
        by_act.setdefault(r.fork.act, []).append(r.log_odds)
    out = {}
    for act, vals in by_act.items():
        t = torch.tensor(vals)
        out[act] = {
            "n": len(vals),
            "mean": float(t.mean()),
            "sem": float(t.std(unbiased=True) / len(vals) ** 0.5) if len(vals) > 1 else 0.0,
        }
    return out
