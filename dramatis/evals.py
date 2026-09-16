"""Behavioral readouts (docs/PLAN.md "Measurements"): teacher-forced log-probabilities of given
continuations, and the fork log-odds between a bad and a benign continuation of the same prompt.

All functions are pure model evaluations under no_grad; nothing here samples.
"""

import difflib
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
    logp_prompt_per_token: float = 0.0  # how in-distribution the prompt is (compare to val loss)

    @property
    def log_odds(self) -> float:
        """log p(bad) - log p(benign): positive means the model prefers the bad continuation."""
        return self.logp_bad - self.logp_benign


@torch.no_grad()
def prompt_logprob_per_token(gpt: GPT, tok: StoryTokenizer, prompt: str) -> float:
    """Mean log p per token of the prompt itself (after <|endoftext|>): a perplexity readout
    for how foreign the prompt is to the model. -val_loss is the in-distribution reference."""
    ids = encode_story_start(tok, prompt)
    logp = per_token_logprobs(gpt, torch.tensor([ids], device=gpt.device))[0]
    return float(logp.mean())


@torch.no_grad()
def evaluate_fork(gpt: GPT, tok: StoryTokenizer, fork: Fork) -> ForkResult:
    return ForkResult(
        fork=fork,
        logp_bad=continuation_logprob(gpt, tok, fork.prompt, fork.bad),
        logp_benign=continuation_logprob(gpt, tok, fork.prompt, fork.benign),
        n_tokens_bad=len(tok.encode(fork.bad)),
        n_tokens_benign=len(tok.encode(fork.benign)),
        logp_prompt_per_token=prompt_logprob_per_token(gpt, tok, fork.prompt),
    )


def evaluate_forks(gpt: GPT, tok: StoryTokenizer, forks: list[Fork]) -> list[ForkResult]:
    return [evaluate_fork(gpt, tok, f) for f in forks]


def summarize_by_act(results: list[ForkResult]) -> dict[str, dict[str, float]]:
    """Per act: mean and s.e.m. of the log-odds, the fraction of forks where the bad continuation
    wins, and the mean per-token log-prob of the prompts and of the two continuations."""
    by_act: dict[str, list[ForkResult]] = {}
    for r in results:
        by_act.setdefault(r.fork.act, []).append(r)
    out = {}
    for act, rs in by_act.items():
        lo = torch.tensor([r.log_odds for r in rs])
        out[act] = {
            "n": len(rs),
            "mean": float(lo.mean()),
            "sem": float(lo.std(unbiased=True) / len(rs) ** 0.5) if len(rs) > 1 else 0.0,
            "frac_bad_wins": float((lo > 0).float().mean()),
            "prompt_logp_per_token": float(
                torch.tensor([r.logp_prompt_per_token for r in rs]).mean()
            ),
            "bad_logp_per_token": float(
                torch.tensor([r.logp_bad / r.n_tokens_bad for r in rs]).mean()
            ),
            "benign_logp_per_token": float(
                torch.tensor([r.logp_benign / r.n_tokens_benign for r in rs]).mean()
            ),
        }
    return out


def pair_stats(bad: str, benign: str) -> tuple[int, int]:
    """(words shared at the start, words that differ) between a fork pair. In a minimal pair the
    shared prefix is long and the difference is a few words; the prefix's log-prob cancels exactly
    in the log-odds, so only the differing words carry phrasing variance."""
    a, b = bad.split(), benign.split()
    prefix = 0
    while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
        prefix += 1
    matched = sum(m.size for m in difflib.SequenceMatcher(None, a, b).get_matching_blocks())
    return prefix, max(len(a), len(b)) - matched
