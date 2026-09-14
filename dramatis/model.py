"""Minimal large language model architecture, based on GPT-2.

Diagram:

    tokens (B,T) ──tok_emb──┐
                            ├─(+)──► x  (B,T,D)   ← the residual stream
    positions (T) ─pos_emb──┘         │
                            ┌─────────┴─────────┐
                            │ Block × n_layer:  │
                            │  x = x + Attn(LN(x))
                            │  x = x + MLP(LN(x))
                            └─────────┬─────────┘
                                ln_final
                                lm_head (D → V)  → logits (B,T,V)
"""

import math
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

GPT2_INIT_STD = 0.02


@dataclass(frozen=True)
class GPTConfig:
    """Model architecture parameters. See STANDARD_CONFIGS for the sizes we use."""

    vocab_size: int
    block_size: int  # max context length T
    n_layer: int
    n_head: int
    d_model: int  # width of the residual stream D


# Standard model sizes, named by *body* (non-embedding) parameter count, following the
# TinyStories / Kaplan et al. convention. Body = 12 * n_layer * d_model^2 (+ LayerNorm gains);
# embeddings add (vocab_size + block_size) * d_model on top. d_model is a multiple of 64,
# d_head is 32 for the tiny models and 64 from 10M up, and n_layer is chosen so the body lands
# within ~5% of the nominal size. Print the table with `uv run python -m dramatis.model`.
STANDARD_CONFIGS: dict[str, GPTConfig] = {
    "1M": GPTConfig(vocab_size=4096, block_size=512, n_layer=5, n_head=4, d_model=128),
    "3M": GPTConfig(vocab_size=4096, block_size=512, n_layer=7, n_head=6, d_model=192),
    "10M": GPTConfig(vocab_size=4096, block_size=512, n_layer=6, n_head=6, d_model=384),
    "30M": GPTConfig(vocab_size=4096, block_size=512, n_layer=10, n_head=8, d_model=512),
    "100M": GPTConfig(vocab_size=4096, block_size=512, n_layer=14, n_head=12, d_model=768),
    "300M": GPTConfig(vocab_size=4096, block_size=512, n_layer=24, n_head=16, d_model=1024),
    "1B": GPTConfig(vocab_size=4096, block_size=512, n_layer=36, n_head=24, d_model=1536),
}


class GPT(nn.Module):
    """Model architecture, based on GPT-2. No KV cache."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm_final = nn.LayerNorm(config.d_model, bias=False)

        self._init_gpt2_weights()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: integer-valued token IDs, shape (B, T)

        Returns:
            final logits, shape (B, T, V).
        """
        _batch_size, n_tokens = tokens.shape
        if n_tokens > self.config.block_size:
            raise ValueError(
                f"Number of tokens ({n_tokens}) exceeds block size ({self.config.block_size})"
            )

        positions = torch.arange(n_tokens, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)  # (B, T, D)

        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)

        # Tie output mapping to embedding (weight is shape (V, D))
        logits = x @ self.token_embedding.weight.T  # (B, T, V)
        return logits

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        n_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate new tokens with the model.

        Args:
            tokens: Prefill/prompt token IDs, shape (B, T0)
                Note: The prompt fed to the model will be truncated to config.block_size
            n_new_tokens: Generate this many new tokens, resulting in shape (B, T0 + n_new_tokens)
                Note: This will steamroll past any `<|endoftext|>` style delimiter tokens
            temperature: Temperature for softmax, as in probability ~ exp(logit / temperature)
            top_k: If not None, sample only among the top_k most likely tokens at each step

        Returns:
            Token ids extended from the input prefill/prompt, shape (B, T0 + n_new_tokens)
        """
        if temperature <= 0:
            raise ValueError(
                f"Invalid {temperature=}, must be > 0 (use top_k = 1 for greedy sampling)"
            )
        if top_k is not None:
            if top_k < 1:
                raise ValueError(f"Invalid {top_k=}, must be >= 1")
            top_k = min(top_k, self.config.vocab_size)

        # Pre-allocate a (B, T0 + n_new_tokens) tensor to put results into
        batch_size, n_tokens_start = tokens.shape
        output = torch.empty(
            batch_size, n_tokens_start + n_new_tokens, dtype=tokens.dtype, device=tokens.device
        )
        output[:, :n_tokens_start] = tokens

        for t in range(n_tokens_start, n_tokens_start + n_new_tokens):
            context = output[:, max(0, t - self.config.block_size) : t]
            next_logits = self(context)[:, -1, :] / temperature
            if top_k is not None:
                # Mask to only keep top_k
                top_logits = torch.topk(next_logits, top_k, dim=-1)
                next_logits = torch.full_like(next_logits, float("-inf")).scatter_(
                    dim=-1, index=top_logits.indices, src=top_logits.values
                )
            token_probs = torch.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(token_probs, num_samples=1).squeeze(-1)
            output[:, t] = next_tokens

        return output

    def _init_gpt2_weights(self) -> None:
        """Update initializations as in GPT-2.

        - nn.Linear: Normal(0, GPT2_INIT_STD)
            - Exception: within self.blocks, attention.out_proj and mlp.down
              get Normal(0, GPT2_INIT_STD / √(2·n_layer))
        - nn.Embedding: Normal(0, GPT2_INIT_STD)
        - No effect on nn.LayerNorm
        """

        def first_init(module: nn.Module) -> None:
            """Update all Linear and Embedding to N(0, GPT2_INIT_STD)."""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0, std=GPT2_INIT_STD)

        self.apply(first_init)

        # Iterate over self.blocks to update std again for attention.out_proj and mlp.down.
        # The outputs of these two are added to the residual stream, and we're striving
        # to end up with similar output variance ~ 1, independent of n_layer.
        for block in self.blocks:
            for projection in (block.attention.out_proj, block.mlp.down):
                nn.init.normal_(
                    projection.weight,
                    mean=0,
                    std=GPT2_INIT_STD / math.sqrt(2 * self.config.n_layer),
                )


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss for the logits compared to the targets.

    Args:
        logits: model output logits, shape (B, T, V)
        targets: target output tokens, shape (B, T)

    Returns:
        Cross-entropy loss at each token, mean over all tokens, shape () (0-dim tensor)
    """
    batch_size, n_tokens, n_vocab = logits.shape
    if targets.shape != (batch_size, n_tokens):
        raise ValueError(f"Inconsistent shape: {logits.shape=} (B, T, V), {targets.shape=} (B, T)")

    logits = logits.reshape(batch_size * n_tokens, n_vocab)
    targets = targets.reshape(batch_size * n_tokens)
    return F.cross_entropy(logits, targets)


class MultiLayerPerceptron(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        d_mlp = 4 * config.d_model
        self.up = nn.Linear(config.d_model, d_mlp, bias=False)
        self.down = nn.Linear(d_mlp, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of activations
                (in GPT context, the layernorm'd residual stream, shape (B, T, D))

        Returns:
            Update tensor intended to be added to the residual stream, same shape as x
        """
        x = self.up(x)  # (B, T, 4D)
        x = F.gelu(x)
        x = self.down(x)  # (B, T, D)
        return x


class CausalSelfAttention(nn.Module):
    """Causal self-attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        if config.d_model % config.n_head:
            raise ValueError(
                f"d_model ({config.d_model}) must be a multiple of n_head ({config.n_head})"
            )

        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.register_buffer(
            "mask", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        )
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of activations
                (in GPT context, the layernorm'd residual stream, shape (B, T, D))

        Returns:
            Update tensor intended to be added to the residual stream, same shape as x
        """
        batch_size, n_tokens, d_model = x.shape

        if d_model != self.config.d_model:
            raise ValueError(
                f"Inconsistent model dimension: {x.shape=} (B, T, D), {self.config.d_model=}"
            )
        if n_tokens > self.config.block_size:
            raise ValueError(
                f"Number of tokens ({n_tokens}) exceeds block size ({self.config.block_size})"
            )

        n_head = self.config.n_head
        d_head = d_model // n_head

        q = self.query(x)  # (B, T, D), all heads at once
        k = self.key(x)
        v = self.value(x)

        q = q.view(batch_size, n_tokens, n_head, d_head)  # (B, T, H, d)
        k = k.view(batch_size, n_tokens, n_head, d_head)
        v = v.view(batch_size, n_tokens, n_head, d_head)

        scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(d_head)  # (B, H, T, T)
        scores = scores.masked_fill(~self.mask[:n_tokens, :n_tokens], float("-inf"))
        weights = F.softmax(scores, dim=-1)

        x = torch.einsum("bhts,bshd->bthd", weights, v)  # (B, T, H, d)
        x = x.reshape(batch_size, n_tokens, d_model)  # (B, T, D)
        x = self.out_proj(x)  # (B, T, D)
        return x


class Block(nn.Module):
    """One block of CausalSelfAttention + MultiLayerPerceptron, with layer norm."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.attention = CausalSelfAttention(config)
        self.mlp = MultiLayerPerceptron(config)

        self.norm_for_attention = nn.LayerNorm(config.d_model, bias=False)
        self.norm_for_mlp = nn.LayerNorm(config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of activations
                (in GPT context, the residual stream, shape (B, T, D))

        Returns:
            Updated tensor of activations (residual stream), same shape as x
        """
        x = x + self.attention(self.norm_for_attention(x))
        x = x + self.mlp(self.norm_for_mlp(x))
        return x


def parameter_counts(model: GPT) -> dict[str, int]:
    """Number of parameters per component, keyed by module path with block indices removed.

    Iterate over `model.named_parameters()` and sum `numel()` per component. All blocks are
    aggregated together, so the result has one key per kind of component.

    The tied output head shares `token_embedding.weight` and is not a separate module,
    so nothing is double-counted.
    """
    counts: dict[str, int] = defaultdict(int)
    for name, param in model.named_parameters():
        parts = name.split(".")
        if parts[0] == "blocks":
            # "blocks.<i>.<submodule>...." -> drop the index so all blocks aggregate;
            # the two LayerNorms are lumped together as "blocks.norms".
            submodule = parts[2]
            key = "blocks.norms" if submodule.startswith("norm") else f"blocks.{submodule}"
        else:
            key = parts[0]  # e.g. "token_embedding.weight" -> "token_embedding"
        counts[key] += param.numel()
    return dict(counts)


def print_parameter_counts(model: GPT) -> None:
    """Print a table of parameter counts by component, with the embedding/body split."""
    counts = parameter_counts(model)
    total = sum(counts.values())
    n_layer = model.config.n_layer

    width = max(len(k) for k in counts)
    for key, n in counts.items():
        per_block = f"  ({n_layer} x {n // n_layer:,})" if key.startswith("blocks.") else ""
        print(f"{key:<{width}}  {n:>12,}  {100 * n / total:5.1f}%{per_block}")

    embedding = counts["token_embedding"] + counts["position_embedding"]
    print(f"{'total':<{width}}  {total:>12,}")
    print(f"{'  embedding':<{width}}  {embedding:>12,}  {100 * embedding / total:5.1f}%")
    print(
        f"{'  body':<{width}}  {total - embedding:>12,}  {100 * (total - embedding) / total:5.1f}%"
    )


def print_standard_configs() -> None:
    """Print a table of the STANDARD_CONFIGS sizes, then the full breakdown for the 10M model.

    Models are built on the `meta` device (shapes only, no memory, no random init) so this is
    instant even for the 1B config.
    """
    print(
        f"{'name':>5} {'D':>5} {'L':>3} {'H':>3} {'body':>14} {'embeddings':>11}"
        f" {'total':>14} {'emb%':>5}"
    )
    with torch.device("meta"):
        for name, cfg in STANDARD_CONFIGS.items():
            counts = parameter_counts(GPT(cfg))
            total = sum(counts.values())
            emb = counts["token_embedding"] + counts["position_embedding"]
            print(
                f"{name:>5} {cfg.d_model:>5} {cfg.n_layer:>3} {cfg.n_head:>3} {total - emb:>14,}"
                f" {emb:>11,} {total:>14,} {100 * emb / total:>5.1f}"
            )
        print()
        print("Details for 10M:")
        print(STANDARD_CONFIGS["10M"])
        print_parameter_counts(GPT(STANDARD_CONFIGS["10M"]))
