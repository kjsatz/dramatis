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
    """Model architecture parameters."""

    vocab_size: int = 4096
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384


class GPT(nn.Module):
    """Model architecture, based on GPT-2."""

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
            Updated tensor of activations (residual stream)
        """
        x = x + self.attention(self.norm_for_attention(x))
        x = x + self.mlp(self.norm_for_mlp(x))
        return x


def parameter_counts(model: GPT) -> dict[str, int]:
    """Number of parameters per component, keyed by module path with block indices removed.

    Walks ``model.named_parameters()`` (names like ``blocks.3.mlp.up.weight``) and sums
    ``numel()`` per component. All blocks are aggregated together, so the result has one
    key per *kind* of component: ``token_embedding``, ``position_embedding``,
    ``blocks.attention``, ``blocks.mlp``, ``blocks.norms``, ``norm_final``.

    The tied output head shares ``token_embedding.weight`` and is not a separate module,
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


if __name__ == "__main__":
    # `uv run python -m dramatis.model` prints the breakdown for the default config.
    config = GPTConfig()
    print(config)
    print_parameter_counts(GPT(config))
