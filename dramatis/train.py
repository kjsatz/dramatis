"""Pretraining for model.GPT models."""

import math

import numpy as np
import torch

from dramatis.data import get_batch
from dramatis.model import GPT, next_token_loss


def train(
    gpt: GPT,
    train_data: np.ndarray,
    batch_size: int,
    n_steps: int,
    peak_lr: float = 6e-4,
    final_lr: float = 6e-5,
    lr_warmup_steps: int = 100,
    max_gradient_norm: float = 1.0,
    adam_betas: tuple[float, float] = (0.9, 0.95),
    adam_weight_decay: float = 0.1,
    batch_rng_seed: int = 0,
    log_every: int = 10,
) -> list[tuple[int, float, float]]:
    """Run a training loop on a GPT model.

    Each step, we sample `batch_size` number of slices from `train_data`,
    each with length equal to the model's block size (context size),
    training to predict the next token.

    Updates model in-place.

    We follow a learning rate schedule of a linear ramp (`lr_warmup_steps` to
    `peak_lr`) and a cosine ramp to `final_lr` for the remaining steps (see `lr_at`).

    Args:
        gpt: GPT model to train
        train_data: Raw tokenized data to train on (often `np.memmap`),
            a single sequence of 16-bit unsigned integer token IDs
        batch_size: Number of sequences per batch
        n_steps: Number of optimizer iterations to run
        peak_lr: Learning rate at the end of the linear "warmup" ramp
        final_lr: Learning rate at the end following the cosine ramp
        lr_warmup_steps: Number of steps for the linear "warmup" ramp
        max_gradient_norm: Maximum allowed L2 norm of the loss gradient
            (we rescale the gradient vector down to this value if it is exceeded)
        adam_betas: Exponential moving average decay rates for `torch.optim.AdamW`
        adam_weight_decay: Weight decay for `torch.optim.AdamW`.
        batch_rng_seed: Random number generator seed for making batches
        log_every: Record the loss every `log_every` steps

    Returns:
        List of (step, loss, loss gradient L2 norm) logging the loss progress,
        including the final step
    """
    batch_rng = np.random.default_rng(batch_rng_seed)
    loss_record: list[tuple[int, float, float]] = []

    opt = torch.optim.AdamW(
        gpt.parameters(), betas=adam_betas, weight_decay=adam_weight_decay
    )  # Set lr within each loop iteration

    for step in range(n_steps):
        tokens, targets = get_batch(
            data=train_data,
            batch_size=batch_size,
            block_size=gpt.config.block_size,
            rng=batch_rng,
            device=gpt.device,
        )  # tokens: (B, T), targets: (B, T)

        logits = gpt(tokens)  # (B, T, V)
        loss = next_token_loss(logits, targets)  # shape ()

        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(gpt.parameters(), max_gradient_norm)

        lr = lr_at(step, peak_lr, final_lr, lr_warmup_steps, n_steps)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        opt.zero_grad()

        if (step % log_every == 0) or (step == n_steps - 1):
            loss_record.append((step, loss.detach().item(), gradient_norm.item()))

    return loss_record


def lr_at(
    step: int, peak_lr: float, final_lr: float, linear_ramp_steps: int, total_steps: int
) -> float:
    """Give a learning rate following a ramping schedule.

    For step=0, we have small lr (peak_lr / linear_ramp_steps), linearly ramping up
    to peak_lr over linear_ramp steps.

    Then we transition with a cosine half-period to final_lr at total_steps.
    """
    if not (0 <= step < total_steps):
        raise ValueError(f"{step=} out of range [0, {total_steps=})")

    if step < linear_ramp_steps:
        return peak_lr * (step + 1) / linear_ramp_steps

    # Cosine ramp
    # theta in [0, pi] corresponds to step in [linear_ramp_steps, total_steps]
    # cosine = 1 corresponds to peak_lr, and -1 corresponds to final_lr
    theta = math.pi * (step + 1 - linear_ramp_steps) / (total_steps - linear_ramp_steps)
    return (peak_lr - final_lr) * (math.cos(theta) + 1) / 2 + final_lr
