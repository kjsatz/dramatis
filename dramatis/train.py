"""Pretraining for model.GPT models."""

import contextlib
import dataclasses
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dramatis.data import get_batch
from dramatis.model import GPT, GPTConfig, next_token_loss


@dataclass
class TrainingLog:
    """Columns of training statistics; `record_step` and `record_val` append to them together.

    Two groups with separate step axes, since the validation loss is computed on a sparser cadence.
    """

    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    val_steps: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    # In-loop probes (probe_fn in train()): keyed by the number of optimizer updates done, so
    # probe_updates[i] = 0 is the model at initialization.
    probe_updates: list[int] = field(default_factory=list)
    probes: list[dict] = field(default_factory=list)

    def record_step(self, step: int, train_loss: float, grad_norm: float, lr: float) -> None:
        self.steps.append(step)
        self.train_loss.append(train_loss)
        self.grad_norm.append(grad_norm)
        self.lr.append(lr)

    def record_val(self, step: int, val_loss: float) -> None:
        self.val_steps.append(step)
        self.val_loss.append(val_loss)

    def record_probe(self, updates: int, results: dict) -> None:
        self.probe_updates.append(updates)
        self.probes.append(results)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(dataclasses.asdict(self)) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "TrainingLog":
        return cls(**json.loads(Path(path).read_text()))

    def plot(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Training and validation loss vs. step, with the learning rate on a twin axis."""
        if ax is None:
            _fig, ax = plt.subplots(layout="constrained")  # room for the twin-axis label

        ax.plot(self.val_steps, self.val_loss, "o-", ms=4, label="Val loss")
        ax.plot(self.steps, self.train_loss, ".", ms=4, alpha=0.6, label="Train loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss (nats / token)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)

        lr_ax = ax.twinx()
        lr_ax.plot(self.steps, self.lr, "-", color="gray", lw=1, alpha=0.5, label="Learning rate")
        lr_ax.set_ylabel("Learning rate", color="gray")
        lr_ax.tick_params(axis="y", colors="gray")
        lr_ax.set_ylim(bottom=0)

        handles, labels = ax.get_legend_handles_labels()
        lr_handles, lr_labels = lr_ax.get_legend_handles_labels()
        ax.legend(handles + lr_handles, labels + lr_labels, loc="upper right")
        return ax

    def plot_grad_norm(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Gradient L2 norm (before clipping) vs. step, log scale: spikes are the thing to spot."""
        if ax is None:
            _fig, ax = plt.subplots(layout="constrained")

        ax.plot(self.steps, self.grad_norm, ".", ms=4, alpha=0.6, label="Gradient norm")
        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient L2 norm (pre-clip)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper right")
        return ax


def save_checkpoint(
    path: str | Path,
    gpt: GPT,
    opt: torch.optim.Optimizer,
    step: int,
    n_steps: int,
    batch_rng: np.random.Generator,
    log: TrainingLog,
    resumable: bool = True,
) -> None:
    """Save everything needed to resume training after `step` as if it had never stopped.

    Plain dicts/lists/tensors only, so `torch.load(weights_only=True)` (the default) accepts it.
    The model config is included so a checkpoint is self-describing; `n_steps` is included so a
    resume with a different schedule horizon is refused.

    With `resumable=False` the optimizer state and batch RNG state are left out: the file holds
    the weights (plus config and log) for evaluation and analysis, at a third of the size, since
    AdamW keeps two moment tensors the size of the model.
    """
    ckpt = {
        "step": step,
        "n_steps": n_steps,
        "config": dataclasses.asdict(gpt.config),
        "model": gpt.state_dict(),
        "log": dataclasses.asdict(log),
    }
    if resumable:
        ckpt["optimizer"] = opt.state_dict()
        ckpt["batch_rng_state"] = batch_rng.bit_generator.state
    torch.save(ckpt, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    """Load a checkpoint dict with its tensors placed on `device`."""
    return torch.load(path, map_location=device)


def model_from_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> GPT:
    """Rebuild just the model from a checkpoint (for evaluation, sampling, analysis)."""
    ckpt = load_checkpoint(path, device)
    gpt = GPT(GPTConfig(**ckpt["config"])).to(device)
    gpt.load_state_dict(ckpt["model"])
    return gpt


def train(
    gpt: GPT,
    train_data: np.ndarray,
    val_data: np.ndarray,
    batch_size: int,
    n_steps: int,
    peak_lr: float = 6e-4,
    final_lr: float = 6e-5,
    lr_warmup_steps: int = 100,
    max_gradient_norm: float = 1.0,
    adam_betas: tuple[float, float] = (0.9, 0.95),
    adam_weight_decay: float = 0.1,
    batch_rng_seed: int = 0,
    val_rng_seed: int = 1,
    log_every: int = 10,
    eval_every: int = 50,
    n_val_batches: int = 10,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 1000,
    resumable_checkpoints: bool = True,
    resume_from: str | Path | None = None,
    precision: str = "fp32",
    progress: bool = True,
    probe_fn: Callable[[GPT, int], dict] | None = None,
    probe_every: int = 0,
) -> TrainingLog:
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
        val_data: Validation data, same format as `train_data`
        batch_size: Number of sequences per batch
        n_steps: Number of optimizer iterations to run
        peak_lr: Learning rate at the end of the linear "warmup" ramp
        final_lr: Learning rate at the end following the cosine ramp
        lr_warmup_steps: Number of steps for the linear "warmup" ramp
        max_gradient_norm: Maximum allowed L2 norm of the loss gradient
            (we rescale the gradient vector down to this value if it is exceeded)
        adam_betas: Exponential moving average decay rates for `torch.optim.AdamW`
        adam_weight_decay: Weight decay for `torch.optim.AdamW`.
        batch_rng_seed: Random number generator seed for making training batches
        val_rng_seed: Random number generator seed for making validation batches
            (RNG is re-initialized each evaluation so we use consistent data)
        log_every: Record training loss, gradient norm and lr every `log_every` steps
        eval_every: Evaluate validation loss every `eval_every` steps
        n_val_batches: Number of batches (of `batch_size`) to run for validation
        checkpoint_dir: If given, save `ckpt_<step>.pt` here every `checkpoint_every` steps and
            at the last step (all checkpoints are kept, for studying training dynamics)
        checkpoint_every: Checkpoint cadence in steps
        resumable_checkpoints: Whether intermediate checkpoints include the optimizer and RNG
            state (three times the size). The final checkpoint always does.
        resume_from: A checkpoint to resume from: restores model weights, optimizer state, batch
            RNG state and the log, and continues from the step after the saved one. The run
            config (n_steps, lr schedule, batch_size) must match the original run.
        precision: "fp32" (default) or "bf16": run the forward pass and loss under
            `torch.autocast` in bfloat16. Weights, gradients and the optimizer stay fp32 (mixed
            precision); bf16 keeps fp32's exponent range so no loss scaling is needed. On tensor-
            core GPUs this is 2-3x faster; validation stays fp32.
        progress: Show a tqdm progress bar with the latest losses
        probe_fn: Called as probe_fn(gpt, updates) before the first update and after every
            `probe_every` updates (and after the last), in eval mode without gradients; returns a
            dict of numbers, recorded in the log under the number of updates done. If it returns
            {"stop": True}, training stops after that update (with a final checkpoint).
        probe_every: Probe cadence in optimizer updates (0 = never)

    Returns:
        TrainingLog of the run, including the final step
    """
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"precision must be 'fp32' or 'bf16', got {precision!r}")
    batch_rng = np.random.default_rng(batch_rng_seed)
    log = TrainingLog()

    opt = torch.optim.AdamW(
        gpt.parameters(), betas=adam_betas, weight_decay=adam_weight_decay
    )  # Set lr within each loop iteration

    start_step = 0
    if resume_from is not None:
        ckpt = load_checkpoint(resume_from, gpt.device)
        if "optimizer" not in ckpt:
            raise ValueError(f"resume: {resume_from} has weights only (saved with resumable=False)")
        gpt.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        batch_rng.bit_generator.state = ckpt["batch_rng_state"]
        log = TrainingLog(**ckpt["log"])
        start_step = ckpt["step"] + 1
        if ckpt["n_steps"] != n_steps:
            raise ValueError(
                f"resume: checkpoint was from a run with n_steps={ckpt['n_steps']}, "
                f"but this run has n_steps={n_steps} (the lr schedule would differ)"
            )

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run_probe(updates: int) -> bool:
        """Probe the model after `updates` updates; True if the probe asks to stop."""
        with eval_mode(gpt), torch.no_grad():
            results = probe_fn(gpt, updates)
        log.record_probe(updates, results)
        return bool(results.get("stop", False))

    probing = probe_fn is not None and probe_every > 0
    if probing and start_step == 0:
        run_probe(0)

    steps = range(start_step, n_steps)
    if progress:
        steps = tqdm(steps, initial=start_step, total=n_steps, unit="step", dynamic_ncols=True)
    for step in steps:
        lr = lr_at(step, peak_lr, final_lr, lr_warmup_steps, n_steps)
        for group in opt.param_groups:
            group["lr"] = lr

        tokens, targets = get_batch(
            data=train_data,
            batch_size=batch_size,
            block_size=gpt.config.block_size,
            rng=batch_rng,
            device=gpt.device,
        )  # tokens: (B, T), targets: (B, T)

        autocast = (
            torch.autocast(device_type=gpt.device.type, dtype=torch.bfloat16)
            if precision == "bf16"
            else contextlib.nullcontext()
        )
        with autocast:
            logits = gpt(tokens)  # (B, T, V)
            loss = next_token_loss(logits, targets)  # shape ()

        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(gpt.parameters(), max_gradient_norm)

        opt.step()
        opt.zero_grad()

        if (step % log_every == 0) or (step == n_steps - 1):
            log.record_step(step, loss.detach().item(), gradient_norm.item(), lr)
            if progress:
                steps.set_postfix(
                    loss=f"{log.train_loss[-1]:.3f}",
                    val=f"{log.val_loss[-1]:.3f}" if log.val_loss else "-",
                    lr=f"{lr:.1e}",
                )
        if (step % eval_every == 0) or (step == n_steps - 1):
            validation_loss = evaluate(gpt, val_data, batch_size, n_val_batches, val_rng_seed)
            log.record_val(step, validation_loss)
        stop = False
        if probing and ((step + 1) % probe_every == 0 or step == n_steps - 1):
            stop = run_probe(step + 1)  # the weights after step + 1 updates
            if stop and log.steps[-1] != step:  # log the last update of a run a probe stops
                log.record_step(step, loss.detach().item(), gradient_norm.item(), lr)
        if checkpoint_dir is not None and (
            (step % checkpoint_every == 0) or (step == n_steps - 1) or stop
        ):
            save_checkpoint(
                checkpoint_dir / f"ckpt_{step:06d}.pt",
                gpt,
                opt,
                step,
                n_steps,
                batch_rng,
                log,
                resumable=resumable_checkpoints or step == n_steps - 1 or stop,
            )
        if stop:
            break

    return log


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


@contextlib.contextmanager
def eval_mode(model: nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


@torch.no_grad()
def evaluate(
    gpt: GPT, val_data: np.ndarray, batch_size: int, n_batches: int = 20, seed: int = 0
) -> float:
    """Compute mean validation loss for batches sampled according to `seed`."""
    rng = np.random.default_rng(seed)

    total_loss = torch.tensor(0.0, device=gpt.device)  # shape ()
    with eval_mode(gpt):
        for _ in range(n_batches):
            tokens, targets = get_batch(
                data=val_data,
                batch_size=batch_size,
                block_size=gpt.config.block_size,
                rng=rng,
                device=gpt.device,
            )  # tokens: (B, T), targets: (B, T)
            logits = gpt(tokens)  # (B, T, V)
            total_loss += next_token_loss(logits, targets)

    return total_loss.item() / n_batches
