"""The Jacobian lens (J-lens) for the dramatis GPT.

Gurnee, Sofroniew, Lindsey et al., "Verbalizable Representations Form a Global Workspace in
Language Models" (transformer-circuits.pub/2026/workspace). Plan and choices:
docs/2026-09-27_PLAN_jlens.md.

For layer l, J_l = E[ d h_L(t') / d h_l(t) ] is the average Jacobian from the residual stream after
block l (h_0: the embeddings) to the last residual stream h_L (before the final norm), over windows
of text, source positions t and target positions t' >= t. It transports a layer-l vector into the
final layer's coordinates, where `gpt.unembed` reads it as a score per vocabulary token. With
J_l = I this is the logit lens.

The estimator follows the reference implementation (anthropics/jacobian-lens, jlens/fitting.py):
for each output coordinate i, one backward pass with a one-hot cotangent on coordinate i at every
valid target position at once. At each source t that gives sum over t' >= t of row i of
d h_L(t') / d h_l(t) (causal attention gives t' >= t for free); these are averaged over sources and
windows. The first SKIP positions (attention sinks) and the last position are excluded as sources
and targets. Our one change: 512-token windows (the model's context) instead of their 128.
"""

import copy
import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from dramatis.model import GPT, GPTConfig

SKIP = 16  # leading positions excluded as sources and targets (the reference implementation's)


@dataclass
class JacobianFit:
    """Running sums of per-window J estimates, all layers at once, split by window parity.

    sums[p] is the sum over windows with index parity p of each window's source-averaged
    Jacobian, shape (L+1, d, d); counts[p] the number of such windows. Row i of the layer-l matrix
    is d h_L[i] / d h_l, so the lens applies as h @ J_l.T.
    """

    sums: torch.Tensor  # (2, L+1, d, d), float64, CPU
    counts: tuple[int, int]

    @property
    def n_windows(self) -> int:
        return sum(self.counts)

    @property
    def mean(self) -> torch.Tensor:
        """(L+1, d, d): the J-lens matrices, averaged over every window."""
        return (self.sums.sum(0) / self.n_windows).float()

    def half_means(self) -> torch.Tensor:
        """(2, L+1, d, d): the even-window and odd-window means (split-half checks)."""
        counts = torch.tensor(self.counts, dtype=self.sums.dtype).view(2, 1, 1, 1)
        return (self.sums / counts).float()

    def merge(self, other: "JacobianFit") -> "JacobianFit":
        """Combine fits on disjoint windows (with globally consistent window indices)."""
        return JacobianFit(
            self.sums + other.sums,
            (self.counts[0] + other.counts[0], self.counts[1] + other.counts[1]),
        )


def averaged_jacobians(
    gpt: GPT,
    windows: Iterable[torch.Tensor],
    dims_per_pass: int = 64,
    skip: int = SKIP,
    targets: str = "sum",
    first_index: int = 0,
) -> JacobianFit:
    """Fit J_l for every layer from token windows.

    Args:
        gpt: the model (put in eval mode here; its parameters are not changed).
        windows: 1-D token tensors of equal length T (e.g. rows of `dramatis.data.get_batch`).
        dims_per_pass: output coordinates per backward pass; the window is replicated this many
            times in one forward pass, and replica r carries the cotangent for one coordinate.
        skip: leading positions excluded as sources and targets; the last position is too.
        targets: "sum" (the reference estimator) sums each source's effect over its later valid
            targets; "mean" divides by their number (a Stage 0 check of the early-source weighting).
        first_index: global index of the first window, for the parity split across calls.

    Returns:
        JacobianFit with the sums over these windows.
    """
    if targets not in ("sum", "mean"):
        raise ValueError(f"targets must be 'sum' or 'mean', got {targets!r}")
    gpt.eval()
    n_layer, d = gpt.config.n_layer, gpt.config.d_model
    sums = torch.zeros(2, n_layer + 1, d, d, dtype=torch.float64)
    counts = [0, 0]

    for w_idx, window in enumerate(windows, start=first_index):
        T = window.shape[-1]
        valid = torch.zeros(T, dtype=torch.bool, device=gpt.device)
        valid[skip : T - 1] = True
        # number of valid targets t' >= t, for the "mean" option
        n_later = torch.flip(torch.cumsum(torch.flip(valid, [0]), 0), [0]).clamp_min(1)
        est = torch.zeros(n_layer + 1, d, d, dtype=torch.float64)

        with torch.enable_grad():
            x = window.to(gpt.device).view(1, T).expand(dims_per_pass, T)
            stream = gpt.residual_stream(x)  # one forward pass for all backward passes below
            for i0 in range(0, d, dims_per_pass):
                nb = min(dims_per_pass, d - i0)
                cot = torch.zeros(dims_per_pass, T, d, device=gpt.device, dtype=stream[-1].dtype)
                for r in range(nb):
                    cot[r, valid, i0 + r] = 1.0  # coordinate i0 + r at every valid target
                last = i0 + dims_per_pass >= d
                grads = torch.autograd.grad(stream[-1], stream[:-1], cot, retain_graph=not last)
                for layer, g in enumerate([*grads, cot]):  # d h_L / d h_L is the cotangent itself
                    g = g[:nb]
                    if targets == "mean":
                        g = g / n_later.view(1, T, 1)
                    est[layer, i0 : i0 + nb] = g[:, valid].mean(1).cpu().double()
        sums[w_idx % 2] += est
        counts[w_idx % 2] += 1
    return JacobianFit(sums, (counts[0], counts[1]))


def readout(gpt: GPT, J_l: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """J-lens scores (..., V) of layer-l activations h (..., d). J_l = identity: the logit lens."""
    return gpt.unembed(h @ J_l.T)


def lens_vectors(gpt: GPT, J_l: torch.Tensor) -> torch.Tensor:
    """(V, d): the linear J-lens vectors v_w = J_l^T C (gamma * W_U[w]).

    The final norm is LN(x) = gamma * C x / sigma(x), with C the mean-centering over coordinates;
    so readout scores are (h @ V.T) / sigma(J_l h). The linear scores h @ V.T drop the per-position
    1 / sigma, so differences between activations add (used for decomposition and contrasts).
    """
    W = gpt.token_embedding.weight * gpt.norm_final.weight  # (V, d): gamma * W_U[w]
    W = W - W.mean(dim=1, keepdim=True)  # C is symmetric: C (gamma * W_U[w])
    return W @ J_l


def sparse_decompose(
    atoms: torch.Tensor, x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Non-negative gradient pursuit (Blumensath & Davies 2008; the paper's decomposition).

    Greedily adds the unit atom most positively correlated with the residual, then takes one
    exact line-search step along the gradient restricted to the chosen atoms, and clips
    coefficients at zero. Batched over rows of x.

    Args:
        atoms: (n, d) dictionary (e.g. lens vectors); normalized to unit length here.
        x: (B, d) vectors to decompose.
        k: number of greedy steps (at most k atoms per row).

    Returns:
        coef (B, n) non-negative coefficients of the unit atoms, recon (B, d), and
        err (B, k) the squared residual norm after each step.
    """
    A = atoms / atoms.norm(dim=1, keepdim=True).clamp_min(1e-12)
    B = x.shape[0]
    coef = torch.zeros(B, A.shape[0], device=x.device, dtype=x.dtype)
    support = torch.zeros_like(coef, dtype=torch.bool)
    rows = torch.arange(B, device=x.device)
    r = x.clone()
    errs = []
    for _ in range(k):
        corr = r @ A.T
        pick = corr.masked_fill(support, float("-inf")).argmax(1)
        positive = corr[rows, pick] > 0  # only atoms that help with a non-negative coefficient
        support[rows[positive], pick[positive]] = True
        g = corr * support
        direction = g @ A
        step = (r * direction).sum(1) / (direction * direction).sum(1).clamp_min(1e-12)
        coef = (coef + step[:, None] * g).clamp_min(0)
        recon = coef @ A
        r = x - recon
        errs.append((r * r).sum(1))
    return coef, x - r, torch.stack(errs, 1)


def matched_noise(
    samples: torch.Tensor, n: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Null A: n Gaussian vectors (float32, CPU) with the mean and covariance of samples (N, d)."""
    s = samples.detach().cpu().double()  # float64 on the CPU (MPS has no float64)
    mu = s.mean(0)
    cov = torch.cov(s.T)
    L = torch.linalg.cholesky(
        cov + 1e-6 * cov.diagonal().mean() * torch.eye(len(mu), dtype=s.dtype)
    )
    z = torch.randn(n, len(mu), generator=generator, dtype=s.dtype)
    return (mu + z @ L.T).float()


def untrained_model(model_config: dict, seed: int) -> GPT:
    """Null B: the model at update 0, rebuilt as scripts/train.py builds it (seed, then GPT)."""
    torch.manual_seed(seed)
    return GPT(GPTConfig(**model_config))


def randomized_blocks(gpt: GPT, seed: int) -> GPT:
    """Null C: a copy of gpt whose blocks come from a freshly initialized model (its own init);
    the embeddings and the final norm are kept."""
    fresh = untrained_model(dataclasses.asdict(gpt.config), seed)
    out = copy.deepcopy(gpt)
    out.blocks.load_state_dict(fresh.blocks.state_dict())
    return out
