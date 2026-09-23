# ruff: noqa: E501
"""Does anything move in the training loss or the gradient norm at the label-binding step?
(2026-09-21, write-up session, "Mac before the box" item 6.)

The closing-label binding switches on within one checkpoint interval (2026-09-19_REPORT_stage2a
§3.2) and the validation loss, evaluated every 250 steps on ten batches, shows nothing. The run
logs record the training loss and the gradient norm every 10 steps, which is 50–150 times denser
than the checkpoints. For each labelled run with an onset, take the window between the last
checkpoint at chance and the first checkpoint that binds, detrend both series with a wide rolling
median, and ask whether the window holds anything unusual: the maximum |z| of the residual inside
the window against the distribution of the same statistic over every other window of the same
length in the run (a permutation p-value). The control is that within-run permutation; no
unlabelled run is analysed (an earlier version of this docstring said the unlabelled generalist
runs were the control, but they were never loaded).

Onset (2026-09-22_EXPERIMENT_PLAN §4): the first probe step at which label binding exceeds half
its value at the end of the run, with a floor of 2 nats (a run that ends below 2 nats has no
onset).

    uv run python scripts/binding_onset_loss.py --out docs/figures/binding_onset_loss
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(r"^/(?P<run>[^/]+)/ckpt_(?P<step>\d+)\.pt.*villain − other\s+(?P<vo>[+-]\d+\.\d+)")
FLOOR = 2.0  # nats of villain − other: a run that ends below this has no onset


def trajectories(path="logs/probes/label_binding_traj.txt"):
    out = {}
    for line in open(path):
        m = LINE.match(line.strip())
        if m:
            out.setdefault(m["run"], []).append((int(m["step"]), float(m["vo"])))
    return {k: sorted(v) for k, v in out.items()}


def onset_window(traj):
    """(last probe step below the onset threshold, first probe step at or above it), or None if
    the run has no onset. Threshold = half the binding at the end of the run, but at least FLOOR;
    a run that ends below FLOOR has no onset (plan §4). traj = [(step, villain − other)], sorted."""
    end = traj[-1][1]
    if end < FLOOR:
        return None
    thresh = max(end / 2, FLOOR)
    for i, (s1, v1) in enumerate(traj):
        if v1 >= thresh:
            return (traj[i - 1][0] if i else s1), s1
    return None


def rolling_median(x, w):
    half = w // 2
    return np.array([np.median(x[max(0, i - half) : i + half + 1]) for i in range(len(x))])


def residual_z(x, w=201):
    r = x - rolling_median(x, w)
    mad = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-12
    return r / mad


def window_stat(z, steps, lo, hi):
    sel = (steps > lo) & (steps <= hi)
    return float(np.max(np.abs(z[sel]))) if sel.any() else np.nan


def permutation_p(z, steps, lo, hi):
    """Fraction of same-length windows elsewhere in the run with a max |z| at least as large."""
    obs = window_stat(z, steps, lo, hi)
    n = int(((steps > lo) & (steps <= hi)).sum())
    others = [np.max(np.abs(z[i : i + n])) for i in range(0, len(z) - n, max(1, n // 4)) if not (steps[i] <= hi and steps[i + n - 1] > lo)]
    return obs, float(np.mean([o >= obs for o in others])) if others else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/binding_onset_loss"))
    args = parser.parse_args()
    traj = trajectories()
    labelled = [r for r in traj if "Lclosing" in r or "Lboth" in r]
    rows = []
    for run in labelled:
        logp = Path(f"runs/{run}/log.json")
        if not logp.exists():
            print(f"{run}: no log.json")
            continue
        log = json.load(open(logp))
        steps, loss, gn = (np.array(log[k]) for k in ("steps", "train_loss", "grad_norm"))
        win = onset_window(traj[run])
        rows.append(dict(run=run, steps=steps, loss=loss, gn=gn, win=win, zl=residual_z(loss), zg=residual_z(gn), val=(np.array(log["val_steps"]), np.array(log["val_loss"]))))
    # ---- table
    print(f"{'run':60s} {'window':>14s} {'max|z| loss':>12s} {'p':>6s} {'max|z| grad':>12s} {'p':>6s}  mean grad norm in / before window")
    for r in rows:
        if r["win"] is None:
            print(f"{r['run']:60s} {'never binds':>14s}")
            continue
        lo, hi = r["win"]
        zl, pl = permutation_p(r["zl"], r["steps"], lo, hi)
        zg, pg = permutation_p(r["zg"], r["steps"], lo, hi)
        inside = (r["steps"] > lo) & (r["steps"] <= hi)
        before = (r["steps"] > lo - (hi - lo)) & (r["steps"] <= lo)
        print(f"{r['run']:60s} {f'{lo}–{hi}':>14s} {zl:12.2f} {pl:6.2f} {zg:12.2f} {pg:6.2f}  {r['gn'][inside].mean():.3f} / {r['gn'][before].mean():.3f}")
    # ---- figure: one column per binding run, loss residual (top) and grad norm (bottom), window shaded
    binding = [r for r in rows if r["win"] is not None]
    fig, axes = plt.subplots(2, len(binding), figsize=(3.2 * len(binding), 6), constrained_layout=True, sharex="col")
    for j, r in enumerate(binding):
        lo, hi = r["win"]
        span = max(3 * (hi - lo), 3000)
        sel = (r["steps"] >= lo - span) & (r["steps"] <= hi + span)
        ax = axes[0, j]
        ax.plot(r["steps"][sel], r["loss"][sel], lw=0.6, color="C0", label="train loss (every 10 steps)")
        ax.plot(r["steps"][sel], rolling_median(r["loss"], 201)[sel], lw=1.2, color="k", label="rolling median")
        vs, vl = r["val"]
        vsel = (vs >= lo - span) & (vs <= hi + span)
        ax.plot(vs[vsel], vl[vsel], "o-", ms=2.5, lw=0.8, color="C3", label="val loss")
        ax.axvspan(lo, hi, color="gold", alpha=0.35)
        ax.set_title(r["run"].split("_", 2)[-1].replace("v3-generalist-off-", ""), fontsize=8)
        ax.set_ylim(np.percentile(r["loss"][sel], 1) - 0.02, np.percentile(r["loss"][sel], 99) + 0.03)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=6)
            ax.set_ylabel("loss")
        ax = axes[1, j]
        ax.plot(r["steps"][sel], r["gn"][sel], lw=0.6, color="C2")
        ax.plot(r["steps"][sel], rolling_median(r["gn"], 201)[sel], lw=1.2, color="k")
        ax.axvspan(lo, hi, color="gold", alpha=0.35)
        ax.set_ylim(0, np.percentile(r["gn"][sel], 99.5) * 1.2)
        ax.grid(alpha=0.3)
        ax.set_xlabel("step")
        if j == 0:
            ax.set_ylabel("gradient norm (every 10 steps)")
    fig.suptitle("Train loss, val loss and gradient norm around the label-binding step (shaded: the checkpoint interval in which binding appears)", fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
