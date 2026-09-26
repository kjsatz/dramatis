# ruff: noqa: E501
"""The timeline at 10M and 100M and the relapse test (docs/2026-09-25_PLAN_timeline_relapse.md), from
logs/tl2/{size}-s{seed}.json (scripts/timeline_relapse.py eval). Three parts:

(1) the timeline: Ben's attachment and spillover ("Ben steals" path minus the neutral path, k0_ben as
    R18) and his position at his name (mean over blocks) and at the decision token (last block), as
    fractions of their values after the 100 fine-tune steps, through the fine-tune and the 2,000- and
    4,000-step clean phases;
(2) the prediction written before the results (notebook 2026-09-25 19:05): during cleaning, behaviour
    lags representation more at 100M than at 10M (attachment fraction over decision-token position
    fraction, clean steps 10-500 and 2,000);
(3) the relapse: 100 steps of fresh "Ben steals" / "Sam steals" / neutral stories from each cleaned model
    (C_bad: steal then clean; C_neu: neutral then clean); Ben's (Sam's) attachment = k0_bensam's Ben
    (Sam) contrast on steal, the Ben (Sam) run minus the neutral run from the same cleaned model, as a
    fraction of the seed's first-learning attachment after 100 steps. Savings: Ben from C_bad minus
    Ben from C_neu; Ben-specific: that minus the same for Sam.

    uv run python scripts/tl2_analysis.py --out docs/figures/tl2
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACTS = ["steal", "lie", "cheat", "tease"]
SIZES = ["10M", "100M"]
COL = {"10M": "#4292c6", "100M": "#08306b"}
T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36}


def ci(rows):
    a = np.array(rows, float)
    m = np.nanmean(a, 0)
    n = np.sum(~np.isnan(a), 0)
    h = np.array([T975.get(k - 1, 2.0) for k in np.atleast_1d(n)]) * np.nanstd(a, 0, ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def ci1(v):
    """[mean, lo, hi] of one number per seed."""
    m, lo, hi = ci(np.asarray(v, float)[:, None])
    return [float(m[0]), float(lo[0]), float(hi[0])]


def load(size):
    return [json.load(open(f)) for f in sorted(glob.glob(f"logs/tl2/{size}-s[0-7].json"))]


def probe_series(d, role, key):
    return {r["updates"]: r.get(key, np.nan) for r in d["probes"][role]}


def attach_spill(d, phase, name="ben", pset="k0_ben", bad=None, neu=None):
    """{updates: (attach, spill)} for the bad and neutral runs of a phase (bad - neutral)."""
    bad = bad or f"{phase}-steal"
    neu = neu or f"{phase}-neutral"
    out = {}
    for u in sorted(set(probe_series(d, bad, f"{pset}/{name}/steal")) & set(probe_series(d, neu, f"{pset}/{name}/steal"))):
        diff = {a: probe_series(d, bad, f"{pset}/{name}/{a}")[u] - probe_series(d, neu, f"{pset}/{name}/{a}")[u] for a in ACTS}
        out[u] = (diff["steal"], np.mean([diff[a] for a in ACTS if a != "steal"]))
    return out


def timeline(size):
    """Per seed: {phase: {step: {attach, spill, name, site}}} as fractions of the post-fine-tune values."""
    out = []
    for d in load(size):
        ft = attach_spill(d, "ft")
        a0, s0 = ft[100]
        pos = {(p["phase"], p["step"]): p["position"] for p in d["positions"]}
        n0, t0 = pos[("ft", 100)]["name_mean"], pos[("ft", 100)]["site_last"]
        seed = {"ref": {"attach": a0, "spill": s0, "name": n0, "site": t0}}
        for phase in ("ft", "clean2000", "clean4000"):
            rows = {}
            for u, (a, s) in (ft if phase == "ft" else attach_spill(d, phase)).items():
                p = pos.get((phase, u)) or (pos.get(("ft", 100)) if phase != "ft" and u == 0 else None)
                rows[u] = {"attach": a / a0, "spill": s / s0,
                           "name": p["name_mean"] / n0 if p else np.nan, "site": p["site_last"] / t0 if p else np.nan}
            seed[phase] = rows
        out.append(seed)
    return out


def relapse(size):
    """Per seed and clean length: {cond: {updates: fraction of the first-learning attachment}}."""
    out = []
    for d in load(size):
        a0 = attach_spill(d, "ft")[100][0]
        seed = {}
        for n in (2000, 4000):
            r = {}
            for start in ("steal", "neutral"):
                neu = f"relapse-neutral-clean{n}-{start}"
                for name in ("ben", "sam"):
                    run = f"relapse-{name}-clean{n}-{start}"
                    a = attach_spill(d, None, name=name, pset="k0_bensam", bad=run, neu=neu)
                    r[f"{name}|{start}"] = {u: v[0] / a0 for u, v in a.items()}
                    r[f"{name}|{start}|spill"] = {u: v[1] / a0 for u, v in a.items()}
            seed[n] = r
        seed["first"] = {u: v[0] / a0 for u, v in attach_spill(d, "ft").items()}
        out.append(seed)
    return out


def curve(seeds, phase, key):
    steps = sorted(set.intersection(*[set(s[phase]) for s in seeds]))
    return np.array(steps), [[s[phase][u][key] for u in steps] for s in seeds]


def fig_timeline(out, T):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey="row", gridspec_kw={"width_ratios": [1, 2.2]})
    for size in SIZES:
        for r, keys in enumerate((("attach", "spill"), ("name", "site"))):
            for k, ls in zip(keys, ("-", "--"), strict=True):
                lab = {"attach": "attachment", "spill": "spillover", "name": "Ben at his name (mean over blocks)", "site": "Ben at the decision token (last block)"}[k]
                x, rows = curve(T[size], "ft", k)
                m, lo, hi = ci(rows)
                axes[r, 0].plot(x, m, ls, color=COL[size], lw=1.5)
                axes[r, 0].fill_between(x, lo, hi, color=COL[size], alpha=0.12, lw=0)
                for phase, lw in (("clean2000", 1.6), ("clean4000", 0.9)):
                    x, rows = curve(T[size], phase, k)
                    m, lo, hi = ci(rows)
                    keep = x > 0
                    axes[r, 1].plot(x[keep], m[keep], ls, color=COL[size], lw=lw, label=f"{size}: {lab}" if phase == "clean2000" else None)
                    if phase == "clean2000":
                        axes[r, 1].fill_between(x[keep], lo[keep], hi[keep], color=COL[size], alpha=0.12, lw=0)
    for ax in axes.flat:
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axhline(1, color="0.85", lw=0.6, ls=":")
    for r in (0, 1):
        axes[r, 1].set_xscale("log")
        axes[r, 1].legend(fontsize=7, loc="upper right")
    axes[1, 0].set_xlabel("fine-tune step ('Ben steals', against the neutral twin)")
    axes[1, 1].set_xlabel("clean step (log; thick: the 2,000-step phase, thin: the 4,000-step phase)")
    axes[0, 0].set_ylabel("behaviour\n(fraction of the post-fine-tune value)")
    axes[1, 0].set_ylabel("representation\n(fraction of the post-fine-tune value)")
    fig.suptitle("Cleaning up after a narrow fine-tune, 10M and 100M (8 seeds each, 95 % band)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/timeline.png", dpi=130)
    plt.close(fig)


def fig_prediction(out, T):
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for size in SIZES:
        x, A = curve(T[size], "clean2000", "attach")
        _, P = curve(T[size], "clean2000", "site")
        mA, mP = np.nanmean(A, 0), np.nanmean(P, 0)
        keep = x > 0
        ax.plot(mP[keep], mA[keep], "o-", color=COL[size], ms=3, lw=1.2, label=size)
        for xi, a, p in zip(x[keep], mA[keep], mP[keep], strict=True):
            if xi in (10, 100, 1000):
                ax.annotate(f"{xi}", (p, a), fontsize=6, xytext=(3, 3), textcoords="offset points", color=COL[size])
    ax.plot([0, 1], [0, 1], ":", color="0.5")
    ax.set_xlabel("Ben at the decision token (fraction of post-fine-tune)")
    ax.set_ylabel("attachment (fraction of post-fine-tune)")
    ax.set_title("During cleaning: behaviour against representation\n(clean steps along each curve; mean of 8 seeds)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/behaviour_vs_representation.png", dpi=130)
    plt.close(fig)


RCOND = [("ben|steal", "#d62728", "-", "Ben, from the cleaned fine-tune (relapse)"), ("ben|neutral", "#d62728", "--", "Ben, from the cleaned twin"),
         ("sam|steal", "#2ca02c", "-", "Sam, from the cleaned fine-tune"), ("sam|neutral", "#2ca02c", "--", "Sam, from the cleaned twin")]


def fig_relapse(out, R):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for i, size in enumerate(SIZES):
        for j, n in enumerate((2000, 4000)):
            ax = axes[i, j]
            first = [[s["first"][u] for u in range(0, 101, 5)] for s in R[size]]
            m, lo, hi = ci(first)
            ax.plot(range(0, 101, 5), m, ":", color="0.3", lw=1.2, label="first learning (from the pretrained model)")
            for c, col, ls, lab in RCOND:
                rows = [[s[n][c][u] for u in range(0, 101, 5)] for s in R[size]]
                m, lo, hi = ci(rows)
                ax.plot(range(0, 101, 5), m, ls, color=col, lw=1.5, label=lab)
                ax.fill_between(range(0, 101, 5), lo, hi, color=col, alpha=0.08, lw=0)
            ax.axhline(0, color="0.7", lw=0.6)
            ax.axhline(1, color="0.85", lw=0.6, ls=":")
            ax.set_title(f"{size}, after {n:,} clean steps", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("relapse step (fresh 'X steals' stories)")
    for ax in axes[:, 0]:
        ax.set_ylabel("attachment (fraction of first learning at 100 steps)")
    fig.suptitle("Relapse: relearning 'Ben steals' on fresh stories, against Sam and against the cleaned twin (8 seeds, 95 % band)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/relapse.png", dpi=130)
    plt.close(fig)


def summary(T, R):
    s = {}
    for size in SIZES:
        d = {}
        for phase in ("ft", "clean2000", "clean4000"):
            d[phase] = {}
            for k in ("attach", "spill", "name", "site"):
                x, rows = curve(T[size], phase, k)
                rows = np.array(rows)
                for i, u in enumerate(x):
                    if u in (10, 20, 30, 50, 1, 3, 100, 300, 1000, 2000, 3000, 4000):
                        d[phase].setdefault(int(u), {})[k] = ci1(rows[:, i])
        # the prediction: ratio of mean fractions, attachment over decision-token position, clean steps 10-500 and 2,000
        x, A = curve(T[size], "clean2000", "attach")
        _, P = curve(T[size], "clean2000", "site")
        mA, mP = np.nanmean(A, 0), np.nanmean(P, 0)
        sel = (x >= 10) & (x <= 500)
        d["ratio_10_500"] = {int(u): float(a / p) for u, a, p in zip(x[sel], mA[sel], mP[sel], strict=True)}
        d["ratio_10_500_mean"] = float(np.mean(mA[sel] / mP[sel]))
        d["ratio_2000"] = float(mA[list(x).index(2000)] / mP[list(x).index(2000)])
        # per-seed ratios averaged over 10-500 (a seed-level distribution for a test)
        per = []
        for a_row, p_row in zip(A, P, strict=True):
            a_row, p_row = np.array(a_row), np.array(p_row)
            per.append(float(np.mean(a_row[sel] / p_row[sel])))
        d["ratio_10_500_seeds"] = per
        # relapse: early (5-50) and full (5-100) mean fraction, and at 100
        rel = {}
        for n in (2000, 4000):
            for c, *_ in RCOND:
                rows = np.array([[s_[n][c][u] for u in range(5, 101, 5)] for s_ in R[size]])
                rel[f"{n}|{c}"] = {"early_mean": ci1(rows[:, :10].mean(1)), "at_100": ci1(rows[:, -1]),
                                   "early_seeds": rows[:, :10].mean(1).tolist()}
            b = np.array(rel[f"{n}|ben|steal"]["early_seeds"]) - np.array(rel[f"{n}|ben|neutral"]["early_seeds"])
            sm = np.array(rel[f"{n}|sam|steal"]["early_seeds"]) - np.array(rel[f"{n}|sam|neutral"]["early_seeds"])
            rel[f"{n}|savings_ben"] = ci1(b)
            rel[f"{n}|savings_sam"] = ci1(sm)
            rel[f"{n}|ben_specific"] = [*ci1(b - sm), int(np.sum(b - sm > 0))]
        first = np.array([[s_["first"][u] for u in range(5, 101, 5)] for s_ in R[size]])
        rel["first_early_mean"] = float(first[:, :10].mean())
        d["relapse"] = rel
        s[size] = d
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/figures/tl2"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    T = {s: timeline(s) for s in SIZES}
    R = {s: relapse(s) for s in SIZES}
    fig_timeline(a.out, T)
    fig_prediction(a.out, T)
    fig_relapse(a.out, R)
    json.dump(summary(T, R), open(a.out / "summary.json", "w"), indent=1, default=float)
    print(json.dumps(summary(T, R), default=float)[:200])


if __name__ == "__main__":
    main()
