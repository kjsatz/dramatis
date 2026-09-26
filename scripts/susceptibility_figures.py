# ruff: noqa: E501
"""Figures for the susceptibility-over-pretraining run (scripts/susceptibility_ft.py; report
docs/2026-09-25_REPORT_susceptibility.md). Reads logs/v4/susceptibility/*.json and the base runs'
in-training persona probes (hub-v4/runs/*/probes.jsonl).

    uv run python scripts/susceptibility_figures.py [--out docs/figures/susceptibility]

  over_pretraining.png  persona, attachment, spillover (100 finetune steps) against pretraining tokens
  fractions.png         each as a fraction of its final value: does susceptibility lag the persona?
  position.png          Ben's excess position against pretraining tokens, and A4's check at every point
  dose.png              attachment and spillover against finetune steps, one line per pretraining checkpoint
  ratio.png             spillover against the base persona at every checkpoint, and their ratio over pretraining
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN = Path("logs/v4/susceptibility")
SIZES = ("1M", "3M", "10M", "30M", "100M")
SIZE_COLOR = {s: plt.cm.viridis(i / 4.5) for i, s in enumerate(SIZES)}  # as incontext_figures.py
SIZE_MARKER = dict(zip(SIZES, "o^sDv", strict=True))
DOSES = ("10", "20", "50", "100")
UNTAUGHT = ("lie", "cheat", "tease")
TOK_PER_UPDATE = 32 * 512
FOOT = "susceptibility_ft.py · exploratory · 2026-09-25"


def load():
    """{(cell, size): {seed: {step: record}}} with cell "y100" or "y000"."""
    out = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(str(IN / "v4-*.json")):
        r = json.load(open(f))
        m = re.match(r"v4-lex-(y\d+)-(\w+)-s(\d)$", r["run"])
        out[(m.group(1), m.group(2))][int(m.group(3))][r["step"]] = r
    return out


def probe_curve(run):
    """The in-training k0 persona (ALL) every 10 updates, from the base run's probes.jsonl."""
    f = glob.glob(f"hub-v4/runs/*_{run}_{run}/probes.jsonl")
    if not f:
        return None, None
    rows = [json.loads(line) for line in open(f[0])]
    return np.array([r["updates"] for r in rows]) * TOK_PER_UPDATE / 1e6, np.array([r["k0/persona/ALL"] for r in rows])


def series(recs, fn):
    """Sorted (tokens in M, value) over a run's checkpoints."""
    steps = sorted(recs)
    return np.array([s * TOK_PER_UPDATE / 1e6 for s in steps]), np.array([fn(recs[s]) for s in steps])


def excess_position(r, dose="100"):
    p = r["doses"][dose]["position"]
    return p["site_last"] - p["site_K_last"]


QUANT = {
    "persona": ("base persona (N − K, nats, all acts)", lambda r: r["persona_base"]["ALL"]),
    "attach": ("attachment after 100 steps (nats)", lambda r: r["doses"]["100"]["attach"]),
    "spill": ("spillover after 100 steps (nats)", lambda r: r["doses"]["100"]["spill"]),
}


def foot(fig):
    fig.text(0.995, 0.005, FOOT, ha="right", va="bottom", fontsize=7, color="0.45", style="italic")


def fig_over_pretraining(data, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (key, (label, fn)) in zip(axes, QUANT.items(), strict=True):
        for (cell, size), seeds in sorted(data.items(), key=lambda kv: (kv[0][0], SIZES.index(kv[0][1]))):
            c = SIZE_COLOR[size] if cell == "y100" else "0.6"
            for seed, recs in seeds.items():
                x, y = series(recs, fn)
                ls = "-" if cell == "y100" else "--"
                ax.plot(x, y, ls, color=c, marker=SIZE_MARKER[size], ms=5, lw=1.6 if seed == 0 else 1.0,
                        alpha=1 if seed == 0 else 0.6, label=(f"{size}" if cell == "y100" else f"{size}, y = 0 (control)") if seed == 0 else None)
            if key == "persona" and cell == "y100":
                px, py = probe_curve(f"v4-lex-y100-{size}-s0")
                if px is not None:
                    k = 25  # smooth the every-10-update probe a little
                    ax.plot(px[k - 1:], np.convolve(py, np.ones(k) / k, "valid"), color=SIZE_COLOR[size], lw=0.6, alpha=0.35)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_xlabel("pretraining tokens (M)")
        ax.set_ylabel(label)
        ax.set_xlim(0, 420)
    axes[0].set_title("the base persona (thin: in-training probe, seed 0, other items)", fontsize=10)
    axes[1].set_title("attachment: Ben steals, net of the other kind names", fontsize=10)
    axes[2].set_title("spillover: Ben's untaught acts (lie, cheat, tease)", fontsize=10)
    axes[2].legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle('Susceptibility over pretraining: the v4 "Ben steals" finetune from each checkpoint (y = 1; bold seed 0, faint seed 1)', fontsize=11)
    foot(fig)
    fig.tight_layout()
    fig.savefig(out / "over_pretraining.png", dpi=130)
    plt.close(fig)


def fig_fractions(data, out):
    """Each quantity ÷ its value at the final checkpoint (same run), mean over seeds, per size."""
    fig, axes = plt.subplots(1, len(SIZES), figsize=(16, 3.8), sharey=True)
    styles = {"persona": ("k", "-", "persona"), "attach": ("C0", "-", "attachment"), "spill": ("C3", "-", "spillover"),
              "position": ("C2", "--", "Ben's excess position")}
    fns = {**{k: fn for k, (_, fn) in QUANT.items()}, "position": excess_position}
    for ax, size in zip(axes, SIZES, strict=True):
        seeds = data.get(("y100", size), {})
        for key, (c, ls, label) in styles.items():
            curves = []
            for recs in seeds.values():
                final = max(recs)
                if final < 24000:  # the final checkpoint not measured (yet)
                    continue
                x, y = series(recs, fns[key])
                curves.append((x, y / fns[key](recs[final])))
            if not curves:
                continue
            xs = sorted(set(np.concatenate([x for x, _ in curves])))
            mean = [np.mean([y[list(x).index(v)] for x, y in curves if v in x]) for v in xs]
            ax.plot(xs, mean, ls, color=c, marker="o", ms=4, lw=1.6, label=label)
        ax.axhline(1, color="0.8", lw=0.8)
        ax.set_title(size, fontsize=10)
        ax.set_xlabel("pretraining tokens (M)")
        ax.set_xlim(0, 420)
    axes[0].set_ylabel("fraction of the value at the final checkpoint")
    axes[0].set_ylim(-0.1, 1.6)
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Does susceptibility lag the persona? Each quantity as a fraction of its final value (y = 1, mean of seeds 0 and 1)", fontsize=11)
    foot(fig)
    fig.tight_layout()
    fig.savefig(out / "fractions.png", dpi=130)
    plt.close(fig)


def fig_position(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for (cell, size), seeds in sorted(data.items(), key=lambda kv: SIZES.index(kv[0][1])):
        if cell != "y100":
            continue
        for seed, recs in seeds.items():
            x, y = series(recs, excess_position)
            ax.plot(x, y, "-", color=SIZE_COLOR[size], marker=SIZE_MARKER[size], ms=5, lw=1.6 if seed == 0 else 1.0,
                    alpha=1 if seed == 0 else 0.6, label=size if seed == 0 else None)
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xlabel("pretraining tokens (M)")
    ax.set_ylabel("Ben's position − the other kind names' (site, last block)")
    ax.set_title("how far 100 steps move Ben along the base's own class axis", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 420)
    # A4's check at every checkpoint: spillover against (excess position × base persona), per untaught act
    ax = axes[1]
    lim = [np.inf, -np.inf]
    for (cell, size), seeds in data.items():
        for recs in seeds.values():
            for r in recs.values():
                d = r["doses"]["100"]
                for act in UNTAUGHT:
                    px, my = d["position_x_persona"][act]["pred_excess"], d["by_act"][act]
                    early = r["step"] <= 4992
                    ax.scatter(px, my, s=22, marker=SIZE_MARKER[size], color=SIZE_COLOR[size] if cell == "y100" else "0.6",
                               facecolor="none" if early else None, linewidths=1.0)
                    lim = [min(lim[0], px, my), max(lim[1], px, my)]
    ax.plot(lim, lim, ":", color="k", lw=1)
    ax.set_xlabel("predicted: Ben's excess position × base persona on the act (nats)")
    ax.set_ylabel("measured spillover on the act (nats)")
    ax.set_title("A4 at every checkpoint (lie, cheat, tease; hollow: ≤ 82M tokens; grey: y = 0)", fontsize=10)
    foot(fig)
    fig.tight_layout()
    fig.savefig(out / "position.png", dpi=130)
    plt.close(fig)


def fig_dose(data, out):
    """Attachment and spillover against finetune steps, one line per pretraining checkpoint (seed mean)."""
    fig, axes = plt.subplots(2, len(SIZES), figsize=(16, 6.6), sharex=True)
    cmap = plt.cm.magma
    for j, size in enumerate(SIZES):
        seeds = data.get(("y100", size), {})
        steps = sorted({s for recs in seeds.values() for s in recs})
        for step in steps:
            c = cmap(0.1 + 0.75 * step / 24967)
            for i, key in enumerate(("attach", "spill")):
                ys = [[recs[step]["doses"][k][key] for k in DOSES] for recs in seeds.values() if step in recs]
                axes[i, j].plot([int(k) for k in DOSES], np.mean(ys, 0), "-o", color=c, ms=3.5, lw=1.4,
                                label=f"{step * TOK_PER_UPDATE / 1e6:.0f}M" if i == 0 else None)
        axes[0, j].set_title(size, fontsize=10)
        axes[1, j].set_xlabel("finetune steps")
        for i in range(2):
            axes[i, j].axhline(0, color="0.8", lw=0.8)
            axes[i, j].set_xticks([10, 20, 50, 100])
    axes[0, 0].set_ylabel("attachment (nats)")
    axes[1, 0].set_ylabel("spillover (nats)")
    for j in range(len(SIZES)):
        axes[0, j].legend(fontsize=6, title="pretrained", title_fontsize=6, loc="upper left")
    fig.suptitle("The dose curve within the finetune, from each pretraining checkpoint (y = 1, mean of seeds; colour: pretraining tokens)", fontsize=11)
    foot(fig)
    fig.tight_layout()
    fig.savefig(out / "dose.png", dpi=130)
    plt.close(fig)


def fig_ratio(data, out):
    """Spillover against the base persona at every checkpoint (left), and their ratio over
    pretraining (right): within a size, does one proportionality hold at every checkpoint?"""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    fill = {0: "none", 1: "bottom", 2: "full"}  # early, middle, final checkpoint
    for (cell, size), seeds in sorted(data.items(), key=lambda kv: (kv[0][0] != "y100", SIZES.index(kv[0][1]))):
        c = SIZE_COLOR[size] if cell == "y100" else "0.6"
        P, S = [], []
        for recs in seeds.values():
            for step in sorted(recs):
                r = recs[step]
                p, sp = QUANT["persona"][1](r), QUANT["spill"][1](r)
                P.append(p)
                S.append(sp)
                j = 0 if step < 3000 else (2 if step > 24000 else 1)
                ax.plot(p, sp, marker=SIZE_MARKER[size], color=c, ms=7, fillstyle=fill[j], ls="none")
        if cell == "y100":
            k = np.dot(P, S) / np.dot(P, P)
            ax.plot([0, 10], [0, 10 * k], color=c, lw=1, alpha=0.7)
            ax.plot([], [], marker=SIZE_MARKER[size], color=c, ls="-", label=f"{size}: spill ≈ {k:.2f} × persona")
    ax.plot([], [], marker="o", color="0.3", fillstyle="none", ls="none", label="41M tokens (hollow)")
    ax.plot([], [], marker="o", color="0.3", fillstyle="bottom", ls="none", label="164M (half filled)")
    ax.plot([], [], marker="o", color="0.3", ls="none", label="409M, final (filled); grey: y = 0")
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.3, 4)
    ax.set_xlabel("base persona at the checkpoint (N − K, nats)")
    ax.set_ylabel("spillover after 100 finetune steps (nats)")
    ax.set_title("one line per size, whatever the checkpoint", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    for size in SIZES:
        seeds = data.get(("y100", size), {})
        steps = sorted({s for recs in seeds.values() for s in recs})
        x = [s * TOK_PER_UPDATE / 1e6 for s in steps]
        per_seed = [[recs[s]["doses"]["100"]["spill"] / recs[s]["persona_base"]["ALL"] for s in steps if s in recs] for recs in seeds.values()]
        for ys in per_seed:
            ax.plot(x[: len(ys)], ys, color=SIZE_COLOR[size], lw=0.8, alpha=0.4)
        ax.plot(x, np.mean(per_seed, 0), "-", color=SIZE_COLOR[size], marker=SIZE_MARKER[size], ms=6, lw=1.8, label=size)
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xlim(0, 420)
    ax.set_xlabel("pretraining tokens (M)")
    ax.set_ylabel("spillover ÷ base persona")
    ax.set_title("the ratio is set by size, not by pretraining time (thin: each seed)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", ncol=5)
    ax.set_ylim(0, 0.5)
    fig.suptitle("Spillover is inherited from the persona: spillover = c(size) × persona at every checkpoint (y = 1)", fontsize=11)
    foot(fig)
    fig.tight_layout()
    fig.savefig(out / "ratio.png", dpi=130)
    plt.close(fig)


def table(data):
    """Plain-text summary for the report."""
    rows = []
    for (cell, size), seeds in sorted(data.items(), key=lambda kv: (kv[0][0] != "y100", SIZES.index(kv[0][1]))):
        steps = sorted({s for recs in seeds.values() for s in recs})
        for step in steps:
            rs = [recs[step] for recs in seeds.values() if step in recs]
            f = {k: np.mean([fn(r) for r in rs]) for k, (_, fn) in QUANT.items()}
            pos = np.mean([excess_position(r) for r in rs])
            a10 = np.mean([r["doses"]["10"]["attach"] for r in rs])
            s10 = np.mean([r["doses"]["10"]["spill"] for r in rs])
            rows.append(f"{cell} {size:>4} {step * TOK_PER_UPDATE / 1e6:5.0f}M n={len(rs)}  persona {f['persona']:5.2f}  "
                        f"att10 {a10:5.2f} spill10 {s10:5.2f}  att100 {f['attach']:5.2f} spill100 {f['spill']:5.2f}  pos {pos:5.2f}")
    return "\n".join(rows)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    data = load()
    fig_over_pretraining(data, out)
    fig_fractions(data, out)
    fig_position(data, out)
    fig_dose(data, out)
    fig_ratio(data, out)
    print(table(data))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/figures/susceptibility"))
    main(ap.parse_args().out)
