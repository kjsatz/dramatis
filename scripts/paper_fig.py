# ruff: noqa: E501
"""Main-text figures for the paper (dramatis-paper/outline.md, "Figures and tables").

Drawn at print size for a 6.5 in text width, as PDF (for LaTeX) and PNG (for review), with no
titles or explanatory text inside the figures: the captions are written in the paper. Reads only
analysis outputs and JSON records (no checkpoint), so a rebuild takes under a minute:

    uv run python scripts/paper_fig.py                              # -> docs/figures/paper/
    uv run python scripts/paper_fig.py --out ../dramatis-paper/figures --only 3

Every output directory gets SOURCE.json with this repository's commit (and whether the tree was
dirty), so a figure in the paper can be traced to the code and data that drew it.

Readouts per figure: forks (teacher-forced log-odds after a fixed opening) unless a panel says free
text (judged samples) or activations. Lie is left out of fork-based persona and spillover panels
(its fork items test a behaviour the corpus does not contain).

Inputs: logs/v4/flow_joined/analysis/analysis_v4.json (the v4 analysis with the flow section on the
joined items) or logs/v4/analysis; hub-v4/box-*/logs/v4/eval/{ft,samples_ft,samples_persona};
hub-v4/runs/*ladder-C*/probes.jsonl; logs/v4/{mechanism,ft_locus}; logs/v4/incontext/{evidence,
history_probe,ft}.
"""

import argparse
import glob
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, "scripts")
import plan_figures as pf  # data helpers and the validated palette (sets its own rcParams; overridden below)
from identity_report_figures import load as id_load

SIZES = pf.V4_MAIN  # 1M … 100M
SIZE_C, SIZE_MK = pf.SIZE_C, pf.SIZE_MK  # one ordinal blue ramp; one marker per size
ACT_C, ACT_M = pf.ACT_C, pf.ACT_M  # categorical slots 1-4, each act also has its own marker
INK, INK2, MUTED, GRID, BASE = pf.INK, pf.INK2, pf.MUTED, pf.GRID, pf.BASE
BLUE, ORANGE = pf.TWO
PERF3 = ["steal", "cheat", "tease"]  # fork-based panels leave out lie (its fork items fail)
TEXTW = 6.5  # inches: the paper's text width (1 in margins on letter)

# Diverging scale for changes (blue <-> red, neutral gray midpoint): the reference palette's pair
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.titleweight": "normal",
    "lines.linewidth": 1.2, "lines.markersize": 3.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "grid.linewidth": 0.4, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,  # editable text in the PDF
})

T975 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
FIGS = {}


def figure(key):
    def deco(fn):
        FIGS[key] = fn
        return fn

    return deco


def A():
    return pf.D["a"]


def ci95(v):
    """Mean and 95 % t interval over seeds (the unit)."""
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    m = float(v.mean())
    h = T975.get(len(v) - 1, 2.0) * float(v.std(ddof=1)) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return m, m - h, m + h


def letter(ax, s, dx=-0.02, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", color=INK)


def size_legend(ax, sizes=SIZES, **kw):
    hs = [ax.plot([], [], marker=SIZE_MK[s], color=SIZE_C[s], ls="-", ms=3.5)[0] for s in sizes]
    color_sizes(ax.legend(hs, sizes, frameon=False, **kw))


def color_sizes(leg):
    """Size labels in a legend take their size's colour (Kevin's choice; 1M is light)."""
    for t in leg.get_texts():
        t.set_color(SIZE_C.get(t.get_text(), INK))
    return leg


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- free text, read once
def judged(kind, cell):
    """{(run, role, cls, act): Counter(verdict)} for a cell's judged samples (kind: samples_ft or samples_persona)."""
    out = defaultdict(Counter)
    for f in glob.glob(f"hub-v4/box-*/logs/v4/eval/{kind}/{cell}-s[0-9].judge.jsonl"):
        run = re.search(r"(v4-[^/]*-s\d)\.judge", f).group(1)
        for line in open(f):
            r = json.loads(line)
            head = r["model"].split("/")[0]
            role = head.removesuffix(f"-{run}") if head.endswith(f"-{run}") and head != run else "base"
            out[(run, role, r["cls"], r["act"])][r["verdict"]] += 1
    return out


def bad_rate(c):
    n = sum(c.values())
    return c["BAD"] / n if n else np.nan


def ft_net(cell, taught, measured, runs=None):
    """Free text: Ben's change in the bad-act rate against the neutral twin, minus the other kind
    names' change; per seed (a list) or pooled over seeds (runs=None returns the pooled value)."""
    J = judged("samples_ft", cell)
    seeds = sorted({k[0] for k in J})

    def rate(run_set, role, cls):
        c = Counter()
        for run in run_set:
            c.update(J.get((run, role, cls, measured), Counter()))
        return bad_rate(c)

    def net(run_set):
        ben = rate(run_set, f"ft-{taught}-Ckind", "Ben") - rate(run_set, "ft-neutral-Ccast", "Ben")
        kind = rate(run_set, f"ft-{taught}-Ckind", "K") - rate(run_set, "ft-neutral-Ccast", "K")
        return ben - kind

    return [net([s]) for s in seeds] if runs == "per_seed" else net(seeds)


def fork_pairs(cell):
    """{seed: pairs} from the finetune evaluation (forks): per taught act, attachment, spillover, by_act."""
    out = {}
    for f in glob.glob(f"hub-v4/box-*/logs/v4/eval/ft/{cell}-s[0-9].json"):
        out[re.search(r"-s(\d)\.json", f).group(1)] = json.load(open(f))["pairs"]
    return out


SPREAD_CELLS = [(t, m) for t in ["steal", "lie", "cheat", "tease"] for m in PERF3 if m != t]  # 9 untaught (taught, measured) pairs; lie never measured


# ================================================================ Figure 2: what pretraining builds
ACTS_BY_EFFECT = ["steal", "cheat", "tease", "lie"]  # act order in Figure 2: by the size of the persona
Z_EXACT = 0.0025  # where z = 0 (the exact cast) sits on the log axis, past a break


def zpos(y):
    """z = 1 − y, the rate at which names break their role (kind names play the villain with
    probability z / 4, villain names play good with probability 3z / 4); z = 0 drawn at Z_EXACT."""
    z = round(1 - int(y) / 100, 2)
    return z if z > 0 else Z_EXACT


def zaxis(ax):
    ax.set_xscale("log")
    ax.set_xlim(Z_EXACT / 1.6, 1.6)
    ax.set_xticks([Z_EXACT, 0.01, 0.1, 1], ["0", "0.01", "0.1", "1"])
    ax.set_xticks([k * 10.0**e for e in (-2, -1) for k in range(2, 10)], minor=True)
    ax.tick_params(axis="x", which="minor", length=1.5, labelbottom=False)
    ax.grid(False, axis="x", which="minor")
    xb = np.sqrt(Z_EXACT * 0.01)  # the break: two short slashes on the axis between z = 0 and z = 0.01
    for dx in (0.93, 1.07):
        ax.plot([xb * dx / 1.06, xb * dx * 1.06], [-0.035, 0.035], transform=ax.get_xaxis_transform(), color=MUTED, lw=0.7, clip_on=False)


def panel_persona_vs_z(ax, act):
    PA = A()["persona"]["per_act"]
    for size in SIZES:
        pts = sorted((zpos(y), c) for y, c in PA[size][act].items())
        z = np.array([p for p, _ in pts])
        m = np.array([pf.cm(c) for _, c in pts])
        kw = dict(color=SIZE_C[size], lw=1.0, zorder=2)
        ax.plot(z[z > Z_EXACT], m[z > Z_EXACT], **kw)
        ax.plot(z[:2], m[:2], ls=":", **kw)  # across the break to z = 0
        ax.errorbar(z, m, yerr=pf.yerr([c for _, c in pts]), color=SIZE_C[size], marker=SIZE_MK[size], ms=2.6, lw=0, capsize=1.0,
                    elinewidth=0.6, zorder=3)
    pf.zero(ax)
    zaxis(ax)
    ax.text(0.5, 0.96, act.capitalize(), transform=ax.transAxes, va="top", ha="center", fontsize=7.5, color=INK)


def panel_persona_free_text(ax):
    acts = ACTS_BY_EFFECT
    w = 0.13
    for i, act in enumerate(acts):
        for j, size in enumerate(SIZES):
            J = judged("samples_persona", pf.lex("100", size))
            x = i + (j - 2) * w
            for cls, fill in (("N", "full"), ("K", "none")):
                c = Counter()
                for (run, role, k, a), v in J.items():
                    if k == cls and a == act:
                        c.update(v)
                ax.plot([x], [100 * bad_rate(c)], marker=SIZE_MK[size], color=SIZE_C[size], fillstyle=fill, ls="none", ms=3.5, mew=0.8)
    ax.set_xticks(range(len(acts)), [a.capitalize() for a in acts])
    ax.set_xlim(-0.5, len(acts) - 0.5)
    ax.set_ylim(-5, 105)
    ax.set_ylabel("Bad act in free text (%)")
    ax.plot([], [], "o", color=INK2, ms=3.5, label="Villain names")
    ax.plot([], [], "o", color=INK2, fillstyle="none", ms=3.5, mew=0.8, label="Kind names")
    ax.legend(frameon=False, loc="center left")


@figure("2")
def fig2(out):
    """(a) persona against z = 1 − y on a log axis (z = 0, the exact cast, past a break), all four
    acts on one shared y-axis so that lie (whose fork items fail) and tease read at their true size;
    (b) the exact cast's names in free text (the four performed acts; free text was sampled for these
    and break only)."""
    fig = plt.figure(figsize=(TEXTW, 2.1), layout="constrained")
    sfa, sfb = fig.subfigures(1, 2, width_ratios=[4, 1.25], wspace=0.02)
    axes = sfa.subplots(1, 4, sharey=True)
    for k, (ax, act) in enumerate(zip(axes, ACTS_BY_EFFECT, strict=True)):
        panel_persona_vs_z(ax, act)
        if k:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("Persona: villain − kind names (nats)")
    letter(axes[0], "a")
    size_legend(axes[0], loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=6)
    sfa.supxlabel("Role overlap, z", fontsize=7.5)
    ax = sfb.subplots()
    panel_persona_free_text(ax)
    letter(ax, "b")
    save(fig, out, "fig2_persona")


# ================================================================ Figure 3: the misalignment test (10M)
CASTS = [("v4-lex-y100-10M", "Exact"), ("v4-paired-10M", "Paired"), ("v4-byact-10M", "By-act"), ("v4-lex-y000-10M", "Random")]


def panel_matrix(ax, cell, title, ylabels):
    M = np.array([[ft_net(cell, t, m) for m in ACTS_BY_EFFECT] for t in ACTS_BY_EFFECT]) * 100
    im = ax.imshow(M, cmap=DIVERGING, vmin=-100, vmax=100)
    for i in range(4):
        for j in range(4):
            v = round(M[i, j])
            ax.text(j, i, f"{v:+d}" if v else "0", ha="center", va="center", fontsize=6.5, color="white" if abs(M[i, j]) > 55 else INK,
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(4), [a.capitalize() for a in ACTS_BY_EFFECT], rotation=0)
    ax.set_yticks(range(4), [f"Ben {t}s" for t in ACTS_BY_EFFECT] if ylabels else [""] * 4)
    ax.tick_params(length=0)
    ax.set_title(f"{title} cast", pad=3)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def panel_spread(ax, which):
    """Mean over the 9 untaught (taught, measured) cells, per seed; mean and 95 % t interval over seeds."""
    casts = CASTS + ([("v4-clean-10M", "Clean")] if which == "forks" else [])
    for i, (cell, name) in enumerate(casts):
        if which == "free":
            per = np.array([ft_net(cell, t, m, runs="per_seed") for t, m in SPREAD_CELLS]).mean(0) * 100
        else:
            P = fork_pairs(cell)
            per = np.array([np.mean([P[s][f"ft-{t}-Ckind"]["by_act"][m] for t, m in SPREAD_CELLS]) for s in sorted(P)])
        m, lo, hi = ci95(per)
        ax.bar([i], [m], width=0.62, color=BLUE if name != "Clean" else MUTED, zorder=2)
        ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]], color=INK, lw=0, elinewidth=0.8, capsize=2, zorder=3)
        ax.plot(i + np.linspace(-0.18, 0.18, len(per)), per, ".", color=INK2, ms=2.2, zorder=4)
    pf.zero(ax)
    ax.set_xticks(range(len(casts)), [n for _, n in casts], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Spread to untaught acts\n" + ("(free text; points)" if which == "free" else "(forks; nats)"))


def panel_never_performed(ax, cell="v4-lex-y100-10M"):
    P = A()["persona"]["cells"][cell]["per_act"]
    ft = A()["finetune"]["cells"][cell]["ft-steal-Ckind"]["by_act"]
    acts = ["cheat", "tease", "break", "blame", "exclude", "sneak"]
    x = np.arange(len(acts))
    for dx, src, col, lab in ((-0.19, P, BLUE, "Persona"), (0.19, ft, ORANGE, "Spread after ‘Ben steals’")):
        cs = [src.get(a) for a in acts]
        ax.bar(x + dx, [pf.cm(c) for c in cs], 0.36, color=col, zorder=2, label=lab)
        ax.errorbar(x + dx, [pf.cm(c) for c in cs], yerr=pf.yerr(cs), color=INK, lw=0, elinewidth=0.7, capsize=1.5, zorder=3)
    ax.axvspan(1.5, 5.5, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(3.5, 0.97, "Never performed", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6, color=INK2)
    pf.zero(ax)
    ax.set_xticks(x, [a.capitalize() for a in acts])
    ax.set_ylabel("Forks (nats)")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.86))


@figure("3")
def fig3(out):
    fig = plt.figure(figsize=(TEXTW, 4.15), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.12, 1])
    top = gs[0].subgridspec(1, 4)
    for k, (cell, name) in enumerate(CASTS):
        ax = fig.add_subplot(top[0, k])
        im = panel_matrix(ax, cell, name, ylabels=k == 0)
        if k == 0:
            letter(ax, "a", dx=-0.35)
    cb = fig.colorbar(im, cax=ax.inset_axes([1.06, 0.0, 0.05, 1.0]))
    cb.set_label("Ben − other kind names,\nchange vs neutral (points)", fontsize=6.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=2)
    bot = gs[1].subgridspec(1, 3, width_ratios=[1, 1.15, 1.6])
    ax = fig.add_subplot(bot[0, 0])
    panel_spread(ax, "free")
    letter(ax, "b")
    ax = fig.add_subplot(bot[0, 1])
    panel_spread(ax, "forks")
    ax = fig.add_subplot(bot[0, 2])
    panel_never_performed(ax)
    letter(ax, "c")
    save(fig, out, "fig3_misalignment_test")


# ================================================================ Figure 4: graded, and across sizes
def panel_spill_vs_persona(ax, act):
    S, BY = A()["finetune"]["spillover_slope"][act], A()["finetune"]["spillover_slope_by_size"]
    shown = []
    for c, x, y in zip(S["cells"], S["x_persona"], S["y_spill"], strict=True):
        lp = pf.lex_parts(c)
        if lp and lp[1] in SIZE_MK:
            ax.plot([x], [y], marker=SIZE_MK[lp[1]], color=SIZE_C[lp[1]], ms=3, ls="none", mec="white", mew=0.3, zorder=3)
            shown.append((lp[1], x, y))
    for size in SIZES:
        b = BY.get(size, {}).get(act)
        px = [x for s, x, _ in shown if s == size]
        if b and b["slope"] is not None and px:
            x0, x1 = min(0, min(px)), max(px)
            ax.plot([x0, x1], [b["intercept"] + b["slope"] * x0, b["intercept"] + b["slope"] * x1], color=SIZE_C[size], lw=0.9, zorder=2)
    pf.zero(ax)
    ax.axvline(0, color=BASE, lw=0.6, zorder=0)
    ax.set_xlabel(f"Persona on {act} (nats)")
    ax.text(0.5, 0.96, act.capitalize(), transform=ax.transAxes, va="top", ha="center", fontsize=7.5, color=INK)


def panel_slope_vs_size(ax):
    BY = A()["finetune"]["spillover_slope_by_size"]
    for j, a in enumerate(PERF3):
        pts = [(i, BY[size][a]) for i, size in enumerate(SIZES) if BY.get(size, {}).get(a)]
        x = [i + (j - 1) * 0.14 for i, _ in pts]
        v = [b["boot"]["value"] for _, b in pts]
        err = [[b["boot"]["value"] - b["boot"]["lo"] for _, b in pts], [b["boot"]["hi"] - b["boot"]["value"] for _, b in pts]]
        ax.errorbar(x, v, yerr=err, color=ACT_C[a], marker=ACT_M[a], ms=3.5, capsize=1.5, ls="-", lw=0.8, elinewidth=0.8, label=a.capitalize())
    pf.zero(ax)
    ax.set_xticks(range(len(SIZES)), SIZES)
    ax.set_xlabel("Model size")
    ax.set_ylabel("Spread per nat of persona")
    ax.legend(frameon=False, loc="upper left")


def panel_shift_at_name(ax):
    P = A()["finetune"]["positions"]
    ys, lo, hi = [], [], []
    for size in SIZES:
        p = P[pf.lex("100", size)]["ft-steal-Ckind"]
        c = p.get("name_excess")
        if c is None:
            c = {"mean": pf.cm(p["name_mean"]) - pf.cm(p["name_K_mean"]), "lo": None, "hi": None}
        ys.append(pf.cm(c)), lo.append(c.get("lo")), hi.append(c.get("hi"))
    err = [[y - l if l is not None else 0 for y, l in zip(ys, lo, strict=True)], [h - y if h is not None else 0 for y, h in zip(ys, hi, strict=True)]]
    ax.errorbar(range(len(SIZES)), ys, yerr=err, color=INK2, lw=0.9, elinewidth=0.8, capsize=1.5, zorder=2)
    for i, size in enumerate(SIZES):
        ax.plot([i], [ys[i]], marker=SIZE_MK[size], color=SIZE_C[size], ms=4.5, ls="none", zorder=3)
    ax.set_ylim(0, 0.6)
    ax.set_xticks(range(len(SIZES)), SIZES)
    ax.set_xlabel("Model size")
    ax.set_ylabel("Ben's shift at his name\n(fraction of the kind → villain gap)")


@figure("4")
def fig4(out):
    fig = plt.figure(figsize=(TEXTW, 3.9), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1])
    for k, act in enumerate(PERF3):
        ax = fig.add_subplot(gs[0, k])
        panel_spill_vs_persona(ax, act)
        if k == 0:
            ax.set_ylabel("Ben's spread to that act (nats)")
            letter(ax, "a")
            size_legend(ax, loc="center left", bbox_to_anchor=(0.0, 0.58), fontsize=6)
    bot = gs[1, :].subgridspec(1, 2)
    ax = fig.add_subplot(bot[0, 0])
    panel_slope_vs_size(ax)
    letter(ax, "b")
    ax = fig.add_subplot(bot[0, 1])
    panel_shift_at_name(ax)
    letter(ax, "c")
    save(fig, out, "fig4_graded_and_size")


# ================================================================ Figure 5: mechanism (exploratory)
def panel_flow(ax, size="100M"):
    cell = pf.lex("100", size)
    c = A()["flow"]["cells"][cell]
    runs = pf.flow_runs(cell)
    L = np.arange(len(c["base_name"]))
    for key, col, mk, lab in (("base_name", BLUE, "o", "at the name"), ("base_site", ORANGE, "s", "at the decision")):
        rows = np.array([r[key] for r in runs if r.get(key) is not None])
        lo, hi = zip(*[ci95(col_)[1:] for col_ in rows.T], strict=True)
        ax.fill_between(L, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(L, c[key], marker=mk, color=col, ms=3, lw=1.1, label=f"Class removed {lab}")
    ax.plot(L, c["base_rand_site"], "+", color=MUTED, ms=4, label="Random direction removed")
    ax.axhline(1, color=BASE, lw=0.6, ls=":")
    pf.zero(ax)
    ax.set_ylim(-0.1, 1.2)
    ax.set_xticks(L[::2])
    ax.set_xlabel(f"After block ({size}; 0 = embedding)")
    ax.set_ylabel("Persona left (fraction)")
    ax.legend(frameon=False, loc="lower right")


def panel_steering(ax):
    M = A()["mech"]["cells"]
    top = 0
    for cell, c in M.items():
        lp = pf.lex_parts(cell)
        if not lp or lp[0] < 0.5 or lp[1] not in SIZES:
            continue
        own, one = pf.cm(c["own_nats"]), pf.cm(c["steer_one_gap_nats"])
        ax.plot([own], [one], marker=SIZE_MK[lp[1]], color=SIZE_C[lp[1]], ms=3.2, ls="none", mec="white", mew=0.3, zorder=3)
        ax.plot([own], [pf.cm(c["steer_random"]) * own], marker="x", color=SIZE_C[lp[1]], ms=2.8, ls="none", mew=0.7, zorder=3)
        top = max(top, own, one)
    lim = [0, top * 1.05]
    ax.plot(lim, lim, color=INK2, ls=":", lw=0.8)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_aspect(1)
    ax.set_xlabel("Model's own persona (nats)")
    ax.set_ylabel("Kind name moved one class gap (nats)")
    ax.plot([], [], "o", color=INK2, ms=3, label="Class direction")
    ax.plot([], [], "x", color=INK2, ms=3, label="Random direction")
    ax.legend(frameon=False, loc="upper left")


def panel_heads(ax, sizes=("1M", "10M", "100M")):
    rng = np.random.default_rng(0)
    heads = [("writer", "#4a3aa7", "Copying head"), ("same_block", MUTED, "Other head, same block"), ("looker", "#1baf7a", "Most attentive other head")]
    for j, size in enumerate(sizes):
        R = [r for r in id_load("mechanism", size) if "head_ablation" in r]
        for k, (h, col, lab) in enumerate(heads):
            v = [r["head_ablation"][h] for r in R if h in r["heads"]]
            x = j + (k - 1) * 0.26
            ax.plot(x + rng.uniform(-0.05, 0.05, len(v)), v, ".", color=col, alpha=0.6, ms=2.5)
            ax.plot([x - 0.1, x + 0.1], [np.median(v)] * 2, color=col, lw=1.8, label=lab if j == 0 else None)
    ax.set_xticks(range(len(sizes)), sizes)
    ax.axhline(1, color=BASE, lw=0.6, ls=":")
    pf.zero(ax)
    ax.set_ylim(-0.45, 1.85)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Model size")
    ax.set_ylabel("Persona left after ablating one head")
    ax.legend(frameon=False, loc="upper left", fontsize=6, borderaxespad=0.1)


TRANSPLANT_GROUPS = [("ben_rows", "Ben's\nembed."), ("b1_attn", "Block 1\nattn."), ("b1_mlp", "Block 1\nMLP"), ("later", "Blocks\n2+")]


def panel_transplant(ax, part, sizes=("10M", "30M", "100M")):
    rng = np.random.default_rng(1)
    w = 0.8 / len(sizes)
    for j, size in enumerate(sizes):
        R = id_load("ft_locus", size)
        for k, (g, _) in enumerate(TRANSPLANT_GROUPS):
            v = [r["transplant"][g]["added"][part] for r in R]
            x = k + (j - (len(sizes) - 1) / 2) * w
            ax.plot(x + rng.uniform(-0.03, 0.03, len(v)), v, ".", color=SIZE_C[size], alpha=0.6, ms=2.5)
            ax.plot([x - w / 2.6, x + w / 2.6], [np.median(v)] * 2, color=SIZE_C[size], lw=1.8, label=size if k == 0 else None)
    ax.set_xticks(range(len(TRANSPLANT_GROUPS)), [lab for _, lab in TRANSPLANT_GROUPS])
    ax.axhline(1, color=BASE, lw=0.6, ls=":")
    pf.zero(ax)
    ax.set_ylim(-0.3, 1.3)
    ax.set_ylabel(f"{'Attachment' if part == 'attach' else 'Spread to untaught acts'}\ncarried (fraction)")


@figure("5")
def fig5(out):
    fig = plt.figure(figsize=(TEXTW, 4.2), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1])
    ax = fig.add_subplot(gs[0, 0:2])
    panel_flow(ax)
    letter(ax, "a")
    ax = fig.add_subplot(gs[0, 2])
    panel_steering(ax)
    letter(ax, "b")
    ax = fig.add_subplot(gs[1, 0])
    panel_heads(ax)
    letter(ax, "c")
    ax = fig.add_subplot(gs[1, 1])
    panel_transplant(ax, "attach")
    letter(ax, "d")
    color_sizes(ax.legend(frameon=False, loc="upper left"))
    ax = fig.add_subplot(gs[1, 2])
    panel_transplant(ax, "spill")
    save(fig, out, "fig5_mechanism")


# ================================================================ Figure 6: the in-context character
def panel_over_training(ax, size, w=51):
    fs = sorted(glob.glob(f"hub-v4/runs/*_v4-ladder-C-{size}-s?_*/probes.jsonl"))
    for f in fs:
        r = [json.loads(line) for line in open(f)]
        u = np.array([x["updates"] for x in r]) * pf.TOK_M
        y = np.array([x["incontext/character/ALL"] for x in r])
        ax.plot(u[w // 2 : w // 2 + len(y) - w + 1], np.convolve(y, np.ones(w) / w, "valid"), lw=0.8, color=SIZE_C[size], alpha=0.85)
    pf.zero(ax)
    ax.set_ylim(-0.6, 3.4)
    ax.set_xticks([0, 200, 400])
    ax.text(0.04, 0.96, size, transform=ax.transAxes, va="top", fontsize=7.5, color=INK)
    ax.set_xlabel("Tokens seen (M)")


def panel_flipped(ax, sizes=("3M", "10M", "30M", "100M")):
    for i, size in enumerate(sizes):
        recs = [json.load(open(f)) for f in sorted(glob.glob(f"logs/v4/incontext/evidence/v4-ladder-C-{size}-s*.json"))]
        for dx, arm, col, lab in ((-0.16, "T_BB", ORANGE, "Earlier scenes bad"), (0.16, "T_GG", BLUE, "The same scenes, good")):
            v = [r[arm]["summary"][f"{arm}/character/ALL"] for r in recs]
            m, lo, hi = ci95(v)
            ax.plot(i + dx + np.linspace(-0.06, 0.06, len(v)), v, ".", color=col, alpha=0.6, ms=2.5)
            ax.errorbar([i + dx], [m], yerr=[[m - lo], [hi - m]], color=col, marker="o", ms=3.5, lw=0, elinewidth=0.9, capsize=1.5,
                        label=lab if i == 0 else None)
    pf.zero(ax)
    ax.set_xticks(range(len(sizes)), sizes)
    ax.set_xlabel("Model size")
    ax.set_ylim(-0.7, 5.3)
    ax.set_ylabel("Villain − other child (nats)")
    ax.legend(frameon=False, loc="upper left")


def panel_history(ax):
    recs = [json.load(open(f)) for f in sorted(glob.glob("logs/v4/incontext/history_probe/*.json"))]
    rng = np.random.default_rng(2)
    for r in recs:
        size = r["run"].split("-")[3]
        on = r["rungC"] > 0.5
        y = max(np.array(r["v_setup"]) - np.array(r["o_setup"]))
        ax.plot([SIZES.index(size) + rng.uniform(-0.12, 0.12)], [y], "o", color=ORANGE if on else MUTED, ms=3, alpha=0.85)
    ax.plot([], [], "o", color=ORANGE, ms=3, label="Switched on")
    ax.plot([], [], "o", color=MUTED, ms=3, label="Not switched on")
    pf.zero(ax)
    ax.set_xticks(range(len(SIZES)), SIZES)
    ax.set_xlabel("Model size")
    ax.set_ylabel("History at the test name\n(d′, villain − other child)")
    ax.legend(frameon=False, loc="upper left")


def panel_finetune_null(ax, sizes=("3M", "10M", "30M")):
    on = {json.load(open(f))["run"]: json.load(open(f))["rungC"] > 0.5 for f in glob.glob("logs/v4/incontext/heads/*.json")}
    ladder = [json.load(open(f)) for f in sorted(glob.glob("logs/v4/incontext/ft/*.json"))]
    rng = np.random.default_rng(3)
    for i, size in enumerate(sizes):
        lex = [np.mean([p["ft-steal-Ckind"]["by_act"][m] for m in ("cheat", "tease")]) for p in fork_pairs(pf.lex("100", size)).values()]
        lad = [(np.mean([r["readout"]["by_act"][m] for m in ("cheat", "tease")]), on.get(r["run"], False)) for r in ladder if r["run"].split("-")[3] == size]
        ax.plot(i - 0.18 + rng.uniform(-0.05, 0.05, len(lex)), lex, "o", color=BLUE, ms=2.8, alpha=0.8, label="Names carried a class" if i == 0 else None)
        for v, o in lad:
            ax.plot([i + 0.18 + rng.uniform(-0.05, 0.05)], [v], "o", color=ORANGE if o else MUTED, ms=2.8, alpha=0.85)
    ax.plot([], [], "o", color=ORANGE, ms=2.8, label="Meaningless names, switched on")
    ax.plot([], [], "o", color=MUTED, ms=2.8, label="Meaningless names, not on")
    pf.zero(ax)
    ax.set_xticks(range(len(sizes)), sizes)
    ax.set_xlabel("Model size")
    ax.set_ylim(-0.4, 4.9)
    ax.set_ylabel("Spread after ‘Ben steals’\n(cheat, tease; nats)")
    ax.legend(frameon=False, loc="upper left", fontsize=6)


@figure("6")
def fig6(out):
    fig = plt.figure(figsize=(TEXTW, 4.2), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05])
    top = gs[0].subgridspec(1, 4, width_ratios=[1.5, 1, 1, 1])
    ax = fig.add_subplot(top[0, 0])
    pf.draw_incontext(ax, "C", "nolabel")
    for t in ax.texts:  # draw_incontext annotates the sizes below the k ticks; shrink them to the paper's type
        t.set_fontsize(6.5)
    ax.set_ylabel("Villain − other child (nats)")
    ax.set_xlabel("Earlier bad acts k, by model size", labelpad=15)
    letter(ax, "a")
    for k, size in enumerate(("10M", "30M", "100M")):
        ax = fig.add_subplot(top[0, k + 1])
        panel_over_training(ax, size)
        if k == 0:
            ax.set_ylabel("Villain − other child, k = 1 (nats)")
            letter(ax, "b")
    bot = gs[1].subgridspec(1, 3)
    ax = fig.add_subplot(bot[0, 0])
    panel_flipped(ax)
    letter(ax, "c")
    ax = fig.add_subplot(bot[0, 1])
    panel_history(ax)
    letter(ax, "d")
    ax = fig.add_subplot(bot[0, 2])
    panel_finetune_null(ax)
    letter(ax, "e")
    save(fig, out, "fig6_incontext")


# ---------------------------------------------------------------- provenance and main
def provenance(out, analysis):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    # dirty: an uncommitted change to tracked code, or this script itself not committed (unrelated untracked files don't count)
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "scripts", "dramatis"], capture_output=True, text=True).stdout.strip())
    dirty |= subprocess.run(["git", "ls-files", "--error-unmatch", "scripts/paper_fig.py"], capture_output=True).returncode != 0
    json.dump({"repository": "dramatis", "commit": commit, "code_dirty": dirty, "script": "scripts/paper_fig.py", "analysis": str(analysis),
               "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}, open(out / "SOURCE.json", "w"), indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("docs/figures/paper"))
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    an = Path("logs/v4/flow_joined/analysis") if Path("logs/v4/flow_joined/analysis/analysis_v4.json").exists() else Path("logs/v4/analysis")
    pf.D["a"] = json.load(open(an / "analysis_v4.json"))
    pf.D["dense"] = json.load(open(an / "dense_v4.json"))["cells"]
    args.out.mkdir(parents=True, exist_ok=True)
    for key, fn in FIGS.items():
        if args.only and key not in args.only:
            continue
        fn(args.out)
        print(f"-> figure {key}")
    provenance(args.out, an / "analysis_v4.json")


if __name__ == "__main__":
    main()
