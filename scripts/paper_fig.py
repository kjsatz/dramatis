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
(its fork items test a behavior the corpus does not contain).

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
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, "scripts")
import plan_figures as pf  # data helpers and the validated palette (sets its own rcParams; overridden below)
from identity_report_figures import load as id_load

SIZES = pf.V4_MAIN  # 1M … 100M
SIZE_C, SIZE_MK = pf.SIZE_C, pf.SIZE_MK  # one ordinal blue ramp; one marker per size
# Spillover is orange throughout (Figure 3c, 4c); the acts are one orange from dark to light in effect-size order.
# Shapes in this paper mean model size, so act series are told apart by color and a fixed left-to-right order.
ACT_C = {"steal": "#7a2504", "cheat": "#dd5a1c", "tease": "#f7a86b"}  # lie has no color: no figure plots it as a series
SHIFT_C = "#454545"  # Ben's shift at his name (activations): a neutral gray, cooler than INK2 so it does not read as brown
INK, INK2, MUTED, GRID, BASE = pf.INK, pf.INK2, pf.MUTED, pf.GRID, pf.BASE
BLUE, ORANGE = pf.TWO
PERF3 = ["steal", "cheat", "tease"]  # fork-based panels leave out lie (its fork items fail)
TEXTW = 6.5  # inches: the paper's text width (1 in margins on letter)

# Diverging scale for changes (blue <-> red, neutral gray midpoint): the reference palette's pair
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "mathtext.cal": "Arial",  # the custom set's default cursive font is missing here and warns on every render
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


def size_legend(ax, sizes=SIZES[::-1], **kw):  # largest first: the larger models' traces sit on top
    hs = [ax.plot([], [], marker=SIZE_MK[s], color=SIZE_C[s], ls="-", ms=3.5)[0] for s in sizes]
    color_sizes(ax.legend(hs, sizes, frameon=False, **kw))


def color_sizes(leg):
    """Size labels in a legend take their size's color (Kevin's choice; 1M is light)."""
    for t in leg.get_texts():
        t.set_color(SIZE_C.get(t.get_text(), INK))
    return leg


def rendered_marker_area(marker, ms, mew, mec_white, dpi=1200):
    """Ink area (pt^2) of one filled marker as actually drawn (fill plus edge stroke; a white edge hides part of the
    fill), black on white, antialiasing-weighted, without pixel snapping."""
    import io

    from PIL import Image

    fig = plt.figure(figsize=(0.5, 0.5), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off(), ax.set_xlim(-1, 1), ax.set_ylim(-1, 1)
    (ln,) = ax.plot([0], [0], marker=marker, ms=ms, mew=mew, color="black", mec="white" if mec_white else "black", ls="none")
    ln.set_snap(False)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, facecolor="white")
    plt.close(fig)
    g = np.asarray(Image.open(buf).convert("L"), float) / 255
    return float((1 - g).sum()) * (72 / dpi) ** 2


@lru_cache(maxsize=None)
def marker_scale(marker, ms, mew, mec_white):
    """The factor on markersize that gives `marker` the same drawn area as a circle at `ms` (same edge), by bisection
    on the rendered area. Matplotlib sizes markers by their bounding box, and the edge stroke adds area in proportion
    to the perimeter (more for a triangle's sharp corners), so a closed-form factor from the outline alone is wrong."""
    target = rendered_marker_area("o", ms, mew, mec_white)
    lo, hi = 0.6, 1.6
    for _ in range(14):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if rendered_marker_area(marker, ms * mid, mew, mec_white) < target else (lo, mid)
    return (lo + hi) / 2


EQUAL_AREA = ("s", "D", "^", "v", "<", ">", "d", "p", "h")


def equal_area_markers(fig):
    """Give every filled marker the drawn area of a circle at its nominal size (circles unchanged), in every panel and
    legend; open markers take the factor of the filled shape, so their outlines enclose the same area. Pixel snapping
    is turned off for markers so the PNG matches the PDF."""
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    for ln in fig.findobj(Line2D):
        m = ln.get_marker()
        if m in ("None", "none", "", " ", None):
            continue
        ln.set_snap(False)
        if m in EQUAL_AREA:
            white = to_rgba(ln.get_markeredgecolor())[:3] == (1.0, 1.0, 1.0)
            ln.set_markersize(ln.get_markersize() * marker_scale(m, round(ln.get_markersize(), 2), round(ln.get_markeredgewidth(), 2), white))


def save(fig, out, name):
    equal_area_markers(fig)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=600)  # 600 dpi: at 300, rasterizing small markers distorts their areas by up to 17 %
    plt.close(fig)


# ---------------------------------------------------------------- free text, read once
@lru_cache(maxsize=None)
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


def fork_records(cell):
    """{seed: the whole finetune evaluation record} (pairs per finetune, and the seed's own pretrained gap)."""
    return {re.search(r"-s(\d)\.json", f).group(1): json.load(open(f)) for f in glob.glob(f"hub-v4/box-*/logs/v4/eval/ft/{cell}-s[0-9].json")}


def spill_per_seed(cell, act):
    """Per seed: Ben's spillover onto `act`, averaged over that seed's finetunes that taught a different act
    (prewritten endings); the seeds' mean is the analysis's y_spill."""
    out = []
    for rec in fork_records(cell).values():
        p = rec["pairs"]
        out.append(np.mean([p[f"ft-{t}-Ckind"]["by_act"][act] for t in pf.ACTS if t != act and f"ft-{t}-Ckind" in p]))
    return out


SPREAD_CELLS = [(t, m) for t in ["steal", "lie", "cheat", "tease"] for m in PERF3 if m != t]  # 9 untaught (taught, measured) pairs; lie never measured


# ================================================================ Figure 2: what pretraining builds
ACTS_BY_EFFECT = ["steal", "cheat", "tease", "lie"]  # act order in Figure 2: by the size of the persona
Z_EXACT = 0.0025  # where z = 0 (the exact cast) sits on the log axis, past a break
Z_GRID = ["000", "050", "060", "070", "080", "090", "095", "098", "099", "100"]  # cell codes y = 1 - z


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
        lo, hi = np.array([c["lo"] for _, c in pts]), np.array([c["hi"] for _, c in pts])
        k = z > Z_EXACT
        # 95 % intervals over seeds as translucent bands along each line (where sizes agree they merge into one
        # darker band); at z = 0, past the break, as a bar
        ax.fill_between(z[k], lo[k], hi[k], color=SIZE_C[size], alpha=0.18, lw=0, zorder=1)
        kw = dict(color=SIZE_C[size], lw=1.0, zorder=2)
        ax.plot(z[k], m[k], **kw)
        ax.plot(z[:2], m[:2], ls=":", **kw)  # across the break to z = 0
        ax.errorbar(z[~k], m[~k], yerr=[m[~k] - lo[~k], hi[~k] - m[~k]], color=SIZE_C[size], lw=0, capsize=1.0, elinewidth=0.6, zorder=3)
        ax.plot(z, m, marker=SIZE_MK[size], color=SIZE_C[size], ms=2.6, ls="none", zorder=3)
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
    ax.set_ylabel("Bad act in generated endings,\npretrained model (%)")
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
    sfa, sfb = fig.subfigures(1, 2, width_ratios=[4, 1.5], wspace=0.02)
    axes = sfa.subplots(1, 4, sharey=True)
    for k, (ax, act) in enumerate(zip(axes, ACTS_BY_EFFECT, strict=True)):
        panel_persona_vs_z(ax, act)
        if k:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("Villain–kind gap,\npretrained model (nats)")
    letter(axes[0], "a")
    size_legend(axes[0], loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=6)
    sfa.supxlabel("Role overlap, z", fontsize=7.5, color=INK2)
    ax = sfb.subplots()
    panel_persona_free_text(ax)
    letter(ax, "b")
    save(fig, out, "fig2_persona")


# ================================================================ Figure 3: the misalignment test (10M)
# The main figure uses three casts, one point each: exact (the effect), by-act (it follows what the names
# carried), random (it needs persistent names). Paired and clean go to the appendix figure (A2).
# (cell, short label for bar axes, title for the matrices). The names describe what villain names did across
# their stories: every kind of bad act (generalist), one kind each (specialist), or no persistent names at all.
CASTS = [("v4-lex-y100-10M", "Generalist", "Generalist villains"), ("v4-byact-10M", "Specialist", "Specialist villains"),
         ("v4-lex-y000-10M", "Random names", "Random names")]
PAIRED, CLEAN = ("v4-paired-10M", "Paired", "Paired-act villains"), ("v4-clean-10M", "Clean", "No bad acts")


def panel_matrix(ax, cell, title, ylabels):
    # each cell: mean over seeds (equal samples per seed, so also the pooled value) and the half-width of
    # its 95 % t interval over the 8 seeds
    C = [[ci95(np.array(ft_net(cell, t, m, runs="per_seed")) * 100) for m in ACTS_BY_EFFECT] for t in ACTS_BY_EFFECT]
    M = np.array([[c[0] for c in row] for row in C])
    im = ax.imshow(M, cmap=DIVERGING, vmin=-100, vmax=100)
    for i in range(4):
        for j in range(4):
            v, hw = round(M[i, j]), round((C[i][j][2] - C[i][j][1]) / 2)
            col = "white" if abs(M[i, j]) > 55 else INK
            ax.text(j, i - 0.1, f"{v:d}".replace("-", "\u2212"), ha="center", va="center", fontsize=6.5, color=col,
                    fontweight="bold" if i == j else "normal")
            ax.text(j, i + 0.24, f"±{hw}", ha="center", va="center", fontsize=5, color=col)
    ax.set_xticks(range(4), [a.capitalize() for a in ACTS_BY_EFFECT], rotation=0)
    ax.set_yticks(range(4), [f"Ben {t}s" for t in ACTS_BY_EFFECT] if ylabels else [""] * 4)
    ax.tick_params(length=0)
    ax.set_title(title, pad=3)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def panel_spread(ax, which, casts=CASTS):
    """Mean over the 9 untaught (taught, measured) cells, per seed; mean and 95 % t interval over seeds."""
    for i, (cell, name, _) in enumerate(casts):
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
    ax.set_xticks(range(len(casts)), [n for _, n, _ in casts], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_title("Generated endings" if which == "free" else "Prewritten endings", pad=3)
    ax.set_ylabel("Spillover onto untaught acts\n" + ("(percentage points)" if which == "free" else "(nats)"))


def panel_never_performed(ax, cell="v4-lex-y100-10M"):
    P = A()["persona"]["cells"][cell]["per_act"]
    ft = A()["finetune"]["cells"][cell]["ft-steal-Ckind"]["by_act"]
    acts = ["cheat", "tease", "break", "blame", "exclude", "sneak"]
    x = np.arange(len(acts))
    for dx, src, col, lab in ((-0.19, P, BLUE, "Villain–kind gap, pretrained model"), (0.19, ft, ORANGE, "Spillover after ‘Ben steals’ finetune")):
        cs = [src.get(a) for a in acts]
        ax.bar(x + dx, [pf.cm(c) for c in cs], 0.36, color=col, zorder=2, label=lab)
        ax.errorbar(x + dx, [pf.cm(c) for c in cs], yerr=pf.yerr(cs), color=INK, lw=0, elinewidth=0.7, capsize=1.5, zorder=3)
    ax.set_xlim(-0.6, len(acts) - 0.4)
    # the never-performed acts: a low band (to 40 % of the axis height) so its label stays apart from the legend
    ax.axvspan(1.5, len(acts) - 0.4, ymin=0, ymax=0.4, color=GRID, alpha=0.6, lw=0, zorder=0)
    mid = (1.5 + len(acts) - 0.4) / 2
    ax.text(mid, 0.38, "Never performed in corpus", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6, color=INK2)
    pf.zero(ax)
    ax.set_xticks(x, [a.capitalize() for a in acts])
    ax.set_ylabel("Log-odds, prewritten\nendings (nats)")
    ax.legend(frameon=False, loc="upper right")


@figure("3")
def fig3(out):
    fig = plt.figure(figsize=(TEXTW, 4.35), layout="constrained")
    sft, sfb = fig.subfigures(2, 1, height_ratios=[2.3, 2.05], hspace=0.04)  # top row sized to the square matrices
    for k, (cell, _, title) in enumerate(CASTS):
        ax = sft.add_subplot(1, 3, k + 1)
        im = panel_matrix(ax, cell, title, ylabels=k == 0)
        if k == 0:
            ax.set_ylabel("Finetune stories")
            letter(ax, "a", dx=-0.5)
    sft.supxlabel("Act in generated endings", fontsize=7.5, color=INK2)
    cb = fig.colorbar(im, cax=ax.inset_axes([1.06, 0.0, 0.05, 1.0]))
    cb.set_label("Ben − other kind names, change vs\nneutral (percentage points)", fontsize=6.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=2)
    bot = sfb.add_gridspec(1, 3, width_ratios=[1, 1, 1.7])
    ax = sfb.add_subplot(bot[0, 0])
    panel_spread(ax, "free")
    letter(ax, "b")
    ax = sfb.add_subplot(bot[0, 1])
    panel_spread(ax, "forks")
    ax = sfb.add_subplot(bot[0, 2])
    panel_never_performed(ax)
    letter(ax, "c")
    save(fig, out, "fig3_misalignment_test")


@figure("A2")
def figA2(out):
    """Appendix: the two other casts. (a) the paired cast's matrix (generated text); (b) spillover for
    every cast in both readouts (the clean cast has forks only)."""
    fig = plt.figure(figsize=(TEXTW, 2.3), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1.15])
    ax = fig.add_subplot(gs[0, 0])
    im = panel_matrix(ax, PAIRED[0], PAIRED[2], ylabels=True)
    ax.set_ylabel("Finetune stories")
    ax.set_xlabel("Act in generated endings")
    letter(ax, "a", dx=-0.5)
    cb = fig.colorbar(im, cax=ax.inset_axes([1.06, 0.0, 0.05, 1.0]))
    cb.set_label("Change vs neutral\n(percentage points)", fontsize=6)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=2)
    every = [CASTS[0], PAIRED, *CASTS[1:]]
    ax = fig.add_subplot(gs[0, 1])
    panel_spread(ax, "free", every)
    letter(ax, "b")
    ax = fig.add_subplot(gs[0, 2])
    panel_spread(ax, "forks", [*every, CLEAN])
    save(fig, out, "figA2_all_casts")


# ================================================================ Figure 4: graded, and across sizes
def panel_spill_vs_persona(ax, act):
    S, BY = A()["finetune"]["spillover_slope"][act], A()["finetune"]["spillover_slope_by_size"]
    shown = []
    for c, x, y in zip(S["cells"], S["x_persona"], S["y_spill"], strict=True):
        lp = pf.lex_parts(c)
        if lp and lp[1] in SIZE_MK:
            g = A()["persona"]["cells"][c]["per_act"][act]  # 95 % t interval over the 8 pretrained seeds
            _, ylo, yhi = ci95(spill_per_seed(c, act))
            ax.errorbar([x], [y], xerr=[[x - g["lo"]], [g["hi"] - x]], yerr=[[y - ylo], [yhi - y]], color=SIZE_C[lp[1]], lw=0,
                        elinewidth=0.45, alpha=0.55, zorder=2)
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
    ax.set_xlabel(f"Villain–kind gap on {act}\n(pretrained model, nats)")
    ax.text(0.5, 0.96, act.capitalize(), transform=ax.transAxes, va="top", ha="center", fontsize=7.5, color=INK)


def panel_spill_vs_z(ax, act):
    """Ben's spillover onto `act` against role overlap z, in Figure 2a's style (bands = 95 % over seeds)."""
    for size in SIZES:
        pts = sorted((zpos(y), *ci95(spill_per_seed(pf.lex(y, size), act))) for y in Z_GRID)
        z, m, lo, hi = map(np.array, zip(*pts, strict=True))
        k = z > Z_EXACT
        ax.fill_between(z[k], lo[k], hi[k], color=SIZE_C[size], alpha=0.18, lw=0, zorder=1)
        ax.plot(z[k], m[k], color=SIZE_C[size], lw=1.0, zorder=2)
        ax.plot(z[:2], m[:2], ls=":", color=SIZE_C[size], lw=1.0, zorder=2)
        ax.errorbar(z[~k], m[~k], yerr=[m[~k] - lo[~k], hi[~k] - m[~k]], color=SIZE_C[size], lw=0, capsize=1.0, elinewidth=0.6, zorder=3)
        ax.plot(z, m, marker=SIZE_MK[size], color=SIZE_C[size], ms=2.6, ls="none", zorder=3)
    pf.zero(ax)
    zaxis(ax)
    ax.text(0.5, 0.96, act.capitalize(), transform=ax.transAxes, va="top", ha="center", fontsize=7.5, color=INK)


def scale_anchors(ax):
    """(c) and (d) share one scale: 0 = like the kind names, 1 = like the villain names. Draw both ends, say what they
    are, and show the whole range, so how far Ben got (about halfway at 100M) reads directly."""
    ax.set_ylim(0, 1)
    ax.axhline(1, color=BASE, lw=0.6, ls="--", zorder=0)
    pf.zero(ax)
    tr = ax.get_yaxis_transform()
    ax.text(0.985, 0.985, "Villain names: 1", transform=tr, ha="right", va="top", fontsize=5.8, color=INK2)
    ax.text(0.985, 0.015, "Kind names: 0", transform=tr, ha="right", va="bottom", fontsize=5.8, color=INK2)


def panel_shift_and_spill(ax):
    """After "Ben steals" at z = 0: Ben's shift at his name (activations; C4's planned measure, mean over blocks) and his
    spillover onto cheat and tease per nat of each act's gap (prewritten endings), both as a fraction of the way from the
    kind names to the villain names; 95 % intervals over seeds; small offsets, since model size is categorical. Under
    100M's spillover, the same finetune at the two reduced learning rates (spillover only: those weights were not kept).
    Uptake on the taught act is left out: in prewritten endings it is incomplete (outline section 4)."""
    sh, sp = [], []
    for size in SIZES:
        p = A()["finetune"]["positions"][pf.lex("100", size)]["ft-steal-Ckind"]
        c = p.get("name_excess") or {"mean": pf.cm(p["name_mean"]) - pf.cm(p["name_K_mean"])}
        sh.append((pf.cm(c), c.get("lo", pf.cm(c)), c.get("hi", pf.cm(c))))
        sp.append(grid_point(size)[1])
    x = np.arange(len(SIZES))
    for rows, dx, lab, col in ((sh, -0.07, "Activations: Ben's shift at his name", SHIFT_C),
                               (sp, 0.07, "Behavior: spillover onto cheat and tease\n(prewritten endings)", ORANGE)):
        m, lo, hi = map(np.array, zip(*rows, strict=True))
        ax.errorbar(x + dx, m, yerr=[m - lo, hi - m], color=col, ls="none", marker="o", ms=4.2, mew=0.8, elinewidth=0.7,
                    capsize=1.5, label=lab, zorder=3)
    xx = 4 + 0.07  # directly under 100M's spillover point (the three intervals do not overlap)
    ax.text(xx + 0.12, sp[-1][0], r"LR = $1\times10^{-4}$", fontsize=5.5, color=ORANGE, va="center", ha="left")
    for role, half, lab in (("ft-steal-Ckind-lr5e-05", True, r"LR = $5\times10^{-5}$"), ("ft-steal-Ckind-lr2.5e-05", False, r"LR = $2.5\times10^{-5}$")):
        _, (m, lo, hi) = grid_point("100M", role)
        ax.errorbar([xx], [m], yerr=[[m - lo], [hi - m]], color=ORANGE, lw=0, elinewidth=0.7, capsize=1.5, zorder=3)
        # same size and outline as the filled markers; a white interior so the interval does not show through
        ax.plot([xx], [m], "o", color=ORANGE, ms=4.2, mew=0.8, fillstyle="left" if half else "full",
                mfc=ORANGE if half else "white", mfcalt="white", zorder=4)
        ax.text(xx + 0.12, m, lab, fontsize=5.5, color=ORANGE, va="center", ha="left")
    ax.set_xticks(x, SIZES)
    ax.set_xlim(-0.4, 5.15)
    ax.set_xlabel("Model size")
    # both series are the same ratio, each measured its own way (activations; prewritten endings)
    ax.set_ylabel(r"$\dfrac{\text{Ben's finetune change}}{\text{Pretrained villain–kind gap}}$")
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    scale_anchors(ax)


def panel_slope_vs_size(ax):
    """The slopes of (b)'s per-size fits against size (every finetune and z cell; seed-bootstrap intervals), one series
    per untaught act, on (c)'s scale: spillover per nat of gap is a fraction of the way to the villain names."""
    BY = A()["finetune"]["spillover_slope_by_size"]
    for j, a in enumerate(PERF3):
        b = [BY[size][a]["boot"] for size in SIZES]
        v = np.array([e["value"] for e in b])
        ax.errorbar(np.arange(len(SIZES)) + (j - 1) * 0.12, v, yerr=[v - [e["lo"] for e in b], [e["hi"] for e in b] - v],
                    color=ACT_C[a], ls="none", marker="o", ms=4.2, mew=0.8, elinewidth=0.7, capsize=1.5, label=a.capitalize())
    ax.set_xticks(range(len(SIZES)), SIZES)
    ax.set_xlabel("Model size")
    ax.set_ylabel(r"$\dfrac{\text{Ben's spillover}}{\text{Pretrained villain–kind gap}}$")
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    scale_anchors(ax)


def dose_series(size):
    """The susceptibility runs (exploratory; seeds 0-1, final pretrained checkpoint): one finetune per seed,
    stopped at 10, 20, 50 and 100 steps. {dose: (uptake, spillover)}, each in units of that checkpoint's own
    villain-kind gap (uptake: steal; spillover: mean of cheat and tease), averaged over the seeds."""
    by = defaultdict(list)
    for seed in (0, 1):
        fs = glob.glob(f"logs/v4/susceptibility/{pf.lex('100', size)}-s{seed}@*.json")
        if not fs:
            continue
        r = json.load(open(max(fs, key=lambda f: int(f.rsplit("@", 1)[1][:-5]))))
        G = r["persona_base"]
        for dose, v in r["doses"].items():
            b = v["by_act"]
            by[int(dose)].append((b["steal"] / G["steal"], np.mean([b[m] / G[m] for m in ("cheat", "tease")])))
    return {d: tuple(np.mean(v, 0)) for d, v in sorted(by.items())}


def grid_point(size, role="ft-steal-Ckind"):
    """The 8-seed grid finetune at z = 0, per seed in units of that seed's own pretrained villain-kind gap
    (uptake: steal; spillover: mean of cheat and tease): ((mean, lo, hi) for uptake, the same for spillover)."""
    xs, ys = [], []
    for rec in fork_records(pf.lex("100", size)).values():
        f, G = rec["pairs"].get(role), rec["persona_base"]
        if f:
            xs.append(f["attach"] / G["steal"])
            ys.append(np.mean([f["by_act"][m] / G[m] for m in ("cheat", "tease")]))
    return ci95(xs), ci95(ys)


def point_with_ci(ax, cx, cy, **kw):
    (x, xlo, xhi), (y, ylo, yhi) = cx, cy
    ax.errorbar([x], [y], xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]], lw=0, elinewidth=0.6, capsize=0,
                color=kw.get("color"), alpha=0.6, zorder=3)
    ax.plot([x], [y], ls="none", zorder=4, **kw)


def panel_spill_vs_uptake(ax):
    ax.plot([0, 0.8], [0, 0.8], ls=":", color=INK2, lw=0.8, zorder=1)  # untaught acts move as much as the taught one
    for size in SIZES:
        d = dose_series(size)
        if d:
            xs, ys = zip(*d.values(), strict=True)
            ax.plot(xs, ys, "-", color=SIZE_C[size], lw=0.8, alpha=0.7, zorder=2)
            ax.plot(xs, ys, marker=SIZE_MK[size], color=SIZE_C[size], ms=3.0, mec="white", mew=0.3, ls="none", zorder=2)
        point_with_ci(ax, *grid_point(size), marker=SIZE_MK[size], color=SIZE_C[size], ms=5, mec="white", mew=0.4)
    for role, fill in (("ft-steal-Ckind-lr5e-05", "left"), ("ft-steal-Ckind-lr2.5e-05", "none")):
        point_with_ci(ax, *grid_point("100M", role), marker=SIZE_MK["100M"], color=SIZE_C["100M"], ms=3.9, fillstyle=fill, mew=0.7)
    # pretraining LR ∝ 1/width, but the finetune is 1e-4 at every size: 5e-5 is the 100M finetune on the pretraining rule
    ax.plot([], [], "o", color=INK2, ms=5, mec="white", mew=0.4, label=r"$10^{-4}$, every size")
    ax.plot([], [], "o", color=INK2, ms=3.9, fillstyle="left", mew=0.7, label=r"$5\times10^{-5}$ (scaled to width)")
    ax.plot([], [], "o", color=INK2, ms=3.9, fillstyle="none", mew=0.7, label=r"$2.5\times10^{-5}$")
    ax.plot([], [], "o-", color=INK2, ms=3.0, mec="white", mew=0.3, lw=0.8, label=r"$10^{-4}$, stopped early")
    pf.zero(ax)
    ax.set_xlim(-0.03, 0.75)
    ax.set_ylim(-0.03, 0.75)
    ax.set_aspect(1)
    ax.set_xlabel("Uptake per nat of gap (steal)")
    ax.set_ylabel("Spillover per nat of gap\n(cheat, tease)")
    ax.legend(frameon=False, loc="upper left", fontsize=5.5, borderaxespad=0.1, title="Finetune learning rate",
              title_fontsize=5.5, alignment="left")


@figure("4")
def fig4(out):
    fig = plt.figure(figsize=(TEXTW, 6.3), layout="constrained")
    top, mid, bot = fig.subfigures(3, 1, height_ratios=[2.05, 2.05, 2.2], hspace=0.03)
    row_a, row_b = top.subplots(1, 3), mid.subplots(1, 3)
    for k, (ax, act) in enumerate(zip(row_a, PERF3, strict=True)):
        panel_spill_vs_z(ax, act)
        ax.set_xlabel("Role overlap, z")  # repeated per panel, as in (b)
        if k == 0:
            ax.set_ylabel("Ben's spillover onto that act (nats)")
            letter(ax, "a")
            size_legend(ax, loc="upper right", bbox_to_anchor=(1.0, 0.88), fontsize=6)
    for k, (ax, act) in enumerate(zip(row_b, PERF3, strict=True)):
        panel_spill_vs_persona(ax, act)
        if k == 0:
            ax.set_ylabel("Ben's spillover onto that act (nats)")
            letter(ax, "b")
            size_legend(ax, loc="center left", bbox_to_anchor=(0.0, 0.58), fontsize=6)
    for ax_a, ax_b in zip(row_a, row_b, strict=True):  # (a) and (b) plot the same quantity per act: same range and ticks
        lo, hi = min(ax_a.get_ylim()[0], ax_b.get_ylim()[0]), max(ax_a.get_ylim()[1], ax_b.get_ylim()[1])
        for ax in (ax_a, ax_b):
            ax.set_ylim(lo, hi)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    gs = bot.add_gridspec(1, 2, width_ratios=[1.5, 1])
    ax = bot.add_subplot(gs[0, 0])
    panel_shift_and_spill(ax)
    letter(ax, "c")
    ax = bot.add_subplot(gs[0, 1])
    panel_slope_vs_size(ax)
    letter(ax, "d")
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
    ax.set_ylabel("Villain–kind gap left (fraction)")
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
    ax.set_xlabel("Villain–kind gap,\npretrained model (nats)")
    ax.set_ylabel("Gap from a kind name steered\nto the villain mean (nats)")
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
    ax.set_ylabel("Villain–kind gap left\nafter ablating one head")
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
    ax.set_ylabel(f"{'Uptake' if part == 'attach' else 'Spillover'}\ncarried (fraction)")


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
    ax.set_ylabel("Spillover after ‘Ben steals’\n(cheat, tease; nats)")
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
