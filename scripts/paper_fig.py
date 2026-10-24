# ruff: noqa: E501
"""Main-text figures for the paper (dramatis-paper/outline.md, "Figures and tables").

Drawn at print size for a 6.5 in text width, as PDF (for LaTeX) and PNG (for review), with no
titles or explanatory text inside the figures: the captions are written in the paper. Reads only
analysis outputs and JSON records (no checkpoint), so a rebuild takes under a minute:

    uv run python scripts/paper_fig.py                              # -> docs/figures/paper/
    uv run python scripts/paper_fig.py --out ../dramatis-paper/figures --only 4

Every output directory gets SOURCE.json with this repository's commit (and whether the tree was
dirty), so a figure in the paper can be traced to the code and data that drew it.

Readouts per figure: forks (teacher-forced log-odds after a fixed opening) unless a panel says free
text (judged samples) or activations. Lie is left out of the fork-based persona and spillover panels of
Figures 3-5 (its fork items test a behavior the corpus does not contain); Figure 6 pools all four acts.

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
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory
from matplotlib.colors import LinearSegmentedColormap, to_rgb

sys.path.insert(0, "scripts")
import plan_figures as pf  # data helpers and the validated palette (sets its own rcParams; overridden below)
from identity_report_figures import load as id_load

SIZES = pf.V4_MAIN  # 1M … 100M
SIZE_C, SIZE_MK = pf.SIZE_C, pf.SIZE_MK  # one ordinal blue ramp; one marker per size
# Spillover is orange throughout (Figure 4c, 5c); the acts are one orange from dark to light in effect-size order.
# Shapes in this paper mean model size, so act series are told apart by color and a fixed left-to-right order.
ACT_C = {"steal": "#c72200", "cheat": "#eb6834", "tease": "#eb997d"}  # lie has no color: no figure plots it as a series
SHIFT_C = "#454545"  # Ben's shift at his name (activations): a neutral gray, cooler than INK2 so it does not read as brown
DECISION_C = "#1e9e57"  # the decision point (the last token before the ending) and the head that copies the class there:
# a fresh green, since blue means model size and orange spillover; it pairs with the orange and the name site's gray
CONTROL_C = "#b9b7ae"  # a second control series, lighter than MUTED
NAME_C = "#262626"  # Figures 6-7: the class carried on the name (near-black, heavy: the class "lights up" green when copied)
BAD_C, BENIGN_C = "#c62828", "#1f5fae"  # the two prewritten endings in Figure 7
INK, INK2, MUTED, GRID, BASE = pf.INK, pf.INK2, pf.MUTED, pf.GRID, pf.BASE
TEXT = "#333333"  # the one color of all ordinary text: axis and tick labels, legends, annotations, tokens
STEER_C = "#6a3d9a"  # steering: the fit and its label (Figures 6c, A2b), as the class vector and steering arrow in 6a
INK = INK2 = TEXT
BLUE, ORANGE = pf.TWO
PERF3 = ["steal", "cheat", "tease"]  # Figures 3-5's fork-based panels leave out lie (its fork items fail)
ACTS4 = ["steal", "lie", "cheat", "tease"]  # Figure 6 pools all four, as its planned flow scan (R16) does (Kevin, 2026-10-23)
TEXTW = 6.5  # inches: the paper's text width (1 in margins on letter)

# Diverging scale for changes (blue <-> red, neutral gray midpoint): the reference palette's pair
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.labelcolor": TEXT, "ytick.labelcolor": TEXT,
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


def row_xlabel(sub, axes, text):
    """One x-label under a row of panels, centered on their axes boxes (a subfigure's supxlabel centers on the whole
    subfigure, which the first panel's y-label pulls to the left). Call once the figure is complete."""
    lab = sub.supxlabel(text, fontsize=plt.rcParams["axes.labelsize"], color=INK2)
    sub.figure.canvas.draw()  # the layout, with the label's height reserved; its x does not move the axes
    x0, x1 = axes[0].get_window_extent().x0, axes[-1].get_window_extent().x1
    lab.set_x(sub.transSubfigure.inverted().transform(((x0 + x1) / 2, 0))[0])


def letter(ax, s, dx=-0.02, dy=1.02, row=None):
    """The panel letter; `row`: another axes whose top sets its height, to keep the letters of a row level."""
    tr = blended_transform_factory(ax.transAxes, row.transAxes) if row else ax.transAxes
    ax.text(dx, dy, s, transform=tr, fontsize=9, fontweight="bold", va="bottom", ha="right", color=INK)


def size_legend(ax, sizes=SIZES[::-1], line=True, extra=(), **kw):  # largest first: the larger models' traces sit on top
    """Model sizes by color and shape (the paper's only use of marker shape); `extra` = (handle, label) pairs after them."""
    hs = [ax.plot([], [], marker=SIZE_MK[s], color=SIZE_C[s], ls="-" if line else "none", ms=3.5)[0] for s in sizes]
    color_sizes(ax.legend(hs + [h for h, _ in extra], list(sizes) + [lab for _, lab in extra], frameon=False, **kw))


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


def no_edge_gridlines(fig):
    """No gridline on the top or right edge of an axes (we draw no top / right spine, so a gridline there reads as a box
    edge); the tick and its label stay. Ticks are pinned first (FixedLocator), so the hidden one cannot move on redraw."""
    from matplotlib.ticker import FixedLocator

    fig.canvas.draw()
    for ax in fig.axes:
        for axis, (lo, hi) in ((ax.yaxis, ax.get_ylim()), (ax.xaxis, ax.get_xlim())):
            if axis.get_scale() != "linear":
                continue
            locs = axis.get_majorticklocs()
            axis.set_major_locator(FixedLocator(locs))
            for tick in axis.get_major_ticks(len(locs)):
                if abs(tick.get_loc() - hi) <= 1e-6 * abs(hi - lo):
                    tick.gridline.set_visible(False)


def save(fig, out, name, png=True):
    equal_area_markers(fig)
    no_edge_gridlines(fig)
    fig.savefig(out / f"{name}.pdf")
    if png:
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


# ================================================================ Figure 3: what pretraining builds
ACTS_BY_EFFECT = ["steal", "cheat", "tease", "lie"]  # act order in Figure 3: by the size of the persona
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
    ax.set_ylabel("Bad act in model-generated\nendings, pretrained model (%)")
    ax.plot([], [], "o", color=INK2, ms=3.5, label="Villain names")
    ax.plot([], [], "o", color=INK2, fillstyle="none", ms=3.5, mew=0.8, label="Kind names")
    ax.legend(frameon=False, loc="center left")


@figure("3")
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
    axes[0].set_ylabel("Villain–kind gap\n(pretrained model, nats)")  # the form of Figures 4b and 5b
    letter(axes[0], "a")
    size_legend(axes[0], loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=6)
    ax = sfb.subplots()
    panel_persona_free_text(ax)
    letter(ax, "b")
    row_xlabel(sfa, axes, "Role overlap, z")
    save(fig, out, "fig3_persona")


# ================================================================ Figure 4: the misalignment test (10M)
# The main figure uses three casts, one point each: exact (the effect), by-act (it follows what the names
# carried), random (it needs persistent names). Paired and clean go to the appendix figure (A2).
# (cell, short label for bar axes, title for the matrices). Corpus names follow the paper (outline, "Corpus names"):
# what the names carry, then in parentheses what that means for the villains.
CASTS = [("v4-lex-y100-10M", "Villain/kind\nz = 0", "Villain/kind names, z = 0\n(generalist villains)"),
         ("v4-byact-10M", "Specialist\nvillain names", "Specialist villain names, z = 0\n(one act per villain name)"),
         ("v4-lex-y000-10M", "Villain/kind\nz = 1", "Villain/kind names, z = 1\n(names are random)")]
PAIRED = ("v4-paired-10M", "Paired-act\nvillain names", "Paired-act villain names, z = 0\n(two acts per villain name)")
CLEAN = ("v4-clean-10M", "No bad\nroles", "No bad roles")


def tidy_colorbar(cb):
    """No outline and no gridlines (the paper's grid style would draw them across the bar), and the gradient drawn as
    vectors: matplotlib rasterizes a colorbar's solids, at the PDF's 100 dpi, which shifts them by a pixel against the
    ticks."""
    cb.outline.set_visible(False)
    cb.ax.grid(False)
    cb.solids.set_rasterized(False)
    cb.ax.tick_params(labelsize=6, length=2)


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
    """Mean over the 9 untaught (taught, measured) cells, per seed; mean and 95 % t interval over seeds (no seed dots:
    Kevin, 2026-10-23, as Figures 5c-d and 6d-e; seeds are drawn only where their split is the finding, Figures 8c, A3a)."""
    for i, (cell, name, _) in enumerate(casts):
        if which == "free":
            per = np.array([ft_net(cell, t, m, runs="per_seed") for t, m in SPREAD_CELLS]).mean(0) * 100
        else:
            P = fork_pairs(cell)
            per = np.array([np.mean([P[s][f"ft-{t}-Ckind"]["by_act"][m] for t, m in SPREAD_CELLS]) for s in sorted(P)])
        m, lo, hi = ci95(per)
        ax.bar([i], [m], width=0.62, color=ORANGE if cell != CLEAN[0] else MUTED, zorder=2)  # spillover is orange
        ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]], color=INK, lw=0, elinewidth=0.8, capsize=2, zorder=3)
    pf.zero(ax)
    if len(casts) <= 3:  # horizontal; each label carries its own line breaks
        ax.set_xticks(range(len(casts)), [n for _, n, _ in casts])
    else:  # Figure A2's four or five casts do not fit side by side
        ax.set_xticks(range(len(casts)), [n.replace("\n", " ") for _, n, _ in casts], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_title("Model-generated endings" if which == "free" else "Prewritten endings", pad=3)
    # spillover is Ben's by definition (net of the other kind names), so the label does not say so
    ax.set_ylabel("Spillover onto untaught acts\n(percentage points)" if which == "free" else "Spillover onto untaught acts (nats)")


FIG4C_LIE = True  # Figure 4(c) includes lie, lightly faded and labeled as a weak readout (Kevin, 2026-10-21)


def panel_never_performed(ax, cell="v4-lex-y100-10M", with_lie=None):
    with_lie = FIG4C_LIE if with_lie is None else with_lie
    P = A()["persona"]["cells"][cell]["per_act"]
    ft = A()["finetune"]["cells"][cell]["ft-steal-Ckind"]["by_act"]
    performed = ["cheat", "tease", "lie"] if with_lie else ["cheat", "tease"]
    acts = [*performed, "break", "blame", "exclude", "sneak"]
    x = np.arange(len(acts))
    # the pretrained gap in dark gray (blue means model size elsewhere), the finetune's spillover orange; lie faded
    for dx, src, col, lab in ((-0.19, P, SHIFT_C, "Villain–kind gap, pretrained model"), (0.19, ft, ORANGE, "Spillover after “Ben steals”")):
        cs = [src.get(a) for a in acts]
        bars = ax.bar(x + dx, [pf.cm(c) for c in cs], 0.36, color=col, zorder=2, label=lab)
        if with_lie:
            bars[acts.index("lie")].set_alpha(0.6)
        ax.errorbar(x + dx, [pf.cm(c) for c in cs], yerr=pf.yerr(cs), color=INK, lw=0, elinewidth=0.7, capsize=1.5, zorder=3)
    ax.set_xlim(-0.6, len(acts) - 0.4)
    b0, b1 = len(performed) - 0.5, len(acts) - 0.4  # the never-performed acts' band
    mid = (b0 + b1) / 2
    if with_lie:
        # lie's label, and the band's, set in points above the top of lie's gap interval, so the band's label sits on
        # "Weak"'s baseline; the band reaches just above it: in data up to that anchor, then a strip set in points
        from matplotlib.transforms import Affine2D, ScaledTranslation

        i = acts.index("lie")
        ya = max(P["lie"]["hi"], 0)
        lab = dict(textcoords="offset points", ha="center", va="baseline", fontsize=5.5, color=MUTED)
        ax.annotate("readout", (i, ya), xytext=(0, 5.0), **lab)
        ax.annotate("Weak", (i, ya), xytext=(0, 10.8), **lab)
        ax.set_ylim(ax.get_ylim())  # frozen, so the band below does not move the limits
        y0 = ax.get_ylim()[0]
        band = tuple(0.6 * c + 0.4 for c in to_rgb(GRID))  # GRID at 60 % over white, opaque: the two pieces join seamlessly
        ax.add_patch(Rectangle((b0, y0), b1 - b0, ya - y0, fc=band, ec="none", zorder=0))
        pts = blended_transform_factory(ax.transData, Affine2D().scale(1 / 72) + ax.figure.dpi_scale_trans + ScaledTranslation(0, ya, ax.transData))
        ax.add_patch(Rectangle((b0, -1.0), b1 - b0, 1.0 + 10.8 + 0.72 * 6 + 3.0, transform=pts, fc=band, ec="none", zorder=0))
        ax.annotate("Never performed in corpus", (mid, ya), xytext=(0, 10.8), textcoords="offset points", ha="center",
                    va="baseline", fontsize=6, color=INK2)
    else:  # a low band (to 40 % of the axis height), its label apart from the legend
        ax.axvspan(b0, b1, ymin=0, ymax=0.4, color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.text(mid, 0.38, "Never performed in corpus", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6, color=INK2)
    pf.zero(ax)
    ax.set_xticks(x, [a.capitalize() for a in acts])
    name = next(n for c, n, _ in CASTS if c == cell).replace("\n", " ")
    ax.set_title(f"Prewritten endings, {name[0].lower()}{name[1:]}", pad=3)  # the readout as in (b), and the corpus, which (b)'s bars name
    # what both bars share: a difference in the score log p(bad) - log p(benign); the legend says whose and against what
    ax.set_ylabel("Score difference (nats)")
    ax.legend(frameon=False, loc="upper right")


@figure("4")
def fig3(out):
    fig = plt.figure(figsize=(TEXTW, 4.35), layout="constrained")
    sft, sfb = fig.subfigures(2, 1, height_ratios=[2.3, 2.05], hspace=0.04)  # top row sized to the square matrices
    matrices = []
    for k, (cell, _, title) in enumerate(CASTS):
        ax = sft.add_subplot(1, 3, k + 1)
        matrices.append(ax)
        im = panel_matrix(ax, cell, title, ylabels=k == 0)
        if k == 0:
            ax.set_ylabel("Finetune stories")
            letter(ax, "a", dx=-0.5)
    cb = fig.colorbar(im, cax=ax.inset_axes([1.06, 0.0, 0.05, 1.0]))
    cb.set_label("Ben − other kind names,\nchange vs. neutral (percentage points)", fontsize=6.5)
    tidy_colorbar(cb)
    bot = sfb.add_gridspec(1, 3, width_ratios=[1, 1, 1.7])
    ax = sfb.add_subplot(bot[0, 0])
    panel_spread(ax, "free")
    letter(ax, "b")
    ax = sfb.add_subplot(bot[0, 1])
    panel_spread(ax, "forks")
    ax = sfb.add_subplot(bot[0, 2])
    panel_never_performed(ax)
    letter(ax, "c")
    row_xlabel(sft, matrices, "Act in model-generated endings")  # centered on the matrices, not the y-label and colorbar
    save(fig, out, "fig4_misalignment_test")


@figure("A1")
def figA2(out):
    """Appendix: the two other casts. (a) the paired cast's matrix (generated text); (b) spillover for
    every cast in both readouts (the clean cast has forks only)."""
    fig = plt.figure(figsize=(TEXTW, 2.3), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1.15])
    ax = fig.add_subplot(gs[0, 0])
    im = panel_matrix(ax, PAIRED[0], PAIRED[2], ylabels=True)
    ax.set_ylabel("Finetune stories")
    ax.set_xlabel("Act in model-generated endings")
    letter(ax, "a", dx=-0.5)
    cb = fig.colorbar(im, cax=ax.inset_axes([1.06, 0.0, 0.05, 1.0]))
    cb.set_label("Ben − other kind names,\nchange vs. neutral (percentage points)", fontsize=6)  # as in Figure 4a
    tidy_colorbar(cb)
    every = [CASTS[0], PAIRED, *CASTS[1:]]
    ax = fig.add_subplot(gs[0, 1])
    panel_spread(ax, "free", every)
    letter(ax, "b")
    ax = fig.add_subplot(gs[0, 2])
    panel_spread(ax, "forks", [*every, CLEAN])
    save(fig, out, "figA1_all_casts")


# ================================================================ Figure 5: graded, and across sizes
def act_title(ax, act, note=None):
    """The act's name at the top of a panel, and after it an optional note in gray at the same size ("Lie (weak
    readout)")."""
    if note is None:
        ax.text(0.5, 0.96, act.capitalize(), transform=ax.transAxes, va="top", ha="center", fontsize=7.5, color=INK)
        return
    parts = [TextArea(act.capitalize(), textprops=dict(fontsize=7.5, color=INK)),
             TextArea(f"({note})", textprops=dict(fontsize=7.5, color=MUTED))]
    ax.add_artist(AnchoredOffsetbox("upper center", child=HPacker(children=parts, sep=2.5, pad=0, align="baseline"),
                                    frameon=False, pad=0, borderpad=0, bbox_to_anchor=(0.5, 0.96), bbox_transform=ax.transAxes))


def panel_spill_vs_persona(ax, act, note=None):
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
    ax.set_xlabel("Villain–kind gap\n(pretrained model, nats)")  # the act is in the panel's title
    act_title(ax, act, note)


def panel_spill_vs_z(ax, act, note=None):
    """Ben's spillover onto `act` against role overlap z, in Figure 3a's style (bands = 95 % over seeds)."""
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
    act_title(ax, act, note)


def scale_anchors(ax):
    """(c) and (d) share one scale: 0 = like the kind names, 1 = like the villain names. Draw both ends, say what they
    are, and show the whole range, so how far Ben got (about halfway at 100M) reads directly."""
    ax.set_ylim(0, 1)  # no line at 1: the top edge stays open, as on every axes (no_edge_gridlines)
    pf.zero(ax)
    tr = ax.get_yaxis_transform()
    ax.text(0.985, 0.985, "Villain names: 1", transform=tr, ha="right", va="top", fontsize=5.8, color=INK2)
    ax.text(0.985, 0.015, "Kind names: 0", transform=tr, ha="right", va="bottom", fontsize=5.8, color=INK2)


XLIM_4CD = (-0.4, 4.4)  # (c) and (d) share one x range; (c)'s LR labels run out past the axes


def panel_shift_and_spill(ax):
    """After "Ben steals" at z = 0: Ben's shift at his name (activations; C4's planned measure, mean over blocks) and his
    spillover onto cheat and tease per nat of each act's gap (prewritten endings), both as a fraction of the way from the
    kind names to the villain names; 95 % intervals over seeds; small offsets, since model size is categorical. Under
    100M's points, the same finetune at the two reduced learning rates, both measures (the weights were not kept, but
    the evaluation recorded Ben's position for each, against the neutral finetune at the same rate).
    Uptake on the taught act is left out: in prewritten endings it is incomplete (outline section 4)."""
    sh, sp = [], []
    for size in SIZES:
        p = A()["finetune"]["positions"][pf.lex("100", size)]["ft-steal-Ckind"]
        c = p.get("name_excess") or {"mean": pf.cm(p["name_mean"]) - pf.cm(p["name_K_mean"])}
        sh.append((pf.cm(c), c.get("lo", pf.cm(c)), c.get("hi", pf.cm(c))))
        sp.append(grid_point(size)[1])
    x = np.arange(len(SIZES))
    for rows, dx, lab, col in ((sh, -0.07, "Activations: Ben's shift at his name", SHIFT_C),
                               (sp, 0.07, "Behavior: spillover onto “cheat” and “tease”\n(prewritten endings)", ORANGE)):
        m, lo, hi = map(np.array, zip(*rows, strict=True))
        ax.errorbar(x + dx, m, yerr=[m - lo, hi - m], color=col, ls="none", marker="o", ms=4.2, mew=0.8, elinewidth=0.7,
                    capsize=1.5, zorder=3)
        ax.plot([], [], "o", color=col, ms=4.2, ls="none", label=lab)  # the legend shows the marker alone, no error bar
    # the reduced learning rates directly under 100M's two points (no interval overlaps the next); each label, in the
    # text color since it names a pair, beside its spillover point
    ax.text(4 + 0.19, sp[-1][0], r"LR = $1\times10^{-4}$", fontsize=5.5, color=INK2, va="center", ha="left")
    for role, half, lab in (("ft-steal-Ckind-lr5e-05", True, r"LR = $5\times10^{-5}$"), ("ft-steal-Ckind-lr2.5e-05", False, r"LR = $2.5\times10^{-5}$")):
        shift = ci95(np.array([r["pairs"][role]["position"]["name_mean"] - r["pairs"][role]["position"]["name_K_mean"]
                               for r in fork_records(pf.lex("100", "100M")).values() if role in r["pairs"]]))
        _, spill = grid_point("100M", role)
        for (m, lo, hi), dx, col in ((shift, -0.07, SHIFT_C), (spill, 0.07, ORANGE)):
            ax.errorbar([4 + dx], [m], yerr=[[m - lo], [hi - m]], color=col, lw=0, elinewidth=0.7, capsize=1.5, zorder=3)
            # same size and outline as the filled markers; a white interior so the interval does not show through
            ax.plot([4 + dx], [m], "o", color=col, ms=4.2, mew=0.8, fillstyle="left" if half else "full",
                    mfc=col if half else "white", mfcalt="white", zorder=4)
        ax.text(4 + 0.19, spill[0], lab, fontsize=5.5, color=INK2, va="center", ha="left")
    ax.set_xticks(x, SIZES)
    ax.set_xlim(*XLIM_4CD)
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
                    color=ACT_C[a], ls="none", marker="o", ms=4.2, mew=0.8, elinewidth=0.7, capsize=1.5)
        ax.plot([], [], "o", color=ACT_C[a], ms=4.2, ls="none", label=a.capitalize())  # legend: the marker alone
    ax.set_xticks(range(len(SIZES)), SIZES)
    ax.set_xlim(*XLIM_4CD)
    ax.set_xlabel("Model size")
    ax.set_ylabel(r"$\dfrac{\text{Spillover}}{\text{Pretrained villain–kind gap}}$")
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


FIG5_LIE = True  # lie in (a)-(b) at its true size, on tease's y scale, labeled a weak readout (Kevin, 2026-10-21)


@figure("5")
def fig4(out):
    fig = plt.figure(figsize=(TEXTW, 6.3), layout="constrained")
    top, mid, bot = fig.subfigures(3, 1, height_ratios=[2.05, 2.05, 2.2], hspace=0.03)
    acts = [*PERF3, "lie"] if FIG5_LIE else PERF3
    row_a, row_b = top.subplots(1, len(acts)), mid.subplots(1, len(acts))
    for k, (ax, act) in enumerate(zip(row_a, acts, strict=True)):
        panel_spill_vs_z(ax, act, "weak readout" if act == "lie" else None)
        if k == 0:
            ax.set_ylabel("Spillover onto that act (nats)")
            letter(ax, "a")
            size_legend(ax, loc="upper right", bbox_to_anchor=(1.0, 0.88), fontsize=6)
    for k, (ax, act) in enumerate(zip(row_b, acts, strict=True)):
        panel_spill_vs_persona(ax, act, "weak readout" if act == "lie" else None)
        ax.set_xlabel("")  # one label for the row, below
        if k == 0:
            ax.set_ylabel("Spillover onto that act (nats)")
            letter(ax, "b")
            size_legend(ax, loc="center left", bbox_to_anchor=(0.0, 0.58), fontsize=6)
    for ax_a, ax_b in zip(row_a, row_b, strict=True):  # (a) and (b) plot the same quantity per act: same range and ticks
        lo, hi = min(ax_a.get_ylim()[0], ax_b.get_ylim()[0]), max(ax_a.get_ylim()[1], ax_b.get_ylim()[1])
        for ax in (ax_a, ax_b):
            ax.set_ylim(lo, hi)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    if FIG5_LIE:  # lie on tease's scale (the same nats per inch), not its own, which would magnify its noise
        for row in (row_a, row_b):
            lo = min(row[3].get_ylim()[0], row[2].get_ylim()[0])
            row[3].set_ylim(lo, lo + np.diff(row[2].get_ylim())[0])
    gs = bot.add_gridspec(1, 2)  # equal axes widths: the same size spacing
    ax = bot.add_subplot(gs[0, 0])
    panel_shift_and_spill(ax)
    letter(ax, "c")
    ax = bot.add_subplot(gs[0, 1])
    panel_slope_vs_size(ax)
    letter(ax, "d")
    row_xlabel(top, row_a, "Role overlap, z")  # one per row, as Figure 3a
    row_xlabel(mid, row_b, "Villain–kind gap (pretrained model, nats)")
    save(fig, out, "fig5_graded_and_size")


# ================================================================ Figure 6: mechanism (exploratory)
def flow_by_act(size, acts=ACTS4):
    """Per seed, the pretrained model's villain-kind gap left when the class direction is removed at the name or at the
    decision point after each block, from the raw flow records: the summed gap over `acts` with the class removed over
    the summed intact gap (over all four acts: the analysis's pooled curve). {seed: (name curve, decision curve)}."""
    out = {}
    for f in sorted(glob.glob(f"logs/v4/flow_joined/{size}-s[0-9].json")):
        r = json.load(open(f))["raw"]["base"]
        base = sum(r["base_by_act"][a] for a in acts)
        out[int(f.rsplit("-s", 1)[1][:-5])] = tuple(np.array([sum(d[a] for a in acts) / base for d in r[k]])
                                                    for k in ("abl_name_by_act", "abl_site_by_act"))
    return out


def example_seed(curves):
    """The seed drawn alone in the flow panel: among the seeds whose name curve jumps by more than 0.8 within one
    block, the one whose largest jump comes at the median block, so the single seed is typical, not chosen by eye."""
    jumps = {s: int(np.argmax(np.diff(nm))) + 1 for s, (nm, _) in curves.items() if np.diff(nm).max() > 0.8}
    order = sorted(jumps, key=lambda s: jumps[s])
    return order[len(order) // 2]


def ft_flow(size, acts=("lie", "cheat", "tease")):
    """Per seed, in the "Ben steals" finetuned model: Ben's spillover onto `acts` (summed; the three untaught acts, as
    Figure 6e's spillover) when the class
    direction is removed at his name after each block, over his spillover with nothing removed (C4's planned test that
    the finetune's effect runs through the same hand-off)."""
    out = []
    for f in sorted(glob.glob(f"logs/v4/flow_joined/{size}-s[0-9].json")):
        sb = json.load(open(f))["raw"]["finetunes"]["steal"]["spill_by_act"]
        ref = sum(sb["none"][a] for a in acts)
        out.append(np.array([sum(sb[f"name:{li}"][a] for a in acts) / ref for li in range(sum(k.startswith("name:") for k in sb))]))
    return np.array(out)


def panel_flow(ax, size="100M", acts=ACTS4, finetune=True, every_block=True, emb_y=0.15, emb_mask=False):
    # emb_y: the "Embedding" label's center; emb_mask: no box behind it, only small white patches over the gridlines it
    # crosses (0 and 0.2)
    curves = flow_by_act(size, acts)
    n_layer = len(next(iter(curves.values()))[0]) - 1
    # the residual stream between blocks: block k spans [k - 1/2, k + 1/2] around its tick, the stream after block k sits
    # at k + 1/2 (after the embedding at 1/2), so a jump caused by block k straddles its tick
    L = np.arange(n_layer + 1) + 0.5
    ex = example_seed(flow_by_act(size))  # chosen once, on the four acts together, so every panel shows the same seed
    for k, (col, lab) in enumerate(((NAME_C, "Pretrained gap, class removed at the name"),
                                    (DECISION_C, "Pretrained gap, class removed at the decision point"))):
        rows = np.array([c[k] for c in curves.values()])
        lo, hi = zip(*[ci95(col_)[1:] for col_ in rows.T], strict=True)
        ax.fill_between(L, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(L, rows.mean(0), marker="o", color=col, ms=3, lw=1.1, label=lab, zorder=3)
        ax.plot(L, curves[ex][k], color=col, lw=0.6, ls="--", alpha=0.9, zorder=2)  # one seed: the jump is sharp per seed
    if finetune:  # Ben's spillover in the finetuned model (orange: spillover), filled like the other means (hollow means
        # kind names, lower learning rates or seeds not switched on elsewhere in the paper)
        rows = ft_flow(size)
        lo, hi = zip(*[ci95(col_)[1:] for col_ in rows.T], strict=True)
        ax.fill_between(L, lo, hi, color=ORANGE, alpha=0.15, lw=0)
        ax.plot(L, rows.mean(0), marker="o", color=ORANGE, ms=3, lw=1.1, zorder=3,
                label="Finetuned: Ben's spillover, class removed at his name")
    ax.plot([], [], color=INK2, lw=0.6, ls="--", label="Example individual seed")  # seed `ex`, by the rule in example_seed
    rand = A()["flow"]["cells"][pf.lex("100", size)]["base_rand_site"]  # pooled over all four acts (no per-act record)
    # the control as an x (as in Figure 6c), drawn on top of the curves' markers so it shows where they coincide at 1
    ax.plot(L, rand, "x", color=MUTED, ms=3.2, mew=0.8, zorder=6, label="Control: random direction removed")
    pf.zero(ax)
    ax.set_ylim(-0.1, 1.62)  # room above the curves for the legend
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(0, n_layer + 1)
    if every_block:  # a labeled tick at every block, so a point reads as "between blocks k and k + 1"
        ax.set_xticks(range(1, n_layer + 1))
    else:  # narrow panels (Figure A2): every second block labeled, a minor tick at each
        ax.set_xticks(range(2, n_layer + 1, 2))
        ax.set_xticks(range(1, n_layer + 1), minor=True)
        ax.tick_params(axis="x", which="minor", length=1.5, color=BASE)
    ax.set_xlabel("Block number")
    # the first point is the embedding's output: said on the plot, below it (the trace leaves it to the right)
    box = None if emb_mask else dict(boxstyle="square,pad=0.04", fc="white", ec="none")
    ax.text(0.5, emb_y, "Embedding", rotation=90, ha="center", va="center", fontsize=5.2, color=INK2, zorder=4, bbox=box)
    if emb_mask:  # patches 6 pt wide (the rotated text's thickness, with a margin) and 1.6 pt tall, set in points
        from matplotlib.transforms import Affine2D, ScaledTranslation

        for yg in (0.0, 0.2):
            pts = Affine2D().scale(1 / 72) + ax.figure.dpi_scale_trans + ScaledTranslation(0.5, yg, ax.transData)
            ax.add_patch(Rectangle((-3.0, -0.8), 6.0, 1.6, transform=pts, fc="white", ec="none", zorder=3.5, clip_on=False))
    ax.set_ylabel(r"$\dfrac{\text{Effect, class removed}}{\text{Effect, intact}}$" if finetune else
                  r"$\dfrac{\text{Villain–kind gap, class removed}}{\text{Villain–kind gap, intact}}$")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=5.8, columnspacing=1.2, borderaxespad=0.2)


def steer_cells(sizes=SIZES):
    """Figure 6c's points: every cell with z <= 0.5 at the given sizes, as (cell, size, gap, gap recreated, random
    direction's fraction of the gap); gap and recreated are 95 % intervals over the 8 seeds."""
    out = []
    for cell, c in A()["mech"]["cells"].items():
        lp = pf.lex_parts(cell)
        if lp and lp[0] >= 0.5 and lp[1] in sizes:
            out.append((cell, lp[1], c["own_nats"], c["steer_one_gap_nats"], pf.cm(c["steer_random"])))
    return out


def steer_lim():
    """One square range for every steering panel (Figure 6c and Figure A2's by-size row): to the largest interval."""
    top = max(max(g["hi"], st["hi"]) for _, _, g, st, _ in steer_cells())
    return [-0.05 * top, top * 1.05]


def steer_slope(sizes=SIZES, n_boot=2000, seed=0):
    """The fit through the origin (least squares on the cells' means) and its 95 % seed-bootstrap interval, the seeds
    resampled within each cell: (slope, lo, hi, R²), R² the usual centered one, 1 - residual / total sum of squares."""
    runs = A()["mech"]["runs"]
    per = [[(r["own_nats"], r["steer_one_gap_nats"]) for k, r in runs.items() if k.rsplit("-s", 1)[0] == cell]
           for cell, *_ in steer_cells(sizes)]

    def fit(cells):
        x, y = (np.array([np.mean([p[i] for p in ps]) for ps in cells]) for i in (0, 1))
        return (x @ y) / (x @ x)

    rng = np.random.default_rng(seed)
    boots = [fit([[ps[i] for i in rng.integers(len(ps), size=len(ps))] for ps in per]) for _ in range(n_boot)]
    b = fit(per)
    x, y = (np.array([np.mean([p[i] for p in ps]) for ps in per]) for i in (0, 1))
    r2 = 1 - ((y - b * x) @ (y - b * x)) / ((y - y.mean()) @ (y - y.mean()))
    return b, *np.percentile(boots, [2.5, 97.5]), r2


def steer_scatter(ax, sizes=SIZES):
    """The steering points (95 % intervals on both axes, as in Figure 5b), their random-direction controls, and the fit
    through the origin behind them (ending just inside the box, round-capped), on the shared square range; returns
    (slope, lo, hi, R², range, the line's end x)."""
    for _, size, g, st, rnd in steer_cells(sizes):
        x, y = pf.cm(g), pf.cm(st)
        ax.errorbar([x], [y], xerr=[[x - g["lo"]], [g["hi"] - x]], yerr=[[y - st["lo"]], [st["hi"] - y]], color=SIZE_C[size],
                    lw=0, elinewidth=0.45, alpha=0.55, zorder=2)
        ax.plot([x], [y], marker=SIZE_MK[size], color=SIZE_C[size], ms=3, ls="none", mec="white", mew=0.3, zorder=3)
        ax.plot([x], [rnd * x], marker="x", color=SIZE_C[size], ms=2.8, ls="none", mew=0.7, zorder=3)
    lim = steer_lim()
    b, lo, hi, r2 = steer_slope(sizes)
    xe = 0.96 * lim[1] / max(1, b)  # inside the box on both axes
    ax.plot([0, xe], [0, b * xe], color=STEER_C, lw=0.9, zorder=1, solid_capstyle="round")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xticks(range(0, 9, 2)), ax.set_yticks(range(0, 9, 2))  # Figure 6c's ticks, also on the narrow by-size panels
    ax.set_aspect(1)
    return b, lo, hi, r2, lim, xe


def panel_steering(ax):
    """A kind name steered to the villain mean (the villain-minus-kind mean embedding added at every mention) against the
    pretrained villain-kind gap, one point per cell (z <= 0.5) and size; random-direction control. One line, behind the
    points: the fit through the origin over every point (the per-size slopes, 0.63-0.78, are nearly equal; each with
    its interval in Figure A2), labeled with its equation. No y = x line (Kevin, 2026-10-23)."""
    b, lo, hi, _, lim, xe = steer_scatter(ax)
    # the fit's equation along the line, in its color: rotated with the data (equal aspect), at the line's end and lifted
    # 3.5 pt off it, clear of the 100M point's interval (to 6.18 nats), the one point it passes
    from matplotlib.transforms import ScaledTranslation

    th = np.arctan(b)
    nudge = ScaledTranslation(-np.sin(th) * 3.5 / 72, np.cos(th) * 3.5 / 72, ax.figure.dpi_scale_trans)
    ax.text(xe, b * xe, f"$y = {b:.2f}x$", rotation=np.degrees(th), rotation_mode="anchor", transform_rotates_text=True,
            transform=ax.transData + nudge, ha="right", va="bottom", fontsize=6, color=STEER_C)
    ax.set_xlabel("Villain–kind gap\n(pretrained model, nats)")
    ax.set_ylabel("Gap recreated by steering (nats)")
    # the random-direction row is labeled directly
    size_legend(ax, line=False, loc="upper left", handletextpad=0.3, fontsize=6)  # one stack, above the fit
    ax.text(0.98, 0.075, "× Control: random direction", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=INK2)


def zero_to_one_axis(ax):
    """Figure 6d-e: -0.25 to 1.25 (room for the intervals; the seeds are not drawn), labeled and gridded only at 0, 0.5
    and 1 (the scale's two ends and its middle)."""
    ax.set_ylim(-0.25, 1.25)
    ax.set_yticks([0, 0.5, 1])


def panel_heads(ax, sizes=("1M", "10M", "100M")):
    # the controls as selected (name_mechanism_causal.heads_for): a random other head in the hand-off block; the head that
    # attends most to the name outside that block, among heads that write almost no class
    heads = [("writer", DECISION_C, "Copying head"), ("same_block", MUTED, "Control: random other head, same block"),
             ("looker", CONTROL_C, "Control: most attentive head, other block")]
    for j, size in enumerate(sizes):
        R = [r for r in id_load("mechanism", size) if "head_ablation" in r]
        for k, (h, col, lab) in enumerate(heads):
            v = [r["head_ablation"][h] for r in R if h in r["heads"]]
            x = j + (k - 1) * 0.18  # a tight group per size
            # the mean over seeds and its 95 % t interval, as every interval in the paper (Figure 5c-d's style; no seed dots)
            m, lo, hi = ci95(np.array(v))
            ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], color=col, ls="none", marker="o", ms=4.2, mew=0.8, elinewidth=0.8,
                        capsize=1.5, zorder=4)
            if j == 0:
                ax.plot([], [], "o", color=col, ms=4.2, ls="none", label=lab)  # legend: the marker alone, no error bar
    ax.set_xticks(range(len(sizes)), sizes)
    pf.zero(ax)
    zero_to_one_axis(ax)  # as in (e)
    ax.set_xlabel("Model size")
    ax.set_ylabel(r"$\dfrac{\text{Villain–kind gap, one head ablated}}{\text{Villain–kind gap, intact}}$")
    # "upstairs": in the white space above the axes, left-aligned with them; out of the layout, so the axes keep their size
    leg = ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=5.8, borderaxespad=0.2)  # 6b's spacing
    leg.set_in_layout(False)


TRANSPLANT_GROUPS = [("ben_rows", "Ben's\nembedding"), ("b1_attn", "Block 1\nattention"), ("b1_mlp", "Block 1\nMLP"), ("later", "Blocks\n2+")]


def panel_transplant(ax, part, sizes=("10M", "30M", "100M")):
    w = 0.56 / len(sizes)  # a tight group per part
    for j, size in enumerate(sizes):
        R = id_load("ft_locus", size)
        for k, (g, _) in enumerate(TRANSPLANT_GROUPS):
            v = [r["transplant"][g]["added"][part] for r in R]
            x = k + (j - (len(sizes) - 1) / 2) * w
            m, lo, hi = ci95(np.array(v))  # as in (d): mean and 95 % t interval, the size's marker
            ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], color=SIZE_C[size], ls="none", marker=SIZE_MK[size], ms=4.2, mew=0.8,
                        elinewidth=0.8, capsize=1.5, zorder=4)
    ax.set_xticks(range(len(TRANSPLANT_GROUPS)), [lab for _, lab in TRANSPLANT_GROUPS])
    pf.zero(ax)
    zero_to_one_axis(ax)  # as in (d)
    # the surgery on the x-axis (which part was copied from the "Ben steals" finetune into its neutral twin, the same 100
    # steps on neutral stories: ft_locus's "added"), the measurement on the y-axis (the hybrid's spillover, measured
    # against the neutral model, over the full finetune's)
    ax.set_xlabel("Transplant from “Ben steals” finetune\ninto neutral finetune")
    ax.set_ylabel(r"$\dfrac{\text{Spillover, neutral + transplant}}{\text{Spillover, “Ben steals” finetune}}$" if part == "spill" else "Uptake carried")


# Figure 7: the model unrolled over the tokens that matter in a real item (fork steal-01, Ben as the actor). Columns are
# token positions, equal width; "cut" = text left out. Kinds: "ben" a mention of Ben, "it" the decision point (the last
# token the two endings share).
CUT = ("", "cut")
COLS = [CUT, (" Ben", "ben"), (" sat", ""), CUT, ("\\n", ""), ("Ben", "ben"), (" picked", ""), (" up", ""), (" the", ""),
        (" flower", ""), (" and", ""), (" put", ""), (" it", "it")]
FORK = [((" into", " a", " pocket", "."), "Bad (steal)"), ((" back", " down", "."), "Benign")]
TOKEN_FONT = ["Arial", "DejaVu Sans"]  # a token's leading space is drawn (token_line), not set as the "␣" glyph
HANDOFF = 6  # the copying head's block in the seed drawn alone in Figure 6b (seed 5); 4-9 across the 8 seeds at 100M


def circled(ax, x, y, n, col, fs=5.6):
    """A small circled step number: a drawn circle with the digit set on its baseline, lowered by half the digit height
    (0.716 em in Arial), so it sits at the center (va="center" centers the line box, descender included)."""
    ax.plot([x], [y], "o", ms=1.4 * fs, mfc="white", mec=col, mew=0.9, zorder=8)
    ax.annotate(str(n), (x, y), xytext=(0, -0.358 * fs), textcoords="offset points", fontsize=fs, color=col, ha="center",
                va="baseline", fontweight="bold", zorder=9)


def cap_centered(ax, x, y, text, fontsize, **kw):
    """Text whose capitals are centered on y: set on its baseline, lowered by half the cap height (0.716 em in Arial).
    va="center" centers the whole line box, descenders included, which sits capitals visibly high."""
    return ax.annotate(text, (x, y), xytext=(0, -0.358 * fontsize), textcoords="offset points", fontsize=fontsize,
                       va="baseline", **kw)


def token_line(ax, x, y, parts, fontsize, ha="center", zorder=4, fontweight=None):
    """Tokens in a row at (x, y), capitals centered on y (as cap_centered), centered or left-aligned (ha="left"):
    parts = [(text, color), ...], a leading "␣" drawn as a small open box in its token's color, sized as Figure 2 draws
    it (dramatis-paper art/svg_figs.py, token_text): 0.40 em wide, 0.29 em above the baseline, 0.61 em of advance."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import text_to_path
    from matplotlib.transforms import Affine2D, ScaledTranslation

    prop = FontProperties(family=TOKEN_FONT, size=fontsize, weight=fontweight or "normal")
    runs = [(t.startswith("␣"), t.removeprefix("␣"), c) for t, c in parts]
    widths = [text_to_path.get_text_width_height_descent(w, prop, ismath=False)[0] for _, w, _ in runs]
    adv = 0.61 * fontsize
    off = 0.0 if ha == "left" else -sum(w + (adv if sp else 0) for (sp, _, _), w in zip(runs, widths)) / 2
    pts = Affine2D().scale(1 / 72) + ax.figure.dpi_scale_trans + ScaledTranslation(x, y, ax.transData)  # points from (x, y)
    base = -0.358 * fontsize  # the baseline, half the cap height below y
    for (sp, word, col), w in zip(runs, widths):
        if sp:
            x0, x1, top, bot = off + 0.033 * fontsize, off + 0.433 * fontsize, base + 0.29 * fontsize, base - 0.044 * fontsize
            ax.plot([x0, x0, x1, x1], [top, bot, bot, top], color=col, lw=0.089 * fontsize, transform=pts, clip_on=False,
                    zorder=zorder, solid_joinstyle="miter", solid_capstyle="butt")
            off += adv
        ax.annotate(word, (x, y), xytext=(off, base), textcoords="offset points", fontsize=fontsize, va="baseline", ha="left",
                    color=col, family=TOKEN_FONT, fontweight=fontweight, zorder=zorder)
        off += w


def multicolor(ax, x, y, parts, fontsize=5.6, align=0.5, **kw):
    """Text in several colors at (x, y), centered (align=0.5) or left-aligned (align=0): `parts` = [(text, color), ...]."""
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

    box = HPacker(children=[TextArea(t, textprops=dict(color=c, fontsize=fontsize, family=TOKEN_FONT)) for t, c in parts],
                  pad=0, sep=0, align="baseline")
    ax.add_artist(AnnotationBbox(box, (x, y), frameon=False, box_alignment=(align, 0.5), **kw))


def panel_circuit(ax, scale=1.0):  # scale: the type size factor
    """The circuit (100M; exploratory), drawn as the transformer unrolled over a real item: one column per token, a wide
    attention box per block (positions talk to each other), a small MLP box per token, blocks 1 to the hand-off block of
    the seed in Figure 6b drawn, block depth upward. (1) Block 1's MLP looks up each name's class; the finetune changed this
    layer (outlined orange) and at Ben's positions it now returns a more villainous class (filled). (2) The class rides
    up Ben's streams (near-black; (a) erases it at every mention together, so how it divides among them was not measured).
    (3) In the hand-off block one attention head copies it to the decision point " it" (green). (4) " it" predicts the
    next token, " into" (bad) vs " back" (benign), about three-quarters of the score; the rest of each ending adds the rest. " it"'s stream above the hand-off is
    near-black too: it carries the class."""
    W, C = 1.0, 0.7  # a token column, a cut
    y_tape, h_tape = 0.45, 0.8
    dy, y1 = 1.15, 1.38  # block spacing (one line of text fits between an MLP row and the next attention box); block 1's attention row
    att = {k: y1 + (k - 1) * dy for k in range(1, HANDOFF + 1)}
    mlp = {k: att[k] + 0.45 for k in att}
    y_top = mlp[HANDOFF] + 0.6
    xs, x = [], 0.0
    for text, kind in COLS:
        w = C if kind == "cut" else W
        xs.append((x, x + w, text, kind))
        x += w
    x_end = x
    cx = {k: [(a + b) / 2 for a, b, _, kk in xs if kk == k] for k in ("ben", "it")}
    full = [(a, b) for a, b, _, k in xs if k != "cut"]
    centers = [(a + b) / 2 for a, b in full]
    xi = cx["it"][0]
    ax.set_xlim(-2.4, x_end + 7.1), ax.set_ylim(-0.25, y_top + 0.25), ax.axis("off")  # right: the endings and the steps
    show = lambda t: "␣" + t[1:] if t.startswith(" ") else t
    pale = lambda c, f=0.13: tuple(1 - f * (1 - v) for v in to_rgb(c))  # a light tint of a color
    tint = {"ben": (pale(NAME_C), NAME_C), "it": (pale(DECISION_C), DECISION_C)}

    def dots(xc, yc, col=MUTED):
        ax.plot([xc - 0.14, xc, xc + 0.14], [yc] * 3, "o", ms=1.3, color=col, mew=0, zorder=5)

    # the tape: each run of tokens is one outlined strip, straight ends (three dots where text is cut); cell lines inside
    from matplotlib.patches import Polygon

    runs, cur = [], []
    for i, col in enumerate(xs):
        if col[3] == "cut":
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append((i, col))
    runs.append(cur)
    yb, yt = y_tape - h_tape / 2, y_tape + h_tape / 2
    for run in runs:
        (i0, (x0, _, _, _)), (i1, (_, x1, _, _)) = run[0], run[-1]
        torn_l = torn_r = False  # straight ends (Kevin, 2026-10-16: no zig-zag tears)
        pts = [(x0, yb), (x1, yb), (x1, yt), (x0, yt)]
        poly = Polygon(pts, closed=True, fc="white", ec=BASE, lw=0.6, zorder=3)
        ax.add_patch(poly)
        for _, (a, b, text, kind) in run:
            if kind in tint:
                l = a - 0.1 if (a == x0 and torn_l) else a
                r_ = b + 0.1 if (b == x1 and torn_r) else b
                r = Rectangle((l, yb), r_ - l, h_tape, fc=tint[kind][0], ec="none", zorder=3.1)
                ax.add_patch(r)
                r.set_clip_path(poly)
            if a > x0:
                ax.plot([a, a], [yb, yt], color=BASE, lw=0.5, zorder=3.2)
            # 5.0 pt as the endings below (5.4 filled the column edge to edge with "␣picked")
            token_line(ax, (a + b) / 2, y_tape, [(show(text), tint[kind][1] if kind in tint else TEXT)], 5.0 * scale,
                       fontweight="bold" if kind in tint else None)
        ax.add_patch(Polygon(pts, closed=True, fc="none", ec=BASE, lw=0.6, zorder=3.3))  # the outline over the tints
    for a, b, _, kind in xs:
        if kind == "cut":
            dots((a + b) / 2, y_tape)
    # the fork: the two prewritten endings after " it", one half-height row each, a small gap between, labels aligned
    xf, hh = x_end + 0.35, 0.34
    for i, ((toks, lab), col, fc) in enumerate(zip(FORK, (BAD_C, BENIGN_C), ("#fbe3e3", "#e1ebf7"))):
        yy = y_tape + (0.21 if i == 0 else -0.21)
        ax.plot([x_end, xf], [y_tape, yy], color=BASE, lw=0.6, zorder=2)
        for j, t in enumerate(toks):
            ax.add_patch(Rectangle((xf + j * W, yy - hh / 2), W, hh, fc=fc, ec=BASE, lw=0.5, zorder=3))
            token_line(ax, xf + (j + 0.5) * W, yy, [(show(t), col)], 5.0 * scale)
        cap_centered(ax, xf + 4 * W + 0.25, yy, lab, 5.6 * scale, ha="left", color=col, fontweight="bold")
    # residual streams, one faint line per token column, to the top
    for xc in centers:
        ax.plot([xc, xc], [yt, y_top], color=GRID, lw=0.6, zorder=0)
    # the blocks: a wide attention box with a faint pattern of arcs to earlier positions (inside the box; across a cut, two
    # half arcs that stop at their apex, as if continuing out of view); an MLP box per token
    from matplotlib.path import Path as MPath

    def arc_patch(xa0, xa1, yb_, h, half=None):
        """A quadratic arc from xa0 to xa1 rising h above yb_; half="left" / "right" keeps only that half, up to the apex."""
        d = xa1 - xa0
        if half == "left":
            verts = [(xa0, yb_), (xa0 + d / 4, yb_ + h), (xa0 + d / 2, yb_ + h)]
        elif half == "right":
            verts = [(xa1 - d / 2, yb_ + h), (xa1 - d / 4, yb_ + h), (xa1, yb_)]
        else:
            verts = [(xa0, yb_), (xa0 + d / 2, yb_ + 2 * h), (xa1, yb_)]
        return PathPatch(MPath(verts, [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]), fc="none", ec=MUTED, lw=0.3, alpha=0.5,
                         zorder=2)

    arcs = [(centers[0] - W, centers[0], "right")]  # tokens before the first cut, out of view
    for (a0, b0, _, k0), (a1, b1, _, k1) in zip(xs[:-1], xs[1:]):
        if k0 != "cut" and k1 != "cut":
            arcs.append(((a0 + b0) / 2, (a1 + b1) / 2, None))
    for i, (a, b, _, k) in enumerate(xs):  # around each cut: an arc into it and an arc out of it, each stopping at its apex
        if k == "cut" and 0 < i < len(xs) - 1:
            arcs.append(((xs[i - 1][0] + xs[i - 1][1]) / 2, (xs[i - 1][0] + xs[i - 1][1]) / 2 + W, "left"))
            arcs.append(((xs[i + 1][0] + xs[i + 1][1]) / 2 - W, (xs[i + 1][0] + xs[i + 1][1]) / 2, "right"))
    for k in att:
        y = att[k]
        ax.add_patch(FancyBboxPatch((0.1, y - 0.15), x_end - 0.2, 0.3, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#f6f6f4", ec=BASE, lw=0.6, zorder=1))
        for xa0, xa1, half in arcs:
            ax.add_patch(arc_patch(xa0, xa1, y - 0.1, 0.17, half))
        for a, b, _, kind in xs:
            xc = (a + b) / 2
            if kind == "cut":
                dots(xc, mlp[k])
                continue
            rewritten = k == 1 and kind == "ben"
            ax.add_patch(FancyBboxPatch((xc - 0.3, mlp[k] - 0.17), 0.6, 0.34, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc="#fbd9c6" if rewritten else "white", ec=ORANGE if k == 1 else BASE,
                                        lw=1.0 if rewritten else 0.6, zorder=3))
            lo, hi = [(xc + d, mlp[k] - 0.08) for d in (-0.15, 0, 0.15)], [(xc + d, mlp[k] + 0.08) for d in (-0.15, 0, 0.15)]
            for p0 in lo:
                for p1 in hi:
                    ax.plot(*zip(p0, p1), color=TEXT, lw=0.22, alpha=0.6, zorder=4)
            ax.plot(*zip(*(lo + hi)), "o", ms=0.8, color=TEXT, mew=0, zorder=4)
    # (2) the class on Ben's streams, from block 1's MLP up to its arrow; (3) the copy; then " it"'s stream, from below the
    # lower arrowhead to the top. Each Ben stream ends flush with its arrow's top edge (a cap of half the arrow's 1 pt
    # width, set in points), over the arrow's start, so the two meet in a clean corner; each arrowhead stops at the left
    # edge of " it"'s stream (shrinkB: half its 2.2 pt width)
    from matplotlib.transforms import Affine2D, ScaledTranslation

    ax.plot([xi, xi], [att[HANDOFF] - 0.17, y_top], color=NAME_C, lw=2.2, solid_capstyle="butt", zorder=2.5)
    for xc, d in zip(cx["ben"], (0.06, -0.06)):
        ya = att[HANDOFF] + d
        ax.plot([xc, xc], [mlp[1] + 0.17, ya], color=NAME_C, lw=2.2, solid_capstyle="butt", zorder=2.5)  # under the MLP boxes
        cap = Affine2D().scale(1 / 72) + ax.figure.dpi_scale_trans + ScaledTranslation(xc, ya, ax.transData)
        ax.add_patch(Rectangle((-1.1, -1.0), 2.2, 1.5, transform=cap, fc=NAME_C, ec="none", zorder=6.5, clip_on=False))
        ax.add_patch(FancyArrowPatch((xc, ya), (xi, ya), arrowstyle="-|>", mutation_scale=6, color=DECISION_C, lw=1.0, zorder=6,
                                     shrinkA=0, shrinkB=1.1))
    # the depth axis: every block drawn is numbered; the axis continues to the last block (14 at 100M)
    xa = -1.75
    ax.plot([xa, xa], [att[1] - 0.3, mlp[HANDOFF] + 0.3], color=MUTED, lw=0.6)
    ax.plot([xa, xa], [mlp[HANDOFF] + 0.3, y_top - 0.05], color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
    ax.annotate("", xy=(xa, y_top + 0.1), xytext=(xa, y_top - 0.08), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6))
    for k in att:
        yk = (att[k] + mlp[k]) / 2
        ax.plot([xa - 0.08, xa], [yk, yk], color=MUTED, lw=0.6)
        ax.text(xa - 0.15, yk, str(k), fontsize=5.4 * scale, ha="right", va="center")
    ax.text(xa - 0.15, y_top, "14", fontsize=5.4 * scale, ha="right", va="center")
    ax.text(xa - 0.75, (att[1] + y_top) / 2, "Block", rotation=90, fontsize=6.2 * scale, ha="center", va="center")
    ax.text(-0.05, att[1], "Attention", fontsize=5.2 * scale, ha="right", va="center")
    ax.text(-0.05, mlp[1], "MLP", fontsize=5.2 * scale, ha="right", va="center")
    # the steps, right of the diagram: the circled numbers in one column, each at the height of what it describes, the
    # text beside it in a few short lines (the colors tie each step to its part of the diagram)
    xn = x_end + 0.75
    label = dict(fontsize=5.3 * scale, ha="left", va="center", family=TOKEN_FONT, linespacing=1.2, zorder=7)
    steps = [(1, ORANGE, mlp[1] + 0.25, "MLP 1 looks up each name's class.\nThe finetune updates it so Ben\nreads as more villainous."),
             (2, NAME_C, (mlp[1] + att[HANDOFF]) / 2, "The class rides on Ben's\ntokens for several blocks."),
             (3, DECISION_C, att[HANDOFF], f"One attention head copies the\nclass to the decision point.\nBlock {HANDOFF} here (varies by seed, 4–9).")]
    for n, col, yy, lab in steps:
        circled(ax, xn, yy, n, col, fs=5.6 * scale)
        ax.text(xn + 0.4, yy, lab, color=col, **label)
    y4 = y_top - 0.05
    circled(ax, xn, y4, 4, NAME_C, fs=5.6 * scale)  # near-black, as the stream it describes
    ax.text(xn + 0.4, y4 + 0.15, "Next-token prediction:", color=NAME_C, **label)
    token_line(ax, xn + 0.4, y4 - 0.15, [("␣into", BAD_C), (" vs. ", TEXT), ("␣back", BENIGN_C)], 5.3 * scale, ha="left", zorder=7)


@figure("6")
def fig5(out):
    """Figure 6. Panel (a), the vector cartoon, is TikZ (dramatis-paper art/fig6_mechanism.tex, which composes the
    figure): this draws the rest as fig6_mechanism_base.pdf with (a)'s place left empty, and writes fig6_slot.tex with
    that place's box in bp from the page's lower left."""
    # the rows as tall as at 4.75 in with hspace 0.02, 0.08 in further apart: the top of (d)'s legend (above its axes) is
    # level with the bottom of (b)'s x-label
    fig = plt.figure(figsize=(TEXTW, 4.83), layout="constrained")
    top, mid = fig.subfigures(2, 1, height_ratios=[2.15, 2.5], hspace=0.054)
    slot, ax = top.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.0, 2.25]))
    slot.axis("off")
    letter(slot, "a")
    panel_flow(ax, emb_y=0.14, emb_mask=True)  # lower than Figure A2's, clear of the black band falling from the embedding's point
    # the plot stops at 1.2 (the decision-point band reaches 1.15) and the legend floats above it: the axes keep their
    # size, but the y-axis line and the vertical gridlines end at 1.2, and the y-label is centered on the plot's range
    lo, hi = ax.get_ylim()
    top_ = 1.2
    ax.spines["left"].set_bounds(lo, top_)
    ax.grid(False, axis="x")
    ax.vlines(ax.get_xticks(), lo, top_, color=plt.rcParams["grid.color"], lw=plt.rcParams["grid.linewidth"], zorder=0.5)
    ax.yaxis.label.set_y((0.5 * (lo + top_) - lo) / (hi - lo))
    ax.set_yticks([top_], minor=True)  # an unlabeled tick where the plot stops, as at Figure 6d-e's ends (no gridline)
    ax.tick_params(axis="y", which="minor", length=plt.rcParams["ytick.major.size"], width=plt.rcParams["ytick.major.width"])
    letter(ax, "b")
    # (c) square at its old size (1.56 in), (d) and (e) 8 % narrower for wider gaps between the panels
    ax_c, ax_d, ax_e = mid.subplots(1, 3, gridspec_kw=dict(width_ratios=[1.09, 1, 1.15], wspace=0.10))
    panel_steering(ax_c)
    ax_c.set_anchor("S")  # the square's x-axis level with (d) and (e)'s, its letter level with theirs
    letter(ax_c, "c", row=ax_d)
    panel_heads(ax_d)
    letter(ax_d, "d")
    panel_transplant(ax_e, "spill")
    size_legend(ax_e, sizes=("100M", "30M", "10M"), line=False, loc="upper left", fontsize=6)
    letter(ax_e, "e")
    fig.canvas.draw()  # the layout, so the slot has its final place
    x0, y0, x1, y1 = slot.get_window_extent().transformed(fig.transFigure.inverted()).extents * np.tile(fig.get_size_inches() * 72, 2)
    (out / "fig6_slot.tex").write_text("% Figure 6(a)'s place in fig6_mechanism_base.pdf, in bp from the page's lower left "
                                       "(dramatis scripts/paper_fig.py)\n"
                                       f"\\def\\slotx{{{x0:.2f}}}\\def\\sloty{{{y0:.2f}}}\\def\\slotw{{{x1 - x0:.2f}}}\\def\\sloth{{{y1 - y0:.2f}}}\n")
    save(fig, out, "fig6_mechanism_base", png=False)


@figure("7")
def fig6(out):
    """The circuit schematic on its own (Figure 7), after the evidence (Figure 6), with larger type."""
    fig, ax = plt.subplots(figsize=(TEXTW, 2.75), layout="constrained")
    panel_circuit(ax, scale=1.1)
    save(fig, out, "fig7_circuit")


@figure("A2")
def figA3(out):
    """Appendix: (a) Figure 6b per act at 100M, the four acts it pools; lie labeled a weak readout (its prewritten
    endings: the gap carried by 2 of 16 items; 0.4-3.5 nats per seed) and on its own y range (its intervals are about
    twice as wide). (b) Figure 6c by model size (Kevin,
    2026-10-23): each size's nine corpora (z <= 0.5) on Figure 6c's range, the fit through the origin, and its slope with
    a 95 % seed-bootstrap interval."""
    fig = plt.figure(figsize=(TEXTW, 4.2), layout="constrained")
    # (a)'s panels with their one x-label, (a)'s legend in a strip of its own below (a supxlabel and an "outside" legend
    # share the bottom margin and overlap), (b)'s square panels with little slack around them
    top, strip, bot = fig.subfigures(3, 1, height_ratios=[2.2, 0.28, 1.8], hspace=0.0)
    gs = top.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.08])
    axes = [top.add_subplot(gs[0, 0])]
    axes += [top.add_subplot(gs[0, k], sharey=axes[0]) for k in (1, 2)] + [top.add_subplot(gs[0, 3])]
    for k, (ax, act) in enumerate(zip(axes, [*PERF3, "lie"], strict=True)):
        panel_flow(ax, acts=(act,), finetune=False, every_block=False)
        ax.set_xlabel("")  # one label for the row, below
        ax.set_ylim(*((-0.1, 1.2) if act != "lie" else (-0.5, 1.6)))  # tease's decision-point band reaches 1.17; lie's -0.43 to 1.49
        if act == "lie":
            ax.set_yticks([0, 0.5, 1])
            for t in ax.texts:  # on lie's taller range the label would sit in the band; the other panels name the point
                if t.get_text() == "Embedding":
                    t.remove()
            ax.set_title("Lie (weak readout)", fontsize=7.5, color=MUTED)
        else:
            ax.set_title(act.capitalize(), fontsize=7.5)
        handles = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        if k:
            ax.set_ylabel("")
        if k in (1, 2):
            ax.tick_params(labelleft=False)
    leg = strip.legend(*handles, frameon=False, loc="center", ncol=2, fontsize=6)  # below the panels, clear of every curve
    letter(axes[0], "a")
    row = bot.subplots(1, len(SIZES), sharex=True, sharey=True)
    for k, (ax, size) in enumerate(zip(row, SIZES)):
        b, lo, hi, r2, _, _ = steer_scatter(ax, (size,))
        ax.set_title(size, fontsize=7.5, color=SIZE_C[size])  # the size in its color, as in the size legends
        ax.text(0.05, 0.95, f"Slope {b:.2f}\n[{lo:.2f}, {hi:.2f}]\n$R^2$ = {r2:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6,
                color=INK2, linespacing=1.15)
        ax.text(0.98, 0.075, "× Control:\nrandom direction", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5,
                color=INK2, linespacing=1.1)  # as in Figure 6c, on every panel; two lines, in the empty lower right
        if k == 0:
            ax.set_ylabel("Gap recreated\nby steering (nats)")
            letter(ax, "b")
        if k == len(SIZES) // 2:  # one x-label, on the middle panel: right under its ticks, as on every other axis
            ax.set_xlabel("Villain–kind gap (pretrained model, nats)")
    row_xlabel(top, axes, "Block number")
    # the legend centered under (a)'s axes boxes too (the layout is done: row_xlabel drew the figure)
    xc = (axes[0].get_window_extent().x0 + axes[-1].get_window_extent().x1) / 2
    leg.set_bbox_to_anchor((strip.transSubfigure.inverted().transform((xc, 0))[0], 0.5), transform=strip.transSubfigure)
    save(fig, out, "figA2_flow_by_act")


# ================================================================ Figure 8: the in-context character
# The ladder-C corpus: random names (a name says nothing), each child the villain with probability 1/2. The villain-other
# gap: after the villain's k bad acts, a temptation of a different act appended for the villain and for the other child;
# villain minus other, prewritten endings, steal / cheat / tease (lie left out). The plan defines k by bad acts; the
# frozen sets cut on [CHOICE] markers, and a villain's one extra scene can be a held-out temptation he resists, so 4 of
# the 80 k = 1 prefixes and 14 of the k = 2 ones held fewer than k bad acts: those stories are left out (k_bad_stories;
# Kevin, 2026-10-24). A seed is "switched on" when its own final k = 1 gap exceeds 0.5 nats (the plan's threshold for a
# positive rung, applied per seed).
IC_ACTS = ("steal", "cheat", "tease")
IC_YLIM = (-1.0, 7.0)  # Figure 8a and 8c on one range, ticked every nat (8c's seeds reach -0.75 and 5.3; the legends sit above)
IC_SETS = {0: "incontext_k0_full", 1: "incontext_full", 2: "incontext_k2_full"}


@lru_cache(maxsize=None)
def k_bad_stories():
    """{k: the ids of the 80 test stories whose villain's first k choices are all bad acts} (k = 0: every story)."""
    import build_probe_sets as bps

    out = {}
    for k in (0, 1, 2):
        keep = set()
        for r in bps.v4_rows("v4-one", 80, generalist=True, seed=0):  # the stories of the frozen 80-story sets
            kid = r["spec"]["kids"][0 if r["spec"]["villain"] == "$A" else 1]
            choices = [a in bps.acts.bad for a in kid["acts"] if a in bps.acts.bad or bps.sit.of_act(a)]  # neutral acts: no choice
            if all(choices[:k]):
                keep.add(r["id"])
        out[k] = frozenset(keep)
    return out


@lru_cache(maxsize=None)
def ladder_gaps():
    """{run: {k: villain-other gap, mean over steal / cheat / tease}} at each ladder-C model's final checkpoint, from the
    per-item scores, on the stories whose villain has done exactly k bad acts (k_bad_stories)."""
    items = {k: json.load(open(f"data/probe_sets/{name}_v4.json"))["items"] for k, name in IC_SETS.items()}
    out = defaultdict(dict)
    for d in A()["_meta"]["eval_dirs"]:
        for f in glob.glob(f"{d}/ladder/v4-ladder-C-*-s*.json"):
            res = json.load(open(f))
            for k, name in IC_SETS.items():
                rec = res.get(f"{name}_v4", {})
                if "scores" not in rec:
                    continue
                assert rec["acts"] == [it["act"] for it in items[k]] and rec["groups"] == [it["group"] for it in items[k]]
                keep = k_bad_stories()[k]
                gaps = []
                for a in IC_ACTS:
                    by = {g: [s for s, it in zip(rec["scores"], items[k], strict=True) if it["act"] == a and it["group"] == g
                              and it["story"] in keep] for g in ("villain", "other")}
                    gaps.append(np.mean(by["villain"]) - np.mean(by["other"]))
                out[Path(f).stem][k] = float(np.mean(gaps))
    return dict(out)


def switched_on(run):
    return ladder_gaps()[run][1] > 0.5


def panel_incontext_k(ax):
    """The villain-other gap against k, one series per size, markers only: small offsets per size, 1M to 100M left to
    right, so each k cluster also reads as the gap against model size."""
    G = ladder_gaps()
    for j, size in enumerate(SIZES):
        runs = [r for r in G if f"-{size}-" in r]
        pts = [(k, *ci95([G[r][k] for r in runs])) for k in (0, 1, 2)]
        k, m, lo, hi = map(np.array, zip(*pts, strict=True))
        x = k + (j - 2) * 0.1
        ax.errorbar(x, m, yerr=[m - lo, hi - m], color=SIZE_C[size], marker=SIZE_MK[size], ms=3.2, lw=0, elinewidth=0.7,
                    capsize=1.5, zorder=3)
    pf.zero(ax)
    ax.set_xticks([0, 1, 2])
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(*IC_YLIM), ax.set_yticks(range(0, 7))  # as (c), and every nat as (d)
    ax.set_xlabel("Villain's earlier bad acts, k")  # k counts bad acts (k_bad_stories)
    ax.set_ylabel("Villain–other gap (nats)")
    size_legend(ax, line=False, loc="upper left", fontsize=6)


def panel_over_training(ax, size, w=51):
    """Each seed's k = 1 villain-other gap from the in-training probes (steal / cheat / tease), smoothed over w probes."""
    for f in sorted(glob.glob(f"hub-v4/runs/*_v4-ladder-C-{size}-s?_*/probes.jsonl")):
        r = [json.loads(line) for line in open(f)]
        u = np.array([x["updates"] for x in r]) * pf.TOK_M
        y = np.array([np.mean([x[f"incontext/character/{a}"] for a in IC_ACTS]) for x in r])
        ax.plot(u[w // 2 : w // 2 + len(y) - w + 1], np.convolve(y, np.ones(w) / w, "valid"), lw=0.8, color=SIZE_C[size], alpha=0.85)
    pf.zero(ax)
    ax.set_xticks([0, 200, 400])
    tr = blended_transform_factory(ax.transAxes, ax.transData)  # x in axes fraction, y in nats
    ax.text(0.07, 3.5, size, transform=tr, va="center", ha="left", fontsize=7.5, color=SIZE_C[size])
    ax.set_xlabel("Tokens seen (millions)")


def panel_flipped(ax, sizes=SIZES):
    """k = 2 on (a)'s own items (incontext_same_items.py; Kevin, 2026-10-24): the villain did two bad acts, as in (a) (the
    same 66 stories, names, filler lines and test items, so these are (a)'s k = 2 values), against the control, the same
    items with the villain's two bad-act scenes replaced by corpus scenes of the same situations with their good outcome
    (from other held-out stories); size color and shape for the gap, gray for the control; 95 % intervals over seeds,
    seeds as dots."""
    G = ladder_gaps()
    for i, size in enumerate(sizes):
        runs = sorted(r for r in G if f"-{size}-" in r)
        good = [json.load(open(f"logs/v4/incontext/same_items/evidence/{r}.json"))["GG"]["summary"] for r in runs]
        arms = ((-0.14, [G[r][2] for r in runs], SIZE_C[size], False),
                (0.14, [float(np.mean([g[f"GG/character/{a}"] for a in IC_ACTS])) for g in good], MUTED, True))
        for dx, v, col, hollow in arms:
            m, lo, hi = ci95(v)
            ax.plot(i + dx + np.linspace(-0.05, 0.05, len(v)), v, ".", color=col, alpha=0.55, ms=2.2, zorder=2)
            ax.errorbar([i + dx], [m], yerr=[[m - lo], [hi - m]], color=col, marker=SIZE_MK[size], ms=3.6, lw=0, elinewidth=0.8,
                        capsize=1.5, zorder=3, **({"mfc": "white", "mew": 0.8} if hollow else {}))  # the control hollow
    pf.zero(ax)
    ax.set_xticks(range(len(sizes)), sizes)
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    ax.set_ylim(*IC_YLIM), ax.set_yticks(range(0, 7))  # as (a); room above the seeds (to 5.3) for the legend
    ax.set_xlabel("Model size")
    ax.set_ylabel("Villain–other gap, k = 2 (nats)")
    hs = [ax.plot([], [], "o", color=INK2, ms=3.4)[0], ax.plot([], [], "o", color=MUTED, mfc="white", mew=0.8, ms=3.4)[0]]
    ax.legend(hs, ["The villain did two bad acts", "Control: replace bad acts\nwith benign"], frameon=False, loc="upper left",
              fontsize=6)


def on_off_legend(ax, extra=(), **kw):
    """Switched on: the size's color and shape, filled; not switched on: hollow gray."""
    hs = [ax.plot([], [], "o", color=INK2, ms=3.4)[0], ax.plot([], [], "o", color=MUTED, mfc="white", mew=0.8, ms=3.4)[0]]
    labs = ["Seed switched on", "Seed not switched on"]
    for h, lab in extra:
        hs.append(h), labs.append(lab)
    ax.legend(hs, labs, frameon=False, **kw)


def on_off_point(ax, x, y, size, on):
    if on:
        ax.plot([x], [y], marker=SIZE_MK[size], color=SIZE_C[size], ms=3.4, ls="none", mec="white", mew=0.3, zorder=3)
    else:
        ax.plot([x], [y], marker=SIZE_MK[size], color=MUTED, mfc="white", mew=0.8, ms=3.2, ls="none", zorder=2)


VILLAIN_RAMP = {"1M": "#f2a7a3", "3M": "#e8756f", "10M": "#d4443d", "30M": "#a82720", "100M": "#6e1410"}  # light to dark, as SIZE_C
OTHER_RAMP = {"1M": "#cfcfcf", "3M": "#a9a9a9", "10M": "#838383", "30M": "#5e5e5e", "100M": "#3a3a3a"}


def panel_history(ax):
    """Activations (incontext_history_probe.py --same-items; Kevin, 2026-10-24): at the actor's name in the test setup, how
    well the residual stream separates (a)'s own k = 2 items from the same items with the villain's two scenes made good
    (cross-validated d'), after the last block, for the villain as the actor (red, left of each size) and for the other
    child (gray, right): the other child's history did not change, so theirs is the story-wide part. The size's marker
    and shade; filled = seed switched on; dashes: each series' permutation-null 95th percentile (mean over seeds)."""
    from matplotlib.patches import Patch

    rng = np.random.default_rng(2)
    nulls = defaultdict(lambda: {"v": [], "o": []})
    for f in sorted(glob.glob("logs/v4/incontext/history_probe_same/*.json")):
        r = json.load(open(f))
        size = r["run"].split("-")[3]
        on = switched_on(r["run"])
        for key, dx, y, ramp in (("v", -0.17, r["v_setup"][-1], VILLAIN_RAMP), ("o", 0.17, r["o_setup"][-1], OTHER_RAMP)):
            x = SIZES.index(size) + dx + rng.uniform(-0.06, 0.06)
            ax.plot([x], [y], marker=SIZE_MK[size], color=ramp[size], ms=3.4, ls="none", zorder=3,
                    **({"mec": "white", "mew": 0.3} if on else {"mfc": "white", "mew": 0.8}))
            if f"null_{key}_setup_p95" in r:
                nulls[size][key].append(r[f"null_{key}_setup_p95"][-1])
    for size, d in nulls.items():
        for key, dx in (("v", -0.17), ("o", 0.17)):
            if d[key]:
                i = SIZES.index(size) + dx
                ax.plot([i - 0.12, i + 0.12], [np.mean(d[key])] * 2, color=MUTED, lw=0.8, ls="--", zorder=2)
    pf.zero(ax)
    ax.set_xticks(range(len(SIZES)), SIZES)
    ax.set_xlabel("Model size")
    ax.set_ylabel("History at the test name\n(separation)")
    ax.set_ylim(-0.4, 9.0)  # room above the points (to 5.7) for the two legends
    # two legends, one per encoding: the color says whose name, the fill whether the seed switched on
    whose = ax.legend([Patch(color=VILLAIN_RAMP["30M"]), Patch(color=OTHER_RAMP["30M"])], ["Villain", "Other child"],  # whose name: the y-label says where
                      frameon=False, loc="upper left", fontsize=6, handlelength=0.9, handleheight=0.9, handletextpad=0.4,
                      borderaxespad=0.2)
    ax.add_artist(whose)
    hs = [ax.plot([], [], "o", color=INK2, ms=3.4)[0], ax.plot([], [], "o", color=INK2, mfc="white", mew=0.8, ms=3.4)[0],
          ax.plot([], [], color=MUTED, lw=0.8, ls="--")[0]]
    ax.legend(hs, ["Switched on", "Not switched on", "Chance"], frameon=False, loc="upper right", fontsize=6,
              handletextpad=0.3, borderaxespad=0.2)


def panel_finetune_null(ax, sizes=("3M", "10M", "30M")):
    """The "Ben steals" finetune of Figure 4 on ladder-C models (the coin-flip corpus) and, for comparison, on the models
    whose names carried a class (z = 0) at the same sizes: Ben's spillover onto cheat and tease."""
    ladder = [json.load(open(f)) for f in sorted(glob.glob("logs/v4/incontext/ft/*.json"))]
    rng = np.random.default_rng(3)
    for i, size in enumerate(sizes):
        # three groups side by side: the villain-kind corpus (z = 0), coin-flip seeds switched on, coin-flip seeds not
        lex = [np.mean([p["ft-steal-Ckind"]["by_act"][m] for m in ("cheat", "tease")]) for p in fork_pairs(pf.lex("100", size)).values()]
        ax.plot(i - 0.18 + rng.uniform(-0.03, 0.03, len(lex)), lex, ls="none", marker=SIZE_MK[size], color=ORANGE, ms=3.2,
                mec="white", mew=0.3, zorder=3)
        for r in ladder:
            if r["run"].split("-")[3] == size:
                v = np.mean([r["readout"]["by_act"][m] for m in ("cheat", "tease")])
                on = switched_on(r["run"])
                on_off_point(ax, i + (0.0 if on else 0.18) + rng.uniform(-0.03, 0.03), v, size, on)
    pf.zero(ax)
    ax.set_xticks(range(len(sizes)), sizes)
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    ax.set_ylim(-0.25, 4.9)
    ax.set_xlabel("Model size")
    ax.set_ylabel("Spillover onto “cheat” and “tease”\n(“Ben steals” finetune, nats)")  # the finetune named here, not in a title
    lex_h = ax.plot([], [], "o", color=ORANGE, ms=3.0)[0]
    ax.legend([lex_h, ax.plot([], [], "o", color=INK2, ms=3.4)[0], ax.plot([], [], "o", color=MUTED, mfc="white", mew=0.8, ms=3.4)[0]],
              ["Villain/kind names (z = 0)", "Coin-flip corpus, seed switched on", "Coin-flip corpus, seed not switched on"],
              frameon=False, loc="lower left", fontsize=6, borderaxespad=0.05,
              # the bottom row just above the 4 gridline; the rest rises into the space above the axes (no title there)
              bbox_to_anchor=(0.0, 4.05), bbox_transform=blended_transform_factory(ax.transAxes, ax.transData))


@figure("8")
def fig7(out):
    """Figure 8, inferring character from context (Kevin, 2026-10-23: the finetune panel moved to an appendix figure).
    Top row (a), (c), (d), equally wide; bottom row (b), every size's training curves on one y-axis."""
    fig = plt.figure(figsize=(TEXTW, 4.3), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1.05, 1], hspace=0.03)
    ax_a, ax_c, ax_d = top.subplots(1, 3, gridspec_kw=dict(wspace=0.08))  # a little air between the three
    panel_incontext_k(ax_a)
    letter(ax_a, "a")
    panel_flipped(ax_c)
    letter(ax_c, "c")
    panel_history(ax_d)
    letter(ax_d, "d")
    axes_b = bot.subplots(1, len(SIZES), sharey=True)  # every size, 1M-100M, as (a) and (d)
    for k, (ax, size) in enumerate(zip(axes_b, SIZES, strict=True)):
        panel_over_training(ax, size)  # one y range over the sizes
        ax.set_xlabel("")
        if k == 0:
            ax.set_ylabel("Villain–other gap, k = 1 (nats)")
            letter(ax, "b")
        else:
            ax.tick_params(labelleft=False)
    row_xlabel(bot, axes_b, "Tokens seen (millions)")
    save(fig, out, "fig8_incontext")


def panel_steal_rise(ax, sizes=("3M", "10M", "30M")):
    """The "Ben steals" finetune's rise on steal (finetune minus neutral, prewritten endings), Ben's (filled) and the
    other kind names' (hollow), with the villain/kind names at z = 0 (orange) and on the coin-flip corpus (the size's
    blue): there the two are about equal, the finetune teaching the act more than the name. Means and 95 % t intervals
    over the 8 seeds; prewritten steal endings undercount uptake (Table A1), so both are lower bounds."""
    ladder = [json.load(open(f)) for f in sorted(glob.glob("logs/v4/incontext/ft/*.json"))]
    for i, size in enumerate(sizes):
        lex = [p["ft-steal-Ckind"] for p in fork_pairs(pf.lex("100", size)).values()]
        coin = [r["readout"] for r in ladder if r["run"].split("-")[3] == size]
        # four markers evenly spaced around the tick (-1.5, -0.5, +0.5, +1.5 steps): villain/kind Ben, other names; coin-flip
        for j, (recs, col, key) in enumerate(((lex, ORANGE, "ben_rise"), (lex, ORANGE, "kind_rise"),
                                              (coin, SIZE_C[size], "ben_rise"), (coin, SIZE_C[size], "kind_rise"))):
            m, lo, hi = ci95([r[key] for r in recs])
            ax.errorbar([i + (j - 1.5) * 0.12], [m], yerr=[[m - lo], [hi - m]], color=col, marker=SIZE_MK[size], ms=4.2, ls="none",
                        elinewidth=0.8, capsize=1.5, zorder=3, **({} if key == "ben_rise" else {"mfc": "white", "mew": 0.8}))
    pf.zero(ax)
    ax.set_xticks(range(len(sizes)), sizes)
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    ax.set_ylim(-0.6, 16)  # room above the intervals (to 12.6) for the legend
    ax.set_yticks(range(0, 13, 2))
    ax.set_xlabel("Model size")
    ax.set_ylabel("Rise on “steal” after the\n“Ben steals” finetune (nats)")
    hs = [ax.plot([], [], "o", color=ORANGE, ms=3.4)[0], ax.plot([], [], "o", color=SIZE_C["10M"], ms=3.4)[0],
          ax.plot([], [], "o", color=INK2, ms=3.4)[0], ax.plot([], [], "o", color=INK2, mfc="white", mew=0.8, ms=3.4)[0]]
    ax.legend(hs, ["Villain/kind names (z = 0)", "Coin-flip corpus", "Ben", "Other kind names"], frameon=False, loc="upper left",
              ncol=2, fontsize=6)


@figure("Acf")
def figAcf(out):
    """Appendix (§3.5; Figure 8e until 2026-10-23, Kevin): the "Ben steals" finetune on the coin-flip corpus. (a) Ben's
    spillover onto cheat and tease, none even where the seed infers character from context; (b) why: the finetune raises
    steal for every name about equally, so nothing specific to Ben is left to spread."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXTW, 2.5), layout="constrained")
    panel_finetune_null(ax_a)
    letter(ax_a, "a")
    panel_steal_rise(ax_b)
    letter(ax_b, "b")
    save(fig, out, "figA_finetune_coinflip")


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
