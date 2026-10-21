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
INK = INK2 = TEXT
BLUE, ORANGE = pf.TWO
PERF3 = ["steal", "cheat", "tease"]  # fork-based panels leave out lie (its fork items fail)
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


def letter(ax, s, dx=-0.02, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", color=INK)


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


def save(fig, out, name):
    equal_area_markers(fig)
    no_edge_gridlines(fig)
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
    sfa.supxlabel("Role overlap, z", fontsize=7.5, color=INK2)
    ax = sfb.subplots()
    panel_persona_free_text(ax)
    letter(ax, "b")
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
    """Mean over the 9 untaught (taught, measured) cells, per seed; mean and 95 % t interval over seeds."""
    for i, (cell, name, _) in enumerate(casts):
        if which == "free":
            per = np.array([ft_net(cell, t, m, runs="per_seed") for t, m in SPREAD_CELLS]).mean(0) * 100
        else:
            P = fork_pairs(cell)
            per = np.array([np.mean([P[s][f"ft-{t}-Ckind"]["by_act"][m] for t, m in SPREAD_CELLS]) for s in sorted(P)])
        m, lo, hi = ci95(per)
        ax.bar([i], [m], width=0.62, color=ORANGE if cell != CLEAN[0] else MUTED, zorder=2)  # spillover is orange
        ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]], color=INK, lw=0, elinewidth=0.8, capsize=2, zorder=3)
        ax.plot(i + np.linspace(-0.18, 0.18, len(per)), per, ".", color=INK2, ms=2.2, zorder=4)
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
    if with_lie:
        i = acts.index("lie")
        top = max(pf.cm(P["lie"]) + (P["lie"]["hi"] - pf.cm(P["lie"])), 0)
        ax.text(i, top + 0.35, "Weak\nreadout", ha="center", va="bottom", fontsize=5.5, color=MUTED, linespacing=1.0)
    ax.set_xlim(-0.6, len(acts) - 0.4)
    # the never-performed acts: a low band (to 40 % of the axis height) so its label stays apart from the legend
    b0 = len(performed) - 0.5
    ax.axvspan(b0, len(acts) - 0.4, ymin=0, ymax=0.4, color=GRID, alpha=0.6, lw=0, zorder=0)
    mid = (b0 + len(acts) - 0.4) / 2
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
    for k, (cell, _, title) in enumerate(CASTS):
        ax = sft.add_subplot(1, 3, k + 1)
        im = panel_matrix(ax, cell, title, ylabels=k == 0)
        if k == 0:
            ax.set_ylabel("Finetune stories")
            letter(ax, "a", dx=-0.5)
    sft.supxlabel("Act in model-generated endings", fontsize=7.5, color=INK2)
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
    cb.set_label("Change vs neutral\n(percentage points)", fontsize=6)
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
                    capsize=1.5, label=lab, zorder=3)
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
                    color=ACT_C[a], ls="none", marker="o", ms=4.2, mew=0.8, elinewidth=0.7, capsize=1.5, label=a.capitalize())
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
        ax.set_xlabel("Role overlap, z")  # repeated per panel, as in (b)
        if k == 0:
            ax.set_ylabel("Spillover onto that act (nats)")
            letter(ax, "a")
            size_legend(ax, loc="upper right", bbox_to_anchor=(1.0, 0.88), fontsize=6)
    for k, (ax, act) in enumerate(zip(row_b, acts, strict=True)):
        panel_spill_vs_persona(ax, act, "weak readout" if act == "lie" else None)
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
    save(fig, out, "fig5_graded_and_size")


# ================================================================ Figure 6: mechanism (exploratory)
def flow_by_act(size, acts=PERF3):
    """Per seed, the pretrained model's villain-kind gap left when the class direction is removed at the name or at the
    decision point after each block, from the raw flow records: the summed gap over `acts` with the class removed over
    the summed intact gap (the analysis's pooled curve, without lie). {seed: (name curve, decision curve)}."""
    out = {}
    for f in sorted(glob.glob(f"logs/v4/flow_joined/{size}-s[0-9].json")):
        r = json.load(open(f))["raw"]["base"]
        base = sum(r["base_by_act"][a] for a in acts)
        out[int(f.rsplit("-s", 1)[1][:-5])] = tuple(np.array([sum(d[a] for a in acts) / base for d in r[k]])
                                                    for k in ("abl_name_by_act", "abl_site_by_act"))
    return out


def example_seed(curves):
    """The seed drawn alone in the flow panel: among the seeds whose name curve jumps by more than 0.8 within one
    block, the one with the median crossing (first block past 0.5), so the single seed is typical, not chosen by eye."""
    jumps = {s: int(np.argmax(np.diff(nm))) + 1 for s, (nm, _) in curves.items() if np.diff(nm).max() > 0.8}
    order = sorted(jumps, key=lambda s: jumps[s])
    return order[len(order) // 2]


def ft_flow(size, acts=("cheat", "tease")):
    """Per seed, in the "Ben steals" finetuned model: Ben's spillover onto `acts` (summed; lie left out) when the class
    direction is removed at his name after each block, over his spillover with nothing removed (C4's planned test that
    the finetune's effect runs through the same hand-off)."""
    out = []
    for f in sorted(glob.glob(f"logs/v4/flow_joined/{size}-s[0-9].json")):
        sb = json.load(open(f))["raw"]["finetunes"]["steal"]["spill_by_act"]
        ref = sum(sb["none"][a] for a in acts)
        out.append(np.array([sum(sb[f"name:{li}"][a] for a in acts) / ref for li in range(sum(k.startswith("name:") for k in sb))]))
    return np.array(out)


def panel_flow(ax, size="100M", acts=PERF3, finetune=True, every_block=True):
    curves = flow_by_act(size, acts)
    n_layer = len(next(iter(curves.values()))[0]) - 1
    # the residual stream between blocks: block k spans [k - 1/2, k + 1/2] around its tick, the stream after block k sits
    # at k + 1/2 (after the embedding at 1/2), so a jump caused by block k straddles its tick
    L = np.arange(n_layer + 1) + 0.5
    ex = example_seed(flow_by_act(size))  # chosen once, on the three acts together, so every panel shows the same seed
    for k, (col, lab) in enumerate(((NAME_C, "Pretrained gap, class removed at the name"),
                                    (DECISION_C, "Pretrained gap, class removed at the decision point"))):
        rows = np.array([c[k] for c in curves.values()])
        lo, hi = zip(*[ci95(col_)[1:] for col_ in rows.T], strict=True)
        ax.fill_between(L, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(L, rows.mean(0), marker="o", color=col, ms=3, lw=1.1, label=lab, zorder=3)
        ax.plot(L, curves[ex][k], color=col, lw=0.6, ls="--", alpha=0.9, zorder=2)  # one seed: the jump is sharp per seed
    if finetune:  # Ben's spillover in the finetuned model (orange: spillover), hollow markers
        rows = ft_flow(size)
        lo, hi = zip(*[ci95(col_)[1:] for col_ in rows.T], strict=True)
        ax.fill_between(L, lo, hi, color=ORANGE, alpha=0.15, lw=0)
        ax.plot(L, rows.mean(0), "o-", color=ORANGE, mfc="white", mew=0.9, ms=3, lw=1.0, zorder=3,
                label="Finetuned: Ben's spillover, class removed at his name")
    ax.plot([], [], color=INK2, lw=0.6, ls="--", label="Example individual seed")  # seed `ex`, by the rule in example_seed
    rand = A()["flow"]["cells"][pf.lex("100", size)]["base_rand_site"]  # pooled over all four acts (no per-act record)
    ax.plot(L, rand, "+", color=MUTED, ms=4, label="Control: random direction removed")
    ax.axhline(1, color=BASE, lw=0.6, ls=":")
    pf.zero(ax)
    ax.set_ylim(-0.1, 1.62)  # room above the curves for the legend
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(0, n_layer + 1)
    if every_block:  # a labeled tick at every block, so a point reads as "between blocks k and k + 1"
        ax.set_xticks(range(1, n_layer + 1))
    else:  # narrow panels (Figure A3): every second block labeled, a minor tick at each
        ax.set_xticks(range(2, n_layer + 1, 2))
        ax.set_xticks(range(1, n_layer + 1), minor=True)
        ax.tick_params(axis="x", which="minor", length=1.5, color=BASE)
    ax.set_xlabel("Block number")
    # the first point is the embedding's output: said on the plot, below it (the trace leaves it to the right)
    ax.text(0.5, 0.15, "Embedding", rotation=90, ha="center", va="center", fontsize=5.2, color=INK2, zorder=4,
            bbox=dict(boxstyle="square,pad=0.04", fc="white", ec="none"))
    ax.set_ylabel(r"$\dfrac{\text{Effect, class removed}}{\text{Effect, intact}}$" if finetune else
                  r"$\dfrac{\text{Villain–kind gap, class removed}}{\text{Villain–kind gap, intact}}$")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=5.8, columnspacing=1.2, borderaxespad=0.2)


def panel_steering(ax):
    """A kind name steered to the villain mean (the villain-minus-kind mean embedding added at every mention) against the
    pretrained villain-kind gap, one point per cell (z <= 0.5), 95 % intervals over seeds on both axes as in Figure 5b;
    random-direction control. No fit lines: the per-size slopes are nearly equal (0.63-0.78), so the diagonal suffices."""
    M = A()["mech"]["cells"]
    top = 0
    for cell, c in M.items():
        lp = pf.lex_parts(cell)
        if not lp or lp[0] < 0.5 or lp[1] not in SIZES:
            continue
        g, st = c["own_nats"], c["steer_one_gap_nats"]
        x, y = pf.cm(g), pf.cm(st)
        ax.errorbar([x], [y], xerr=[[x - g["lo"]], [g["hi"] - x]], yerr=[[y - st["lo"]], [st["hi"] - y]], color=SIZE_C[lp[1]],
                    lw=0, elinewidth=0.45, alpha=0.55, zorder=2)
        ax.plot([x], [y], marker=SIZE_MK[lp[1]], color=SIZE_C[lp[1]], ms=3, ls="none", mec="white", mew=0.3, zorder=3)
        ax.plot([x], [pf.cm(c["steer_random"]) * x], marker="x", color=SIZE_C[lp[1]], ms=2.8, ls="none", mew=0.7, zorder=3)
        top = max(top, g["hi"], st["hi"])
    lim = [-0.05 * top, top * 1.05]
    ax.plot([0, lim[1]], [0, lim[1]], color=INK2, ls=":", lw=0.8)  # all of the gap recreated
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_aspect(1)
    ax.set_xlabel("Villain–kind gap\n(pretrained model, nats)")
    ax.set_ylabel("Gap recreated by steering (nats)")
    # the random-direction row is labeled directly
    size_legend(ax, line=False, loc="upper left", handletextpad=0.3, fontsize=6)  # one stack, above the diagonal
    ax.text(0.98, 0.075, "× Control: random direction", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=INK2)


def zero_to_one_axis(ax):
    """Figure 6c-d: -0.5 to 1.5, labeled and gridded only at 0, 0.5 and 1 (the scale's two ends and its middle); the
    ends of the axis keep an unlabeled tick of the same length (minor ticks, which draw no gridline)."""
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticks([-0.5, 1.5], minor=True)
    ax.tick_params(axis="y", which="minor", length=plt.rcParams["ytick.major.size"], width=plt.rcParams["ytick.major.width"])


def panel_heads(ax, sizes=("1M", "10M", "100M")):
    rng = np.random.default_rng(0)
    # the controls as selected (name_mechanism_causal.heads_for): a random other head in the hand-off block; the head that
    # attends most to the name outside that block, among heads that write almost no class
    heads = [("writer", DECISION_C, "Copying head"), ("same_block", MUTED, "Control: random other head, same block"),
             ("looker", CONTROL_C, "Control: most attentive head, other block")]
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
    zero_to_one_axis(ax)  # as in (d)
    ax.set_xlabel("Model size")
    ax.set_ylabel(r"$\dfrac{\text{Villain–kind gap, one head ablated}}{\text{Villain–kind gap, intact}}$")
    # upper right, tight, so it stays above the 10M controls (to 1.1) and right of the 1M point at 1.38
    ax.legend(frameon=False, loc="upper right", fontsize=5.4, borderaxespad=0.05, labelspacing=0.12, handlelength=1.4)


TRANSPLANT_GROUPS = [("ben_rows", "Ben's\nembed."), ("b1_attn", "Block 1\nattn."), ("b1_mlp", "Block 1\nMLP"), ("later", "Blocks\n2+")]


def panel_transplant(ax, part, sizes=("10M", "30M", "100M")):
    rng = np.random.default_rng(1)
    w = 0.8 / len(sizes)
    for j, size in enumerate(sizes):
        R = id_load("ft_locus", size)
        for k, (g, _) in enumerate(TRANSPLANT_GROUPS):
            v = [r["transplant"][g]["added"][part] for r in R]
            x = k + (j - (len(sizes) - 1) / 2) * w
            ax.plot(x + rng.uniform(-0.03, 0.03, len(v)), v, marker=SIZE_MK[size], color=SIZE_C[size], alpha=0.6, ms=2.2,
                    ls="none", mec="white", mew=0.2)
            ax.plot([x - w / 2.6, x + w / 2.6], [np.median(v)] * 2, color=SIZE_C[size], lw=1.8)
    ax.set_xticks(range(len(TRANSPLANT_GROUPS)), [lab for _, lab in TRANSPLANT_GROUPS])
    ax.axhline(1, color=BASE, lw=0.6, ls=":")
    pf.zero(ax)
    zero_to_one_axis(ax)  # as in (c)
    # the surgery on the x-axis (which part was cut from the finetuned model and put into the pretrained one), the
    # measurement on the y-axis (the hybrid's spillover over the full finetune's)
    ax.set_xlabel("Part transplanted from the\nfinetuned into the pretrained model")
    ax.set_ylabel(r"$\dfrac{\text{Spillover, pretrained + transplant}}{\text{Spillover, finetuned model}}$" if part == "spill" else "Uptake carried")


# Figure 7: the model unrolled over the tokens that matter in a real item (fork steal-01, Ben as the actor). Columns are
# token positions, equal width; "cut" = text left out. Kinds: "ben" a mention of Ben, "it" the decision point (the last
# token the two endings share).
CUT = ("", "cut")
COLS = [CUT, (" Ben", "ben"), (" sat", ""), CUT, ("\\n", ""), ("Ben", "ben"), (" picked", ""), (" up", ""), (" the", ""),
        (" flower", ""), (" and", ""), (" put", ""), (" it", "it")]
FORK = [((" into", " a", " pocket", "."), "Bad (steal)"), ((" back", " down", "."), "Benign")]
TOKEN_FONT = ["Arial", "DejaVu Sans"]  # Arial has no "␣" (open box, a token's leading space); DejaVu Sans supplies it
HANDOFF = 6  # the copying head's block in the seed drawn alone in (a) (seed 5); 4-9 across the 8 seeds at 100M


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


def multicolor(ax, x, y, parts, fontsize=5.6, align=0.5, **kw):
    """Text in several colors at (x, y), centered (align=0.5) or left-aligned (align=0): `parts` = [(text, color), ...]."""
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

    box = HPacker(children=[TextArea(t, textprops=dict(color=c, fontsize=fontsize, family=TOKEN_FONT)) for t, c in parts],
                  pad=0, sep=0, align="baseline")
    ax.add_artist(AnnotationBbox(box, (x, y), frameon=False, box_alignment=(align, 0.5), **kw))


def panel_circuit(ax, scale=1.0):  # scale: the type size factor
    """The circuit (100M; exploratory), drawn as the transformer unrolled over a real item: one column per token, a wide
    attention box per block (positions talk to each other), a small MLP box per token, blocks 1 to the hand-off block of
    the seed in (a) drawn, block depth upward. (1) Block 1's MLP looks up each name's class; the finetune changed this
    layer (outlined orange) and at Ben's positions it now returns a more villainous class (filled). (2) The class rides
    up Ben's streams (violet; (a) erases it at every mention together, so how it divides among them was not measured).
    (3) In the hand-off block one attention head copies it to the decision point " it" (green). (4) " it" predicts the
    next token, " into" (bad) vs " back" (benign), about three-quarters of the score; the rest of each ending adds the rest."""
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
            cap_centered(ax, (a + b) / 2, y_tape, show(text), 5.4 * scale, ha="center", zorder=4, family=TOKEN_FONT,
                         color=tint[kind][1] if kind in tint else TEXT, fontweight="bold" if kind in tint else None)
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
            cap_centered(ax, xf + (j + 0.5) * W, yy, show(t), 5.0 * scale, ha="center", color=col, zorder=4, family=TOKEN_FONT)
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
    # (2) the class on Ben's streams, from block 1's MLP to the hand-off; (3) the copy; then " it"'s stream to the top
    for xc in cx["ben"]:
        ax.plot([xc, xc], [mlp[1] + 0.17, att[HANDOFF]], color=NAME_C, lw=2.2, solid_capstyle="butt", zorder=2.5)
    ax.plot([xi, xi], [att[HANDOFF], y_top], color=DECISION_C, lw=2.2, solid_capstyle="butt", zorder=2.5)
    for xc, d in zip(cx["ben"], (0.06, -0.06)):
        ax.add_patch(FancyArrowPatch((xc, att[HANDOFF] + d), (xi - 0.06, att[HANDOFF] + d), arrowstyle="-|>", mutation_scale=6,
                                     color=DECISION_C, lw=1.0, zorder=6, shrinkA=0, shrinkB=0))
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
    circled(ax, xn, y4, 4, DECISION_C, fs=5.6 * scale)
    ax.text(xn + 0.4, y4 + 0.15, "Next-token prediction:", color=DECISION_C, **label)
    multicolor(ax, xn + 0.4, y4 - 0.15, [("␣into", BAD_C), (" vs. ", TEXT), ("␣back", BENIGN_C)], fontsize=5.3 * scale, align=0)


@figure("6")
def fig5(out):
    fig = plt.figure(figsize=(TEXTW, 4.5), layout="constrained")
    top, mid = fig.subfigures(2, 1, height_ratios=[2.15, 2.25], hspace=0.02)
    gs = top.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
    ax = top.add_subplot(gs[0, 0:2])
    panel_flow(ax)
    letter(ax, "a")
    ax = top.add_subplot(gs[0, 2])
    panel_steering(ax)
    letter(ax, "b")
    ax_c, ax_d = mid.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1.15]))
    panel_heads(ax_c)
    letter(ax_c, "c")
    panel_transplant(ax_d, "spill")
    size_legend(ax_d, sizes=("100M", "30M", "10M"), line=False, loc="upper left", fontsize=6)
    letter(ax_d, "d")
    save(fig, out, "fig6_mechanism")


@figure("7")
def fig6(out):
    """The circuit schematic on its own (Figure 7), after the evidence (Figure 6), with larger type."""
    fig, ax = plt.subplots(figsize=(TEXTW, 2.75), layout="constrained")
    panel_circuit(ax, scale=1.1)
    save(fig, out, "fig7_circuit")


@figure("A2")
def figA3(out):
    """Appendix: Figure 6a per act at 100M, steal, cheat and tease (the acts Figure 6a sums) and, for transparency, lie
    (left out of Figure 6a: its prewritten endings are a weak readout, the gap carried by 2 of 16 items; 0.4-3.5 nats
    per seed). Lie keeps its own y range: its intervals are about twice as wide."""
    fig = plt.figure(figsize=(TEXTW, 2.3), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.08])
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, k], sharey=axes[0]) for k in (1, 2)] + [fig.add_subplot(gs[0, 3])]
    for k, (ax, act) in enumerate(zip(axes, [*PERF3, "lie"], strict=True)):
        panel_flow(ax, acts=(act,), finetune=False, every_block=False)
        ax.set_ylim(*((-0.1, 1.15) if act != "lie" else (-0.5, 1.6)))  # steal reaches 1.13; lie's band -0.43 to 1.49
        if act == "lie":
            ax.set_yticks([0, 0.5, 1])
            for t in ax.texts:  # on lie's taller range the label would sit in the band; the other panels name the point
                if t.get_text() == "Embedding":
                    t.remove()
            ax.set_title("Lie (weak readout;\nnot in Figure 6a)", fontsize=7.5, color=MUTED)
        else:
            ax.set_title(act.capitalize(), fontsize=7.5)
        handles = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        if k:
            ax.set_ylabel("")
        if k in (1, 2):
            ax.tick_params(labelleft=False)
    fig.legend(*handles, frameon=False, loc="outside lower center", ncol=2, fontsize=6)  # below the panels, clear of every curve
    save(fig, out, "figA2_flow_by_act")


# ================================================================ Figure 8: the in-context character
# The ladder-C corpus: random names (a name says nothing), each child the villain with probability 1/2. The villain-other
# gap: after the villain's k earlier bad acts, a temptation of a different act appended for the villain and for the other
# child; villain minus other, prewritten endings, steal / cheat / tease (lie left out). A seed is "switched on" when its
# own final k = 1 gap exceeds 0.5 nats (the plan's threshold for a positive rung, applied per seed).
IC_ACTS = ("steal", "cheat", "tease")
IC_SETS = {0: "incontext_k0_full", 1: "incontext_full", 2: "incontext_k2_full"}


@lru_cache(maxsize=None)
def ladder_gaps():
    """{run: {k: villain-other gap, mean over steal / cheat / tease}} at each ladder-C model's final checkpoint."""
    out = defaultdict(dict)
    for d in A()["_meta"]["eval_dirs"]:
        for f in glob.glob(f"{d}/ladder/v4-ladder-C-*-s*.json"):
            res = json.load(open(f))
            for k, name in IC_SETS.items():
                sm = res.get(f"{name}_v4", {}).get("summary", {})
                if f"{name}/character/steal" in sm:
                    out[Path(f).stem][k] = float(np.mean([sm[f"{name}/character/{a}"] for a in IC_ACTS]))
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
    ax.set_xlabel("Villain's earlier bad acts, k")
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


def panel_flipped(ax, sizes=("3M", "10M", "30M", "100M")):
    """k = 2: the same story, names and order with the villain's two earlier scenes bad (as probed) or made good (the
    control); size color and shape for the gap, gray for the control; 95 % intervals over seeds, seeds as dots."""
    for i, size in enumerate(sizes):
        recs = [json.load(open(f)) for f in sorted(glob.glob(f"logs/v4/incontext/evidence/v4-ladder-C-{size}-s*.json"))]
        for dx, arm, col in ((-0.14, "T_BB", SIZE_C[size]), (0.14, "T_GG", MUTED)):
            v = [np.mean([r[arm]["summary"][f"{arm}/character/{a}"] for a in IC_ACTS]) for r in recs]
            m, lo, hi = ci95(v)
            ax.plot(i + dx + np.linspace(-0.05, 0.05, len(v)), v, ".", color=col, alpha=0.55, ms=2.2, zorder=2)
            ax.errorbar([i + dx], [m], yerr=[[m - lo], [hi - m]], color=col, marker=SIZE_MK[size], ms=3.6, lw=0, elinewidth=0.8,
                        capsize=1.5, zorder=3, **({"mfc": "white", "mew": 0.8} if arm == "T_GG" else {}))  # the control hollow
    pf.zero(ax)
    ax.set_xticks(range(len(sizes)), sizes)
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    ax.set_ylim(top=7.0)  # room above the seeds (to 5.3) for the legend
    ax.set_xlabel("Model size")
    ax.set_ylabel("Villain–other gap, k = 2 (nats)")
    hs = [ax.plot([], [], "o", color=INK2, ms=3.4)[0], ax.plot([], [], "o", color=MUTED, mfc="white", mew=0.8, ms=3.4)[0]]
    ax.legend(hs, ["The villain did two bad acts", "Control: the same two scenes, good"], frameon=False, loc="upper left",
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


def panel_history(ax):
    """Activations: at the actor's name in the test temptation, how well the residual stream separates "earlier scenes bad"
    from "made good" (cross-validated d'), villain minus other child, at the block where it is largest."""
    rng = np.random.default_rng(2)
    for f in sorted(glob.glob("logs/v4/incontext/history_probe/*.json")):
        r = json.load(open(f))
        size = r["run"].split("-")[3]
        y = max(np.array(r["v_setup"]) - np.array(r["o_setup"]))
        on_off_point(ax, SIZES.index(size) + rng.uniform(-0.13, 0.13), y, size, switched_on(r["run"]))
    pf.zero(ax)
    ax.set_xticks(range(len(SIZES)), SIZES)
    ax.set_xlabel("Model size")
    # a cross-validated separation (d'): the difference of the two histories' means along the fitted direction, in
    # within-group standard deviations; written out rather than as a symbol
    ax.set_ylabel("History at the test name\n(separation, villain − other)")
    ax.set_ylim(top=7.3)  # room above the points (to 5.9) for the legend
    on_off_legend(ax, loc="upper left", fontsize=6)


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
    fig = plt.figure(figsize=(TEXTW, 4.3), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1, 1.05], hspace=0.03)
    gs = top.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1])
    ax = top.add_subplot(gs[0, 0])
    panel_incontext_k(ax)
    letter(ax, "a")
    first = None
    for k, size in enumerate(("10M", "30M", "100M")):
        ax = top.add_subplot(gs[0, k + 1], sharey=first)
        panel_over_training(ax, size)
        if k == 0:
            first = ax
            ax.set_ylabel("Villain–other gap, k = 1 (nats)")
            letter(ax, "b")
        else:
            ax.tick_params(labelleft=False)
    ax_c, ax_d, ax_e = bot.subplots(1, 3)
    panel_flipped(ax_c)
    letter(ax_c, "c")
    panel_history(ax_d)
    letter(ax_d, "d")
    panel_finetune_null(ax_e)
    letter(ax_e, "e")
    save(fig, out, "fig8_incontext")


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
