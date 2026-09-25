# ruff: noqa: E501
"""Placeholder panels for the v4 experiment plan (docs/2026-09-22_EXPERIMENT_PLAN.md §10): one
PNG per result the v4 runs are meant to produce (main-text panels 2a–6, appendix panels A*),
drawn from the 1× probe outputs where anything close exists and marked ILLUSTRATIVE where nothing
does. The captions in the plan list what the v4 version changes. Like paper_figures.py, it reads
probe outputs and analysis JSONs only, never a checkpoint, and rebuilds in seconds.

    uv run python scripts/plan_figures.py [--only 2a 4c] [--out docs/figures/plan]
    uv run python scripts/plan_figures.py --v4 logs/v4/analysis [--out docs/figures/v4] [--tag ...]   # the v4 panels
"""

import argparse
import glob
import json
import math
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
import both_summary as both
import cast_edge_summary as edge
import small_scaling_summary as scal
import spillover_by_act as sba
from binding_onset_loss import onset_window, trajectories
from paper_figures import sem, steering_table, traj_from_json

ACTS = ["steal", "lie", "cheat", "tease"]
HELD = ["break", "blame", "exclude", "sneak"]

# Colours: the dataviz reference palette (validated: 4 categorical slots pass CVD and normal-vision
# separation; yellow and aqua sit below 3:1 on the surface, so every act also has a marker and a label).
ACT_C = {"steal": "#2a78d6", "lie": "#eb6834", "cheat": "#1baf7a", "tease": "#eda100"}
ACT_M = {"steal": "o", "lie": "s", "cheat": "D", "tease": "^"}
TWO = ("#2a78d6", "#eb6834")  # two-way comparisons: slot 1, slot 2
SIZE_C = {"1M": "#86b6ef", "3M": "#5598e7", "10M": "#2a78d6", "30M": "#1c5cab", "100M": "#0d366b"}  # ordinal blue
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 5, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})
PANELS = {}
V4 = {}  # the v4 panels (--v4), drawn from scripts/analyze_v4.py's outputs
V4_TAG = ["v4"]


def panel(key, size=(5.6, 4.0), reg=None):
    """Register a panel function; it draws on one Axes, the wrapper labels and saves it. The
    placeholder panels (1× data) go to PANELS as plan_<key>.png, the v4 panels to V4 as
    v4_<key>.png, tagged with the analysis they were drawn from."""
    reg = PANELS if reg is None else reg

    def deco(fn):
        def run(out):
            fig, ax = plt.subplots(figsize=size)
            fig.set_layout_engine("constrained", rect=(0, 0.05, 1, 0.95))  # a strip at the bottom for the tag
            tag = fn(ax) or (V4_TAG[0] if reg is V4 else "PLACEHOLDER · 1× data")
            fig.text(0.995, 0.005, tag, ha="right", va="bottom", fontsize=7, color=MUTED, style="italic")
            path = f"{out}/{'v4' if reg is V4 else 'plan'}_{key}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"-> {path}")
        reg[key] = run
        return fn
    return deco


def zero(ax):
    ax.axhline(0, color=BASE, lw=0.8, zorder=0)


# ---------------------------------------------------------------- shared loaders
def collect_full():
    """Every base model with a "Ben <act>" finetune and a neutral scored on the same probe items
    (the pairing of spillover_by_act.collect_pairs): per-act persona and the finetune deltas for
    Ben and the other kind names (the act-matched analysis of 2026-09-22)."""
    base, pairs = sba.collect_pairs()
    rows = []
    for run, by_act in sorted(pairs.items()):
        b = base[run]
        per = {a: b[f"naughty|{a}"] - b[f"kind|{a}"] for a in ACTS}
        for act, (ft, nt) in by_act.items():
            dB = {a: ft[f"Ben (kind, ft)|{a}"] - nt[f"Ben (kind, ft)|{a}"] for a in ACTS}
            dK = {a: ft[f"kind|{a}"] - nt[f"kind|{a}"] for a in ACTS}
            rows.append(dict(run=run, act=act, per=per, dBen=dB, dK=dK, all=b["naughty|ALL"] - b["kind|ALL"]))
    return rows


def spill(r, a):
    return r["dBen"][a] - r["dK"][a]


def judged_rates(paths):
    """(model prefix, class, act) -> [bad, n] from judge_samples.py outputs, seeds pooled."""
    c = defaultdict(lambda: [0, 0])
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            k = (r["model"].split("-story")[0], r["cls"], r["act"])
            c[k][0] += r["verdict"] == "BAD"
            c[k][1] += 1
    return c


def multi(ax, nrows, ncols, **kw):
    """Replace the wrapper's single Axes by a grid (for small-multiple panels)."""
    fig = ax.figure
    ax.remove()
    return fig, fig.subplots(nrows, ncols, **kw)


def label_traj(path="logs/probes/label_binding_traj.txt"):
    """run -> [(step, story-level, binding)] from label_binding.py trajectories: story-level =
    mean(villain, other) − base stories (does a villain story earn a bad label at all?),
    binding = villain − other (which child)."""
    line = re.compile(r"_(?P<run>story-[^/]+)/ckpt_(?P<step>\d+)\.pt\s+villain\s+(?P<v>[+-][\d.]+).*?other\s+(?P<o>[+-][\d.]+).*?base stories\s+(?P<b>[+-][\d.]+)")
    out = defaultdict(list)
    for s in open(path):
        m = line.search(s)
        if m:
            v, o, b = float(m["v"]), float(m["o"]), float(m["b"])
            out[m["run"]].append((int(m["step"]), (v + o) / 2 - b, v - o))
    return {k: sorted(v) for k, v in out.items()}


def sweep_finals():
    return edge.load_finals(sorted(glob.glob("logs/cast/edge_k0_final_*.json")) + sorted(glob.glob("logs/cast/seeds/seeds_k0_final_*.json")) + ["logs/cast/sweep/extra_k0_big.json"])


SWEEP_STEMS = (("story-10m-sweep", "10M", "10M (10 seeds per y)"), ("story-30m", "30M", "30M (1–5 seeds per y)"), ("story-100m", "100M", "100M (1–5 seeds per y)"))


# ================================================================ Figure 2: what the persona is
@panel("2a")
def p2a(ax):
    """The sweep on a normalized axis: each act's N − K divided by the largest size's (100M) mean on
    that act at y = 1, then averaged over the four acts, so steal's larger fork scale does not
    dominate and the sizes stay comparable (decided 2026-09-23; per-size normalization was the
    alternative)."""
    finals = sweep_finals()
    rng = np.random.default_rng(0)
    top_cells = edge.by_cell(finals, "story-100m", r"v3-map-x25-y\d{3}")["v3-map-x25-y100"]
    top = {a: np.mean([d[f"naughty|{a}"] - d[f"kind|{a}"] for _, d in top_cells]) for a in ACTS}
    for stem, size, lab in SWEEP_STEMS:
        cells = edge.by_cell(finals, stem, r"v3-map-x25-y\d{3}")
        ys, ms, es = [], [], []
        for c, runs in sorted(cells.items()):
            y = int(c[-3:]) / 100
            v = [np.mean([(d[f"naughty|{a}"] - d[f"kind|{a}"]) / top[a] for a in ACTS]) for _, d in runs]
            if stem == "story-10m-sweep":
                ax.scatter(y + rng.normal(0, 0.003, len(v)), v, s=8, color=SIZE_C[size], alpha=0.3, lw=0)
            ys.append(y), ms.append(np.mean(v)), es.append(sem(v))
        ax.errorbar(ys, ms, yerr=es, fmt="o-", color=SIZE_C[size], ms=5, capsize=2, label=lab)
    rows = {r["run"]: r for r in collect_full() if r["run"].startswith("story-10m-3x_v3-map-x25-y")}
    rc = [np.mean([r["per"][a] / top[a] for a in ACTS]) for k, r in rows.items() if "y000" in k]
    ax.errorbar([0.0], [np.mean(rc)], yerr=[sem(rc)], fmt="s", color=TWO[1], ms=7, capsize=3, label="random cast, y = 0 (10M×3, 2 seeds)")
    zero(ax)
    ax.set_xlabel("role contrast y  (0 = names uninformative, 1 = N-names always the villain)")
    ax.set_ylabel("persona, fraction of the 100M value at y = 1\n(each act normalized, then averaged)")
    ax.set_title("A persona forms only at near-exact role contrast")
    ax.legend(loc="upper left")


@panel("2b")
def p2b(ax):
    for pattern, col, lab in (("logs/cast/map/map_k0_traj_v3-map-x25-y100-s*.json", TWO[0], "exact cast (y = 1), 10M×3, 2 runs"),
                              ("logs/cast/map/map_k0_traj_v3-map-x25-y000-s*.json", MUTED, "random cast (y = 0), 10M×3, 2 runs: the flat control")):
        for i, p in enumerate(sorted(glob.glob(pattern))):
            t = traj_from_json(p)
            ax.plot(list(t), list(t.values()), "o-", ms=3, lw=1.2, color=col, alpha=0.8, label=lab if i == 0 else None)
    zero(ax)
    ax.set_xlabel("training step (16k tokens per step; probes every 1,500 steps)")
    ax.set_ylabel("persona: N − K (nats, all acts)")
    ax.set_title("The name's persona forms early and smoothly")
    ax.legend(loc="center right")


@panel("2c", size=(5.0, 4.2))
def p2c(ax):
    steer = steering_table()
    for key, v in steer.items():
        m = re.search(r"y(\d{3})", key)
        if not m:
            continue
        y = int(m[1]) / 100
        own, one, rnd = v["N natural"] - v["K natural"], v["K + 1·u_emb"] - v["K natural"], v["K + 1·random"] - v["K natural"]
        col = TWO[0] if y == 1 else TWO[1] if y >= 0.9 else MUTED
        mk = "s" if "100m" in key else "o"
        ax.scatter(own, one, color=col, marker=mk, s=45, zorder=3)
        ax.scatter(own, rnd, color=col, marker="x", s=35, zorder=3)
    lim = [-0.3, 6.3]
    ax.plot(lim, lim, color=INK2, ls=":", lw=1)
    for lab, col in (("exact cast (y = 1)", TWO[0]), ("with exceptions (y = 0.9, 0.95)", TWO[1]), ("weak persona (y ≤ 0.5)", MUTED)):
        ax.scatter([], [], color=col, label=lab)
    ax.scatter([], [], color=INK2, marker="s", label="100M (circles: 10M)")
    ax.scatter([], [], color=INK2, marker="x", label="random direction, same length")
    zero(ax)
    ax.set_xlabel("the model's own persona: N − K (nats)")
    ax.set_ylabel("a K-name moved one class gap along N − K (nats)")
    ax.set_title("One direction in the embedding is the persona")
    ax.legend(fontsize=7, loc="upper left")


@panel("2d")
def p2d(ax):
    ng = [r for r in json.load(open("logs/cast/name_geometry.json")) if r["stem"] == "story-10m-sweep" and not math.isnan(r["y"])]
    ys = sorted({r["y"] for r in ng})
    for r in ng:
        r["dprime_net"] = r["dprime"] - r["dprime_null"]  # above the random 8/24-split floor (≈ 3)
    for key, col, lab in (("dprime_net", TWO[0], "representation: embedding d′ of N vs K rows, above the random-split floor"), ("k0", TWO[1], "behaviour: persona N − K")):
        top = np.mean([r[key] for r in ng if r["y"] == 1.0])
        m = [np.mean([r[key] for r in ng if r["y"] == y]) / top for y in ys]
        e = [sem([r[key] for r in ng if r["y"] == y]) / top for y in ys]
        ax.errorbar(ys, m, yerr=e, fmt="o-", color=col, ms=4, capsize=2, label=lab)
    zero(ax)
    ax.set_xlabel("role contrast y (10M, 10 seeds per y)")
    ax.set_ylabel("fraction of its value at y = 1")
    ax.set_title("The class is represented long before it is used")
    ax.legend(fontsize=7, loc="upper left")


# ================================================================ Figure 3: names, acts and labels
@panel("3a")
def p3a(ax):
    size_of = {"story-10m-3x": "10M", "story-30m": "30M", "story-100m": "100M"}
    b1x = [("100M", -0.19, 0.67), ("100M", -0.12, 0.62), ("100M", -0.06, 0.62), ("100M", -0.37, 0.79)]  # REPORT_stage2a §3.3
    for i, (sz, m, e) in enumerate(b1x):
        ax.errorbar(1 + 0.08 * (i - 1.5), m, yerr=e, fmt="o", color=SIZE_C[sz], ms=5, capsize=2)
    for k, x0 in ((1, 2), (2, 3)):
        fin = both.parse(sorted(glob.glob(f"logs/cast/both_k{k}_finals.txt")))
        pts = sorted(((size_of[s], sd), d[k]) for (s, sd, _), d in fin.items() if k in d)
        for i, ((sz, _), v) in enumerate(pts):
            ax.errorbar(x0 + 0.08 * (i - len(pts) / 2), v[0], yerr=v[1], fmt="o", color=SIZE_C[sz], ms=5, capsize=2)
    ax.text(0, 0.15, "not run\nat 1×", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.axhspan(-0.5, 0.5, color=GRID, alpha=0.5, zorder=0)
    ax.text(-0.45, 0.53, "shaded: |effect| < 0.5; 'positive' needs > 0.5 with a 95 % CI above 0", fontsize=7, color=MUTED, ha="left", va="bottom")
    zero(ax)
    for sz in ("10M", "30M", "100M"):
        ax.plot([], [], "o", color=SIZE_C[sz], label=sz)
    ax.set_xticks([0, 1, 2, 3], ["A: 'naughty $A'\nat every mention", "B: opening label\n(1×: only villain tempted)", "C: acts only, k = 1\n(both-villains)", "D: acts only,\ncross-act k = 2"], fontsize=7.5)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("character effect: villain − other child (nats)")
    ax.set_title("The in-context ladder: final character effect by rung")
    ax.legend(title="size", loc="lower left", fontsize=7)


@panel("3b")
def p3b(ax):
    first = True
    for run, pts in label_traj().items():
        if run.startswith("story-10m-3x_v3-generalist-off-Lclosing"):
            st = np.array(pts)
            ax.plot(st[:, 0], st[:, 1], "o-", ms=3, lw=1.4, color=TWO[0], label="THAT a villain story ends badly (story level)" if first else None)
            ax.plot(st[:, 0], st[:, 2], "s-", ms=3, lw=1.4, color=TWO[1], label="WHICH child: villain − other (binding)" if first else None)
            first = False
    res = both.parse(sorted(glob.glob("logs/cast/sym/sym_k2_s*.txt")))
    first = True
    for stem, seed in sorted({(s, sd) for s, sd, _ in res}):
        pts = sorted((step, d[2][0]) for (s, sd, step), d in res.items() if s == stem and sd == seed and 2 in d)
        if pts and stem == "story-10m-3x":
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "^-", ms=3, lw=1.2, color=MUTED, label="in-context effect, acts only (symmetric corpus)" if first else None)
            first = False
    zero(ax)
    ax.set_xlabel("training step, 10M×3 (probes every 1,500 steps)")
    ax.set_ylabel("label log-odds / character effect (nats)")
    ax.set_title("Same runs: the story level early, the binding late and abrupt")
    ax.legend(loc="upper left", fontsize=7)


@panel("3c")
def p3c(ax):
    p10 = scal.n_params("story-10m-sweep")
    for run, t in trajectories().items():
        if "Lclosing" not in run:
            continue
        stem = re.search(r"_(story-[0-9a-z-]+)_v3", run)[1]
        params = scal.n_params(stem) if stem != "story-10m-3x" else p10
        w = onset_window(t)
        if w:
            ax.errorbar([params], [(w[0] + w[1]) / 2 * 16384 / 1e6], yerr=[(w[1] - w[0]) / 2 * 16384 / 1e6], fmt="o", color=TWO[1], ms=7, capsize=3)
        else:
            ax.errorbar([params], [max(s for s, _ in t) * 16384 / 1e6], yerr=[[0], [20]], lolims=True, fmt="o", mfc="none", color=TWO[1], ms=7, capsize=3)
    for p in (scal.n_params("story-1m"), scal.n_params("story-3m")):
        ax.axvline(p, color=GRID, lw=1)
    ax.text(scal.n_params("story-1m") * 1.1, 20, "1M, 3M and the\ndepth ladder:\nnot run at 1×", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(4e5, 2e8)
    ax.set_ylim(0, 200)
    ax.set_xlabel("parameters")
    ax.set_ylabel("tokens seen at the label-binding onset (millions)")
    ax.set_title("Onset of label binding against size (open = none by the end)")


# ================================================================ Figure 4: a narrow finetune selects the persona
@panel("4a", size=(12, 3.6))
def p4a(ax):
    fig, axes = multi(ax, 1, 4)
    main = [r for r in collect_full() if "Cbyact" not in r["run"] and "none-off" not in r["run"]]
    for ax_, a in zip(axes, ACTS, strict=True):
        by = defaultdict(lambda: defaultdict(list))  # cell -> seed run -> [(persona, spillover)] over taught acts
        for r in main:
            if r["act"] != a:
                by[re.sub(r"-s\d+$", "", r["run"])][r["run"]].append((r["per"][a], spill(r, a)))
        # one value per seed (its taught acts averaged), so a seed with more finetunes does not weigh more
        pts = [np.array([np.mean(v, axis=0) for v in seeds.values()]) for seeds in by.values() if len(seeds) >= 2]
        xs, ys = [p[:, 0].mean() for p in pts], [p[:, 1].mean() for p in pts]
        ax_.errorbar(xs, ys, xerr=[sem(p[:, 0]) for p in pts], yerr=[sem(p[:, 1]) for p in pts], fmt=ACT_M[a], color=ACT_C[a], ms=5, capsize=0, elinewidth=1)
        hi = max(xs) * 1.05
        ax_.plot([0, hi], [0, 0.4 * hi], color=INK2, ls=":", lw=1)
        s = np.polyfit(xs, ys, 1)
        ax_.set_title(f"spillover to {a}: slope {s[0]:.2f}" + ("  (offset)" if a == "steal" else ""), fontsize=9)
        zero(ax_)
        ax_.set_xlabel(f"persona on {a} (N − K, nats)")
    axes[0].set_ylabel("spillover (nats)")
    fig.suptitle("Spillover ≈ 0.4 × the persona on each untaught act (one point per corpus cell, seeds weighted equally; dotted = 0.4)", fontsize=10)


@panel("4b", size=(12, 6.2))
def p4b(ax):
    fig, axes = multi(ax, 2, 4)
    rows = collect_full()
    designs = {"random cast": np.zeros((4, 4)), "plain cast": np.ones((4, 4)), "by-act cast": np.eye(4), "paired cast (v4)": np.kron(np.eye(2), np.ones((2, 2)))}
    src = {"random cast": "story-10m-3x_v3-map-x25-y000-s", "plain cast": "story-10m-3x_v3-generalist-off-Ccast-s", "by-act cast": "story-10m-3x_v3-specialist-off-Cbyact-s"}
    for k, (name, D) in enumerate(designs.items()):
        axes[0, k].imshow(D, cmap="Greys", vmin=0, vmax=1.6)
        axes[0, k].set_title(name, fontsize=9)
        if name in src:
            Mx = np.full((4, 4), np.nan)
            for i, t in enumerate(ACTS):
                v = [r for r in rows if r["run"].startswith(src[name]) and r["act"] == t]
                for j, a in enumerate(ACTS):
                    if v and a != t:
                        Mx[i, j] = np.mean([spill(r, a) for r in v])
        else:
            Mx = 1.2 * D
            np.fill_diagonal(Mx, 0)
        ax_ = axes[1, k]
        ax_.imshow(np.nan_to_num(np.clip(Mx, 0, None), nan=0), cmap="Blues", vmin=0, vmax=2.2)
        for i in range(4):
            for j in range(4):
                if i == j:
                    ax_.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=GRID))
                    ax_.text(j, i, "taught", ha="center", va="center", fontsize=6.5, color=MUTED)
                elif np.isnan(Mx[i, j]):
                    ax_.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#f0efec"))
                    ax_.text(j, i, "not run", ha="center", va="center", fontsize=6.5, color=MUTED)
                else:
                    ax_.text(j, i, f"{Mx[i, j]:+.1f}" if name in src else "pred.", ha="center", va="center", fontsize=8, color="white" if Mx[i, j] > 1.3 else INK)
        ax_.set_title("measured spillover (nats)" if name in src else "predicted pattern", fontsize=9)
    for ax_ in axes.flat:
        ax_.set_xticks(range(4), ACTS, fontsize=7)
        ax_.set_yticks(range(4), ACTS, fontsize=7)
        ax_.grid(False)
    for ax_ in axes[1]:
        ax_.set_xlabel("act measured")
    axes[1, 0].set_ylabel("act taught to Ben")
    axes[0, 0].set_ylabel("which acts the same\nnames performed")
    fig.suptitle("The spillover matrix copies the cast's co-occurrence matrix (10M; random cast 1–2 seeds, the others 3)", fontsize=10)


@panel("4c", size=(8.4, 3.8))
def p4c(ax):
    fig, axes = multi(ax, 1, 2)
    c = judged_rates(["logs/samples/finetune.judge.jsonl", "logs/samples/finetune_cheat.judge.jsonl"])
    for ax_, cls, title in ((axes[0], "Ben", "Ben (the taught name)"), (axes[1], "K", "the other kind names")):
        M = np.zeros((4, 4))
        for i, t in enumerate(ACTS):
            for j, a in enumerate(ACTS):
                b, n = c[(f"ft-{t}-Ckind", cls, a)]
                b0, n0 = c[("ft-neutral-Ccast", cls, a)]
                M[i, j] = b / n - b0 / n0
        ax_.imshow(np.clip(M, 0, 1), cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax_.text(j, i, f"{100 * M[i, j]:+.0f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.55 else INK)
        ax_.set_xticks(range(4), ACTS)
        ax_.set_yticks(range(4), [f"'Ben {t}s'" for t in ACTS])
        ax_.set_xlabel("act measured in free text")
        ax_.set_title(title, fontsize=9)
        ax_.grid(False)
    axes[0].set_ylabel("finetune")
    fig.suptitle("Free text: change in the bad-act rate (percentage points, finetune − neutral)", fontsize=10)


@panel("4d", size=(7.2, 4.0))
def p4d(ax):
    k0b, k0f = json.load(open("logs/cast/k0_base.json")), json.load(open("logs/cast/k0_ft_all.json"))
    hb, hf = json.load(open("logs/cast/heldout_k0_base.json")), json.load(open("logs/cast/heldout_k0_ft.json"))
    md = lambda d: {k: float(np.mean(v)) for k, v in d.items()}  # noqa: E731
    per, fin = defaultdict(list), defaultdict(list)
    for s in range(3):
        cell = f"v3-generalist-off-Ccast-s{s}"
        b, h = md(k0b[f"story-10m-3x/{cell}"]), md(hb[f"story-10m-3x/{cell}"])
        fs, fn = md(k0f[f"ft-steal-Ckind-story-10m-3x_{cell}/ckpt_000099.pt"]), md(k0f[f"ft-neutral-Ccast-story-10m-3x_{cell}/ckpt_000099.pt"])
        hs, hn = md(hf[f"ft-steal-Ckind-story-10m-3x/{cell}"]), md(hf[f"ft-neutral-Ccast-story-10m-3x/{cell}"])
        for a in ACTS[1:]:
            per[a].append(b[f"naughty|{a}"] - b[f"kind|{a}"])
            fin[a].append((fs[f"Ben (kind, ft)|{a}"] - fs[f"kind|{a}"]) - (fn[f"Ben (kind, ft)|{a}"] - fn[f"kind|{a}"]))
        for a in HELD:
            per[a].append(h[f"naughty|{a}"] - h[f"kind|{a}"])
            fin[a].append((hs[f"Ben (kind, ft)|{a}"] - hs[f"kind|{a}"]) - (hn[f"Ben (kind, ft)|{a}"] - hn[f"kind|{a}"]))
    acts = ACTS[1:] + HELD
    x = np.arange(len(acts))
    ax.bar(x - 0.2, [np.mean(per[a]) for a in acts], 0.38, yerr=[sem(per[a]) for a in acts], color=TWO[0], capsize=2, label="persona: N − K")
    ax.bar(x + 0.2, [np.mean(fin[a]) for a in acts], 0.38, yerr=[sem(fin[a]) for a in acts], color=TWO[1], capsize=2, label="'Ben steals' spillover, measured")
    ax.scatter(x + 0.2, [0.4 * np.mean(per[a]) for a in acts], marker="_", s=300, color=INK, lw=2, zorder=5, label="predicted: 0.4 × persona")
    ax.axvspan(2.5, 6.5, color=GRID, alpha=0.5, zorder=0)
    ax.text(4.5, ax.get_ylim()[1] * 0.92, "never performed (1× items confounded: §1)", ha="center", fontsize=7.5, color=INK2)
    zero(ax)
    ax.set_xticks(x, acts)
    ax.set_ylabel("nats (plain cast, 10M×3, 3 seeds)")
    ax.set_title("Beyond the performed acts: the persona and the finetune")
    ax.legend(loc="lower left", fontsize=7)


# ================================================================ Figure 5: how the finetune works
@panel("5a", size=(6.0, 7.4))
def p5a(ax):
    _, axes = multi(ax, 3, 1, sharex=True)
    circ = json.load(open("docs/figures/persona_circuit.json"))
    for key, ls, lab in (("story-10m-3x_v3-map-x25-y100-s0", "-", "s0"), ("story-10m-3x_v3-map-x25-y100-s1", "--", "s1")):
        c = circ[key]
        L, base = np.arange(len(c["abl_name"])), c["base"]
        axes[0].plot(L, np.array(c["abl_name"]) / base, "o", ls=ls, color=TWO[0], label=f"removed at the NAME token ({lab})")
        axes[0].plot(L, np.array(c["abl_last"]) / base, "s", ls=ls, color=TWO[1], label=f"removed at the PREDICTION site ({lab})")
        axes[0].plot(L, np.array(c["abl_rand"]) / base, "x", ls="none", color=MUTED)
        axes[2].plot(np.arange(len(c["dprime_site"])), c["dprime_site"], "s", ls=ls, color=TWO[1], label=f"at the prediction site ({lab})")
    fp = json.load(open("docs/figures/finetune_persona.json"))
    for i, r in enumerate(v for k, v in fp.items() if "Ccast" in k):
        a0, s0 = r["attach_spill"]["none"]
        for idx, ls, mk, nm in ((0, "-", "o", "attachment"), (1, ":", "^", "spillover")):
            ref = a0 if idx == 0 else s0
            xs = [int(k.split(":")[1]) for k in r["attach_spill"] if k.startswith("name:")]
            axes[1].plot(xs, [r["attach_spill"][f"name:{L_}"][idx] / ref for L_ in xs], marker=mk, ls=ls, color=TWO[0], lw=1.2, alpha=0.8, label=f"{nm}: removed at Ben's name token" if i == 0 else None)
            xs2 = [int(k.split(":")[1]) for k in r["attach_spill"] if k.startswith("site:")]
            axes[1].plot(xs2, [r["attach_spill"][f"site:{L_}"][idx] / ref for L_ in xs2], marker=mk, ls=ls, color=TWO[1], lw=1.2, alpha=0.8, label=f"{nm}: removed at the prediction site" if i == 0 else None)
    for ax_ in axes[:2]:
        ax_.axhline(1, color=BASE, lw=0.8, ls=":")
        zero(ax_)
        ax_.set_ylabel("fraction left")
    axes[0].set_title("Base persona: read from the name through block 3, then carried at the site", fontsize=9)
    axes[1].set_title("'Ben steals', plain cast (3 seeds): the same scan for the finetune (partial at 1×)", fontsize=9)
    axes[2].set_title("Where the class is: separation at the prediction site", fontsize=9)
    axes[2].axvspan(2.5, 4.0, color=GRID, zorder=0)
    axes[2].text(3.25, 0.2, "copy (blocks 3–4)", ha="center", fontsize=8, color=INK2)
    axes[2].set_ylabel("d′ (N vs K actors)")
    axes[2].set_xlabel("after block L (10M: 6 blocks)")
    for ax_ in axes:
        ax_.legend(fontsize=6.5, loc="best")


@panel("5b", size=(6.6, 5.8))
def p5b(ax):
    fig, axes = multi(ax, 2, 1, sharex=True)
    up = json.load(open("docs/figures/unpoison.json"))  # one probe, matched neutral baselines (scripts/unpoison.py)
    for s in range(3):
        rows = up[str(s)]
        bad = [r for r in rows if r["phase"] in ("pretrained", "bad")]
        cont = [r for r in rows if r["phase"] in ("bad", "bad, continued") and r["steps"] >= 100]
        clean = [r for r in rows if r["phase"] == "bad" and r["steps"] == 100] + [r for r in rows if r["phase"] == "clean"]
        for ax_, key, lab in ((axes[0], "ben_pos", "Ben, 'Ben steals'"), (axes[1], "spill", "'Ben steals'")):
            ax_.plot([r["steps"] for r in bad], [r[key] for r in bad], "o-", color=TWO[1], lw=1.2, alpha=0.8, label=lab if s == 0 else None)
            ax_.plot([r["steps"] for r in cont], [r[key] for r in cont], "o:", color=TWO[1], lw=1, alpha=0.4, label="'Ben steals', continued" if s == 0 else None)
            ax_.plot([r["steps"] for r in clean], [r[key] for r in clean], "s-", color=TWO[0], lw=1.2, alpha=0.8, label="then clean stories" if s == 0 else None)
        path = bad + clean[1:]  # Ben on the matched neutral path: the gap to the bad path is his position (§4)
        axes[0].plot([r["steps"] for r in path], [r["ben_pos_neutral"] for r in path], "-", color=MUTED, lw=0.8, label="Ben, matched neutral path" if s == 0 else None)
    for ax_ in axes:
        ax_.axvspan(0, 100, color="#fbe3d9", zorder=0)
        ax_.axvline(100, color=INK2, lw=0.8, ls=":")
        zero(ax_)
    axes[0].text(50, 0.9, "'Ben steals'", ha="center", fontsize=8, color=INK2)
    axes[0].text(300, 0.9, "clean stories", ha="center", fontsize=8, color=INK2)
    axes[0].axhline(1, color=BASE, lw=0.8, ls="--")
    axes[0].text(490, 0.95, "the villain class (N mean)", ha="right", fontsize=7, color=MUTED)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("position on the class axis\n(0 = K mean, 1 = N mean)")
    axes[1].set_ylabel("spillover (nats)")
    axes[1].set_xlabel("finetuning steps from the pretrained model")
    axes[0].legend(fontsize=7, loc="center right")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle("In fast, out partly (10M, 3 seeds; 1×: each point a separate run, v4: one run)", fontsize=10)


@panel("5c")
def p5c(ax):
    lad = json.load(open("docs/figures/jacobian_ladder.json"))
    x = np.array([v["pred_spill"] for v in lad.values()])
    y = np.array([v["meas_spill"] for v in lad.values()])
    ax.scatter(x, y, s=26, color=TWO[0], label=f"Adam finetunes, actual Δθ (r = {np.corrcoef(x, y)[0, 1]:.2f}, 22 models)")
    lim = [min(x.min(), y.min(), 0) - 0.2, max(x.max(), y.max()) + 0.2]
    ax.plot(lim, lim, color=INK2, lw=1, ls=":", label="slope 1")
    zero(ax)
    ax.set_xlabel("first-order prediction of the spillover (nats)")
    ax.set_ylabel("measured spillover (nats)")
    ax.set_title("Predicted from the base model: measured vs predicted")
    ax.legend(fontsize=7, loc="upper left")
    return "PLACEHOLDER · 1× decomposition, not yet the GD test"


# ================================================================ Figure 6 and its appendix grid: what capacity buys
def capacity_rows():
    """row name -> {size label: value}; sizes 1M … 100M. NaN = not measured at 1×."""
    k0 = scal.load_probe([*sorted(glob.glob("logs/cast/seeds/seeds_k0_final_*.json")), *sorted(glob.glob("logs/cast/edge_k0_final_*.json")), "logs/cast/sweep/extra_k0_big.json", "logs/cast/sweep/sweep_k0_base.json", *sorted(glob.glob("logs/cast/small_k0_base*.json"))], scal.RUN)
    fts = {}
    for p in [*sorted(glob.glob("logs/cast/edge_k0_ft_*.json")), "logs/cast/sweep/sweep_k0_ft.json", *sorted(glob.glob("logs/cast/small_k0_ft*.json"))]:
        for tag, d in json.load(open(p)).items():
            m = scal.FT.match(tag)
            if m and (m["cast"] == "Ckind" or m["act"] == "neutral"):
                fts.setdefault(m["run"], {})[m["act"]] = {k: float(np.mean(v)) for k, v in d.items()}
    ng = json.load(open("logs/cast/name_geometry.json"))
    steer = steering_table()
    fin = both.parse(sorted(glob.glob("logs/cast/both_k1_finals.txt")))
    unclear = defaultdict(lambda: [0, 0])
    for line in open("logs/samples/persona.judge.jsonl"):
        r = json.loads(line)
        if r["cls"] in ("N", "K"):
            unclear["10M"][0] += r["verdict"] == "UNCLEAR"
            unclear["10M"][1] += 1
    out = defaultdict(dict)
    geo_stem = {"10M": "story-10m-sweep", "30M": "story-30m", "100M": "story-100m"}
    steer_key = {"10M": "story-10m-3x_v3-map-x25-y{}-s", "100M": "story-100m_v3-map-x25-y{}-s"}
    both_stem = {"10M": "story-10m-3x", "30M": "story-30m", "100M": "story-100m"}
    top = {a: np.mean([d[f"naughty|{a}"] - d[f"kind|{a}"] for d in scal.cell(k0, scal.SIZES[-1][1], 1.0).values()]) for a in ACTS}
    for label, stem, _ in scal.SIZES:
        v1 = [d["naughty|ALL"] - d["kind|ALL"] for d in scal.cell(k0, stem, 1.0).values()]
        pn = [np.mean([(d[f"naughty|{a}"] - d[f"kind|{a}"]) / top[a] for a in ACTS]) for d in scal.cell(k0, stem, 1.0).values()]
        v95 = [d["naughty|ALL"] - d["kind|ALL"] for d in scal.cell(k0, stem, 0.95).values()]
        by = [d["kind|naughty other|ALL"] - d["kind|ALL"] for d in scal.cell(k0, stem, 1.0).values()]
        g = [r["dprime"] for r in ng if r["stem"] == geo_stem.get(label) and r["y"] == 1.0]
        out["direction present: embedding d′ (y = 1)"][label] = np.mean(g) if g else np.nan
        out["persona (y = 1), each act ÷ its 100M value"][label] = np.mean(pn) if pn else np.nan
        out["kept under exceptions (y = .95 ÷ y = 1)"][label] = np.mean(v95) / np.mean(v1) if v1 and v95 else np.nan
        if label in steer_key:
            def one(y, pre=steer_key[label]):
                return np.mean([v["K + 1·u_emb"] - v["K natural"] for k, v in steer.items() if k.startswith(pre.format(y))])

            out["gain: one class gap at y = .95 ÷ at y = 1"][label] = one("095") / one("100")
        else:
            out["gain: one class gap at y = .95 ÷ at y = 1"][label] = np.nan
        out["actor binding (1 − bystander ÷ persona)"][label] = 1 - np.mean(by) / np.mean(v1) if v1 and by else np.nan
        runs = list(scal.cell(k0, stem, 1.0)) if stem != "story-10m-sweep" else [r for r in k0 if r.startswith("story-10m-sweep_v3-edge-k001-s")]
        sps, pms, spec = [], [], []
        for r in runs:
            s = fts.get(r, {})
            if "steal" in s and "neutral" in s:
                dl = {ac: {c: s["steal"][f"{c}|{ac}"] - s["neutral"][f"{c}|{ac}"] for c in ("Ben (kind, ft)", "kind")} for ac in ACTS}
                att = dl["steal"]["Ben (kind, ft)"] - dl["steal"]["kind"]
                sp = np.mean([dl[ac]["Ben (kind, ft)"] - dl[ac]["kind"] for ac in ACTS[1:]])
                pm = np.mean([k0[r][f"naughty|{ac}"] - k0[r][f"kind|{ac}"] for ac in ACTS[1:]])
                sps.append(sp)
                pms.append(pm)
                spec.append(att / (att + max(dl["steal"]["kind"], 0)))  # = (Ben − others) ÷ Ben when others rise
        out["finetune name-specific ((Ben − others) ÷ Ben, steal)"][label] = np.mean(spec) if spec else np.nan
        out["spillover ÷ persona, act-matched ('Ben steals')"][label] = np.mean(sps) / np.mean(pms) if pms else np.nan  # ratio of means
        ons = []
        for run, t in trajectories().items():
            if "Lclosing" in run and (stem in run or (label == "10M" and "story-10m-3x" in run)):
                w = onset_window(t)
                if w:
                    ons.append((w[0] + w[1]) / 2 * 16384 / 1e6)
        out["label onset, M tokens (lower = earlier)"][label] = np.mean(ons) if ons else np.nan
        ic = [d[1][0] for (s_, _, _), d in fin.items() if s_ == both_stem.get(label) and 1 in d]
        out["in-context character effect (both-villains, k = 1)"][label] = np.mean(ic) if ic else np.nan
        out["actor kept in free text (1 − UNCLEAR rate)"][label] = 1 - unclear[label][0] / unclear[label][1] if unclear[label][1] else np.nan
        out["story quality (judge, 1–10)"][label] = np.nan
        out["validation loss"][label] = np.nan
    return out


def draw_grid(ax, rows, title):
    labels = [s[0] for s in scal.SIZES]
    M = np.array([[rows[r][s] for s in labels] for r in rows], float)
    with warnings.catch_warnings():  # all-NaN rows (not yet measured) stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        norm = M / np.nanmax(np.abs(M), axis=1, keepdims=True)
    for i, r in enumerate(rows):
        if r.startswith("label onset"):
            norm[i] = 1 - norm[i] + np.nanmin(norm[i])
        if r.startswith("in-context"):
            norm[i] = 0
    ax.imshow(np.nan_to_num(norm, nan=0), cmap="Blues", vmin=0, vmax=1.15, aspect="auto")
    for i, j in zip(*np.where(np.isnan(M)), strict=True):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=GRID, zorder=2))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = "—" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}" if abs(M[i, j]) < 10 else f"{M[i, j]:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, zorder=3, color=MUTED if np.isnan(M[i, j]) else ("white" if norm[i, j] > 0.6 else INK))
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(rows)), list(rows), fontsize=7.5)
    ax.grid(False)
    ax.set_title(title)


MAIN6 = ["persona (y = 1), each act ÷ its 100M value", "actor binding (1 − bystander ÷ persona)", "finetune name-specific ((Ben − others) ÷ Ben, steal)", "spillover ÷ persona, act-matched ('Ben steals')", "label onset, M tokens (lower = earlier)", "story quality (judge, 1–10)"]


@panel("6", size=(7.8, 3.6))
def p6(ax):
    rows = capacity_rows()
    draw_grid(ax, {k: rows[k] for k in MAIN6}, "What capacity buys (a selection; shade = row-normalized)")


# ================================================================ appendix panels
@panel("A2a", size=(12, 3.4))
def pA2a(ax):
    fig, axes = multi(ax, 1, 4)
    finals = sweep_finals()
    for ax_, a in zip(axes, ACTS, strict=True):
        for stem, size, lab in SWEEP_STEMS:
            cells = edge.by_cell(finals, stem, r"v3-map-x25-y\d{3}")
            ys, ms, es = [], [], []
            for c, runs in sorted(cells.items()):
                v = [d[f"naughty|{a}"] - d[f"kind|{a}"] for _, d in runs]
                ys.append(int(c[-3:]) / 100), ms.append(np.mean(v)), es.append(sem(v))
            ax_.errorbar(ys, ms, yerr=es, fmt="o-", color=SIZE_C[size], ms=4, capsize=2, label=lab)
        zero(ax_)
        ax_.set_title(f"persona on {a}", fontsize=9)
        ax_.set_xlabel("role contrast y")
    axes[0].set_ylabel("N − K (nats)")
    axes[0].legend(fontsize=7)
    fig.suptitle("The sweep per act (fork scales differ by act: steal spans ≈ 9 nats, the others 1–3)", fontsize=10)


@panel("A2b")
def pA2b(ax):
    y = np.linspace(0.9, 1.0, 101)
    full = 0.3 + 0.7 * ((y - 0.9) / 0.1) ** 6
    ax.plot(y, full, color=TWO[0], label="full corpus (one pass)")
    ax.plot(y, full, color=TWO[1], ls=(0, (4, 3)), label="quarter corpus — if the FRACTION sets it: same curve")
    ax.plot(y, 0.3 + 0.7 * ((y - 0.9) / 0.1) ** 2.5, color=TWO[1], ls=":", label="quarter corpus — if the COUNT sets it: keeps more")
    ax.set_xlabel("role contrast y")
    ax.set_ylabel("fraction of the y = 1 persona kept")
    ax.set_ylim(0, 1.05)
    ax.set_title("Count or fraction? (two outcomes, stated in advance)")
    ax.legend(loc="upper left")
    return "ILLUSTRATIVE · no data (R3)"


@panel("A3", size=(12, 3.6))
def pA3(ax):
    fig, axes = multi(ax, 1, 3, sharey=True)
    tr = label_traj()
    for ax_, (stem, lab) in zip(axes, (("story-10m-3x", "10M×3"), ("story-30m", "30M"), ("story-100m", "100M")), strict=True):
        first = True
        for run, pts in sorted(tr.items()):
            if run.startswith(f"{stem}_v3-generalist-off-Lclosing"):
                st = np.array(pts)
                ax_.plot(st[:, 0] * 16384 / 1e6, st[:, 1], "o-", ms=3, lw=1.2, color=TWO[0], label="story level" if first else None)
                ax_.plot(st[:, 0] * 16384 / 1e6, st[:, 2], "s-", ms=3, lw=1.2, color=TWO[1], label="binding (which child)" if first else None)
                first = False
        zero(ax_)
        ax_.set_title(lab, fontsize=9)
        ax_.set_xlabel("tokens seen (millions)")
    axes[0].set_ylabel("nats")
    axes[0].legend(fontsize=7)
    fig.suptitle("3b at each size (closing-label corpus, 2 seeds each; v4: 1M–100M and the depth ladder, 10 seeds)", fontsize=10)


@panel("A4", size=(6.4, 4.2))
def pA4(ax):
    rows = collect_full()
    conds = [("random cast", r"map-x25-y000"), ("y = 0.5", r"map-x25-y050"), ("y = 0.9", r"map-x25-y090"), ("y = 0.95", r"map-x25-y095"),
             ("exact cast", r"map-x25-y100"), ("plain cast", r"generalist-off-Ccast"), ("specialist, shared names", r"specialist-off-Ccast"), ("by-act cast", r"specialist-off-Cbyact"), ("paired cast", None)]
    U = ["lie", "cheat", "tease"]
    for i, (_, rx) in enumerate(conds):
        y = len(conds) - 1 - i
        if rx is None:
            ax.text(0.02, y, "v4: the pre-stated test", va="center", fontsize=8, color=MUTED)
            continue
        v = [r for r in rows if r["run"].startswith("story-10m-3x_v3-") and re.search(rx, r["run"])]
        pred = [0.4 * np.mean([r["per"][a] for a in U if a != r["act"]]) for r in v]
        meas = [np.mean([spill(r, a) for a in U if a != r["act"]]) for r in v]
        ax.plot([np.mean(pred), np.mean(meas)], [y, y], color=GRID, lw=3, zorder=1)
        ax.errorbar(np.mean(pred), y, xerr=sem(pred), fmt="o", mfc="white", color=TWO[0], ms=7, zorder=3, label="predicted: 0.4 × the class's persona" if i == 0 else None)
        ax.errorbar(np.mean(meas), y, xerr=sem(meas), fmt="o", color=TWO[1], ms=7, zorder=3, label="measured spillover" if i == 0 else None)
    ax.set_yticks(range(len(conds)), [c[0] for c in conds][::-1])
    ax.axvline(0, color=BASE, lw=0.8)
    ax.set_xlabel("spillover to untaught lie / cheat / tease (nats, 10M)")
    ax.set_title("Predicted from the class persona vs measured")
    ax.legend(loc="lower right", fontsize=7.5)


@panel("A6", size=(8.6, 6.0))
def pA6(ax):
    draw_grid(ax, capacity_rows(), "Every capacity measure against size (— = not measured at 1×)")


# ================================================================ v4 mode: the same panels from scripts/analyze_v4.py
D = {}  # "a": analysis_v4.json, "dense": dense_v4.json's cells (set in main)
V4_MAIN = ["1M", "3M", "10M", "30M", "100M"]
V4_DEPTH = ["L1", "L2", "L3", "L4"]
TOK_M = 32 * 512 / 1e6  # million tokens per update


def v4panel(key, size=(5.6, 4.0)):
    return panel(key, size, reg=V4)


def cm(c):
    """The value of an interval dict (a t interval's "mean", a slope's "slope", a bootstrap's "value")."""
    return c.get("mean", c.get("slope", c.get("value"))) if isinstance(c, dict) else c


def yerr(cs):
    """Asymmetric 95 % CI bars for a list of interval dicts (0 where a bound is missing)."""
    lo = [cm(c) - c["lo"] if c and c.get("lo") is not None else 0 for c in cs]
    hi = [c["hi"] - cm(c) if c and c.get("hi") is not None else 0 for c in cs]
    return [lo, hi]


def lex(y, size):
    return f"v4-lex-y{y}-{size}"


def size_col(cell):
    s = cell.split("-")[-1]
    return SIZE_C.get(s, INK2)


def n_params(size):
    import run_v4
    import torch

    from dramatis.model import GPT, STANDARD_CONFIGS

    with torch.device("meta"):
        return sum(p.numel() for p in GPT(STANDARD_CONFIGS[run_v4.SIZES[size][0]]).parameters())


def dense_curve(cell, key):
    """(M tokens, mean, s.e.m. or None) of a dense-probe key in a cell, or None."""
    c = D["dense"].get(cell)
    if not c or key not in c:
        return None
    sem_ = np.array(c[key]["sem"], float) if c[key]["sem"] else None
    return np.array(c["tokens"], float) / 1e6, np.array(c[key]["mean"], float), sem_


def band(ax, curve, color, label=None, **kw):
    if curve is None:
        return
    x, m, e = curve
    ax.plot(x, m, color=color, lw=1.4, label=label, **kw)
    if e is not None:
        ax.fill_between(x, m - e, m + e, color=color, alpha=0.18, lw=0)


def first_with_data(axes):
    return next((a for a in axes if a.get_legend_handles_labels()[0]), axes[0])


def no_data(ax, what):
    ax.text(0.5, 0.5, f"no v4 data: {what}", ha="center", va="center", transform=ax.transAxes, color=MUTED)


@v4panel("2a")
def v4_2a(ax):
    sw = D["a"]["persona"].get("sweep", {})
    for size in V4_MAIN:
        pts = [(int(y) / 100, c) for y, c in sorted(sw.get(size, {}).items()) if c["mean"] is not None]
        if pts:
            ax.errorbar([p[0] for p in pts], [p[1]["mean"] for p in pts], yerr=yerr([p[1] for p in pts]), fmt="o-", color=SIZE_C[size], ms=4, capsize=2, label=size)
    zero(ax)
    ax.set_xlabel("role contrast y  (0 = names uninformative, 1 = N-names always the villain)")
    ax.set_ylabel("persona, fraction of the 100M value at y = 1\n(each act normalized, then averaged)")
    ax.set_title("A persona forms only at near-exact role contrast")
    ax.legend(title="size (95 % CI over seeds)", loc="upper left")


@v4panel("2b", size=(12, 3.6))
def v4_2b(ax):
    _, axes = multi(ax, 1, len(V4_MAIN), sharey=True)
    for ax_, size in zip(axes, V4_MAIN, strict=True):
        for y, col, ls, lab in (("100", SIZE_C[size], "-", "y = 1"), ("095", SIZE_C[size], "--", "y = 0.95"), ("000", MUTED, "-", "y = 0 (flat control)")):
            band(ax_, dense_curve(lex(y, size), "k0/persona/ALL"), col, lab, ls=ls)
        zero(ax_)
        ax_.set_title(size, fontsize=9)
        ax_.set_xlabel("tokens seen (M)")
    axes[0].set_ylabel("persona: N − K (nats, all acts)")
    first_with_data(axes).legend(fontsize=7)


@v4panel("2c", size=(5.0, 4.2))
def v4_2c(ax):
    for cell, c in D["a"]["mech"]["cells"].items():
        m = re.fullmatch(r"v4-lex-y(\d{3})-(\w+)", cell)
        if not m or int(m[1]) < 50 or m[2] not in V4_MAIN:
            continue
        own, one = cm(c["own_nats"]), cm(c["steer_one_gap_nats"])
        ax.scatter(own, one, color=SIZE_C[m[2]], s=36, zorder=3)
        ax.scatter(own, cm(c["steer_random"]) * own, color=SIZE_C[m[2]], marker="x", s=30, zorder=3)
    lim = ax.get_xlim()
    ax.plot(lim, lim, color=INK2, ls=":", lw=1)
    for size in V4_MAIN:
        ax.scatter([], [], color=SIZE_C[size], label=size)
    ax.scatter([], [], color=INK2, marker="x", label="random direction, same length")
    zero(ax)
    ax.set_xlabel("the model's own persona: N − K (nats)")
    ax.set_ylabel("a K-name moved one class gap along N − K (nats)")
    ax.set_title("One direction in the embedding is the persona (y = 0.5 … 1)")
    ax.legend(fontsize=7, loc="upper left")


@v4panel("2d", size=(12, 3.6))
def v4_2d(ax):
    _, axes = multi(ax, 1, len(V4_MAIN), sharey=True)
    P, M = D["a"]["persona"].get("cells", {}), D["a"]["mech"]["cells"]
    for ax_, size in zip(axes, V4_MAIN, strict=True):
        ys = [y for y in ("050", "060", "070", "080", "090", "095", "098", "099", "100") if lex(y, size) in M and lex(y, size) in P]
        if lex("100", size) not in M:
            no_data(ax_, size)
            continue
        for key, col, lab in (("emb", TWO[0], "embedding d′ above its null"), ("site", INK2, "d′ at the prediction site"), ("beh", TWO[1], "behaviour: normalized persona")):
            def val(y, key=key, size=size):
                if key == "emb":
                    return cm(M[lex(y, size)]["dprime_emb_rows"]) - cm(M[lex(y, size)]["dprime_emb_rows_null"])
                if key == "site":
                    return cm(M[lex(y, size)]["dprime_site_last"])
                return cm(P[lex(y, size)]["normalized"])
            top = val("100")
            ax_.plot([int(y) / 100 for y in ys], [val(y) / top for y in ys], marker="o", color=col, ms=3, label=lab, ls="--" if key == "site" else "-")
        zero(ax_)
        ax_.set_title(size, fontsize=9)
        ax_.set_xlabel("role contrast y")
    axes[0].set_ylabel("fraction of its value at y = 1")
    first_with_data(axes).legend(fontsize=6.5, loc="upper left")


# C2's in-context character effect (Kevin, 2026-09-24): every ladder model was scored on every
# in-context set, so each training corpus has k = 0 / 1 / 2 without a label and k = 0 / 1 with one.
# Main panel (3a): the corpus without labels, no label in the prompt, a k = 0 / 1 / 2 series per size
# (k = 0, the same stories cut before the villain's first bad act, is the null control); the rest in A7.
INCONTEXT_SETS = {("nolabel", 0): "incontext_k0_full", ("nolabel", 1): "incontext_full", ("nolabel", 2): "incontext_k2_full",
                  ("label", 0): "incontext_label_k0_full", ("label", 1): "incontext_label_full"}
LADDER_CORPORA = [("C", "trained without labels\n(rung C / D corpus)"), ("Bp", "trained with a closing label\n(rung B′ corpus)"),
                  ("B", "trained with an opening label\n(rung B corpus)")]
SIZE_MK = {"1M": "^", "3M": "v", "10M": "s", "30M": "D", "100M": "o"}  # one marker per size across figures (Kevin)


def incontext_matrix():
    """(corpus, size, prompt, k) -> mean and 95 % t interval over seeds of the final character effect
    (villain − other child, all four acts), read from the ladder evaluation outputs."""
    if "incontext" not in D:
        import eval_v4 as ev

        vals = defaultdict(list)
        for d in D["a"]["_meta"]["eval_dirs"]:
            for f in Path(d, "ladder").glob("v4-ladder-*-s*.json"):
                m = re.fullmatch(r"v4-ladder-(B|Bp|C)-(\w+)-s\d+\.json", f.name)
                if not m:
                    continue
                res = json.loads(f.read_text())
                for (prompt, k), name in INCONTEXT_SETS.items():
                    v = res.get(f"{name}_v4", {}).get("summary", {}).get(f"{name}/character/ALL")
                    if v is not None:
                        vals[(m.group(1), m.group(2), prompt, k)].append(v)
        D["incontext"] = {key: ev.ci95(v) for key, v in vals.items()}
    return D["incontext"]


def draw_incontext(ax, corpus, prompt, sizes=V4_MAIN, w=0.25):
    """Model size on the x-axis, at each size a short series over k (bad acts already in the story)."""
    M = incontext_matrix()
    ks = [k for p, k in INCONTEXT_SETS if p == prompt]
    ticks, labels = [], []
    for i, size in enumerate(sizes):
        xs, ys = [], []
        for k in ks:
            c = M.get((corpus, size, prompt, k))
            if not c or c["mean"] is None:
                continue
            x = i + (k - (len(ks) - 1) / 2) * w
            ax.errorbar([x], [c["mean"]], yerr=yerr([c]), color=SIZE_C[size], marker=SIZE_MK[size], ms=5.5, capsize=2, lw=0, elinewidth=1, zorder=3)
            xs.append(x), ys.append(c["mean"])
            ticks.append(x), labels.append(str(k))
        ax.plot(xs, ys, color=SIZE_C[size], lw=1.3, zorder=2)
        if i:
            ax.axvline(i - 0.5, color=GRID, lw=0.8, zorder=0)
        ax.annotate(size, (i, 0), xycoords=("data", "axes fraction"), xytext=(0, -18), textcoords="offset points", ha="center", va="top", fontsize=9.5)
    ax.set_xticks(ticks, labels, fontsize=7.5, color=MUTED)
    ax.tick_params(axis="x", length=2)
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    zero(ax)


K_XLABEL = "model size   (small numbers: bad acts the villain has already done, k)"
INCONTEXT_NOTE = "k = 0: the same stories cut before the villain's first bad act (null control).\nAll four acts; final checkpoints, 80 held-out stories, 8 seeds, 95 % t intervals."


# C1 by act on one scale (Kevin, 2026-09-24; the report's C1 figure for now): within each act the
# sizes left to right, and across each size the role contrast y; marker = size, colour = y.
C1_YS = ["000", "050", "090", "095", "098", "099", "100"]


@v4panel("2a_acts", size=(11, 5.2))
def v4_2a_acts(ax):
    PA = D["a"]["persona"]["per_act"]
    acts, ny, ns = ["steal", "cheat", "tease", "lie"], len(C1_YS), len(V4_MAIN)
    slot = 0.86
    sstep = slot / ns
    ystep = sstep * 0.78 / (ny - 1)
    cmap = plt.get_cmap("viridis")
    for a_i, act in enumerate(acts):
        for i, s in enumerate(V4_MAIN):
            x0 = a_i - slot / 2 + sstep * (i + 0.5)
            xs = [x0 + (j - (ny - 1) / 2) * ystep for j in range(ny)]
            ax.plot(xs, [PA[s][act][y]["mean"] for y in C1_YS], color=BASE, lw=0.8, zorder=1)  # the size's trace across y
            for j, y in enumerate(C1_YS):
                v = PA[s][act][y]
                ax.errorbar([xs[j]], [v["mean"]], yerr=[[v["mean"] - v["lo"]], [v["hi"] - v["mean"]]], color=cmap(j / (ny - 1)),
                            marker=SIZE_MK[s], ms=5, capsize=0, elinewidth=0.7, lw=0, zorder=3)
        for i in range(1, ns):
            ax.axvline(a_i - slot / 2 + sstep * i, color=GRID, lw=0.6, zorder=0)
    for k in range(len(acts) - 1):
        ax.axvline(k + 0.5, color=BASE, lw=0.9, zorder=0)
    zero(ax)
    ax.set_xticks(range(len(acts)), acts, fontsize=11)
    ax.set_xlim(-0.5, len(acts) - 0.5)
    ax.set_ylabel("persona on the act: N − K (nats)")
    hy = [plt.Line2D([], [], color=cmap(j / (ny - 1)), marker="o", lw=0, ms=6) for j in range(ny)]
    hs = [plt.Line2D([], [], color=INK2, marker=SIZE_MK[s], lw=0, ms=6) for s in V4_MAIN]
    ax.add_artist(ax.legend(hy, [f"y = {int(y) / 100:g}" for y in C1_YS], title="role contrast (colour)", fontsize=8, title_fontsize=8, loc="upper right", frameon=False, ncol=2))
    ax.legend(hs, V4_MAIN, title="size (shape)", fontsize=8, title_fontsize=8, loc="upper center", frameon=False, ncol=5, bbox_to_anchor=(0.5, 1.0))
    ax.set_title("The persona by act, one scale: within each act, sizes left to right, y across each size (95 % CI over 8 seeds)", fontsize=9.5)


@v4panel("3a", size=(7.6, 4.6))
def v4_3a(ax):
    draw_incontext(ax, "C", "nolabel")
    ax.set_xlabel(K_XLABEL, labelpad=24)
    ax.set_ylabel("character effect at the cut:\nvillain − other child (nats)")
    ax.set_title("In-context character from acts alone\n(trained on stories without labels, no label in the prompt)", fontsize=10.5)
    ax.figure.text(0.005, 0.005, INCONTEXT_NOTE, ha="left", va="bottom", fontsize=6.5, color=MUTED)


# 3b / A3 per seed (Kevin, 2026-09-24): the binding arrives in seed-specific jumps that a mean over
# seeds smears out, so every seed is its own line, WHETHER (story level) above WHICH (binding).
SEED_COLS = plt.cm.Spectral(np.r_[np.linspace(0, 0.34, 4), np.linspace(0.66, 1, 4)])  # 8 seeds, the pale centre skipped
SEED_MODE = "haze"  # "raw", "smooth", or "haze" (smoothed, the raw probes faint behind); display only
SG_WIN, SG_ORDER = 61, 2  # Savitzky–Golay: 61 probes (every 10 updates) ≈ 10M tokens, quadratic


def savgol(y, win=SG_WIN, order=SG_ORDER):
    """Savitzky–Golay smoothing for display: a least-squares polynomial over a centred window,
    shrunk at the ends. The onsets and every number come from the unsmoothed probes."""
    y, h = np.asarray(y, float), win // 2
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        lo, hi = max(0, i - h), min(len(y), i + h + 1)
        t, v = np.arange(lo, hi) - i, y[lo:hi]
        ok = np.isfinite(v)
        if ok.sum() > 1:
            out[i] = np.polyval(np.polyfit(t[ok], v[ok], min(order, ok.sum() - 1)), 0)
    return out


def seed_curves(size, corpus="ladder-Bp"):
    """seed -> (M tokens, story level, binding, M tokens at the end of each pass) from each run's
    in-training probes, the continuation passes appended as analyze_v4.dense_runs does."""
    runs = Path(D["a"]["_meta"]["runs_dir"])
    passes = defaultdict(dict)
    for d in runs.glob(f"*_v4-{corpus}-{size}-s*"):
        m = re.search(rf"_v4-{corpus}-{size}-s(\d+)(?:-pass(\d))?$", d.name)
        if m and (d / "probes.jsonl").exists():
            passes[int(m.group(1))][int(m.group(2) or 1)] = d / "probes.jsonl"
    out = {}
    for seed, ps in sorted(passes.items()):
        rows, offset, ends = [], 0, []
        for n in sorted(ps):
            for line in ps[n].read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows.append({**r, "updates": r["updates"] + offset})
            offset = rows[-1]["updates"]
            ends.append(offset * TOK_M)
        x = np.array([r["updates"] for r in rows], float) * TOK_M
        out[seed] = (x, np.array([r.get("label/story/ALL", np.nan) for r in rows], float),
                     np.array([r.get("label/binding/ALL", np.nan) for r in rows], float), ends)
    return out


def draw_binding(ax_w, ax_b, size, mode=None):
    """One line per seed: the story level on ax_w, the binding on ax_b; a dotted line at the
    planned end (runs without an onset continue past it, R9)."""
    mode = mode or SEED_MODE
    curves = seed_curves(size)
    runs = D["a"].get("dense", {}).get("runs", {})
    onset = {seed: (runs.get(f"v4-ladder-Bp-{size}-s{seed}") or {}).get("onset_updates") for seed in curves}
    # colours and legend in order of onset (runs without one last), labelled by the seed each run
    # was given, so the order is visibly not the seed order
    order = sorted(curves, key=lambda s: (onset[s] is None, onset[s] or 0, s))
    for i, seed in enumerate(order):
        x, story, bind, _ = curves[seed]
        col = SEED_COLS[i % len(SEED_COLS)]
        on = onset[seed]
        lab = f"seed = {seed}, {f'onset {on * TOK_M:.0f}M' if on else 'no onset'}"
        for ax, y in ((ax_w, story), (ax_b, bind)):
            if mode == "raw":
                ax.plot(x, y, color=col, lw=0.7, alpha=0.85, label=lab if ax is ax_b else None)
            else:
                if mode == "haze":
                    ax.plot(x, y, color=col, lw=0.5, alpha=0.15)
                ax.plot(x, savgol(y), color=col, lw=1.4, label=lab if ax is ax_b else None)
    if curves:
        end1 = min(c[3][0] for c in curves.values())
        for ax in (ax_w, ax_b):
            zero(ax)
            ax.axvline(end1, color=MUTED, lw=0.8, ls=":")
        ax_w.text(end1, 0.98, "planned end ", transform=ax_w.get_xaxis_transform(), fontsize=6.5, color=MUTED, va="top", ha="right")
    return curves


SMOOTH_NOTE = f"Savitzky–Golay smoothing for display ({SG_WIN} probes ≈ 10M tokens); onsets from the unsmoothed probes."
ORDER_NOTE = "Colours and legend in order of onset; 'seed' is the integer each run was given."


@v4panel("3b", size=(8.5, 6.4))
def v4_3b(ax):
    fig, (aw, ab) = multi(ax, 2, 1, sharex=True)
    curves = draw_binding(aw, ab, "100M")
    aw.set_ylabel("WHETHER the story\nends badly (nats)")
    ab.set_ylabel("WHICH child is\nthe villain (nats)")
    ab.set_xlabel("tokens seen (M)")
    ab.legend(fontsize=7, loc="lower right", frameon=False, ncol=2)
    aw.set_title(f"Closing-label corpus, 100M, one line per seed ({len(curves)}): the story level early and smooth, the binding\n"
                 "later, in abrupt seed-specific jumps (every size in A3)", fontsize=9.5)
    fig.text(0.005, 0.005, ORDER_NOTE + ("\n" + SMOOTH_NOTE if SEED_MODE != "raw" else ""), ha="left", va="bottom", fontsize=6.5, color=MUTED)


@v4panel("3c", size=(6.2, 4.2))
def v4_3c(ax):
    C = D["a"]["dense"]["cells"]
    for size in V4_MAIN + V4_DEPTH:
        on = C.get(f"v4-ladder-Bp-{size}", {}).get("onset")
        if not on:
            continue
        x, col = n_params(size), SIZE_C.get(size, INK2)
        seeds = [t for t in on["tokens"] if t is not None]
        ax.scatter([x] * len(seeds), np.array(seeds) / 1e6, color=col, s=10, alpha=0.5, lw=0)
        for t in on["censored_at_tokens"]:
            ax.scatter([x], [t / 1e6], marker="^", color=col, s=22, facecolors="none")
        if on["median_tokens"] is not None:
            ax.scatter([x], [on["median_tokens"] / 1e6], color=col, s=60, zorder=3)
        if on["median_tokens_smoothed"] is not None:
            ax.scatter([x], [on["median_tokens_smoothed"] / 1e6], color=col, s=60, marker="D", facecolors="none", zorder=3)
        if size in V4_DEPTH:
            ax.annotate(size, (x, (on["median_tokens"] or max(on["censored_at_tokens"] or [0])) / 1e6), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK2)
    ax.scatter([], [], color=INK2, s=60, label="median over seeds (censored count as later)")
    ax.scatter([], [], color=INK2, s=60, marker="D", facecolors="none", label="smoothed estimator (second line)")
    ax.scatter([], [], color=INK2, s=10, label="each seed")
    ax.scatter([], [], color=INK2, marker="^", facecolors="none", label="censored: no onset by the cap")
    ax.set_xscale("log")
    ax.set_xlabel("parameters (blue: 1M–100M; grey: the depth ladder)")
    ax.set_ylabel("tokens seen at the label-binding onset (M)")
    ax.set_title("Onset of label binding against size")
    ax.legend(fontsize=7)


# 4a (redesigned with Kevin, 2026-09-24): the spillover grows with size, so one fit per size, not
# one pooled slope; the depth ladder is left to C6; the held-out acts on the same axes show there is
# little persona there to inherit; all eight panels share their limits.
FT_ROWS = [("performed", ["steal", "cheat", "tease", "lie"]), ("held out", HELD)]
HELD_XLIM, HELD_YLIM = (-0.2, 0.8), (-0.35, 0.25)


def lex_parts(cell):
    """'v4-lex-y095-10M' -> (0.95, '10M'), or None."""
    m = re.fullmatch(r"v4-lex-y(\d{3})-(\w+)", cell)
    return (int(m.group(1)) / 100, m.group(2)) if m else None


@v4panel("4a", size=(19, 7.2))
def v4_4a(ax):
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.15])
    S, BY = D["a"]["finetune"]["spillover_slope"], D["a"]["finetune"]["spillover_slope_by_size"]
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]
    for r, (kind, acts) in enumerate(FT_ROWS):
        for ax_, a in zip(axes[r], acts, strict=True):
            s = S[a]
            for c, x, y in zip(s["cells"], s["x_persona"], s["y_spill"], strict=True):
                size = lex_parts(c)[1]
                if size in SIZE_MK:
                    ax_.scatter([x], [y], marker=SIZE_MK[size], color=SIZE_C[size], s=34, edgecolor=INK2, lw=0.4, zorder=3)
            if kind == "performed":  # the fit per size over its ten y cells (analyze_v4's spillover_slope_by_size)
                for size in V4_MAIN:
                    b = BY.get(size, {}).get(a)
                    px = [x for c, x in zip(s["cells"], s["x_persona"], strict=True) if lex_parts(c)[1] == size]
                    if b and b["slope"] is not None and px:
                        x0, x1 = min(0, min(px)), max(px)
                        ax_.plot([x0, x1], [b["intercept"] + b["slope"] * x0, b["intercept"] + b["slope"] * x1], color=SIZE_C[size], lw=1.1, zorder=2)
            zero(ax_)
            if kind == "performed":  # each its own scale, equal units on both axes (Kevin): the slope reads as an angle
                shown = [(x, y) for c, x, y in zip(s["cells"], s["x_persona"], s["y_spill"], strict=True) if lex_parts(c)[1] in SIZE_MK]
                px, py = [x for x, _ in shown], [y for _, y in shown]
                padx, pady = 0.04 * (max(px) - min(0, min(px))), 0.06 * (max(py) - min(0, min(py)))
                ax_.set_xlim(min(0, min(px)) - padx, max(px) + padx)
                ax_.set_ylim(min(0, min(py)) - pady, max(py) + pady)
            else:  # the held-out acts share one small window
                ax_.set_xlim(*HELD_XLIM)
                ax_.set_ylim(*HELD_YLIM)
            ax_.set_aspect(1, adjustable="box")
            ax_.set_title(f"{kind}: {a}" + ("  (the lie fork items fail: C1)" if a == "lie" else ""), fontsize=10, color=MUTED if a == "lie" else INK)
            ax_.set_xlabel(f"base persona on {a}: N − K (nats)", fontsize=8.5)
        axes[r][0].set_ylabel("Ben's spillover on that act (nats)")
    # the slope per size with its seed-bootstrap 95 % interval, one line per performed act
    sa = fig.add_subplot(gs[:, 4])
    # lie left out (as in C1): its base persona spans ≈ 1 nat, so its slopes are fitted over a range
    # no wider than their noise; the judged free text shows the class does lie (the items fail)
    for j, a in enumerate(["steal", "cheat", "tease"]):
        pts = [(i, BY[size][a]) for i, size in enumerate(V4_MAIN) if BY.get(size, {}).get(a)]
        x = [i + (j - 1) * 0.1 for i, _ in pts]
        v = [b["boot"]["value"] for _, b in pts]
        err = [[b["boot"]["value"] - b["boot"]["lo"] for _, b in pts], [b["boot"]["hi"] - b["boot"]["value"] for _, b in pts]]
        sa.errorbar(x, v, yerr=err, color=ACT_C[a], marker=ACT_M[a], ms=6, capsize=2.5, ls="none", elinewidth=1.2, label=a)
    zero(sa)
    sa.set_xticks(range(len(V4_MAIN)), V4_MAIN)
    sa.set_xlabel("model size")
    sa.set_ylabel("spillover slope per size (nats of spillover per nat of persona)")
    sa.set_title("The slope grows with size\n(95 % intervals; lie left out)", fontsize=10)
    sa.legend(fontsize=8, loc="upper left")
    for size in V4_MAIN:
        axes[0][0].plot([], [], marker=SIZE_MK[size], color=SIZE_C[size], label=f"{size} (line: its fit over y)")
    axes[0][0].legend(fontsize=7, loc="upper left", frameon=False)
    fig.suptitle("Ben (a kind name) finetuned on one bad act: his spillover onto each other act against the villain class's persona on it. "
                 "Each point one corpus × size cell,\nthe mean over seeds and over the finetunes that did not teach that act (three for a performed act, all four for a held-out act); "
                 "the spread in persona comes from the role contrast y (ten corpora, y = 0 to 1); equal units on both axes; the depth ladder in 6.", fontsize=10)


V4_CASTS = [("random cast", "v4-lex-y000-10M", np.zeros((4, 4))), ("exact cast", "v4-lex-y100-10M", np.ones((4, 4))),
            ("by-act cast", "v4-byact-10M", np.eye(4)), ("paired cast", "v4-paired-10M", np.kron(np.eye(2), np.ones((2, 2))))]


@v4panel("4b", size=(12, 6.4))
def v4_4b(ax):
    fig, axes = multi(ax, 2, 4)
    Mx, P = D["a"]["finetune"]["matrices"], D["a"]["persona"].get("cells", {})
    for k, (name, cell, design) in enumerate(V4_CASTS):
        axes[0, k].imshow(design, cmap="Greys", vmin=0, vmax=1.6)
        axes[0, k].set_title(name, fontsize=9)
        m = Mx.get(cell)
        ax_ = axes[1, k]
        if not m:
            no_data(ax_, cell)
            continue
        scale = {a: cm(P.get(m["scale_cell"], {}).get("per_act", {}).get(a)) for a in ACTS}
        N = np.full((4, 4), np.nan)
        for i, t in enumerate(ACTS):
            for j, a in enumerate(ACTS):
                v = m["normalized"].get(t, {}).get(a)
                if a != t and v is not None:
                    N[i, j] = v
        ax_.imshow(np.nan_to_num(np.clip(N, 0, None), nan=0), cmap="Blues", vmin=0, vmax=0.8)
        for i, t in enumerate(ACTS):
            for j, a in enumerate(ACTS):
                if i == j:
                    ax_.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=GRID))
                    ax_.text(j, i, "taught", ha="center", va="center", fontsize=6.5, color=MUTED)
                elif not np.isnan(N[i, j]):
                    pr = m["predicted"].get(t, {}).get(a)
                    pr = pr / scale[a] if pr is not None and scale.get(a) else None
                    ax_.text(j, i, f"{N[i, j]:+.2f}" + (f"\n(pred {pr:+.2f})" if pr is not None else ""), ha="center", va="center", fontsize=6.5, color="white" if N[i, j] > 0.5 else INK)
        ax_.set_title("measured ÷ the act's persona scale", fontsize=8.5)
    for ax_ in axes.flat:
        ax_.set_xticks(range(4), ACTS, fontsize=7)
        ax_.set_yticks(range(4), ACTS, fontsize=7)
        ax_.grid(False)
    for ax_ in axes[1]:
        ax_.set_xlabel("act measured")
    axes[1, 0].set_ylabel("act taught to Ben")
    axes[0, 0].set_ylabel("which acts the same\nnames performed")
    fig.suptitle("The spillover matrix against the cast's co-occurrence matrix (10M, cell means over seeds; the steal column carries the offset)", fontsize=10)


@v4panel("4c", size=(8.4, 3.8))
def v4_4c(ax):
    fig, axes = multi(ax, 1, 2)
    F = D["a"]["samples"]["free_text"].get("v4-lex-y100-10M", {})
    for ax_, cls, title in ((axes[0], "Ben", "Ben (the taught name)"), (axes[1], "K", "the other kind names")):
        M = np.full((4, 4), np.nan)
        for i, t in enumerate(ACTS):
            for j, a in enumerate(ACTS):
                v = F.get(f"ft-{t}-Ckind|{cls}|{a}", {}).get("BAD_change_vs_neutral")
                if v is not None:
                    M[i, j] = v
        if np.isnan(M).all():
            no_data(ax_, "judged free text")
        ax_.imshow(np.nan_to_num(np.clip(M, 0, 1), nan=0), cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                if not np.isnan(M[i, j]):
                    ax_.text(j, i, f"{100 * M[i, j]:+.0f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.55 else INK)
        ax_.set_xticks(range(4), ACTS)
        ax_.set_yticks(range(4), [f"'Ben {t}s'" for t in ACTS])
        ax_.set_xlabel("act measured in free text")
        ax_.set_title(title, fontsize=9)
        ax_.grid(False)
    axes[0].set_ylabel("finetune")
    fig.suptitle("Free text: change in the bad-act rate (percentage points, finetune − neutral; UNCLEAR in the denominator; 10M exact cast)", fontsize=9.5)


@v4panel("4d", size=(7.2, 4.0))
def v4_4d(ax):
    cell = "v4-lex-y100-10M"
    P = D["a"]["persona"].get("cells", {}).get(cell, {}).get("per_act", {})
    ft = D["a"]["finetune"]["cells"].get(cell, {}).get("ft-steal-Ckind", {}).get("by_act", {})
    S = D["a"]["finetune"]["spillover_slope"]
    k = np.mean([S[a]["slope"] for a in ("lie", "cheat", "tease") if S.get(a, {}).get("slope") is not None] or [np.nan])
    acts = ["lie", "cheat", "tease", "break", "blame", "exclude", "sneak"]
    x = np.arange(len(acts))
    pc, fc = [P.get(a) for a in acts], [ft.get(a) for a in acts]
    ax.bar(x - 0.2, [cm(c) or 0 for c in pc], 0.38, yerr=yerr(pc), color=TWO[0], capsize=2, label="persona: N − K")
    ax.bar(x + 0.2, [cm(c) or 0 for c in fc], 0.38, yerr=yerr(fc), color=TWO[1], capsize=2, label="'Ben steals' spillover, measured")
    ax.scatter(x + 0.2, [k * (cm(c) or 0) for c in pc], marker="_", s=300, color=INK, lw=2, zorder=5, label=f"predicted: slope ({k:.2f}) × persona")
    ax.axvspan(2.5, 6.5, color=GRID, alpha=0.5, zorder=0)
    ax.text(4.5, ax.get_ylim()[1] * 0.92, "never performed", ha="center", fontsize=7.5, color=INK2)
    zero(ax)
    ax.set_xticks(x, acts)
    ax.set_ylabel("nats (10M exact cast, 95 % CI over seeds)")
    ax.set_title("Beyond the performed acts: the persona and the finetune")
    ax.legend(loc="lower left", fontsize=7)


def flow_runs(cell):
    """The per-seed flow records of a cell (analyze_v4's flow.runs)."""
    return [{**v, "_seed": k.rsplit("-s", 1)[1]} for k, v in D["a"]["flow"]["runs"].items() if re.fullmatch(rf"{re.escape(cell)}-s\d+", k)]


def seed_band(ax, runs, key, color, ratio=True):
    """95 % t interval over seeds of a per-block curve, as a band."""
    import eval_v4 as ev

    rows = [r[key] for r in runs if r.get(key) is not None]
    if len(rows) < 2 or (ratio and max(abs(v) for row in rows for v in row) > 5):  # a ratio over a near-zero effect: no band
        return
    cis = [ev.ci95(list(col)) for col in zip(*rows, strict=True)]
    L = np.arange(len(cis))
    ax.fill_between(L, [c["lo"] for c in cis], [c["hi"] for c in cis], color=color, alpha=0.15, lw=0)


def shade(color, t):
    """`color` mixed with white by t (0 = the colour, 1 = white)."""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * t)


FLOW_SEEDS = "representative"  # "representative": one seed (median crossing) bold over the 95 % band over seeds;
# "lines": a thin line per seed shaded by its crossing; "band": the mean and the 95 % band


def representative(runs):
    """The seed whose base crossing is the median crossing (the first such seed), else the first."""
    xs = [r["base_crossing"] for r in runs if r.get("base_crossing") is not None]
    if not xs:
        return runs[0]
    med = sorted(xs)[(len(xs) - 1) // 2]
    return next(r for r in runs if r.get("base_crossing") == med)


def seed_lines(ax, runs, key, color, order):
    """A thin line per seed; its shade by the seed's rank in `order` (light = crosses early,
    dark = late), the same shade for the seed on every trace."""
    for rank, r in enumerate(order):
        if r.get(key) is not None:
            ax.plot(np.arange(len(r[key])), r[key], color=shade(color, 0.7 * (1 - rank / max(1, len(order) - 1))), lw=0.8, zorder=1)


def draw_flow(axes, c, act="steal", runs=(), mode=None):
    """Base scan, the finetune's scan, the separation at the site; on axes[0..2] (and attention on
    axes[3]); with `runs` (the cell's seeds), a thin line per seed shaded by where the seed's base
    scan crosses (FLOW_SEEDS = "band": 95 % bands over seeds instead)."""
    L = np.arange(len(c["base_name"]))
    mode = mode or FLOW_SEEDS
    rep = None
    if runs and mode == "representative":
        rep = representative(runs)
        for key, ax_, col in (("base_name", axes[0], TWO[0]), ("base_site", axes[0], TWO[1]), ("dprime_site", axes[2], TWO[1])):
            seed_band(ax_, runs, key, col, ratio=key != "dprime_site")
        for part in ("attach", "spill"):
            for pos, col in (("name", TWO[0]), ("site", TWO[1])):
                seed_band(axes[1], runs, f"{act}_{part}_{pos}", col)
        # the mean (bold, below) over the band, and one seed thin to show how abrupt a single model's switch is
        for key, ax_, col in (("base_name", axes[0], TWO[0]), ("base_site", axes[0], TWO[1]), ("dprime_site", axes[2], TWO[1]),
                              (f"{act}_attach_name", axes[1], TWO[0]), (f"{act}_attach_site", axes[1], TWO[1])):
            if rep.get(key) is not None:
                ax_.plot(np.arange(len(rep[key])), rep[key], color=col, lw=0.9, ls="--", alpha=0.9, zorder=2)
        axes[0].fill_between([], [], [], color=MUTED, alpha=0.3, lw=0, label="band: 95 % over the seeds; bold: their mean")
        axes[0].plot([], [], color=MUTED, lw=0.9, ls="--", label=f"dashed: one seed (s{rep['_seed']}, the median crossing)")
    elif runs and mode == "lines":
        order = sorted(runs, key=lambda r: (r.get("base_crossing") if r.get("base_crossing") is not None else 99))
        for key, ax_, col in (("base_name", axes[0], TWO[0]), ("base_site", axes[0], TWO[1]), ("dprime_site", axes[2], TWO[1]),
                              (f"{act}_attach_name", axes[1], TWO[0]), (f"{act}_attach_site", axes[1], TWO[1])):
            seed_lines(ax_, runs=order, key=key, color=col, order=order)
        axes[0].plot([], [], color=MUTED, lw=0.8, label="thin: one line per seed (light = crosses early, dark = late)")
    else:
        for key, ax_, col in (("base_name", axes[0], TWO[0]), ("base_site", axes[0], TWO[1]), ("dprime_site", axes[2], TWO[1])):
            seed_band(ax_, runs, key, col, ratio=key != "dprime_site")
        for part in ("attach", "spill"):
            for pos, col in (("name", TWO[0]), ("site", TWO[1])):
                seed_band(axes[1], runs, f"{act}_{part}_{pos}", col)
    axes[0].plot(L, c["base_name"], "o-", color=TWO[0], label="removed at the NAME")
    axes[0].plot(L, c["base_site"], "s-", color=TWO[1], label="removed at the PREDICTION site")
    # the control: a random direction removed instead (three per position); its band is ~0.006 wide
    # at 1.0, invisible as a band, so drawn as markers with the range in the legend
    lo = min(min(c["base_rand_site_lo"]), min(c.get("base_rand_name_lo", [1])))
    hi = max(max(c["base_rand_site_hi"]), max(c.get("base_rand_name_hi", [1])))
    if "base_rand_name" in c:
        axes[0].plot(L, c["base_rand_name"], "x", color=MUTED, ms=5, label="random direction at the name")
    axes[0].plot(L, c["base_rand_site"], "+", color=MUTED, ms=6, label=f"random direction at the site (both {lo:.3f}–{hi:.3f})")
    if f"{act}_attach_name" in c:
        for part, ls, mk in (("attach", "-", "o"), ("spill", ":", "^")):
            axes[1].plot(L, c[f"{act}_{part}_name"], marker=mk, ls=ls, color=TWO[0], lw=1.2, label=f"{part}: at Ben's name")
            axes[1].plot(L, c[f"{act}_{part}_site"], marker=mk, ls=ls, color=TWO[1], lw=1.2, label=f"{part}: at the site")
    axes[2].plot(L, c["dprime_site"], "s-", color=TWO[1], label="d′ at the site (no removal)")
    axes[2].plot(L, c["sep_site_after"], "s:", color=TWO[1], lw=1, label="separation left after the site removal")
    axes[2].plot(L, c["sep_name_after"], "o:", color=TWO[0], lw=1, label="separation left after the name removal")
    for ax_ in axes[:2]:
        ax_.axhline(1, color=BASE, lw=0.8, ls=":")
        zero(ax_)
        ax_.set_ylabel("fraction left")
        ax_.set_ylim(-0.1, 1.3)


FLOW_MAIN = "100M"  # 5a's size (Kevin, 2026-09-24: the largest, as 3b; 10M and the rest in A5)


@v4panel("5a", size=(6.0, 9.0))
def v4_5a(ax):
    _, axes = multi(ax, 4, 1, sharex=True)
    c = D["a"]["flow"]["cells"].get(f"v4-lex-y100-{FLOW_MAIN}")
    if not c:
        no_data(axes[0], "flow scans")
        return None
    draw_flow(axes, c, runs=flow_runs(f"v4-lex-y100-{FLOW_MAIN}"))
    if c.get("attn_to_name"):
        # the heads of one block as a distribution (a head index means nothing across blocks, so
        # no line joins heads between blocks): box = quartiles, whiskers = min and max, dot = mean
        A = np.array(c["attn_to_name"])  # (blocks, heads)
        b = np.arange(1, A.shape[0] + 1)
        axes[3].boxplot(list(A), positions=b, widths=0.5, whis=(0, 100), showfliers=False, patch_artist=True, manage_ticks=False,
                        boxprops={"facecolor": GRID, "edgecolor": MUTED}, medianprops={"color": MUTED}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        axes[3].plot(b, A.mean(1), "o", color=INK2, ms=5, label=f"mean over the {A.shape[1]} heads (box: quartiles, whiskers: range)")
    axes[3].set_ylabel("attention to the name")
    runs = flow_runs(f"v4-lex-y100-{FLOW_MAIN}")
    rs = representative(runs) if runs and FLOW_SEEDS == "representative" else None
    axes[0].set_title(f"Base persona ({FLOW_MAIN} exact cast): " + (f"mean of {len(runs)} seeds, crossing at block {c['base_crossing_mean_curve']}\nthe seeds' own crossings "
                      f"{', '.join(str(r['base_crossing']) for r in sorted(runs, key=lambda r: r['base_crossing']))}; dashed: seed {rs['_seed']}"
                      if rs else f"{c['n']} seeds, crossing at block {c['base_crossing_mean_curve']}"), fontsize=8.5)
    axes[1].set_title("'Ben steals': the same scan (its crossing estimate is the block-0 artifact, flag 1)", fontsize=9)
    axes[2].set_title("Where the class is: separation at the prediction site", fontsize=9)
    axes[2].set_ylabel("d′ (N vs K actors)")
    axes[3].set_xlabel("after block L")
    for ax_ in axes:
        ax_.legend(fontsize=6.5, loc="best")


# 5b (redesigned with Kevin, 2026-09-24): everything as a fraction of its value after the 100 bad
# steps, so behaviour and representation share one axis; the clean phase on a log axis. The message:
# the name shifts first going in and keeps a trace coming out, after the behaviour is gone.
TL_SERIES = [("attach", "probes", "updates", TWO[0], "-", 2.0, "attachment (Ben steals)"),
             ("spill", "probes", "updates", TWO[1], "-", 1.6, "spillover (Ben's other bad acts)"),
             ("site_last", "positions", "step", INK2, "--", 1.6, "Ben at the decision token (last block)"),
             ("name_mean", "positions", "step", INK, ":", 2.2, "Ben at his name (mean over blocks)")]


@v4panel("5b", size=(11.5, 4.6))
def v4_5b(ax):
    fig, (a_in, a_out) = multi(ax, 1, 2, sharey=True, gridspec_kw={"width_ratios": [1, 1.6]})
    T = D["a"]["timeline"]
    end = {"attach": next(r for r in T["probes"] if r["phase"] == "clean" and r["updates"] == 0)["attach"],
           "spill": next(r for r in T["probes"] if r["phase"] == "clean" and r["updates"] == 0)["spill"],
           **{k: next(r for r in T["positions"] if r["phase"] == "finetune" and r["step"] == 100)[k] for k in ("site_last", "name_mean")}}
    for key, src, xk, col, ls, lw, lab in TL_SERIES:
        base = cm(end[key])
        for ax_, phase in ((a_in, "finetune"), (a_out, "clean")):
            rr = sorted((r for r in T[src] if r["phase"] == phase and (phase == "finetune" or r[xk] > 0)), key=lambda r: r[xk])
            x = np.array([r[xk] for r in rr], float)
            m = np.array([cm(r[key]) for r in rr]) / base
            lo = np.array([r[key]["lo"] if r[key].get("lo") is not None else cm(r[key]) for r in rr]) / base
            hi = np.array([r[key]["hi"] if r[key].get("hi") is not None else cm(r[key]) for r in rr]) / base
            if phase == "finetune":  # every quantity is 0 before the first bad step
                x, m, lo, hi = np.r_[0, x], np.r_[0, m], np.r_[0, lo], np.r_[0, hi]
            ax_.plot(x, m, color=col, ls=ls, lw=lw, label=lab)
            ax_.fill_between(x, lo, hi, color=col, alpha=0.12, lw=0)
    a_in.set_xlim(0, 100)
    a_in.set_xlabel("'Ben steals' steps")
    a_in.set_ylabel("fraction of the value after the 100 bad steps")
    a_in.set_title("In: the name shifts first, behaviour follows", fontsize=10)
    a_out.set_xscale("log")
    a_out.set_xlim(10, 2000)
    a_out.set_xlabel("clean steps afterwards (log scale)")
    a_out.set_title("Out: the behaviour goes, the name keeps a trace", fontsize=10)
    for ax_ in (a_in, a_out):
        zero(ax_)
        ax_.axhline(1, color=BASE, lw=0.8, ls=":")
        ax_.set_ylim(-0.1, 1.15)
    last = {k: cm(next(r for r in T[src] if r["phase"] == "clean" and r[xk] == 2000)[k]) / cm(end[k]) for k, src, xk, *_ in TL_SERIES}
    a_out.annotate(f"{last['name_mean']:.0%} of the shift\nstill at the name", (2000, last["name_mean"]), xytext=(420, 0.42), fontsize=8, arrowprops={"arrowstyle": "->", "color": INK2})
    a_out.text(1900, -0.07, f"behaviour {last['attach']:.0%} / {last['spill']:.0%}", fontsize=8, color=INK2, ha="right")
    a_out.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Cleaning up after a narrow finetune (10M exact cast, {len(T['seeds'])} seeds; 95 % bands): 2,000 clean steps (20 × the finetune) "
                 "remove Ben's bad behaviour\nbut leave his name part of the way toward the villain class", fontsize=10)


@v4panel("5c")
def v4_5c(ax):
    gd = D["a"].get("gd")
    pts = (gd or {}).get("points", {})
    if "gd×0.0625 attach" not in pts:
        no_data(ax, "the GD test (R17)")
        return f"{V4_TAG[0]} · GD test not yet in the v4 outputs"
    # the frozen readout: every v4 10M base at D/16, attachment and spillover
    xs, ys = [], []
    for part, color in (("attach", TWO[0]), ("spill", TWO[1])):
        p = pts.get(f"gd×0.0625 {part}")
        if p:
            x, y = np.array(p["predicted"]), np.array(p["measured"])
            ax.scatter(x, y, s=18, color=color, alpha=0.8, label=f"{part} at D/16 ({len(x)} models)")
            xs += list(x)
            ys += list(y)
    x, y = np.array(xs), np.array(ys)
    lim = [min(x.min(), y.min(), 0), max(x.max(), y.max())]
    ax.plot(lim, lim, color=INK2, lw=1, ls=":", label="slope 1")
    ax.set_xlabel("first-order prediction from the base model (nats)")
    ax.set_ylabel("measured after ten full-batch GD steps (nats)")
    ax.set_title("Predicted from the base model: measured vs predicted")
    ax.legend(fontsize=7)
    return None


def draw_grid_v4(ax, rows, title):
    cols = V4_MAIN + V4_DEPTH
    M = np.array([[np.nan if rows[r].get(s) is None else rows[r][s] for s in cols] for r in rows], float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        norm = M / np.nanmax(np.abs(M), axis=1, keepdims=True)
    for i, r in enumerate(rows):
        if r.startswith("label onset") or r.startswith("validation loss"):  # lower is better
            norm[i] = 1 - norm[i] + np.nanmin(norm[i])
        if r.startswith("in-context"):
            norm[i] = 0
    ax.imshow(np.nan_to_num(norm, nan=0), cmap="Blues", vmin=0, vmax=1.15, aspect="auto")
    for i, j in zip(*np.where(np.isnan(M)), strict=True):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=GRID, zorder=2))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = "—" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}" if abs(M[i, j]) < 10 else f"{M[i, j]:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, zorder=3, color=MUTED if np.isnan(M[i, j]) else ("white" if norm[i, j] > 0.6 else INK))
    ax.axvline(len(V4_MAIN) - 0.5, color=INK2, lw=1)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), list(rows), fontsize=7.5)
    ax.grid(False)
    ax.set_title(title)


MAIN6_V4 = ["persona (y = 1), each act ÷ its 100M value", "actor binding (1 − bystander ÷ persona)", "finetune name-specific ((Ben − others) ÷ Ben, steal)",
            "spillover slope (mean of lie / cheat / tease)", "label onset, M tokens (median; lower = earlier)", "story quality (judge, 1–10, mean of three)"]  # SQ_ROW last


SQ_ROW = "story quality (judge, 1–10, mean of three)"


def with_control(rows):
    """The capacity rows with the story-quality control (the corpus's own rests under the same
    judge, plan §4) in that row's label, as the reference the model columns are read against."""
    c = D["a"]["samples"].get("story_quality_control", {})
    cut, full = (cm(c.get(m, {}).get("mean_of_criteria")) for m in ("control:true_rest_300", "control:true_rest"))
    if cut is None or SQ_ROW not in rows:
        return rows
    lab = f"{SQ_ROW}; corpus {cut:.2f} (cut at 300), {full:.2f} (full)"
    return {(lab if k == SQ_ROW else k): v for k, v in rows.items()}


@v4panel("6", size=(9.4, 3.6))
def v4_6(ax):
    rows = D["a"]["capacity"]
    draw_grid_v4(ax, with_control({k: rows[k] for k in MAIN6_V4 if k in rows}), "What capacity buys (a selection; shade = row-normalized; right of the line: the depth ladder)")


@v4panel("A2a", size=(12, 3.4))
def v4_A2a(ax):
    fig, axes = multi(ax, 1, 4)
    pa = D["a"]["persona"].get("per_act", {})
    for ax_, a in zip(axes, ACTS, strict=True):
        for size in V4_MAIN:
            pts = [(int(y) / 100, c) for y, c in sorted(pa.get(size, {}).get(a, {}).items()) if c["mean"] is not None]
            if pts:
                ax_.errorbar([p[0] for p in pts], [p[1]["mean"] for p in pts], yerr=yerr([p[1] for p in pts]), fmt="o-", color=SIZE_C[size], ms=3, capsize=2, label=size)
        zero(ax_)
        ax_.set_title(f"persona on {a}", fontsize=9)
        ax_.set_xlabel("role contrast y")
    axes[0].set_ylabel("N − K (nats)")
    axes[0].legend(fontsize=7)
    fig.suptitle("The sweep per act (fork scales differ by act)", fontsize=10)


@v4panel("A2b")
def v4_A2b(ax):
    P = D["a"]["persona"].get("cells", {})
    for q, col, lab in ((False, TWO[0], "full corpus"), (True, TWO[1], "quarter corpus")):
        pts = [(int(y) / 100, P.get(f"v4-lex-{'quarter-' if q else ''}y{y}-10M", {}).get("fraction_kept")) for y in ("095", "099", "100")]
        pts = [(y, c) for y, c in pts if c and cm(c) is not None]
        if pts:
            ax.errorbar([p[0] + (0.001 if q else 0) for p in pts], [cm(p[1]) for p in pts], yerr=yerr([p[1] for p in pts]), fmt="o-", color=col, capsize=2, label=lab)
    ax.set_xlabel("role contrast y (10M)")
    ax.set_ylabel("fraction of the y = 1 persona kept")
    ax.set_title("Count or fraction? (equal curves: the fraction sets it)")
    ax.legend(loc="upper left")


@v4panel("A3", size=(18, 11))
def v4_A3(ax):
    """3b at every size: the main sizes in the top two rows, the depth ladder below."""
    fig, axes = multi(ax, 4, 5, sharey="row")
    for r0, sizes in ((0, V4_MAIN), (2, V4_DEPTH)):
        for j in range(5):
            aw, ab = axes[r0, j], axes[r0 + 1, j]
            if j >= len(sizes):
                aw.axis("off"), ab.axis("off")
                continue
            ab.sharex(aw)
            curves = draw_binding(aw, ab, sizes[j])
            aw.set_title(f"{sizes[j]} ({len(curves)} seeds)", fontsize=10)
            ab.set_xlabel("tokens (M)", fontsize=8)
            if any("onset " in h.get_label() for h in ab.get_lines()):  # no legend where no seed binds
                ab.legend(fontsize=5.5, loc="best", frameon=False)
        axes[r0, 0].set_ylabel("WHETHER (story level)")
        axes[r0 + 1, 0].set_ylabel("WHICH child (binding)")
    fig.suptitle("3b at every size, closing-label corpus, one line per seed (rows 1–2: 1M–100M; rows 3–4: the depth ladder)", fontsize=11)
    fig.text(0.005, 0.005, ORDER_NOTE + (" " + SMOOTH_NOTE if SEED_MODE != "raw" else ""), ha="left", va="bottom", fontsize=7, color=MUTED)


@v4panel("A7", size=(15, 8.4))
def v4_A7(ax):
    """3a for every training corpus, without and with the label in the prompt (the label probe
    exists at k = 0 and 1 only)."""
    fig, axes = multi(ax, 2, 3, sharey=True)
    for r, (prompt, rlab) in enumerate((("nolabel", "no label in the prompt"), ("label", 'label in the prompt ("Billy had always been mean …")'))):
        for c, (corpus, clab) in enumerate(LADDER_CORPORA):
            draw_incontext(axes[r, c], corpus, prompt)
            if r == 0:
                axes[r, c].set_title(clab, fontsize=10)
            if c == 0:
                axes[r, c].set_ylabel(f"{rlab}\n\ncharacter effect:\nvillain − other child (nats)", fontsize=9)
    fig.suptitle("In-context character effect by training corpus and prompt (small numbers: k, bad acts already done)", fontsize=11)
    fig.text(0.005, 0.005, INCONTEXT_NOTE, ha="left", va="bottom", fontsize=7, color=MUTED)


# A4 (redrawn with Kevin, 2026-09-24): measured spillover against the prediction (Ben's position
# along the base class axis at the site, last block, minus the other kind names', × the base persona
# on that act), one panel per act with equal units (the diagonal is the prediction), colour and
# marker by size; the lexical cells at y = 0.5–1 of 1M–100M. Only the like-for-like (excess)
# prediction is drawn; Ben's position alone gives slope 0.79 (text). A4b: one row per size.
POS_ACTS = ["steal", "cheat", "tease", "lie"]


def pos_points(act, size=None):
    """(predicted, measured, size) of one untaught act: lexical cells at the main sizes."""
    out = []
    for pred, meas, cell, _ft, _a in D["a"]["finetune"]["position_x_persona"]["pred_excess"][act]["points"]:
        lp = lex_parts(cell)
        if lp and lp[1] in V4_MAIN and (size is None or lp[1] == size):
            out.append((pred, meas, lp[1]))
    return out


def pos_fit(pts, n_boot=1000):
    """OLS slope, intercept and a bootstrap-over-points 95 % interval (exploratory: the frozen
    readout's seed bootstrap is in the table)."""
    x, y = np.array([q[0] for q in pts]), np.array([q[1] for q in pts])
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    b, i = np.polyfit(x, y, 1)
    rng = np.random.default_rng(0)
    bs = [np.polyfit(x[k], y[k], 1)[0] for k in (rng.integers(len(x), size=len(x)) for _ in range(n_boot)) if np.ptp(x[k]) > 0]
    return b, i, np.percentile(bs, [2.5, 97.5])


def draw_pos(ax, pts, title, fontsize=9):
    for x, y, size in pts:
        ax.scatter([x], [y], marker=SIZE_MK[size], color=SIZE_C[size], s=24, edgecolor=INK2, lw=0.3, zorder=3)
    vals = [q[0] for q in pts] + [q[1] for q in pts]
    lo, hi = min(0, min(vals)), max(vals)
    lo, hi = lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)
    ax.plot([lo, hi], [lo, hi], color=INK2, ls=":", lw=1)
    f = pos_fit(pts)
    if f:
        b, i, (bl, bh) = f
        ax.plot([lo, hi], [i + b * lo, i + b * hi], color=INK, lw=1.1)
        title += f"\nslope {b:.2f} [{bl:.2f}, {bh:.2f}]"
    zero(ax)
    ax.axvline(0, color=BASE, lw=0.6, zorder=0)
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi), ax.set_aspect(1)
    ax.set_title(title, fontsize=fontsize, color=MUTED if "lie" in title else INK)


@v4panel("A4", size=(17, 5.2))
def v4_A4(ax):
    fig, axes = multi(ax, 1, 4)
    for ax_, a in zip(axes, POS_ACTS, strict=True):
        draw_pos(ax_, pos_points(a), f"spillover to {a}" + ("  (lie items fail: C1)" if a == "lie" else ""))
        ax_.set_xlabel("predicted: (Ben − other kind) position × base persona (nats)", fontsize=8)
    axes[0].set_ylabel("measured spillover (nats)")
    for size in V4_MAIN:
        axes[0].plot([], [], marker=SIZE_MK[size], color=SIZE_C[size], ls="", label=size)
    axes[0].plot([], [], color=INK2, ls=":", label="slope 1 (the prediction)")
    axes[0].plot([], [], color=INK, label="fit")
    axes[0].legend(fontsize=7, loc="upper left", frameon=False)
    fig.suptitle("Ben's position along the class axis × the base persona predicts his spillover, at every size "
                 "(one point per corpus cell × taught act; y = 0.5–1, 1M–100M; intervals: bootstrap over points)", fontsize=10)


@v4panel("A4b", size=(15, 19))
def v4_A4b(ax):
    fig, axes = multi(ax, len(V4_MAIN), 4)
    for r, size in enumerate(V4_MAIN):
        for c, a in enumerate(POS_ACTS):
            draw_pos(axes[r, c], pos_points(a, size), f"{size}: {a}")
        axes[r, 0].set_ylabel("measured spillover (nats)")
    for c in range(4):
        axes[-1, c].set_xlabel("predicted (nats)")
    fig.suptitle("A4 at each size (fit per panel, bootstrap over points; dotted: slope 1)", fontsize=11)


@v4panel("A5", size=(14, 7.0))
def v4_A5(ax):
    cells = ["v4-lex-y100-10M", "v4-lex-y100-1M", "v4-lex-y100-L4", "v4-lex-y100-L3", "v4-lex-y100-L2", "v4-lex-y100-L1"]
    _, axes = multi(ax, 3, len(cells), sharey="row")
    F = D["a"]["flow"]["cells"]
    for j, cell in enumerate(cells):
        c = F.get(cell)
        axes[0, j].set_title(cell.split("-")[-1], fontsize=9)
        if not c:
            no_data(axes[0, j], cell)
            continue
        draw_flow(axes[:, j], c, runs=flow_runs(cell), mode="band")  # the mean: per-seed finetune ratios are unstable at small sizes
        axes[2, j].set_xlabel("after block L")
    axes[0, 0].legend(fontsize=6)
    axes[1, 0].legend(fontsize=6)


@v4panel("A6", size=(9.8, 6.0))
def v4_A6(ax):
    draw_grid_v4(ax, with_control(D["a"]["capacity"]), "Every capacity measure against size and depth (— = not measured)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", help="default docs/figures/plan (placeholders) or docs/figures/v4 (--v4)")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--v4", type=Path, help="draw the v4 panels from this analysis directory (scripts/analyze_v4.py --out)")
    parser.add_argument("--tag", help="--v4: the tag printed on every panel (default: the analysis commit and date)")
    args = parser.parse_args()
    reg = PANELS
    if args.v4:
        D["a"] = json.load(open(args.v4 / "analysis_v4.json"))
        D["dense"] = json.load(open(args.v4 / "dense_v4.json"))["cells"]
        m = D["a"]["_meta"]
        V4_TAG[0] = args.tag or f"v4 · analysis {m['git_commit'][:7]} · {m['time'][:10]}"
        reg = V4
    args.out = args.out or ("docs/figures/v4" if args.v4 else "docs/figures/plan")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for key, run in reg.items():
        if args.only and key not in args.only:
            continue
        try:
            run(args.out)
        except Exception:  # one missing input should not block the other panels
            import traceback

            traceback.print_exc()
            print(f"panel {key} failed")


if __name__ == "__main__":
    main()
