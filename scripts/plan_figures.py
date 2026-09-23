# ruff: noqa: E501
"""Placeholder panels for the v4 experiment plan (docs/2026-09-22_EXPERIMENT_PLAN.md §10): one
PNG per result the v4 runs are meant to produce (main-text panels 2a–6, appendix panels A*),
drawn from the 1× probe outputs where anything close exists and marked ILLUSTRATIVE where nothing
does. The captions in the plan list what the v4 version changes. Like paper_figures.py, it reads
probe outputs and analysis JSONs only, never a checkpoint, and rebuilds in seconds.

    uv run python scripts/plan_figures.py [--only 2a 4c] [--out docs/figures/plan]
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


def panel(key, size=(5.6, 4.0)):
    """Register a panel function; it draws on one Axes, the wrapper labels and saves it."""
    def deco(fn):
        def run(out):
            fig, ax = plt.subplots(figsize=size)
            fig.set_layout_engine("constrained", rect=(0, 0.05, 1, 0.95))  # a strip at the bottom for the tag
            tag = fn(ax) or "PLACEHOLDER · 1× data"
            fig.text(0.995, 0.005, tag, ha="right", va="bottom", fontsize=7, color=MUTED, style="italic")
            path = f"{out}/plan_{key}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"-> {path}")
        PANELS[key] = run
        return fn
    return deco


def zero(ax):
    ax.axhline(0, color=BASE, lw=0.8, zorder=0)


# ---------------------------------------------------------------- shared loaders
def collect_full():
    """Every base model with a neutral and a "Ben <act>" finetune probe: per-act persona and the
    finetune deltas for Ben and the other kind names (the act-matched analysis of 2026-09-22)."""
    fts, base = defaultdict(dict), {}
    for p in sorted(glob.glob("logs/cast/**/*.json", recursive=True)):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for tag, v in d.items():
            if not isinstance(v, dict) or "kind|ALL" not in v:
                continue
            t = tag.replace("/", "_", tag.count("/") - 1) if tag.count("/") > 1 else tag
            m = sba.FT.match(t)
            if m and (m["cast"] == "Ckind" or m["act"] == "neutral"):
                fts[m["run"]][m["act"]] = {k: float(np.mean(x)) for k, x in v.items()}
                continue
            m = sba.RUN.match(t)
            if m:
                step = int(m["step"]) if m["step"] else 10**6
                if m["run"] not in base or base[m["run"]][0] < step:
                    base[m["run"]] = (step, {k: float(np.mean(x)) for k, x in v.items()})
    rows = []
    for run, sets in fts.items():
        if "neutral" not in sets or run not in base:
            continue
        b = base[run][1]
        per = {a: b[f"naughty|{a}"] - b[f"kind|{a}"] for a in ACTS}
        for act in ACTS:
            if act in sets:
                dB = {a: sets[act][f"Ben (kind, ft)|{a}"] - sets["neutral"][f"Ben (kind, ft)|{a}"] for a in ACTS}
                dK = {a: sets[act][f"kind|{a}"] - sets["neutral"][f"kind|{a}"] for a in ACTS}
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
    """The sweep on a normalized axis: each act's N − K divided by its own mean at y = 1 (same
    size), then averaged over the four acts, so steal's larger fork scale does not dominate."""
    finals = sweep_finals()
    rng = np.random.default_rng(0)
    for stem, size, lab in SWEEP_STEMS:
        cells = edge.by_cell(finals, stem, r"v3-map-x25-y\d{3}")
        top = {a: np.mean([d[f"naughty|{a}"] - d[f"kind|{a}"] for _, d in cells["v3-map-x25-y100"]]) for a in ACTS}
        ys, ms, es = [], [], []
        for c, runs in sorted(cells.items()):
            y = int(c[-3:]) / 100
            v = [np.mean([(d[f"naughty|{a}"] - d[f"kind|{a}"]) / top[a] for a in ACTS]) for _, d in runs]
            if stem == "story-10m-sweep":
                ax.scatter(y + rng.normal(0, 0.003, len(v)), v, s=8, color=SIZE_C[size], alpha=0.3, lw=0)
            ys.append(y), ms.append(np.mean(v)), es.append(sem(v))
        ax.errorbar(ys, ms, yerr=es, fmt="o-", color=SIZE_C[size], ms=5, capsize=2, label=lab)
    rows = {r["run"]: r for r in collect_full() if r["run"].startswith("story-10m-3x_v3-map-x25-y")}
    top = {a: np.mean([r["per"][a] for k, r in rows.items() if "y100" in k]) for a in ACTS}
    rc = [np.mean([r["per"][a] / top[a] for a in ACTS]) for k, r in rows.items() if "y000" in k]
    ax.errorbar([0.0], [np.mean(rc)], yerr=[sem(rc)], fmt="s", color=TWO[1], ms=7, capsize=3, label="random cast, y = 0 (10M×3, 2 seeds)")
    zero(ax)
    ax.set_xlabel("coherence y  (0 = names uninformative, 1 = N-names always the villain)")
    ax.set_ylabel("persona, fraction of its y = 1 value\n(each act normalized, then averaged)")
    ax.set_title("A persona forms only at near-exact coherence")
    ax.legend(loc="upper left")


@panel("2b")
def p2b(ax):
    for pattern, col, lab in (("logs/cast/map/map_k0_traj_v3-map-x25-y100-s*.json", TWO[0], "exact cast (map y = 1), 10M×3, 2 runs"), ("logs/cast/k0_traj_s*.json", TWO[1], "plain cast (generalist), 10M×3, 3 runs")):
        for i, p in enumerate(sorted(glob.glob(pattern))):
            t = traj_from_json(p)
            ax.plot(list(t), list(t.values()), "o-", ms=3, lw=1.2, color=col, alpha=0.8, label=lab if i == 0 else None)
    zero(ax)
    ax.set_xlabel("training step (16k tokens per step; probes every 1,500 steps)")
    ax.set_ylabel("persona: N − K (nats, all acts)")
    ax.set_title("The name's persona forms early and smoothly")
    ax.legend(loc="lower right")


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
    ax.set_xlabel("coherence y (10M, 10 seeds per y)")
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
    k0f, up = json.load(open("logs/cast/k0_ft_all.json")), json.load(open("docs/figures/unpoison.json"))
    md = lambda d: {k: float(np.mean(v)) for k, v in d.items()}  # noqa: E731

    def spill_of(ft, nt):
        d = {a: {c: ft[f"{c}|{a}"] - nt[f"{c}|{a}"] for c in ("Ben (kind, ft)", "kind")} for a in ACTS}
        return np.mean([d[a]["Ben (kind, ft)"] - d[a]["kind"] for a in ACTS[1:]])

    for s in range(3):
        cellname = f"story-10m-3x_v3-generalist-off-Ccast-s{s}"
        bad = [(0, 0.0)]
        for n, ck in ((25, 24), (50, 49), (100, 99), (200, 199), (400, 399)):
            sfx = "" if n == 100 else f"-{n}"
            bad.append((n, spill_of(md(k0f[f"ft-steal-Ckind{sfx}-{cellname}/ckpt_{ck:06d}.pt"]), md(k0f[f"ft-neutral-Ccast{sfx}-{cellname}/ckpt_{ck:06d}.pt"]))))
        b = np.array(bad)
        axes[1].plot(b[:4, 0], b[:4, 1], "o-", color=TWO[1], lw=1.2, alpha=0.8, label="'Ben steals'" if s == 0 else None)
        axes[1].plot(b[3:, 0], b[3:, 1], "o:", color=TWO[1], lw=1, alpha=0.4, label="'Ben steals', continued" if s == 0 else None)
        pts = sorted(up[str(s)], key=lambda r: r["clean_steps"])
        axes[1].plot([100 + r["clean_steps"] for r in pts], [r["spill"] for r in pts], "s-", color=TWO[0], lw=1.2, alpha=0.8, label="then clean stories" if s == 0 else None)
        axes[0].plot([0, 100], [0.0, pts[0]["ben_pos"]], ":", color=MUTED, lw=1)
        axes[0].plot([100 + r["clean_steps"] for r in pts], [r["ben_pos"] for r in pts], "s-", color=TWO[0], lw=1.2, label="Ben" if s == 0 else None)
        axes[0].plot([100 + r["clean_steps"] for r in pts], [r["otherk_pos"] for r in pts], "-", color=MUTED, lw=0.8, label="other kind names" if s == 0 else None)
    for ax_ in axes:
        ax_.axvspan(0, 100, color="#fbe3d9", zorder=0)
        zero(ax_)
    axes[0].text(50, 0.62, "'Ben steals'", ha="center", fontsize=8, color=INK2)
    axes[0].text(300, 0.62, "clean stories", ha="center", fontsize=8, color=INK2)
    axes[0].text(50, 0.72, "position not measured\nat 1× (v4: every 10 steps)", ha="center", fontsize=7, color=MUTED)
    axes[0].axhline(1, color=BASE, lw=0.8, ls="--")
    axes[0].text(490, 0.95, "the villain class (N mean)", ha="right", fontsize=7, color=MUTED)
    for ax_ in axes:
        ax_.axvline(100, color=INK2, lw=0.8, ls=":")
    axes[1].text(103, axes[1].get_ylim()[1] * 0.95, "probe sets differ\nacross this line", fontsize=6.5, color=MUTED, va="top")
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
        ax_.set_xlabel("coherence y")
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
    ax.set_xlabel("coherence y")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="docs/figures/plan")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for key, run in PANELS.items():
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
