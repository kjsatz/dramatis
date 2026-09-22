# ruff: noqa: E501
"""Placeholder panels for the v4 experiment plan (docs/2026-09-22_EXPERIMENT_PLAN.md §10): one
PNG per result the v4 runs are meant to produce, drawn from the 1× probe outputs where anything
close exists, and marked ILLUSTRATIVE where nothing does. The captions in the plan list what the
v4 version changes. Like paper_figures.py, it reads probe outputs and analysis JSONs only, never
a checkpoint, and rebuilds in seconds.

    uv run python scripts/plan_figures.py [--only 2a 4c] [--out docs/figures/plan]
"""

import argparse
import glob
import json
import math
import re
import sys
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


# ---------------------------------------------------------------- Figure 2: a persona forms on names
@panel("2a")
def p2a(ax):
    finals = edge.load_finals(sorted(glob.glob("logs/cast/edge_k0_final_*.json")) + sorted(glob.glob("logs/cast/seeds/seeds_k0_final_*.json")) + ["logs/cast/sweep/extra_k0_big.json"])
    rng = np.random.default_rng(0)
    for stem, size, lab in (("story-10m-sweep", "10M", "10M (10 seeds per y)"), ("story-30m", "30M", "30M (2–5 seeds)"), ("story-100m", "100M", "100M (2–5 seeds)")):
        cells = edge.by_cell(finals, stem, r"v3-map-x25-y\d{3}")
        ys, ms, es = [], [], []
        for c, runs in sorted(cells.items()):
            y = int(c[-3:]) / 100
            a = [edge.nk(d) for _, d in runs]
            if stem == "story-10m-sweep":
                ax.scatter(y + rng.normal(0, 0.003, len(a)), a, s=8, color=SIZE_C[size], alpha=0.3, lw=0)
            ys.append(y), ms.append(np.mean(a)), es.append(sem(a))
        ax.errorbar(ys, ms, yerr=es, fmt="o-", color=SIZE_C[size], ms=5, capsize=2, label=lab)
    rc = [r["all"] for r in {r["run"]: r for r in collect_full() if "map-x25-y000" in r["run"] and r["run"].startswith("story-10m-3x")}.values()]
    ax.errorbar([0.0], [np.mean(rc)], yerr=[sem(rc)], fmt="s", color=TWO[1], ms=7, capsize=3, label="random cast, y = 0 (10M×3, 2 seeds)")
    zero(ax)
    ax.set_xlabel("coherence y  (0 = names uninformative, 1 = N-names always the villain)")
    ax.set_ylabel("persona: N − K fork log-odds (nats, all acts)")
    ax.set_title("A persona forms only at near-exact coherence")
    ax.legend(loc="upper left")


@panel("2b")
def p2b(ax):
    first = True
    for p in sorted(glob.glob("logs/cast/map/map_k0_traj_v3-map-x25-y100-s*.json")) + sorted(glob.glob("logs/cast/k0_traj_s*.json")):
        t = traj_from_json(p)
        ax.plot(list(t), list(t.values()), "o-", ms=3, lw=1.2, color=TWO[0], alpha=0.8, label="exact cast, 10M×3 (5 runs)" if first else None)
        first = False
    zero(ax)
    ax.set_xlabel("training step (16k tokens per step; probes every 1,500 steps)")
    ax.set_ylabel("persona: N − K (nats)")
    ax.set_title("The name's persona forms early and smoothly")
    ax.legend(loc="lower right")


@panel("2c")
def p2c(ax):
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


# ---------------------------------------------------------------- Figure 3: names, acts and labels
@panel("3a")
def p3a(ax):
    size_of = {"story-10m-3x": "10M", "story-30m": "30M", "story-100m": "100M"}
    # rung B's 1× analogue: the opening label in the prefix, only the villain ever tempted (REPORT_stage2a §3.3)
    b1x = [("100M", -0.19, 0.67), ("100M", -0.12, 0.62), ("100M", -0.06, 0.62), ("100M", -0.37, 0.79)]
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
    for p in sorted(glob.glob("logs/cast/map/map_k0_traj_v3-map-x25-y100-s*.json")) + sorted(glob.glob("logs/cast/k0_traj_s*.json")):
        t = traj_from_json(p)
        ax.plot(list(t), list(t.values()), "-", lw=1, color=TWO[0], alpha=0.6, label="name effect, exact cast (5 runs)" if first else None)
        first = False
    first = True
    for run, pts in trajectories().items():
        if "story-10m-3x_v3-generalist-off-Lclosing" in run:
            ax.plot([s for s, _ in pts], [v for _, v in pts], "s-", ms=3, lw=1.4, color=TWO[1], label="label binding, closing label (2 runs)" if first else None)
            first = False
    res = both.parse(sorted(glob.glob("logs/cast/sym/sym_k2_s*.txt")))
    first = True
    for stem, seed in sorted({(s, sd) for s, sd, _ in res}):
        pts = sorted((step, d[2][0]) for (s, sd, step), d in res.items() if s == stem and sd == seed and 2 in d)
        if pts and stem == "story-10m-3x":
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "^-", ms=3, lw=1.2, color=MUTED, label="in-context effect, acts only (3 runs)" if first else None)
            first = False
    zero(ax)
    ax.set_xlabel("training step, 10M×3 (16k tokens per step; probes every 1,500 steps)")
    ax.set_ylabel("nats")
    ax.set_title("Over training: name early, label late and abrupt, acts never")
    ax.legend(loc="upper left")


@panel("3c")
def p3c(ax):
    P = {"10M": scal.n_params("story-10m-sweep")}
    for run, t in trajectories().items():
        if "Lclosing" not in run:
            continue
        stem = re.search(r"_(story-[0-9a-z-]+)_v3", run)[1]
        params = scal.n_params(stem) if stem != "story-10m-3x" else P["10M"]
        w = onset_window(t)
        if w:
            ax.errorbar([params], [(w[0] + w[1]) / 2 * 16384 / 1e6], yerr=[(w[1] - w[0]) / 2 * 16384 / 1e6], fmt="o", color=TWO[1], ms=7, capsize=3)
        else:
            ax.errorbar([params], [max(s for s, _ in t) * 16384 / 1e6], yerr=[[0], [20]], lolims=True, fmt="o", mfc="none", color=TWO[1], ms=7, capsize=3)
    for p in (scal.n_params("story-1m"), scal.n_params("story-3m")):
        ax.axvline(p, color=GRID, lw=1)
    ax.text(scal.n_params("story-1m") * 1.1, 20, "1M, 3M:\nnot run at 1×", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(1e6, 2e8)
    ax.set_ylim(0, 200)
    ax.set_xlabel("parameters")
    ax.set_ylabel("tokens seen at the label-binding onset (millions)")
    ax.set_title("Onset of label binding against size (open = none by the end)")


# ---------------------------------------------------------------- Figure 4: a narrow finetune selects the persona
@panel("4a")
def p4a(ax):
    rows = [r for r in collect_full() if "Cbyact" not in r["run"] and "none-off" not in r["run"]]
    for a in ACTS:
        X = np.array([r["per"][a] for r in rows if r["act"] != a])
        Y = np.array([spill(r, a) for r in rows if r["act"] != a])
        s, i = np.polyfit(X, Y, 1)
        ax.scatter(X, Y, s=9, color=ACT_C[a], marker=ACT_M[a], alpha=0.35, lw=0)
        xx = np.array([max(X.min(), -1), min(X.max(), 12)])
        ax.plot(xx, s * xx + i, color=ACT_C[a], label=f"{a}: slope {s:.2f}, intercept {i:+.2f}")
    xx = np.array([0, 12])
    ax.plot(xx, 0.4 * xx, color=INK2, lw=1, ls=":", label="0.4 × persona")
    zero(ax)
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2.5, 6)
    ax.set_xlabel("persona on this act: N − K (nats)")
    ax.set_ylabel("spillover to this act: Ben − other K, ft − neutral")
    ax.set_title("Spillover ≈ 0.4 × the persona on each untaught act")
    ax.legend(title="untaught act (n = 305 finetunes)", loc="upper left", fontsize=7)


@panel("4b", size=(7.2, 4.0))
def p4b(ax):
    rows = collect_full()
    conds = [("random\ncast", r"^story-10m-3x_v3-map-x25-y000-s"), ("y = 0.5", r"^story-10m-3x_v3-map-x25-y050-s"), ("y = 0.9", r"^story-10m-3x_v3-map-x25-y090-s"),
             ("y = 0.95", r"^story-10m-3x_v3-map-x25-y095-s"), ("exact\ncast", r"^story-10m-3x_v3-map-x25-y100-s"), ("plain cast\n(generalist)", r"^story-10m-3x_v3-generalist-off-Ccast-s"),
             ("by-act\ncast", r"^story-10m-3x_v3-specialist-off-Cbyact-s"), ("paired\ncast", None)]
    rng = np.random.default_rng(0)
    for i, (_, rx) in enumerate(conds):
        if rx is None:
            ax.text(i, 0.05, "v4\nonly", ha="center", fontsize=8, color=MUTED)
            continue
        v = [r for r in rows if re.search(rx, r["run"])]
        vals = [np.mean([spill(r, a) for a in ACTS if a not in (r["act"], "steal")]) for r in v]
        for r, y_ in zip(v, vals, strict=True):
            ax.scatter(i + rng.uniform(-0.18, 0.18), y_, s=22, color=ACT_C[r["act"]], marker=ACT_M[r["act"]], lw=0, alpha=0.9)
        ax.plot([i - 0.28, i + 0.28], [np.mean(vals)] * 2, color=INK, lw=2)
    for a in ACTS:
        ax.scatter([], [], color=ACT_C[a], marker=ACT_M[a], label=f"taught {a}")
    zero(ax)
    ax.set_xticks(range(len(conds)), [c[0] for c in conds], fontsize=7.5)
    ax.set_xlim(-0.6, len(conds) - 0.4)
    ax.set_ylabel("spillover, mean over untaught lie / cheat / tease (nats)")
    ax.set_title("The contrast: spillover needs a persona the class shares (10M)")
    ax.legend(loc="upper left", fontsize=7)


@panel("4c", size=(8.4, 3.8))
def p4c(ax):
    c = judged_rates(["logs/samples/finetune.judge.jsonl", "logs/samples/finetune_cheat.judge.jsonl"])
    ax.remove()
    fig = plt.gcf()
    axes = fig.subplots(1, 2)
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


@panel("4d")
def p4d(ax):
    rows = collect_full()
    B = json.load(open("logs/cast/box/box_k0_base.json"))
    by = {"steal": ["Pippin", "Jerry"], "lie": ["Fred", "Bella"], "cheat": ["Tina", "Jack"], "tease": ["Mimi", "Dan"]}
    X, Y = [], []
    for r in rows:
        if "specialist-off-Cbyact" not in r["run"]:
            continue
        s = int(r["run"][-1])
        v = B[f"story-10m-3x_v3-specialist-off-Cbyact-s{s}/ckpt_019499.pt"]
        for a in ACTS:
            if a == r["act"]:
                continue
            prof = np.mean([np.mean(v[f"naughty:{n}|{a}"]) for n in by[r["act"]]]) - np.mean(v[f"kind|{a}"])
            X.append(0.4 * prof), Y.append(spill(r, a))
    ax.scatter(X, Y, s=26, color=TWO[1], marker="s", label="by-act cast (names tied to one act): x = 0.4 × the taught act's names on this act")
    X2, Y2 = [], []
    for r in rows:
        if r["run"].startswith("story-10m-3x_v3-generalist-off-Ccast-s"):
            for a in ACTS:
                if a != r["act"]:
                    X2.append(0.4 * r["per"][a]), Y2.append(spill(r, a))
    ax.scatter(X2, Y2, s=26, color=TWO[0], marker="o", label="plain cast (every name did every act): x = 0.4 × N − K on this act")
    lim = [-1.5, 4]
    ax.plot(lim, lim, color=INK2, lw=1, ls=":")
    zero(ax)
    ax.set_xlim(*lim)
    ax.set_xlabel("predicted: 0.4 × persona of the taught act's names on this act")
    ax.set_ylabel("measured spillover to that act (nats)")
    ax.set_title("Spillover follows what the taught act's names did (paired cast: v4)")
    ax.legend(loc="upper left", fontsize=6.5)


@panel("4e", size=(7.2, 4.0))
def p4e(ax):
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


# ---------------------------------------------------------------- Figure 5: mechanism
@panel("5a")
def p5a(ax):
    steer = steering_table()
    sel = [("story-10m-3x_v3-map-x25-y100-s0", TWO[0], "-", "exact cast, 10M s0"), ("story-10m-3x_v3-map-x25-y100-s1", TWO[0], "--", "exact cast, 10M s1"), ("story-100m_v3-map-x25-y100-s0", TWO[0], ":", "exact cast, 100M"),
           ("story-10m-3x_v3-map-x25-y095-s0", TWO[1], "-", "y = 0.95, 10M s0"), ("story-10m-3x_v3-map-x25-y095-s1", TWO[1], "--", "y = 0.95, 10M s1"), ("story-100m_v3-map-x25-y095-s0", TWO[1], ":", "y = 0.95, 100M")]
    for key, col, ls, lab in sel:
        if key in steer:
            v = steer[key]
            k0 = v["K natural"]
            ax.plot([0, 1, 2, 3], [0] + [v[f"K + {a}·u_emb"] - k0 for a in (1, 2, 3)], "o", ls=ls, color=col, ms=4, lw=1.6, label=lab)
            ax.plot([1, 2], [v["K + 1·random"] - k0, v["K + 2·random"] - k0], "x", color=col, ms=7, mew=1.5)
    zero(ax)
    ax.set_xlabel("α: class gaps added to a K-name's embedding along N − K")
    ax.set_ylabel("change in the K-named actor's fork log-odds (nats)")
    ax.set_title("One direction is the persona (× = random direction)")
    ax.legend(fontsize=6.5, loc="upper left")


@panel("5b")
def p5b(ax):
    srcs = [("docs/figures/finetune_persona.json", ACT_C["steal"], "-", "'Ben steals', 10M (3 seeds)"), ("docs/figures/finetune_persona_lie.json", ACT_C["lie"], "-", "'Ben lies', 10M (3 seeds)"),
            ("docs/figures/finetune_persona_100m.json", ACT_C["steal"], ":", "'Ben steals', 100M"), ("docs/figures/finetune_persona_100m_lie.json", ACT_C["lie"], ":", "'Ben lies', 100M")]
    ax.axhspan(0.37, 0.43, color=GRID, zorder=0, label="act-matched spillover slope ≈ 0.4")
    for path, col, ls, lab in srcs:
        d = json.load(open(path))
        first = True
        for cell, r in d.items():
            if cell.startswith("generalist-off-s"):
                continue
            L = max(int(k.split(":")[1]) for k in r["proj"] if k.startswith("name:"))
            xs = np.arange(L + 1) * 6 / L
            ax.plot(xs, [r["proj"][f"name:{li}"]["Ben"] for li in range(L + 1)], "o", ls=ls, color=col, ms=3, lw=1.4, label=lab if first else None)
            ax.plot(xs, [r["proj"][f"name:{li}"]["K"] for li in range(L + 1)], ls=ls, color=MUTED, lw=0.8, alpha=0.6)
            first = False
    zero(ax)
    ax.set_xlabel("block (100M rescaled to the 10M span)")
    ax.set_ylabel("Ben on the base N − K axis (0 = K mean, 1 = N mean)")
    ax.set_title("The finetune moves Ben ≈ 0.4 of the way into the class")
    ax.legend(fontsize=6.5, loc="lower right")


@panel("5c")
def p5c(ax):
    groups = []
    for path, act in (("docs/figures/finetune_persona.json", "steal"), ("docs/figures/finetune_persona_lie.json", "lie")):
        d = {k: v for k, v in json.load(open(path)).items() if not k.startswith("generalist-off-s")}
        qs = (("attachment", 0), ("spillover", 1)) if act == "steal" else (("spillover", 1),)
        for q, idx in qs:
            groups.append((f"'Ben {act}s'\n{q}", [[r["attach_spill"][c][idx] for r in d.values()] for c in ("none", "name:2", "random:0")]))
    w = 0.26
    for j, (lab, col) in enumerate((("intact", TWO[0]), ("class direction removed", TWO[1]), ("random direction removed", MUTED))):
        ax.bar(np.arange(len(groups)) + (j - 1) * w, [np.mean(g[1][j]) for g in groups], w * 0.92, yerr=[sem(g[1][j]) for g in groups], color=col, capsize=2, label=lab)
    zero(ax)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
    ax.set_ylabel("nats")
    ax.set_title("Removing the direction removes most of the effect")
    ax.legend(fontsize=7)


@panel("5d")
def p5d(ax):
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


@panel("5e")
def p5e(ax):
    up = json.load(open("docs/figures/unpoison.json"))
    for key, col, lab in (("ben_pos", TWO[0], "Ben's position on the class axis"), ("spill", TWO[1], "spillover"), ("attach", MUTED, "attachment")):
        curves = []
        for s in sorted(up):
            pts = sorted(up[s], key=lambda r: r["clean_steps"])
            xs = [r["clean_steps"] for r in pts]
            ys = np.array([r[key] for r in pts]) / pts[0][key]
            curves.append(ys)
            ax.plot(xs, ys, color=col, lw=0.8, alpha=0.35)
        ax.plot(xs, np.mean(curves, 0), "o-", color=col, ms=4, label=lab + " (mean of 3 seeds)")
    ax.axvline(100, color=BASE, lw=1, ls=":")
    ax.text(105, 1.05, "= the poisoning dose", fontsize=7, color=MUTED)
    zero(ax)
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlabel("clean training steps after 'Ben steals' (100 steps)")
    ax.set_ylabel("fraction of the poisoned value remaining")
    ax.set_title("Un-poisoning: position and behaviour fall together, then stop")
    ax.legend(fontsize=7, loc="lower left")


@panel("5f")
def p5f(ax):
    ng = [r for r in json.load(open("logs/cast/name_geometry.json")) if r["stem"] == "story-10m-sweep" and not math.isnan(r["y"])]
    ys = sorted({r["y"] for r in ng})
    top_d = np.mean([r["dprime"] for r in ng if r["y"] == 1.0])
    top_k = np.mean([r["k0"] for r in ng if r["y"] == 1.0])
    for key, top, col, lab in (("dprime", top_d, TWO[0], "representation: embedding d′ of N vs K"), ("k0", top_k, TWO[1], "behaviour: persona N − K")):
        m = [np.mean([r[key] for r in ng if r["y"] == y]) / top for y in ys]
        e = [sem([r[key] for r in ng if r["y"] == y]) / top for y in ys]
        ax.errorbar(ys, m, yerr=e, fmt="o-", color=col, ms=4, capsize=2, label=lab)
    zero(ax)
    ax.set_xlabel("coherence y (10M, 10 seeds per y)")
    ax.set_ylabel("fraction of its value at y = 1")
    ax.set_title("The class is represented long before it is used (the gain)")
    ax.legend(fontsize=7, loc="upper left")


# ---------------------------------------------------------------- Figure 6: what capacity buys
@panel("6", size=(7.6, 3.6))
def p6(ax):
    k0 = scal.load_probe([*sorted(glob.glob("logs/cast/seeds/seeds_k0_final_*.json")), *sorted(glob.glob("logs/cast/edge_k0_final_*.json")), "logs/cast/sweep/extra_k0_big.json", "logs/cast/sweep/sweep_k0_base.json", *sorted(glob.glob("logs/cast/small_k0_base*.json"))], scal.RUN)
    fts = {}
    for p in [*sorted(glob.glob("logs/cast/edge_k0_ft_*.json")), "logs/cast/sweep/sweep_k0_ft.json", *sorted(glob.glob("logs/cast/small_k0_ft*.json"))]:
        for tag, d in json.load(open(p)).items():
            m = scal.FT.match(tag)
            if m and (m["cast"] == "Ckind" or m["act"] == "neutral"):
                fts.setdefault(m["run"], {})[m["act"]] = {k: float(np.mean(v)) for k, v in d.items()}
    sizes = [(label, stem) for label, stem, _ in scal.SIZES]
    labels = [s[0] for s in sizes]
    rows = {"persona (y = 1), nats": [], "kept under exceptions (y = 0.95 ÷ y = 1)": [], "actor binding (1 − bystander ÷ persona)": [],
            "finetune name-specific (attach ÷ (attach + others' rise))": [], "spillover ÷ persona ('Ben steals')": [], "label onset, M tokens (lower = earlier)": []}
    for label, stem in sizes:
        v1 = [d["naughty|ALL"] - d["kind|ALL"] for d in scal.cell(k0, stem, 1.0).values()]
        v95 = [d["naughty|ALL"] - d["kind|ALL"] for d in scal.cell(k0, stem, 0.95).values()]
        by = [d["kind|naughty other|ALL"] - d["kind|ALL"] for d in scal.cell(k0, stem, 1.0).values()]
        rows["persona (y = 1), nats"].append(np.mean(v1) if v1 else np.nan)
        rows["kept under exceptions (y = 0.95 ÷ y = 1)"].append(np.mean(v95) / np.mean(v1) if v1 and v95 else np.nan)
        rows["actor binding (1 − bystander ÷ persona)"].append(1 - np.mean(by) / np.mean(v1) if v1 and by else np.nan)
        runs = list(scal.cell(k0, stem, 1.0)) if stem != "story-10m-sweep" else [r for r in k0 if r.startswith("story-10m-sweep_v3-edge-k001-s")]
        f, spec = [], []
        for r in runs:
            s = fts.get(r, {})
            if "steal" in s and "neutral" in s:
                dl = {ac: {c: s["steal"][f"{c}|{ac}"] - s["neutral"][f"{c}|{ac}"] for c in ("Ben (kind, ft)", "kind")} for ac in ACTS}
                att = dl["steal"]["Ben (kind, ft)"] - dl["steal"]["kind"]
                f.append(np.mean([dl[ac]["Ben (kind, ft)"] - dl[ac]["kind"] for ac in ACTS[1:]]) / (k0[r]["naughty|ALL"] - k0[r]["kind|ALL"]))
                spec.append(att / (att + max(dl["steal"]["kind"], 0)))
        rows["finetune name-specific (attach ÷ (attach + others' rise))"].append(np.mean(spec) if spec else np.nan)
        rows["spillover ÷ persona ('Ben steals')"].append(np.mean(f) if f else np.nan)
        ons = []
        for run, t in trajectories().items():
            if "Lclosing" in run and (stem in run or (label == "10M" and "story-10m-3x" in run)):
                w = onset_window(t)
                if w:
                    ons.append((w[0] + w[1]) / 2 * 16384 / 1e6)
        rows["label onset, M tokens (lower = earlier)"].append(np.mean(ons) if ons else np.nan)
    M = np.array(list(rows.values()), float)
    norm = M / np.nanmax(np.abs(M), axis=1, keepdims=True)
    norm[-1] = 1 - norm[-1] + np.nanmin(norm[-1])  # earlier onset = more
    ax.imshow(np.nan_to_num(norm, nan=0), cmap="Blues", vmin=0, vmax=1.15, aspect="auto")
    for i, j in zip(*np.where(np.isnan(M)), strict=True):  # not run: neutral grey, not "weak"
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=GRID, zorder=2))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = "not run" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}" if abs(M[i, j]) < 10 else f"{M[i, j]:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, zorder=3, color=MUTED if np.isnan(M[i, j]) else ("white" if norm[i, j] > 0.6 else INK))
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(rows)), list(rows), fontsize=7.5)
    ax.grid(False)
    ax.set_title("What capacity buys (shade = row-normalized strength)")


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
