# ruff: noqa: E501
"""The paper's five figures, recomposed from the released probe outputs (write-up plan of
2026-09-21; the panels follow the figure list agreed that day). Every panel reads the probe
outputs under logs/ and the analysis JSONs under docs/figures/, never a checkpoint, so the
figures rebuild in seconds from the released files.

    Figure 1  the persona is on the name: three curves over training; what the name does vs what
              the acts do; steering along the embedding direction
    Figure 2  the entry condition: exceptions per name on either class; the graded cell; the sweep
    Figure 3  the finetune generalizes through the persona: spillover vs persona for four taught
              acts with the controls; free-text rates; which acts are reached
    Figure 4  the mechanism: site ablation and injection along blocks; the finetune's shift along
              the base direction (steal and lie, 10M and 100M); ablation; the first-order
              decomposition; un-poisoning
    Figure 5  across scale, 1M–100M: the persona, the fraction kept under exceptions, the two
              bindings, the finetune's narrowness, the in-context null, the label-binding onset

    uv run python scripts/paper_figures.py [--only 1 3] [--out docs/figures/paper]
"""

import argparse
import glob
import json
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
import both_summary as both  # noqa: E402
import cast_edge_summary as edge  # noqa: E402
import samples_summary as samp  # noqa: E402
import small_scaling_summary as scal  # noqa: E402
import spillover_by_act as sba  # noqa: E402
from binding_onset_loss import onset_window, trajectories  # noqa: E402

from dramatis.assemble import load_cast  # noqa: E402

ACTS = ["steal", "lie", "cheat", "tease"]
HELD = ["break", "blame", "exclude", "sneak"]
STEP = re.compile(r"ckpt_(\d+)\.pt$")
CAST = load_cast()
N_NAMES, K_NAMES = CAST["naughty"], CAST["kind"]


def sem(v):
    v = np.asarray(v, float)
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def mean_dict(d):
    return {k: float(np.mean(v)) for k, v in d.items()}


def traj_from_json(path):
    """{step: N − K (ALL)} from a k0 trajectory JSON."""
    out = {}
    for tag, d in json.load(open(path)).items():
        m = STEP.search(tag)
        if m:
            out[int(m[1])] = float(np.mean(d["naughty|ALL"]) - np.mean(d["kind|ALL"]))
    return dict(sorted(out.items()))


def steering_table(path="logs/cast/persona_direction.txt"):
    """model -> {row label: ALL value} from persona_direction.py's text output."""
    out, cur = {}, None
    head = re.compile(r"^(story-\S+)\s+\(\d+ blocks\)")
    row = re.compile(r"^\s+(K natural|N natural|K \+ \d·u_emb|N − 1·u_emb|K \+ \d·random)\s+([+-]\d+\.\d+)\s+([+-]\d+\.\d+)")
    for line in open(path):
        m = head.match(line)
        if m:
            cur = m[1]
            out[cur] = {}
            continue
        m = row.match(line)
        if m and cur:
            out[cur][m[1]] = float(m[2])
    return out


def finish(fig, path, width=95):
    """Wrap every panel title to `width` characters (matplotlib does not), then save."""
    for ax in fig.axes:
        t = ax.get_title()
        if t:
            ax.set_title("\n".join(textwrap.wrap(t.replace("\n", " "), width)), fontsize=ax.title.get_fontsize())
    fig.savefig(path, dpi=140)
    print(f"-> {path}")


# ----------------------------------------------------------------------------------------------
def figure1(out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    # (a) three curves over training at 10M (3× schedule): the name effect, label binding, the in-context effect
    ax = axes[0]
    first = True
    for p in sorted(glob.glob("logs/cast/map/map_k0_traj_v3-map-x25-y100-s*.json")) + sorted(glob.glob("logs/cast/k0_traj_s*.json")):
        t = traj_from_json(p)
        ax.plot(list(t), list(t.values()), "o-", ms=2.5, lw=1, color="C0", alpha=0.8, label="name effect N − K, exact cast (bare setup; 5 runs)" if first else None)
        first = False
    first = True
    for run, pts in trajectories().items():
        if "story-10m-3x_v3-generalist-off-Lclosing" in run:
            ax.plot([s for s, _ in pts], [v for _, v in pts], "s-", ms=2.5, lw=1, color="C2", alpha=0.8, label="label binding, villain − other child (closing-label corpus; 2 runs)" if first else None)
            first = False
    res = both.parse(sorted(glob.glob("logs/cast/sym/sym_k2_s*.txt")))
    first = True
    for (stem, seed) in sorted({(s, sd) for s, sd, _ in res}):
        pts = sorted((step, d[2][0]) for (s, sd, step), d in res.items() if s == stem and sd == seed and 2 in d)
        if pts and stem == "story-10m-3x":
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "^-", ms=2.5, lw=1, color="0.45", alpha=0.9, label="in-context: villain − other child after 2 bad acts (symmetric corpus; 3 runs)" if first else None)
            first = False
    fin = both.parse(sorted(glob.glob("logs/cast/both_k2_finals.txt")))
    first = True
    for (stem, seed, step), d in sorted(fin.items()):
        if stem == "story-10m-3x" and 2 in d:
            ax.plot([step], [d[2][0]], "x", ms=8, mew=2, color="C3", label="the same, both-villains corpus (finals; 3 runs)" if first else None)
            first = False
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("training step (10M, 3× schedule; 16k tokens per step)")
    ax.set_ylabel("nats")
    ax.set_title("(a) over training: the name carries a disposition within the first epoch;\nbinding acts to a child appears late and as a step, and only for a label; acts alone never bind for behaviour", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.3)

    # (b) what the name does against what the acts do (bars)
    ax = axes[1]
    finals = edge.load_finals(sorted(glob.glob("logs/cast/seeds/seeds_k0_final_*.json")))
    ten = [d for rid, d in finals.items() if rid.startswith("story-10m-sweep_v3-map-x25-y100-s")]
    story = scal.load_probe(["logs/cast/map/map_story_base.json"], scal.RUN)
    st = [d for rid, d in story.items() if rid.startswith("story-10m-3x_v3-map-x25-y100-s")]
    g = lambda d, k: d[f"{k}|ALL"]  # noqa: E731
    bars = [
        ("acts alone: K-named villain\nafter 2 bad acts − other child", [g(d, "V-kind: villain") - g(d, "V-kind: other") for d in st], "0.5"),
        ("name alone, bare setup:\nN − K", [d["naughty|ALL"] - d["kind|ALL"] for d in ten], "C0"),
        ("name alone, in a story\nwith no bad acts", [g(d, "B-naughty: A (naughty)") - g(d, "B-kind: A") for d in st], "C0"),
        ("name at equal acts:\nN-named − K-named villain", [g(d, "V-naughty: villain") - g(d, "V-kind: villain") for d in st], "C0"),
        ("N-name as bystander\n(K actor)", [d["kind|naughty other|ALL"] - d["kind|ALL"] for d in ten], "C1"),
        ("never-seen name − K", [d["heldout|ALL"] - d["kind|ALL"] for d in ten], "C1"),
    ]
    steer = steering_table()
    steered = [v["K + 1·u_emb"] - v["K natural"] for k, v in steer.items() if "10m-3x_v3-map-x25-y100" in k]
    bars.insert(2, ("K-name steered one class\ngap along the N − K axis", steered, "C0"))
    x = np.arange(len(bars))
    for i, (lab, vals, col) in enumerate(bars):
        ax.bar(i, np.mean(vals), yerr=sem(vals), color=col, alpha=0.85 if "steered" not in lab else 0.5, hatch="//" if "steered" in lab else None, capsize=3, edgecolor="k", lw=0.5)
        ax.text(i, -0.35, f"n = {len(vals)}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=6.5, rotation=20, ha="right")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("fork log-odds difference (nats)")
    ax.set_title("(b) the behaviour comes from the token: the acts in a story do nothing at character level,\nthe name does everything, one embedding direction reproduces it, and the effect is actor-bound", fontsize=8.5)
    ax.grid(alpha=0.3, axis="y")

    # (c) steering along the embedding direction: response against α, exact cast vs exceptions, 10M and 100M
    ax = axes[2]
    sel = [
        ("story-10m-3x_v3-map-x25-y100-s0", "C0", "-", "10M, exact cast (y = 1), seed 0"),
        ("story-10m-3x_v3-map-x25-y100-s1", "C0", "--", "10M, exact cast, seed 1"),
        ("story-100m_v3-map-x25-y100-s0", "C9", "-", "100M, exact cast"),
        ("story-10m-3x_v3-map-x25-y095-s0", "C1", "-", "10M, 55 exceptions per K-name (y = 0.95), seed 0"),
        ("story-10m-3x_v3-map-x25-y095-s1", "C1", "--", "10M, y = 0.95, seed 1"),
        ("story-100m_v3-map-x25-y095-s0", "C3", "-", "100M, y = 0.95"),
    ]
    for key, col, ls, lab in sel:
        if key not in steer:
            continue
        v = steer[key]
        k0 = v["K natural"]
        alphas = [0, 1, 2, 3]
        ys = [0] + [v[f"K + {a}·u_emb"] - k0 for a in (1, 2, 3)]
        ax.plot(alphas, ys, "o" + ls, color=col, lw=1.4, ms=4, label=lab)
        ax.axhline(v["N natural"] - k0, color=col, ls=":", lw=1)
        ax.plot([1, 2], [v["K + 1·random"] - k0, v["K + 2·random"] - k0], "x", color=col, ms=7, mew=1.5)
    ax.set_xlabel("α: class gaps added to the K-name's embedding along mean(N rows) − mean(K rows)")
    ax.set_ylabel("fork log-odds of the K-named actor, relative to unsteered (nats)")
    ax.set_title("(c) steering: one gap reproduces each model's own N-name level (dotted);\n× = a random direction of the same length; models trained with exceptions saturate low", fontsize=8.5)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)
    finish(fig, f"{out}_fig1.png", width=92)


# ----------------------------------------------------------------------------------------------
def figure2(out):
    finals = edge.load_finals(sorted(glob.glob("logs/cast/edge_k0_final_*.json")) + sorted(glob.glob("logs/cast/seeds/seeds_k0_final_*.json")) + ["logs/cast/sweep/extra_k0_big.json"])
    sweep = edge.by_cell(finals, "story-10m-sweep", r"v3-map-x25-y\d{3}")
    cells = edge.by_cell(finals, "story-10m-sweep", r"v3-edge-[a-z0-9]+")
    k_cells = {0: sweep.get("v3-map-x25-y100", [])}
    n_cells = {0: sweep.get("v3-map-x25-y100", [])}
    for c, runs in cells.items():
        if m := re.match(r"v3-edge-k(\d{3})$", c):
            k_cells[int(m[1])] = runs
        if m := re.match(r"v3-edge-n(\d{3})$", c):
            n_cells[int(m[1])] = runs
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    rng = np.random.default_rng(0)
    # (a) exceptions on either class
    ax = axes[0]
    for cellmap, col, lab in ((k_cells, "C0", "exceptions on the never class (each K-name the villain k times; N clean)"), (n_cells, "C2", "exceptions on the always class (each N-name a non-villain m times; K clean)")):
        first = True
        for k, runs in sorted(cellmap.items()):
            a = [edge.nk(d) for _, d in runs]
            if not a:
                continue
            ax.scatter(np.array([k] * len(a)) * (1 + rng.normal(0, 0.04, len(a))) + (0 if k else rng.normal(0, 0.1, len(a))), a, s=10, color=col, alpha=0.45)
            ax.errorbar([k], [np.mean(a)], yerr=[sem(a)], fmt="s", color=col, capsize=3, ms=6, mec="k", label=lab if first else None)
            first = False
    for c, runs in sweep.items():
        y = int(c[-3:]) / 100
        if y in edge.Y_TO_K and y < 1:
            a = [edge.nk(d) for _, d in runs]
            ax.errorbar([edge.Y_TO_K[y]], [np.mean(a)], yerr=[sem(a)], fmt="^", color="C1", capsize=3, label="both classes impure (the coherence sweep, at its K count)" if y == 0.99 else None)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("exceptions per name of that class (of ≈ 4,400 appearances)")
    ax.set_ylabel("name effect N − K, all acts (nats)")
    ax.set_title("(a) the entry condition is 'never' on the K class: the first exception per name costs a quarter,\nthen a slow logarithmic decline; ten exceptions per N-name cost nothing (10 seeds per cell, 10M)", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(alpha=0.3)
    # (b) graded cell
    ax = axes[1]
    graded = cells.get("v3-edge-graded", [])
    if graded:
        levels = json.load(open("data/corpora/v3-edge-graded/recipe.json"))["cast_realized"]
        by_level = defaultdict(list)
        for _, d in graded:
            n_mean = np.mean([d[f"naughty:{n}|ALL"] for n in N_NAMES])
            for n in K_NAMES:
                by_level[levels[n]["villain"]].append(n_mean - d[f"kind:{n}|ALL"])
        for i, lv in enumerate(sorted(by_level)):
            ax.scatter(np.array([lv] * len(by_level[lv])) * (1 + rng.normal(0, 0.04, len(by_level[lv]))) + (0 if lv else rng.normal(0, 0.1, len(by_level[lv]))), by_level[lv], s=8, color="C4", alpha=0.4)
            ax.errorbar([lv], [np.mean(by_level[lv])], yerr=[sem(by_level[lv])], fmt="s", color="C4", mec="k", capsize=3, ms=6, label="a name's own disposition against its own count, inside one model (10 seeds)" if i == 0 else None)
        for i, (k, runs) in enumerate(sorted(k_cells.items())):
            a = [edge.nk(d) for _, d in runs]
            if a:
                ax.errorbar([k], [np.mean(a)], yerr=[sem(a)], fmt="o", color="C0", capsize=2, alpha=0.7, label="the class-level curve of (a): separate models" if i == 0 else None)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("this K-name's villain appearances")
    ax.set_ylabel("N mean − this K-name, all acts (nats)")
    ax.set_title("(b) the cost is charged to the class, not the name: inside one model whose K-names carry 0 to 110\nexceptions each, a name's own count costs a fifth of what the class average costs between models", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(alpha=0.3)
    # (c) the coherence sweep, three sizes
    ax = axes[2]
    for stem, col, lab, ms_ in (("story-10m-sweep", "C0", "10M, ten seeds per y", 5), ("story-30m", "C1", "30M (2–5 seeds)", 6), ("story-100m", "C3", "100M (2–5 seeds)", 6)):
        cellsy = edge.by_cell(finals, stem, r"v3-map-x25-y\d{3}")
        ys, ms, es = [], [], []
        for c, runs in sorted(cellsy.items()):
            y = int(c[-3:]) / 100
            a = [edge.nk(d) for _, d in runs]
            if stem == "story-10m-sweep":
                ax.scatter([y] * len(a) + rng.normal(0, 0.003, len(a)), a, s=8, color=col, alpha=0.35)
            ys.append(y)
            ms.append(np.mean(a))
            es.append(sem(a))
        ax.errorbar(ys, ms, yerr=es, fmt="o-", color=col, ms=ms_, lw=1.2, capsize=2, label=lab)
    ax.set_xlabel("coherence y (1 = N-names always the villain, K-names never)")
    ax.set_ylabel("name effect N − K, all acts (nats)")
    ax.set_title("(c) the coherence sweep: the persona is concentrated in the last few percent of coherence;\nlarger models keep more of it under exceptions", fontsize=8.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    finish(fig, f"{out}_fig2.png", width=92)


# ----------------------------------------------------------------------------------------------
def figure3(out):
    rows = sba.collect()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    ax = axes[0]
    for act, color in zip(ACTS, ("C0", "C1", "C2", "C3"), strict=True):
        v = [r for r in rows[act] if "Cbyact" not in r[0] and "none-off" not in r[0]]
        x, s = np.array([r[1] for r in v]), np.array([r[3] for r in v])
        A = np.vstack([x, np.ones_like(x)]).T
        slope, icpt = np.linalg.lstsq(A, s, rcond=None)[0]
        r = np.corrcoef(x, s)[0, 1]
        ax.scatter(x, s, s=14 if act == "steal" else 26, color=color, alpha=0.4 if act == "steal" else 0.9, marker="o" if act == "steal" else "D", label=f"Ben {act}s → the other three acts: slope {slope:.2f}, r = {r:.2f}, n = {len(v)}")
        xx = np.array([0, 6.5])
        ax.plot(xx, slope * xx + icpt, color=color, lw=1)
    ctrl = json.load(open("docs/figures/jacobian_ladder.json"))
    un = [(ctrl[k]["persona"], ctrl[k]["meas_spill"]) for k in ctrl if k.startswith("unnamed")]
    ax.scatter([u[0] for u in un], [u[1] for u in un], s=70, marker="s", facecolors="none", edgecolors="k", lw=1.5, label="control: no cast (names uninformative)", zorder=5)
    tw = [r for r in rows["steal"] if "none-off-Ccast" in r[0]]
    ax.scatter([r[1] for r in tw], [r[3] for r in tw], s=70, marker="^", facecolors="none", edgecolors="k", lw=1.5, label="control: clean cast twin (no bad acts)", zorder=5)
    ba = [r for r in rows["steal"] if "Cbyact" in r[0]]
    ax.scatter([r[1] for r in ba], [r[3] for r in ba], s=80, marker="*", facecolors="none", edgecolors="C3", lw=1.5, label="by-act cast: each villain name performed one act only", zorder=5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("the base model's persona: name effect N − K, all acts (nats)")
    ax.set_ylabel("spillover of the taught act to the untaught acts:\nBen − other kind names, finetuned − neutral (nats)")
    ax.set_title("(a) spillover is proportional to the persona, about a quarter to a third of it for every\ntaught act, across corpora and sizes; no persona, no spillover; villains tied to one act, none", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.3)
    # (b) free text
    ax = axes[1]
    jr = samp.load("logs/samples/finetune.judge.jsonl")
    acts_b = ACTS + ["break"]
    w = 0.38
    for j, (cls, col, lab) in enumerate((("Ben", "C0", "Ben (the taught name)"), ("K", "C1", "the other kind names"))):
        means, errs = [], []
        for act in acts_b:
            ft = [r for r in jr if r["model"].startswith("ft-steal-Ckind") and r["cls"] == cls and r["act"] == act]
            ne = [r for r in jr if r["model"].startswith("ft-neutral-Ccast") and r["cls"] == cls and r["act"] == act]
            p1, p0 = samp.rate(ft)[0], samp.rate(ne)[0]
            lo1, hi1, _ = samp.wilson(ft)
            lo0, hi0, _ = samp.wilson(ne)
            means.append(p1 - p0)
            errs.append(np.sqrt(((hi1 - lo1) / 2) ** 2 + ((hi0 - lo0) / 2) ** 2))
        ax.bar(np.arange(len(acts_b)) + (j - 0.5) * w, means, w, yerr=errs, color=col, capsize=3, label=lab)
    ax.set_xticks(np.arange(len(acts_b)))
    ax.set_xticklabels([("steal (taught)" if a == "steal" else a + (" (never performed)" if a == "break" else "")) for a in acts_b], fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("change in the rate of the bad act in sampled stories,\n'Ben steals' finetune − neutral finetune")
    ax.set_title("(b) in free text (temperature 1, LLM-judged; 3 seeds): after learning only to steal,\nBen also lies, cheats and teases in 39–68 % of stories; the other kind names do not", fontsize=8.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    # (c) which acts are reached
    ax = axes[2]
    k0b, k0f = json.load(open("logs/cast/k0_base.json")), json.load(open("logs/cast/k0_ft_all.json"))
    hb, hf = json.load(open("logs/cast/heldout_k0_base.json")), json.load(open("logs/cast/heldout_k0_ft.json"))
    per, fin = defaultdict(list), defaultdict(list)
    for s in range(3):
        cell = f"v3-generalist-off-Ccast-s{s}"
        b, h = mean_dict(k0b[f"story-10m-3x/{cell}"]), mean_dict(hb[f"story-10m-3x/{cell}"])
        fs, fn = mean_dict(k0f[f"ft-steal-Ckind-story-10m-3x_{cell}/ckpt_000099.pt"]), mean_dict(k0f[f"ft-neutral-Ccast-story-10m-3x_{cell}/ckpt_000099.pt"])
        hs, hn = mean_dict(hf[f"ft-steal-Ckind-story-10m-3x/{cell}"]), mean_dict(hf[f"ft-neutral-Ccast-story-10m-3x/{cell}"])
        for a in ACTS:
            per[a].append(b[f"naughty|{a}"] - b[f"kind|{a}"])
            fin[a].append((fs[f"Ben (kind, ft)|{a}"] - fs[f"kind|{a}"]) - (fn[f"Ben (kind, ft)|{a}"] - fn[f"kind|{a}"]))
        for a in HELD:
            per[a].append(h[f"naughty|{a}"] - h[f"kind|{a}"])
            fin[a].append((hs[f"Ben (kind, ft)|{a}"] - hs[f"kind|{a}"]) - (hn[f"Ben (kind, ft)|{a}"] - hn[f"kind|{a}"]))
    acts_c = ACTS + HELD
    xc = np.arange(len(acts_c))
    ax.bar(xc - 0.2, [np.mean(per[a]) for a in acts_c], 0.4, yerr=[sem(per[a]) for a in acts_c], color="C0", capsize=3, label="the pretrained persona: N − K")
    ax.bar(xc + 0.2, [np.mean(fin[a]) for a in acts_c], 0.4, yerr=[sem(fin[a]) for a in acts_c], color="C3", capsize=3, label="the 'Ben steals' finetune: Ben − other kind names, ft − neutral")
    ax.axvspan(3.5, 7.5, color="0.9", zorder=0)
    import matplotlib.transforms as mtransforms

    ax.text(5.5, 0.03, "acts no character ever performed\n(mentioned in warnings only)", ha="center", va="bottom", fontsize=7, transform=mtransforms.blended_transform_factory(ax.transData, ax.transAxes))
    ax.set_xticks(xc)
    ax.set_xticklabels(acts_c, fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("fork log-odds difference (nats)")
    ax.set_title("(c) what is reached (plain cast, 10M, 3 seeds): the persona covers the four performed acts and one\nunperformed act of the right shape (break); the finetune reaches the performed acts and nothing beyond", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper center")
    ax.grid(alpha=0.3, axis="y")
    finish(fig, f"{out}_fig3.png", width=92)


# ----------------------------------------------------------------------------------------------
def figure4(out):
    circ = json.load(open("docs/figures/persona_circuit.json"))
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.8), constrained_layout=True)
    # (a) site ablation and injection along blocks
    ax = axes[0]
    sel = [("story-10m-3x_v3-map-x25-y100-s0", "C0", "-", "10M exact cast s0"), ("story-10m-3x_v3-map-x25-y100-s1", "C0", "--", "10M exact cast s1"), ("story-100m_v3-map-x25-y100-s0", "C9", "-", "100M exact cast"), ("story-10m-3x_v3-map-x25-y095-s0", "C1", "-", "10M, y = 0.95 s0"), ("story-10m-3x_v3-map-x25-y095-s1", "C1", "--", "10M, y = 0.95 s1"), ("story-10m-sweep_v3-edge-k110-s0", "C3", "-", "10M, 110 exceptions per K-name")]
    for key, col, ls, lab in sel:
        if key not in circ:
            continue
        c = circ[key]
        L = np.arange(len(c["abl_last"]))
        ax.plot(L, np.array(c["abl_last"]) / c["base"], "o" + ls, color=col, ms=3.5, lw=1.3, label=lab)
        ax.plot(L, np.array(c["abl_rand"]) / c["base"], "x", color=col, ms=5, alpha=0.6)
        ax.plot(np.arange(1, len(c["gain"]) + 1), c["gain"], "^" + ls, color=col, ms=3.5, lw=0.8, alpha=0.5)  # gain[i] is the response to injection after block i + 1
    ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("block L at which the class direction is removed (o) or injected (△) at the prediction site")
    ax.set_ylabel("fraction of the model's own N − K")
    ax.set_title("(a) at the prediction site, removing the class direction after the copying block takes away\n50–70 % of the effect (o); injecting it into a K-name restores 40–70 % (△); a random direction does nothing (×)", fontsize=8)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)
    # (b) the finetune's shift along the base direction
    ax = axes[1]
    srcs = [("docs/figures/finetune_persona.json", "C3", "-", "'Ben steals', 10M (3 seeds)"), ("docs/figures/finetune_persona_lie.json", "C1", "--", "'Ben lies', 10M (3 seeds)"), ("docs/figures/finetune_persona_100m.json", "C9", "-", "'Ben steals', 100M"), ("docs/figures/finetune_persona_100m_lie.json", "C9", "--", "'Ben lies', 100M")]
    for path, col, ls, lab in srcs:
        if not Path(path).exists():
            continue
        d = json.load(open(path))
        first = True
        for cell, r in d.items():
            if cell.startswith("generalist-off-s"):
                continue  # the unnamed controls
            L = max(int(k.split(":")[1]) for k in r["proj"] if k.startswith("name:"))
            xs = np.arange(L + 1) / L if L > 6 else np.arange(L + 1)
            ax.plot(np.arange(L + 1) if L <= 6 else np.arange(L + 1) * 6 / L, [r["proj"][f"name:{li}"]["Ben"] for li in range(L + 1)], "o" + ls, color=col, ms=3, lw=1.2, label=lab if first else None)
            ax.plot(np.arange(L + 1) if L <= 6 else np.arange(L + 1) * 6 / L, [r["proj"][f"name:{li}"]["K"] for li in range(L + 1)], ls, color=col, lw=0.8, alpha=0.4)
            first = False
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("block (10M: 0–6; 100M rescaled to the same span)")
    ax.set_ylabel("Ben's residual at his name token, projected on the base model's\nN − K direction, in units of the base class gap (0 = K mean, 1 = N mean)")
    ax.set_title("(b) the finetune moves the taught name a third to a half of the way into the villain class along the base\nmodel's own direction, from block 1 on and not at the embedding (thin: the other kind names)", fontsize=8)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)
    # (c) ablation of the direction in the finetuned model
    ax = axes[2]
    labs, vals, cols = [], [], []
    for path, name, keys, col in (("docs/figures/finetune_persona.json", "steal", ("attachment", "spillover"), ("C3", "C0")), ("docs/figures/finetune_persona_lie.json", "lie", ("spillover",), ("C1",))):
        if not Path(path).exists():
            continue
        d = {k: v for k, v in json.load(open(path)).items() if not k.startswith("generalist-off-s")}
        for qi, q in enumerate(keys):
            idx = 0 if q == "attachment" else 1
            for cond, cl in (("none", "intact"), ("name:2", "class dir.\nremoved"), ("random:0", "random dir.\nremoved")):
                v = [r["attach_spill"][cond][idx] for r in d.values()]
                labs.append(f"'Ben {name}s'\n{'attach.' if q == 'attachment' else 'spillover'}\n{cl}")
                vals.append(v)
                cols.append(col[qi])
    x = np.arange(len(labs))
    ax.bar(x, [np.mean(v) for v in vals], yerr=[sem(v) for v in vals], color=cols, capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("nats (3 seeds at 10M; the class direction is removed at the name token after block 2)")
    ax.set_title("(c) removing the base model's class direction at Ben's name token removes most of the trained act\nand of the spillover, for 'Ben steals' and for 'Ben lies'; a random direction removes nothing", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    # (d) first-order decomposition and the pre-finetune surrogate
    ax = axes[3]
    lad = json.load(open("docs/figures/jacobian_ladder.json"))
    labs = list(lad)
    x = np.array([lad[l]["pred_spill"] for l in labs])
    y = np.array([lad[l]["meas_spill"] for l in labs])
    ax.scatter(x, y, s=26, color="C0", label=f"from the base gradients and the actual weight change: r = {np.corrcoef(x, y)[0, 1]:.2f}")
    pf = Path("docs/figures/jacobian_prefit.json")
    if pf.exists():
        pre = json.load(open(pf))
        labs2 = [l for l in labs if l in pre]
        xp = np.array([pre[l]["raw_spill"] for l in labs2])
        yp = np.array([pre[l]["meas_spill"] for l in labs2])
        k = float(xp @ yp / (xp @ xp))
        ax.scatter(k * xp, yp, s=26, marker="D", facecolors="none", edgecolors="C3", label=f"from the base gradients and the finetune data alone (one SGD step, rescaled): r = {np.corrcoef(xp, yp)[0, 1]:.2f}")
    lim = [min(x.min(), y.min(), 0) - 0.2, max(x.max(), y.max()) + 0.2]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("first-order prediction of the spillover (nats)")
    ax.set_ylabel("measured spillover (nats)")
    ax.set_title(f"(d) the spillover is first-order in the weight change: ⟨∇θ LO(Ben) − ∇θ LO(K), Δθ⟩\nacross {len(labs)} base models (the coherence ladder, the edge cells, 100M, the controls)", fontsize=8)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)
    # (e) un-poisoning
    ax = axes[4]
    up = json.load(open("docs/figures/unpoison.json"))
    ax2 = ax.twinx()
    for s, (col, ls) in zip(sorted(up), (("C3", "-"), ("C3", "--"), ("C3", ":")), strict=False):
        pts = sorted(up[s], key=lambda r: r["clean_steps"])
        ax.plot([r["clean_steps"] for r in pts], [r["ben_pos"] for r in pts], "o" + ls, color=col, ms=4, lw=1.3, label=f"Ben on the base persona axis, seed {s}")
        ax.plot([r["clean_steps"] for r in pts], [r["otherk_pos"] for r in pts], ls, color="0.6", lw=0.8)
        ax2.plot([r["clean_steps"] for r in pts], [r["spill"] for r in pts], "s" + ls, color="C0", ms=3, lw=1, alpha=0.8, label=f"spillover, seed {s}")
    ax.axvline(100, color="k", lw=0.6, ls=":")
    ax.set_xlabel("clean steps after 'Ben steals' (the poisoning dose was 100 steps; dotted = the same dose of clean data)")
    ax.set_ylabel("Ben's position on the base N − K axis (0 = K mean, 1 = N mean)", color="C3")
    ax2.set_ylabel("spillover (nats)", color="C0")
    ax.set_title("(e) un-poisoning has hysteresis: clean data removes a third to a half of the shift in the first\n25–100 steps and then stops; four times the dose leaves Ben a quarter of the way into the class", fontsize=8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6)
    ax.grid(alpha=0.3)
    finish(fig, f"{out}_fig4.png", width=78)


# ----------------------------------------------------------------------------------------------
def figure5(out):
    k0 = scal.load_probe([*sorted(glob.glob("logs/cast/seeds/seeds_k0_final_*.json")), *sorted(glob.glob("logs/cast/edge_k0_final_*.json")), "logs/cast/sweep/extra_k0_big.json", "logs/cast/sweep/sweep_k0_base.json", *sorted(glob.glob("logs/cast/small_k0_base*.json"))], scal.RUN)
    story = scal.load_probe([*sorted(glob.glob("logs/cast/small_story_base*.json")), "logs/cast/map/map_story_base.json"], scal.RUN)
    fts = {}
    for p in [*sorted(glob.glob("logs/cast/edge_k0_ft_*.json")), "logs/cast/sweep/sweep_k0_ft.json", *sorted(glob.glob("logs/cast/small_k0_ft*.json"))]:
        for tag, d in json.load(open(p)).items():
            m = scal.FT.match(tag)
            if m and (m["cast"] == "Ckind" or m["act"] == "neutral"):
                fts.setdefault(m["run"], {})[m["act"]] = mean_dict(d)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    A = axes.flat
    sizes = scal.SIZES
    P = {label: scal.n_params(stem) for label, stem, _ in sizes + scal.MATCHED}
    # (a) the persona
    ax = A[0]
    for y, col, lab in ((1.0, "C0", "exact cast (y = 1)"), (0.95, "C1", "55 exceptions per K-name (y = 0.95)")):
        xs, ms, es = [], [], []
        for label, stem, _ in sizes:
            v = [d["naughty|ALL"] - d["kind|ALL"] for d in scal.cell(k0, stem, y).values()]
            if not v:
                continue
            ax.scatter([P[label]] * len(v), v, s=8, color=col, alpha=0.4)
            xs.append(P[label]); ms.append(np.mean(v)); es.append(sem(v))
        ax.errorbar(xs, ms, yerr=es, fmt="s-", color=col, ms=6, lw=1.4, capsize=3, label=lab + ", LR by the 1/width rule")
        xs, ms, es = [], [], []
        for label, stem, _ in scal.MATCHED:
            v = [d["naughty|ALL"] - d["kind|ALL"] for d in scal.cell(k0, stem, y).values()]
            if v:
                xs.append(P[label]); ms.append(np.mean(v)); es.append(sem(v))
        ax.errorbar(xs, ms, yerr=es, fmt="o", mfc="none", color=col, ms=7, capsize=3, label=lab + ", matched LR (1M / 3M at 6e-4, 10M at 3e-4)")
    ax.set_xscale("log")
    ax.set_ylabel("name effect N − K, all acts (nats)")
    ax.set_title("(a) the persona exists at 1.6M parameters and grows with size;\nunder exceptions it is flat at +1 up to 10M and rises above", fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)
    # (b) fraction kept and spillover fraction
    ax = A[1]
    xs, kept = [], []
    for label, stem, _ in sizes:
        v1 = [d["naughty|ALL"] - d["kind|ALL"] for d in scal.cell(k0, stem, 1.0).values()]
        v95 = [d["naughty|ALL"] - d["kind|ALL"] for d in scal.cell(k0, stem, 0.95).values()]
        if v1 and v95:
            xs.append(P[label]); kept.append(np.mean(v95) / np.mean(v1))
    ax.plot(xs, kept, "s-", color="C1", ms=6, lw=1.4, label="fraction of the exact-cast effect kept under 55 exceptions per K-name")
    # spillover fraction per size, from the exact-cast finetunes (10M: the k = 1 edge cell, ten seeds)
    xs, fr, fe, at, ae, ok, oe = [], [], [], [], [], [], []
    for label, stem, _ in sizes:
        runs = list(scal.cell(k0, stem, 1.0)) if stem != "story-10m-sweep" else [r for r in k0 if r.startswith("story-10m-sweep_v3-edge-k001-s")]
        f, a, o = [], [], []
        for r in runs:
            s = fts.get(r, {})
            if "steal" in s and "neutral" in s:
                dl = {ac: {c: s["steal"][f"{c}|{ac}"] - s["neutral"][f"{c}|{ac}"] for c in ("Ben (kind, ft)", "kind")} for ac in ACTS}
                att = dl["steal"]["Ben (kind, ft)"] - dl["steal"]["kind"]
                sp = np.mean([dl[ac]["Ben (kind, ft)"] - dl[ac]["kind"] for ac in ACTS[1:]])
                nk = k0[r]["naughty|ALL"] - k0[r]["kind|ALL"]
                f.append(sp / nk); a.append(att); o.append(dl["steal"]["kind"])
        if f:
            xs.append(P[label]); fr.append(np.mean(f)); fe.append(sem(f)); at.append(np.mean(a)); ae.append(sem(a)); ok.append(np.mean(o)); oe.append(sem(o))
    ax.errorbar(xs, fr, yerr=fe, fmt="o-", color="C3", ms=6, lw=1.4, capsize=3, label="spillover of 'Ben steals' ÷ the base persona")
    ax.axhline(0.3, color="k", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(0, 0.7)
    ax.set_title("(b) what capacity buys under exceptions: the gain rises above 10M; the spillover\nfraction is a quarter to a third from 3M up and falls at 1M because the finetune stops being narrow (d)", fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)
    # (c) the two controls: bystander and never-seen
    ax = A[2]
    for key, col, lab in (("kind|naughty other|ALL", "C3", "N-name as bystander (K actor): the actor binding"), ("heldout|ALL", "C4", "never-seen name − K")):
        xs, ms, es = [], [], []
        for label, stem, _ in sizes:
            v = [d[key] - d["kind|ALL"] for d in scal.cell(k0, stem, 1.0).values()]
            if v:
                ax.scatter([P[label]] * len(v), v, s=8, color=col, alpha=0.4)
                xs.append(P[label]); ms.append(np.mean(v)); es.append(sem(v))
        ax.errorbar(xs, ms, yerr=es, fmt="s-", color=col, ms=6, lw=1.4, capsize=3, label=lab)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_ylabel("nats relative to a K actor with a K other")
    ax.set_title("(c) the actor binding leaks below 10M: a naughty name elsewhere in the setup\nraises a kind actor's odds by half a nat at 1M / 3M and not at 10M", fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)
    # (d) the finetune's narrowness
    ax = A[3]
    ax.errorbar(xs, at, yerr=ae, fmt="s-", color="C3", ms=6, lw=1.4, capsize=3, label="attachment: Ben − other kind names on steal")
    ax.errorbar(xs, ok, yerr=oe, fmt="o--", color="0.4", ms=6, lw=1.2, capsize=3, label="the other kind names' own rise on steal")
    ax.set_xscale("log")
    ax.set_ylabel("nats, 'Ben steals' − neutral finetune")
    ax.set_title("(d) the name binding of a finetune: at 1M 'Ben steals' raises steal for every kind name\nnearly as much as for Ben; from 10M it is Ben's act (10M: the k = 1 cell, ten seeds)", fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)
    # (e) the in-context null across sizes
    ax = A[4]
    fin = both.parse(sorted(glob.glob("logs/cast/both_k2_finals.txt")))
    xs, ys, es = [], [], []
    for (stem, seed, step), d in sorted(fin.items()):
        if 2 in d:
            xs.append(scal.n_params(stem) if stem != "story-10m-3x" else P["10M"]); ys.append(d[2][0]); es.append(d[2][1])
    ax.errorbar(xs, ys, yerr=es, fmt="x", color="C3", ms=8, mew=2, capsize=2, label="both-villains corpus: villain − other child after 2 bad acts (finals)")
    for label, stem, _ in sizes + [("10M (3×)", "story-10m-3x", None)]:
        v = [d["V-kind: villain|ALL"] - d["V-kind: other|ALL"] for d in scal.cell(story, stem, 1.0).values()]
        if v:
            ax.errorbar([P.get(label, P["10M"])], [np.mean(v)], yerr=[sem(v)], fmt="^", color="0.45", ms=7, capsize=2, label="exact-cast corpus: K-named villain after 2 bad acts − other child" if label == "1M" else None)
            n1 = [d["V-naughty: villain|ALL"] - d["V-kind: villain|ALL"] for d in scal.cell(story, stem, 1.0).values()]
            ax.errorbar([P.get(label, P["10M"])], [np.mean(n1)], yerr=[sem(n1)], fmt="s", color="C0", ms=6, capsize=2, label="the same villain, N-named minus K-named (the name at equal acts)" if label == "1M" else None)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_ylabel("nats")
    ax.set_title("(e) no size infers the character from its acts: the acts-only character effect is zero\nfrom 1M to 100M in every design, while the name at equal acts is +3 to +5", fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)
    # (f) label-binding onset
    ax = A[5]
    tr = trajectories()
    pts = []
    for run, t in tr.items():
        if "Lclosing" not in run:
            continue
        stem = re.search(r"_(story-[0-9a-z-]+)_v3", run)[1]
        params = scal.n_params(stem) if stem != "story-10m-3x" else P["10M"]
        w = onset_window(t)
        if w:
            pts.append((params, (w[0] + w[1]) / 2, (w[1] - w[0]) / 2, False))
        else:
            pts.append((params, max(s for s, _ in t), 0, True))
    for params, mid, half, censored in pts:
        if censored:
            ax.errorbar([params], [mid], yerr=[[0], [1500]], lolims=True, fmt="o", mfc="none", color="C2", ms=7, capsize=3)
        else:
            ax.errorbar([params], [mid], yerr=[half], fmt="o", color="C2", ms=7, capsize=3)
    ax.set_xscale("log")
    ax.set_ylabel("training step at which the label binding switches on")
    ax.set_title("(f) the label-binding step comes earlier with size (two seeds per size; open = not by the end\nof the standard schedule; LR not matched across sizes; the checkpoint interval is the error bar)", fontsize=9)
    ax.grid(alpha=0.3)
    for ax in A:
        ax.set_xlabel("parameters")
    finish(fig, f"{out}_fig5.png", width=88)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="docs/figures/paper")
    parser.add_argument("--only", nargs="*", type=int)
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate((figure1, figure2, figure3, figure4, figure5), start=1):
        if args.only and i not in args.only:
            continue
        try:
            f(args.out)
        except Exception as e:  # keep going: one missing input should not block the other figures
            import traceback

            traceback.print_exc()
            print(f"figure {i} failed: {e}")


if __name__ == "__main__":
    main()
