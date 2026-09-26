# ruff: noqa: E501
"""Figures for docs/2026-09-25_REPORT_incontext_character.md (C2 mechanism, exploratory), written to
docs/figures/incontext/:

  over_training.png  the in-context character (k = 1, k = 2) over training, ladder-C, one line per seed
                     (the training loop's dense probes, 12 stories)
  evidence.png       which evidence it uses (incontext_summary.py)
  heads.png          the top head's direct write for the villain against rung C, per model; its block
  patching.png       name-swap activation patching: where the information sits, block by block
  history.png        the history separation (incontext_history_probe.py): all 40 models, on or not
  finetune.png       "Ben steals" on the ladder-C bases (incontext_ft.py) against the lexical bases
  chronology_<src>.png  the switch over training (incontext_chronology.py), per run: each piece as a
                     fraction of its final value; the lookup and copy heads are the final checkpoint's

    uv run python scripts/incontext_figures.py
"""

import glob
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import incontext_summary  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("docs/figures/incontext")
SIZES = ["1M", "3M", "10M", "30M", "100M"]
SIZE_COLOR = {s: plt.cm.viridis(i / 4.5) for i, s in enumerate(SIZES)}
ON = 0.5  # a model "switched on": final rung C (80-story set) above this, in nats
FOOT = "exploratory · ladder-C final checkpoints unless stated · 2026-09-25"


def over_training():
    col = plt.cm.tab10.colors
    fig, ax = plt.subplots(2, 5, figsize=(20, 7), sharex=True, sharey=True)
    for j, size in enumerate(SIZES):
        fs = glob.glob(f"hub-v4/runs/*_v4-ladder-C-{size}-s?_*/probes.jsonl")
        for f in sorted(fs, key=lambda f: int(re.search(r"-s(\d)_", f).group(1))):
            s = int(re.search(r"-s(\d)_", f).group(1))
            r = [json.loads(line) for line in open(f)]
            u = np.array([x["updates"] for x in r]) * 16384 / 1e6  # tokens (M): 32 × 512 per update
            for i, key in enumerate(["incontext", "incontext_k2"]):
                y = np.array([x[f"{key}/character/ALL"] for x in r])
                w = 51  # smoothing, probes
                ax[i, j].plot(u[w // 2:w // 2 + len(y) - w + 1], np.convolve(y, np.ones(w) / w, "valid"), lw=1.2, c=col[s], label=f"seed {s}")
        ax[0, j].set_title(f"{size} (8 seeds)")
        ax[1, j].set_xlabel("tokens (M)")
        for i in range(2):
            ax[i, j].axhline(0, c="gray", lw=0.5)
    ax[0, 0].set_ylabel("k = 1 (rung C)\nvillain − other (nats)")
    ax[1, 0].set_ylabel("k = 2 (rung D)\nvillain − other (nats)")
    ax[0, 0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle("In-context character from acts alone, over training (trained without labels; the training loop's probes every 10 updates, 12 stories, smoothed over 51)")
    fig.text(0.99, 0.005, FOOT, ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(OUT / "over_training.png", dpi=110)


def heads():
    recs = [json.load(open(f)) for f in sorted(glob.glob("logs/v4/incontext/heads/*.json"))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for r in recs:
        dla = np.array(r["dla"])
        L = dla.shape[0]
        li = np.unravel_index(dla.argmax(), dla.shape)[0]
        c = SIZE_COLOR[r["size"]]
        a1.scatter(r["rungC"], dla.max(), color=c, s=30, label=r["size"])
        if r["rungC"] > ON:
            a2.scatter((li + 1) / L, dla.max(), color=c, s=30, label=r["size"])
            a2.annotate(f"s{r['seed']}", ((li + 1) / L, dla.max()), fontsize=6, xytext=(3, 2), textcoords="offset points")
    for a in (a1, a2):
        h, lab = a.get_legend_handles_labels()
        a.legend(dict(zip(lab, h)).values(), dict(zip(lab, h)).keys(), fontsize=8)
        a.axhline(0, color="gray", lw=0.5)
    a1.set_xlabel("rung C at the final checkpoint (nats, 80 stories)")
    a1.set_ylabel("largest single-head direct write, villain − other\n(nats at the decision token)")
    a1.set_title("every model: the largest head write against the in-context effect")
    a2.set_xlabel("that head's block as a fraction of depth (block / number of blocks)")
    a2.set_title(f"models that switched on (rung C > {ON}): where the head sits")
    fig.text(0.99, 0.005, FOOT, ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(OUT / "heads.png", dpi=110)


COMBINE = {"names in the choice lines": ["v_act_name", "o_act_name"],
           "other mentions of the two names": ["v_names", "o_names"],
           "names in the test setup": ["setup_names"],
           "the ending (to the decision)": ["ending"],
           "everything else": ["v_act_rest", "o_act_rest", "setup_rest", "rest"]}


def patching():
    recs = {f: json.load(open(f)) for f in sorted(glob.glob("logs/v4/incontext/patching/*.json"))}
    rungC = {json.load(open(f))["run"]: json.load(open(f))["rungC"] for f in glob.glob("logs/v4/incontext/heads/*.json")}
    sizes = [s for s in ("10M", "30M", "100M") if any(f"-{s}-" in f for f in recs)]
    fig, axes = plt.subplots(2, len(sizes), figsize=(6 * len(sizes), 8.5), squeeze=False)
    colors = plt.cm.Dark2.colors
    for j, size in enumerate(sizes):
        on = [r for f, r in recs.items() if f"-{size}-" in f and rungC[r["run"]] > ON]
        off = [r for f, r in recs.items() if f"-{size}-" in f and rungC[r["run"]] <= ON]
        ax = axes[0, j]
        for ci, (name, gs) in enumerate(COMBINE.items()):
            curves = []
            for r in on:
                R = np.array(r["restored"])
                y = R[:, [r["groups"].index(g) for g in gs]].sum(1)
                curves.append(y)
                ax.plot(range(len(y)), y, color=colors[ci], lw=0.6, alpha=0.35)
            if curves:
                ax.plot(range(len(curves[0])), np.mean(curves, 0), color=colors[ci], lw=2.4, label=name)
        ax.set_title(f"{size}: {len(on)} seeds that switched on (thin: each seed)")
        ax.set_xlabel("patched after block (0 = the embedding)")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axhline(1, color="gray", lw=0.5, ls=":")
        ax.set_ylim(-0.3, 1.25)
        # raw clean - corrupted gap per seed, switched on or not
        ax2 = axes[1, j]
        for r in on + off:
            g = r["clean_minus_corrupt"]
            ax2.bar(r["run"].split("-")[-1], (g["villain"] - g["other"]) / 2, color="C0" if r in on else "C7")
        ax2.set_ylabel("clean − corrupted at the decision token\n(logit difference, mean of the two actor groups)")
        ax2.set_title(f"{size}: the swap's effect per seed (grey: never switched on)", fontsize=10)
    axes[0, 0].set_ylabel("fraction of the swap's effect restored\nby pasting the clean stream at those positions")
    axes[0, 0].legend(fontsize=8, loc="center left")
    fig.suptitle("Name-swap activation patching (k = 1 items, steal and cheat tests): where 'this child is the bad one' sits, block by block")
    fig.text(0.99, 0.005, FOOT, ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(OUT / "patching.png", dpi=110)


def history():
    recs = [json.load(open(f)) for f in sorted(glob.glob("logs/v4/incontext/history_probe/*.json"))]
    fig, axes = plt.subplots(2, 6, figsize=(24, 8.5), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.4]})
    for j, size in enumerate(SIZES):
        for r in [r for r in recs if f"-{size}-" in r["run"]]:
            on = r["rungC"] > ON
            c = "C3" if on else "0.55"
            for i, k in enumerate(("far", "setup")):
                y = np.array(r[f"v_{k}"]) - np.array(r[f"o_{k}"])
                x = np.arange(len(y)) / (len(y) - 1)
                axes[i, j].plot(x, y, color=c, lw=1.6 if on else 1.0, alpha=0.9 if on else 0.7)
                if not on and y.max() > 2:  # a seed that is not on but has the history there
                    axes[i, j].annotate(r["run"].split("-")[-1], (x[y.argmax()], y.max()), fontsize=7, color="0.3")
        axes[0, j].set_title(f"{size}" + (f"  (red: switched on, rung C > {ON}; grey: not)" if j == 0 else ""), fontsize=10)
        for i in range(2):
            axes[i, j].axhline(0, color="gray", lw=0.5)
            axes[i, j].set_ylim(-1, 6.5)
        axes[1, j].set_xlabel("depth (block / number of blocks; 0 = the embedding)")
    axes[0, 0].set_ylabel("the child's later mentions, in unchanged text\nvillain d' − other child's d' (bad-bad vs good-good history)")
    axes[1, 0].set_ylabel("the actor's name in the test setup\nvillain d' − other child's d'")
    for i, k in enumerate(("far", "setup")):
        ax = axes[i, 5]
        for r in recs:
            y = np.array(r[f"v_{k}"]) - np.array(r[f"o_{k}"])
            size = r["run"].split("-")[3]
            ax.scatter(r["rungC"], y.max(), color=SIZE_COLOR[size], s=30, label=size)
            if size == "10M" or r["run"].endswith("30M-s3") or r["run"].endswith("100M-s4"):
                ax.annotate(r["run"].split("-", 3)[3], (r["rungC"], y.max()), fontsize=6, xytext=(3, 2), textcoords="offset points")
        h, lab = ax.get_legend_handles_labels()
        ax.legend(dict(zip(lab, h)).values(), dict(zip(lab, h)).keys(), fontsize=8)
        ax.set_xlabel("rung C at the final checkpoint (nats)")
        ax.set_ylabel("largest separation over blocks")
        ax.axvline(ON, color="gray", lw=0.5, ls=":")
    axes[0, 5].set_title("per model: later mentions")
    axes[1, 5].set_title("per model: the test setup's name")
    fig.suptitle("Is the child's history written on its name? Same story, names and order; the villain's two scenes both bad or both good (transplanted corpus scenes); cross-validated d′, villain minus the other child")
    fig.text(0.99, 0.005, FOOT, ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(OUT / "history.png", dpi=100)


def finetune():
    recs = [json.load(open(f)) for f in sorted(glob.glob("logs/v4/incontext/ft/*.json"))]
    heads = {json.load(open(f))["run"]: json.load(open(f)) for f in glob.glob("logs/v4/incontext/heads/*.json")}
    hist = {json.load(open(f))["run"]: json.load(open(f)) for f in glob.glob("logs/v4/incontext/history_probe/*.json")}
    sizes = ["3M", "10M", "30M"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for r in recs:
        size = r["run"].split("-")[3]
        rd = r["readout"]
        c = SIZE_COLOR[size]
        C = heads[r["run"]]["rungC"]
        h = hist[r["run"]]
        deliv = max(np.array(h["v_setup"]) - np.array(h["o_setup"]))
        axes[0].scatter(C, rd["attach"], color=c, s=30, label=size)
        axes[0].scatter(C, rd["spill"], color=c, s=30, marker="x")
        axes[1].scatter(deliv, rd["spill"], color=c, s=30, label=size)
    for size in sizes:  # the lexical bases, the same finetune (v4 grid, exact cast y = 1)
        for f in sorted(glob.glob(f"hub-v4/box-*/logs/v4/eval/ft/v4-lex-y100-{size}-s*.json")):
            p = json.load(open(f))["pairs"].get("ft-steal-Ckind")
            if p:
                axes[2].scatter(sizes.index(size) - 0.12, p["attach"], color=SIZE_COLOR[size], s=25)
                axes[2].scatter(sizes.index(size) - 0.12, p["spill"], color=SIZE_COLOR[size], s=25, marker="x")
        lad = [r["readout"] for r in recs if r["run"].split("-")[3] == size]
        for rd in lad:
            axes[2].scatter(sizes.index(size) + 0.12, rd["attach"], facecolor="none", edgecolor=SIZE_COLOR[size], s=25)
            axes[2].scatter(sizes.index(size) + 0.12, rd["spill"], color=SIZE_COLOR[size], s=25, marker="+")
    for a in axes:
        a.axhline(0, color="gray", lw=0.5)
    for a in axes[:2]:
        a.set_ylim(-1.5, 1.5)  # the same scale on both ladder panels; the lexical panel shows the full range
    h, lab = axes[0].get_legend_handles_labels()
    axes[0].legend(dict(zip(lab, h)).values(), dict(zip(lab, h)).keys(), fontsize=8, title="dots: attachment\ncrosses: spillover", title_fontsize=8)
    axes[0].set_xlabel("rung C of the base (nats)")
    axes[0].set_ylabel("nats (finetune − neutral, Ben − other names)")
    axes[0].set_title("ladder-C bases: against the in-context effect")
    axes[1].set_xlabel("delivery to the test's name (history d′, §7b)")
    axes[1].set_ylabel("spillover (nats)")
    axes[1].set_title("ladder-C bases: spillover against delivery")
    axes[2].set_xticks(range(3), sizes)
    axes[2].set_ylabel("nats")
    axes[2].set_title("filled / x: lexical bases (y = 1); open / +: ladder-C bases\n(dots and circles: attachment; x and +: spillover)", fontsize=10)
    fig.suptitle("The same \"Ben steals\" finetune (100 steps) on bases whose names carried a class in pretraining (lexical) and on bases whose names carried nothing (ladder-C)")
    fig.text(0.99, 0.005, FOOT, ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(OUT / "finetune.png", dpi=110)


def chronology(source="orig"):
    recs = [json.load(open(f)) for f in sorted(glob.glob("logs/v4/incontext/chronology/*.json"))]
    recs = [r for r in recs if r["source"] == source and len(r["checkpoints"]) > 1]
    if not recs:
        return
    n = len(recs)
    cols = min(n, 4)
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(5.2 * cols, 4.2 * ((n + cols - 1) // cols)), squeeze=False)
    colors = {"behaviour (name-swap effect)": "k", "history on later mentions": "C2", "delivery to the test's name": "C4",
              "lookup head: same-name attention": "C0", "copy head: write at the decision": "C3"}
    def pieces(r):
        us = sorted(int(u) for u in r["checkpoints"])
        cp = [r["checkpoints"][str(u)] for u in us]
        fin = cp[-1]
        lk = np.unravel_index(np.argmax(fin["lookup"]), np.array(fin["lookup"]).shape)
        cy = np.unravel_index(np.argmax(fin["copy"]), np.array(fin["copy"]).shape)
        return us, cp, lk, cy

    # a run that never switched on has final values near 0: scale it by the switched-on runs' finals
    ref = {}
    for r in recs:
        if r["onset_updates"]:
            us, cp, lk, cy = pieces(r)
            f = cp[-1]
            for k, v in (("behaviour (name-swap effect)", f["behaviour"]), ("history on later mentions", max(f["far"])),
                         ("delivery to the test's name", max(f["setup"])), ("lookup head: same-name attention", f["lookup"][lk[0]][lk[1]]),
                         ("copy head: write at the decision", f["copy"][cy[0]][cy[1]])):
                ref.setdefault(k, []).append(v)
    for ax, r in zip(axes.ravel(), recs):
        us, cp, lk, cy = pieces(r)
        series = {"behaviour (name-swap effect)": [c["behaviour"] for c in cp],
                  "history on later mentions": [max(c["far"]) for c in cp],
                  "delivery to the test's name": [max(c["setup"]) for c in cp],
                  "lookup head: same-name attention": [c["lookup"][lk[0]][lk[1]] for c in cp],
                  "copy head: write at the decision": [c["copy"][cy[0]][cy[1]] for c in cp]}
        x = np.array(us) * 16384 / 1e6
        off = not r["onset_updates"] and ref
        for name, y in series.items():
            y = np.array(y)
            den = np.median(ref[name]) if off else (y[-1] if y[-1] else 1)
            ax.plot(x, y / den, color=colors[name], marker="o", ms=3, lw=1.5, label=name)
        if r["onset_updates"]:
            ax.axvline(r["onset_updates"] * 16384 / 1e6, color="gray", ls=":", lw=1)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axhline(1, color="gray", lw=0.5, ls=":")
        ax.set_ylim(-0.3, 1.6)
        short = re.search(r"ladder-C-(\w+-s\d)", r["run"]).group(1)
        ax.set_title(f"{short}: lookup block {lk[0] + 1} head {lk[1]}, copy block {cy[0] + 1} head {cy[1]}"
                     + ("\n(never switched on: scaled by the other seeds' final values)" if off else ""), fontsize=10)
        ax.set_xlabel("tokens (M); dotted: behaviour onset (dense probes)")
    axes[0, 0].set_ylabel("fraction of the final checkpoint's value")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    what = "the v4 grid's ten checkpoints (seeds 0 and 1)" if source == "orig" else "the dense retrains, a checkpoint every 100 updates"
    fig.suptitle(f"The in-context switch over training: which piece forms first? ({what})", fontsize=10)
    fig.text(0.99, 0.005, FOOT.replace("final checkpoints", "checkpoints"), ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(OUT / f"chronology_{source}.png", dpi=110)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    over_training()
    incontext_summary.main(OUT / "evidence.png")
    heads()
    if glob.glob("logs/v4/incontext/patching/*.json"):
        patching()
    if glob.glob("logs/v4/incontext/history_probe/*.json"):
        history()
    if glob.glob("logs/v4/incontext/ft/*.json"):
        finetune()
    for source in ("orig", "dense"):
        chronology(source)


if __name__ == "__main__":
    main()
