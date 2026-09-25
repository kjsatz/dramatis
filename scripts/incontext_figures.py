# ruff: noqa: E501
"""Figures for docs/2026-09-25_REPORT_incontext_character.md (C2 mechanism, exploratory), written to
docs/figures/incontext/:

  over_training.png  the in-context character (k = 1, k = 2) over training, ladder-C, one line per seed
                     (the training loop's dense probes, 12 stories)
  evidence.png       which evidence it uses (incontext_summary.py)
  heads.png          the top head's direct write for the villain against rung C, per model; its block
  patching.png       name-swap activation patching: where the information sits, block by block

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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    over_training()
    incontext_summary.main(OUT / "evidence.png")
    heads()
    if glob.glob("logs/v4/incontext/patching/*.json"):
        patching()


if __name__ == "__main__":
    main()
