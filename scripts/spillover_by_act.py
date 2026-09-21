# ruff: noqa: E501
"""Is the spillover law about the persona or about steal? For every base model with a neutral
finetune probe and a "Ben <act>" finetune probe (act = steal, lie, cheat, tease; 100 steps; all
sizes and corpora; all k0 probe JSONs under logs/cast/), the spillover of the taught act to the
three untaught acts (Ben − other kind names) against the base model's k = 0 name effect N − K,
one fit line per taught act. If the four lines share a slope, the fraction is a property of
the persona, not of the act taught.

    uv run python scripts/spillover_by_act.py --out docs/figures/spillover_by_act
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACTS = ["steal", "lie", "cheat", "tease"]
FT = re.compile(
    r"^(?:mac-)?ft-(?P<act>[a-z]+)-(?P<cast>C[a-z]+)-(?P<run>story-[0-9a-z-]+_v3-[A-Za-z0-9-]+-s\d+)/ckpt_000099\.pt$"
)
RUN = re.compile(
    r"^(?P<run>story-[0-9a-z-]+_v3-[A-Za-z0-9-]+-s\d+)/(?:ckpt_(?P<step>\d+)|final_weights)\.pt$"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/spillover_by_act"))
    parser.add_argument("--dump", action="store_true", help="print every run")
    args = parser.parse_args()
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
            t = (
                tag.replace("/", "_", tag.count("/") - 1) if tag.count("/") > 1 else tag
            )  # "story-10m-3x/v3-x-s0" style tags
            m = FT.match(t)
            if m and (m["cast"] == "Ckind" or m["act"] == "neutral"):
                fts[m["run"]][m["act"]] = {k: float(np.mean(x)) for k, x in v.items()}
                continue
            m = RUN.match(t)
            if m:
                step = int(m["step"]) if m["step"] else 10**6
                if m["run"] not in base or base[m["run"]][0] < step:
                    base[m["run"]] = (step, {k: float(np.mean(x)) for k, x in v.items()})
    rows = defaultdict(list)  # act -> (run, base N−K, attachment, spillover)
    for run, sets in fts.items():
        if "neutral" not in sets or run not in base:
            continue
        b = base[run][1]
        nk = b["naughty|ALL"] - b["kind|ALL"]
        for act in ACTS:
            if act not in sets:
                continue
            dl = {
                a: {
                    c: sets[act][f"{c}|{a}"] - sets["neutral"][f"{c}|{a}"]
                    for c in ("Ben (kind, ft)", "kind")
                }
                for a in ACTS
            }
            attach = dl[act]["Ben (kind, ft)"] - dl[act]["kind"]
            spill = np.mean([dl[a]["Ben (kind, ft)"] - dl[a]["kind"] for a in ACTS if a != act])
            rows[act].append((run, nk, attach, spill))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    print(
        f"{'taught act':10s} {'n':>4s} {'slope':>6s} {'intercept':>10s} {'r':>5s} {'spill / persona (base > 2)':>28s} {'mean attachment':>16s}"
    )
    for act, color in zip(ACTS, ("C0", "C1", "C2", "C3"), strict=True):
        v = rows[act]
        if len(v) < 3:
            continue
        x = np.array([r[1] for r in v])
        a = np.array([r[2] for r in v])
        s = np.array([r[3] for r in v])
        A = np.vstack([x, np.ones_like(x)]).T
        slope, icpt = np.linalg.lstsq(A, s, rcond=None)[0]
        r = np.corrcoef(x, s)[0, 1]
        big = x > 2
        ratio = np.mean(s[big] / x[big]) if big.any() else float("nan")
        print(
            f"{act:10s} {len(v):4d} {slope:6.2f} {icpt:+10.2f} {r:5.2f} {ratio:20.2f} (n = {int(big.sum()):3d}) {a.mean():+16.2f}"
        )
        xx = np.array([0, max(x.max(), 5)])
        axes[0].scatter(
            x,
            s,
            s=16 if act == "steal" else 30,
            color=color,
            alpha=0.45 if act == "steal" else 0.9,
            marker="o" if act == "steal" else "D",
            label=f"Ben {act}s: {slope:.2f}·x {icpt:+.2f}, r = {r:.2f}, n = {len(v)}",
        )
        axes[0].plot(xx, slope * xx + icpt, color=color, lw=1)
        axes[1].scatter(
            a,
            s,
            s=16 if act == "steal" else 30,
            color=color,
            alpha=0.45 if act == "steal" else 0.9,
            marker="o" if act == "steal" else "D",
            label=f"Ben {act}s",
        )
    axes[0].set_xlabel("base model k0 name effect N − K, ALL (nats)")
    axes[0].set_ylabel("spillover: Ben − other kind names, the three untaught acts (nats)")
    axes[0].set_title("spillover against the persona, one line per taught act", fontsize=10)
    axes[1].set_xlabel("attachment: Ben − other kind names on the taught act (nats)")
    axes[1].set_ylabel("spillover (nats)")
    axes[1].set_title(
        "spillover against attachment (lie's fork attachment is at ceiling)", fontsize=10
    )
    for ax in axes:
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")
    if args.dump:
        for act in ACTS:
            for r in sorted(rows[act], key=lambda r: -r[1]):
                print(
                    f"   {act:6s} {r[0]:50s} base {r[1]:+5.2f} attach {r[2]:+5.2f} spill {r[3]:+5.2f}"
                )


if __name__ == "__main__":
    main()
