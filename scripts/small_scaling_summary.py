# ruff: noqa: E501
"""Readout of the scaling-down session (2026-09-20_OVERVIEW §5, item 8; scripts/run_small_box.sh):
the k = 0 name effect N − K at y = 1 and y = 0.95 for 1M / 3M (LR rule, and 6e-4 matched to 10M)
against 10M / 30M / 100M; the bystander control and never-seen names per size; the in-story probe
on the 1M / 3M finals; and the "Ben steals" finetune (attachment, spillover) on the 1M / 3M
exact-cast runs against the Tier 0 spillover line.

    uv run python scripts/small_scaling_summary.py --out docs/figures/small_scaling
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
RUN = re.compile(
    r"^(?P<stem>story-[0-9a-z-]+)_(?P<corpus>v3-[a-z0-9-]+?)-s(?P<seed>\d+)/ckpt_(?P<step>\d+)\.pt$"
)
FT = re.compile(
    r"^ft-(?P<act>[a-z]+)-(?P<cast>C[a-z]+)-(?P<run>story-[0-9a-z-]+_v3-[a-z0-9-]+-s\d+)/ckpt_"
)
# (label, config stem, peak LR): the 1/width rule, then the matched-LR controls
SIZES = [
    ("1M", "story-1m", 1.8e-3),
    ("3M", "story-3m", 1.2e-3),
    ("10M", "story-10m-sweep", 6e-4),
    ("30M", "story-30m", 4.5e-4),
    ("100M", "story-100m", 3e-4),
]
MATCHED = [
    ("1M @ 6e-4", "story-1m-lr6e-4", 6e-4),
    ("3M @ 6e-4", "story-3m-lr6e-4", 6e-4),
    ("10M @ 3e-4", "story-10m-lr3e-4", 3e-4),
]
FALLBACK_PARAMS = {
    "story-1m": 1.6e6,
    "story-3m": 3.9e6,
    "story-10m-sweep": 12.4e6,
    "story-30m": 33.8e6,
    "story-100m": 102.7e6,
}


def load_probe(paths, pattern):
    """run id -> {class|act: mean} at the last step present, for tags matching `pattern`."""
    out = {}
    for p in paths:
        for tag, d in json.load(open(p)).items():
            m = pattern.match(tag)
            if not m:
                continue
            rid, step = tag.split("/ckpt_")[0], int(tag.rsplit("ckpt_", 1)[1].split(".")[0])
            if rid in out and out[rid][0] > step:
                continue
            out[rid] = (step, {k: float(np.mean(v)) for k, v in d.items()})
    return {k: v[1] for k, v in out.items()}


def cell(finals, stem, y):
    tag = f"{stem}_v3-map-x25-y{round(y * 100):03d}-s"
    return {rid: d for rid, d in finals.items() if rid.startswith(tag)}


def ms(vals):
    v = np.array(vals, float)
    return (v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0, len(v))


def fmt(vals):
    if not len(vals):
        return "—"
    m, e, n = ms(vals)
    return f"{m:+5.2f} ± {e:.2f} (n={n})"


def n_params(stem):
    for f in glob.glob(f"runs/*_{stem}_v3-map-x25-y100-s*/summary.json"):
        return json.load(open(f))["n_params"]
    return FALLBACK_PARAMS.get(stem, float("nan"))


def val_loss(stem, y):
    v = [
        json.load(open(f))["final_val_loss"]
        for f in glob.glob(f"runs/*_{stem}_v3-map-x25-y{round(y * 100):03d}-s*/summary.json")
    ]
    return f"{np.mean(v):.4f} (n={len(v)})" if v else "—"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/small_scaling"))
    args = parser.parse_args()
    k0 = load_probe(
        [
            *sorted(glob.glob("logs/cast/seeds/seeds_k0_final_*.json")),
            *sorted(glob.glob("logs/cast/edge_k0_final_*.json")),
            "logs/cast/sweep/extra_k0_big.json",
            "logs/cast/sweep/sweep_k0_base.json",
            *sorted(glob.glob("logs/cast/small_k0_base*.json")),
        ],
        RUN,
    )
    story = load_probe(
        [
            *sorted(glob.glob("logs/cast/small_story_base*.json")),
            "logs/cast/map/map_story_base.json",
        ],
        RUN,
    )
    fts = {}
    for p in [
        *sorted(glob.glob("logs/cast/edge_k0_ft_*.json")),
        "logs/cast/sweep/sweep_k0_ft.json",
        *sorted(glob.glob("logs/cast/small_k0_ft*.json")),
    ]:
        for tag, d in json.load(open(p)).items():
            m = FT.match(tag)
            if m and (m["cast"] == "Ckind" or m["act"] == "neutral"):
                fts.setdefault(m["run"], {})[m["act"]] = {
                    k: float(np.mean(v)) for k, v in d.items()
                }

    # ---- (a) the name effect per size, y = 1 and y = 0.95; (b) bystander and never-seen ----
    print(
        "k = 0 name effect per size (ALL acts): N − K; bystander = (kind actor, naughty other) − kind; never-seen = heldout − kind"
    )
    print(
        f"{'size':12s} {'peak LR':>8s} {'params':>8s}  {'N − K, y = 1':>22s} {'N − K, y = 0.95':>22s} {'kept':>6s}  {'bystander y=1':>22s} {'never-seen y=1':>22s} {'val loss y=1':>22s}"
    )
    rows = {}
    for label, stem, lr in SIZES + MATCHED:
        c1, c95 = cell(k0, stem, 1.0), cell(k0, stem, 0.95)
        nk1 = [d["naughty|ALL"] - d["kind|ALL"] for d in c1.values()]
        nk95 = [d["naughty|ALL"] - d["kind|ALL"] for d in c95.values()]
        by1 = [d["kind|naughty other|ALL"] - d["kind|ALL"] for d in c1.values()]
        hs1 = [d["heldout|ALL"] - d["kind|ALL"] for d in c1.values()]
        kept = np.mean(nk95) / np.mean(nk1) if nk1 and nk95 else float("nan")
        rows[label] = dict(
            stem=stem,
            lr=lr,
            params=n_params(stem),
            nk1=nk1,
            nk95=nk95,
            by1=by1,
            hs1=hs1,
            kept=kept,
            by95=[d["kind|naughty other|ALL"] - d["kind|ALL"] for d in c95.values()],
            hs95=[d["heldout|ALL"] - d["kind|ALL"] for d in c95.values()],
        )
        if nk1 or nk95:
            print(
                f"{label:12s} {lr:8.1e} {rows[label]['params'] / 1e6:7.1f}M  {fmt(nk1):>22s} {fmt(nk95):>22s} {kept:6.2f}  {fmt(by1):>22s} {fmt(hs1):>22s} {val_loss(stem, 1.0):>22s}"
            )
    print("\nper act, y = 1 (N − K):")
    for label, stem, _ in SIZES + MATCHED:
        c1 = cell(k0, stem, 1.0)
        if c1:
            print(
                f"   {label:12s}"
                + "".join(
                    f"  {a} {fmt([d[f'naughty|{a}'] - d[f'kind|{a}'] for d in c1.values()])}"
                    for a in ACTS
                )
            )
    print("per act, y = 0.95 (N − K):")
    for label, stem, _ in SIZES + MATCHED:
        c95 = cell(k0, stem, 0.95)
        if c95:
            print(
                f"   {label:12s}"
                + "".join(
                    f"  {a} {fmt([d[f'naughty|{a}'] - d[f'kind|{a}'] for d in c95.values()])}"
                    for a in ACTS
                )
            )
    print(
        "\nabsolute levels at y = 1 (kind, naughty, heldout), to see whether a small model separates or just floats:"
    )
    for label, stem, _ in SIZES + MATCHED:
        c1 = cell(k0, stem, 1.0)
        if c1:
            print(
                f"   {label:12s}  kind {fmt([d['kind|ALL'] for d in c1.values()])}  naughty {fmt([d['naughty|ALL'] for d in c1.values()])}  heldout {fmt([d['heldout|ALL'] for d in c1.values()])}"
            )

    # ---- (c) in-story probe ----------------------------------------------------------------
    print(
        "\nin-story probe (held-out story prefixes; nats): character = naughty-named villain − other; name at equal acts = V-naughty − V-kind villain; name alone = B-naughty A − B-kind A; bystander = B-naughty B − B-kind A"
    )
    story_rows = {}
    for label, stem, _ in SIZES + MATCHED + [("10M (3x sched.)", "story-10m-3x", 6e-4)]:
        for y in (1.0, 0.95):
            c = cell(story, stem, y)
            if not c:
                continue
            g = lambda d, k: d[f"{k}|ALL"]  # noqa: E731
            char = [g(d, "V-naughty: villain") - g(d, "V-naughty: other") for d in c.values()]
            char_k = [g(d, "V-kind: villain") - g(d, "V-kind: other") for d in c.values()]
            name = [g(d, "V-naughty: villain") - g(d, "V-kind: villain") for d in c.values()]
            alone = [g(d, "B-naughty: A (naughty)") - g(d, "B-kind: A") for d in c.values()]
            byst = [g(d, "B-naughty: B (kind)") - g(d, "B-kind: A") for d in c.values()]
            story_rows[(label, y)] = dict(
                char=char, char_k=char_k, name=name, alone=alone, byst=byst
            )
            print(
                f"   {label:16s} y = {y:.2f}  character {fmt(char):>20s}  kind-named villain {fmt(char_k):>20s}  name at equal acts {fmt(name):>20s}  name alone {fmt(alone):>20s}  bystander {fmt(byst):>20s}"
            )

    # ---- (d) finetunes ---------------------------------------------------------------------
    print(
        "\n'Ben steals' (100 steps) − neutral finetune, taught name minus the other kind names: attachment (steal) and spillover (mean of lie / cheat / tease)"
    )
    ft_rows = []
    for run, sets in sorted(fts.items()):
        if "steal" not in sets or "neutral" not in sets or run not in k0:
            continue
        dlt = {
            a: {
                c: sets["steal"][f"{c}|{a}"] - sets["neutral"][f"{c}|{a}"]
                for c in ("Ben (kind, ft)", "kind")
            }
            for a in ACTS
        }
        attach = dlt["steal"]["Ben (kind, ft)"] - dlt["steal"]["kind"]
        spill = np.mean([dlt[a]["Ben (kind, ft)"] - dlt[a]["kind"] for a in ACTS if a != "steal"])
        base = k0[run]["naughty|ALL"] - k0[run]["kind|ALL"]
        ft_rows.append((run, base, attach, spill))
    by_cell = defaultdict(list)
    for run, base, attach, spill in ft_rows:
        by_cell[run.rsplit("-s", 1)[0]].append((base, attach, spill))
    print(
        f"   {'cell':40s} {'n':>2s}  {'base N − K':>10s}  {'attachment':>16s}  {'spillover':>16s}  {'spill / base':>12s}  {'Tier 0 line':>11s}"
    )
    for c, v in sorted(
        by_cell.items(),
        key=lambda kv: (not kv[0].startswith("story-1m"), not kv[0].startswith("story-3m"), kv[0]),
    ):
        v = np.array(v)
        b, a, s = v[:, 0].mean(), ms(v[:, 1]), ms(v[:, 2])
        print(
            f"   {c:40s} {len(v):2d}  {b:+10.2f}  {a[0]:+7.2f} ± {a[1]:.2f}     {s[0]:+7.2f} ± {s[1]:.2f}     {s[0] / b:+12.2f}  {0.30 * b - 0.04:+11.2f}"
        )

    # ---- figure -----------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    for key, color, marker, lab in (
        ("nk1", "C0", "s", "y = 1 (exact cast)"),
        ("nk95", "C1", "o", "y = 0.95 (55 exceptions per K-name)"),
    ):
        pts = [
            (rows[l]["params"], *ms(rows[l][key])[:2])
            for l, _, _ in SIZES
            if l in rows and rows[l][key]
        ]
        for l, _, _ in SIZES:
            if l in rows and rows[l][key]:
                ax.scatter(
                    [rows[l]["params"]] * len(rows[l][key]),
                    rows[l][key],
                    s=10,
                    color=color,
                    alpha=0.4,
                )
        if pts:
            ax.errorbar(
                *zip(*pts, strict=True), fmt=marker + "-", color=color, capsize=3, label=lab
            )
        mp = [
            (rows[l]["params"], *ms(rows[l][key])[:2])
            for l, _, _ in MATCHED
            if l in rows and rows[l][key]
        ]
        if mp:
            ax.errorbar(
                *zip(*mp, strict=True),
                fmt=marker,
                mfc="none",
                color=color,
                capsize=3,
                label=lab.split(" (")[0] + ", LR matched (1M / 3M at 6e-4, 10M at 3e-4)",
            )
    ax.set_xscale("log")
    ax.set_xticks([rows[l]["params"] for l, _, _ in SIZES if l in rows])
    ax.set_xticklabels([l for l, _, _ in SIZES if l in rows])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("model size (parameters); filled = 1/width LR rule")
    ax.set_ylabel("k0 name effect N − K, ALL (nats)")
    ax.set_title("(a) the persona across five sizes", fontsize=10)
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for key, color, marker, lab in (
        ("by1", "C3", "s", "bystander: kind actor with a naughty other, y = 1"),
        ("by95", "C3", "o", "bystander, y = 0.95"),
        ("hs1", "C4", "s", "never-seen name − kind, y = 1"),
        ("hs95", "C4", "o", "never-seen, y = 0.95"),
    ):
        pts = [
            (rows[l]["params"], *ms(rows[l][key])[:2])
            for l, _, _ in SIZES
            if l in rows and rows[l][key]
        ]
        if pts:
            ax.errorbar(
                *zip(*pts, strict=True),
                fmt=marker + ("-" if key.endswith("1") else "--"),
                color=color,
                capsize=3,
                label=lab,
                mfc="none" if key.endswith("95") else color,
            )
    ax.set_xscale("log")
    ax.set_xticks([rows[l]["params"] for l, _, _ in SIZES if l in rows])
    ax.set_xticklabels([l for l, _, _ in SIZES if l in rows])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("model size (parameters)")
    ax.set_ylabel("nats relative to a kind actor with a kind other")
    ax.set_title(
        "(b) is the persona actor-bound? (bystander ≈ 0) and the unfamiliar-name default",
        fontsize=10,
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    labels = [l for l, y in story_rows if y == 1.0]
    keys = [
        ("char", "character (acts in the story)"),
        ("name", "name at equal acts"),
        ("alone", "name alone, no acts"),
        ("byst", "bystander"),
    ]
    w = 0.2
    for i, (k, lab) in enumerate(keys):
        m = [ms(story_rows[(l, 1.0)][k]) for l in labels]
        ax.bar(
            np.arange(len(labels)) + (i - 1.5) * w,
            [x[0] for x in m],
            w,
            yerr=[x[1] for x in m],
            capsize=2,
            label=lab,
        )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("nats")
    ax.set_title(
        "(c) in-story probe at y = 1: what the story does vs what the name does", fontsize=10
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    if ft_rows:
        x = np.array([r[1] for r in ft_rows])
        s = np.array([r[3] for r in ft_rows])
        small = np.array(
            [r[0].startswith("story-1m") or r[0].startswith("story-3m") for r in ft_rows]
        )
        ax.scatter(
            x[~small],
            s[~small],
            s=14,
            color="C0",
            alpha=0.5,
            label="10M / 30M / 100M runs (edge session)",
        )
        for stem, color, lab in (("story-1m_", "C2", "1M"), ("story-3m_", "C1", "3M")):
            sel = np.array([r[0].startswith(stem) for r in ft_rows])
            if sel.any():
                ax.scatter(
                    x[sel],
                    s[sel],
                    s=40,
                    color=color,
                    marker="D",
                    label=f"{lab} exact-cast runs",
                    zorder=3,
                )
        xx = np.array([0, max(x.max(), 4)])
        ax.plot(xx, 0.30 * xx - 0.04, "k--", lw=0.8, label="Tier 0 fit: 0.30·x − 0.04")
        ax.legend(fontsize=7)
    ax.set_xlabel("base N − K, ALL (nats)")
    ax.set_ylabel("spillover of 'Ben steals' to lie / cheat / tease (nats)")
    ax.set_title("(d) does the spillover fraction hold at 1M / 3M?", fontsize=10)
    ax.grid(alpha=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
