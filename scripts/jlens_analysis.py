# ruff: noqa: E501
"""The J-lens Stages 1–3a from the box outputs (docs/2026-09-27_PLAN_jlens.md): the FROZEN band rule
applied to every measured model, summary tables and figures.

    uv run python scripts/jlens_analysis.py [--box logs/v4/jlens/box] [--out docs/figures/jlens]

Reads <box>/stats/*.json (and <box>/profile/*.json); writes <box>/summary.json and
  scale.png        Stage 1: per size, the fraction of seeds with a band (y = 1 and the y = 0 control), the band's
                   extent, its peakedness, capacity, and class content (J-lens against logit lens)
  scale_blocks.png Stage 1: peakedness and next-token agreement against relative depth, every size and seed
  maturation.png   Stage 2: the same summaries at 41M / 82M / 164M / 409M tokens (seeds 0-1)
  formation.png    Stage 2: dense ladder-C 30M against updates, with the in-context switch marked
  finetune.png     Stage 3a: what "Ben steals" writes onto Ben in the base lens, against size, and over the
                   tl2 finetune and cleaning steps
"""

import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EX_MIN, NT_MAX = 0.10, 0.5  # the frozen band rule (plan, frozen at 43d9948)
SIZES = ["L1", "L2", "L3", "L4", "1M", "3M", "10M", "30M", "100M"]
LABEL = {"L1": "0.2M", "L2": "0.4M", "L3": "0.6M", "L4": "0.8M", "1M": "1M", "3M": "3M", "10M": "10M", "30M": "30M", "100M": "100M"}
SIZE_COLOR = {s: plt.cm.viridis(i / 8.5) for i, s in enumerate(SIZES)}
SWITCH = {0: 7920, 1: 7590, 2: 14560, 3: None}
TOK_PER_UPDATE = 32 * 512
FOOT = "scripts/jlens_analysis.py · band rule frozen at 43d9948 · exploratory"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def blocks_series(rec, lens, key):
    fin = rec["blocks"][-1][lens]["next_token_top1"]
    out = []
    for b in rec["blocks"]:
        x = b[lens]
        if key == "ex":
            out.append(x["kurtosis"]["centered"]["exceedance"])
        elif key == "nt":
            out.append(x["next_token_top1"] / fin if fin else float("nan"))
        elif key == "content":
            c = x.get("class_profile")
            out.append((c["norm"] - c["null_mean"]) / max(c["null_sd"], 1e-9) if c else float("nan"))
        elif key == "active":
            out.append(x["active_tokens"]["mean"])
    return np.array(out)


def band(rec):
    """The frozen rule: blocks with exceedance >= 0.10 and next-token agreement < 0.5 x final; the longest
    contiguous run (others listed)."""
    ex, nt = blocks_series(rec, "jlens", "ex"), blocks_series(rec, "jlens", "nt")
    ok = [i for i in range(len(ex) - 1) if ex[i] >= EX_MIN and nt[i] < NT_MAX]  # the final block is the output
    runs, cur = [], []
    for i in ok:
        if cur and i != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    return max(runs, key=len) if runs else [], ok


def parse(path):
    m = re.match(r"(v4-.+?)@(\d+)(.*)\.json$", Path(path).name)
    run, step, tag = m.group(1), int(m.group(2)), m.group(3)
    info = {"run": run, "step": step, "tag": tag, "null": tag.lstrip("-") or None}
    mm = re.match(r"v4-(lex-y\d+|ladder-C)-(\w+?)-s(\d+)(-dense)?$", run)
    info.update(corpus=mm.group(1), size=mm.group(2), seed=int(mm.group(3)), dense=bool(mm.group(4)))
    return info


def summarize(rec):
    b, ok = band(rec)
    L = len(rec["blocks"]) - 1
    ex, act = blocks_series(rec, "jlens", "ex"), blocks_series(rec, "jlens", "active")
    cj, cl = blocks_series(rec, "jlens", "content"), blocks_series(rec, "logit", "content")
    nt = blocks_series(rec, "jlens", "nt")
    inner = [i for i in range(L)]  # every non-output block, for the continuous companions
    motor_ok = [i for i in inner if nt[i] < NT_MAX]
    return {"has_band": bool(b), "band": b, "qualifying": ok, "L": L,
            "band_lo": b[0] / L if b else None, "band_hi": b[-1] / L if b else None, "band_frac": len(b) / L if b else 0.0,
            "peak_ex_nonmotor": float(max(ex[i] for i in motor_ok)) if motor_ok else float("nan"),
            "ex_band": float(np.mean(ex[b])) if b else float("nan"),
            "active_band": float(np.mean(act[b])) if b else float("nan"),
            "content_j_band": float(np.nanmax(cj[b])) if b else float("nan"),
            "content_l_band": float(np.nanmax(cl[b])) if b else float("nan"),
            "content_j_max": float(np.nanmax(cj[1:L])) if L > 1 else float("nan"),
            "content_l_max": float(np.nanmax(cl[1:L])) if L > 1 else float("nan")}


def load(box):
    out = []
    for f in sorted(glob.glob(str(Path(box) / "stats" / "*.json"))):
        info = parse(f)
        rec = json.load(open(f))
        info.update(summarize(rec))
        info["rec"] = rec
        out.append(info)
    return out


def cell_table(rows):
    """Per (corpus, size, null, step class): k of n seeds with a band, the Wilson interval, and medians."""
    groups = defaultdict(list)
    for r in rows:
        if r["dense"]:
            continue
        final = r["step"] > 24000
        groups[(r["corpus"], r["size"], r["null"], "final" if final else r["step"])].append(r)
    table = []
    for key, rs in groups.items():
        k, n = sum(r["has_band"] for r in rs), len(rs)
        lo, hi = wilson(k, n)
        def med(f, rs=rs):
            vals = [r[f] for r in rs if r[f] is not None]
            return float(np.nanmedian(vals)) if vals else float("nan")

        table.append({"corpus": key[0], "size": key[1], "null": key[2], "step": key[3], "k": k, "n": n, "lo": lo, "hi": hi,
                      "has_band_cell": lo > 0.5, "band_lo": med("band_lo") if k else None, "band_hi": med("band_hi") if k else None,
                      "band_frac": med("band_frac"), "ex_band": med("ex_band"), "active_band": med("active_band"),
                      "content_j_band": med("content_j_band"), "content_l_band": med("content_l_band"),
                      "content_j_max": med("content_j_max"), "content_l_max": med("content_l_max"),
                      "peak_ex_nonmotor": med("peak_ex_nonmotor"), "seeds": sorted(r["seed"] for r in rs)})
    return table


def fig_scale(table, out):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.4))
    x = np.arange(len(SIZES))
    for corpus, mk, ls in (("lex-y100", "o", "-"), ("lex-y000", "s", "--")):
        rows = {t["size"]: t for t in table if t["corpus"] == corpus and t["null"] is None and t["step"] == "final"}
        xs = [i for i, s in enumerate(SIZES) if s in rows]
        def get(f, rows=rows, xs=xs):
            return [rows[SIZES[i]][f] for i in xs]

        name = "y = 1" if corpus == "lex-y100" else "y = 0 (control)"
        frac = [rows[SIZES[i]]["k"] / rows[SIZES[i]]["n"] for i in xs]
        err = np.array([[f - rows[SIZES[i]]["lo"], rows[SIZES[i]]["hi"] - f] for f, i in zip(frac, xs, strict=True)]).T
        axes[0].errorbar(xs, frac, yerr=err, fmt=mk + ls, capsize=3, label=name)
        for i in xs:
            t = rows[SIZES[i]]
            if t["k"]:
                axes[1].plot([i + (0.12 if corpus == "lex-y000" else 0)] * 2, [t["band_lo"], t["band_hi"]], "-", lw=5 if corpus == "lex-y100" else 2, color="C0" if corpus == "lex-y100" else "C1", alpha=0.7)
        axes[2].plot(xs, get("ex_band"), mk + ls, label=name)
        axes[3].plot(xs, get("active_band"), mk + ls, label=name)
        if corpus == "lex-y100":
            axes[4].plot(xs, get("content_j_band"), mk + "-", color="C0", label="J-lens (y = 1), max over band")
            axes[4].plot(xs, get("content_l_band"), mk + ":", color="C2", label="logit lens (y = 1)")
        else:
            axes[4].plot(xs, get("content_j_band"), mk + "--", color="C1", label="J-lens (y = 0 control)")
    for key, mk in (("untrained", "x"), ("randomized", "+")):
        rows = {t["size"]: t for t in table if t["null"] == key}
        xs = [i for i, s in enumerate(SIZES) if s in rows]
        axes[0].plot(xs, [rows[SIZES[i]]["k"] / rows[SIZES[i]]["n"] for i in xs], mk, color="0.4", ms=8, label=f"{key} null (seed 0)")
    axes[0].axhline(0.5, color="0.8", lw=0.8)
    titles = ["(a) fraction of seeds with a band\n(Wilson 95 %; a size has one if the lower bound > ½)",
              "(b) the band's extent in relative depth\n(median over seeds with one)",
              "(c) peakedness in the band\n(mean exceedance; chance 0.01)",
              "(d) capacity: active tokens per position in the band",
              "(e) class content at names in the band\n(max over band blocks, null sd; median over seeds)"]
    for ax, t in zip(axes, titles, strict=True):
        ax.set_xticks(x, [LABEL[s] for s in SIZES], fontsize=8)
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("size (0.2M–0.8M: 1–4 layers at width 128)")
    axes[1].set_ylim(0, 1)
    axes[4].axhline(2, color="0.8", lw=0.8, ls=":")
    axes[0].legend(fontsize=7)
    axes[4].legend(fontsize=7)
    fig.suptitle("Stage 1: does a J-space band exist, by size? (finals at one epoch, 409M tokens; y = 1: 8 seeds, y = 0: 4 seeds)", fontsize=11)
    fig.text(0.995, 0.005, FOOT, ha="right", va="bottom", fontsize=7, color="0.45", style="italic")
    fig.tight_layout()
    fig.savefig(out / "scale.png", dpi=130)
    plt.close(fig)


def fig_scale_blocks(rows, out):
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharey="row")
    for i, size in enumerate([s for s in SIZES if s not in ("L1",)]):
        ax = axes.flat[i]
        for r in rows:
            if r["size"] != size or r["dense"] or r["step"] < 24000:
                continue
            rec = r["rec"]
            L = len(rec["blocks"]) - 1
            xd = np.arange(L + 1) / L
            if r["null"]:
                c, ls, lw, al = "0.4", ":" if r["null"] == "untrained" else "-.", 1.2, 0.9
            else:
                c, ls, lw, al = ("C0" if r["corpus"] == "lex-y100" else "C1"), "-", 0.9, 0.55
            ax.plot(xd, blocks_series(rec, "jlens", "ex"), ls, color=c, lw=lw, alpha=al)
            ax.plot(xd, blocks_series(rec, "jlens", "nt"), ls if r["null"] else "--", color=c, lw=lw * 0.8, alpha=al * 0.6)
        ax.axhline(EX_MIN, color="C3", lw=0.8, ls="--")
        ax.axhline(NT_MAX, color="C3", lw=0.8, ls=":")
        ax.set_title(LABEL[size], fontsize=10)
        ax.set_xlabel("block / L")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.05, 0.5, "solid: peakedness (exceedance)\nshort dashes: next-token ÷ final\nblue: y = 1 seeds; orange: y = 0\ngrey dotted: untrained; dash-dot: randomized\nred: the rule's thresholds (0.10, 0.5)\n(0.2M, one layer, has no inner block)", fontsize=9)
    fig.suptitle("Stage 1: the lens statistics block by block, every size and seed (finals)", fontsize=11)
    fig.text(0.995, 0.005, FOOT, ha="right", va="bottom", fontsize=7, color="0.45", style="italic")
    fig.tight_layout()
    fig.savefig(out / "scale_blocks.png", dpi=120)
    plt.close(fig)


def fig_maturation(rows, out):
    """Seeds 0-1 at every time point (the finals too), so the curves compare like with like."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for corpus, ls in (("lex-y100", "-"), ("lex-y000", "--")):
        for size in SIZES:
            rs = [r for r in rows if r["corpus"] == corpus and r["size"] == size and not r["null"] and not r["dense"] and r["seed"] in (0, 1)]
            steps = sorted({r["step"] for r in rs})
            if len(steps) < 2:
                continue
            xs = [st * TOK_PER_UPDATE / 1e6 for st in steps]
            at = [[r for r in rs if r["step"] == st] for st in steps]
            c = SIZE_COLOR[size]
            axes[0].plot(xs, [np.mean([r["has_band"] for r in a]) for a in at], ls, marker="o", ms=3, color=c, label=LABEL[size] if corpus == "lex-y100" else None)
            axes[1].plot(xs, [np.mean([r["peak_ex_nonmotor"] for r in a]) for a in at], ls, marker="o", ms=3, color=c)
            axes[2].plot(xs, [np.mean([r["band_frac"] for r in a]) for a in at], ls, marker="o", ms=3, color=c)
    titles = ["fraction of seeds 0-1 with a band", "peak peakedness among non-motor blocks", "band width (fraction of blocks)"]
    for ax, t in zip(axes, titles, strict=True):
        ax.set_xlabel("pretraining tokens (M)")
        ax.set_title(t, fontsize=10)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Stage 2: maturation over pretraining (seeds 0-1; solid y = 1, dashed y = 0)", fontsize=11)
    fig.text(0.995, 0.005, FOOT, ha="right", va="bottom", fontsize=7, color="0.45", style="italic")
    fig.tight_layout()
    fig.savefig(out / "maturation.png", dpi=130)
    plt.close(fig)


def probe_curve(seed):
    f = glob.glob(f"logs/incontext_dense/*_v4-ladder-C-30M-s{seed}_dense/probes.jsonl")
    if not f:
        return None, None
    rows = [json.loads(line) for line in open(f[0])]
    u = np.array([r["updates"] for r in rows])
    a = np.array([(r["incontext/character/ALL"] + r["incontext_k2/character/ALL"]) / 2 for r in rows])
    w = 51
    return u[w // 2:w // 2 + len(a) - w + 1], np.convolve(a, np.ones(w) / w, "valid")


def fig_formation(rows, out):
    dense = [r for r in rows if r["dense"]]
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for ax, seed in zip(axes, range(4), strict=True):
        rs = sorted([r for r in dense if r["seed"] == seed], key=lambda r: r["step"])
        if not rs:
            continue
        u = [r["step"] for r in rs]
        ax.plot(u, [r["peak_ex_nonmotor"] for r in rs], "-o", ms=2.5, color="C0", label="peak peakedness among non-motor blocks")
        ax.plot(u, [r["band_frac"] for r in rs], "-", color="C2", lw=1.2, label="band width (fraction of blocks)")
        ax.plot(u, [r["active_band"] if r["has_band"] else np.nan for r in rs], ".", color="C4", ms=3, label="active tokens in the band")
        pu, pv = probe_curve(seed)
        if pu is not None:
            ax2 = ax.twinx()
            ax2.plot(pu, pv, color="0.5", lw=1, label="in-context behaviour (rung C, k = 1 and 2, nats)")
            ax2.set_ylabel("behaviour (nats)", fontsize=8, color="0.5")
        if SWITCH[seed]:
            ax.axvline(SWITCH[seed], color="C3", ls="--", lw=1)
        ax.axhline(EX_MIN, color="C3", lw=0.6, ls=":")
        ax.set_title(f"seed {seed}: " + (f"switch at {SWITCH[seed]} updates" if SWITCH[seed] else "never switches"), fontsize=9)
        ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
    axes[0].legend(fontsize=7, loc="upper left")
    axes[-1].set_xlabel("pretraining updates (dense 30M ladder-C retrains; 1 update = 16,384 tokens)")
    fig.suptitle("Stage 2: formation of the J-space structure against the in-context switch (30M, names carry no class)", fontsize=11)
    fig.text(0.995, 0.003, FOOT, ha="right", va="bottom", fontsize=7, color="0.45", style="italic")
    fig.tight_layout()
    fig.savefig(out / "formation.png", dpi=120)
    plt.close(fig)


def ben_position(run):
    fs = glob.glob(f"hub-v4/box-*/logs/v4/eval/ft/{run}.json")
    if not fs:
        return None
    pairs = json.load(open(fs[0]))["pairs"]
    p = next(v for k, v in pairs.items() if k.startswith("ft-steal"))["position"]
    return {"name": p["name_last"] - p["name_K_last"], "site": p["site_last"] - p["site_K_last"]}


def fig_finetune(box, rows, out):
    profs = {Path(f).stem: json.load(open(f)) for f in glob.glob(str(Path(box) / "profile" / "*.json"))}
    if not profs:
        return
    bands = {(r["run"]): r["band"] for r in rows if not r["dense"] and r["step"] > 24000 and not r["null"]}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    xs, beta_b, beta_z, pos = [], [], [], []
    for name, p in profs.items():
        if name.endswith("-tl2"):
            continue
        m = re.match(r"v4-lex-y100-(\w+?)-s(\d)$", name)
        if not m:
            continue
        size = m.group(1)
        layers = p["pairs"][0]["layers"]
        b = bands.get(name) or list(range(1, len(layers) - 1)) or [0]  # the band's blocks (all inner blocks without one)
        xs.append(SIZES.index(size))
        beta_b.append(float(np.mean([layers[li]["no_names"]["beta"] for li in b])))
        beta_z.append(float(np.mean([(layers[li]["no_names"]["beta"] - layers[li]["no_names"]["null_mean"]) / max(layers[li]["no_names"]["null_sd"], 1e-9) for li in b])))
        pb = ben_position(name)
        pos.append(pb["name"] if pb else np.nan)
    P, B = np.array(pos, dtype=float), np.array(beta_b)
    ok = ~np.isnan(P)
    print(f"3a: {ok.sum()} models; β against Ben's position: slope through origin {P[ok] @ B[ok] / (P[ok] @ P[ok]):.2f}, r {np.corrcoef(P[ok], B[ok])[0, 1]:.2f}; median z {np.median(beta_z):.1f}")
    for i, s in enumerate(SIZES):
        sel = [k for k in range(len(xs)) if xs[k] == i]
        if sel:
            print(f"   {LABEL[s]:5s} β {np.median(B[sel]):.2f}  position {np.nanmedian(P[sel]):.2f}  z {np.median(np.array(beta_z)[sel]):.1f}")
    jit = np.random.default_rng(0).uniform(-0.15, 0.15, len(xs))
    axes[0].scatter(np.array(xs) + jit, beta_b, s=18, color="C0", label="β (no name tokens), mean over the band")
    axes[0].scatter(np.array(xs) + jit, pos, s=18, marker="x", color="C3", label="Ben's excess position at his name (A4, last block)")
    axes[0].set_xticks(range(len(SIZES)), [LABEL[s] for s in SIZES], fontsize=8)
    axes[0].set_title("(a) the fraction of the villain profile Ben acquires\n(\"Ben steals\", 100 steps), against size", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].scatter(pos, beta_b, s=18, c=[SIZE_COLOR[SIZES[i]] for i in xs])
    lim = [0, max(0.1, np.nanmax(pos + beta_b) * 1.1)]
    axes[1].plot(lim, lim, ":", color="k")
    axes[1].set_xlabel("Ben's excess position (A4, at his name)")
    axes[1].set_ylabel("β in the base lens")
    axes[1].set_title("(b) β against the measured position, per model\n(dotted: equal)", fontsize=9)
    for name, p in profs.items():
        if not name.endswith("-tl2"):
            continue
        m = re.match(r"v4-lex-y100-(\w+?)-s(\d)-tl2$", name)
        size = m.group(1)
        b = bands.get(name[:-4]) or [1]
        rows_t = []
        for pair in p["pairs"]:
            phase = "ft" if pair["steal"].startswith("tl2-ft") else "clean"
            step = int(pair["step"])
            beta = float(np.mean([pair["layers"][li]["no_names"]["beta"] for li in b]))
            rows_t.append((phase, step, beta))
        ft = [(s, v) for ph, s, v in rows_t if ph == "ft"]
        cl = [(s, v) for ph, s, v in rows_t if ph == "clean"]
        x = [s for s, _ in ft] + [100 + s for s, _ in cl]
        y = [v for _, v in ft] + [v for _, v in cl]
        axes[2].plot(x, y, "-o", ms=3, color=SIZE_COLOR[size], alpha=0.7, label=f"{LABEL[size]} s{m.group(2)}")
    axes[2].set_xscale("symlog", linthresh=10)
    axes[2].axvline(100, color="0.7", lw=0.8)
    axes[2].set_xlabel("finetune steps (≤ 100), then 100 + clean steps")
    axes[2].set_title("(c) over the finetune and through cleaning (tl2)\nβ, mean over the band", fontsize=9)
    axes[2].legend(fontsize=6, ncol=2)
    fig.suptitle("Stage 3a: what the finetune writes onto Ben, read in the base model's lens", fontsize=11)
    fig.text(0.995, 0.005, FOOT, ha="right", va="bottom", fontsize=7, color="0.45", style="italic")
    fig.tight_layout()
    fig.savefig(out / "finetune.png", dpi=130)
    plt.close(fig)


def main(box, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = load(box)
    table = cell_table(rows)
    summary = {"rule": {"ex_min": EX_MIN, "nt_max": NT_MAX, "frozen_at": "43d9948"},
               "cells": table,
               "models": [{k: v for k, v in r.items() if k != "rec"} for r in rows]}
    (Path(box) / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    for t in sorted(table, key=lambda t: (t["corpus"], str(t["null"]), SIZES.index(t["size"]) if t["size"] in SIZES else 99, str(t["step"]))):
        print(f"{t['corpus']:9s} {LABEL.get(t['size'], t['size']):5s} {t['null'] or ''!s:10s} {t['step']!s:6s} band {t['k']}/{t['n']} [{t['lo']:.2f},{t['hi']:.2f}]"
              f" extent {t['band_lo'] if t['band_lo'] is None else round(t['band_lo'], 2)}–{t['band_hi'] if t['band_hi'] is None else round(t['band_hi'], 2)}"
              f" ex {t['ex_band']:.2f} act {t['active_band']:.2f} content J {t['content_j_band']:+.1f} L {t['content_l_band']:+.1f}")
    fig_scale(table, out)
    fig_scale_blocks(rows, out)
    fig_maturation(rows, out)
    fig_formation(rows, out)
    fig_finetune(box, rows, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="logs/v4/jlens/box")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/jlens"))
    a = ap.parse_args()
    main(a.box, a.out)
