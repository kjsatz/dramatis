# ruff: noqa: E501
"""Readout of the LLM-judged samples (scripts/sample_forks.py + judge_samples.py): bad-act rates
by name class and act for the persona block, attachment and spillover in rate terms for the
finetune block, and the consistency of sampled rates with the fork log-odds.

    uv run python scripts/samples_summary.py --out docs/figures/samples
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dramatis.runids import parse, read_probe_json, step_rank

ACTS = ["steal", "lie", "cheat", "tease", "break"]
CLS_TO_FORK = {"N": "naughty", "K": "kind", "H": "heldout", "Ben": "Ben (kind, ft)"}


def load(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []


def rate(rows, key="BAD"):
    n = len(rows)
    p = sum(r["verdict"] == key for r in rows) / n if n else np.nan
    se = np.sqrt(p * (1 - p) / n) if n else np.nan
    return p, se, n


def wilson(rows, key="BAD"):
    n = len(rows)
    if not n:
        return np.nan, np.nan, 0
    p = sum(r["verdict"] == key for r in rows) / n
    z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h, n


def decided(rows):
    """BAD / (BAD + BENIGN): the BAD rate among the continuations the judge could decide."""
    n = sum(r["verdict"] in ("BAD", "BENIGN") for r in rows)
    return sum(r["verdict"] == "BAD" for r in rows) / n if n else np.nan


def rates(rows):
    """"BAD x% [Wilson CI] · UNCLEAR y% · BAD/(BAD+BENIGN) z%" for one table cell."""
    p, _, n = rate(rows)
    lo, hi, _ = wilson(rows)
    return f"{p:6.1%} [{lo:.0%}–{hi:.0%}] n={n:3d} · {rate(rows, 'UNCLEAR')[0]:4.0%} · {decided(rows):6.1%}"


def run_id(model):
    """The canonical run name (dramatis.runids) of a sample's model tag."""
    rid = parse(model)
    if rid is None:
        raise ValueError(f"sample model tag in no known format: {model!r}")
    return rid.run


def fork_logodds():
    """(canonical run name, fork class, act) -> mean log-odds, from the k0 probe JSONs (base and
    finetuned) at each run's last probed checkpoint (finetunes: their final checkpoint only)."""
    out, rank = {}, {}
    for p in ("logs/cast/k0_base.json", "logs/cast/k0_ft_all.json", "logs/cast/heldout_k0_base.json", "logs/cast/heldout_k0_ft.json"):
        if not Path(p).exists():
            continue
        for rid, _, d in read_probe_json(p)[0]:
            if rid.act is not None and not rid.ft_final:
                continue
            r = step_rank(rid)
            for k, v in d.items():
                cls, act = k.rsplit("|", 1)
                key = (rid.run, cls, act)
                if key not in rank or rank[key] <= r:
                    out[key], rank[key] = float(np.mean(v)), r
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--persona", default="logs/samples/persona.judge.jsonl")
    parser.add_argument("--finetune", nargs="+", default=["logs/samples/finetune.judge.jsonl", "logs/samples/finetune_cheat.judge.jsonl"])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/samples"))
    args = parser.parse_args()
    P, Fr = load(args.persona), [r for f in args.finetune for r in load(f)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)

    # ---- (a) persona: bad-act rate by class and act, pooled over the three seeds
    print("(a) persona block, pooled over models: BAD / all verdicts [Wilson 95 % CI] · UNCLEAR / all · BAD / (BAD + BENIGN)")
    print(f"{'act':6s}" + "".join(f"{c:>46s}" for c in ("N-names", "K-names", "never-seen")))
    ax = axes[0]
    w = 0.25
    bars = {cls: ([], [], [], []) for cls in ("N", "K", "H")}
    for i, act in enumerate(ACTS):
        line = f"{act:6s}"
        for j, cls in enumerate(("N", "K", "H")):
            rows = [r for r in P if r["cls"] == cls and r["act"] == act]
            if not rows:
                line += f"{'—':>46s}"
                continue
            p = rate(rows)[0]
            l, h, _ = wilson(rows)
            xs, ys, lo, hi = bars[cls]
            xs.append(i + (j - 1) * w)
            ys.append(p)
            lo.append(p - l)
            hi.append(h - p)
            line += rates(rows).rjust(46)
        print(line)
    for cls, (xs, ys, lo, hi) in bars.items():
        ax.bar(xs, ys, width=w, yerr=[lo, hi], capsize=2, label={"N": "N-names (always villains)", "K": "K-names (never)", "H": "never-seen names"}[cls])
    ax.set_xticks(range(len(ACTS)))
    ax.set_xticklabels(ACTS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of samples judged BAD")
    ax.set_title("(a) the lexical persona in sampled stories (plain cast, 3 seeds)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    print("   per model, N and K on steal: " + "; ".join(f"{m.split('_')[-1]}: N {rate([r for r in P if r['model'] == m and r['cls'] == 'N' and r['act'] == 'steal'])[0]:.0%} K {rate([r for r in P if r['model'] == m and r['cls'] == 'K' and r['act'] == 'steal'])[0]:.0%}" for m in sorted({r["model"] for r in P})))
    print("   UNCLEAR overall: " + f"{rate(P, 'UNCLEAR')[0]:.1%}" + f"; BAD / (BAD + BENIGN) overall {decided(P):.1%}")

    # ---- (b) finetune: attachment and spillover in rate terms
    if Fr:
        print("\n(b) finetune block, pooled over seeds: neutral → finetune for the taught name (Ben) and the other K-names;")
        print("    BAD / all verdicts, UNCLEAR / all, BAD / (BAD + BENIGN); 'Ben − K change' is in BAD / all")
        sets = defaultdict(list)  # (finetune set, base cell, seed) -> rows, by canonical run id
        for r in Fr:
            rid = parse(r["model"])
            if rid is None or rid.act is None:
                raise ValueError(f"finetune sample model tag in no known format: {r['model']!r}")
            sets[(rid.ft, rid.cell, rid.seed)].append(r)
        rows_out = []
        ax = axes[1]
        cells = sorted({c for (_, c, _) in sets})
        for base_cell in cells:  # every base cell, in a fixed order
            seeds = sorted({s for (_, c, s) in sets if c == base_cell})
            if len(cells) > 1:
                print(f"   {base_cell} (seeds {seeds})")
            for setname, label in (("ft-steal-Ckind", "Ben steals"), ("ft-lie-Ckind", "Ben lies"), ("ft-cheat-Ckind", "Ben cheats"), ("ft-tease-Ckind", "Ben teases")):
                for act in ACTS:
                    vals = {}
                    for cls in ("Ben", "K"):
                        ft = [r for s in seeds for r in sets.get((setname, base_cell, s), []) if r["cls"] == cls and r["act"] == act]
                        ne = [r for s in seeds for r in sets.get(("ft-neutral-Ccast", base_cell, s), []) if r["cls"] == cls and r["act"] == act]
                        vals[cls] = (rate(ft)[0], rate(ne)[0], len(ft), len(ne), rate(ft, "UNCLEAR")[0], rate(ne, "UNCLEAR")[0], decided(ft), decided(ne))
                    if not vals["Ben"][2]:
                        continue
                    d_ben, d_k = vals["Ben"][0] - vals["Ben"][1], vals["K"][0] - vals["K"][1]
                    rows_out.append((label, act, vals["Ben"][1], vals["Ben"][0], vals["K"][1], vals["K"][0], d_ben - d_k))

                    def cell(v):
                        return f"BAD {v[1]:5.1%} → {v[0]:5.1%}  UNCLEAR {v[5]:4.0%} → {v[4]:4.0%}  BAD/decided {v[7]:5.1%} → {v[6]:5.1%} (n={v[3]},{v[2]})"

                    print(f"   {label:12s} {act:6s}  Ben {cell(vals['Ben'])}   other K {cell(vals['K'])}   Ben − K change {d_ben - d_k:+.1%}")
        sel = [r for r in rows_out if r[1] == "steal" or r[0] == "Ben steals"]
        x = np.arange(len(sel))
        ax.bar(x - 0.15, [r[3] - r[2] for r in sel], width=0.3, label="Ben: finetune − neutral")
        ax.bar(x + 0.15, [r[5] - r[4] for r in sel], width=0.3, label="other K-names: finetune − neutral")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r[0]}\n→ {r[1]}" for r in sel], fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("change in BAD rate")
        ax.set_title("(b) attachment and spillover in sampled stories (3 seeds)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")

    # ---- (c) consistency: sampled rate vs fork log-odds per (model, class, act)
    fl = fork_logodds()
    pts, missing = [], 0
    for r_set in (P, Fr):
        cells = defaultdict(list)
        for r in r_set:
            cells[(run_id(r["model"]), r["cls"], r["act"])].append(r)
        for (m, cls, act), rows in sorted(cells.items()):
            fc = CLS_TO_FORK.get(cls)
            if not fc:
                continue
            if (m, fc, act) not in fl:
                missing += 1
                continue
            pts.append((fl[(m, fc, act)], rate(rows)[0], cls, act, decided(rows)))
    ax = axes[2]
    if pts:
        xs = np.linspace(-12, 14, 200)
        ax.plot(xs, 1 / (1 + np.exp(-xs)), "k--", lw=0.8, label="σ(fork log-odds)")
        for act, marker in zip(ACTS, "osD^v", strict=True):
            sub = [p for p in pts if p[3] == act]
            if sub:
                ax.scatter([p[0] for p in sub], [p[1] for p in sub], marker=marker, s=22, alpha=0.75, label=act)
        def ranks(v):
            """Ranks with ties averaged (many sampled rates tie at 0 %), so ρ does not depend on
            the order of the cells."""
            v = np.asarray(v, float)
            r = np.empty(len(v))
            r[np.argsort(v, kind="stable")] = np.arange(len(v))
            for x in np.unique(v):
                r[v == x] = r[v == x].mean()
            return r
        rho = float(np.corrcoef(ranks([p[0] for p in pts]), ranks([p[1] for p in pts]))[0, 1])
        dec = [p for p in pts if not np.isnan(p[4])]
        rho_d = float(np.corrcoef(ranks([p[0] for p in dec]), ranks([p[4] for p in dec]))[0, 1])
        ax.set_title(f"(c) sampled BAD rate vs fork log-odds, per (model, class, act): Spearman ρ = {rho:.2f}, n = {len(pts)}", fontsize=8)
        print(f"\n(c) consistency: Spearman ρ = {rho:.2f} over {len(pts)} cells (BAD / all verdicts); {rho_d:.2f} over {len(dec)} cells with BAD / (BAD + BENIGN)")
    if missing:
        print(f"   WARNING: {missing} sampled (model, class, act) cells have no fork log-odds and are left out")
    ax.set_xlabel("fork log-odds (teacher-forced, two fixed endings)")
    ax.set_ylabel("sampled BAD rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
