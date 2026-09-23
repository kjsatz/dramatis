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
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dramatis.runids import is_probe_result, read_probe_json, step_rank, warn

ACTS = ["steal", "lie", "cheat", "tease"]
# The finetune and base-run tags in canonical form (dramatis.runids), v3 and v4 corpora alike.
# collect() parses tags with dramatis.runids; these stay for the scripts that import them.
FT = re.compile(
    r"^(?:mac-)?ft-(?P<act>[a-z]+)-(?P<cast>C[a-z]+)-(?P<run>[a-z][a-z0-9-]*_v\d+-[A-Za-z0-9-]+-s\d+)/ckpt_000099\.pt$"
)
RUN = re.compile(
    r"^(?P<run>[a-z][a-z0-9-]*_v\d+-[A-Za-z0-9-]+-s\d+)/(?:ckpt_(?P<step>\d+)|final_weights)\.pt$"
)
USED = ("Ben (kind, ft)", "kind")  # the classes the finetune deltas use
BASE_KEYS = [f"{c}|{a}" for c in ("naughty", "kind") for a in ["ALL", *ACTS]]


def comparable(a, b):
    """Were a finetune entry and its neutral scored on the same probe items? New probe files
    carry per-class item hashes in "_meta"; old ones drew every item from one random stream
    shared by all classes, so their items match only if the same classes were probed (the
    --per-name "kind:<Name>" classes and the finetune-name singletons both change every item)."""
    if a["hashes"] is not None and b["hashes"] is not None:
        return all(a["hashes"].get(c) is not None and a["hashes"].get(c) == b["hashes"].get(c) for c in USED)
    return a["classes"] == b["classes"]


def pick_pair(fts, neutrals):
    """(finetune, neutral) entries for one base and act, or None if no finetune entry has a
    comparable neutral. Finetunes: box runs before Mac ("mac-") runs, then the latest file first.
    Neutrals: from the same file and machine first, then the same file, then the latest."""
    for f in sorted(reversed(fts), key=lambda f: f["mac"]):  # stable: latest first within
        for n in sorted(reversed(neutrals), key=lambda n: (n["path"] != f["path"], n["mac"] != f["mac"])):
            if comparable(f, n):
                return f, n
    return None


def collect_pairs():
    """(base, pairs): base = run -> the base model's k = 0 means (latest checkpoint); pairs = run ->
    act -> (finetune means, neutral means) for every "Ben <act>" finetune under logs/cast/ that has
    a neutral scored on the same probe items. Pairs that have none are skipped with a warning."""
    fts, base = defaultdict(lambda: defaultdict(list)), {}
    for p in sorted(glob.glob("logs/cast/**/*.json", recursive=True)):
        try:
            entries, meta = read_probe_json(p)
        except ValueError:  # not JSON
            continue
        hashes = meta.get("items_sha256") if meta else None
        for rid, tag, v in entries:
            if not is_probe_result(v):
                continue
            means = {k: float(np.mean(x)) for k, x in v.items()}
            if rid.act is not None:  # a finetune: the 100-step final checkpoint only
                if rid.steps is None and rid.ft_final and (
                    (rid.act in ACTS and rid.cast == "Ckind") or (rid.act == "neutral" and rid.cast == "Ccast")
                ):
                    classes = frozenset(k.rsplit("|", 1)[0] for k in v)
                    fts[rid.base][rid.act].append(dict(path=p, tag=tag, mac=rid.mac, means=means, hashes=hashes, classes=classes))
                continue
            if not all(k in v for k in BASE_KEYS):  # e.g. the held-out-act probe: not a base
                continue
            rank = step_rank(rid)
            if rid.base not in base or base[rid.base][0] < rank:
                base[rid.base] = (rank, means)
    pairs, excluded = defaultdict(dict), []
    for run, sets in sorted(fts.items()):
        if "neutral" not in sets or run not in base:
            continue
        for act in ACTS:
            if act not in sets:
                continue
            pair = pick_pair(sets[act], sets["neutral"])
            if pair is None:
                excluded.append((sets[act][-1], sets["neutral"][-1]))
                continue
            pairs[run][act] = (pair[0]["means"], pair[1]["means"])
    for f, n in excluded:
        warn(
            f"spillover_by_act: excluded {f['tag']} ({f['path']}) vs {n['tag']} ({n['path']}): "
            "the finetune and its neutral were scored on different probe items"
        )
    return {r: v[1] for r, v in base.items()}, pairs


def collect():
    """act -> [(run, base N − K, attachment, spillover)] over every base model with a neutral and
    a "Ben <act>" finetune probe under logs/cast/ scored on the same items."""
    base, pairs = collect_pairs()
    rows = defaultdict(list)  # act -> (run, base N−K, attachment, spillover)
    for run, by_act in sorted(pairs.items()):
        b = base[run]
        nk = b["naughty|ALL"] - b["kind|ALL"]
        for act, (ft, ne) in by_act.items():
            dl = {a: {c: ft[f"{c}|{a}"] - ne[f"{c}|{a}"] for c in USED} for a in ACTS}
            attach = dl[act]["Ben (kind, ft)"] - dl[act]["kind"]
            spill = np.mean([dl[a]["Ben (kind, ft)"] - dl[a]["kind"] for a in ACTS if a != act])
            rows[act].append((run, nk, attach, spill))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/spillover_by_act"))
    parser.add_argument("--dump", action="store_true", help="print every run")
    args = parser.parse_args()
    rows = collect()
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
    # ---- re-fit (2026-09-21): intercepts with bootstrap CIs, through-origin slopes, the fraction
    # above +2 nats as a distribution, a pooled fit, and the steal line by model size.
    rng = np.random.default_rng(0)

    def fits(x, s):
        A = np.vstack([x, np.ones_like(x)]).T
        slope, icpt = np.linalg.lstsq(A, s, rcond=None)[0]
        boots = []
        for _ in range(2000):
            i = rng.integers(len(x), size=len(x))
            boots.append(np.linalg.lstsq(A[i], s[i], rcond=None)[0][1])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        origin = float(x @ s / (x @ x))
        big = x > 2
        frac = s[big] / x[big] if big.any() else np.array([np.nan])
        return slope, icpt, lo, hi, origin, np.corrcoef(x, s)[0, 1], np.median(frac), np.percentile(frac, 25), np.percentile(frac, 75), int(big.sum())

    print("\nre-fit: OLS slope and intercept [95 % bootstrap CI], through-origin slope, r; the fraction spill / persona above +2 nats as median [IQR]")
    print(f"{'set':22s} {'n':>4s} {'slope':>6s} {'intercept [CI]':>22s} {'origin':>7s} {'r':>5s} {'fraction > 2: median [IQR]':>28s} {'n>2':>4s}")
    allx, alls = [], []
    for act in ACTS:
        v = rows[act]
        if len(v) < 3:
            continue
        x, s = np.array([r[1] for r in v]), np.array([r[3] for r in v])
        allx += list(x)
        alls += list(s)
        f = fits(x, s)
        print(f"{'Ben ' + act + 's':22s} {len(v):4d} {f[0]:6.2f} {f[1]:+7.2f} [{f[2]:+.2f}, {f[3]:+.2f}] {f[4]:7.2f} {f[5]:5.2f} {f[6]:14.2f} [{f[7]:.2f}, {f[8]:.2f}] {f[9]:4d}")
    x, s = np.array(allx), np.array(alls)
    f = fits(x, s)
    print(f"{'all four acts pooled':22s} {len(x):4d} {f[0]:6.2f} {f[1]:+7.2f} [{f[2]:+.2f}, {f[3]:+.2f}] {f[4]:7.2f} {f[5]:5.2f} {f[6]:14.2f} [{f[7]:.2f}, {f[8]:.2f}] {f[9]:4d}")
    for size in ("1m", "3m", "10m", "30m", "100m"):
        v = [r for r in rows["steal"] if re.search(rf"story-{size}(?:-|_)", r[0])]
        if len(v) < 3:
            continue
        x, s = np.array([r[1] for r in v]), np.array([r[3] for r in v])
        f = fits(x, s)
        print(f"{'Ben steals, ' + size:22s} {len(v):4d} {f[0]:6.2f} {f[1]:+7.2f} [{f[2]:+.2f}, {f[3]:+.2f}] {f[4]:7.2f} {f[5]:5.2f} {f[6]:14.2f} [{f[7]:.2f}, {f[8]:.2f}] {f[9]:4d}")
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
