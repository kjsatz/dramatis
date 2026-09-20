# ruff: noqa: E501
"""Name-embedding geometry across runs (NEXT_SESSION.md Tier 0, item 3). For every run with a
`name_embeddings.pt` (the 44 pool names' rows of the tied embedding table): the direction
u = mean(N-names) − mean(K-names), the separation of the two classes along it (d′ = gap over the
pooled within-class sd), a chance level for d′ from random 8/24 splits of the same 32 rows, where
the 12 never-seen names sit on u (0 = K mean, 1 = N mean), and — for the finetuned checkpoints
of the plain cast — how far the taught name (Ben, Lucy, Tom, Ollie) moves along the *base* model's
u. Joined with each run's k = 0 name effect from the probe JSONs.

    uv run python scripts/name_geometry.py --out docs/figures/name_geometry
"""

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dramatis.assemble import load_cast

BASE_JSONS = [
    "logs/cast/map/map_k0_base.json",
    "logs/cast/sweep/sweep_k0_base.json",
    "logs/cast/sweep/extra_k0_big.json",
    "logs/cast/box/box_k0_base.json",
    "logs/cast/k0_base.json",
    *sorted(glob.glob("logs/cast/seeds/seeds_k0_final_*.json")),
]
RUN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_(?P<id>.+)$")
CELL = re.compile(r"^(?P<stem>story-[0-9a-z-]+)_(?P<corpus>v3-.+)-s(?P<seed>\d)$")
Y = re.compile(r"v3-map-x25-y(?P<y>\d{3})$")


def run_id(tag):
    return tag.split("/ckpt_")[0].replace("/", "_")


def load_k0():
    out = {}
    for p in BASE_JSONS:
        for tag, d in json.load(open(p)).items():
            step = int(tag.split("/ckpt_")[1].split(".")[0]) if "/ckpt_" in tag else 10**9
            rid = run_id(tag)
            if rid in out and out[rid][0] > step:
                continue
            out[rid] = (
                step,
                float(np.mean(d["naughty|ALL"])) - float(np.mean(d["kind|ALL"])),
                float(np.mean(d["naughty|steal"])) - float(np.mean(d["kind|steal"])),
            )
    return {k: v[1:] for k, v in out.items()}


def geometry(E, names, N, K, H, rng, n_null=200):
    """E: (44, d) rows. Returns dict of scalars."""
    idx = {n: i for i, n in enumerate(names)}
    iN, iK, iH = [idx[n] for n in N], [idx[n] for n in K], [idx[n] for n in H]
    X = E[iN + iK]
    X = X - X.mean(0, keepdims=True)
    En, Ek = X[: len(iN)], X[len(iN) :]

    def dprime(A, B):
        u = A.mean(0) - B.mean(0)
        u = u / (np.linalg.norm(u) + 1e-12)
        pa, pb = A @ u, B @ u
        return (pa.mean() - pb.mean()) / np.sqrt((pa.var(ddof=1) + pb.var(ddof=1)) / 2), u

    d, u = dprime(En, Ek)
    null = []
    for _ in range(n_null):
        perm = rng.permutation(len(X))
        null.append(dprime(X[perm[: len(iN)]], X[perm[len(iN) :]])[0])
    null = np.array(null)
    ph = (E[iH] - E[iN + iK].mean(0)) @ u
    pk, pn = Ek @ u, En @ u
    held = ph.mean() / (pn.mean() - pk.mean())  # 0 = the 32-name centroid, in units of the K→N gap
    cos = lambda A, B: float(
        np.mean(
            (A / np.linalg.norm(A, axis=1, keepdims=True))
            @ (B / np.linalg.norm(B, axis=1, keepdims=True)).T
        )
    )  # noqa: E731
    return {
        "dprime": float(d),
        "dprime_null": float(null.mean()),
        "dprime_z": float((d - null.mean()) / null.std()),
        "gap_norm": float((pn.mean() - pk.mean()) / np.mean(np.linalg.norm(X, axis=1))),
        "heldout_pos": float(held),
        "cos_NN": cos(En, En),
        "cos_KK": cos(Ek, Ek),
        "cos_NK": cos(En, Ek),
        "u": u,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/name_geometry"))
    args = parser.parse_args()
    cast = load_cast()
    N, K, H = cast["naughty"], cast["kind"], cast["heldout"]
    k0 = load_k0()
    rng = np.random.default_rng(0)
    rows, ft_rows, base_u = [], [], {}
    files = sorted(glob.glob("runs/*/name_embeddings.pt"))
    for f in files:
        rid = RUN.match(Path(f).parent.name)["id"]
        e = torch.load(f, map_location="cpu", weights_only=False)
        E = e["wte"].float().numpy()
        if rid.startswith("ft-"):
            ft_rows.append((rid, E, e["names"]))
            continue
        g = geometry(E, e["names"], N, K, H, rng)
        base_u[rid] = (g["u"], E, e["names"])
        m = CELL.match(rid)
        stem, corpus, seed = (m["stem"], m["corpus"], int(m["seed"])) if m else (rid, "", -1)
        ym = Y.search(corpus)
        y = int(ym["y"]) / 100 if ym else np.nan
        eff = k0.get(rid, (np.nan, np.nan))
        rows.append(
            dict(
                run=rid,
                stem=stem,
                corpus=corpus,
                seed=seed,
                y=y,
                k0=eff[0],
                k0_steal=eff[1],
                **{k: v for k, v in g.items() if k != "u"},
            )
        )

    # ---- table by cell
    print(
        f"{'cell':46s} {'n':>2s} {'k0 N−K':>8s} {'d′':>6s} {'null':>5s} {'z':>6s} {'gap/‖e‖':>8s} {'held':>6s} {'cosNN':>6s} {'cosKK':>6s} {'cosNK':>6s}"
    )
    cells = sorted({(r["stem"], r["corpus"]) for r in rows}, key=lambda c: (c[1], c[0]))
    for stem, corpus in cells:
        rs = [r for r in rows if r["stem"] == stem and r["corpus"] == corpus]
        mean = lambda k, rs=rs: np.nanmean([r[k] for r in rs])  # noqa: E731
        print(
            f"{stem + '/' + corpus:46s} {len(rs):2d} {mean('k0'):+8.2f} {mean('dprime'):6.2f} {mean('dprime_null'):5.2f} {mean('dprime_z'):6.1f} {mean('gap_norm'):8.3f} {mean('heldout_pos'):6.2f} {mean('cos_NN'):6.2f} {mean('cos_KK'):6.2f} {mean('cos_NK'):6.2f}"
        )

    # ---- finetune drift along the base model's u (plain cast, Mac finetunes)
    NAME_OF = {"Ckind": "Ben", "Clucy": "Lucy", "Ctom": "Tom", "Cnew": "Ollie", "Ccast": None}
    print(
        "\nFinetune drift of the taught name's row (Mac finetunes of the plain cast), measured against the base model:"
    )
    print(
        "  pos = projection on the base N−K direction u (0 = base K mean, 1 = base N mean); |Δ| = the row's total"
    )
    print(
        "  movement in units of the base K→N gap; cos = cosine of that movement with u. Neutral rows give the baseline."
    )
    print(
        f"{'finetune':58s} {'name':>5s} {'pos before':>10s} {'pos after':>9s} {'|Δ|/gap':>8s} {'cos(Δ,u)':>8s} {'other K |Δ|/gap':>15s} {'other K pos':>11s}"
    )
    FT = re.compile(
        r"^ft-(?P<act>[a-z]+)-(?P<cast>C[a-z]+)(?:-(?P<steps>\d+))?-(?P<base>story-.+-s\d)$"
    )
    for rid, E, names in sorted(ft_rows, key=lambda t: (t[0].split("-story-")[1], t[0])):
        m = FT.match(rid)
        if not m or m["base"] not in base_u or m["cast"] not in NAME_OF:
            continue
        u, E0, _names0 = base_u[m["base"]]
        idx = {n: i for i, n in enumerate(names)}
        c = E0[[idx[n] for n in N + K]].mean(0)
        pos = lambda M, ns, idx=idx, c=c, u=u: float(np.mean((M[[idx[n] for n in ns]] - c) @ u))  # noqa: E731
        k0m, n0m = pos(E0, K), pos(E0, N)
        gap = n0m - k0m
        scale = lambda v, k0m=k0m, gap=gap: (v - k0m) / gap  # noqa: E731
        taught = (
            NAME_OF[m["cast"]] or "Ben"
        )  # for the neutral set, report Ben's row as the baseline
        others = [n for n in K if n != taught]
        dv = E[idx[taught]] - E0[idx[taught]]
        do = np.mean([np.linalg.norm(E[idx[n]] - E0[idx[n]]) for n in others])
        cos = float(dv @ u / (np.linalg.norm(dv) + 1e-12))
        print(
            f"{rid:58s} {taught:>5s} {scale(pos(E0, [taught])):+10.2f} {scale(pos(E, [taught])):+9.2f} {np.linalg.norm(dv) / gap:8.3f} {cos:+8.2f} {do / gap:15.3f} {scale(pos(E, others)):+11.2f}"
        )

    # ---- figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    colors = {
        "story-10m-sweep": "C0",
        "story-10m": "C4",
        "story-10m-3x": "C3",
        "story-30m": "C1",
        "story-100m": "C2",
    }
    ax = axes[0]
    for stem, color in colors.items():
        pts = [(r["y"], r["dprime"]) for r in rows if r["stem"] == stem and not np.isnan(r["y"])]
        if pts:
            ax.scatter(*zip(*pts, strict=True), s=16, color=color, alpha=0.7, label=stem[6:])
    nulls = [r["dprime_null"] for r in rows if not np.isnan(r["y"])]
    ax.axhline(np.mean(nulls), color="k", ls="--", lw=0.8, label="random 8/24 split")
    ax.set_xlabel("coherence y (x = 0.25)")
    ax.set_ylabel("d′ of N vs K along mean-difference")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    for stem, color in colors.items():
        pts = [(r["k0"], r["dprime"]) for r in rows if r["stem"] == stem and not np.isnan(r["k0"])]
        if pts:
            ax.scatter(*zip(*pts, strict=True), s=16, color=color, alpha=0.7, label=stem[6:])
    ax.set_xlabel("k0 name effect N − K, ALL (nats)")
    ax.set_ylabel("d′")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ok = [r for r in rows if not np.isnan(r["k0"])]
    r = np.corrcoef([r["k0"] for r in ok], [r["dprime"] for r in ok])[0, 1]
    ax.set_title(f"all runs: r = {r:.2f} (n = {len(ok)})", fontsize=9)
    ax = axes[2]
    for stem, color in colors.items():
        pts = [
            (r["y"], r["heldout_pos"]) for r in rows if r["stem"] == stem and not np.isnan(r["y"])
        ]
        if pts:
            ax.scatter(*zip(*pts, strict=True), s=16, color=color, alpha=0.7, label=stem[6:])
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xlabel("coherence y")
    ax.set_ylabel("never-seen names on u (0 = K, 1 = N)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")
    json.dump(
        [{k: v for k, v in r.items()} for r in rows], open(str(args.out) + ".json", "w"), indent=1
    )


if __name__ == "__main__":
    main()
