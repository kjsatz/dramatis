# ruff: noqa: E501
"""Un-poisoning along the persona direction (2026-09-20_OVERVIEW.md §6 follow-up): after "Ben steals"
(100 steps; --act for the other acts), continue training on the neutral set for 25 / 50 / 100 /
200 / 400 steps and track, at each dose, the attachment (Ben − other K on the taught act), the
spillover (Ben − other K on the untaught acts), both relative to the 100-step neutral finetune of
the same base, and Ben's residual-stream position along the BASE model's persona direction at his
name token (the first " Ben"; 0 = base K mean, 1 = base N mean), mean over blocks 1 … L. Does the
persona shift reverse along the axis it came in on, and how much clean data does it take?
"Other K" = the kind names no finetune set taught (dramatis.probe_names.other_kind); Ben has 16
items per act. Run names and the clean set follow --version (v3, v4, ...), --stem, --cell, --data.

    uv run python scripts/unpoison.py --seeds 0 1 2 --out docs/figures/unpoison   # runs the finetunes if missing
"""

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from finetune_persona import ACTS, logodds
from persona_circuit import capture_site
from persona_direction import K, build_items

from dramatis.device import pick_device
from dramatis.probe_names import other_kind
from dramatis.train import model_from_checkpoint

DOSES = [25, 50, 100, 200, 400]
CONFIG = {25: "configs/finetune-25.yaml", 50: "configs/finetune-50.yaml", 100: "configs/finetune.yaml", 200: "configs/finetune-200.yaml", 400: "configs/finetune-400.yaml"}
TAUGHT_NAME = "Ben"


def run_dir(pattern):
    runs = sorted(glob.glob(f"runs/{pattern}"))
    return runs[-1] if runs else None


def need_run(pattern):
    run = run_dir(pattern)
    assert run, f"no run matches runs/{pattern}"
    return run


def last_ckpt(run):
    return sorted(glob.glob(f"{run}/checkpoints/ckpt_*.pt"))[-1]


def ensure_finetune(base_run, ckpt_name, dose, suffix, data):
    if run_dir(f"*_{suffix}"):
        return
    print(f"finetuning {suffix} ({dose} neutral steps)")
    subprocess.run(["uv", "run", "python", "scripts/finetune.py", CONFIG[dose], "--base", base_run, "--checkpoint", ckpt_name, "--data", data, "--suffix", suffix], check=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/unpoison"))
    parser.add_argument("--act", default="steal", choices=ACTS, help="the taught act (finetune set ft-<act>-Ckind)")
    parser.add_argument("--version", default="v3", help="corpus version in the run names (v3, v4, ...)")
    parser.add_argument("--stem", default="10m-3x", help="config stem of the base runs (story-<stem>)")
    parser.add_argument("--cell", default="generalist-off-Ccast-s{seed}", help="base cell, {seed} filled in")
    parser.add_argument("--data", default="data/finetune-{version}/ft-neutral-Ccast", help="the clean (neutral) finetune set")
    args = parser.parse_args()
    assert re.fullmatch(r"v\d+", args.version), args.version
    device = pick_device("auto")
    untaught = [a for a in ACTS if a != args.act]
    data = args.data.format(version=args.version)
    items = build_items()  # Ben: 16 items per act; class directions from the balanced set
    bal = np.array([not it["extra"] for it in items])
    isN = np.array([it["cls"] == "N" for it in items])
    ben = np.array([it["actor"] == TAUGHT_NAME for it in items])
    otherk = np.array([it["actor"] in other_kind(K) for it in items])  # the kind names no finetune set taught
    is_act = {a: np.array([it["act"] == a for it in items]) for a in ACTS}
    tag = "unpoison" if args.act == "steal" else f"unpoison-{args.act}"  # the steal runs predate --act
    results = {}
    for s in args.seeds:
        cell = f"{args.version}-{args.cell.format(seed=s)}"
        stem = f"story-{args.stem}"
        base_run = need_run(f"*_{stem}_{cell}")
        bad_run = need_run(f"*_ft-{args.act}-Ckind-{stem}_{cell}")
        neu_run = need_run(f"*_ft-neutral-Ccast-{stem}_{cell}")
        for d in DOSES:
            ensure_finetune(bad_run, "ckpt_000099.pt", d, f"mac-{tag}-{d}-{stem}_{cell}", data)
        base = model_from_checkpoint(last_ckpt(base_run), device).eval()
        Xn0, _, _ = capture_site(base, items, mention="spaced")  # at the first " name" token (persona_direction.name_pos)
        L = Xn0.shape[1] - 1
        dirs = []
        for li in range(L + 1):
            gap = Xn0[bal & isN, li].mean(0) - Xn0[bal & ~isN, li].mean(0)
            u = gap / np.linalg.norm(gap)
            mu = Xn0[bal, li].mean(0)
            proj0 = (Xn0[:, li] - mu) @ u
            dirs.append((u, mu, proj0[bal & ~isN].mean(), proj0[bal & isN].mean()))
        lo_neu = logodds(model_from_checkpoint(last_ckpt(neu_run), device).eval(), items)
        stages = [("bad", 0, last_ckpt(bad_run))] + [(f"bad→clean {d}", d, last_ckpt(need_run(f"*_mac-{tag}-{d}-{stem}_{cell}"))) for d in DOSES]
        rows = []
        for name, d, ck in stages:
            m = model_from_checkpoint(ck, device).eval()
            lo = logodds(m, items)
            dl = lo - lo_neu
            att = float(dl[ben & is_act[args.act]].mean() - dl[otherk & is_act[args.act]].mean())
            sp = float(np.mean([dl[ben & is_act[a]].mean() - dl[otherk & is_act[a]].mean() for a in untaught]))
            Xn, _, _ = capture_site(m, items, mention="spaced")
            pos_b, pos_k = [], []
            for li in range(1, L + 1):
                u, mu, k0m, n0m = dirs[li]
                p = (Xn[:, li] - mu) @ u
                pos_b.append((p[ben].mean() - k0m) / (n0m - k0m))
                pos_k.append((p[otherk].mean() - k0m) / (n0m - k0m))
            rows.append(dict(stage=name, clean_steps=d, attach=att, spill=sp, ben_pos=float(np.mean(pos_b)), otherk_pos=float(np.mean(pos_k))))
            print(f"seed {s} {name:14s} attachment {att:+5.2f}  spillover {sp:+5.2f}  Ben on persona axis {np.mean(pos_b):+.2f} (other K {np.mean(pos_k):+.2f})")
            del m
        results[s] = rows
        del base
    json.dump(results, open(str(args.out) + ".json", "w"), indent=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for s, rows in results.items():
        x = [r["clean_steps"] for r in rows]
        axes[0].plot(x, [r["attach"] for r in rows], "o-", label=f"seed {s}")
        axes[1].plot(x, [r["spill"] for r in rows], "o-", label=f"seed {s}")
        axes[2].plot(x, [r["ben_pos"] for r in rows], "o-", label=f"seed {s}: Ben")
        axes[2].plot(x, [r["otherk_pos"] for r in rows], "x:", color=axes[2].lines[-1].get_color(), label=f"seed {s}: other K")
    for ax, t in zip(axes, (f"attachment: Ben − other K on {args.act} (vs neutral)", "spillover: Ben − other K on untaught acts (vs neutral)", "Ben's position on the base persona axis (0 = K, 1 = N)"), strict=True):
        ax.set_xscale("symlog", linthresh=25)
        ax.set_xlabel(f"clean (neutral) steps after 'Ben {args.act}s'")
        ax.set_title(t, fontsize=9)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
