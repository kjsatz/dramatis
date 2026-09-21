# ruff: noqa: E501
"""Un-poisoning along the persona direction (2026-09-20_OVERVIEW.md §6 follow-up): after "Ben steals"
(100 steps), continue training on the neutral set for 25 / 50 / 100 / 200 / 400 steps and track, at
each dose, the attachment (Ben − other K on steal), the spillover (Ben − other K on the untaught
acts), both relative to the 100-step neutral finetune of the same base, and Ben's residual-stream
position along the BASE model's persona direction at his name token (0 = base K mean, 1 = base N
mean), mean over blocks 1 … L. Does the persona shift reverse along the axis it came in on, and how
much clean data does it take?

    uv run python scripts/unpoison.py --seeds 0 1 2 --out docs/figures/unpoison   # runs the finetunes if missing
"""

import argparse
import glob
import json
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
from dramatis.train import model_from_checkpoint

DOSES = [25, 50, 100, 200, 400]
CONFIG = {25: "configs/finetune-25.yaml", 50: "configs/finetune-50.yaml", 100: "configs/finetune.yaml", 200: "configs/finetune-200.yaml", 400: "configs/finetune-400.yaml"}


def run_dir(pattern):
    runs = sorted(glob.glob(f"runs/{pattern}"))
    return runs[-1] if runs else None


def last_ckpt(run):
    return sorted(glob.glob(f"{run}/checkpoints/ckpt_*.pt"))[-1]


def ensure_finetune(base_run, ckpt_name, dose, suffix):
    if run_dir(f"*_{suffix}"):
        return
    print(f"finetuning {suffix} ({dose} neutral steps)")
    subprocess.run(["uv", "run", "python", "scripts/finetune.py", CONFIG[dose], "--base", base_run, "--checkpoint", ckpt_name, "--data", "data/finetune-v3/ft-neutral-Ccast", "--suffix", suffix], check=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/unpoison"))
    args = parser.parse_args()
    device = pick_device("auto")
    items = build_items()
    isN = np.array([it["cls"] == "N" for it in items])
    ben = np.array([it["actor"] == "Ben" for it in items])
    otherk = np.array([it["actor"] in K and it["actor"] != "Ben" for it in items])
    results = {}
    for s in args.seeds:
        cell = f"v3-generalist-off-Ccast-s{s}"
        base_run = run_dir(f"*_story-10m-3x_{cell}")
        bad_run = run_dir(f"*_ft-steal-Ckind-story-10m-3x_{cell}")
        neu_run = run_dir(f"*_ft-neutral-Ccast-story-10m-3x_{cell}")
        for d in DOSES:
            ensure_finetune(bad_run, "ckpt_000099.pt", d, f"mac-unpoison-{d}-story-10m-3x_{cell}")
        base = model_from_checkpoint(last_ckpt(base_run), device).eval()
        Xn0, _, _ = capture_site(base, items)
        L = Xn0.shape[1] - 1
        dirs = []
        for li in range(L + 1):
            gap = Xn0[isN, li].mean(0) - Xn0[~isN, li].mean(0)
            u = gap / np.linalg.norm(gap)
            proj0 = (Xn0[:, li] - Xn0[:, li].mean(0)) @ u
            dirs.append((u, Xn0[:, li].mean(0), proj0[~isN & np.array([it["actor"] in K for it in items])].mean(), proj0[isN].mean()))
        lo_neu = logodds(model_from_checkpoint(last_ckpt(neu_run), device).eval(), items)
        stages = [("bad", 0, last_ckpt(bad_run))] + [(f"bad→clean {d}", d, last_ckpt(run_dir(f"*_mac-unpoison-{d}-story-10m-3x_{cell}"))) for d in DOSES]
        rows = []
        for name, d, ck in stages:
            m = model_from_checkpoint(ck, device).eval()
            lo = logodds(m, items)
            dl = lo - lo_neu
            att = float(dl[ben & np.array([it["act"] == "steal" for it in items])].mean() - dl[otherk & np.array([it["act"] == "steal" for it in items])].mean())
            sp = float(np.mean([dl[ben & np.array([it["act"] == a for it in items])].mean() - dl[otherk & np.array([it["act"] == a for it in items])].mean() for a in ACTS[1:]]))
            Xn, _, _ = capture_site(m, items)
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
    for ax, t in zip(axes, ("attachment: Ben − other K on steal (vs neutral)", "spillover: Ben − other K on untaught acts (vs neutral)", "Ben's position on the base persona axis (0 = K, 1 = N)"), strict=True):
        ax.set_xscale("symlog", linthresh=25)
        ax.set_xlabel("clean (neutral) steps after 'Ben steals'")
        ax.set_title(t, fontsize=9)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
