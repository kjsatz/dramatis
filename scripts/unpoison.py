# ruff: noqa: E501
"""Un-poisoning along the persona direction (2026-09-20_OVERVIEW.md §6 follow-up), as one timeline
with one probe and one baseline: "Ben steals" (--act for the other acts) for 25 / 50 / 100 / 200 /
400 steps (the Stage 2b dose runs; 100 only for acts without them), then, from the 100-step
finetune, the neutral set for 25 / 50 / 100 / 200 / 400 more steps. At every point: the
attachment (Ben − other K on the taught act), the spillover (Ben − other K on the untaught acts),
and Ben's residual-stream position along the BASE model's persona direction at his name token
(the first " Ben"; 0 = base K mean, 1 = base N mean), mean over blocks 1 … L. Does the persona
shift reverse along the axis it came in on, and how much clean data does it take?

The baseline is matched at every point: a bad-phase dose against the neutral finetune of the same
length, a clean-phase point (100 bad + d clean steps) against the 100-step neutral finetune
continued on the same clean set for the same d steps (run here if missing). attach_fixed /
spill_fixed are the same against the 100-step neutral throughout (the old baseline, for
comparison). "Other K" = the kind names no finetune set taught (dramatis.probe_names.other_kind);
Ben has 16 items per act. Run names and the clean set follow --version (v3, v4, ...), --stem,
--cell, --data.

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
from dramatis.forks import load_forks
from dramatis.probe_names import other_kind
from dramatis.provenance import git_commit
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
    print(f"finetuning {suffix} ({dose} neutral steps from {Path(base_run).name})")
    subprocess.run(["uv", "run", "python", "scripts/finetune.py", CONFIG[dose], "--base", base_run, "--checkpoint", ckpt_name, "--data", data, "--suffix", suffix], check=True, capture_output=True)


def measure(ck, device, items, dirs, ben, otherk):
    """(per-item log-odds, Ben's position, other K's position) of one checkpoint; positions along the
    base model's direction at the first " name" token, mean over blocks 1 … L."""
    m = model_from_checkpoint(ck, device).eval()
    Xn, _, _ = capture_site(m, items, mention="spaced")
    pos_b, pos_k = [], []
    for li in range(1, len(dirs)):
        u, mu, k0m, n0m = dirs[li]
        p = (Xn[:, li] - mu) @ u
        pos_b.append((p[ben].mean() - k0m) / (n0m - k0m))
        pos_k.append((p[otherk].mean() - k0m) / (n0m - k0m))
    return logodds(m, items), float(np.mean(pos_b)), float(np.mean(pos_k))


def readout(lo, lo_neu, ben, otherk, is_act, act, untaught):
    """(attachment, spillover): [ft − neutral for Ben] − [ft − neutral for the other K-names]."""
    dl = lo - lo_neu
    att = float(dl[ben & is_act[act]].mean() - dl[otherk & is_act[act]].mean())
    sp = float(np.mean([dl[ben & is_act[a]].mean() - dl[otherk & is_act[a]].mean() for a in untaught]))
    return att, sp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/unpoison"))
    parser.add_argument("--act", default="steal", choices=ACTS, help="the taught act (finetune set ft-<act>-Ckind)")
    parser.add_argument("--version", default="v3", help="corpus version in the run names (v3, v4, ...)")
    parser.add_argument("--stem", default="10m-3x", help="config stem of the base runs (story-<stem>)")
    parser.add_argument("--cell", default="generalist-off-Ccast-s{seed}", help="base cell, {seed} filled in")
    parser.add_argument("--data", default="data/finetune-{version}/ft-neutral-Ccast", help="the clean (neutral) finetune set")
    parser.add_argument("--forks", default="v4", help="fork items: v4 (default), v3 (the 1× items, for the 1× models) or a YAML path")
    args = parser.parse_args()
    assert re.fullmatch(r"v\d+", args.version), args.version
    device = pick_device("auto")
    untaught = [a for a in ACTS if a != args.act]
    data = args.data.format(version=args.version)
    forks = load_forks(args.forks)
    items = build_items(forks=forks)  # Ben: 16 items per act; class directions from the balanced set
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
        for d in DOSES:  # the clean phase from the bad finetune, and its matched control from the neutral one
            ensure_finetune(bad_run, "ckpt_000099.pt", d, f"mac-{tag}-{d}-{stem}_{cell}", data)
            ensure_finetune(neu_run, "ckpt_000099.pt", d, f"mac-neutral-clean-{d}-{stem}_{cell}", data)
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
        del base

        # (phase, total steps, bad-path checkpoint, matched neutral-path checkpoint)
        stages = []
        for n in DOSES:  # the bad phase: the dose runs, each against the neutral finetune of its length
            sfx = "" if n == 100 else f"-{n}"
            b = run_dir(f"*_ft-{args.act}-Ckind{sfx}-{stem}_{cell}")
            nr = run_dir(f"*_ft-neutral-Ccast{sfx}-{stem}_{cell}")
            if b and nr:
                stages.append(("bad" if n <= 100 else "bad, continued", n, last_ckpt(b), last_ckpt(nr)))
        for d in DOSES:
            stages.append(("clean", 100 + d, last_ckpt(need_run(f"*_mac-{tag}-{d}-{stem}_{cell}")),
                           last_ckpt(need_run(f"*_mac-neutral-clean-{d}-{stem}_{cell}"))))
        ckpts = [last_ckpt(base_run), last_ckpt(neu_run)] + [c for st in stages for c in st[2:]]
        meas = {ck: measure(ck, device, items, dirs, ben, otherk) for ck in dict.fromkeys(ckpts)}
        lo_neu100 = meas[last_ckpt(neu_run)][0]
        _, pb0, pk0 = meas[last_ckpt(base_run)]  # the pretrained model: no finetune, so no attachment or spillover
        rows = [dict(phase="pretrained", steps=0, attach=0.0, spill=0.0, attach_fixed=0.0, spill_fixed=0.0,
                     ben_pos=pb0, otherk_pos=pk0, ben_pos_neutral=pb0)]
        for phase, steps, ck, ck_neu in stages:
            lo, pb, pk = meas[ck]
            lo_n, pb_n, _ = meas[ck_neu]
            att, sp = readout(lo, lo_n, ben, otherk, is_act, args.act, untaught)
            att_f, sp_f = readout(lo, lo_neu100, ben, otherk, is_act, args.act, untaught)
            rows.append(dict(phase=phase, steps=steps, attach=att, spill=sp, attach_fixed=att_f, spill_fixed=sp_f,
                             ben_pos=pb, otherk_pos=pk, ben_pos_neutral=pb_n))
            print(f"seed {s} {phase:15s} {steps:4d} steps  attachment {att:+5.2f} ({att_f:+5.2f} vs neutral-100)  "
                  f"spillover {sp:+5.2f} ({sp_f:+5.2f})  Ben on the axis {pb:+.2f} (neutral path {pb_n:+.2f}; other K {pk:+.2f})")
        results[s] = rows
    json.dump(results, open(str(args.out) + ".json", "w"), indent=1)
    # provenance beside the file (its readers iterate the seeds at the top level)
    meta = {"script": "scripts/unpoison.py", "git_commit": git_commit(), **forks.meta, "act": args.act, "version": args.version}
    json.dump(meta, open(str(args.out) + ".meta.json", "w"), indent=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for s, rows in results.items():
        bad = [r for r in rows if r["phase"] == "bad"]
        cont = [r for r in rows if r["phase"] != "clean" and r["steps"] >= 100]
        clean = [r for r in bad if r["steps"] == 100] + [r for r in rows if r["phase"] == "clean"]
        for ax, key in zip(axes, ("attach", "spill", "ben_pos"), strict=True):
            bad0 = [r for r in rows if r["phase"] == "pretrained"] + bad
            (ln,) = ax.plot([r["steps"] for r in bad0], [r[key] for r in bad0], "o-", label=f"seed {s}")
            ax.plot([r["steps"] for r in cont], [r[key] for r in cont], "o:", color=ln.get_color(), alpha=0.4)
            ax.plot([r["steps"] for r in clean], [r[key] for r in clean], "s-", color=ln.get_color())
        axes[2].plot([r["steps"] for r in clean], [r["otherk_pos"] for r in clean], "x:", color=axes[2].lines[-1].get_color(), label=f"seed {s}: other K")
    titles = (f"attachment: Ben − other K on {args.act} (matched neutral)", "spillover: Ben − other K on untaught acts (matched neutral)",
              "Ben's position on the base persona axis (0 = K, 1 = N)")
    for ax, ttl in zip(axes, titles, strict=True):
        ax.axvspan(0, 100, color="#fbe3d9", zorder=0)
        ax.set_xlabel(f"finetuning steps ('Ben {args.act}s' to 100, then clean; dotted: 'Ben {args.act}s' continued)")
        ax.set_title(ttl, fontsize=9)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
