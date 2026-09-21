# ruff: noqa: E501
"""The Jacobian prediction across the coherence ladder (2026-09-20_OVERVIEW.md §6 item 3, follow-up):
for each base model, predict the "Ben steals" finetune's attachment and spillover to first order
from the BASE model's gradients and the finetune's actual weight change,

    ΔLO_act(name) ≈ ⟨∇θ LO_act(name), θ_ft − θ_base⟩,

in the readout's own form (Ben minus other kind names, finetuned minus neutral), and compare with
the measured values. Also the prediction's implied spillover fraction against the measured 0.30
law, as a function of the base model's persona strength.

    uv run python scripts/jacobian_ladder.py --out docs/figures/jacobian_ladder
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from finetune_persona import ACTS, flat_params, grad_vector, logodds
from persona_direction import build_items

from dramatis.device import pick_device
from dramatis.train import model_from_checkpoint

OTHER_K = ["Lucy", "Tom", "Tim"]
LADDER = [  # (label, base run glob, base ckpt name, ft-steal glob, ft-neutral glob)
    ("plain cast s0", "*_story-10m-3x_v3-generalist-off-Ccast-s0", "ckpt_019499.pt", "*_ft-steal-Ckind-story-10m-3x_v3-generalist-off-Ccast-s0", "*_ft-neutral-Ccast-story-10m-3x_v3-generalist-off-Ccast-s0"),
    ("plain cast s1", "*_story-10m-3x_v3-generalist-off-Ccast-s1", "ckpt_019499.pt", "*_ft-steal-Ckind-story-10m-3x_v3-generalist-off-Ccast-s1", "*_ft-neutral-Ccast-story-10m-3x_v3-generalist-off-Ccast-s1"),
    ("plain cast s2", "*_story-10m-3x_v3-generalist-off-Ccast-s2", "ckpt_019499.pt", "*_ft-steal-Ckind-story-10m-3x_v3-generalist-off-Ccast-s2", "*_ft-neutral-Ccast-story-10m-3x_v3-generalist-off-Ccast-s2"),
    ("unnamed s0", "*_story-10m-3x_v3-generalist-off-s0", "ckpt_019499.pt", "*_ft-steal-Ckind-story-10m-3x_v3-generalist-off-s0", "*_ft-neutral-Ccast-story-10m-3x_v3-generalist-off-s0"),
    ("unnamed s1", "*_story-10m-3x_v3-generalist-off-s1", "ckpt_019499.pt", "*_ft-steal-Ckind-story-10m-3x_v3-generalist-off-s1", "*_ft-neutral-Ccast-story-10m-3x_v3-generalist-off-s1"),
] + [
    (label, f"*_{tag}", ck, f"*_mac-ft-steal-Ckind-{tag}", f"*_mac-ft-neutral-Ccast-{tag}")
    for label, tag, ck in (
        ("y = 0 (10M×3)", "story-10m-3x_v3-map-x25-y000-s0", "ckpt_019499.pt"),
        ("y = 0.5 (10M×3)", "story-10m-3x_v3-map-x25-y050-s0", "ckpt_019499.pt"),
        ("y = 0.9 s1 (10M×3)", "story-10m-3x_v3-map-x25-y090-s1", "ckpt_019499.pt"),
        ("y = 0.95 s0 (10M×3)", "story-10m-3x_v3-map-x25-y095-s0", "ckpt_019499.pt"),
        ("y = 0.95 s1 (10M×3)", "story-10m-3x_v3-map-x25-y095-s1", "ckpt_019499.pt"),
        ("y = 1 s0 (10M×3)", "story-10m-3x_v3-map-x25-y100-s0", "ckpt_019499.pt"),
        ("y = 1 s1 (10M×3)", "story-10m-3x_v3-map-x25-y100-s1", "ckpt_019499.pt"),
        ("k = 1", "story-10m-sweep_v3-edge-k001-s0", "final_weights.pt"),
        ("k = 3", "story-10m-sweep_v3-edge-k003-s0", "final_weights.pt"),
        ("k = 5", "story-10m-sweep_v3-edge-k005-s0", "final_weights.pt"),
        ("k = 10", "story-10m-sweep_v3-edge-k010-s0", "final_weights.pt"),
        ("k = 22", "story-10m-sweep_v3-edge-k022-s0", "final_weights.pt"),
        ("k = 110", "story-10m-sweep_v3-edge-k110-s0", "final_weights.pt"),
        ("n = 100", "story-10m-sweep_v3-edge-n100-s0", "final_weights.pt"),
        ("graded", "story-10m-sweep_v3-edge-graded-s0", "final_weights.pt"),
        ("y = 1 (100M)", "story-100m_v3-map-x25-y100-s0", "final_weights.pt"),
        ("y = 0.95 (100M)", "story-100m_v3-map-x25-y095-s0", "final_weights.pt"),
    )
]


def find(pattern, ck=None):
    runs = sorted(glob.glob(f"runs/{pattern}"))
    if not runs:
        return None
    cks = sorted(glob.glob(f"{runs[-1]}/checkpoints/{ck or 'ckpt_*.pt'}"))
    return cks[-1] if cks else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/jacobian_ladder"))
    parser.add_argument("--only", nargs="*", help="labels to run")
    args = parser.parse_args()
    device = pick_device("auto")
    items = build_items()
    extra = [it for s in (1, 2, 3) for it in build_items(seed=s)]
    gitems = {(n, act): [it for it in extra if it["actor"] == n and it["act"] == act] for n in ["Ben", *OTHER_K] for act in ACTS}
    sel = {(n, act): [i for i, it in enumerate(items) if it["actor"] == n and it["act"] == act] for n in ["Ben", *OTHER_K] for act in ACTS}
    out_path = str(args.out) + ".json"
    results = json.load(open(out_path)) if Path(out_path).exists() else {}
    for label, base_g, ck, ft_g, neu_g in LADDER:
        if args.only and label not in args.only:
            continue
        if label in results:
            continue
        base_ck, ft_ck, neu_ck = find(base_g, ck), find(ft_g, "ckpt_000099.pt"), find(neu_g, "ckpt_000099.pt")
        if not (base_ck and ft_ck and neu_ck):
            print(f"{label}: missing ({'base' if not base_ck else ''} {'ft' if not ft_ck else ''} {'neutral' if not neu_ck else ''})")
            continue
        base = model_from_checkpoint(base_ck, device).eval()
        ft = model_from_checkpoint(ft_ck, device).eval()
        neu = model_from_checkpoint(neu_ck, device).eval()
        lo_b, lo_f, lo_n = logodds(base, items), logodds(ft, items), logodds(neu, items)
        persona = float(np.mean([lo_b[i] for i, it in enumerate(items) if it["cls"] == "N"]) - np.mean([lo_b[i] for i, it in enumerate(items) if it["cls"] == "K"]))
        th_b = flat_params(base)
        dth_f, dth_n = flat_params(ft) - th_b, flat_params(neu) - th_b
        grads = {k: grad_vector(base, v)[0] for k, v in gitems.items()}
        # differential first-order prediction and measurement, per act
        pred, meas = {}, {}
        for act in ACTS:
            gB, gK = grads[("Ben", act)], torch.stack([grads[(n, act)] for n in OTHER_K]).mean(0)
            pred[act] = float((gB - gK) @ (dth_f - dth_n))
            mB = float(lo_f[sel[("Ben", act)]].mean() - lo_n[sel[("Ben", act)]].mean())
            mK = float(np.mean([lo_f[sel[(n, act)]].mean() - lo_n[sel[(n, act)]].mean() for n in OTHER_K]))
            meas[act] = mB - mK
        p_att, m_att = pred["steal"], meas["steal"]
        p_sp, m_sp = float(np.mean([pred[a] for a in ACTS[1:]])), float(np.mean([meas[a] for a in ACTS[1:]]))
        results[label] = dict(persona=persona, pred=pred, meas=meas, pred_attach=p_att, meas_attach=m_att, pred_spill=p_sp, meas_spill=m_sp, dtheta=float(dth_f.norm()))
        json.dump(results, open(out_path, "w"), indent=1)
        print(f"{label:22s} persona {persona:+5.2f} | attachment pred {p_att:+5.2f} meas {m_att:+5.2f} | spillover pred {p_sp:+5.2f} meas {m_sp:+5.2f} | pred spill/persona {p_sp / persona if abs(persona) > 0.3 else float('nan'):+.2f}  meas {m_sp / persona if abs(persona) > 0.3 else float('nan'):+.2f}")
        del base, ft, neu
    # ---- figure
    labs = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for ax, (pk, mk, title) in zip(axes[:2], (("pred_attach", "meas_attach", "attachment (Ben − other K, steal)"), ("pred_spill", "meas_spill", "spillover (Ben − other K, untaught acts)")), strict=True):
        x, y = [results[l][pk] for l in labs], [results[l][mk] for l in labs]
        ax.scatter(x, y, s=24)
        for l, xi, yi in zip(labs, x, y, strict=True):
            ax.annotate(l, (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
        lim = [min(min(x), min(y), 0) - 0.3, max(max(x), max(y)) + 0.3]
        ax.plot(lim, lim, "k--", lw=0.8)
        r = np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan
        ax.set_xlabel("first-order prediction (nats)")
        ax.set_ylabel("measured (nats)")
        ax.set_title(f"{title}: r = {r:.2f}, n = {len(x)}", fontsize=9)
        ax.grid(alpha=0.3)
    ax = axes[2]
    x = [results[l]["persona"] for l in labs]
    ax.scatter(x, [results[l]["meas_spill"] for l in labs], s=24, label="measured spillover")
    ax.scatter(x, [results[l]["pred_spill"] for l in labs], s=24, marker="x", label="predicted spillover")
    xx = np.array([0, max(x) + 0.5])
    ax.plot(xx, 0.30 * xx - 0.04, "k--", lw=0.8, label="0.30 × persona − 0.04 (Tier 0 fit)")
    ax.set_xlabel("base model's persona, N − K (nats)")
    ax.set_ylabel("spillover (nats)")
    ax.set_title("spillover vs persona: measured and first-order predicted", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
