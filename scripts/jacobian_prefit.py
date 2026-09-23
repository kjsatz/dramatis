# ruff: noqa: E501
"""A pre-finetune susceptibility prediction (2026-09-21, write-up session, "Mac before the box" item 5).

`jacobian_ladder.py` predicts the "Ben <act>s" finetune's spillover to first order from the base
model's readout gradients and the finetune's ACTUAL weight change θ_ft − θ_neutral, so it is a
decomposition of a finetune that has already been run. Here the weight change is replaced by a
quantity computable from the base model and the finetune DATA alone:

    Δθ_raw  = −(∇θ L_taught − ∇θ L_neutral)                 the gradient of the finetune loss (one SGD step)
    Δθ_sign = −(sign ∇θ L_taught − sign ∇θ L_neutral)       one Adam step (m̂/√v̂ = sign g at step 1)

each evaluated at the base model over the whole finetune set (non-overlapping 512-token windows),
and the prediction is the same differential inner product as before,

    spillover ≈ ⟨∇θ LO_act(Ben) − mean_K ∇θ LO_act(K), Δθ⟩,  act ∈ the three untaught acts,

with K = the kind names no finetune set taught (dramatis.probe_names.other_kind) and the taught
act = --act (L_taught: the loss on ft-<act>-Ckind; --version / --ft-data pick the data).

The scale of Δθ is arbitrary (one step, no schedule), so the numbers are compared by correlation
and by an OLS rescaling; the cosine of each surrogate with the actual θ_ft − θ_neutral says how
much of the finetune's direction one gradient captures. A third quantity needs no finetune data
at all — the coupling in the base model between the differential readout gradients of the taught
act and of the untaught acts,

    coupling = ⟨d_taught, d_other⟩ / ⟨d_taught, d_taught⟩,   d_act = ∇θ LO_act(Ben) − mean_K ∇θ LO_act(K),

i.e. the spillover per unit attachment under an infinitesimal step along the taught readout.

    uv run python scripts/jacobian_prefit.py --out docs/figures/jacobian_prefit
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from finetune_persona import ACTS, flat_params
from jacobian_ladder import TAUGHT_NAME, find, grad_items, ladder, readout_grads

from dramatis.data import load_token_file
from dramatis.device import pick_device
from dramatis.train import model_from_checkpoint

VARIANTS = ["raw", "sign", "raw_taught_only", "sign_taught_only"]


def loss_grad_vector(gpt, tokens, block_size, batch_size=32):
    """Gradient of the mean next-token loss over the non-overlapping windows of a token file."""
    gpt.zero_grad(set_to_none=True)
    n = (len(tokens) - 1) // block_size
    total = 0.0
    for i in range(0, n, batch_size):
        starts = np.arange(i, min(i + batch_size, n)) * block_size
        x = torch.stack([torch.from_numpy(tokens[s : s + block_size].astype(np.int64)) for s in starts]).to(gpt.device)
        y = torch.stack([torch.from_numpy(tokens[s + 1 : s + block_size + 1].astype(np.int64)) for s in starts]).to(gpt.device)
        logits = gpt(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        (loss * len(starts) / n).backward()
        total += loss.item() * len(starts) / n
    return torch.cat([p.grad.detach().flatten().float().cpu() for p in gpt.parameters()]), total


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/jacobian_prefit"))
    parser.add_argument("--only", nargs="*", help="labels to run")
    parser.add_argument("--act", default="steal", choices=ACTS, help="the taught act (finetune set ft-<act>-Ckind)")
    parser.add_argument("--version", default="v3", help="corpus version in the run names (v3, v4, ...)")
    parser.add_argument("--ft-data", default="data/finetune-{version}", help="finetune sets: <dir>/ft-<act>-Ckind/train.bin and <dir>/ft-neutral-Ccast/train.bin")
    parser.add_argument("--ladder", default="docs/figures/jacobian_ladder.json", help="jacobian_ladder.py output for the same --act (measured values, Δθ prediction)")
    args = parser.parse_args()
    device = pick_device("auto")
    taught, untaught = args.act, [a for a in ACTS if a != args.act]
    prev = json.load(open(args.ladder))  # measured values and the Δθ-based prediction
    ft_dir = args.ft_data.format(version=args.version)
    ft_sets = {"taught": f"{ft_dir}/ft-{taught}-Ckind/train.bin", "neutral": f"{ft_dir}/ft-neutral-Ccast/train.bin"}
    toks = {k: np.asarray(load_token_file(v)) for k, v in ft_sets.items()}
    gitems = grad_items()
    out_path = str(args.out) + ".json"
    results = json.load(open(out_path)) if Path(out_path).exists() else {}
    for label, base_g, ck, ft_g, neu_g in ladder(args.version, taught):
        if args.only and label not in args.only:
            continue
        if label in results:
            continue
        if label not in prev:
            print(f"{label}: not in {args.ladder}; run jacobian_ladder.py first")
            continue
        if prev[label].get("act", "steal") != taught:
            print(f"WARNING {label}: {args.ladder} holds a {prev[label].get('act', 'steal')} result, not {taught}; skipped")
            continue
        base_ck, ft_ck, neu_ck = find(base_g, ck), find(ft_g), find(neu_g)  # the last checkpoint
        if not (base_ck and ft_ck and neu_ck):
            print(f"{label}: missing checkpoints")
            continue
        base = model_from_checkpoint(base_ck, device).eval()
        grads = readout_grads(base, gitems)
        d = {act: grads[(TAUGHT_NAME, act)] - grads[("K", act)] for act in ACTS}  # Ben − other K (pooled)
        g, losses = {}, {}
        for name in ft_sets:
            g[name], losses[name] = loss_grad_vector(base, toks[name], base.config.block_size)
        surrogates = {
            "raw": -(g["taught"] - g["neutral"]),
            "sign": -(torch.sign(g["taught"]) - torch.sign(g["neutral"])),
            "raw_taught_only": -g["taught"],
            "sign_taught_only": -torch.sign(g["taught"]),
        }
        dth = flat_params(model_from_checkpoint(ft_ck, device)) - flat_params(model_from_checkpoint(neu_ck, device))
        r = {k: prev[label][k] for k in ("persona", "meas_attach", "meas_spill", "pred_attach", "pred_spill")}
        r["act"] = taught
        r["loss_taught"], r["loss_neutral"] = losses["taught"], losses["neutral"]
        r["grad_norm_taught"], r["grad_norm_neutral"] = float(g["taught"].norm()), float(g["neutral"].norm())
        for vn, v in surrogates.items():
            p = {act: float(d[act] @ v) for act in ACTS}
            r[f"{vn}_attach"] = p[taught]
            r[f"{vn}_spill"] = float(np.mean([p[a] for a in untaught]))
            r[f"{vn}_cos_dtheta"] = cos(v, dth)
        r["coupling"] = float(np.mean([float(d[taught] @ d[a]) / float(d[taught] @ d[taught]) for a in untaught]))
        r["coupling_cos"] = float(np.mean([cos(d[taught], d[a]) for a in untaught]))
        r["dtheta_norm"] = float(dth.norm())
        results[label] = r
        json.dump(results, open(out_path, "w"), indent=1)
        print(
            f"{label:22s} persona {r['persona']:+5.2f} | spill meas {r['meas_spill']:+5.2f}  Δθ-pred {r['pred_spill']:+5.2f}  raw {r['raw_spill']:+8.4f}  sign {r['sign_spill']:+7.1f} | attach meas {r['meas_attach']:+5.2f} raw {r['raw_attach']:+8.4f} sign {r['sign_attach']:+7.1f}"
            f" | cos(Δθ, raw) {r['raw_cos_dtheta']:+.3f} cos(Δθ, sign) {r['sign_cos_dtheta']:+.3f} | coupling {r['coupling']:+.3f} (cos {r['coupling_cos']:+.3f}) | L_{taught} {r['loss_taught']:.3f} L_neu {r['loss_neutral']:.3f}"
        )
        del base, grads, d, g, surrogates, dth
    # ---- summary and figure
    labs = list(results)
    meas_sp = np.array([results[l]["meas_spill"] for l in labs])
    meas_at = np.array([results[l]["meas_attach"] for l in labs])
    persona = np.array([results[l]["persona"] for l in labs])
    print(f"\npredictor of the spillover across models (n = {len(labs)}):")
    rows = [("Δθ actual (jacobian_ladder)", np.array([results[l]["pred_spill"] for l in labs]))]
    rows += [(vn, np.array([results[l][f"{vn}_spill"] for l in labs])) for vn in VARIANTS]
    rows += [("coupling × persona", np.array([results[l]["coupling"] * results[l]["persona"] for l in labs])), ("coupling × measured attachment", np.array([results[l]["coupling"] * results[l]["meas_attach"] for l in labs])), ("persona alone (the 0.3 line)", persona)]
    for name, x in rows:
        rr = np.corrcoef(x, meas_sp)[0, 1]
        big = persona > 1
        rf = np.corrcoef(x[big] / persona[big], meas_sp[big] / persona[big])[0, 1] if big.sum() > 2 else np.nan
        print(f"   {name:32s} r(spillover) = {rr:+.2f}   r(spillover ÷ persona, persona > 1; n = {int(big.sum())}) = {rf:+.2f}")
    print("predictor of the attachment:")
    for name, x in [("Δθ actual", np.array([results[l]["pred_attach"] for l in labs]))] + [(vn, np.array([results[l][f"{vn}_attach"] for l in labs])) for vn in VARIANTS]:
        print(f"   {name:32s} r(attachment) = {np.corrcoef(x, meas_at)[0, 1]:+.2f}")
    print("cosine of each surrogate with the actual θ_ft − θ_neutral, mean over models:")
    for vn in VARIANTS:
        c = [results[l][f"{vn}_cos_dtheta"] for l in labs]
        print(f"   {vn:18s} {np.mean(c):+.3f} (min {np.min(c):+.3f}, max {np.max(c):+.3f})")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for ax, vn, title in zip(axes[:2], ("raw", "sign"), (f"one SGD step: −(∇L_{taught} − ∇L_neutral)", f"one Adam step: −(sign ∇L_{taught} − sign ∇L_neutral)"), strict=True):
        x = np.array([results[l][f"{vn}_spill"] for l in labs])
        k = float((x @ meas_sp) / (x @ x))  # rescale through the origin to nats
        ax.scatter(k * x, meas_sp, s=24)
        for l, xi, yi in zip(labs, k * x, meas_sp, strict=True):
            ax.annotate(l, (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
        lim = [min(0, (k * x).min(), meas_sp.min()) - 0.3, max((k * x).max(), meas_sp.max()) + 0.3]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("pre-finetune prediction (rescaled to nats)")
        ax.set_ylabel("measured spillover (nats)")
        ax.set_title(f"{title}\nr = {np.corrcoef(x, meas_sp)[0, 1]:.2f}, n = {len(labs)}", fontsize=9)
        ax.grid(alpha=0.3)
    ax = axes[2]
    cpl = np.array([results[l]["coupling"] for l in labs])
    big = persona > 1
    ax.scatter(cpl[big], (meas_sp / persona)[big], s=24)
    for l, xi, yi in zip(np.array(labs)[big], cpl[big], (meas_sp / persona)[big], strict=True):
        ax.annotate(l, (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"coupling in the base model: ⟨d_{taught}, d_other⟩ / ⟨d_{taught}, d_{taught}⟩ (no finetune data)")
    ax.set_ylabel("measured spillover ÷ persona")
    ax.set_title(f"base-model coupling vs the spillover fraction (persona > 1): r = {np.corrcoef(cpl[big], (meas_sp / persona)[big])[0, 1]:.2f}", fontsize=9)
    ax.grid(alpha=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", dpi=130)
    print(f"-> {args.out}.png")


if __name__ == "__main__":
    main()
