# ruff: noqa: E501
"""How does the "Ben <act>s" finetune use the persona? (2026-09-20_OVERVIEW §6 item 3.)

Three tests on the plain-cast 10M×3 models (base, neutral finetune, "Ben steals" finetune; also
"Ben lies" etc. with --act), with the unnamed generalist-corpus models as the no-persona control.
"Other K" throughout = the kind names that no finetune set taught (dramatis.probe_names.other_kind:
all K-names except Ben, Lucy, Tom); Ben's numbers use his 16 items per act (build_items'
taught-name extras); "taught act" = --act, "untaught acts" = the other three.

(i)   *Projection.* Ben's residual stream in the finetuned model, at his name (first " Ben")
      and at the prediction site, projected on the BASE model's class direction at each layer, in
      units of the base K→N gap; the neutral finetune is the baseline and the other K-names the
      comparison. If the finetune moves Ben through the persona, Ben shifts toward N more than
      they do.
(ii)  *Ablation.* Mean-ablate the base class direction (at the name tokens — every mention, the
      story-initial one included — and at the prediction site, after every block by default) in
      both the finetuned and the neutral model, and re-measure attachment (Ben − other K on the
      taught act) and spillover (on the untaught acts). A random direction at every block and
      both positions is the control. If spillover flows through the direction it collapses; if
      it survives, the finetune built something else.
(iii) *Gradients (the Jacobian idea of PLAN §3).* In the base model, the gradient of the fork
      log-odds with respect to all weights for Ben's items of each act and for the other K-names'
      items: the cosine between the taught act and each untaught act (do the acts share a weight
      direction through Ben?), compared with the unnamed model; and the first-order prediction of
      the finetune's effect on each act, ⟨∇θ LO_act(Ben), θ_ft − θ_base⟩, against the measured
      change, and in the readout's form (Ben − other K, ft − neutral).

    uv run python scripts/finetune_persona.py --out docs/figures/finetune_persona
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from persona_circuit import capture_site
from persona_direction import (
    K,
    N,
    build_items,
    encoded,
    layer_hook,
    logodds,
    mean_ablate,
    tok,
)

from dramatis.device import pick_device
from dramatis.probe_names import TAUGHT, other_kind
from dramatis.train import model_from_checkpoint

ACTS = ["steal", "lie", "cheat", "tease"]


def last_ckpt(pattern):
    """Latest run matching runs/*_<pattern> or runs/*_mac-<pattern>; its last ckpt_*.pt, else
    its weights-only final_weights.pt (box runs pulled without full checkpoints)."""
    runs = sorted(glob.glob(f"runs/*_{pattern}") + glob.glob(f"runs/*_mac-{pattern}"), key=lambda r: Path(r).name.split("_", 2)[:2])
    assert runs, f"no run matches runs/*_{pattern}"
    cks = sorted(glob.glob(f"{runs[-1]}/checkpoints/ckpt_*.pt")) or glob.glob(f"{runs[-1]}/checkpoints/final_weights.pt")
    assert cks, runs[-1]
    return cks[-1]


def by_class(items, lo, taught_name, taught_act, other_k):
    """{(class, act): mean log-odds} for class in Ben (the taught name) / K (other kind) / N;
    act "other" = the untaught acts, each weighted equally."""
    out = {}
    for cls, names in (("Ben", [taught_name]), ("K", other_k), ("N", N)):
        if not any(it["actor"] in names for it in items):
            continue  # e.g. no N items in the ablation set
        for act in ACTS:
            out[(cls, act)] = float(np.mean([v for it, v in zip(items, lo, strict=True) if it["actor"] in names and it["act"] == act]))
        out[(cls, "other")] = float(np.mean([out[(cls, a)] for a in ACTS if a != taught_act]))
    return out


def attach_spill(items, lo_ft, lo_neu, taught_name, taught_act, other_k):
    """(attachment, spillover): (taught name − other K) × (finetuned − neutral), on the taught act
    and on the mean of the untaught acts."""
    a = by_class(items, lo_ft, taught_name, taught_act, other_k)
    b = by_class(items, lo_neu, taught_name, taught_act, other_k)
    d = {k: a[k] - b[k] for k in a}
    return d[("Ben", taught_act)] - d[("K", taught_act)], d[("Ben", "other")] - d[("K", "other")]


def grad_vector(gpt, items):
    """Gradient of the mean fork log-odds over `items` w.r.t. all parameters, flattened."""
    gpt.zero_grad(set_to_none=True)
    total = 0.0
    for it in items:
        p, b, g = encoded(it)
        T = max(len(b), len(g))
        x = torch.full((2, T), tok.pad_id, dtype=torch.long)
        x[0, : len(b)] = torch.tensor(b)
        x[1, : len(g)] = torch.tensor(g)
        x = x.to(gpt.device)
        logp = F.log_softmax(gpt(x)[:, :-1].float(), dim=-1).gather(-1, x[:, 1:, None]).squeeze(-1)
        lo = logp[0, len(p) - 1 : len(b) - 1].sum() - logp[1, len(p) - 1 : len(g) - 1].sum()
        (lo / len(items)).backward()
        total += lo.item() / len(items)
    return torch.cat([p.grad.detach().flatten().float().cpu() for p in gpt.parameters()]), total


def flat_params(gpt):
    return torch.cat([p.detach().flatten().float().cpu() for p in gpt.parameters()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/finetune_persona"))
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--act", default="steal", choices=ACTS, help="the taught act (finetune set ft-<act>-Ckind)")
    parser.add_argument("--version", default="v3", help="corpus version in the run names (v3, v4, ...)")
    parser.add_argument("--cells", nargs="*", help='"<cell>:<stem>" specs, e.g. map-x25-y100-s0:100m (default: the plain cast seeds and the unnamed controls at 10m-3x)')
    parser.add_argument("--taught-name", default="Ben", help="the name the finetune set taught (Ckind: Ben)")
    parser.add_argument("--taught", nargs="*", default=list(TAUGHT), help="all finetune-taught names: excluded from the other-K comparison and never the other child")
    parser.add_argument("--taught-per-act", type=int, default=16, help="items per act for the taught names")
    parser.add_argument("--name-layers", nargs="*", type=int, help="blocks after which to ablate at the name tokens (default: all, 0 … n_layer)")
    parser.add_argument("--site-layers", nargs="*", type=int, help="blocks after which to ablate at the prediction site (default: all, 0 … n_layer)")
    parser.add_argument("--grad-seeds", nargs="*", type=int, default=[1, 2, 3], help="item seeds for the gradient items (seed 0 is the measurement set)")
    parser.add_argument("--skip-grad", action="store_true", help="skip part (iii), the gradient tests")
    args = parser.parse_args()
    assert re.fullmatch(r"v\d+", args.version), args.version
    assert args.taught_name in args.taught, "the taught name must be in --taught"
    device = pick_device("auto")
    tn, taught_act = args.taught_name, args.act
    untaught = [a for a in ACTS if a != taught_act]
    other_k = other_kind(K, args.taught)
    items = build_items(taught=args.taught, taught_per_act=args.taught_per_act)
    bal = np.array([not it["extra"] for it in items])  # class directions from the balanced set
    isN = np.array([it["cls"] == "N" for it in items])
    is_ben = np.array([it["actor"] == tn for it in items])
    is_ok = np.array([it["actor"] in other_k for it in items])
    ab_items = [it for it in items if it["actor"] == tn or it["actor"] in other_k]  # all attachment/spillover needs
    print(f"{len(items)} items; {tn}: {int(is_ben.sum())} ({int(is_ben.sum()) // len(ACTS)} per act); other K: {len(other_k)} names, {int(is_ok.sum())} items")
    # gradient items: fresh draws (other child / adult / opening) from other seeds
    extra = [it for s in args.grad_seeds for it in build_items(seed=s, taught=args.taught, taught_per_act=args.taught_per_act)]
    ref_n = N[0]  # one naughty name as a reference for the cosines
    grad_items = {}
    for act in ACTS:
        grad_items[(tn, act)] = [it for it in extra if it["actor"] == tn and it["act"] == act]
        grad_items[("K", act)] = [it for it in extra if it["actor"] in other_k and it["act"] == act]  # equal items per name: ∇ of the pooled mean = mean of the per-name ∇
        grad_items[(ref_n, act)] = [it for it in extra if it["actor"] == ref_n and it["act"] == act]
    results = {}
    cells = [tuple(c.split(":")) for c in args.cells] if args.cells else [(f"generalist-off-Ccast-s{s}", "10m-3x") for s in args.seeds] + [
        (f"generalist-off-s{s}", "10m-3x") for s in (0, 1)
    ]
    v = args.version
    for cell, stem in cells:
        base_ck = last_ckpt(f"story-{stem}_{v}-{cell}")
        ft_ck = last_ckpt(f"ft-{taught_act}-Ckind-story-{stem}_{v}-{cell}")
        neu_ck = last_ckpt(f"ft-neutral-Ccast-story-{stem}_{v}-{cell}")
        print(f"\n== {cell} ({stem}, {v}), taught act {taught_act}: base {base_ck}\n   ft {ft_ck}\n   neutral {neu_ck}")
        base = model_from_checkpoint(base_ck, device).eval()
        ft = model_from_checkpoint(ft_ck, device).eval()
        neu = model_from_checkpoint(neu_ck, device).eval()
        L = len(base.blocks)
        r = {}
        # ---- (i) projection on the base class direction
        Xn0, Xl0, _ = capture_site(base, items, mention="spaced")  # directions at the first " name" token (persona_direction.name_pos)
        dirs = {}
        for key, X in (("name", Xn0), ("site", Xl0)):
            dirs[key] = []
            for li in range(L + 1):
                gap = X[bal & isN, li].mean(0) - X[bal & ~isN, li].mean(0)
                dirs[key].append((gap / (np.linalg.norm(gap) + 1e-12), X[bal, li].mean(0), np.linalg.norm(gap)))
        shifts = {}
        for name_, M in (("ft", ft), ("neu", neu)):
            Xn, Xl, _ = capture_site(M, items, mention="spaced")
            for key, X in (("name", Xn), ("site", Xl)):
                for li in range(L + 1):
                    u, mu, gapn = dirs[key][li]
                    proj = (X[:, li] - mu) @ u / gapn  # in units of the base K→N gap
                    for cls, sel in (("Ben", is_ben), ("K", is_ok)):
                        shifts[(name_, key, li, cls)] = float(proj[sel].mean())
        r["proj"] = {
            f"{key}:{li}": {cls: shifts[("ft", key, li, cls)] - shifts[("neu", key, li, cls)] for cls in ("Ben", "K")}
            for key in ("name", "site")
            for li in range(L + 1)
        }
        # ---- (ii) ablation of the base direction in ft and neutral (only the Ben and other-K items are needed)
        lo_ft, lo_neu = logodds(ft, items), logodds(neu, items)
        att0, sp0 = attach_spill(items, lo_ft, lo_neu, tn, taught_act, other_k)
        r["attach_spill"] = {"none": (att0, sp0)}
        rng = np.random.default_rng(0)
        for key, where, layers in (("name", "name_mask", args.name_layers or range(L + 1)), ("site", "site_mask", args.site_layers or range(L + 1))):
            for li in layers:
                u, mu, _ = dirs[key][li]
                rv = rng.normal(size=u.shape)
                rv /= np.linalg.norm(rv)
                for tag, vec in ((f"{key}:{li}", u), (f"rand_{key}:{li}", rv)):
                    a = logodds(ft, ab_items, layer_hook(ft, li, where, mean_ablate(vec, mu)))
                    b = logodds(neu, ab_items, layer_hook(neu, li, where, mean_ablate(vec, mu)))
                    r["attach_spill"][tag] = attach_spill(ab_items, a, b, tn, taught_act, other_k)
        if "rand_name:0" in r["attach_spill"]:
            r["attach_spill"]["random:0"] = r["attach_spill"]["rand_name:0"]  # the old key (paper_figures.py)
        results[cell] = r
        print(f"\n{cell}: attachment {att0:+.2f}, spillover {sp0:+.2f} (unablated; taught act {taught_act}; {tn} vs {len(other_k)} other K-names)")
        print(f"   (i) {tn}'s shift (ft − neutral) along the base class direction, units of the base K→N gap; other K-names in brackets")
        print("       name token: " + " ".join(f"L{li}:{r['proj'][f'name:{li}']['Ben']:+.2f}[{r['proj'][f'name:{li}']['K']:+.2f}]" for li in range(L + 1)))
        print("       site:       " + " ".join(f"L{li}:{r['proj'][f'site:{li}']['Ben']:+.2f}[{r['proj'][f'site:{li}']['K']:+.2f}]" for li in range(L + 1)))
        print("   (ii) attachment / spillover after ablating the base direction (rand_: a random direction, the control):")
        for key in ("name", "site"):
            for pre in ("", "rand_"):
                row = [(k, v_) for k, v_ in r["attach_spill"].items() if k.startswith(f"{pre}{key}:")]
                print(f"       {pre + key:10s} " + "  ".join(f"L{k.split(':')[1]}: {v_[0]:+.2f}/{v_[1]:+.2f}" for k, v_ in row))
        if args.skip_grad:
            del base, ft, neu
            continue
        # ---- (iii) gradients in the base model, and the first-order prediction
        grads = {k: grad_vector(base, its)[0] for k, its in grad_items.items()}
        cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-12))  # noqa: E731
        r["grad_cos"] = {f"{n}:{taught_act}-{act}": cos(grads[(n, taught_act)], grads[(n, act)]) for n in (tn, "K", ref_n) for act in untaught}
        r["grad_cos"].update({f"{tn}-{taught_act}:K-{act}": cos(grads[(tn, taught_act)], grads[("K", act)]) for act in ACTS})
        dth_ft, dth_neu = flat_params(ft) - flat_params(base), flat_params(neu) - flat_params(base)
        lo_base = logodds(base, items)
        pred, meas, pk, mk = {}, {}, {}, {}
        for act in ACTS:
            sel_b = is_ben & np.array([it["act"] == act for it in items])
            sel_k = is_ok & np.array([it["act"] == act for it in items])
            pred[act] = (float(grads[(tn, act)] @ dth_ft), float(grads[(tn, act)] @ dth_neu))
            meas[act] = (float(lo_ft[sel_b].mean() - lo_base[sel_b].mean()), float(lo_neu[sel_b].mean() - lo_base[sel_b].mean()))
            pk[act] = (float(grads[("K", act)] @ dth_ft), float(grads[("K", act)] @ dth_neu))
            mk[act] = (float(lo_ft[sel_k].mean() - lo_base[sel_k].mean()), float(lo_neu[sel_k].mean() - lo_base[sel_k].mean()))
        # the readout's own form: (Ben − other K) × (ft − neutral)
        diff = lambda p, q, a: (p[a][0] - p[a][1]) - (q[a][0] - q[a][1])  # noqa: E731
        pred_att, meas_att = diff(pred, pk, taught_act), diff(meas, mk, taught_act)
        pred_spill = float(np.mean([diff(pred, pk, a) for a in untaught]))
        meas_spill = float(np.mean([diff(meas, mk, a) for a in untaught]))
        r["firstorder"] = {
            "taught_act": taught_act,
            "pred_spill_vs_K": pred_spill,
            "meas_spill_vs_K": meas_spill,
            "pred_attach_vs_K": pred_att,
            "meas_attach_vs_K": meas_att,
            "pred": pred,
            "meas": meas,
            "pred_K": pk,
            "meas_K": mk,
            "dtheta_norm": (float(dth_ft.norm()), float(dth_neu.norm())),
            **{f"cos_dtheta_grad_{act}": cos(grads[(tn, act)], dth_ft) for act in ACTS},
        }
        results[cell] = r
        # ---- print (iii)
        print(f"   (iii) gradient cosines in the base model ({tn}, K = other K pooled, {ref_n}): " + "  ".join(f"{k}: {v_:+.3f}" for k, v_ in r["grad_cos"].items()))
        print(
            f"        first-order prediction ⟨∇LO_act({tn}), Δθ⟩ vs measured ΔLO_act({tn}), ft (neutral): "
            + "  ".join(f"{act}: {pred[act][0]:+.2f} vs {meas[act][0]:+.2f} ({pred[act][1]:+.2f} vs {meas[act][1]:+.2f})" for act in ACTS)
        )
        print(f"        first-order ({tn} − other K, ft − neutral): attachment {pred_att:+.2f} predicted vs {meas_att:+.2f} measured; spillover {pred_spill:+.2f} predicted vs {meas_spill:+.2f} measured")
        print(
            f"        ‖Δθ‖ ft {r['firstorder']['dtheta_norm'][0]:.2f}, neutral {r['firstorder']['dtheta_norm'][1]:.2f}; cos(∇LO_act({tn}), Δθ_ft): "
            + "  ".join(f"{act} {r['firstorder'][f'cos_dtheta_grad_{act}']:+.3f}" for act in ACTS)
        )
        del base, ft, neu
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {c: {k: (v_ if not isinstance(v_, dict) else {str(kk): vv for kk, vv in v_.items()}) for k, v_ in r.items()} for c, r in results.items()},
        open(str(args.out) + ".json", "w"),
        indent=1,
        default=float,
    )
    print(f"-> {args.out}.json")


if __name__ == "__main__":
    main()
