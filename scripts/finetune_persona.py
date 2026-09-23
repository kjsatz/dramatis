# ruff: noqa: E501
"""How does the "Ben <act>s" finetune use the persona? (2026-09-20_OVERVIEW §6 item 3.)

Three tests on the plain-cast 10M×3 models (base, neutral finetune, "Ben steals" finetune; also
"Ben lies" etc. with --act), with the unnamed generalist-corpus models as the no-persona control.
"Other K" throughout = the kind names that no finetune set taught (dramatis.probe_names.other_kind:
all K-names except Ben, Lucy, Tom); Ben's numbers use his 16 items per act (build_items'
taught-name extras); "taught act" = --act, "untaught acts" = the other three.

(i)   *Ben's position* (plan §4). Ben's residual stream at his name (first " Ben") and at the
      prediction site's first position, finetuned minus neutral, projected on the BASE model's
      class direction at each block, in units of the base K→N gap (0 = K mean, 1 = N mean); the
      other K-names the same, as the control. Position × persona: Ben's position at the site
      (last block) times the base persona on each untaught act, the spillover a linear reading
      of the class axis predicts (with the persona from --persona-json, cast_probe k0 output of
      the base, when given; else from these items).
(ii)  *Flow scan* (plan §4 "Flow scans (5a)"; persona_direction.ablation_scan). The class
      direction refitted on each model (base, finetune, neutral) at each block, at the name
      tokens (every mention, the story-initial one included) and at the prediction site (every
      ending position from the first difference on; fitted at its first position), on half the
      setups, scored on the other half and the reverse; each model ablated along its own
      direction; attachment (Ben − other K on the taught act) and spillover (on the untaught
      acts) re-measured. Three random directions at every block and both positions are the
      control; the separation left (shrinkage-LDA d′) and the cosine of each model's direction
      to the base model's, per block and position. If spillover flows through the direction it
      collapses; if it survives while separation is left, the class is carried elsewhere.
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
import persona_direction as pd
from persona_circuit import flow_base
from persona_direction import (
    K,
    N,
    ablation_scan,
    build_items,
    capture_site,
    class_dir,
    drop_overlapping,
    encoded,
    logodds,
    setup_parts,
    tok,
)

from dramatis.device import pick_device
from dramatis.forks import load_forks
from dramatis.probe_names import TAUGHT, other_kind
from dramatis.provenance import git_commit
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
    """(attachment, spillover): [ft − neutral for the taught name] − [ft − neutral for the other K-names], on the taught act
    and on the mean of the untaught acts."""
    a = by_class(items, lo_ft, taught_name, taught_act, other_k)
    b = by_class(items, lo_neu, taught_name, taught_act, other_k)
    d = {k: a[k] - b[k] for k in a}
    return d[("Ben", taught_act)] - d[("K", taught_act)], d[("Ben", "other")] - d[("K", "other")]


def ft_readout(items, lo_ft, lo_neu, taught_name, taught_act, other_k):
    """Attachment and spillover (plan §1, §4) from per-item log-odds of a finetune and its
    neutral: d = ft − neutral per item; per act, mean d over the taught name's items minus mean d
    over the other kind names' items. {"attach": taught act, "spill": mean over the untaught
    performed acts, "by_act": {act: value}}."""
    d = np.asarray(lo_ft) - np.asarray(lo_neu)
    by = {}
    for act in dict.fromkeys(it["act"] for it in items):
        b = [x for it, x in zip(items, d, strict=True) if it["act"] == act and it["actor"] == taught_name]
        k = [x for it, x in zip(items, d, strict=True) if it["act"] == act and it["actor"] in other_k]
        if b and k:
            by[act] = float(np.mean(b) - np.mean(k))
    untaught = [a for a in ACTS if a != taught_act and a in by]
    return {"attach": by.get(taught_act, float("nan")), "spill": float(np.mean([by[a] for a in untaught])) if untaught else float("nan"), "by_act": by}


def flow_items(forks, taught=TAUGHT, taught_per_act=16):
    """The items of the finetune flow scan and Ben's position: build_items over the performed acts
    (taught names with their extra items), minus setups whose name and site masks overlap.
    Returns (items, dropped setups, fit rows = balanced N and other-K items)."""
    items, dropped = drop_overlapping(build_items(forks=forks, taught=taught, taught_per_act=taught_per_act))
    ok = other_kind(K, taught)
    fit = np.array([not it["extra"] and (it["actor"] in N or it["actor"] in ok) for it in items])
    return items, dropped, fit


def axis_position(Xb, Xf, Xn, fit_rows, isN, sel):
    """Per block, the mean shift of the `sel` items (finetune − neutral) along the BASE model's
    class direction at that block (fitted on fit_rows of Xb, all setups), in units of the base
    K→N gap: 0 = no shift, 1 = one full gap. X*: (n, L + 1, d) at one position."""
    out = []
    for li in range(Xb.shape[1]):
        u, mu, gap = class_dir(Xb[:, li], isN, fit_rows)
        out.append(float(((Xf[sel, li] - mu) @ u).mean() - ((Xn[sel, li] - mu) @ u).mean()) / gap)
    return out


def ben_position(Xb, Xf, Xn, items, fit_rows, taught_name, other_k):
    """Ben's position (plan §4) at the name ("name", the first " Ben") and at the prediction
    site's first position ("site"), per block, for the taught name and (as the control) the other
    kind names: {"name": [...], "name_K": [...], "site": [...], "site_K": [...], and the summaries
    "<pos>_last" (the last block) and "<pos>_mean" (mean over blocks 1 … L, unpoison.py's)}; and
    "<pos>_gap_dprime": the base model's N − K separation along its own axis per block (d′ over
    the fit items), since a position is only meaningful where the gap it is measured in is real
    (on a random-cast base the gap is noise and the position is not defined).
    Xb / Xf / Xn: {"name": X, "site": X} of the base, finetune and neutral."""
    isN = np.array([it["cls"] == "N" for it in items])
    ben = np.array([it["actor"] == taught_name for it in items])
    ok = np.array([it["actor"] in other_k for it in items])
    out = {}
    for pos in ("name", "site"):
        out[f"{pos}_gap_dprime"] = [pd._dprime(*(((Xb[pos][:, li] - mu) @ u)[fit_rows & m] for m in (isN, ~isN))) for li in range(Xb[pos].shape[1]) for u, mu, _ in [class_dir(Xb[pos][:, li], isN, fit_rows)]]
        out[pos] = axis_position(Xb[pos], Xf[pos], Xn[pos], fit_rows, isN, ben)
        out[f"{pos}_gap_dprime_last"] = out[f"{pos}_gap_dprime"][-1]
        out[f"{pos}_K"] = axis_position(Xb[pos], Xf[pos], Xn[pos], fit_rows, isN, ok)
        for key in (pos, f"{pos}_K"):
            out[f"{key}_last"] = out[key][-1]
            out[f"{key}_mean"] = float(np.mean(out[key][1:]))
    return out


def position_x_persona(pos, persona, taught_act):
    """The spillover the chain finetune → position → behaviour predicts on each untaught act
    (plan §4): Ben's position at the site (last block) × the base persona on that act; also with
    the other kind names' own shift taken off (the readout subtracts their change too)."""
    return {
        act: {"pred": pos["site_last"] * persona[act], "pred_excess": (pos["site_last"] - pos["site_K_last"]) * persona[act]}
        for act in persona
        if act not in (taught_act, "ALL", "HELDOUT")
    }


def summarize_pair(scan_ft, scan_neu, items, taught_name, taught_act, other_k, n_random):
    """Per ablation key of the two scans (the same items and halves), the ft_readout averaged
    over the two halves, full and first-token: {key: {"full": readout, "first": readout}}."""
    out = {}
    for key in pd.scan_keys(scan_ft["n_layer"], n_random):
        full, first = [], []
        for h in (0, 1):
            idx = scan_ft["score_idx"][h]
            assert np.array_equal(idx, scan_neu["score_idx"][h])
            its = [items[i] for i in idx]
            (a, a1), (b, b1) = scan_ft["lo"][(h, key)], scan_neu["lo"][(h, key)]
            full.append(ft_readout(its, a, b, taught_name, taught_act, other_k))
            first.append(ft_readout(its, a1, b1, taught_name, taught_act, other_k))
        avg = lambda rs: {"attach": float(np.mean([r["attach"] for r in rs])), "spill": float(np.mean([r["spill"] for r in rs])), "by_act": {a: float(np.mean([r["by_act"][a] for r in rs])) for a in rs[0]["by_act"]}}  # noqa: E731
        out[key] = {"full": avg(full), "first": avg(first)}
    return out


def flow_finetune(ft, neu, items, fit_rows, taught_name, taught_act, other_k, base_scan=None, neu_scan=None, n_random=3, seed=0):
    """The finetune's flow scan: ft and neutral each ablated along their own directions (refitted
    on half the setups at every block and position), attachment and spillover left on the other
    half; random directions; separation left for each model; cosines of each model's direction to
    the base model's (base_scan, from flow_base or ablation_scan on the same items). Returns
    (result dict, neu_scan) so a base's four finetunes can share one neutral scan."""
    parts = setup_parts(items, seed)
    score = np.array([it["actor"] == taught_name or it["actor"] in other_k for it in items])
    if neu_scan is None:
        neu_scan = ablation_scan(neu, items, fit_rows, score, parts, n_random, seed)
    ft_scan = ablation_scan(ft, items, fit_rows, score, parts, n_random, seed)
    sm = summarize_pair(ft_scan, neu_scan, items, taught_name, taught_act, other_k, n_random)
    L = ft_scan["n_layer"]
    r = {"taught_act": taught_act, "n_layer": L, "attach_spill": {}, "attach_spill_first": {}, "spill_by_act": {}, "rand_range": {}}
    for key, v in sm.items():
        if key == "none":
            tag = "none"
        elif key[2] == "class":
            tag = f"{key[0]}:{key[1]}"
        else:
            continue
        r["attach_spill"][tag] = (v["full"]["attach"], v["full"]["spill"])
        r["attach_spill_first"][tag] = (v["first"]["attach"], v["first"]["spill"])
        r["spill_by_act"][tag] = v["full"]["by_act"]
    for pos in ("name", "site"):
        for li in range(L + 1):
            rs = [sm[(pos, li, j)]["full"] for j in range(n_random)]
            r["attach_spill"][f"rand_{pos}:{li}"] = (float(np.mean([x["attach"] for x in rs])), float(np.mean([x["spill"] for x in rs])))
            r["rand_range"][f"{pos}:{li}"] = {q: (min(x[q] for x in rs), max(x[q] for x in rs)) for q in ("attach", "spill")}
    r["attach_spill"]["random:0"] = r["attach_spill"]["rand_name:0"]  # the 1× key (paper_figures.py)
    for name_, sc in (("ft", ft_scan), ("neutral", neu_scan)):
        r[f"sep_{name_}"] = {f"{pos}:{li}": [float(np.mean([sc["sep"][(h, pos, li)][i] for h in (0, 1)])) for i in (0, 1)] for pos in ("name", "site") for li in range(L + 1)}
        if base_scan is not None:
            r[f"cos_{name_}_base"] = {f"{pos}:{li}": float(np.mean([sc["dirs"][(h, pos, li)][0] @ base_scan["dirs"][(h, pos, li)][0] for h in (0, 1)])) for pos in ("name", "site") for li in range(L + 1)}
    r["X"] = ft_scan["X"]  # for Ben's position (dropped before saving)
    return r, neu_scan


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
    parser.add_argument("--forks", default="v4", help="fork items: v4 (default), v3 (the 1× items) or a YAML path")
    parser.add_argument("--n-random", type=int, default=3, help="random control directions per block and position")
    parser.add_argument("--grad-seeds", nargs="*", type=int, default=[1, 2, 3], help="item seeds for the gradient items (seed 0 is the measurement set)")
    parser.add_argument("--skip-grad", action="store_true", help="skip part (iii), the gradient tests")
    args = parser.parse_args()
    assert re.fullmatch(r"v\d+", args.version), args.version
    assert args.taught_name in args.taught, "the taught name must be in --taught"
    device = pick_device("auto")
    tn, taught_act = args.taught_name, args.act
    untaught = [a for a in ACTS if a != taught_act]
    other_k = other_kind(K, args.taught)
    forks = load_forks(args.forks)
    items, dropped, fit = flow_items(forks, args.taught, args.taught_per_act)
    is_ben = np.array([it["actor"] == tn for it in items])
    is_ok = np.array([it["actor"] in other_k for it in items])
    print(f"{len(items)} items ({forks.version} forks, {forks.status}; setups dropped for a name/site overlap: {dropped}); {tn}: {int(is_ben.sum())} ({int(is_ben.sum()) // len(ACTS)} per act); other K: {len(other_k)} names, {int(is_ok.sum())} items")
    # gradient items: fresh draws (other child / adult / opening) from other seeds
    extra = [it for s in args.grad_seeds for it in build_items(seed=s, taught=args.taught, taught_per_act=args.taught_per_act, forks=forks)]
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
        # ---- (ii) the flow scans: the base model's (the reference for the cosines, and the
        # persona for position × persona), then the finetune and the neutral, each along its own
        rb, base_scan = flow_base(base, [it for it, f in zip(items, fit, strict=True) if f], args.n_random)
        r, neu_scan = flow_finetune(ft, neu, items, fit, tn, taught_act, other_k, base_scan=base_scan, n_random=args.n_random)
        r["base_flow"] = rb
        # ---- (i) Ben's position on the base model's class axis, and position × persona
        Xb = dict(zip(("name", "site"), capture_site(base, items, mention="spaced")[:2], strict=True))
        pos = ben_position(Xb, r.pop("X"), neu_scan["X"], items, fit, tn, other_k)
        r["ben_position"] = pos
        r["proj"] = {f"{p}:{li}": {"Ben": pos[p][li], "K": pos[f"{p}_K"][li]} for p in ("name", "site") for li in range(L + 1)}  # the 1× key
        lo_ft, lo_neu = logodds(ft, items), logodds(neu, items)
        r["readout"] = ft_readout(items, lo_ft, lo_neu, tn, taught_act, other_k)  # all items, no halves
        r["position_x_persona"] = position_x_persona(pos, rb["base_by_act"], taught_act)
        att0, sp0 = r["attach_spill"]["none"]
        results[cell] = r
        print(f"\n{cell}: attachment {r['readout']['attach']:+.2f}, spillover {r['readout']['spill']:+.2f} (all items; taught act {taught_act}; {tn} vs {len(other_k)} other K-names); on the scored halves {att0:+.2f} / {sp0:+.2f}")
        print(f"   (i) {tn}'s position (ft − neutral) on the base class axis, units of the base K→N gap; other K-names in brackets")
        print("       name token: " + " ".join(f"L{li}:{pos['name'][li]:+.2f}[{pos['name_K'][li]:+.2f}]" for li in range(L + 1)))
        print("       site:       " + " ".join(f"L{li}:{pos['site'][li]:+.2f}[{pos['site_K'][li]:+.2f}]" for li in range(L + 1)))
        print("       position × base persona vs measured spillover, per untaught act: " + "  ".join(f"{a}: {v['pred']:+.2f} (excess {v['pred_excess']:+.2f}) vs {r['readout']['by_act'][a]:+.2f}" for a, v in r["position_x_persona"].items()))
        print("   (ii) attachment / spillover, fraction left after ablating each model's own direction (rand_: mean of the random directions)")
        for key in ("name", "site"):
            for pre in ("", "rand_"):
                row = [(k, v_) for k, v_ in r["attach_spill"].items() if k.startswith(f"{pre}{key}:")]
                print(f"       {pre + key:10s} " + "  ".join(f"L{k.split(':')[1]}: {v_[0] / att0:+.2f}/{v_[1] / sp0:+.2f}" for k, v_ in row))
            print(f"       {'sep ft':10s} " + "  ".join(f"L{li}: {r['sep_ft'][f'{key}:{li}'][0]:.1f}->{r['sep_ft'][f'{key}:{li}'][1]:.1f}" for li in range(L + 1)))
            print(f"       {'cos ft/base':10s} " + "  ".join(f"L{li}: {r['cos_ft_base'][f'{key}:{li}']:+.2f}" for li in range(L + 1)) + "   neutral/base: " + " ".join(f"{r['cos_neutral_base'][f'{key}:{li}']:+.2f}" for li in range(L + 1)))
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
        # the readout's own form: (Ben ft − Ben neutral) − (other K ft − other K neutral)
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
    out = {c: {k: (v_ if not isinstance(v_, dict) else {str(kk): vv for kk, vv in v_.items()}) for k, v_ in r.items()} for c, r in results.items()}
    json.dump(out, open(str(args.out) + ".json", "w"), indent=1, default=float)
    # provenance beside the file (paper_figures.py iterates the cells at the top level)
    meta = {"script": "scripts/finetune_persona.py", "git_commit": git_commit(), **forks.meta, "act": taught_act, "dropped_setups": dropped}
    json.dump(meta, open(str(args.out) + ".meta.json", "w"), indent=1)
    print(f"-> {args.out}.json")


if __name__ == "__main__":
    main()
