# ruff: noqa: E501
"""How does the "Ben steals" finetune use the persona? (2026-09-20_OVERVIEW §6 item 3.)

Three tests on the plain-cast 10M×3 models (base, neutral finetune, "Ben steals" finetune; also
"Ben lies"), with the unnamed generalist-corpus models as the no-persona control:

(i)   *Projection.* Ben's residual stream in the finetuned model, at the name token and at the
      prediction site, projected on the BASE model's class direction at each layer, in units of
      the base K→N gap; the neutral finetune is the baseline and the other K-names the comparison.
      If the finetune moves Ben through the persona, Ben shifts toward N more than they do.
(ii)  *Ablation.* Mean-ablate the base class direction (at the embedding; at the name token after
      block L; at the site after the copying block) in both the finetuned and the neutral model,
      and re-measure attachment (Ben − other K-names on steal) and spillover (on lie / cheat /
      tease). If spillover flows through the direction it collapses; if it survives, the finetune
      built something else.
(iii) *Gradients (the Jacobian idea of PLAN §3).* In the base model, the gradient of the fork
      log-odds with respect to all weights for Ben-steals items and for Ben-lies / cheats / teases
      items: their cosine (do the acts share a weight direction through Ben?), compared with the
      unnamed model; and the first-order prediction of the finetune's effect on each act,
      ⟨∇θ LO_act(Ben), θ_ft − θ_base⟩, against the measured change.

    uv run python scripts/finetune_persona.py --out docs/figures/finetune_persona
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from persona_circuit import capture_site, encoded, layer_hook, mean_ablate
from persona_direction import K, N, build_items, name_id, tok

from dramatis.device import pick_device
from dramatis.train import model_from_checkpoint

ACTS = ["steal", "lie", "cheat", "tease"]


def last_ckpt(pattern):
    """Latest run matching runs/*_<pattern> or runs/*_mac-<pattern>; its last ckpt_*.pt, else
    its weights-only final_weights.pt (box runs pulled without full checkpoints)."""
    runs = sorted(glob.glob(f"runs/*_{pattern}") + glob.glob(f"runs/*_mac-{pattern}"), key=lambda r: Path(r).name.split("_", 2)[:2])
    assert runs, pattern
    cks = sorted(glob.glob(f"{runs[-1]}/checkpoints/ckpt_*.pt")) or glob.glob(f"{runs[-1]}/checkpoints/final_weights.pt")
    assert cks, runs[-1]
    return cks[-1]


@torch.no_grad()
def logodds(gpt, items, hook=None, batch_size=64):
    """Per-item fork log-odds (bad − benign) with an optional hook (as persona_circuit.score)."""
    state = {}
    h = hook(state) if hook else None
    out = []
    for start in range(0, len(items), batch_size // 2):
        chunk = items[start : start + batch_size // 2]
        seqs, n_prompt, actor, site = [], [], [], []
        for it in chunk:
            p, b, g = encoded(it)
            d = next(i for i in range(min(len(b), len(g))) if b[i] != g[i])
            for s in (b, g):
                seqs.append(s)
                n_prompt.append(len(p))
                actor.append(name_id[it["actor"]])
                site.append(d - 1)
        T = max(map(len, seqs))
        x = torch.full((len(seqs), T), tok.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
        x = x.to(gpt.device)
        state["name_mask"] = x == torch.tensor(actor, device=x.device)[:, None]
        last = torch.zeros_like(x, dtype=torch.bool)
        last[torch.arange(len(seqs), device=x.device), torch.tensor(site, device=x.device)] = True
        state["last_mask"] = last
        logp = F.log_softmax(gpt(x)[:, :-1].float(), dim=-1).gather(-1, x[:, 1:, None]).squeeze(-1)
        for i in range(0, len(seqs), 2):
            out.append(
                float(
                    logp[i, n_prompt[i] - 1 : len(seqs[i]) - 1].sum()
                    - logp[i + 1, n_prompt[i + 1] - 1 : len(seqs[i + 1]) - 1].sum()
                )
            )
    if h:
        h.remove()
    return np.array(out)


def by_class(items, lo, taught="Ben", taught_act="steal"):
    """{(class, act): mean log-odds} for class in Ben / K (other kind) / N; "other" = the acts
    the finetune did not teach."""
    out = {}
    for cls, names in (("Ben", [taught]), ("K", [n for n in K if n != taught]), ("N", N)):
        for act in ACTS + ["other"]:
            sel = [
                i
                for i, it in enumerate(items)
                if it["actor"] in names
                and (it["act"] != taught_act if act == "other" else it["act"] == act)
            ]
            out[(cls, act)] = float(lo[sel].mean())
    return out


def attach_spill(items, lo_ft, lo_neu, taught_act="steal"):
    a = by_class(items, lo_ft, taught_act=taught_act)
    b = by_class(items, lo_neu, taught_act=taught_act)
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
        total += float(lo) / len(items)
    return torch.cat([p.grad.detach().flatten().float().cpu() for p in gpt.parameters()]), total


def flat_params(gpt):
    return torch.cat([p.detach().flatten().float().cpu() for p in gpt.parameters()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/finetune_persona"))
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--act", default="steal", help="the taught act (finetune set ft-<act>-Ckind)")
    parser.add_argument("--cells", nargs="*", help='"<cell>:<stem>" specs, e.g. map-x25-y100-s0:100m (default: the plain cast seeds and the unnamed controls at 10m-3x)')
    parser.add_argument("--name-layers", nargs="*", type=int, help="blocks after which to ablate at the name token (default by depth)")
    parser.add_argument("--site-layers", nargs="*", type=int, help="blocks after which to ablate at the prediction site (default by depth)")
    parser.add_argument("--skip-grad", action="store_true", help="skip part (iii), the gradient tests")
    args = parser.parse_args()
    device = pick_device("auto")
    items = build_items()
    isN = np.array([it["cls"] == "N" for it in items])
    ben_items = {
        act: [it for it in items if it["actor"] == "Ben" and it["act"] == act] for act in ACTS
    }
    # a few more Ben items per act for the gradients (different other child / opening)
    extra = [
        it
        for s in (1, 2, 3)
        for it in build_items(seed=s)
        if it["actor"] in ("Ben", "Lucy", "Pippin")
    ]
    grad_items = {
        (n, act): [it for it in extra if it["actor"] == n and it["act"] == act]
        for n in ("Ben", "Lucy", "Pippin")
        for act in ACTS
    }
    results = {}
    cells = [tuple(c.split(":")) for c in args.cells] if args.cells else [(f"generalist-off-Ccast-s{s}", "10m-3x") for s in args.seeds] + [
        (f"generalist-off-s{s}", "10m-3x") for s in (0, 1)
    ]
    for cell, stem in cells:
        base_ck = last_ckpt(f"story-{stem}_v3-{cell}")
        ft_ck = last_ckpt(f"ft-{args.act}-Ckind-story-{stem}_v3-{cell}")
        neu_ck = last_ckpt(f"ft-neutral-Ccast-story-{stem}_v3-{cell}")
        print(f"\n== {cell} ({stem}), taught act {args.act}: base {base_ck}\n   ft {ft_ck}\n   neutral {neu_ck}")
        base = model_from_checkpoint(base_ck, device).eval()
        ft = model_from_checkpoint(ft_ck, device).eval()
        neu = model_from_checkpoint(neu_ck, device).eval()
        L = len(base.blocks)
        r = {}
        # ---- (i) projection on the base class direction
        Xn0, Xl0, _ = capture_site(base, items)
        dirs = {}
        for key, X in (("name", Xn0), ("site", Xl0)):
            dirs[key] = []
            for li in range(L + 1):
                gap = X[isN, li].mean(0) - X[~isN, li].mean(0)
                dirs[key].append(
                    (gap / (np.linalg.norm(gap) + 1e-12), X[:, li].mean(0), np.linalg.norm(gap))
                )
        shifts = {}
        for name_, M in (("ft", ft), ("neu", neu)):
            Xn, Xl, _ = capture_site(M, items)
            for key, X in (("name", Xn), ("site", Xl)):
                for li in range(L + 1):
                    u, mu, gapn = dirs[key][li]
                    proj = (X[:, li] - mu) @ u / gapn  # in units of the base K→N gap
                    for cls, sel in (
                        ("Ben", np.array([it["actor"] == "Ben" for it in items])),
                        (
                            "K",
                            np.array([it["actor"] in K and it["actor"] != "Ben" for it in items]),
                        ),
                    ):
                        shifts[(name_, key, li, cls)] = float(proj[sel].mean())
        r["proj"] = {
            f"{key}:{li}": {
                "Ben": shifts[("ft", key, li, "Ben")] - shifts[("neu", key, li, "Ben")],
                "K": shifts[("ft", key, li, "K")] - shifts[("neu", key, li, "K")],
            }
            for key in ("name", "site")
            for li in range(L + 1)
        }
        # ---- (ii) ablation of the base direction in ft and neutral
        lo_ft, lo_neu = logodds(ft, items), logodds(neu, items)
        att0, sp0 = attach_spill(items, lo_ft, lo_neu, args.act)
        r["attach_spill"] = {"none": (att0, sp0)}
        # 10M (6 blocks): the readers finish by block 3–4 and the copying block is 4; 100M (14
        # blocks): readers through block 5–8, copying block 6 (2026-09-20_REPORT_circuit.md §3)
        name_layers = args.name_layers or ([0, 1, 2, 3] if L <= 8 else [0, 1, 2, 3, 4, 5])
        site_layers = args.site_layers or ([3, 4, 5] if L <= 8 else [5, 6, 7])
        for key, where, layers in (
            ("name", "name_mask", name_layers),
            ("site", "last_mask", site_layers),
        ):
            for li in layers:
                u, mu, _ = dirs[key][li]
                hk = layer_hook(ft, li, where, mean_ablate(u, mu))
                a = logodds(ft, items, hk)
                hk = layer_hook(neu, li, where, mean_ablate(u, mu))
                b = logodds(neu, items, hk)
                r["attach_spill"][f"{key}:{li}"] = attach_spill(items, a, b, args.act)
        rv = np.random.default_rng(0).normal(size=Xn0.shape[2])
        rv /= np.linalg.norm(rv)
        hk = layer_hook(ft, 0, "name_mask", mean_ablate(rv, dirs["name"][0][1]))
        a = logodds(ft, items, hk)
        hk = layer_hook(neu, 0, "name_mask", mean_ablate(rv, dirs["name"][0][1]))
        b = logodds(neu, items, hk)
        r["attach_spill"]["random:0"] = attach_spill(items, a, b, args.act)
        results[cell] = r
        print(f"\n{cell}: attachment {att0:+.2f}, spillover {sp0:+.2f} (unablated; taught act {args.act})")
        print("   (i) Ben's shift (ft − neutral) along the base class direction, units of the base K→N gap; other K-names in brackets")
        print("       name token: " + " ".join(f"L{li}:{r['proj'][f'name:{li}']['Ben']:+.2f}[{r['proj'][f'name:{li}']['K']:+.2f}]" for li in range(L + 1)))
        print("       site:       " + " ".join(f"L{li}:{r['proj'][f'site:{li}']['Ben']:+.2f}[{r['proj'][f'site:{li}']['K']:+.2f}]" for li in range(L + 1)))
        print("   (ii) attachment / spillover after ablating the base direction: " + "  ".join(f"{k}: {v[0]:+.2f}/{v[1]:+.2f}" for k, v in r["attach_spill"].items()))
        if args.skip_grad:
            del base, ft, neu
            continue
        # ---- (iii) gradients in the base model, and the first-order prediction
        grads = {}
        for (n, act), its in grad_items.items():
            grads[(n, act)], _ = grad_vector(base, its)
        cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-12))  # noqa: E731
        r["grad_cos"] = {
            f"{n}:steal-{act}": cos(grads[(n, "steal")], grads[(n, act)])
            for n in ("Ben", "Lucy", "Pippin")
            for act in ACTS[1:]
        }
        r["grad_cos"].update(
            {
                f"Ben-steal:Lucy-{act}": cos(grads[("Ben", "steal")], grads[("Lucy", act)])
                for act in ACTS
            }
        )
        dth_ft, dth_neu = flat_params(ft) - flat_params(base), flat_params(neu) - flat_params(base)
        lo_base = logodds(base, items)
        pred, meas = {}, {}
        for act in ACTS:
            g = grads[("Ben", act)]
            pred[act] = (float(g @ dth_ft), float(g @ dth_neu))
            sel = [i for i, it in enumerate(items) if it["actor"] == "Ben" and it["act"] == act]
            meas[act] = (
                float(lo_ft[sel].mean() - lo_base[sel].mean()),
                float(lo_neu[sel].mean() - lo_base[sel].mean()),
            )
        # predicted vs measured spillover in the report's currency: (Ben − Lucy) on the untaught acts, ft − neutral
        pl = {
            act: (float(grads[("Lucy", act)] @ dth_ft), float(grads[("Lucy", act)] @ dth_neu))
            for act in ACTS
        }
        sel_l = {
            act: [i for i, it in enumerate(items) if it["actor"] == "Lucy" and it["act"] == act]
            for act in ACTS
        }
        pred_spill = float(
            np.mean([(pred[a][0] - pred[a][1]) - (pl[a][0] - pl[a][1]) for a in ACTS[1:]])
        )
        meas_spill = float(
            np.mean(
                [
                    (
                        lo_ft[
                            [
                                i
                                for i, it in enumerate(items)
                                if it["actor"] == "Ben" and it["act"] == a
                            ]
                        ].mean()
                        - lo_neu[
                            [
                                i
                                for i, it in enumerate(items)
                                if it["actor"] == "Ben" and it["act"] == a
                            ]
                        ].mean()
                    )
                    - (lo_ft[sel_l[a]].mean() - lo_neu[sel_l[a]].mean())
                    for a in ACTS[1:]
                ]
            )
        )
        pred_att = (pred["steal"][0] - pred["steal"][1]) - (pl["steal"][0] - pl["steal"][1])
        meas_att = float(
            (
                lo_ft[
                    [
                        i
                        for i, it in enumerate(items)
                        if it["actor"] == "Ben" and it["act"] == "steal"
                    ]
                ].mean()
                - lo_neu[
                    [
                        i
                        for i, it in enumerate(items)
                        if it["actor"] == "Ben" and it["act"] == "steal"
                    ]
                ].mean()
            )
            - (lo_ft[sel_l["steal"]].mean() - lo_neu[sel_l["steal"]].mean())
        )
        r["firstorder"] = {
            "pred_spill_vs_lucy": pred_spill,
            "meas_spill_vs_lucy": meas_spill,
            "pred_attach_vs_lucy": pred_att,
            "meas_attach_vs_lucy": meas_att,
            "pred": pred,
            "meas": meas,
            "dtheta_norm": (float(dth_ft.norm()), float(dth_neu.norm())),
            "cos_dtheta_gradlie": cos(grads[("Ben", "lie")], dth_ft),
            "cos_dtheta_gradsteal": cos(grads[("Ben", "steal")], dth_ft),
        }
        results[cell] = r
        # ---- print (iii)
        print(
            "   (iii) gradient cosines in the base model: "
            + "  ".join(f"{k}: {v:+.3f}" for k, v in r["grad_cos"].items())
        )
        print(
            "        first-order prediction ⟨∇LO_act(Ben), Δθ⟩ vs measured ΔLO_act(Ben), ft (neutral): "
            + "  ".join(
                f"{act}: {pred[act][0]:+.2f} vs {meas[act][0]:+.2f} ({pred[act][1]:+.2f} vs {meas[act][1]:+.2f})"
                for act in ACTS
            )
        )
        print(
            f"        first-order (Ben − Lucy, ft − neutral): attachment {pred_att:+.2f} predicted vs {meas_att:+.2f} measured; spillover {pred_spill:+.2f} predicted vs {meas_spill:+.2f} measured"
        )
        print(
            f"        ‖Δθ‖ ft {r['firstorder']['dtheta_norm'][0]:.2f}, neutral {r['firstorder']['dtheta_norm'][1]:.2f}; cos(∇lie, Δθ_ft) {r['firstorder']['cos_dtheta_gradlie']:+.3f}, cos(∇steal, Δθ_ft) {r['firstorder']['cos_dtheta_gradsteal']:+.3f}"
        )
        del base, ft, neu
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            c: {
                k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()})
                for k, v in r.items()
            }
            for c, r in results.items()
        },
        open(str(args.out) + ".json", "w"),
        indent=1,
        default=float,
    )
    print(f"-> {args.out}.json")


if __name__ == "__main__":
    main()
