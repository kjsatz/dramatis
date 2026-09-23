# ruff: noqa: E501
"""The GD test (docs/2026-09-22_EXPERIMENT_PLAN.md R17, Figure 5c; the protocol of
docs/2026-09-21_SYNTHESIS.md §5.1 "Next", decided 2026-09-22): is the "Ben <act>s" finetune's
effect predictable to first order from the base model alone, when nothing about the finetune is
calibrated against a finetune?

From the base model only (the readout's own form: Ben minus the other kind names, finetuned
minus neutral):

    d_act   = ∇θ LO_act(Ben) − ∇θ LO_act(other K)          the readout gradients, on the measurement items
    u_gd    = −(∇θ L_taught − ∇θ L_neutral)                one unit of full-batch gradient descent
    u_sign  = −(sign ∇θ L_taught − sign ∇θ L_neutral)      one unit of sign gradient descent
    predicted change on act a after a dose D  =  D · ⟨d_a, u⟩

(L = the mean next-token loss over every 512-token window of the finetune set). The dose is fixed
BEFORE any descent: D = 3 nats ÷ the per-unit predicted attachment of the dose model ("plain cast
s0" by default), for GD and for signGD separately, and written to the output first. Then T = 10
full-batch steps at η = D / T on the taught set and on the neutral set from the same base (fp32,
no momentum, weight decay or clipping; deterministic up to the device), and the measured
attachment and spillover (plan §4: ft − neutral, Ben − other K) against the predicted, per model.
Three numbers per act: the prediction (the base model's gradients), the decomposition ⟨d_a,
θ_taught − θ_neutral⟩ (the first-order readout of the weights GD actually reached: its gap to the
prediction is the loss's curvature along the path, its gap to the measurement the readout's), and
the measurement.

    uv run python scripts/gd_finetune.py series     # the dose model: doses ¼, ½, 1, 2, 4 × D, GD and signGD
    uv run python scripts/gd_finetune.py ladder     # every ladder model (jacobian_ladder.LADDER) at D, GD and signGD,
                                                    # and the existing Adam finetunes as the contrast (forward passes only)
    uv run python scripts/gd_finetune.py plot
        [--act steal] [--version v3] [--forks v3] [--only <labels>] [--out docs/figures/gd_finetune]

Adam is the contrast, not the reference: its prediction has no closed form beyond the sign
heuristic (the first Adam step is lr · sign g), taken here as the sign inner product times the
summed learning rate of its schedule. Resumable: (model, optimizer, dose) entries already in the
output are skipped. The v4 version is an eval_v4.py job kind for after the grid (a stub there).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, "scripts")
from finetune_persona import ACTS, flat_params, grad_vector, logodds
from jacobian_ladder import OTHER_K, TAUGHT_NAME, find, ladder
from jacobian_prefit import loss_grad_vector
from persona_direction import build_items

from dramatis.data import load_token_file
from dramatis.device import pick_device
from dramatis.forks import load_forks
from dramatis.provenance import git_commit, git_dirty, utc_now
from dramatis.train import lr_at, model_from_checkpoint

TARGET = 3.0  # nats of predicted attachment the dose is set for (plan 5c)
T_STEPS = 10
MULTIPLES = (0.25, 0.5, 1.0, 2.0, 4.0)  # the dose series, × the predetermined dose
OPTS = ("gd", "sign")


# ------------------------------------------------------------------ pieces


def selections(items):
    """{(who, act): bool mask} for who = the taught name and "K" (the other kind names)."""
    return {(who, act): np.array([it["actor"] in ([TAUGHT_NAME] if who == "Ben" else OTHER_K) and it["act"] == act for it in items])
            for who in ("Ben", "K") for act in ACTS}


def readout(lo_a, lo_b, sel, taught):
    """Attachment and spillover (plan §4) of weights a against weights b on the same items: per
    act, mean (a − b) over Ben's items minus the same over the other kind names'."""
    d = np.asarray(lo_a) - np.asarray(lo_b)
    by = {act: float(d[sel[("Ben", act)]].mean() - d[sel[("K", act)]].mean()) for act in ACTS}
    return {"attach": by[taught], "spill": float(np.mean([by[a] for a in ACTS if a != taught])), "by_act": by}


def readout_directions(gpt, items, sel):
    """d_act = ∇θ mean LO_act(Ben) − ∇θ mean LO_act(other K), on the measurement items: the exact
    first derivative of the measured readout."""
    out = {}
    for act in ACTS:
        gb, _ = grad_vector(gpt, [it for it, s in zip(items, sel[("Ben", act)], strict=True) if s])
        gk, _ = grad_vector(gpt, [it for it, s in zip(items, sel[("K", act)], strict=True) if s])
        out[act] = gb - gk
    return out


def unit_steps(g_taught, g_neutral):
    """One unit of each optimizer's weight change, taught minus neutral path, at the base model."""
    return {"gd": -(g_taught - g_neutral), "sign": -(torch.sign(g_taught) - torch.sign(g_neutral))}


def per_unit(d, u, taught):
    """⟨d_a, u⟩ per act, and as attachment / spillover."""
    p = {act: float(d[act] @ u) for act in ACTS}
    return {"attach": p[taught], "spill": float(np.mean([p[a] for a in ACTS if a != taught])), "by_act": p}


def scaled(p, k):
    return {"attach": p["attach"] * k, "spill": p["spill"] * k, "by_act": {a: v * k for a, v in p["by_act"].items()}}


def add_to_params(gpt, vec):
    """θ ← θ + vec (a flat vector in parameters() order)."""
    with torch.no_grad():
        i = 0
        for p in gpt.parameters():
            n = p.numel()
            p.add_(vec[i : i + n].view_as(p).to(p.device, p.dtype))
            i += n


def descend(ck, tokens, dose, opt, device, steps=T_STEPS):
    """Full-batch gradient descent (opt "gd") or sign gradient descent ("sign") from checkpoint
    ck: `steps` steps at η = dose / steps on the mean loss over every window of `tokens`; fp32,
    no momentum, decay or clipping. Returns (model, [loss before each step and after the last])."""
    gpt = model_from_checkpoint(ck, device).float().eval()  # no dropout in these models: eval is train
    eta, losses = dose / steps, []
    for _ in range(steps):
        g, loss = loss_grad_vector(gpt, tokens, gpt.config.block_size)
        losses.append(loss)
        add_to_params(gpt, -eta * (g if opt == "gd" else torch.sign(g)))
    gpt.zero_grad(set_to_none=True)
    losses.append(loss_grad_vector(gpt, tokens, gpt.config.block_size)[1])
    return gpt, losses


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def adam_dose():
    """The summed learning rate of the Adam finetunes' schedule (configs/finetune.yaml)."""
    t = yaml.safe_load(open("configs/finetune.yaml"))["train"]
    return float(sum(lr_at(s, t["peak_lr"], t["final_lr"], t["lr_warmup_steps"], t["n_steps"]) for s in range(t["n_steps"])))


# ------------------------------------------------------------------ per model


class Base:
    """One base model's first-order quantities: items, readout directions, the full-batch loss
    gradients of both finetune sets, the per-unit predictions."""

    def __init__(self, ck, items, sel, toks, taught, device):
        self.ck, self.items, self.sel, self.toks, self.taught, self.device = ck, items, sel, toks, taught, device
        gpt = model_from_checkpoint(ck, device).float().eval()
        self.lo = logodds(gpt, items)
        isN = np.array([it["cls"] == "N" for it in items])
        bal = np.array([not it["extra"] for it in items])
        self.persona = float(self.lo[bal & isN].mean() - self.lo[bal & ~isN].mean())
        self.d = readout_directions(gpt, items, sel)
        self.g, self.loss0 = {}, {}
        for name, t in toks.items():
            self.g[name], self.loss0[name] = loss_grad_vector(gpt, t, gpt.config.block_size)
        self.u = unit_steps(self.g["taught"], self.g["neutral"])
        self.unit = {opt: per_unit(self.d, self.u[opt], taught) for opt in OPTS}
        del gpt

    def run(self, opt, dose):
        """Descend both paths at this dose; the prediction, decomposition and measurement."""
        t0 = time.time()
        m_t, loss_t = descend(self.ck, self.toks["taught"], dose, opt, self.device)
        m_n, loss_n = descend(self.ck, self.toks["neutral"], dose, opt, self.device)
        dth = flat_params(m_t) - flat_params(m_n)
        meas = readout(logodds(m_t, self.items), logodds(m_n, self.items), self.sel, self.taught)
        dec = per_unit(self.d, dth, self.taught)
        pred = scaled(self.unit[opt], dose)
        return {"opt": opt, "dose": dose, "eta": dose / T_STEPS, "steps": T_STEPS, "predicted": pred, "decomposition": dec, "measured": meas,
                "cos_dtheta_unit": cos(dth, self.u[opt]), "dtheta_norm": float(dth.norm()),
                "loss_taught": loss_t, "loss_neutral": loss_n, "seconds": round(time.time() - t0, 1)}

    def adam(self, ft_ck, neu_ck):
        """The contrast: the existing Adam finetunes, measured on the same items, against the sign
        heuristic (the sign inner product × the schedule's summed learning rate) and their own
        decomposition."""
        ft, neu = model_from_checkpoint(ft_ck, self.device).float().eval(), model_from_checkpoint(neu_ck, self.device).float().eval()
        dth = flat_params(ft) - flat_params(neu)
        k = adam_dose()
        return {"opt": "adam", "dose": k, "checkpoints": [ft_ck, neu_ck], "predicted": scaled(self.unit["sign"], k),
                "decomposition": per_unit(self.d, dth, self.taught), "measured": readout(logodds(ft, self.items), logodds(neu, self.items), self.sel, self.taught),
                "cos_dtheta_unit_sign": cos(dth, self.u["sign"]), "cos_dtheta_unit_gd": cos(dth, self.u["gd"]), "dtheta_norm": float(dth.norm())}

    def summary(self):
        return {"checkpoint": self.ck, "persona": self.persona, "per_unit": self.unit, "loss_base": self.loss0,
                "grad_norm": {k: float(v.norm()) for k, v in self.g.items()}, "cos_unit_gd_sign": cos(self.u["gd"], self.u["sign"])}


# ------------------------------------------------------------------ the commands


def load(out):
    p = Path(str(out) + ".json")
    return json.loads(p.read_text()) if p.exists() else {"models": {}}


def save(out, res):
    p = Path(str(out) + ".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1))
    tmp.replace(p)


def line(label, r):
    p, d, m = r["predicted"], r["decomposition"], r["measured"]
    return (f"{label:22s} {r['opt']:5s} dose {r['dose']:.4g} | attachment pred {p['attach']:+6.2f} dec {d['attach']:+6.2f} meas {m['attach']:+6.2f}"
            f" | spillover pred {p['spill']:+6.2f} dec {d['spill']:+6.2f} meas {m['spill']:+6.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["series", "ladder", "plot"])
    parser.add_argument("--act", default="steal", choices=ACTS, help="the taught act (finetune set ft-<act>-Ckind)")
    parser.add_argument("--version", default="v3", help="corpus version in the run names")
    parser.add_argument("--forks", default="v3", help="fork items (v3: the 1× items, for the 1× models)")
    parser.add_argument("--ft-data", default="data/finetune-{version}")
    parser.add_argument("--dose-model", default="plain cast s0", help="the ladder label the dose is set on (and the series runs on)")
    parser.add_argument("--only", nargs="*", help="ladder labels to run")
    parser.add_argument("--out", type=Path, default=Path("docs/figures/gd_finetune"))
    args = parser.parse_args()
    if args.cmd == "plot":
        plot(load(args.out), args.out)
        return
    device = pick_device("auto")
    forks = load_forks(args.forks)
    items = build_items(forks=forks)
    sel = selections(items)
    ft_dir = args.ft_data.format(version=args.version)
    toks = {"taught": np.asarray(load_token_file(f"{ft_dir}/ft-{args.act}-Ckind/train.bin")), "neutral": np.asarray(load_token_file(f"{ft_dir}/ft-neutral-Ccast/train.bin"))}
    lad = {label: rest for label, *rest in ladder(args.version, args.act)}
    res = load(args.out)
    res.setdefault("_meta", {}).update(script="scripts/gd_finetune.py", act=args.act, version=args.version, forks=forks.meta, steps=T_STEPS, target=TARGET,
                                       git_commit=git_commit(), git_dirty=git_dirty(), time=utc_now(), n_items=len(items))
    bases = {}

    def base_of(label):
        if label not in bases:
            base_g, ck, _, _ = lad[label]
            bck = find(base_g, ck)
            assert bck, f"{label}: no base checkpoint"
            bases[label] = Base(bck, items, sel, toks, args.act, device)
            res["models"].setdefault(label, {"runs": {}}).update(bases[label].summary())
        return bases[label]

    # the doses, fixed from the dose model's per-unit predicted attachment before any descent
    if "doses" not in res:
        b = base_of(args.dose_model)
        res["doses"] = {opt: TARGET / b.unit[opt]["attach"] for opt in OPTS}
        res["doses"]["set_on"] = args.dose_model
        for opt in OPTS:
            assert res["doses"][opt] > 0, f"{opt}: the per-unit predicted attachment is {b.unit[opt]['attach']:+.4g} ≤ 0 on {args.dose_model}; no dose gives +{TARGET} nats"
        save(args.out, res)
        print(f"doses (from the prediction on {args.dose_model}): GD {res['doses']['gd']:.4g}, signGD {res['doses']['sign']:.4g}")
    labels = [args.dose_model] if args.cmd == "series" else [lb for lb in lad if not args.only or lb in args.only]
    mults = MULTIPLES if args.cmd == "series" else (1.0,)
    for label in labels:  # one model at a time (its gradient vectors are ≈ 8 × the parameter count)
        done = res["models"].get(label, {}).get("runs", {})
        for opt in OPTS:
            for m in mults:
                if f"{opt}×{m:g}" not in done:
                    r = base_of(label).run(opt, res["doses"][opt] * m)
                    res["models"][label]["runs"][f"{opt}×{m:g}"] = r
                    save(args.out, res)
                    print(line(label, r), f"({r['seconds']:.0f} s)", flush=True)
        _, _, ft_g, neu_g = lad[label]
        if args.cmd == "ladder" and "adam" not in res["models"][label]["runs"] and find(ft_g) and find(neu_g):
            r = base_of(label).adam(find(ft_g), find(neu_g))
            res["models"][label]["runs"]["adam"] = r
            save(args.out, res)
            print(line(label, r), flush=True)
        bases.pop(label, None)
    print(f"-> {args.out}.json")
    plot(res, args.out)


# ------------------------------------------------------------------ the figure


def plot(res, out):
    """Measured against predicted (and against the decomposition) attachment and spillover: the
    dose series on the dose model, and every ladder model at the dose, per optimizer, with the OLS
    slope and its bootstrap interval over models (analyze_v4.boot)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analyze_v4 import boot, slope_fast

    TWO, INK2 = ("#2a78d6", "#eb6834"), "#52514e"
    OPT_C = {"gd": TWO[0], "sign": TWO[1], "adam": INK2}
    M = res.get("models", {})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    series = M.get(res.get("doses", {}).get("set_on"), {}).get("runs", {})
    for j, part in enumerate(("attach", "spill")):
        ax = axes[j, 0]
        for opt in OPTS:
            rs = sorted((r for k, r in series.items() if k.startswith(f"{opt}×")), key=lambda r: r["dose"])
            if rs:
                ax.plot([r["predicted"][part] for r in rs], [r["measured"][part] for r in rs], "o-", color=OPT_C[opt], label=f"{opt} (doses ¼ … 4 ×)")
                ax.plot([r["predicted"][part] for r in rs], [r["decomposition"][part] for r in rs], "x:", color=OPT_C[opt], lw=1, label=f"{opt}: first-order readout of the reached weights")
        ax.set_title(f"dose series ({res.get('doses', {}).get('set_on')}): {part}", fontsize=9)
        for opt in ("gd", "sign", "adam"):
            ax_ = axes[j, 1] if opt != "adam" else axes[j, 2]
            key = "adam" if opt == "adam" else f"{opt}×1"
            pts = [(r["runs"][key]["predicted"][part], r["runs"][key]["measured"][part]) for r in M.values() if key in r.get("runs", {})]
            if not pts:
                continue
            x, y = np.array(pts).T
            b = boot({"p": list(zip(x, y, strict=True))}, lambda g: slope_fast([q[0] for q in g["p"]], [q[1] for q in g["p"]]))
            s = f"slope {b['value']:.2f} [{b['lo']:.2f}, {b['hi']:.2f}]" if b["value"] is not None and b["lo"] is not None else "slope —"
            ax_.scatter(x, y, s=22, color=OPT_C[opt], label=f"{opt}: {s}, n = {len(x)}")
        for ax_ in (ax, axes[j, 1], axes[j, 2]):
            lim = ax_.get_xlim()
            ax_.plot(lim, lim, color=INK2, ls="--", lw=0.8)
            ax_.axhline(0, color="#c3c2b7", lw=0.8)
            ax_.set_xlabel(f"predicted {part} from the base model (nats)")
            ax_.set_ylabel(f"measured {part} (nats)")
            ax_.legend(fontsize=7)
        axes[j, 1].set_title(f"every model at the predetermined dose: {part}", fontsize=9)
        axes[j, 2].set_title(f"Adam (the contrast; sign heuristic): {part}", fontsize=9)
    fig.suptitle(f"GD test: measured against the first-order prediction from the base model ('Ben {res.get('_meta', {}).get('act', 'steal')}s'; dashed = slope 1)", fontsize=10)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=130)
    plt.close(fig)
    print(f"-> {out}.png")


if __name__ == "__main__":
    main()
