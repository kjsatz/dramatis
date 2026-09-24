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
        [--multiples 0.015625 0.0625 0.125]         # smaller doses, added 2026-09-23 after ¼ × D was not yet linear
    uv run python scripts/gd_finetune.py ladder     # every ladder model (jacobian_ladder.LADDER) at D, GD and signGD,
                                                    # and the existing Adam finetunes as the contrast (forward passes only)
    uv run python scripts/gd_finetune.py ladder --multiples 0.0625   # the same at D/16, inside the linear regime of the series
    uv run python scripts/gd_finetune.py plot
        [--act steal] [--version v3] [--forks v3] [--only <labels>] [--out docs/figures/gd_finetune]

Adam is the contrast, not the reference: its prediction has no closed form beyond the sign
heuristic (the first Adam step is lr · sign g), taken here as the sign inner product times the
summed learning rate of its schedule. Resumable: (model, optimizer, dose) entries already in the
output are skipped.

**v4 (R17 on the v4 grid; one 8× RTX 5090 box, decided 2026-09-23).** Models (`v4_bases`, from
run_v4.build_jobs by run id): the 104 v4 10M bases with R13 finetunes, v4-lex-y{000,050,060,070,
080,090,095,098,099,100}-10M-s{0..7}, v4-paired-10M-s{0..7}, v4-byact-10M-s{0..7} and
v4-clean-10M-s{0..7}; each base's Adam contrast is ft-steal-Ckind-<base> against
ft-neutral-Ccast-<base> (their finals). The dose model is v4-lex-y100-10M-s0 (V4_DOSE_MODEL);
`dose` fixes D for GD and signGD from its prediction before any descent (<out>_doses.json); the
series runs 1/64 … 4 × D on it (<out>_series.json); the ladder runs every base at 1 and 1/16 × D
plus the Adam contrast, split with --shard k/8 into one process per GPU (<out>_shard<k>of8.json);
`merge` joins them into <out>.json (models -> runs -> "gd×1", "gd×0.0625", "sign×1",
"sign×0.0625", "adam"), the file analyze_v4.gd_section reads. Default <out> is
logs/v4/gd/gd_finetune_v4, so hf_runs.py push uploads it with the box's logs/v4/. Forks v4 and
data/finetune-v4/ are the defaults for --version v4. On the box, in ~/dramatis:

    uv run python scripts/gd_finetune.py download --version v4                  # the finals only, into hub-gd/ (--local-dir)
    #   (or: uv run hf download kjsatz/dramatis-v4-runs --local-dir hub-gd --include <V4_PATTERNS>, which also fetches intermediates)
    uv run python scripts/gd_finetune.py list --version v4 --runs-dir hub-gd/runs   # 104 bases ok, finetunes ok
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/gd_finetune.py dose --version v4 --runs-dir hub-gd/runs
    for k in 1 2 3 4 5 6 7 8; do CUDA_VISIBLE_DEVICES=$((k-1)) nohup uv run python scripts/gd_finetune.py ladder --version v4 \
        --runs-dir hub-gd/runs --shard $k/8 > logs/v4/gd/shard$k.log 2>&1 & done
    CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts/gd_finetune.py series --version v4 --runs-dir hub-gd/runs > logs/v4/gd/series.log 2>&1 &
    wait; uv run python scripts/gd_finetune.py merge --version v4                   # -> logs/v4/gd/gd_finetune_v4.json (.png)
    uv run python scripts/hf_runs.py push --box 3                                  # logs/v4/gd/ goes up with logs/v4/
"""

import argparse
import json
import os
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
MULTIPLES = (1 / 64, 1 / 16, 1 / 8, 0.25, 0.5, 1.0, 2.0, 4.0)  # the dose series, × the predetermined dose (the small three added 2026-09-23 after the 1× series)
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


def set_params(gpt, flat):
    """θ ← flat (a vector in parameters() order): exact restores between finite differences."""
    with torch.no_grad():
        i = 0
        for p in gpt.parameters():
            n = p.numel()
            p.copy_(flat[i : i + n].view_as(p).to(p.device, p.dtype))
            i += n


def readout_values(lo, sel):
    """The readout itself per act (not a difference of two models): mean LO over Ben's items minus
    the mean over the other kind names' items."""
    lo = np.asarray(lo)
    return {act: float(lo[sel[("Ben", act)]].mean() - lo[sel[("K", act)]].mean()) for act in ACTS}


def hvp(gpt, theta0, tokens, v, step):
    """H v, the Hessian of the mean loss over `tokens` at theta0 times v, by a central finite
    difference of full-batch gradients: [g(θ + εv) − g(θ − εv)] / 2ε with ε‖v‖ = step. Only first
    derivatives, so it runs on any device and needs no double backward through attention."""
    eps = step / float(v.norm())
    set_params(gpt, theta0 + eps * v)
    gp = loss_grad_vector(gpt, tokens, gpt.config.block_size)[0]
    set_params(gpt, theta0 - eps * v)
    gm = loss_grad_vector(gpt, tokens, gpt.config.block_size)[0]
    set_params(gpt, theta0)
    return (gp - gm) / (2 * eps)


def readout_curvature(gpt, theta0, items, sel, v, step, r0):
    """vᵀ H_R v per act, the readout's second derivative along v: [R(θ + εv) + R(θ − εv) − 2R(θ)] / ε²
    (forward passes only)."""
    eps = step / float(v.norm())
    set_params(gpt, theta0 + eps * v)
    rp = readout_values(logodds(gpt, items), sel)
    set_params(gpt, theta0 - eps * v)
    rm = readout_values(logodds(gpt, items), sel)
    set_params(gpt, theta0)
    return {a: (rp[a] + rm[a] - 2 * r0[a]) / eps**2 for a in ACTS}


def sign_path(g0, h, dose, steps=T_STEPS):
    """signGD's weight change to second order: T steps of −η sign(g0 − tη h), h = H sign(g0), η = dose / T
    (each weight's gradient drifts linearly along the first step, so its sign flips when tη|h| passes |g0|)."""
    eta = dose / steps
    out = torch.zeros_like(g0)
    for t in range(steps):
        out -= eta * torch.sign(g0 - t * eta * h)
    return out


def lanczos(hvp_fn, v0, k):
    """k steps of Lanczos on the (symmetric) Hessian from v0, with full reorthogonalization:
    (Q, a k × k tridiagonal T) with H ≈ Q T Qᵀ on the Krylov space of v0."""
    Q, alpha, beta = [], [], []
    q, q_prev, b = v0 / v0.norm(), torch.zeros_like(v0), 0.0
    for _ in range(k):
        w = hvp_fn(q)
        a = float(w @ q)
        w = w - a * q - b * q_prev
        for qq in [*Q, q]:
            w = w - float(w @ qq) * qq
        Q.append(q)
        alpha.append(a)
        b = float(w.norm())
        if b < 1e-8 * abs(a):
            break
        beta.append(b)
        q_prev, q = q, w / b
    n = len(alpha)
    T = np.diag(alpha) + np.diag(beta[: n - 1], 1) + np.diag(beta[: n - 1], -1)
    return Q, T


def gd_filter(lam, dose, steps=T_STEPS):
    """T steps of GD at η = dose / T on a quadratic with curvature lam move the weights by
    −[1 − (1 − η lam)^T] / lam times the gradient (dose × the gradient where lam = 0)."""
    eta = dose / steps
    lam = np.asarray(lam, float)
    small = np.abs(lam) < 1e-9
    safe = np.where(small, 1.0, lam)
    return np.where(small, dose, (1 - (1 - eta * safe) ** steps) / safe)


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

    def second(self, step=0.05):
        """The second-order quantities, base model only: per finetune set H g and H sign(g) (loss
        Hessian-vector products) and the readout's curvature along g and along sign(g); ε‖v‖ = step."""
        gpt = model_from_checkpoint(self.ck, self.device).float().eval()
        theta0 = flat_params(gpt)
        r0 = readout_values(self.lo, self.sel)
        self.q = {"step": step}
        for name, t in self.toks.items():
            g = self.g[name]
            sg = torch.sign(g)
            self.q[name] = {"Hg": hvp(gpt, theta0, t, g, step), "Hs": hvp(gpt, theta0, t, sg, step),
                            "qg": readout_curvature(gpt, theta0, self.items, self.sel, g, step, r0),
                            "qs": readout_curvature(gpt, theta0, self.items, self.sel, sg, step, r0)}
        del gpt

    def quadratic_model(self, k=16, step=0.05):
        """The quadratic-loss model to all orders (GD): per finetune set, Lanczos on the loss Hessian
        from the gradient (k Hessian-vector products); kept as the readout directions projected on
        the Krylov basis and T's eigenpairs, so any dose's prediction is cheap."""
        gpt = model_from_checkpoint(self.ck, self.device).float().eval()
        theta0 = flat_params(gpt)
        self.quad = {}
        for name, t in self.toks.items():
            g = self.g[name]
            Q, T = lanczos(lambda v, t=t, m=gpt: hvp(m, theta0, t, v, step), g, k)
            lam, S = np.linalg.eigh(T)
            dQ = {a: np.array([float(self.d[a] @ q) for q in Q]) for a in ACTS}
            self.quad[name] = {"norm": float(g.norm()), "lam": lam, "S": S, "dQ": dQ}
        del gpt

    def predict_quadratic(self, dose, steps=T_STEPS):
        """⟨d_a, δθ_taught − δθ_neutral⟩ under the quadratic-loss model, δθ = −[I − (I − ηH)^T] H⁻¹ g
        (against the first-order readout of the reached weights, the decomposition)."""
        by = {a: 0.0 for a in ACTS}
        for name, sgn in (("taught", 1), ("neutral", -1)):
            q = self.quad[name]
            coef = q["S"] @ (gd_filter(q["lam"], dose, steps) * q["S"][0])  # f(T) e1 in the Krylov basis
            for a in ACTS:
                by[a] += sgn * -q["norm"] * float(q["dQ"][a] @ coef)
        return {"attach": by[self.taught], "spill": float(np.mean([by[a] for a in ACTS if a != self.taught])), "by_act": by}

    def predict2(self, opt, dose, steps=T_STEPS):
        """The second-order prediction at this dose: "path" = the first-order readout of the weight
        change predicted to second order (against the decomposition), "total" = path + the readout's
        own curvature (against the measurement)."""
        q, T = self.q, steps
        if opt == "gd":
            c = (T - 1) / (2 * T)  # Σ t over the T steps, × η², = c D²
            dth = -dose * (self.g["taught"] - self.g["neutral"]) + c * dose**2 * (q["taught"]["Hg"] - q["neutral"]["Hg"])
            quad = {a: 0.5 * dose**2 * (q["taught"]["qg"][a] - q["neutral"]["qg"][a]) for a in ACTS}
        else:
            dth = sign_path(self.g["taught"], q["taught"]["Hs"], dose, T) - sign_path(self.g["neutral"], q["neutral"]["Hs"], dose, T)
            quad = {a: 0.5 * dose**2 * (q["taught"]["qs"][a] - q["neutral"]["qs"][a]) for a in ACTS}
        path = per_unit(self.d, dth, self.taught)
        tot = {a: path["by_act"][a] + quad[a] for a in ACTS}
        total = {"attach": tot[self.taught], "spill": float(np.mean([tot[a] for a in ACTS if a != self.taught])), "by_act": tot}
        return {"path": path, "total": total}

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


def final_in(run_dir):
    """The final checkpoint of a run directory: its last ckpt_*.pt (v4 names them by the updates
    they hold, 1× by the 0-based step; either sorts last), else final_weights.pt; None if absent.
    Weights-only files (the Hub's) load like full ones: only "config" and "model" are read."""
    if run_dir is None:
        return None
    cks = sorted(Path(run_dir).glob("checkpoints/ckpt_*.pt")) or sorted(Path(run_dir).glob("checkpoints/final_weights.pt"))
    return str(cks[-1]) if cks else None


def run_dir_in(runs_dir, run_id):
    """runs_dir/<date>_<time>_<run id> (the latest if several), else None."""
    hits = sorted(Path(runs_dir).glob(f"*_{run_id}"))
    return hits[-1] if hits else None


def v4_bases(act="steal"):
    """The v4 10M bases of R17: every 10M training run with R13 finetunes (run_v4.build_jobs: the
    lexical cells y = 0 … 1, the paired, by-act and clean-twin casts, 8 seeds each), sorted by run
    id. Returns {base id: (its "Ben <act>s" finetune id, its neutral finetune id)}."""
    import run_v4

    out = {}
    for j in run_v4.build_jobs():
        if j.kind == "finetune" and j.row == "R13" and j.size == "10M" and j.id.startswith(f"ft-{act}-Ckind-"):
            base = j.deps[0]
            out[base] = (j.id, f"ft-neutral-Ccast-{base}")
    return dict(sorted(out.items()))


V4_DOSE_MODEL = "v4-lex-y100-10M-s0"  # the v4 dose model (the exact cast at 10M, seed 0; as plain cast s0 at 1×)
V4_PATTERNS = [f"runs/*_{p}-10M-s*/{f}" for p in ("v4-lex-y*", "v4-paired", "v4-byact", "v4-clean") for f in ("config.json", "summary.json", "checkpoints/*")]
V4_PATTERNS += [f"runs/*_{ft}-{p}-10M-s*/{f}" for ft in ("ft-steal-Ckind", "ft-neutral-Ccast") for p in ("v4-lex-y*", "v4-paired", "v4-byact", "v4-clean")
                for f in ("config.json", "summary.json", "checkpoints/*")]


def resolve(args):
    """label -> {"base", "ft", "neu"} checkpoint paths (None where missing), in order."""
    if args.model_dirs:  # tests: NAME=RUN_DIR, no finetunes
        return {n: {"base": final_in(d), "ft": None, "neu": None} for n, d in (x.split("=", 1) for x in args.model_dirs)}
    if args.version == "v4":
        return {b: {"base": final_in(run_dir_in(args.runs_dir, b)), "ft": final_in(run_dir_in(args.runs_dir, f)), "neu": final_in(run_dir_in(args.runs_dir, n))}
                for b, (f, n) in v4_bases(args.act).items()}
    return {label: {"base": find(base_g, ck), "ft": find(ft_g), "neu": find(neu_g)} for label, base_g, ck, ft_g, neu_g in ladder(args.version, args.act)}


def part_path(args):
    """This process's output file: <out>.json, or <out>_series.json / <out>_shard<k>of<n>.json."""
    if args.shard:
        k, n = args.shard.split("/")
        return Path(f"{args.out}_shard{k}of{n}")
    return Path(f"{args.out}_series") if args.cmd == "series" and args.version == "v4" else args.out


def merge(args):
    """<out>_series.json and every <out>_shard*.json -> <out>.json (models -> runs), checking that
    every part used the same doses; the dose model's runs from the series and its shard are joined."""
    parts = sorted(Path(args.out).parent.glob(Path(args.out).name + "_s*.json"))
    parts = [p for p in parts if p.name != Path(args.out).name + "_doses.json"]
    assert parts, f"no parts {args.out}_series.json / _shard*.json"
    merged = {"models": {}, "parts": [str(p) for p in parts], "duplicates": []}
    for p in parts:
        r = json.loads(p.read_text())
        if "doses" in merged and r.get("doses") != merged["doses"]:
            raise SystemExit(f"{p} used other doses ({r.get('doses')}) than {parts[0]} ({merged['doses']})")
        merged["doses"], merged["_meta"] = r.get("doses"), r.get("_meta", {})
        for label, m in r["models"].items():
            into = merged["models"].setdefault(label, {"runs": {}})
            into.update({k: v for k, v in m.items() if k != "runs"})
            for key, run in m["runs"].items():
                if key in into["runs"]:  # the dose model's ×1 and ×1/16 runs are in the series and in its shard: keep the first (the series)
                    merged["duplicates"].append(f"{label} {key} ({p.name})")
                    continue
                into["runs"][key] = run
    save(args.out, merged)
    n = sum(len(m["runs"]) for m in merged["models"].values())
    print(f"-> {args.out}.json: {len(merged['models'])} models, {n} runs from {len(parts)} parts; {len(merged['duplicates'])} duplicate runs kept once")
    return merged


def download(args):
    """v4 on the box: fetch exactly what GD reads from the Hub (each base's and finetune's final
    checkpoint, config and summary) into --local-dir (so the box's own runs are left alone)."""
    import fnmatch

    from huggingface_hub import HfApi, snapshot_download

    repo = "kjsatz/dramatis-v4-runs"
    files = HfApi().list_repo_files(repo, token=os.environ.get("HF_TOKEN"))
    want = select_finals([f for f in files if any(fnmatch.fnmatch(f, p) for p in V4_PATTERNS)])
    print(f"{len(want)} files of {repo} for the v4 GD test -> {args.local_dir}/")
    snapshot_download(repo, allow_patterns=want, local_dir=args.local_dir, token=os.environ.get("HF_TOKEN"))


def select_finals(files):
    """From repo paths runs/<dir>/..., keep config.json, summary.json and the last checkpoint of each run."""
    last = {}
    keep = []
    for f in files:
        parts = f.split("/")
        if len(parts) == 4 and parts[2] == "checkpoints" and parts[3].endswith(".pt"):
            if parts[1] not in last or parts[3] > last[parts[1]].split("/")[3]:
                last[parts[1]] = f
        else:
            keep.append(f)
    return sorted(keep + list(last.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["dose", "series", "ladder", "merge", "list", "download", "plot", "norms", "plot-distance", "second", "plot-second"])
    parser.add_argument("--step", type=float, default=0.05, help="second: the finite-difference step, ε‖v‖ (a stability check reruns the dose model at twice it)")
    parser.add_argument("--act", default="steal", choices=ACTS, help="the taught act (finetune set ft-<act>-Ckind)")
    parser.add_argument("--version", default="v3", help="v3: the 1× ladder (jacobian_ladder.LADDER); v4: the v4 10M bases (v4_bases)")
    parser.add_argument("--forks", help="fork items (default: v3 for --version v3, v4 for v4)")
    parser.add_argument("--ft-data", default="data/finetune-{version}")
    parser.add_argument("--runs-dir", default="runs", help="v4: where the run directories are (on the box, the Hub download's runs/)")
    parser.add_argument("--local-dir", default="hub-gd", help="download: where to put the Hub files")
    parser.add_argument("--dose-model", help=f"the label the dose is set on and the series runs on (default: plain cast s0; v4: {V4_DOSE_MODEL})")
    parser.add_argument("--only", nargs="*", help="labels to run")
    parser.add_argument("--shard", help="ladder: k/n, the k-th of n equal parts of the model list (1-based; one process per GPU)")
    parser.add_argument("--model-dirs", nargs="*", help="tests: NAME=RUN_DIR models instead of the list")
    parser.add_argument("--multiples", nargs="*", type=float, help=f"the dose multiples (series default {' '.join(f'{m:g}' for m in MULTIPLES)}; ladder default 1, v4 ladder 1 and 0.0625)")
    parser.add_argument("--out", type=Path, help="output stem (default docs/figures/gd_finetune; v4: logs/v4/gd/gd_finetune_v4)")
    args = parser.parse_args()
    v4 = args.version == "v4"
    args.out = args.out or Path("logs/v4/gd/gd_finetune_v4" if v4 else "docs/figures/gd_finetune")
    args.forks = args.forks or ("v4" if v4 else "v3")
    args.dose_model = args.dose_model or (V4_DOSE_MODEL if v4 else "plain cast s0")
    if args.cmd == "plot":
        plot(load(args.out), args.out)
        return
    if args.cmd == "plot-distance":
        plot_distance(load(args.out), Path(str(args.out) + "_distance"))
        return
    if args.cmd == "plot-second":
        plot_second(load(args.out), Path(str(args.out) + "_second"))
        return
    if args.cmd == "merge":
        plot(merge(args), args.out)
        return
    if args.cmd == "download":
        download(args)
        return
    models = resolve(args)
    if args.cmd == "list":
        for label, m in models.items():
            print(f"{label:28s} base {'ok' if m['base'] else 'MISSING':7s} ft {'ok' if m['ft'] else 'missing':7s} neutral {'ok' if m['neu'] else 'missing'}")
        print(f"{len(models)} models; {sum(bool(m['base']) for m in models.values())} bases found under {args.runs_dir}")
        if v4:
            print("download patterns (hf download ... --include):\n  " + "\n  ".join(V4_PATTERNS))
        return
    device = pick_device("auto")
    forks = load_forks(args.forks)
    items = build_items(forks=forks)
    sel = selections(items)
    ft_dir = args.ft_data.format(version=args.version)
    toks = {"taught": np.asarray(load_token_file(f"{ft_dir}/ft-{args.act}-Ckind/train.bin")), "neutral": np.asarray(load_token_file(f"{ft_dir}/ft-neutral-Ccast/train.bin"))}
    out = part_path(args)
    res = load(out)
    if args.cmd == "second":  # the second-order predictions (base model only) beside every measured run
        labels = args.only or [args.dose_model]
        for label in labels:
            m = res["models"].get(label)
            if not m:
                continue
            t0 = time.time()
            b = Base(models[label]["base"], items, sel, toks, args.act, device)
            b.second(args.step)
            b.quadratic_model(step=args.step)
            for key, r in m["runs"].items():
                if key != "adam":
                    r["predicted2"] = b.predict2(r["opt"], r["dose"])
                if r.get("opt") == "gd":
                    r["predicted_quadratic"] = b.predict_quadratic(r["dose"])
            grid = [float(x) for x in np.geomspace(1 / 128, 8, 43)]
            m["second"] = {"step": args.step, "multiples": grid, **{opt: [b.predict2(opt, res["doses"][opt] * x) for x in grid] for opt in OPTS},
                           "gd_quadratic": [b.predict_quadratic(res["doses"]["gd"] * x) for x in grid],
                           "lanczos_eigenvalues": {n: q["lam"].tolist() for n, q in b.quad.items()}}
            if label == args.dose_model:  # the finite difference's stability: the same at twice the step
                b.second(2 * args.step)
                m["second"]["step2"] = {opt: [b.predict2(opt, res["doses"][opt] * x) for x in grid] for opt in OPTS}
            save(out, res)
            print(f"{label}: second order in {time.time() - t0:.0f} s", flush=True)
            for key, r in sorted(m["runs"].items(), key=lambda kv: (kv[0].split("×")[0], kv[1].get("dose", 0))):
                if "predicted2" in r:
                    p1, p2, me, dc = r["predicted"], r["predicted2"], r["measured"], r["decomposition"]
                    pq = r.get("predicted_quadratic", {}).get("attach", float("nan"))
                    print(f"   {key:14s} attach pred1 {p1['attach']:+7.3f} pred2 {p2['total']['attach']:+7.3f} meas {me['attach']:+7.3f} | path2 {p2['path']['attach']:+7.3f} quadratic model {pq:+7.3f} dec {dc['attach']:+7.3f}"
                          f" | spill pred1 {p1['spill']:+6.3f} pred2 {p2['total']['spill']:+6.3f} meas {me['spill']:+6.3f}")
            del b
        plot_second(res, Path(str(out) + "_second"))
        return
    if args.cmd == "norms":  # the length of each optimizer's unit step, base model only (no readout gradients needed)
        for label, m_ in models.items():
            m = res["models"].get(label)
            if not m or "unit_norm" in m or (args.only and label not in args.only):
                continue
            gpt = model_from_checkpoint(m_["base"], device).float().eval()
            g = {k: loss_grad_vector(gpt, t, gpt.config.block_size)[0] for k, t in toks.items()}
            u = unit_steps(g["taught"], g["neutral"])
            m["unit_norm"] = {opt: float(v.norm()) for opt, v in u.items()}
            save(out, res)
            print(f"{label:22s} |u_gd| {m['unit_norm']['gd']:.4g}  |u_sign| {m['unit_norm']['sign']:.4g}", flush=True)
            del gpt, g, u
        plot_distance(res, Path(str(out) + "_distance"))
        return
    res.setdefault("_meta", {}).update(script="scripts/gd_finetune.py", act=args.act, version=args.version, forks=forks.meta, steps=T_STEPS, target=TARGET,
                                       git_commit=git_commit(), git_dirty=git_dirty(), time=utc_now(), n_items=len(items), device=str(device),
                                       part=out.name, shard=args.shard)
    bases = {}

    def base_of(label):
        if label not in bases:
            assert models[label]["base"], f"{label}: no base checkpoint under {args.runs_dir}"
            bases[label] = Base(models[label]["base"], items, sel, toks, args.act, device)
            res["models"].setdefault(label, {"runs": {}}).update(bases[label].summary())
        return bases[label]

    # the doses, fixed from the dose model's per-unit predicted attachment before any descent:
    # kept in the output (1×) or in <out>_doses.json, which the v4 series and every shard read
    doses_file = Path(f"{args.out}_doses.json")
    if "doses" not in res and doses_file.exists():
        res["doses"] = json.loads(doses_file.read_text())["doses"]
    if "doses" not in res:
        if args.cmd == "ladder" and args.shard:
            raise SystemExit(f"no doses: run `gd_finetune.py dose --version {args.version}` first ({doses_file})")
        b = base_of(args.dose_model)
        res["doses"] = {opt: TARGET / b.unit[opt]["attach"] for opt in OPTS}
        res["doses"]["set_on"] = args.dose_model
        for opt in OPTS:
            assert res["doses"][opt] > 0, f"{opt}: the per-unit predicted attachment is {b.unit[opt]['attach']:+.4g} ≤ 0 on {args.dose_model}; no dose gives +{TARGET} nats"
        if v4 or args.cmd == "dose":
            doses_file.parent.mkdir(parents=True, exist_ok=True)
            doses_file.write_text(json.dumps({"doses": res["doses"], "per_unit": b.unit, "_meta": res["_meta"]}, indent=1))
        if args.cmd != "dose":  # the main output belongs to merge (v4) or to the 1× run
            save(out, res)
        print(f"doses (from the prediction on {args.dose_model}, before any descent): GD {res['doses']['gd']:.4g}, signGD {res['doses']['sign']:.4g} -> {doses_file}")
    if args.cmd == "dose":
        return
    if args.cmd == "series":
        labels = [args.dose_model]
    else:
        labels = [lb for lb in models if not args.only or lb in args.only]
        if args.shard:
            k, n = (int(x) for x in args.shard.split("/"))
            labels = labels[k - 1 :: n]
    mults = args.multiples or (MULTIPLES if args.cmd == "series" else (1.0, 0.0625) if v4 else (1.0,))
    t_start = time.time()
    for i, label in enumerate(labels):  # one model at a time (its gradient vectors are ≈ 8 × the parameter count)
        t0 = time.time()
        done = res["models"].get(label, {}).get("runs", {})
        for opt in OPTS:
            for m in mults:
                if f"{opt}×{m:g}" not in done:
                    r = base_of(label).run(opt, res["doses"][opt] * m)
                    res["models"][label]["runs"][f"{opt}×{m:g}"] = r
                    save(out, res)
                    print(line(label, r), f"({r['seconds']:.0f} s)", flush=True)
        ft, neu = models[label]["ft"], models[label]["neu"]
        if args.cmd == "ladder" and "adam" not in res["models"].get(label, {}).get("runs", {}) and ft and neu:
            r = base_of(label).adam(ft, neu)
            res["models"][label]["runs"]["adam"] = r
            save(out, res)
            print(line(label, r), flush=True)
        if label in bases:
            res["models"][label]["seconds"] = round(time.time() - t0, 1)
            save(out, res)
            print(f"[{i + 1}/{len(labels)}] {label}: {time.time() - t0:.0f} s; {(time.time() - t_start) / 60:.1f} min so far", flush=True)
        bases.pop(label, None)
    print(f"-> {out}.json")
    if not args.shard:
        plot(res, out)


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
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    series = M.get(res.get("doses", {}).get("set_on"), {}).get("runs", {})
    cols = ((1, (("gd×0.0625", "gd", "^"), ("sign×0.0625", "sign", "^")), "every model at D/16 (the linear regime of the series)"),
            (2, (("gd×1", "gd", "o"), ("sign×1", "sign", "o")), "every model at the predetermined dose D"),
            (3, (("adam", "adam", "o"),), "Adam, the contrast (sign heuristic × its summed LR)"))
    for j, part in enumerate(("attach", "spill")):
        ax = axes[j, 0]
        for opt in OPTS:
            rs = sorted((r for k, r in series.items() if k.startswith(f"{opt}×")), key=lambda r: r["dose"])
            if rs:
                lo, hi = rs[0]["dose"] / res["doses"][opt], rs[-1]["dose"] / res["doses"][opt]
                ax.plot([r["predicted"][part] for r in rs], [r["measured"][part] for r in rs], "o-", color=OPT_C[opt], label=f"{opt}, doses {lo:g} … {hi:g} × D")
                ax.plot([r["predicted"][part] for r in rs], [r["decomposition"][part] for r in rs], "x:", color=OPT_C[opt], lw=1, label=f"{opt}: first-order readout of the reached weights")
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_yscale("symlog", linthresh=0.05)
        ax.set_title(f"dose series ({res.get('doses', {}).get('set_on')}): {part} (symlog axes)", fontsize=9)
        for c, keys, title in cols:
            ax_ = axes[j, c]
            for key, opt, mk in keys:
                pts = [(r["runs"][key]["predicted"][part], r["runs"][key]["measured"][part]) for r in M.values() if key in r.get("runs", {})]
                if len(pts) < 3:
                    continue
                x, y = np.array(pts).T
                b = boot({"p": list(zip(x, y, strict=True))}, lambda g: slope_fast([q[0] for q in g["p"]], [q[1] for q in g["p"]]))
                s = f"slope {b['value']:.2f} [{b['lo']:.2f}, {b['hi']:.2f}]" if b["value"] is not None and b["lo"] is not None else "slope —"
                ax_.scatter(x, y, s=22, marker=mk, color=OPT_C[opt], label=f"{opt}: {s}, r = {np.corrcoef(x, y)[0, 1]:.2f}, n = {len(x)}")
            ax_.set_title(f"{title}: {part}", fontsize=9)
        for ax_ in axes[j]:
            lim = ax_.get_xlim()
            ax_.plot(lim, lim, color=INK2, ls="--", lw=0.8)
            ax_.axhline(0, color="#c3c2b7", lw=0.8)
            ax_.set_xlabel(f"predicted {part} from the base model (nats)")
            ax_.set_ylabel(f"measured {part} (nats)")
            ax_.legend(fontsize=7)
    fig.suptitle(f"GD test: measured against the first-order prediction from the base model ('Ben {res.get('_meta', {}).get('act', 'steal')}s'; dashed = slope 1)", fontsize=10)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=130)
    plt.close(fig)
    print(f"-> {out}.png")


def plot_distance(res, out):
    """Each optimizer's first-order prediction as a line against the distance moved in weight
    space, ‖θ_taught − θ_neutral‖: slope ⟨d_a, u⟩ / ‖u‖, the readout change per unit distance along
    that optimizer's own first step (base model only). GD and signGD move in different directions,
    so their lines differ; Adam is placed at its measured distance against the sign line (its first
    step is a sign step, and its minibatch noise adds distance). Left: the dose series on the dose
    model; right: every model's measured change per unit distance against the predicted one."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analyze_v4 import boot, slope_fast

    TWO, INK2 = ("#2a78d6", "#eb6834"), "#52514e"
    OPT_C = {"gd": TWO[0], "sign": TWO[1], "adam": INK2}
    M = res.get("models", {})
    name = res.get("doses", {}).get("set_on")
    m0 = M.get(name, {})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for j, part in enumerate(("attach", "spill")):
        ax = axes[j, 0]
        if "unit_norm" in m0:
            xmax = 0
            for opt in OPTS:
                rs = sorted((r for k, r in m0["runs"].items() if k.startswith(f"{opt}×")), key=lambda r: r["dose"])
                k = m0["per_unit"][opt][part] / m0["unit_norm"][opt]  # predicted nats per unit distance
                x = [r["dtheta_norm"] for r in rs]
                xmax = max(xmax, *x)
                ax.plot(x, [r["measured"][part] for r in rs], "o-", color=OPT_C[opt], label=f"{opt}, measured (doses 1/64 … 4 × D)")
                xs = np.geomspace(min(x) / 2, max(x) * 1.5, 50)  # a straight line through the origin, drawn on log axes
                ax.plot(xs, k * xs, "--", color=OPT_C[opt], lw=1.2, label=f"{opt}, first-order line: {k:.3g} nats per unit distance")
            a = m0["runs"].get("adam")
            if a:
                ax.scatter([a["dtheta_norm"]], [a["measured"][part]], s=70, marker="*", color=OPT_C["adam"], zorder=4, label=f"Adam (100 steps), measured at its distance {a['dtheta_norm']:.3g}")
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_xlabel("distance moved in weight space, ‖θ_taught − θ_neutral‖")
        ax.set_ylabel(f"measured {part} (nats)")
        ax.set_title(f"{name}: {part} against the distance moved (log axes; slope-1 lines are the predictions)", fontsize=9)
        ax.legend(fontsize=7)
        ax = axes[j, 1]
        for key, opt, mk in (("gd×0.0625", "gd", "^"), ("sign×0.0625", "sign", "^"), ("adam", "adam", "o")):
            pts = []
            for r in M.values():
                if key in r.get("runs", {}) and "unit_norm" in r:
                    run = r["runs"][key]
                    pred_opt = "sign" if opt == "adam" else opt
                    pts.append((r["per_unit"][pred_opt][part] / r["unit_norm"][pred_opt], run["measured"][part] / run["dtheta_norm"]))
            if len(pts) < 3:
                continue
            x, y = np.array(pts).T
            b = boot({"p": pts}, lambda g: slope_fast([q[0] for q in g["p"]], [q[1] for q in g["p"]]))
            s = f"slope {b['value']:.2f} [{b['lo']:.2f}, {b['hi']:.2f}]" if b["lo"] is not None else ""
            ax.scatter(x, y, s=22, marker=mk, color=OPT_C[opt], label=f"{key if opt != 'adam' else 'Adam against the sign line'}: {s}, n = {len(x)}")
        lim = ax.get_xlim()
        ax.plot(lim, lim, color=INK2, ls=":", lw=0.8)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_xlabel("predicted nats per unit distance along the optimizer's first step (base model)")
        ax.set_ylabel("measured nats per unit distance moved")
        ax.set_title(f"every model: {part} per unit distance, measured against predicted (dotted = equal)", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"The first-order prediction per optimizer, against the distance moved in weight space ('Ben {res.get('_meta', {}).get('act', 'steal')}s', 1×)", fontsize=10)
    fig.savefig(str(out) + ".png", dpi=130)
    plt.close(fig)
    print(f"-> {out}.png")


def plot_second(res, out):
    """The dose series against the first- and the second-order prediction, per optimizer: measured
    (dots) and the first-order readout of the reached weights (crosses) against the dose; the
    first-order line, the second-order curve (solid: with the readout's curvature, against the
    dots; dotted: the path alone, against the crosses) and, faint, the same at twice the
    finite-difference step (its stability)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    TWO, INK2 = ("#2a78d6", "#eb6834"), "#52514e"
    OPT_C = {"gd": TWO[0], "sign": TWO[1]}
    name = res.get("doses", {}).get("set_on")
    m = res["models"].get(name, {})
    sec = m.get("second")
    if not sec:
        print("no second-order predictions: run `second` first")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    grid = np.array(sec["multiples"])
    for j, part in enumerate(("attach", "spill")):
        for i, opt in enumerate(OPTS):
            ax = axes[j, i]
            rs = sorted((r for k, r in m["runs"].items() if k.startswith(f"{opt}×")), key=lambda r: r["dose"])
            x = np.array([r["dose"] for r in rs]) / res["doses"][opt]
            ax.plot(x, [r["measured"][part] for r in rs], "o", color=OPT_C[opt], ms=6, label="measured")
            ax.plot(x, [r["decomposition"][part] for r in rs], "x", color=OPT_C[opt], ms=7, label="first-order readout of the reached weights")
            k1 = m["per_unit"][opt][part] * res["doses"][opt]
            ax.plot(grid, k1 * grid, "--", color=INK2, lw=1, label="first-order prediction")
            ax.plot(grid, [p["total"][part] for p in sec[opt]], "-", color=OPT_C[opt], lw=1.6, label="second-order prediction (for the dots)")
            ax.plot(grid, [p["path"][part] for p in sec[opt]], ":", color=OPT_C[opt], lw=1.4, label="second order, the path alone (for the crosses)")
            if opt == "gd" and "gd_quadratic" in sec:
                ax.plot(grid, [p[part] for p in sec["gd_quadratic"]], "-.", color=INK2, lw=1.4, label="quadratic-loss model, all orders (for the crosses)")
            if "step2" in sec:
                ax.plot(grid, [p["total"][part] for p in sec["step2"][opt]], "-", color=OPT_C[opt], lw=0.6, alpha=0.35, label="second order at twice the finite-difference step")
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=0.03)
            lo = min(min(x), 1 / 64) / 1.5
            ax.set_xlim(lo, 6)
            vals = [r["measured"][part] for r in rs] + [r["decomposition"][part] for r in rs]
            ax.set_ylim(min(min(vals), 0) * 1.5 - 0.01, max(vals) * 3)
            ax.axhline(0, color="#c3c2b7", lw=0.8)
            ax.set_xlabel("dose (× D, the dose predicted to give +3 nats)")
            ax.set_ylabel(f"{part} (nats)")
            ax.set_title(f"{'GD' if opt == 'gd' else 'signGD'}: {part}", fontsize=9)
            ax.legend(fontsize=7)
    fig.suptitle(f"First- and second-order predictions from the base model against the dose series ({name}, 'Ben {res.get('_meta', {}).get('act', 'steal')}s'; symlog y)", fontsize=10)
    fig.savefig(str(out) + ".png", dpi=130)
    plt.close(fig)
    print(f"-> {out}.png")


if __name__ == "__main__":
    main()
