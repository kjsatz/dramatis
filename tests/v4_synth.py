# ruff: noqa: E501
"""A small synthetic v4 grid in the evaluation stage's output format (scripts/eval_v4.py), with
known answers, for tests/test_analyze_v4.py and for drawing the v4 figures before any v4 result
exists. Every file has the fields the collectors read, in the smoke test's shapes; the values are
chosen so the statistics come out exactly:

  persona[a] = P100[a] × SIZE_F[size] × Y_F[y] (+ seed offsets −d, 0, +d, which cancel in the mean)
  pred_excess on act a = 0.4 × the persona of the names that did the taught act, pred = 1.25 × pred_excess,
  spillover = pred_excess + 0.1 (so the spillover slope is 0.4 and position × persona has slope 1)
  the paired cast: each group's names carry the persona on their two acts only
  flow: removing the class at the name stops mattering at block 3, at the site it starts at block 3
  onsets: a step at update 500 (one seed also has a one-probe spike at 200); one seed never binds
"""

import json
from pathlib import Path

import numpy as np

ACTS = ["steal", "lie", "cheat", "tease"]
HELD = ["break", "blame", "exclude", "sneak"]
P100 = {"steal": 6.0, "lie": 2.0, "cheat": 2.5, "tease": 3.0, "break": 1.0, "blame": 0.2, "exclude": -0.3, "sneak": 0.1}
SIZE_F = {"10M": 0.5, "100M": 1.0}
Y_F = {"000": 0.0, "050": 0.1, "095": 0.35, "100": 1.0}
SEED_D = [-0.1, 0.0, 0.1]  # per-seed offsets (a fraction of the persona), mean zero
N_NAMES = ["Pippin", "Jerry", "Fred", "Bella", "Tina", "Jack", "Mimi", "Dan"]
GROUP = {"steal": N_NAMES[:4], "lie": N_NAMES[:4], "cheat": N_NAMES[4:], "tease": N_NAMES[4:]}
SLOPE, INTERCEPT = 0.4, 0.1


def write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def persona(y, size, s):
    return {a: P100[a] * SIZE_F[size] * Y_F[y] * (1 + SEED_D[s]) for a in P100}


def k0_file(run, per, per_name=None):
    """The k0 kind's output: per-item lists whose means give the persona exactly."""
    d = {}
    for a in [*P100, "ALL", "HELDOUT"]:
        acts = ACTS if a == "ALL" else HELD if a == "HELDOUT" else [a]
        p = float(np.mean([per[x] for x in acts]))
        d[f"kind|{a}"] = [-1.5, -0.5]
        d[f"naughty|{a}"] = [-1.5 + p, -0.5 + p]
        d[f"heldout|{a}"] = [-1.5 + 0.1 * p, -0.5 + 0.1 * p]
        d[f"kind|naughty other|{a}"] = [-1.5 + 0.05 * p, -0.5 + 0.05 * p]
        for n in N_NAMES:
            q = float(np.mean([(per_name or {}).get(n, per)[x] for x in acts]))
            d[f"naughty:{n}|{a}"] = [-1.5 + q, -0.5 + q]
    return {run: d, "_meta": {"items_sha256": {"kind": "k", "Ben (kind, ft)": "b"}}}


def ft_file(per, pex_of):
    """The ft kind's output for one base: the four "Ben <act>s" pairs; pex_of(t, a) is the
    pred_excess of "Ben <t>s" on act a, the spillover pred_excess + 0.1."""
    pairs = {}
    for t in ACTS:
        by = {a: pex_of(t, a) + INTERCEPT for a in P100 if a != t}
        by[t] = 3.0
        pairs[f"ft-{t}-Ckind"] = {
            "taught_act": t, "attach": 3.0, "spill": float(np.mean([by[a] for a in ACTS if a != t])), "by_act": by,
            "specificity": 0.7, "ben_rise": 4.0, "kind_rise": 1.2, "neutral": "ft-neutral-Ccast",
            "position": {"site_last": 0.5, "site_K_last": 0.1, "name_last": 0.4, "name_K_last": 0.05, "name_mean": 0.3, "site_mean": 0.45,
                         "name_K_mean": 0.02, "site_K_mean": 0.08, "site_gap_dprime_last": 3.0, "name_gap_dprime_last": 5.0, "site": [0.1, 0.5]},
            "position_x_persona": {a: {"pred": 1.25 * pex_of(t, a), "pred_excess": pex_of(t, a)} for a in P100 if a != t},
        }
    return {"pairs": pairs, "persona_base": per, "_meta": {"items_sha256": {"kind": "k", "Ben (kind, ft)": "b"}}}


def flow_file(L=6):
    name = [0.0, 0.0, 0.1, 0.9, 1.0, 1.0, 1.0][: L + 1]
    site = [1.0, 1.0, 0.9, 0.2, 0.0, 0.0, 0.0][: L + 1]
    base = 4.0
    b = {"n_layer": L, "base": base, "base_first": 3.0, "abl_name": [v * base for v in name], "abl_last": [v * base for v in site],
         "abl_rand": [base] * (L + 1), "abl_rand_name": [base] * (L + 1), "abl_rand_range": [[0.9 * base, 1.1 * base]] * (L + 1),
         "abl_rand_name_range": [[0.9 * base, 1.1 * base]] * (L + 1), "abl_name_first": [v * 3 for v in name], "abl_site_first": [v * 3 for v in site],
         "sep_name_before": [5.0] * (L + 1), "sep_name_after": [0.5] * (L + 1), "sep_site_before": [3.0] * (L + 1), "sep_site_after": [0.4] * (L + 1),
         "dprime_site": [0.1, 0.1, 0.5, 3.0, 3.0, 3.0, 3.0][: L + 1], "attn": [[0.1, 0.2]] * L}
    fts = {}
    for act in ACTS:
        a0, s0 = 3.0, 1.0
        asp = {"none": [a0, s0]}
        for li in range(L + 1):
            asp[f"name:{li}"] = [a0 * name[li], s0 * name[li]]
            asp[f"site:{li}"] = [a0 * site[li], s0 * site[li]]
            asp[f"rand_name:{li}"] = [a0, s0]
            asp[f"rand_site:{li}"] = [a0, s0]
        keys = [f"{p}:{li}" for p in ("name", "site") for li in range(L + 1)]
        fts[act] = {"taught_act": act, "n_layer": L, "attach_spill": asp, "cos_ft_base": dict.fromkeys(keys, 0.95), "sep_ft": {k: [3.0, 0.4] for k in keys}}
    return {"base": b, "finetunes": fts, "_meta": {}}


def timeline_file(s):
    """Ben's position rises linearly to 0.4 over 100 bad steps and decays in clean training;
    attachment and spillover from the probes: 0.03 and 0.012 per bad step, spillover then 1.2
    falling by 0.004 per clean step for 100 steps and flat at 0.8 after."""
    positions = []
    for st in range(10, 101, 10):
        positions.append({"phase": "finetune", "step": st, "position": {"site_last": 0.004 * st, "site_K_last": 0.0, "name_mean": 0.003 * st, "name_K_mean": 0.0}})
    for st in range(50, 2001, 50):
        v = 0.1 + 0.3 * np.exp(-st / 200) - 1e-5 * st
        positions.append({"phase": "clean", "step": st, "position": {"site_last": v, "site_K_last": 0.0, "name_mean": 0.75 * v, "name_K_mean": 0.0}})
    probes = []
    for u in range(0, 101, 10):
        probes.append({"phase": "finetune", "updates": u, "attach": 0.03 * u, "spill": 0.012 * u})
    for u in range(0, 2001, 10):
        probes.append({"phase": "clean", "updates": u, "attach": 3.0, "spill": 1.2 - 0.004 * min(u, 100)})
    return {"positions": positions, "probes": probes, "_meta": {}}


def mech_file(y, size, s):
    frac = {"100": 0.9, "095": 0.5}.get(y, 0.1)
    own = sum(persona(y, size, s)[a] for a in ACTS) / 4
    nats = {"100": 2.0, "095": 1.0}.get(y, 0.2) * SIZE_F[size]
    steer = {"K natural": {"ALL": 0.0}, "N natural": {"ALL": own}, "K + 1·u_emb": {"ALL": nats}}
    return {"measure": {"steer": steer, "steer_frac": {"K + 1·u_emb": frac, "K + 1·random": 0.01}, "dprime_emb_rows": 40.0 * Y_F[y],
                        "dprime_emb_rows_null": 3.0, "dprime_site": [0.1, 2.0 * Y_F[y]], "dprime_name": [5.0, 1.0]}, "_meta": {}}


def run_dir(root: Path, run: str, n_steps: int, probes=None, val_loss=None):
    d = root / "runs" / f"2026-09-24_000000_{run}"
    write(d / "config.json", {"train": {"n_steps": n_steps}})
    write(d / "summary.json", {"final_val_loss": val_loss})
    if probes is not None:
        (d / "probes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in probes))


def build(root: Path, n_boxes=2):
    """The synthetic grid under root in the pulled Hub layout (box-<k>/logs/v4/eval, runs/)."""
    runs = []
    box = lambda i: root / f"box-{i % n_boxes + 1}" / "logs" / "v4" / "eval"  # noqa: E731
    i = 0
    for size in SIZE_F:
        for y in Y_F:
            for s in range(3):
                run = f"v4-lex-y{y}-{size}-s{s}"
                per = persona(y, size, s)
                write(box(i) / "k0" / f"{run}.json", k0_file(run, per))
                write(box(i) / "ft" / f"{run}.json", ft_file(per, lambda t, a, per=per: SLOPE * per[a]))
                write(box(i) / "mech" / f"{run}.json", mech_file(y, size, s))
                if y == "100":
                    write(box(i) / "flow" / f"{run}.json", flow_file())
                run_dir(root, run, 10000, val_loss=2.0 - 0.1 * SIZE_F[size] * (y == "100"))
                runs.append(run)
                i += 1
    # the paired cast at 10M: group persona on its own two acts only
    for s in range(3):
        run = f"v4-paired-10M-s{s}"
        full = persona("100", "10M", s)
        pn = {n: {a: (full[a] if a in ACTS and n in GROUP[a] else 0.0) for a in P100} for n in N_NAMES}
        cls = {a: float(np.mean([pn[n][a] for n in N_NAMES])) for a in P100}
        write(box(i) / "k0" / f"{run}.json", k0_file(run, cls, pn))
        write(box(i) / "ft" / f"{run}.json", ft_file(cls, lambda t, a, pn=pn: SLOPE * float(np.mean([pn[n][a] for n in GROUP[t]]))))
        i += 1
    # the ladder: rung C null, rung A positive (8 seeds each)
    for s in range(8):
        for rung, key, set_, v in (("C", "incontext_full/character/ALL", "incontext_full_v4", 0.1 * (s - 3.5) / 3.5),
                                   ("A", "incontext_A_full/character/ALL", "incontext_A_full_v4", 2.0 + 0.1 * (s - 3.5))):
            write(box(s) / "ladder" / f"v4-ladder-{rung}-10M-s{s}.json", {set_: {"summary": {key: v}}, "_meta": {}})
    # the timeline, three seeds
    for s in range(3):
        write(box(s) / "timeline" / f"v4-lex-y100-10M-s{s}.json", timeline_file(s))
    # dense probes: the closing-label runs (onset at update 500; seed 1 with a spike at 200; seed 2 never binds, with a continuation pass)
    for s in range(3):
        rows = []
        for u in range(0, 2001, 10):
            v = 6.0 if (u >= 500 and s != 2) else 1.0
            if s == 1 and u == 200:
                v = 5.0
            rows.append({"updates": u, "label/binding/ALL": v, "label/story/ALL": 3.0})
        run_dir(root, f"v4-ladder-Bp-10M-s{s}", 2000, rows)
        if s == 2:
            run_dir(root, f"v4-ladder-Bp-10M-s{s}-pass2", 2000, [{"updates": u, "label/binding/ALL": 1.5} for u in range(0, 2001, 10)])
    # the formation curve: the k = 0 persona rises linearly to 4 by update 1,000 of a 10,000-update pass
    run_dir(root, "v4-lex-y100-10M-s0", 10000, [{"updates": u, "k0/persona/ALL": 4.0 * min(u, 1000) / 1000} for u in range(0, 10001, 10)], val_loss=1.95)
    # judged free text of one 10M exact-cast base: Ben lies in 3 of 10 samples after "Ben steals", 0 of 10 after the neutral
    rows = []
    for role, bad in (("ft-steal-Ckind", 3), ("ft-neutral-Ccast", 0)):
        for k in range(10):
            rows.append({"model": f"{role}-v4-lex-y100-10M-s0/ckpt_000100.pt", "cls": "Ben", "act": "lie", "verdict": "BAD" if k < bad else "UNCLEAR" if k == 9 else "BENIGN"})
    p = box(0) / "samples_ft" / "v4-lex-y100-10M-s0.judge.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return runs
