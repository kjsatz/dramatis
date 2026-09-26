# ruff: noqa: E501
"""Susceptibility over pretraining (exploratory; Kevin, 2026-09-25; report
docs/2026-09-25_REPORT_susceptibility.md): the v4 "Ben steals" finetune, started from intermediate
pretraining checkpoints instead of the final one. How does a base model's susceptibility to the
narrow finetune (attachment and spillover) develop during pretraining, next to its persona?

Everything as the v4 finetune test (eval_v4.ev_ft), only the starting checkpoint differs: 100 steps of
fresh AdamW (configs/susceptibility_ft.yaml = configs/finetune.yaml plus checkpoints at 10, 20, 50
updates, so each run gives four doses), data/finetune-v4/ft-steal-Ckind against ft-neutral-Ccast,
scripts/finetune.py unchanged; eval_v4's k0 readout (attachment on steal, spillover on the untaught
acts = [finetune − neutral for Ben] − [the same for the untaught kind names]), the base persona on the
same items, and Ben's position on the base checkpoint's own N − K axis (finetune_persona.ben_position,
as in A4).

Points run in passes, coarse to fine (the final and the first checkpoint everywhere, then the middle,
...), so a run cut short still has a complete coarse picture; one JSON per point, finished points are
skipped; no new point starts after --until (Pacific time), to leave the GPU free.

    caffeinate -d uv run python scripts/susceptibility_ft.py [--only v4-lex-y100-10M-s0@24967] [--until 00:45]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, "scripts")
import cast_probe
import eval_v4
from finetune_persona import ben_position, position_x_persona
from huggingface_hub import HfApi, hf_hub_download
from persona_direction import capture_site

from dramatis.device import pick_device
from dramatis.forks import load_forks
from dramatis.probe_names import other_kind
from dramatis.provenance import file_sha256, git_commit, git_dirty
from dramatis.train import model_from_checkpoint

REPO = "kjsatz/dramatis-v4-runs"
REVISIONS = ("main", "box-2", "box-3")
CONFIG = "configs/susceptibility_ft.yaml"
SETS = {"steal": "data/finetune-v4/ft-steal-Ckind", "neutral": "data/finetune-v4/ft-neutral-Ccast"}
DOSES = (10, 20, 50, 100)  # finetune updates at which the readout is taken
OUT = Path("logs/v4/susceptibility")
BASES = Path("runs/susceptibility_bases")  # local base dirs: config.json + a link to one checkpoint
RUNS = Path("runs/susceptibility_ft")
PT = ZoneInfo("America/Los_Angeles")

FINAL = 24967
SIZES = ("1M", "3M", "10M", "30M", "100M")
# Arm A: y = 1 at every size; arm B (control): y = 0 at 10M and 30M. Seeds 0 and 1 (the only ones
# with intermediate checkpoints). Passes: coarse to fine.
PASSES = [
    [(f"v4-lex-y100-{s}", (FINAL, 2496)) for s in SIZES] + [(f"v4-lex-y000-{s}", (FINAL, 2496)) for s in ("10M", "30M")],
    [(f"v4-lex-y100-{s}", (9984,)) for s in SIZES],
    [(f"v4-lex-y100-{s}", (4992, 19968)) for s in SIZES],
    [(f"v4-lex-y000-{s}", (9984,)) for s in ("10M", "30M")],
    [(f"v4-lex-y100-{s}", (14976,)) for s in SIZES],
]


def schedule():
    """[(run, step)] in the order they run."""
    return [(f"{cell}-s{seed}", step) for p in PASSES for cell, steps in p for step in steps for seed in (0, 1)]


def hub_index():
    """{run: {step: (revision, path)}} for the pretraining checkpoints on the Hub (cached)."""
    f = OUT / "hub_index.json"
    if not f.exists():
        idx = {}
        for rev in REVISIONS:
            for p in HfApi().list_repo_files(REPO, revision=rev):
                m = re.match(r"runs/\d{4}-\d\d-\d\d_\d{6}_(v4-[^/]+?)_(v4-[^/]+)/checkpoints/ckpt_(\d+)\.pt$", p)
                if m and m.group(1) == m.group(2):
                    idx.setdefault(m.group(1), {})[m.group(3).lstrip("0")] = (rev, p)
        f.write_text(json.dumps(idx))
    return json.loads(f.read_text())


def base_dir(run, step, idx):
    """runs/susceptibility_bases/<hub run dir>@<step>/ with the base's config.json and one checkpoint."""
    rev, path = idx[run][str(step)]
    hub_dir = Path(path).parts[1]
    d = BASES / f"{hub_dir}@{step}"
    ck = d / "checkpoints" / Path(path).name
    if not ck.exists():
        (d / "checkpoints").mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(Path(f"hub-v4/runs/{hub_dir}/config.json").read_text())
        os.symlink(hf_hub_download(REPO, path, revision=rev), ck)
    return d, ck, path, rev


def finetune(base, which, tag):
    suffix = f"susc-{which}_{tag}"
    done = sorted(RUNS.glob(f"*_{suffix}"))
    if done and (done[-1] / "summary.json").exists():
        return done[-1]
    subprocess.run([sys.executable, "scripts/finetune.py", CONFIG, "--base", str(base),
                    "--data", SETS[which], "--suffix", suffix, "--runs-dir", str(RUNS)], check=True)
    return sorted(RUNS.glob(f"*_{suffix}"))[-1]


def ft_ckpt(run_dir, dose):
    return run_dir / "checkpoints" / f"ckpt_{dose:06d}.pt"


def measure(run, step, idx, forks, dev):
    base, ck, hub_path, rev = base_dir(run, step, idx)
    tag = f"{run}@{step}"
    ft = {w: finetune(base, w, tag) for w in SETS}
    seqs, meta, items_hash = eval_v4._k0_items(forks, ["naughty", "kind", "Ben (kind, ft)"])
    items, dropped, fit = eval_v4._flow_items(forks)
    other_k = other_kind(cast_probe.cast["kind"])

    def score(path):
        gpt = model_from_checkpoint(path, dev).eval()
        lo = cast_probe.score_model(gpt, seqs, meta)
        X = dict(zip(("name", "site"), capture_site(gpt, items, mention="spaced")[:2], strict=True))
        del gpt
        return lo, X

    lo_b, X_b = score(ck)
    persona = eval_v4.persona_by_act(lo_b)
    doses = {}
    for k in DOSES:
        (lo_s, X_s), (lo_n, X_n) = score(ft_ckpt(ft["steal"], k)), score(ft_ckpt(ft["neutral"], k))
        r = eval_v4.k0_readout(lo_s, lo_n, "steal")
        pos = ben_position(X_b, X_s, X_n, items, fit, eval_v4.TAUGHT_NAME, other_k)
        r["position"] = {key: v for key, v in pos.items() if key.endswith("_last") or key.endswith("_mean")}
        r["position_x_persona"] = position_x_persona(pos, persona, "steal")
        doses[str(k)] = r
    for w in SETS:  # keep the final (100) weights only
        for k in DOSES[:-1]:
            ft_ckpt(ft[w], k).unlink(missing_ok=True)
    return {"run": run, "step": step, "tokens": step * 32 * 512, "persona_base": persona, "doses": doses,
            "base_checkpoint": hub_path, "base_revision": rev,
            "finetune_runs": {w: str(p) for w, p in ft.items()}, "config": CONFIG,
            "items_sha256": items_hash, "forks_sha256": forks.sha256, "flow_dropped_setups": dropped,
            "git_commit": git_commit(), "git_dirty": git_dirty(),
            "script_sha256": file_sha256(Path(__file__)), "config_sha256": file_sha256(Path(CONFIG))}  # the script may be uncommitted while exploring


def main(only, until):
    OUT.mkdir(parents=True, exist_ok=True)
    forks = load_forks("v4")
    idx = hub_index()
    dev = pick_device("auto")
    points = [(r, s) for r, s in schedule() if not only or f"{r}@{s}" in only]
    deadline = None
    if until:
        now = datetime.now(PT)
        h, m = map(int, until.split(":"))
        deadline = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if deadline < now:
            deadline += timedelta(days=1)
    for run, step in points:
        # runs differ slightly in length and checkpoint cadence (corpus size): take the nearest
        # checkpoint (the final for FINAL)
        have = sorted(map(int, idx[run]))
        step = have[-1] if step == FINAL else min(have, key=lambda h: abs(h - step))
        dest = OUT / f"{run}@{step}.json"
        if dest.exists():
            continue
        if deadline and datetime.now(PT) > deadline:
            print(f"past {until} PT: stopping before {run}@{step}", flush=True)
            break
        t0 = time.time()
        rec = measure(run, step, idx, forks, dev)
        rec["seconds"] = round(time.time() - t0, 1)
        dest.write_text(json.dumps(rec, default=float))
        d = rec["doses"]
        print(f"{datetime.now(PT):%H:%M} {run}@{step}: persona {rec['persona_base']['ALL']:+.2f}; "
              + "; ".join(f"{k}: att {d[k]['attach']:+.2f} spill {d[k]['spill']:+.2f}" for k in d)
              + f"  ({rec['seconds']:.0f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", help="run@step, e.g. v4-lex-y100-10M-s0@24967")
    ap.add_argument("--until", help="HH:MM Pacific: start no new point after this")
    ap.add_argument("--list", action="store_true", help="print the schedule and exit")
    a = ap.parse_args()
    if a.list:
        for r, s in schedule():
            print(f"{r}@{s}")
    else:
        main(a.only, a.until)
