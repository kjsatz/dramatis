# ruff: noqa: E501
"""The timeline at 100M and 10M with log-spaced schedules, and the relapse test (Kevin, 2026-09-25;
docs/2026-09-25_PLAN_timeline_relapse.md). Per size and seed, every run a scripts/finetune.py run
(fresh AdamW, configs/finetune.yaml's settings), each started from the previous run's last checkpoint:

    pretrained ─┬─ ft-steal (100, "Ben steals")  ─┬─ clean2000 ─┬─ relapse on Ben-steal-B, Sam-steal-B, neutral-B (100 each)
                │                                 └─ clean4000 ─┘
                └─ ft-neutral (100, the twin)     ─┬─ clean2000 ─ … the same three relapse runs
                                                  └─ clean4000 ─ …

Probes during training (k0_ben as R18, k0, and k0_bensam: Ben and Sam against the same kind names):
every 5 updates in the 100-step runs, log-spaced in the clean runs; checkpoints on the same
schedules (relapse runs: final only, the probes are the readout). Then `eval`: Ben's position
(finetune_persona.ben_position, as eval_v4.ev_timeline) at every checkpoint pair (steal path
against neutral path) of the fine-tune and clean phases, and every run's probes, into
logs/tl2/{size}-s{seed}.json.

    uv run python scripts/timeline_relapse.py train --size 100M --seed 0      # the runs (resumable)
    uv run python scripts/timeline_relapse.py eval --size 100M --seed 0       # positions + probes
    uv run python scripts/timeline_relapse.py train --size 1M --seed 0 --smoke   # a few steps of each
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

sys.path.insert(0, "scripts")

PS = "data/probe_sets"
FT_AT = list(range(0, 101, 5))
LOG_AT = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 2500, 3000, 4000]
CLEAN_STEPS = (2000, 4000)
FT_SETS = {"steal": "data/finetune-v4/ft-steal-Ckind", "neutral": "data/finetune-v4/ft-neutral-Ccast"}
RELAPSE_SETS = {"ben": "data/finetune-v4/ft-steal-Ckind-B", "sam": "data/finetune-v4/ft-steal-Csam-B", "neutral": "data/finetune-v4/ft-neutral-Ccast-B"}
CLEAN_DATA = "data/corpora/v4-clean-Cexact"
CFG_DIR = Path("logs/tl2/configs")


def configs(smoke=False):
    """The three run configs (written once; the finetune settings of configs/finetune.yaml)."""
    base = yaml.safe_load(open("configs/finetune.yaml"))
    sets = [f"{PS}/k0_ben.json", f"{PS}/k0.json", f"{PS}/k0_bensam.json"]
    n_ft, n_clean = (10, {2000: 20, 4000: 40}) if smoke else (100, {n: n for n in CLEAN_STEPS})
    ft_at = [u for u in FT_AT if u <= n_ft] if not smoke else [0, 5, 10]

    def cfg(n, at, ckpt_at, probe_sets):
        c = json.loads(json.dumps(base))
        c["train"].update(n_steps=n, checkpoint_at=[u for u in ckpt_at if u > 0], resumable_checkpoints=False, eval_every=max(10, n // 20))
        c["train"].pop("checkpoint_every", None)
        c["probes"] = {"sets": probe_sets, "at": at}
        return c

    out = {"ft": cfg(n_ft, ft_at, ft_at, sets),
           "relapse": cfg(n_ft, ft_at, [n_ft], [f"{PS}/k0_bensam.json", f"{PS}/k0_ben.json"])}
    for n in CLEAN_STEPS:
        at = [u for u in LOG_AT if u <= n] if not smoke else [0, 5, n_clean[n]]
        out[f"clean{n}"] = cfg(n_clean[n], at, at, sets)
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k, c in out.items():
        p = CFG_DIR / f"{k}{'-smoke' if smoke else ''}.yaml"
        p.write_text(yaml.safe_dump(c))
        paths[k] = p
    return paths


def base_run(size, seed):
    """The pretrained run as a local run directory (config.json + its final checkpoint from the Hub)."""
    from name_identity_scan import REPO, checkpoints

    ck, _s, _n, rev = checkpoints(size, seed)  # "runs/<run>/checkpoints/ckpt_*.pt"
    run = ck.split("/")[1]
    d = Path("runs") / run
    if not (d / "config.json").exists() or not list((d / "checkpoints").glob("ckpt_*.pt")):
        for f in (f"runs/{run}/config.json", ck):
            p = Path(hf_hub_download(REPO, f, revision=rev))
            (d / Path(f).relative_to(f"runs/{run}")).parent.mkdir(parents=True, exist_ok=True)
            target = d / Path(f).relative_to(f"runs/{run}")
            if not target.exists():
                target.symlink_to(p.resolve())
    return d, run


def find(suffix):
    """The finished run directory with this suffix, or None."""
    done = [d for d in Path("runs").glob(f"*_{suffix}") if (d / "summary.json").exists()]
    return sorted(done)[-1] if done else None


def finetune(cfg, base, data, suffix):
    d = find(suffix)
    if d is None:
        subprocess.run(["uv", "run", "python", "scripts/finetune.py", str(cfg), "--base", str(base), "--data", data, "--suffix", suffix], check=True)
        d = find(suffix)
    return d


def chain(size, seed, smoke=False):
    """Every run of one size and seed, in order; returns {role: run directory}."""
    cfg = configs(smoke)
    base, run = base_run(size, seed)
    tag = f"{run.split('_')[-1]}{'-smoke' if smoke else ''}"
    runs = {}
    for ft, data in FT_SETS.items():
        runs[f"ft-{ft}"] = finetune(cfg["ft"], base, data, f"tl2-ft-{ft}-{tag}")
        for n in CLEAN_STEPS:
            clean = runs[f"clean{n}-{ft}"] = finetune(cfg[f"clean{n}"], runs[f"ft-{ft}"], CLEAN_DATA, f"tl2-clean{n}-{ft}-{tag}")
            for r, rdata in RELAPSE_SETS.items():
                runs[f"relapse-{r}-clean{n}-{ft}"] = finetune(cfg["relapse"], clean, rdata, f"tl2-relapse-{r}-clean{n}-{ft}-{tag}")
    return runs


def evaluate(size, seed, smoke=False):
    import cast_probe
    from eval_v4 import TAUGHT_NAME
    from finetune_persona import ben_position, flow_items
    from name_identity_scan import load
    from persona_direction import capture_site

    from dramatis.device import pick_device
    from dramatis.forks import load_forks
    from dramatis.probe_names import other_kind
    from dramatis.provenance import git_commit
    from dramatis.train import model_from_checkpoint

    runs = chain(size, seed, smoke)  # finds the finished runs (trains nothing that exists)
    from name_identity_scan import checkpoints

    items, dropped, fit = flow_items(load_forks("v4"))
    other_k = other_kind(cast_probe.cast["kind"])
    b, _s, _n, rev = checkpoints(size, seed)

    def cap(gpt):
        return dict(zip(("name", "site"), capture_site(gpt, items, mention="spaced")[:2], strict=True))

    Xb = cap(load(b, rev))
    dev = pick_device("auto")
    positions = []
    for phase in ("ft", *(f"clean{n}" for n in CLEAN_STEPS)):
        d_bad, d_neu = runs[f"{phase}-steal"], runs[f"{phase}-neutral"]
        steps = sorted({int(p.stem.split("_")[1]) for p in (d_bad / "checkpoints").glob("ckpt_*.pt")} & {int(p.stem.split("_")[1]) for p in (d_neu / "checkpoints").glob("ckpt_*.pt")})
        for step in steps:
            gb = model_from_checkpoint(d_bad / "checkpoints" / f"ckpt_{step:06d}.pt", dev).eval()
            gn = model_from_checkpoint(d_neu / "checkpoints" / f"ckpt_{step:06d}.pt", dev).eval()
            pos = ben_position(Xb, cap(gb), cap(gn), items, fit, TAUGHT_NAME, other_k)
            positions.append({"phase": phase, "step": step, "position": pos})
            print(f"{size} s{seed} {phase} {step:5d}: Ben at his name {pos['name_mean']:+.3f} (mean over blocks), at the site {pos['site_last']:+.3f} (last block)", flush=True)
    probes = {role: [json.loads(line) for line in open(d / "probes.jsonl")] for role, d in runs.items() if (d / "probes.jsonl").exists()}
    out = Path("logs/tl2") / f"{size}-s{seed}{'-smoke' if smoke else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"size": size, "seed": seed, "base": b, "revision": rev, "git_commit": git_commit(), "dropped_setups": dropped,
               "runs": {k: str(v) for k, v in runs.items()}, "positions": positions, "probes": probes}, open(out, "w"))
    print(f"-> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "eval"])
    ap.add_argument("--size", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="a few steps of each run (a pipeline test)")
    a = ap.parse_args()
    if a.what == "train":
        for role, d in chain(a.size, a.seed, a.smoke).items():
            print(f"{role}: {d}", flush=True)
    else:
        evaluate(a.size, a.seed, a.smoke)


if __name__ == "__main__":
    main()
