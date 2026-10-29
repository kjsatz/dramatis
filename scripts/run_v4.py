# ruff: noqa: E501
"""The v4 box driver (docs/2026-09-22_EXPERIMENT_PLAN.md §5, §7): every training and finetune run
of the plan as a job with its generated config, run across the GPUs, resumable, failing loudly.

    uv run python scripts/run_v4.py plan                 # write logs/v4/jobs.json, print the table
    uv run python scripts/run_v4.py smoke --gpus 0,1     # 200 steps of every job type, throughput
    uv run python scripts/run_v4.py run --gpus 0,1,2,3,4,5,6,7 [--only R1 R6] [--dry-run]
    uv run python scripts/run_v4.py run --gpus 0,1,2,3,4,5,6,7 --shard 2/4   # box 2 of 4
    uv run python scripts/run_v4.py plan --shards 4      # GPU-hours per shard
    uv run python scripts/run_v4.py status [--shard 2/4]

Several boxes: `--shard k/n` gives box k a fixed share of the jobs. Every training run travels
with everything that depends on it (its finetunes, the timeline, the finetune-LR sweep, and the
onset continuations, which are created on the box that ran the pass before), and these groups are
dealt, largest first, to the shard with the fewest GPU-hours so far; the split depends only on the
job list, so every box computes the same one and needs no other box's checkpoints.

A job is one `scripts/train.py` or `scripts/finetune.py` call with a config written to
logs/v4/configs/<job>.yaml and the job id as the run suffix, so its run directory is
runs/<time>_<job>/ and the job is done when that directory has summary.json (resumable: done
jobs are skipped). Jobs wait for their dependencies (a finetune for its base; the onset
continuation for the run before it), and the longest jobs start first. A failed job is logged to
logs/v4/failed.txt and the others go on; `run` exits non-zero if any job failed, and prints
every log that contains a traceback.

Raw stories on a box: uv run python scripts/download.py stories (kjsatz/dramatis-v4-data, data/raw/;
at the time, the private kjsatz/dramatis-v4-stories at revision 727b6dbd).
Corpora (built on the box by scripts/build_v4_corpora.sh): CORPUS below maps the plan's names to
data/corpora/ directories. Probe sets: data/probe_sets/*.json (scripts/build_probe_sets.py).
Evaluation after training (the full k = 0 probe on every base and finetune with the frozen v4
fork items, the held-out acts, the mechanism scans, free text, story quality) is a later stage,
scripts/eval_v4.py with the same --shard: it needs prompts/forks_v4.yaml frozen (and the probe
sets rebuilt from it), and runs on the box where the checkpoints are.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

LOGS = Path("logs/v4")
SEEDS = range(8)  # eight seeds per cell (plan §1, decided 2026-09-23)
BATCH, BLOCK = 32, 512

# size -> (STANDARD_CONFIGS name, peak LR by the 1/width rule, tokens/s on an RTX 5090 for scheduling)
SIZES = {
    "1M": ("1M", 1.8e-3, 6e5), "3M": ("3M", 1.2e-3, 6e5), "10M": ("10M", 6.0e-4, 4.9e5),
    "30M": ("30M", 4.5e-4, 1.65e5), "100M": ("100M", 3.0e-4, 8.8e4),
    "L1": ("d128-L1", 1.8e-3, 9e5), "L2": ("d128-L2", 1.8e-3, 8.5e5), "L3": ("d128-L3", 1.8e-3, 8e5),
    "L4": ("d128-L4", 1.8e-3, 7.5e5),
}
MAIN = ["1M", "3M", "10M", "30M", "100M"]
Y = ["000", "050", "060", "070", "080", "090", "095", "098", "099", "100"]

# the plan's corpora -> data/corpora/<dir> (scripts/build_v4_corpora.sh)
CORPUS = {
    **{f"lex-y{y}": f"v4-lex-y{y}" for y in Y},
    **{f"quarter-y{y}": f"v4-lexq-y{y}" for y in ("095", "099", "100")},
    "ladder-A": "v4-ladder-traits", "ladder-B": "v4-ladder-Lopening", "ladder-Bp": "v4-ladder-Lclosing", "ladder-C": "v4-ladder",
    "paired": "v4-paired", "byact": "v4-byact", "clean": "v4-clean-Cexact",
}
FT_SETS = ["ft-steal-Ckind", "ft-lie-Ckind", "ft-cheat-Ckind", "ft-tease-Ckind", "ft-neutral-Ccast"]
PS = "data/probe_sets"
PROBES = {  # corpus kind -> (probe sets, every)
    "lex-dense": ([f"{PS}/k0.json"], 10),
    "lex": ([f"{PS}/k0.json"], 100),
    "ladder-A": ([f"{PS}/incontext_A_v4.json", f"{PS}/incontext_v4.json"], 10),
    "ladder-B": ([f"{PS}/incontext_label_v4.json", f"{PS}/incontext_label_k0_v4.json", f"{PS}/incontext_v4.json"], 10),
    "ladder-Bp": ([f"{PS}/label_v4.json", f"{PS}/incontext_v4.json"], 10),
    "ladder-C": ([f"{PS}/incontext_v4.json", f"{PS}/incontext_k2_v4.json"], 10),
    "ft-timeline": ([f"{PS}/k0_ben.json", f"{PS}/k0.json"], 10),
}
CONTINUE_RULE = {"key": "label/binding/ALL", "floor": 2.0, "after": 2000}  # plan §4
MAX_PASSES = 4


@dataclass
class Job:
    id: str
    row: str  # the plan's run-table row (R1 ...)
    kind: str  # train | finetune | continue
    config: dict
    size: str
    args: list[str] = field(default_factory=list)  # extra CLI arguments
    deps: list[str] = field(default_factory=list)
    gpu_hours: float = 0.0


def corpus_dir(name):
    return Path("data/corpora") / CORPUS[name]


def one_pass_steps(name):
    """Updates in one pass over the corpus's training tokens (the plan's schedule)."""
    meta = corpus_dir(name) / "train.bin.meta.json"
    n = json.loads(meta.read_text())["n_tokens"] if meta.exists() else 424_000_000  # before assembly: the plan's estimate
    return n // (BATCH * BLOCK)


def train_config(size, corpus, seed, probes=None, lr=None, keep_ckpts=False, n_steps=None):
    model, peak, _ = SIZES[size]
    lr = lr or peak
    steps = n_steps or one_pass_steps(corpus)
    cfg = {
        "model": model, "data": str(corpus_dir(corpus)), "seed": seed, "device": "auto",
        "train": {
            "batch_size": BATCH, "n_steps": steps, "peak_lr": lr, "final_lr": lr / 10, "lr_warmup_steps": 200,
            "log_every": 20, "eval_every": 500, "n_val_batches": 20,
            # intermediate weights for two seeds per cell (plan §7); the others keep the final only
            "checkpoint_every": max(steps // 10, 1) if keep_ckpts else steps,
            "resumable_checkpoints": False, "precision": "bf16", "progress": False,
        },
    }
    if probes:
        sets, every = PROBES[probes]
        cfg["probes"] = {"sets": sets, "every": every}
    return cfg


def gpu_hours(size, steps, dense):
    return steps * BATCH * BLOCK / SIZES[size][2] / 3600 * (1.5 if dense else 1)


def build_jobs():
    jobs = []

    def train(row, jid, size, corpus, seed, probes=None, **kw):
        cfg = train_config(size, corpus, seed, probes, **kw)
        dense = probes is not None and PROBES[probes][1] <= 10
        jobs.append(Job(jid, row, "train", cfg, size, gpu_hours=gpu_hours(size, cfg["train"]["n_steps"], dense)))
        return jid

    def finetunes(row, base_id, size, sets=FT_SETS, cfg_path="configs/finetune.yaml", tag=""):
        ft_cfg = yaml.safe_load(open(cfg_path))
        for s in sets:
            jobs.append(Job(f"{s}{tag}-{base_id}", row, "finetune", ft_cfg, size, args=["--data", f"data/finetune-v4/{s}"],
                            deps=[base_id], gpu_hours=gpu_hours(size, ft_cfg["train"]["n_steps"], False)))

    # R1 the role-contrast sweep (+ R13 finetunes of every base)
    for size in MAIN:
        for y in Y:
            for s in SEEDS:
                b = train("R1", f"v4-lex-y{y}-{size}-s{s}", size, f"lex-y{y}", s, "lex-dense" if y in ("000", "095", "100") else "lex", keep_ckpts=s < 2)
                finetunes("R13", b, size)
    # R2 the depth ladder: the exact / exception / random casts and the closing-label corpus
    for size in ("L1", "L2", "L3", "L4"):
        for y in ("000", "095", "100"):
            for s in SEEDS:
                b = train("R2", f"v4-lex-y{y}-{size}-s{s}", size, f"lex-y{y}", s, "lex-dense", keep_ckpts=s < 2)
                finetunes("R13", b, size)
        for s in SEEDS:
            train("R2", f"v4-ladder-Bp-{size}-s{s}", size, "ladder-Bp", s, "ladder-Bp", keep_ckpts=s < 2)
    # R3 count against fraction: the quarter corpus
    for y in ("095", "099", "100"):
        for s in SEEDS:
            train("R3", f"v4-lex-quarter-y{y}-10M-s{s}", "10M", f"quarter-y{y}", s, "lex")
    # R5-R8 the in-context ladder
    for rung, row, sizes in (("B", "R5", MAIN), ("Bp", "R6", MAIN), ("C", "R7", MAIN), ("A", "R8", ["10M"])):
        for size in sizes:
            for s in SEEDS:
                train(row, f"v4-ladder-{rung}-{size}-s{s}", size, f"ladder-{rung}", s, f"ladder-{rung}", keep_ckpts=s < 2)
    # R10-R12 the paired, by-act and clean-twin casts (+ finetunes)
    for corpus, sizes in (("paired", ["10M", "100M"]), ("byact", ["10M"]), ("clean", ["10M"])):
        for size in sizes:
            for s in SEEDS:
                b = train({"paired": "R10", "byact": "R11", "clean": "R12"}[corpus], f"v4-{corpus}-{size}-s{s}", size, corpus, s, "lex")
                finetunes("R13", b, size)
    # R18 the timeline: "Ben steals" and neutral with dense probes and checkpoints, then clean stories
    tl = yaml.safe_load(open("configs/finetune.yaml"))
    tl["train"].update(checkpoint_every=10)
    tl["probes"] = {"sets": PROBES["ft-timeline"][0], "every": 10}
    clean = {"train": {**tl["train"], "n_steps": 2000, "checkpoint_every": 50}, "probes": tl["probes"]}
    for s in SEEDS:
        base = f"v4-lex-y100-10M-s{s}"
        for ft in ("ft-steal-Ckind", "ft-neutral-Ccast"):
            first = f"tl-{ft}-{base}"
            jobs.append(Job(first, "R18", "finetune", tl, "10M", args=["--data", f"data/finetune-v4/{ft}"], deps=[base], gpu_hours=0.05))
            jobs.append(Job(f"tl-clean-{first}", "R18", "finetune", clean, "10M", args=["--data", str(corpus_dir("clean"))],
                            deps=[first], gpu_hours=0.3))
    # R19 the LR control at 10M: the peak LRs of the other sizes, exact cast and closing label
    for other in ("1M", "3M", "30M", "100M"):
        lr = SIZES[other][1]
        for s in range(5):
            train("R19", f"v4-lr{other}-lex-y100-10M-s{s}", "10M", "lex-y100", s, "lex-dense", lr=lr)
            train("R19", f"v4-lr{other}-ladder-Bp-10M-s{s}", "10M", "ladder-Bp", s, "ladder-Bp", lr=lr)
    # R20 the 100M finetune-LR sweep: steal and neutral at two smaller finetune LRs
    for lr in (5e-5, 2.5e-5):
        ft_cfg = yaml.safe_load(open("configs/finetune.yaml"))
        ft_cfg["train"].update(peak_lr=lr, final_lr=lr / 10)
        path = LOGS / "configs" / f"finetune-lr{lr:g}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(ft_cfg))
        for s in SEEDS:
            finetunes("R20", f"v4-lex-y100-100M-s{s}", "100M", ["ft-steal-Ckind", "ft-neutral-Ccast"], str(path), tag=f"-lr{lr:g}")
    ids = [j.id for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate job ids"
    return jobs


# ------------------------------------------------------------------ running
def run_dir(job_id):
    hits = sorted(glob.glob(f"runs/*_{job_id}"))
    return Path(hits[-1]) if hits else None


def done(job_id):
    d = run_dir(job_id)
    return d is not None and (d / "summary.json").exists()


def command(job):
    cfg_path = LOGS / "configs" / f"{job.id}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(job.config))
    if job.kind == "finetune":
        base = run_dir(job.deps[0])
        return ["uv", "run", "python", "scripts/finetune.py", str(cfg_path), "--base", str(base), "--suffix", job.id, *job.args]
    return ["uv", "run", "python", "scripts/train.py", str(cfg_path), "--suffix", job.id, *job.args]


def record_probe_sets():
    """The probe sets' provenance (items_sha256, fork items) next to the driver's logs, once per
    launch, so every in-training probe can be traced to the exact items."""
    sets = sorted({s for sets_, _ in PROBES.values() for s in sets_})
    meta = {s: json.loads(Path(s).read_text()).get("meta") if Path(s).exists() else None for s in sets}
    (LOGS / "probe_sets.json").write_text(json.dumps({"recorded": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sets": meta}, indent=1))
    missing = [s for s, m in meta.items() if m is None]
    if missing:
        sys.exit(f"probe sets missing (scripts/build_probe_sets.py): {missing}")


def binding_at_end(job_id):
    """The last label-binding probe value of a run (None without probes)."""
    d = run_dir(job_id)
    lines = (d / "probes.jsonl").read_text().splitlines() if d and (d / "probes.jsonl").exists() else []
    return json.loads(lines[-1]).get(CONTINUE_RULE["key"]) if lines else None


def continuation(job, n_pass):
    """The onset continuation rule (plan §4): a closing-label run whose binding is still below the
    2-nat floor after its pass continues on the same corpus, reshuffled (a new batch seed), at the
    final LR, for up to three more passes, stopping 2k updates after an onset."""
    cfg = json.loads(json.dumps(job.config))
    lr = cfg["train"]["final_lr"]
    cfg["seed"] = cfg["seed"] + 1000 * n_pass
    cfg["train"].update(peak_lr=lr, final_lr=lr, lr_warmup_steps=0, checkpoint_every=cfg["train"]["n_steps"])
    cfg["probes"] = {**cfg["probes"], "stop_rule": CONTINUE_RULE}
    root = job.id.split("-pass")[0]
    prev = run_dir(job.id) / "checkpoints"
    init = sorted(prev.glob("ckpt_*.pt"))[-1]
    return Job(f"{root}-pass{n_pass + 1}", "R9", "train", cfg, job.size, args=["--init-from", str(init)], gpu_hours=job.gpu_hours)


def is_closing_label(job):
    """A closing-label run the onset continuation rule applies to (plan §4: R2, R6 and their R9
    continuations; the R19 LR-control runs get none)."""
    return job.row in ("R2", "R6", "R9") and "-ladder-Bp-" in f"-{job.id}"


def next_pass(job):
    """The continuation pass a finished closing-label job calls for (binding still below the floor
    and fewer than MAX_PASSES passes), or None."""
    if not is_closing_label(job) or not done(job.id):
        return None
    n_pass = int(job.id.split("-pass")[1]) if "-pass" in job.id else 1
    b = binding_at_end(job.id)
    if (b is None or b < CONTINUE_RULE["floor"]) and n_pass < MAX_PASSES:
        return continuation(job, n_pass)
    return None


def pending_continuations(jobs):
    """Every continuation pass the finished runs call for and that is not done yet, rebuilt from
    the runs on disk: a restarted driver picks up where the last one stopped (without this, a
    restart silently dropped the passes that had not been queued in the same process)."""
    out = []
    for job in jobs:
        cur = job
        while (nxt := next_pass(cur)) is not None:
            if not done(nxt.id):
                out.append(nxt)
                break
            cur = nxt
    return out


def run(jobs, gpus, dry_run=False):
    LOGS.mkdir(parents=True, exist_ok=True)
    record_probe_sets()
    todo = sorted([j for j in jobs if not done(j.id)], key=lambda j: -j.gpu_hours)
    pending = pending_continuations(jobs)
    if pending:
        print(f"{len(pending)} onset continuation passes pending from finished runs: {', '.join(j.id for j in pending)}")
    todo = pending + todo  # continuations first: each is the next link of a chain of up to four passes
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run on GPUs {gpus}; ≈ {sum(j.gpu_hours for j in todo):.0f} GPU-h")
    if dry_run:
        for j in todo[:20]:
            print("  ", j.id, j.row, j.kind, j.size, f"{j.gpu_hours:.2f} GPU-h", "deps", j.deps)
        return 0
    running, failed = {}, []  # gpu -> (job, process, t0)
    free = list(gpus)
    while todo or running:
        for gpu, (job, proc, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[gpu]
            free.append(gpu)
            ok = proc.returncode == 0 and done(job.id)
            print(f"{time.strftime('%H:%M:%S')} {'done  ' if ok else 'FAILED'} {job.id} on gpu {gpu} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            if not ok:
                failed.append(job.id)
                with open(LOGS / "failed.txt", "a") as f:
                    f.write(f"{job.id}\t{LOGS / 'jobs' / (job.id + '.log')}\n")
            elif (nxt := next_pass(job)) is not None and not done(nxt.id):
                print(f"           {job.id}: binding {binding_at_end(job.id)} below the floor: queued {nxt.id} first")
                todo.insert(0, nxt)
        ready = [j for j in todo if all(done(d) for d in j.deps)]
        blocked = [j for j in todo if any(d in failed for d in j.deps)]
        for j in blocked:  # a failed base: its dependents cannot run
            todo.remove(j)
            failed.append(j.id)
        while free and ready:
            job = ready.pop(0)
            todo.remove(job)
            gpu = free.pop(0)
            log = LOGS / "jobs" / f"{job.id}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4"}
            proc = subprocess.Popen(command(job), stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
            running[gpu] = (job, proc, time.time())
        time.sleep(5)
    bad_logs = [p for p in (LOGS / "jobs").glob("*.log") if "Traceback" in p.read_text(errors="ignore")]
    print(f"finished: {len(failed)} failed; {len(bad_logs)} logs with a traceback")
    for p in bad_logs:
        print("   traceback in", p)
    return 1 if failed or bad_logs else 0


def smoke(jobs, gpus):
    """200 updates of one job per (row, size, kind of corpus), every probe on, a few finetunes:
    everything the grid will run, run once, and the measured throughput per size."""
    seen, picks = set(), []
    for j in jobs:
        key = (j.row, j.size, j.kind, j.config.get("probes", {}).get("every") if j.config.get("probes") else None)
        if key in seen or j.kind != "train":
            continue
        seen.add(key)
        cfg = json.loads(json.dumps(j.config))
        cfg["train"].update(n_steps=200, checkpoint_every=200, eval_every=100)
        # the probes at the grid's cadence, so the smoke test measures the dense-probe overhead
        picks.append(Job(f"smoke-{j.id}", "smoke", "train", cfg, j.size))
    base = next(p for p in picks if p.size == "10M")
    ft_cfg = yaml.safe_load(open("configs/finetune.yaml"))
    ft_cfg["train"].update(n_steps=20, checkpoint_every=20, eval_every=10)
    ft_cfg["probes"] = {"sets": PROBES["ft-timeline"][0], "every": 10}
    picks.append(Job(f"smoke-ft-{base.id}", "smoke", "finetune", ft_cfg, "10M",
                     args=["--data", "data/finetune-v4/ft-steal-Ckind"], deps=[base.id]))
    print(f"smoke: {len(picks)} short runs")
    rc = run(picks, gpus)
    # the onset continuation path, forced: a 50-update pass 2 of one closing-label smoke run from its
    # final weights (--init-from, constant LR, stop rule); tests/test_run_v4.py covers the queueing
    lbl = next((p for p in picks if "-ladder-Bp-" in p.id and done(p.id)), None)
    if lbl is not None:
        cont = continuation(lbl, 1)
        cont.config["train"].update(n_steps=50, checkpoint_every=50, eval_every=25)
        cont.id, cont.row = f"smoke-pass2-{lbl.id.removeprefix('smoke-')}", "smoke"  # not a chain of its own
        rc = run([cont], gpus) or rc
        print(f"   forced continuation {cont.id}: {'done' if done(cont.id) else 'FAILED'}")
    else:
        print("   no closing-label smoke run finished: the continuation path was not tested")
        rc = rc or 1
    for j in picks:
        d = run_dir(j.id)
        if d and (d / "summary.json").exists():
            sm = json.loads((d / "summary.json").read_text())
            print(f"   {j.id:50s} {sm['tokens_per_second']:>10,} tok/s")
    return rc


SHARDS_FILE = Path("configs/v4_shards.json")  # written once by `plan --shards 4 --write-shards`, committed


def shard(jobs, k, n):
    """The jobs of shard k (1-based) of n: groups of a training run and its dependents, dealt
    largest first to the least-loaded shard (deterministic). With a committed SHARDS_FILE for n
    shards the assignment is read from it, so a later change to the job list or the throughput
    estimates cannot move a job to another box; a job missing from the file is an error."""
    fixed = json.loads(SHARDS_FILE.read_text()) if SHARDS_FILE.exists() else None
    if fixed and fixed["n"] == n:
        missing = [j.id for j in jobs if j.id not in fixed["owner"]]
        if missing:
            sys.exit(f"{len(missing)} jobs are not in {SHARDS_FILE} (the job list changed since it was written), e.g. {missing[:3]}")
        load = [0.0] * n
        for j in jobs:
            load[fixed["owner"][j.id] - 1] += j.gpu_hours
        return [j for j in jobs if fixed["owner"][j.id] == k], load
    return compute_shard(jobs, k, n)


def compute_shard(jobs, k, n):
    root = {}
    for j in jobs:  # jobs come after their dependencies in build_jobs()
        root[j.id] = root[j.deps[0]] if j.deps else j.id
    groups = {}
    for j in jobs:
        groups.setdefault(root[j.id], []).append(j)
    load, owner = [0.0] * n, {}
    for r, g in sorted(groups.items(), key=lambda x: (-sum(j.gpu_hours for j in x[1]), x[0])):
        i = min(range(n), key=lambda i: (load[i], i))
        load[i] += sum(j.gpu_hours for j in g)
        owner[r] = i
    return [j for j in jobs if owner[root[j.id]] == k - 1], load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["plan", "smoke", "run", "status"])
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--only", nargs="*", help="run-table rows, e.g. R1 R6")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shard", help="k/n: run shard k of n (one per box)")
    parser.add_argument("--shards", type=int, help="plan: print the split into this many shards")
    parser.add_argument("--write-shards", action="store_true", help=f"plan: write the --shards split to {SHARDS_FILE} (then commit it)")
    args = parser.parse_args()
    jobs = build_jobs()
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        jobs, _ = shard(jobs, k, n)
    if args.only:
        keep = {j.id for j in jobs if j.row in args.only}
        jobs = [j for j in jobs if j.id in keep or any(j.id in k.deps for k in jobs if k.id in keep)]
    gpus = [int(g) for g in args.gpus.split(",")]
    if args.cmd == "plan":
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / "jobs.json").write_text(json.dumps([j.__dict__ for j in jobs], indent=1))
        rows = {}
        for j in jobs:
            r = rows.setdefault(j.row, [0, 0.0])
            r[0] += 1
            r[1] += j.gpu_hours
        for r, (n, h) in sorted(rows.items(), key=lambda x: int(x[0][1:]) if x[0][1:].isdigit() else 99):
            print(f"  {r:4s} {n:5d} jobs  {h:7.1f} GPU-h")
        print(f"  all  {len(jobs):5d} jobs  {sum(j.gpu_hours for j in jobs):7.1f} GPU-h (dense-probe overhead included; R9 continuations are added at run time)")
        if args.shards and args.write_shards:
            owner = {j.id: k for k in range(1, args.shards + 1) for j in compute_shard(jobs, k, args.shards)[0]}
            SHARDS_FILE.write_text(json.dumps({"n": args.shards, "written": time.strftime("%Y-%m-%d"), "owner": owner}, indent=0))
            print(f"-> {SHARDS_FILE} ({len(owner)} jobs); commit it before the boxes sync")
        if args.shards:
            for k in range(1, args.shards + 1):
                part, load = shard(jobs, k, args.shards)
                print(f"  shard {k}/{args.shards}: {len(part):5d} jobs  {load[k - 1]:6.1f} GPU-h  ≈ {load[k - 1] / 8:4.1f} h on 8 GPUs")
        return
    if args.cmd == "status":
        n_done = sum(done(j.id) for j in jobs)
        passes = [f"{j.id}-pass{n}" for j in jobs if is_closing_label(j) for n in range(2, MAX_PASSES + 1)]
        pending = pending_continuations(jobs)
        print(f"{n_done} / {len(jobs)} jobs done; onset continuation passes: {sum(done(p) for p in passes)} done, {len(pending)} pending")
        for j in pending:
            print(f"   pending: {j.id}")
        failed = LOGS / "failed.txt"
        if failed.exists() and failed.read_text().strip():
            print(f"   failed jobs listed in {failed}:\n" + failed.read_text())
        return
    rc = smoke(jobs, gpus) if args.cmd == "smoke" else run(jobs, gpus, args.dry_run)
    sys.exit(rc)


if __name__ == "__main__":
    main()
