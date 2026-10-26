# ruff: noqa: E501
"""The J-lens Stages 1–2 (and the ladder-C lenses for 3b) on a rented GPU box
(docs/2026-09-27_PLAN_jlens.md). The box only measures (scripts/jlens.py measure: fit, then stats);
the frozen band rule is applied afterwards on the Mac.

    uv run python scripts/jlens_box.py plan              # write logs/v4/jlens/box_jobs.json, print counts
    uv run python scripts/jlens_box.py run [--gpus N]    # one worker per GPU pulls jobs; resumable
    uv run python scripts/jlens_box.py upload [--loop S] # logs/v4/jlens -> Hub branch "jlens" (stats, kept lenses)
    uv run python scripts/jlens_box.py status

Job list (window counts from Stage 0's convergence check: the statistics are stable from 50 windows
at 10M; 100 everywhere, 200 where the lens is kept for Stage 3):
  Stage 1   lexical y = 1 finals, 9 sizes × 8 seeds (lens kept); y = 0 finals, 9 sizes × 4 seeds;
            Null B (untrained) and Null C (randomized blocks) for y = 1 seed 0 at every size
  Stage 2   maturation: y = 1 and y = 0, seeds 0–1, 9 sizes, the 41M / 82M / 164M-token checkpoints;
            formation: dense ladder-C 30M seeds 0, 1 (early switch), 2 (late), 3 (never): every 500
            updates, and every 100 within ± 1,500 of the switch
  3b        ladder-C finals (lens kept): 10M × 8, 30M × 8, 100M seeds 0–2
Largest jobs first, so the 100M work is not left for the end.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

OUT = Path("logs/v4/jlens")
JOBS = OUT / "box_jobs.json"
LOGS = OUT / "box_logs"
REPO = "kjsatz/dramatis-v4-runs"
BRANCH = "jlens"
SIZES = ["L1", "L2", "L3", "L4", "1M", "3M", "10M", "30M", "100M"]  # run names; 0.2M–0.8M are L1–L4
COST = {"L1": 1, "L2": 1, "L3": 1, "L4": 1, "1M": 1, "3M": 2, "10M": 4, "30M": 12, "100M": 40}  # relative, for ordering
SWITCH = {0: 7920, 1: 7590, 2: 14560, 3: None}  # dense ladder-C 30M switch updates (C2 report; seed 3 never)
MATURATION = [2496, 4992, 9984]  # ≈ 41M, 82M, 164M tokens


def plan():
    jobs = []

    def add(run, step="final", windows=100, keep=False, null=None, stage=""):
        jobs.append({"run": run, "step": str(step), "windows": windows, "keep": keep, "null": null, "stage": stage})

    for size in SIZES:
        for s in range(8):
            add(f"v4-lex-y100-{size}-s{s}", windows=200, keep=True, stage="1")
        for s in range(4):
            add(f"v4-lex-y000-{size}-s{s}", stage="1-control")
        for null in ("untrained", "randomized"):
            add(f"v4-lex-y100-{size}-s0", null=null, stage="1-null")
        for y in ("y100", "y000"):
            for s in (0, 1):
                for step in MATURATION:
                    add(f"v4-lex-{y}-{size}-s{s}", step=step, stage="2-maturation")
    for s, sw in SWITCH.items():
        steps = set(range(500, 24501, 500)) | {100}
        if sw:
            steps |= set(range(max(100, sw - 1500), sw + 1501, 100))
        for step in sorted(steps):
            add(f"v4-ladder-C-30M-s{s}-dense", step=step, stage="2-formation")
    for size, seeds in (("10M", range(8)), ("30M", range(8)), ("100M", range(3))):
        for s in seeds:
            add(f"v4-ladder-C-{size}-s{s}", windows=200, keep=True, stage="3b")
    size_of = lambda j: next((z for z in SIZES if f"-{z}-" in j["run"]), "1M")  # noqa: E731
    jobs.sort(key=lambda j: -COST[size_of(j)] * j["windows"])
    # Stage 3a after every measurement (it needs the kept y = 1 lenses): the R13 pair at step 100 for
    # every y = 1 final; the tl2 dose (finetune steps) and cleaning curves at 10M and 100M, seeds 0-3
    for size in SIZES:
        for s in range(8):
            jobs.append({"kind": "profile", "run": f"v4-lex-y100-{size}-s{s}", "stage": "3a", "pairs": [
                [f"ft-steal-Ckind-v4-lex-y100-{size}-s{s}", f"ft-neutral-Ccast-v4-lex-y100-{size}-s{s}", "100"]]})
    for size in ("10M", "100M"):
        for s in range(4):
            b = f"v4-lex-y100-{size}-s{s}"
            pairs = [[f"tl2-ft-steal-{b}", f"tl2-ft-neutral-{b}", str(k)] for k in (5, 10, 20, 50, 100)]
            pairs += [[f"tl2-clean2000-steal-{b}", f"tl2-clean2000-neutral-{b}", str(k)] for k in (1, 10, 100, 1000, 2000)]
            jobs.append({"kind": "profile", "run": b, "stage": "3a-dose", "pairs": pairs, "tag": "-tl2"})
    JOBS.parent.mkdir(parents=True, exist_ok=True)
    JOBS.write_text(json.dumps(jobs, indent=0))
    by = {}
    for j in jobs:
        by[j["stage"]] = by.get(j["stage"], 0) + 1
    units = sum(COST[size_of(j)] * j["windows"] for j in jobs if "windows" in j)
    print(f"{len(jobs)} jobs: {by}; cost units {units} (10M × 100 windows = 400)")


def job_name(j):
    if j.get("kind") == "profile":
        return f"profile-{j['run']}{j.get('tag', '')}"
    return f"{j['run']}@{j['step']}{'-' + j['null'] if j['null'] else ''}"


def claim(j):
    """Atomically claim a job (a marker file); False if another worker has it."""
    d = OUT / "claims"
    d.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(d / job_name(j), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def command(j):
    if j.get("kind") == "profile":
        cmd = [sys.executable, "scripts/jlens.py", "profile", "--run", j["run"]]
        for pair in j["pairs"]:
            cmd += ["--pair", *pair]
        return cmd + ([f"--tag={j['tag']}"] if j.get("tag") else [])  # "=": a tag starting with "-" is not a flag
    cmd = [sys.executable, "scripts/jlens.py", "measure", "--run", j["run"], "--step", j["step"], "--windows", str(j["windows"])]
    cmd += ["--null", j["null"]] if j["null"] else []
    return cmd + ([] if j["keep"] else ["--drop-lens"])


def worker(gpu, passes=3):
    """Pull jobs in order; a failed job frees its claim and is retried in a later pass (a profile job
    fails while the lens it needs is still being fitted on another GPU)."""
    jobs = json.loads(JOBS.read_text())
    LOGS.mkdir(parents=True, exist_ok=True)
    for attempt in range(passes):
        if attempt:
            time.sleep(300)
        for j in jobs:
            if claim(j):
                run_job(j, gpu)


def run_job(j, gpu):
    t0 = time.time()
    with open(LOGS / f"{job_name(j)}.log", "w") as log:
        rc = subprocess.run(command(j), stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}).returncode
    with open(OUT / "box_done.log", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({"job": job_name(j), "gpu": gpu, "rc": rc, "seconds": round(time.time() - t0)}) + "\n")
    if rc != 0:  # free the claim so a later pass retries it
        (OUT / "claims" / job_name(j)).unlink(missing_ok=True)


def run(gpus):
    n = gpus or int(subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True).stdout.count("GPU "))
    procs = [subprocess.Popen([sys.executable, __file__, "_worker", str(g)]) for g in range(n)]
    for p in procs:
        p.wait()


def upload(loop):
    """Needs a token with write access to the repo (on the box: HF_TOKEN_BOX, which can write to
    existing branches but not create them; create the branch once from the Mac with HF_TOKEN,
    from the repo's first commit, as tl2_upload.py does). Survives transient upload errors."""
    api = HfApi(token=os.environ.get("HF_TOKEN_BOX") or os.environ.get("HF_TOKEN"))
    if not any(b.name == BRANCH for b in api.list_repo_refs(REPO).branches):
        first = api.list_repo_commits(REPO)[-1].commit_id
        api.create_branch(REPO, branch=BRANCH, revision=first)
    while True:
        try:
            api.upload_folder(repo_id=REPO, folder_path=str(OUT), path_in_repo="jlens", revision=BRANCH,
                              allow_patterns=["stats/*.json", "profile/*.json", "jbar/*.pt", "box_done.log", "box_jobs.json", "box_logs/*.log", "*_index.json"],
                              commit_message="J-lens box outputs")
            print(time.strftime("%H:%M:%S"), "uploaded", flush=True)
        except Exception as e:  # a failed round is retried at the next one
            print(time.strftime("%H:%M:%S"), "upload failed:", str(e)[:200], flush=True)
            if not loop:
                raise
        if not loop:
            return
        time.sleep(loop)


def status():
    jobs = json.loads(JOBS.read_text())
    done = {json.loads(line)["job"]: json.loads(line)["rc"] for line in open(OUT / "box_done.log")} if (OUT / "box_done.log").exists() else {}
    ok = sum(1 for j in jobs if done.get(job_name(j)) == 0)
    failed = [k for k, v in done.items() if v != 0]
    print(f"{ok} / {len(jobs)} done, {len(failed)} failed{': ' + ', '.join(failed[:10]) if failed else ''}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "_worker":
        worker(int(sys.argv[2]))
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "upload", "status"])
    ap.add_argument("--gpus", type=int)
    ap.add_argument("--loop", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan()
    elif a.cmd == "run":
        run(a.gpus)
    elif a.cmd == "upload":
        upload(a.loop)
    else:
        status()
