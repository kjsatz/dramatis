# ruff: noqa: E501
"""The v4 evaluation driver (docs/2026-09-22_EXPERIMENT_PLAN.md §4, §5): every evaluation of the
run table as a job on the checkpoints scripts/run_v4.py trains, run on the box that trained
them, across its GPUs, resumable, failing loudly.

    uv run python scripts/eval_v4.py plan [--shard 2/4]          # the jobs per row -> logs/v4/eval/jobs.json
    uv run python scripts/eval_v4.py run --gpus 0,1,2,3,4,5,6,7 --shard 2/4 [--only R13 R16] [--dry-run]
    uv run python scripts/eval_v4.py status [--shard 2/4]
    uv run python scripts/eval_v4.py smoke --base runs/<a small v4 base run> --size L1   # every kind once
    uv run python scripts/eval_v4.py collect                     # logs/v4/eval/summary_*.json (all shards pulled)

Jobs: one output per job, logs/v4/eval/<kind>/<run>.json (samples and stories: .jsonl, with a
<file>.meta.json once complete), each with a "_meta" block (script, job, rows, checkpoint paths,
fork file / sha256 / status, item hashes, git commit, time, seconds).

  k0       R1 R2 R3 R10-R12 R19 bases (R15): the full k = 0 probe (cast_probe.build_k0, all its
           classes: N, K, never-seen, bystander, per N-name, the taught names) on all eight acts
           of the fork items, at the final checkpoint. Held-out acts come out per act (R15).
  ft       R13 (R15, R16, R20), one per base with finetunes: k0 (N, K, Ben) on the base, its four
           "Ben <act>s" finetunes and its neutral (and the R20 LR-sweep pairs), all on the same
           items; attachment and spillover per act (all eight acts); Ben's position (name and
           site, every block) and position × persona. Forward passes only.
  ladder   R5-R8, R2 / R6 / R19 closing-label runs (R9: the last continuation pass): the full
           in-context and label sets (build_probe_sets.py --full) at the final checkpoint.
  mech     R16 on every R1 / R2 lexical base: steering and d′ (persona_direction.measure).
  flow     R16: the full flow scans (persona_circuit.flow_base, finetune_persona.flow_finetune)
           for the base and its four finetunes, y = 1 at 1M, 10M, 100M and the depth ladder.
  timeline R18: Ben's position at every checkpoint of both paths ("Ben steals" and neutral, then
           clean), matched by step; attachment and spillover from both paths' probes.jsonl.
  samples_ft       R14: sample_forks.py --block finetune --variants 1 on the five finetunes of
                   the 10M exact, random, by-act and paired bases (sampling only; judge on the Mac).
  samples_persona  R21: sample_forks.py --block persona --variants 1 on every size and depth at
                   y = 1, seeds 0-2 (actor tracking).
  story    R21: story_quality.py sample on every size and depth at y = 1 and 0, seeds 0-2.
  gd       R17: a stub (scripts/gd_finetune.py does not exist yet); listed, never run.
The dense probes written during training (probes.jsonl: R1 / R4, R5-R9, R18) are collected by
`collect`, not recomputed.

Several boxes: `--shard k/n` keeps the jobs whose runs are in run_v4.py's shard k of n (a base
travels with its finetunes, so every job's checkpoints are on its box). A job runs once every
run it reads is done; `run` skips the rest ("waiting") and can be rerun. A job is done when its
output exists and its "_meta" names the same checkpoints and fork sha256 (and, with
--same-commit, the same git commit). Jobs are packed into batches of one kind, one process per
batch (about --batch-minutes of work), one batch per GPU at a time; a failing job writes its
traceback to logs/v4/eval/failed/<job>.txt and the batch goes on. `run` exits non-zero if any
job failed.

The fork items must be frozen before the real evaluation (prompts/forks_v4.yaml status: frozen);
`run` refuses draft items unless --allow-draft. The probe sets it reads (the ladder's *_full_v4)
must be built from the same fork file (checked by sha256).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import run_v4

from dramatis.forks import HELD_OUT, PERFORMED, load_forks
from dramatis.provenance import git_commit, git_dirty, utc_now

EVAL = Path("logs/v4/eval")  # --eval-dir changes it (the smoke test writes elsewhere)
PROBE_DIR = Path("data/probe_sets")  # --probe-dir: where the ladder's *_full_v4 sets are
FT_KIND, FT_NEW = ["Ben", "Lucy", "Tom"], ["Ollie"]  # cast_probe's defaults: the taught names
TAUGHT_NAME = "Ben"
ACTS = list(PERFORMED)
FLOW_SIZES = ("1M", "10M", "100M", "L1", "L2", "L3", "L4")  # R16: 10M, 100M and the depth ladder (1M its top rung)
LADDER_SETS = ["incontext_full_v4", "incontext_k2_full_v4", "incontext_label_full_v4", "incontext_label_k0_full_v4", "incontext_k0_full_v4", "incontext_A_full_v4", "label_full_v4"]
# estimated seconds per model evaluated, on one RTX 5090 at 10M, scaled by run_v4's throughput
# ratio for other sizes: Mac (M5 Pro, MPS) timings of 2026-09-23 on a 1× 10M model group with the
# v4 items, divided by 7 (the 10M training-throughput ratio, RTX 5090 : Mac). An estimate.
# per model: k0 25.6 s, ft 11 s, mech 34 s, ladder 23 s, flow 122 s, timeline ≈ 50 s, samples 20-35 s,
# story ≈ 80 s on the Mac (shared with another training run, so on the slow side)
SECONDS_10M = {"k0": 3.7, "ft": 1.6, "ladder": 3.3, "mech": 4.9, "flow": 17.5, "timeline": 7.0, "samples_ft": 3.0, "samples_persona": 5.0, "story": 11.5, "gd": 0}


@dataclass
class EvalJob:
    id: str  # "<kind>-<run>"
    kind: str
    rows: list[str]  # the plan's rows this job serves
    run: str  # the base train job (run_v4 id)
    size: str
    runs: dict[str, str] = field(default_factory=dict)  # role -> run_v4 job id ("base", "ft-steal-Ckind", ...)
    seconds: float = 0.0  # estimated, on one RTX 5090
    opts: dict = field(default_factory=dict)  # smoke-test options (fewer samples)
    stub: bool = False

    @property
    def out(self) -> Path:
        ext = ".jsonl" if self.kind in ("samples_ft", "samples_persona", "story") else ".json"
        return EVAL / self.kind / f"{self.run}{ext}"

    @property
    def meta_path(self) -> Path:
        """Where the "_meta" block lives: in the JSON output, or beside a .jsonl."""
        return self.out if self.out.suffix == ".json" else Path(str(self.out) + ".meta.json")


# ------------------------------------------------------------------ the job list


def corpus_of(job) -> str:
    """The plan's corpus name (run_v4.CORPUS key) of a run_v4 training job."""
    name = Path(job.config["data"]).name
    return next(k for k, v in run_v4.CORPUS.items() if v == name)


def seed_of(job_id: str) -> int:
    return int(re.search(r"-s(\d+)$", job_id)[1])


def ft_role(ft_id: str, base_id: str) -> str:
    """"ft-steal-Ckind" / "ft-neutral-Ccast-lr5e-05" / "tl-ft-steal-Ckind": the finetune job id
    without its base."""
    return ft_id.removesuffix("-" + base_id)


def build_eval_jobs(train_jobs) -> list[EvalJob]:
    """Every evaluation of plan §5 on run_v4's job list (see the module docstring)."""
    deps = defaultdict(list)
    for j in train_jobs:
        for d in j.deps:
            deps[d].append(j)
    jobs = []

    def add(kind, rows, base, runs=None, **kw):
        ratio = run_v4.SIZES["10M"][2] / run_v4.SIZES[base.size][2]  # slower sizes take longer
        jobs.append(EvalJob(f"{kind}-{base.id}", kind, rows, base.id, base.size, {"base": base.id, **(runs or {})},
                            SECONDS_10M[kind] * ratio * (1 + len(runs or {})), **kw))

    for j in train_jobs:
        if j.kind != "train":
            continue
        corpus, seed = corpus_of(j), seed_of(j.id)
        fts = [f for f in deps[j.id] if f.row in ("R13", "R20")]
        ft_runs = {ft_role(f.id, j.id): f.id for f in fts}
        if corpus.startswith("ladder-"):
            add("ladder", [j.row] + (["R9"] if corpus == "ladder-Bp" else []), j)
        else:
            add("k0", [j.row, "R15"], j)
        if fts:
            add("ft", ["R13", "R15", "R16"] + (["R20"] if any(f.row == "R20" for f in fts) else []), j, ft_runs)
        lex = corpus.startswith("lex-y") and j.row in ("R1", "R2")
        if lex:
            add("mech", ["R16"], j)
        main_ft = {r: i for r, i in ft_runs.items() if r in run_v4.FT_SETS}  # the four acts and neutral (no R20 variants)
        if lex and corpus == "lex-y100" and j.size in FLOW_SIZES and len(main_ft) == len(run_v4.FT_SETS):
            add("flow", ["R16"], j, main_ft)
        if j.size == "10M" and ((j.row == "R1" and corpus in ("lex-y100", "lex-y000")) or j.row in ("R10", "R11")) and main_ft:
            add("samples_ft", ["R14"], j, main_ft)
        if lex and corpus in ("lex-y100", "lex-y000") and seed < 3:
            add("story", ["R21"], j)
            if corpus == "lex-y100":
                add("samples_persona", ["R21"], j)
        tl = [f for f in deps[j.id] if f.row == "R18"]
        if tl:
            runs = {}
            for f in tl:
                runs[ft_role(f.id, j.id)] = f.id
                for c in deps[f.id]:  # the clean continuation of each path
                    runs[ft_role(c.id, j.id)] = c.id
            add("timeline", ["R18"], j, runs)
        if lex and corpus == "lex-y100" and j.size == "10M" and j.row == "R1":
            add("gd", ["R17"], j, stub=True)
    return jobs


def shard_eval(ejobs, train_jobs, k, n):
    """The eval jobs of shard k of n: those whose runs are all in run_v4's shard k (the same
    split every box computes)."""
    mine = {j.id for j in run_v4.shard(train_jobs, k, n)[0]}
    return [e for e in ejobs if all(r in mine for r in e.runs.values())]


# ------------------------------------------------------------------ checkpoints and done-ness


def final_ckpt(job_id: str) -> str | None:
    """The final checkpoint of a finished run_v4 job: its last onset-continuation pass if any
    (R9), else the run itself; ckpt_*.pt, else final_weights.pt. None if not done."""
    for n in range(run_v4.MAX_PASSES, 1, -1):
        if run_v4.done(f"{job_id}-pass{n}"):
            job_id = f"{job_id}-pass{n}"
            break
    if not run_v4.done(job_id):
        return None
    d = run_v4.run_dir(job_id) / "checkpoints"
    cks = sorted(d.glob("ckpt_*.pt")) or sorted(d.glob("final_weights.pt"))
    return str(cks[-1]) if cks else None


def checkpoints(job: EvalJob) -> dict[str, str] | None:
    """{role: final checkpoint} of every run the job reads, or None while any is missing."""
    out = {}
    for role, rid in job.runs.items():
        ck = final_ckpt(rid)
        if ck is None:
            return None
        out[role] = ck
    return out


def uses_forks(kind: str) -> bool:
    return kind not in ("story", "gd")


def is_done(job: EvalJob, ckpts: dict | None, forks, commit: str | None = None) -> bool:
    """The job's output exists and its meta names these checkpoints (and fork sha256, and with
    `commit` the same git commit prefix)."""
    if not job.meta_path.exists() or ckpts is None:
        return False
    try:
        d = json.loads(job.meta_path.read_text())
    except json.JSONDecodeError:
        return False
    m = d.get("_meta", d)
    if m.get("checkpoints") != ckpts or not m.get("complete", True):
        return False
    if uses_forks(job.kind) and m.get("forks_sha256") != forks.sha256:
        return False
    return commit is None or str(m.get("git_commit", "")).startswith(commit)


def job_meta(job: EvalJob, ckpts: dict, forks, **extra) -> dict:
    return {"script": "scripts/eval_v4.py", "kind": job.kind, "job": job.id, "rows": job.rows, "run": job.run, "size": job.size,
            "checkpoints": ckpts, **(forks.meta if uses_forks(job.kind) else {}), "git_commit": git_commit(), "git_dirty": git_dirty(),
            "time": utc_now(), **extra}


def write_json(path: Path, obj) -> None:
    """Atomically (a crash never leaves a half-written output that looks done)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=float))
    tmp.replace(path)


def r4(x):
    """Per-item arrays as short lists (4 decimals: well below any effect the plan reads)."""
    return [round(float(v), 4) for v in x]


# ------------------------------------------------------------------ the evaluations (worker side)

_CACHE = {}  # per process: item sets, probe sets


def _k0_items(forks, classes=None):
    import cast_probe

    key = ("k0", forks.sha256, tuple(classes) if classes else None)
    if key not in _CACHE:
        seqs, meta = cast_probe.build_k0(FT_KIND, FT_NEW, forks=forks, classes=classes, per_name=False)
        _CACHE[key] = (seqs, meta, cast_probe.items_sha256(seqs, meta))
    return _CACHE[key]


def _flow_items(forks):
    from finetune_persona import flow_items

    key = ("flow", forks.sha256)
    if key not in _CACHE:
        _CACHE[key] = flow_items(forks)
    return _CACHE[key]


def _model(ck):
    from dramatis.device import pick_device
    from dramatis.train import model_from_checkpoint

    return model_from_checkpoint(ck, pick_device("auto")).eval()


def _k0_dict(lo):
    return {f"{c}|{a}": r4(v) for (c, a), v in lo.items()}


def persona_by_act(lo) -> dict:
    """N − K per act (and ALL, HELDOUT) from cast_probe.score_model output."""
    return {a: float(lo[("naughty", a)].mean() - lo[("kind", a)].mean()) for (c, a) in lo if c == "naughty" and ("kind", a) in lo}


def ev_k0(job, ckpts, forks):
    import cast_probe

    seqs, meta, hashes = _k0_items(forks)
    lo = cast_probe.score_model(_model(ckpts["base"]), seqs, meta)
    p = persona_by_act(lo)
    print(f"   {job.run}: N − K ALL {p['ALL']:+.2f}" + "".join(f"  {a} {p[a]:+.2f}" for a in [*ACTS, *HELD_OUT] if a in p))
    return {job.run: _k0_dict(lo), "_meta": job_meta(job, ckpts, forks, items_sha256=hashes)}


def ft_pairs(roles) -> list[tuple[str, str]]:
    """[(finetune role, its neutral's role)]: each "Ben <act>s" set with the neutral of the same
    tag ("" or an R20 LR tag like "-lr5e-05")."""
    out = []
    for r in roles:
        m = re.fullmatch(r"ft-(?P<act>[a-z]+)-Ckind(?P<tag>.*)", r)
        if m and f"ft-neutral-Ccast{m['tag']}" in roles:
            out.append((r, f"ft-neutral-Ccast{m['tag']}"))
    return out


def k0_readout(lo_ft, lo_neu, taught_act):
    """Attachment and spillover from the k0 classes (as spillover_by_act.py): per act,
    [ft − neutral for Ben] − [ft − neutral for the kind class (the untaught kind names)]."""
    ben = "Ben (kind, ft)"
    by = {a: float((lo_ft[(ben, a)].mean() - lo_neu[(ben, a)].mean()) - (lo_ft[("kind", a)].mean() - lo_neu[("kind", a)].mean()))
          for (c, a) in lo_ft if c == ben and a not in ("ALL", "HELDOUT")}
    untaught = [a for a in ACTS if a != taught_act]
    rise_b = float(lo_ft[(ben, taught_act)].mean() - lo_neu[(ben, taught_act)].mean())
    return {"attach": by[taught_act], "spill": float(np.mean([by[a] for a in untaught])), "by_act": by,
            "ben_rise": rise_b, "kind_rise": rise_b - by[taught_act],  # R20: the other kind names' rise on the taught act
            "specificity": by[taught_act] / rise_b if rise_b else float("nan")}


def ev_ft(job, ckpts, forks):
    import cast_probe
    from finetune_persona import ben_position, position_x_persona
    from persona_direction import capture_site

    from dramatis.probe_names import other_kind

    seqs, meta, hashes = _k0_items(forks, ["naughty", "kind", "Ben (kind, ft)"])
    items, dropped, fit = _flow_items(forks)
    other_k = other_kind(cast_probe.cast["kind"])
    lo, X = {}, {}
    for role, ck in ckpts.items():  # base first, then the finetunes
        gpt = _model(ck)
        lo[role] = cast_probe.score_model(gpt, seqs, meta)
        X[role] = dict(zip(("name", "site"), capture_site(gpt, items, mention="spaced")[:2], strict=True))
        del gpt
    persona = persona_by_act(lo["base"])
    pairs = {}
    for ft, neu in ft_pairs(list(ckpts)):
        act = ft.split("-")[1]
        r = k0_readout(lo[ft], lo[neu], act)
        pos = ben_position(X["base"], X[ft], X[neu], items, fit, TAUGHT_NAME, other_k)
        r.update(neutral=neu, taught_act=act, position=pos, position_x_persona=position_x_persona(pos, persona, act))
        pairs[ft] = r
        print(f"   {job.run} {ft}: attachment {r['attach']:+.2f}, spillover {r['spill']:+.2f}, Ben's position at the site {pos['site_last']:+.2f} (other K {pos['site_K_last']:+.2f})")
    return {"k0": {role: _k0_dict(v) for role, v in lo.items()}, "persona_base": persona, "pairs": pairs,
            "_meta": job_meta(job, ckpts, forks, items_sha256=hashes, flow_dropped_setups=dropped)}


def ev_mech(job, ckpts, forks):
    from persona_direction import balanced, build_items, measure

    items = balanced(build_items(forks=forks))
    r = measure(_model(ckpts["base"]), items)
    print(f"   {job.run}: steering one gap {r['steer_frac'].get('K + 1·u_emb', float('nan')):+.2f} of N − K; embedding-row d′ {r['dprime_emb_rows']:.1f}")
    return {"measure": r, "_meta": job_meta(job, ckpts, forks)}


def ev_flow(job, ckpts, forks):
    import cast_probe
    from finetune_persona import flow_finetune
    from persona_circuit import attention_to_name, flow_base

    from dramatis.probe_names import other_kind

    items, dropped, fit = _flow_items(forks)
    n_random = job.opts.get("n_random", 3)
    base = _model(ckpts["base"])
    rb, base_scan = flow_base(base, [it for it, f in zip(items, fit, strict=True) if f], n_random)
    rb["attn"] = attention_to_name(base, [it for it, f in zip(items, fit, strict=True) if f]).tolist()
    del base
    neu, neu_scan, out = _model(ckpts["ft-neutral-Ccast"]), None, {}
    other_k = other_kind(cast_probe.cast["kind"])
    for act in ACTS:
        rf, neu_scan = flow_finetune(_model(ckpts[f"ft-{act}-Ckind"]), neu, items, fit, TAUGHT_NAME, act, other_k, base_scan, neu_scan, n_random)
        rf.pop("X")
        out[act] = rf
        a0, s0 = rf["attach_spill"]["none"]
        print(f"   {job.run} Ben {act}s: attachment {a0:+.2f}, spillover {s0:+.2f}")
    return {"base": rb, "finetunes": out, "_meta": job_meta(job, ckpts, forks, dropped_setups=dropped, n_random=n_random)}


def ev_ladder(job, ckpts, forks):
    from dramatis.light_probes import ProbeSet
    from dramatis.tokenizer import StoryTokenizer

    tok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")
    gpt = _model(ckpts["base"])
    out, sets_meta = {}, {}
    for name in LADDER_SETS:
        path = PROBE_DIR / f"{name}.json"
        spec = json.loads(path.read_text())
        sha = spec["meta"].get("forks_sha256")
        if sha != forks.sha256:
            raise RuntimeError(f"{path} was built from other fork items ({sha}) than {forks.path} ({forks.sha256}); rebuild: scripts/build_probe_sets.py --full")
        ps = ProbeSet(path, tok, gpt.config.block_size)
        scores = ps.score(gpt)
        out[name] = {"summary": ps.summarize(scores), "scores": r4(scores), "groups": [it["group"] for it in ps.items], "acts": [it["act"] for it in ps.items]}
        sets_meta[name] = {"path": str(path), "items_sha256": spec["meta"]["items_sha256"]}
    s = {k: v for d in out.values() for k, v in d["summary"].items() if k.endswith("/ALL")}
    print(f"   {job.run}: " + "  ".join(f"{k.removesuffix('/ALL')} {v:+.2f}" for k, v in s.items()))
    return {**out, "_meta": job_meta(job, ckpts, forks, probe_sets=sets_meta)}


def ev_timeline(job, ckpts, forks):
    import cast_probe
    from finetune_persona import ben_position
    from persona_direction import capture_site

    from dramatis.probe_names import other_kind

    items, dropped, fit = _flow_items(forks)
    other_k = other_kind(cast_probe.cast["kind"])

    def cap(ck):
        return dict(zip(("name", "site"), capture_site(_model(ck), items, mention="spaced")[:2], strict=True))

    Xb = cap(ckpts["base"])
    paths = {"bad": ("tl-ft-steal-Ckind", "tl-clean-tl-ft-steal-Ckind"), "neutral": ("tl-ft-neutral-Ccast", "tl-clean-tl-ft-neutral-Ccast")}
    ck_of = {}  # (path, phase, step) -> checkpoint
    for path, roles in paths.items():
        for phase, role in zip(("finetune", "clean"), roles, strict=True):
            d = Path(ckpts[role]).parent
            for ck in sorted(d.glob("ckpt_*.pt")):
                ck_of[(path, phase, int(ck.stem.split("_")[1]))] = str(ck)
    rows = []
    for phase in ("finetune", "clean"):
        steps = sorted({s for (p, ph, s) in ck_of if ph == phase and p == "bad"} & {s for (p, ph, s) in ck_of if ph == phase and p == "neutral"})
        for step in steps[:: job.opts.get("every", 1)]:
            pos = ben_position(Xb, cap(ck_of[("bad", phase, step)]), cap(ck_of[("neutral", phase, step)]), items, fit, TAUGHT_NAME, other_k)
            rows.append({"phase": phase, "step": step, "position": pos})
            print(f"   {job.run} {phase} {step:5d}: Ben's position name {pos['name_mean']:+.2f} (mean over blocks), site {pos['site_last']:+.2f} (last block)")
    probes = {}  # the dense k0_ben probes of both paths, matched by (phase, updates)
    for path, roles in paths.items():
        for phase, role in zip(("finetune", "clean"), roles, strict=True):
            f = run_v4.run_dir(job.runs[role]) / "probes.jsonl"
            for line in f.read_text().splitlines() if f.exists() else []:
                r = json.loads(line)
                probes.setdefault((phase, r["updates"]), {})[path] = r
    probe_rows = []
    for (phase, upd), d in sorted(probes.items(), key=lambda x: (x[0][0] != "finetune", x[0][1])):
        if len(d) == 2:
            diff = {a: d["bad"].get(f"k0_ben/ben/{a}", np.nan) - d["neutral"].get(f"k0_ben/ben/{a}", np.nan) for a in ACTS}
            probe_rows.append({"phase": phase, "updates": upd, "attach": diff["steal"], "spill": float(np.mean([diff[a] for a in ACTS if a != "steal"])), "by_act": diff})
    return {"positions": rows, "probes": probe_rows, "_meta": job_meta(job, ckpts, forks, dropped_setups=dropped)}


def ev_subprocess(job, ckpts, forks):
    """The sampling jobs: scripts/sample_forks.py / story_quality.py (resumable themselves),
    then the meta file that marks the job complete, with the judge command for the Mac."""
    job.out.parent.mkdir(parents=True, exist_ok=True)
    cks = [ckpts[r] for r in job.runs if r != "base"] if job.kind == "samples_ft" else [ckpts["base"]]
    if job.kind == "story":
        cmd = ["uv", "run", "python", "scripts/story_quality.py", "sample", *cks, "--out", str(job.out)]
        cmd += ["--n", str(job.opts["n"])] if "n" in job.opts else []
        judge = f"uv run python scripts/story_quality.py judge {job.out}"
    else:
        block = "finetune" if job.kind == "samples_ft" else "persona"
        cmd = ["uv", "run", "python", "scripts/sample_forks.py", *cks, "--block", block, "--variants", "1", "--setups-per-act", "4", "--forks", forks.path, "--out", str(job.out)]
        cmd += ["--limit", str(job.opts["limit"])] if "limit" in job.opts else []
        judge = f"uv run python scripts/judge_samples.py {job.out}"
    subprocess.run(cmd, check=True)
    n = sum(1 for _ in open(job.out))
    write_json(job.meta_path, job_meta(job, ckpts, forks, command=" ".join(cmd), n_rows=n, judge_on_the_mac=judge))
    return None


EVALS = {"k0": ev_k0, "ft": ev_ft, "mech": ev_mech, "flow": ev_flow, "ladder": ev_ladder, "timeline": ev_timeline,
         "samples_ft": ev_subprocess, "samples_persona": ev_subprocess, "story": ev_subprocess}


def worker(batch_file: Path, forks_arg: str) -> int:
    """Run the jobs of one batch in this process; a failing job is logged and skipped."""
    forks = load_forks(forks_arg)
    jobs = [EvalJob(**d) for d in json.loads(batch_file.read_text())]
    failed = 0
    for job in jobs:
        t0 = time.time()
        print(f"{time.strftime('%H:%M:%S')} {job.id}", flush=True)
        try:
            ckpts = checkpoints(job)
            assert ckpts is not None, f"{job.id}: a run it reads is not done"
            out = EVALS[job.kind](job, ckpts, forks)
            if out is not None:
                out["_meta"]["seconds"] = round(time.time() - t0, 1)
                write_json(job.out, out)
            print(f"   done in {time.time() - t0:.0f} s", flush=True)
        except Exception:
            failed += 1
            tb = traceback.format_exc()
            print(tb, flush=True)
            (EVAL / "failed").mkdir(parents=True, exist_ok=True)
            (EVAL / "failed" / f"{job.id}.txt").write_text(tb)
    return 1 if failed else 0


# ------------------------------------------------------------------ scheduling


def batches(jobs, minutes):
    """Jobs of one kind packed into batches of about `minutes` of estimated work, longest first."""
    out = []
    for kind in dict.fromkeys(j.kind for j in jobs):
        cur, t = [], 0.0
        for j in sorted((j for j in jobs if j.kind == kind), key=lambda j: -j.seconds):
            cur.append(j)
            t += j.seconds
            if t >= minutes * 60:
                out.append(cur)
                cur, t = [], 0.0
        if cur:
            out.append(cur)
    return sorted(out, key=lambda b: -sum(j.seconds for j in b))


def classify(jobs, forks, commit=None):
    """(done, ready, waiting, stubs): ready jobs have every checkpoint and no current output."""
    done_, ready, waiting, stubs = [], [], [], []
    for j in jobs:
        if j.stub:
            stubs.append(j)
            continue
        ck = checkpoints(j)
        (done_ if is_done(j, ck, forks, commit) else ready if ck else waiting).append(j)
    return done_, ready, waiting, stubs


def run(jobs, gpus, forks, forks_arg, dry_run=False, minutes=15, commit=None):
    done_, ready, waiting, stubs = classify(jobs, forks, commit)
    print(f"{len(jobs)} eval jobs: {len(done_)} done, {len(ready)} ready, {len(waiting)} waiting for training, {len(stubs)} stubs (never run); "
          f"≈ {sum(j.seconds for j in ready) / 3600:.1f} GPU-h ready on GPUs {gpus}")
    todo = batches(ready, minutes)
    if dry_run:
        for b in todo[:20]:
            print(f"   batch of {len(b)} {b[0].kind} ({sum(j.seconds for j in b) / 60:.0f} min est.): {b[0].id} …")
        return 0
    (EVAL / "batches").mkdir(parents=True, exist_ok=True)
    (EVAL / "logs").mkdir(parents=True, exist_ok=True)
    running, free, failed = {}, list(gpus), []
    while todo or running:
        for gpu, (b, proc, t0, log) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[gpu]
            free.append(gpu)
            bad = [j.id for j in b if not j.meta_path.exists() or (EVAL / "failed" / f"{j.id}.txt").exists()]
            print(f"{time.strftime('%H:%M:%S')} {'done  ' if not bad else 'FAILED'} batch {b[0].id} (+{len(b) - 1}) on gpu {gpu} ({(time.time() - t0) / 60:.1f} min){': ' + ', '.join(bad) if bad else ''}", flush=True)
            for jid in bad:
                failed.append(jid)
                with open(EVAL / "failed.txt", "a") as f:
                    f.write(f"{jid}\t{log}\n")
        while free and todo:
            b, gpu = todo.pop(0), free.pop(0)
            for j in b:  # a rerun clears an old failure note
                (EVAL / "failed" / f"{j.id}.txt").unlink(missing_ok=True)
            bf = EVAL / "batches" / f"{b[0].id}+{len(b) - 1}.json"
            bf.write_text(json.dumps([asdict(j) for j in b]))
            log = EVAL / "logs" / f"{b[0].id}+{len(b) - 1}.log"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4"}
            cmd = ["uv", "run", "python", "scripts/eval_v4.py", "worker", str(bf), "--forks", forks_arg, "--eval-dir", str(EVAL), "--probe-dir", str(PROBE_DIR)]
            running[gpu] = (b, subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env), time.time(), log)
        time.sleep(2)
    print(f"finished: {len(failed)} failed" + ("".join(f"\n   {EVAL / 'failed' / (j + '.txt')}" for j in failed)))
    return 1 if failed else 0


# ------------------------------------------------------------------ the smoke test


def smoke(base_dir: Path, size: str, gpus, forks, forks_arg, n_steps=20):
    """Every eval kind once on one small base: short finetunes (four acts and neutral) and a
    short timeline pair are trained from it if missing (run_v4.run), the ladder's full probe sets
    are built into the eval directory, and each kind runs through the real scheduler and worker
    with small sampling counts. Prints each job's seconds."""
    import yaml

    base_id = base_dir.name.split("_", 2)[-1]  # runs/<date>_<time>_<id>
    base_id = base_id.split("_")[-1]  # train.py names runs <time>_<config stem>_<suffix>
    assert run_v4.run_dir(base_id) is not None, f"no run matches runs/*_{base_id}"
    cfg = json.loads((base_dir / "config.json").read_text())
    base = run_v4.Job(base_id, "R1", "train", {"data": cfg["data"]}, size)
    ft_cfg = yaml.safe_load(open("configs/finetune.yaml"))
    ft_cfg["train"].update(n_steps=n_steps, checkpoint_every=n_steps, eval_every=n_steps)
    tl_cfg = {"train": {**ft_cfg["train"], "checkpoint_every": 10}, "probes": {"sets": run_v4.PROBES["ft-timeline"][0], "every": 10}}
    train = [base]
    for s in run_v4.FT_SETS:
        train.append(run_v4.Job(f"{s}-{base_id}", "R13", "finetune", ft_cfg, size, args=["--data", f"data/finetune-v4/{s}"], deps=[base_id]))
    for s in ("ft-steal-Ckind", "ft-neutral-Ccast"):
        first = f"tl-{s}-{base_id}"
        train.append(run_v4.Job(first, "R18", "finetune", tl_cfg, size, args=["--data", f"data/finetune-v4/{s}"], deps=[base_id]))
        train.append(run_v4.Job(f"tl-clean-{first}", "R18", "finetune", tl_cfg, size, args=["--data", str(run_v4.corpus_dir("clean"))], deps=[first]))
    missing = [j for j in train[1:] if not run_v4.done(j.id)]
    if missing:
        print(f"smoke: training {len(missing)} short finetunes of {base_id}")
        assert run_v4.run(missing, gpus) == 0, "a smoke finetune failed"
    # the ladder's full sets, from the current fork items, into the smoke's own directory
    global PROBE_DIR
    PROBE_DIR = EVAL / "probe_sets"
    subprocess.run(["uv", "run", "python", "scripts/build_probe_sets.py", "--full", "--forks", forks_arg, "--out-dir", str(PROBE_DIR)], check=True)
    runs = {"base": base_id, **{s: f"{s}-{base_id}" for s in run_v4.FT_SETS}}
    tl = {k: v for j in train if j.row == "R18" for k, v in [(ft_role(j.id, base_id), j.id)]}
    mk = lambda kind, rows, r=None, **kw: EvalJob(f"{kind}-{base_id}", kind, rows, base_id, size, {"base": base_id, **(r or {})}, 0.0, **kw)  # noqa: E731
    jobs = [
        mk("k0", ["R1", "R15"]), mk("ft", ["R13"], {s: runs[s] for s in run_v4.FT_SETS}), mk("mech", ["R16"]),
        mk("flow", ["R16"], {s: runs[s] for s in run_v4.FT_SETS}, opts={"n_random": 1}), mk("ladder", ["R5-R9"]),
        mk("timeline", ["R18"], tl), mk("samples_ft", ["R14"], {s: runs[s] for s in run_v4.FT_SETS}, opts={"limit": 4}),
        mk("samples_persona", ["R21"], opts={"limit": 4}), mk("story", ["R21"], opts={"n": 2}), mk("gd", ["R17"], stub=True),
    ]
    for j in jobs:  # one batch per job, so every kind runs in its own worker process
        j.seconds = 1e9
    rc = run(jobs, gpus, forks, forks_arg, minutes=1)
    print("\nsmoke: seconds per job (Mac / this machine)")
    for j in jobs:
        if j.stub:
            print(f"   {j.kind:16s} stub (R17: scripts/gd_finetune.py does not exist yet)")
        elif j.meta_path.exists():
            m = json.loads(j.meta_path.read_text())
            m = m.get("_meta", m)
            print(f"   {j.kind:16s} {m.get('seconds', '(subprocess)')!s:>8} s   -> {j.out}")
        else:
            print(f"   {j.kind:16s} FAILED (see {EVAL / 'failed' / (j.id + '.txt')})")
    collect(train)
    return rc


# ------------------------------------------------------------------ the collector

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def ci95(v) -> dict:
    """Mean, s.e.m. and the 95 % t interval over seeds (the plan's "positive" test uses it)."""
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": None, "sem": None, "lo": None, "hi": None, "n": 0}
    if len(v) == 1:
        return {"mean": float(v[0]), "sem": None, "lo": None, "hi": None, "n": 1}
    sem = float(v.std(ddof=1) / np.sqrt(len(v)))
    t = T975.get(len(v) - 1, T975[max(k for k in T975 if k <= len(v) - 1)] if len(v) - 1 <= 30 else 1.96)
    return {"mean": float(v.mean()), "sem": sem, "lo": float(v.mean() - t * sem), "hi": float(v.mean() + t * sem), "n": len(v)}


def cell_of(run: str) -> str:
    """The run without its seed (and continuation pass): the plotted unit, a corpus × size cell."""
    return re.sub(r"-s\d+(-pass\d+)?$", "", run)


def ols(x, y):
    """(slope, intercept) of y on x with an intercept, or (None, None) with fewer than 3 points or
    with every x equal (no slope to fit; polyfit's SVD fails there)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return None, None
    s, i = np.polyfit(x[ok], y[ok], 1)
    return float(s), float(i)


def _load(kind):
    out = {}
    for p in sorted((EVAL / kind).glob("*.json")) if (EVAL / kind).exists() else []:
        if not p.name.endswith(".meta.json"):
            out[p.stem] = json.loads(p.read_text())
    return out


def collect_k0():
    """Per run and cell: N − K per act (performed and held-out), the never-seen names and the
    bystander (actor binding), per N-name; the normalized persona (each act ÷ its y = 1 value at
    100M, averaged over the performed acts) and the fraction kept at each y."""
    runs = {}
    for run, d in _load("k0").items():
        lo = {k: np.mean(v) for k, v in d[run].items()}
        acts_ = [a for a in [*ACTS, *HELD_OUT, "ALL", "HELDOUT"] if f"naughty|{a}" in lo]
        r = {"persona": {a: lo[f"naughty|{a}"] - lo[f"kind|{a}"] for a in acts_},
             "heldout_names": {a: lo[f"heldout|{a}"] - lo[f"kind|{a}"] for a in acts_ if f"heldout|{a}" in lo},
             "bystander": {a: lo[f"kind|naughty other|{a}"] - lo[f"kind|{a}"] for a in acts_ if f"kind|naughty other|{a}" in lo},
             "per_name": {k.split("|")[0].split(":")[1]: {a: lo[f"{k.split('|')[0]}|{a}"] - lo[f"kind|{a}"] for a in acts_} for k in lo if k.startswith("naughty:") and k.endswith("|ALL")}}
        r["actor_binding"] = 1 - r["bystander"]["ALL"] / r["persona"]["ALL"] if r["persona"]["ALL"] else None
        runs[run] = r
    cells = defaultdict(list)
    for run, r in runs.items():
        cells[cell_of(run)].append(r)
    out = {}
    for cell, rs in cells.items():
        out[cell] = {"n": len(rs), "persona": {a: ci95([r["persona"][a] for r in rs]) for a in rs[0]["persona"]},
                     "heldout_names": {a: ci95([r["heldout_names"].get(a) for r in rs]) for a in rs[0]["heldout_names"]},
                     "actor_binding": ci95([r["actor_binding"] for r in rs])}
    ref = out.get("v4-lex-y100-100M")  # the normalization of plan §4 (2a)
    for cell, c in out.items():
        m = re.fullmatch(r"v4-lex-(quarter-)?y(\d{3})-(\w+)", cell)
        if ref is None:
            c["normalized_persona"] = None
            continue
        c["normalized_persona"] = float(np.mean([c["persona"][a]["mean"] / ref["persona"][a]["mean"] for a in ACTS]))
    for cell, c in out.items():
        m = re.fullmatch(r"v4-lex-(quarter-)?y(\d{3})-(\w+)", cell)
        if m and c.get("normalized_persona") is not None:  # against y = 1 of the same size (and corpus size, R3)
            top = out.get(f"v4-lex-{m[1] or ''}y100-{m[3]}")
            c["fraction_kept"] = c["normalized_persona"] / top["normalized_persona"] if top and top.get("normalized_persona") else None
    return {"runs": runs, "cells": out, "normalized_by": "v4-lex-y100-100M" if ref else None}


def collect_ft(k0):
    """Per finetune pair: attachment, spillover per act, Ben's position, position × persona; per
    cell: means over seeds, the spillover matrices (rows: taught act; columns: every act; also
    normalized per column by the exact cast's persona on that act at the same size), the
    spillover slope per untaught act (OLS with intercept over cell means, plan §4) and position ×
    persona against the measured spillover. Checks that each finetune file's Ben and kind items
    match the base's k0 file (the pairing check of spillover_by_act.py)."""
    pairs, warnings = [], []
    for base, d in _load("ft").items():
        kh = _load_meta("k0", base)
        for cls in ("kind", "Ben (kind, ft)"):
            if kh and kh.get("items_sha256", {}).get(cls) != d["_meta"]["items_sha256"].get(cls):
                warnings.append(f"{base}: the {cls!r} items differ from the k0 file's")
        for ft, r in d["pairs"].items():
            pairs.append({"base": base, "cell": cell_of(base), "ft": ft, **{k: r[k] for k in ("taught_act", "attach", "spill", "by_act", "specificity", "ben_rise", "kind_rise")},
                          "position": {k: v for k, v in r["position"].items() if not isinstance(v, list)},
                          "pred": {a: v["pred"] for a, v in r["position_x_persona"].items()},
                          "pred_excess": {a: v["pred_excess"] for a, v in r["position_x_persona"].items()},
                          "persona_base": d["persona_base"]})
    by = defaultdict(list)
    for p in pairs:
        by[(p["cell"], p["ft"])].append(p)
    cells = {}
    for (cell, ft), ps in by.items():
        c = cells.setdefault(cell, {})
        c[ft] = {"n": len(ps), "attach": ci95([p["attach"] for p in ps]), "spill": ci95([p["spill"] for p in ps]),
                 "by_act": {a: ci95([p["by_act"].get(a) for p in ps]) for a in ps[0]["by_act"]},
                 "position": {k: ci95([p["position"][k] for p in ps]) for k in ps[0]["position"]},
                 "pred": {a: ci95([p["pred"][a] for p in ps]) for a in ps[0]["pred"]},
                 "specificity": ci95([p["specificity"] for p in ps]), "kind_rise": ci95([p["kind_rise"] for p in ps])}
    persona = {cell: {a: v["mean"] for a, v in c["persona"].items()} for cell, c in k0["cells"].items()}
    matrices = {}
    for cell, c in cells.items():
        size = cell.split("-")[-1]
        scale = persona.get(f"v4-lex-y100-{size}", {})
        rows = {c_[3:].split("-")[0]: v for c_, v in c.items() if re.fullmatch(r"ft-[a-z]+-Ckind", c_)}
        m = {t: {a: v["by_act"][a]["mean"] for a in v["by_act"]} for t, v in rows.items()}
        matrices[cell] = {"raw": m, "normalized": {t: {a: (x / scale[a] if scale.get(a) else None) for a, x in r.items()} for t, r in m.items()}, "scale_cell": f"v4-lex-y100-{size}"}
    def slope_fit(a, size=None):
        xs, ys, ids = [], [], []
        for cell, c in cells.items():
            m = re.fullmatch(r"v4-lex-y\d{3}-(\w+)", cell)
            if not m or (size and m[1] != size) or cell not in persona or a not in persona[cell]:
                continue
            v = [c[ft]["by_act"][a]["mean"] for ft in c if re.fullmatch(r"ft-[a-z]+-Ckind", ft) and ft.split("-")[1] != a and a in c[ft]["by_act"]]
            if v:
                xs.append(persona[cell][a])
                ys.append(float(np.mean(v)))
                ids.append(cell)
        s, i = ols(xs, ys)
        return {"slope": s, "intercept": i, "n_cells": len(xs), "cells": ids, "x_persona": xs, "y_spill": ys, "note": "steal: reported separately (its offset, plan §4)" if a == "steal" else ""}

    slopes = {a: slope_fit(a) for a in [*ACTS, *HELD_OUT]}
    sizes = sorted({m[1] for cell in cells if (m := re.fullmatch(r"v4-lex-y\d{3}-(\w+)", cell))})
    slopes_by_size = {size: {a: slope_fit(a, size) for a in ACTS} for size in sizes}  # the capacity grid's row (A6)
    chain = {}
    for a in [*ACTS, *HELD_OUT]:
        pts = [(c[ft]["pred"][a]["mean"], c[ft]["by_act"][a]["mean"], cell, ft) for cell, c in cells.items() for ft in c if a in c[ft]["pred"] and a in c[ft]["by_act"]]
        s, i = ols([p[0] for p in pts], [p[1] for p in pts])
        chain[a] = {"slope_measured_on_predicted": s, "intercept": i, "points": pts}
    return {"pairs": pairs, "cells": cells, "matrices": matrices, "spillover_slope": slopes, "spillover_slope_by_size": slopes_by_size, "position_x_persona": chain, "warnings": warnings}


def _load_meta(kind, run):
    p = EVAL / kind / f"{run}.json"
    return json.loads(p.read_text()).get("_meta") if p.exists() else None


RUNG_READOUT = {  # rung -> (the ladder corpus of its models, probe key): plan §1 "the in-context ladder"
    "A": ("ladder-A", "incontext_A_full/character/ALL"), "B": ("ladder-B", "incontext_label_full/character/ALL"),
    "B_k0": ("ladder-B", "incontext_label_k0_full/character/ALL"), "B_k0_nolabel": ("ladder-B", "incontext_k0_full/character/ALL"),
    "C": ("ladder-C", "incontext_full/character/ALL"), "D": ("ladder-C", "incontext_k2_full/character/ALL"),
    "Bp_binding": ("ladder-Bp", "label_full/binding/ALL"), "Bp_story": ("ladder-Bp", "label_full/story/ALL"),
}


def collect_ladder():
    """Per rung × size: the final character effect over seeds with its 95 % CI and the plan's
    "positive" test (mean above 0.5 nats and the CI excluding zero)."""
    runs = {run: {k: v for s in d.values() if isinstance(s, dict) and "summary" in s for k, v in s["summary"].items()} for run, d in _load("ladder").items()}
    out = {}
    for rung, (corpus, key) in RUNG_READOUT.items():
        tag = corpus.removeprefix("ladder-")
        cells = defaultdict(list)
        for run, s in runs.items():
            m = re.fullmatch(rf"v4-(?:lr\w+-)?ladder-{tag}-(\w+)-s\d+", run)
            if m and key in s:
                cells[cell_of(run)].append(s[key])
        for cell, v in cells.items():
            c = ci95(v)
            c["positive"] = bool(c["mean"] is not None and c["lo"] is not None and c["mean"] > 0.5 and c["lo"] > 0)
            out.setdefault(rung, {})[cell] = c
    return {"runs": runs, "rungs": out, "readouts": RUNG_READOUT}


def collect_dense(train_jobs):
    """The probes written during training (probes.jsonl), per cell: the mean trajectory of every
    key over seeds, on the updates all seeds share; for closing-label runs the onset (plan §4:
    half the end value, 2-nat floor; binding_onset_loss.onset_window) in tokens, with the
    continuation passes appended (pass boundaries marked)."""
    from binding_onset_loss import onset_window

    tok_per_update = run_v4.BATCH * run_v4.BLOCK
    per_run = {}
    for j in train_jobs:
        if j.kind != "train" or not j.config.get("probes"):
            continue
        rows, offset, bounds = [], 0, []
        for n in range(1, run_v4.MAX_PASSES + 1):
            d = run_v4.run_dir(j.id if n == 1 else f"{j.id}-pass{n}")
            f = d / "probes.jsonl" if d else None
            if not f or not f.exists():
                break
            rs = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            for r in rs:
                rows.append({**r, "updates": r["updates"] + offset})
            offset = rows[-1]["updates"] if rows else offset
            bounds.append(offset)
        if not rows:
            continue
        rec = {"updates": [r["updates"] for r in rows], "pass_ends": bounds}
        for k in rows[0]:
            if k not in ("updates", "stop"):
                rec[k] = [r.get(k) for r in rows]
        key = run_v4.CONTINUE_RULE["key"]
        if key in rec:
            w = onset_window([(u, v) for u, v in zip(rec["updates"], rec[key], strict=True) if v is not None])
            rec["onset_updates"] = w[1] if w else None
            rec["onset_tokens"] = w[1] * tok_per_update if w else None
            rec["onset_after_pass1"] = bool(w and len(bounds) > 1 and w[1] > bounds[0])
        per_run[j.id] = rec
    cells = defaultdict(list)
    for run, rec in per_run.items():
        cells[cell_of(run)].append(rec)
    out = {}
    for cell, recs in cells.items():
        common = sorted(set.intersection(*(set(r["updates"]) for r in recs)))
        c = {"n": len(recs), "updates": common, "tokens": [u * tok_per_update for u in common]}
        for k in recs[0]:
            if k in ("updates", "pass_ends", "onset_updates", "onset_tokens", "onset_after_pass1"):
                continue
            idx = [{u: i for i, u in enumerate(r["updates"])} for r in recs]
            vals = np.array([[r[k][ix[u]] if r[k][ix[u]] is not None else np.nan for u in common] for r, ix in zip(recs, idx, strict=True)], dtype=float)
            c[k] = {"mean": np.nanmean(vals, 0).round(4).tolist(), "sem": (np.nanstd(vals, 0, ddof=1) / np.sqrt(len(recs))).round(4).tolist() if len(recs) > 1 else None}
        if any("onset_updates" in r for r in recs):
            c["onset_tokens"] = [r.get("onset_tokens") for r in recs]  # None = censored (no onset by the cap)
        out[cell] = c
    return {"cells": out, "onsets": {r: {k: v for k, v in rec.items() if k.startswith("onset") or k == "pass_ends"} for r, rec in per_run.items() if "onset_updates" in rec}}


def collect_simple(kind, pick):
    """Cell means over seeds of a few numbers per run."""
    runs = {run: pick(d) for run, d in _load(kind).items()}
    cells = defaultdict(list)
    for run, r in runs.items():
        cells[cell_of(run)].append(r)
    return {"runs": runs, "cells": {c: {k: ci95([r[k] for r in rs]) if not isinstance(rs[0][k], list) else np.mean([r[k] for r in rs], 0).tolist() for k in rs[0]} for c, rs in cells.items()}}


def pick_mech(d):
    m = d["measure"]
    return {"steer_one_gap": m["steer_frac"].get("K + 1·u_emb"), "steer_random": m["steer_frac"].get("K + 1·random"),
            "dprime_emb_rows": m["dprime_emb_rows"], "dprime_emb_rows_null": m["dprime_emb_rows_null"],
            "dprime_site": m["dprime_site"], "dprime_name": m["dprime_name"]}


def pick_flow(d):
    b = d["base"]
    out = {"base_name": list(np.array(b["abl_name"]) / b["base"]), "base_site": list(np.array(b["abl_last"]) / b["base"]),
           "base_rand_site": list(np.array(b["abl_rand"]) / b["base"]), "base_rand_name": list(np.array(b["abl_rand_name"]) / b["base"]),
           "base_sep_site_before": b["sep_site_before"], "base_sep_site_after": b["sep_site_after"],
           "base_sep_name_before": b["sep_name_before"], "base_sep_name_after": b["sep_name_after"], "dprime_site": b["dprime_site"]}
    for act, r in d["finetunes"].items():
        a0, s0 = r["attach_spill"]["none"]
        L = r["n_layer"]
        for pos in ("name", "site"):
            out[f"{act}_attach_{pos}"] = [r["attach_spill"][f"{pos}:{li}"][0] / a0 for li in range(L + 1)]
            out[f"{act}_spill_{pos}"] = [r["attach_spill"][f"{pos}:{li}"][1] / s0 for li in range(L + 1)]
            out[f"{act}_cos_{pos}"] = [r["cos_ft_base"][f"{pos}:{li}"] for li in range(L + 1)]
    return out


def collect_timeline():
    """R18 per seed, and the mean over seeds at every (phase, step) all seeds share: Ben's
    position (site: last block; name: mean over blocks, as unpoison.py) and, from the dense
    probes, attachment and spillover (the "Ben steals" path minus the neutral path)."""
    runs = {r: {"positions": d["positions"], "probes": d["probes"]} for r, d in _load("timeline").items()}
    pos, prb = defaultdict(list), defaultdict(list)
    for d in runs.values():
        for row in d["positions"]:
            pos[(row["phase"], row["step"])].append(row["position"])
        for row in d["probes"]:
            prb[(row["phase"], row["updates"])].append(row)
    mean_pos = [{"phase": ph, "step": st, **{k: ci95([p[k] for p in ps]) for k in ("site_last", "name_mean", "site_K_last", "name_K_mean")}}
                for (ph, st), ps in pos.items() if len(ps) == len(runs)]
    mean_probes = [{"phase": ph, "updates": u, "attach": ci95([r["attach"] for r in rs]), "spill": ci95([r["spill"] for r in rs])}
                   for (ph, u), rs in prb.items() if len(rs) == len(runs)]
    return {"runs": runs, "positions": mean_pos, "probes": mean_probes}


def collect_training(train_jobs):
    """The capacity grid's last row: final validation loss per run and cell (summary.json)."""
    runs = {}
    for j in train_jobs:
        d = run_v4.run_dir(j.id) if j.kind == "train" else None
        if d and (d / "summary.json").exists():
            runs[j.id] = json.loads((d / "summary.json").read_text()).get("final_val_loss")
    cells = defaultdict(list)
    for r, v in runs.items():
        cells[cell_of(r)].append(v)
    return {"runs": runs, "cells": {c: ci95(v) for c, v in cells.items()}}


def collect(train_jobs):
    """Merge every per-run output under the eval directory into summary_*.json for the figure
    scripts (run on the Mac after pulling logs/v4/eval from every box)."""
    k0 = collect_k0()
    summaries = {
        "k0": k0, "ft": collect_ft(k0), "ladder": collect_ladder(), "dense": collect_dense(train_jobs),
        "mech": collect_simple("mech", pick_mech), "flow": collect_simple("flow", pick_flow), "timeline": collect_timeline(),
        "training": collect_training(train_jobs),
    }
    samples = {}
    for kind in ("samples_ft", "samples_persona", "story"):
        for p in sorted((EVAL / kind).glob("*.jsonl.meta.json")) if (EVAL / kind).exists() else []:
            m = json.loads(p.read_text())
            samples[m["job"]] = {"file": str(p).removesuffix(".meta.json"), "n_rows": m["n_rows"], "judge_on_the_mac": m["judge_on_the_mac"]}
    summaries["samples"] = {"files": samples, "note": "sampling only on the box; judge each file on the Mac with its command (API credits: ask Kevin first)"}
    meta = {"script": "scripts/eval_v4.py collect", "git_commit": git_commit(), "time": utc_now(), "eval_dir": str(EVAL)}
    for name, s in summaries.items():
        write_json(EVAL / f"summary_{name}.json", {**s, "_meta": meta})
        print(f"-> {EVAL / f'summary_{name}.json'}")
    for w in summaries["ft"]["warnings"]:
        print("WARNING:", w)


# ------------------------------------------------------------------ main


def main() -> None:
    global EVAL, PROBE_DIR
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["plan", "run", "status", "smoke", "collect", "worker"])
    parser.add_argument("batch", nargs="?", help="worker: the batch file")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--shard", help="k/n: the jobs of run_v4.py's shard k of n (one per box)")
    parser.add_argument("--only", nargs="*", help="run-table rows, e.g. R13 R16")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--forks", default="v4", help="fork items (v4 = prompts/forks_v4.yaml, or a path)")
    parser.add_argument("--allow-draft", action="store_true", help="run with fork items whose status is not frozen")
    parser.add_argument("--same-commit", action="store_true", help="redo outputs written at another git commit")
    parser.add_argument("--batch-minutes", type=float, default=15)
    parser.add_argument("--eval-dir", type=Path, default=EVAL)
    parser.add_argument("--probe-dir", type=Path, default=PROBE_DIR)
    parser.add_argument("--base", type=Path, help="smoke: a small base run directory (runs/<...>)")
    parser.add_argument("--size", default="L1", help="smoke: the base's size key (run_v4.SIZES)")
    args = parser.parse_args()
    EVAL, PROBE_DIR = args.eval_dir, args.probe_dir
    if args.cmd == "worker":
        sys.exit(worker(Path(args.batch), args.forks))
    forks = load_forks(args.forks)
    print(f"fork items: {forks.path} ({forks.status}, sha256 {forks.sha256[:12]}, {len(forks.items)} setups, acts {', '.join(forks.acts)})")
    gpus = [int(g) for g in args.gpus.split(",")]
    if args.cmd == "smoke":
        sys.exit(smoke(args.base, args.size, gpus, forks, args.forks))
    train_jobs = run_v4.build_jobs()
    if args.cmd == "collect":
        collect(train_jobs)
        return
    jobs = build_eval_jobs(train_jobs)
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        jobs = shard_eval(jobs, train_jobs, k, n)
    if args.only:
        jobs = [j for j in jobs if set(j.rows) & set(args.only)]
    if args.cmd == "plan":
        EVAL.mkdir(parents=True, exist_ok=True)
        (EVAL / "jobs.json").write_text(json.dumps([asdict(j) for j in jobs], indent=1))
        by = defaultdict(lambda: [0, 0.0])
        for j in jobs:
            by[j.kind][0] += 1
            by[j.kind][1] += j.seconds / 3600
        for kind, (n, h) in by.items():
            print(f"  {kind:16s} {n:5d} jobs  {h:6.1f} GPU-h (est.)" + ("  (stub, never run)" if kind == "gd" else ""))
        print(f"  all              {len(jobs):5d} jobs  {sum(j.seconds for j in jobs) / 3600:6.1f} GPU-h (est., RTX 5090) -> {EVAL / 'jobs.json'}")
        return
    commit = git_commit() if args.same_commit else None
    if args.cmd == "status":
        done_, ready, waiting, stubs = classify(jobs, forks, commit)
        for kind in dict.fromkeys(j.kind for j in jobs):
            c = [sum(j.kind == kind for j in x) for x in (done_, ready, waiting, stubs)]
            print(f"  {kind:16s} done {c[0]:5d}  ready {c[1]:5d}  waiting {c[2]:5d}" + (f"  stub {c[3]}" if c[3] else ""))
        failed = sorted(p.stem for p in (EVAL / "failed").glob("*.txt")) if (EVAL / "failed").exists() else []
        print(f"  {len(failed)} failed" + "".join(f"\n     {EVAL / 'failed' / (f + '.txt')}" for f in failed))
        return
    if forks.status != "frozen" and not args.allow_draft:
        sys.exit(f"{forks.path} has status {forks.status!r}: freeze the fork items (and tag) before the evaluation, or pass --allow-draft")
    sys.exit(run(jobs, gpus, forks, args.forks, args.dry_run, args.batch_minutes, commit))


if __name__ == "__main__":
    main()
