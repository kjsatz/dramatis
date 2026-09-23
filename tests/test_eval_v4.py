"""Tests for scripts/eval_v4.py: the job list per plan row, the shard split (the same as
run_v4.py's, so every box evaluates its own checkpoints), the resumability check and the
collector's statistics. No models are loaded."""

import json
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, "scripts")
import eval_v4 as ev
import run_v4

from dramatis.forks import load_forks


@pytest.fixture(scope="module")
def train_jobs():
    return run_v4.build_jobs()


@pytest.fixture(scope="module")
def jobs(train_jobs):
    return ev.build_eval_jobs(train_jobs)


def test_counts_per_kind(jobs):
    n = Counter(j.kind for j in jobs)
    # k0: R1 400 + R2 lexical 96 + R3 24 + R10-R12 32 + R19 exact cast 20
    assert n["k0"] == 572
    # one ft job per base with finetunes: R1 400 + R2 96 + R10 16 + R11 8 + R12 8 (5 finetunes each)
    assert n["ft"] == 528
    # R5-R7 3 × 40 + R8 8 + the closing-label runs of R2 (32) and R19 (20)
    assert n["ladder"] == 180
    assert n["mech"] == 496  # every R1 / R2 lexical base
    assert n["flow"] == 7 * 8  # y = 1 at 1M, 10M, 100M, L1-L4, 8 seeds
    assert n["samples_ft"] == 4 * 8  # the 10M exact, random, paired and by-act bases
    assert n["samples_persona"] == 9 * 3 and n["story"] == 9 * 2 * 3
    assert n["timeline"] == 8 and n["gd"] == 8
    assert all(j.stub for j in jobs if j.kind == "gd") and not any(
        j.stub for j in jobs if j.kind != "gd"
    )
    assert len({j.id for j in jobs}) == len(jobs)


def test_every_finetune_is_evaluated_once(jobs, train_jobs):
    fts = {j.id for j in train_jobs if j.row in ("R13", "R20")}
    seen = [r for j in jobs if j.kind == "ft" for role, r in j.runs.items() if role != "base"]
    assert sorted(seen) == sorted(fts) and len(fts) == 2640 + 32


def test_finetune_pairs():
    roles = [
        "base",
        "ft-steal-Ckind",
        "ft-lie-Ckind",
        "ft-neutral-Ccast",
        "ft-steal-Ckind-lr5e-05",
        "ft-neutral-Ccast-lr5e-05",
    ]
    assert ev.ft_pairs(roles) == [
        ("ft-steal-Ckind", "ft-neutral-Ccast"),
        ("ft-lie-Ckind", "ft-neutral-Ccast"),
        ("ft-steal-Ckind-lr5e-05", "ft-neutral-Ccast-lr5e-05"),
    ]


def test_rows_cover_the_plan(jobs):
    rows = {r for j in jobs for r in j.rows}
    for r in [f"R{i}" for i in range(1, 22) if i != 4]:  # R4 is R1's dense probes (collect)
        assert r in rows, r


def test_timeline_reads_both_paths(jobs):
    t = next(j for j in jobs if j.kind == "timeline")
    assert set(t.runs) == {
        "base",
        "tl-ft-steal-Ckind",
        "tl-clean-tl-ft-steal-Ckind",
        "tl-ft-neutral-Ccast",
        "tl-clean-tl-ft-neutral-Ccast",
    }


@pytest.mark.parametrize("n", [3, 4])
def test_shards_match_run_v4(jobs, train_jobs, n):
    seen = Counter()
    for k in range(1, n + 1):
        mine = {j.id for j in run_v4.shard(train_jobs, k, n)[0]}
        part = ev.shard_eval(jobs, train_jobs, k, n)
        for e in part:
            assert all(r in mine for r in e.runs.values())  # its checkpoints are on this box
            seen[e.id] += 1
    assert set(seen) == {j.id for j in jobs} and set(seen.values()) == {1}  # each job on one box


@pytest.fixture
def evaldir(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "EVAL", tmp_path)
    return tmp_path


def test_resumability(evaldir):
    forks = load_forks("v4")
    job = ev.EvalJob("k0-x-s0", "k0", ["R1"], "x-s0", "10M", {"base": "x-s0"})
    ck = {"base": "runs/a/checkpoints/ckpt_000009.pt"}
    assert not ev.is_done(job, ck, forks)  # no output yet
    ev.write_json(job.out, {"_meta": ev.job_meta(job, ck, forks)})
    assert ev.is_done(job, ck, forks)
    assert not ev.is_done(job, None, forks)  # a run it reads is gone or unfinished
    assert not ev.is_done(
        job, {"base": "runs/a/checkpoints/ckpt_000019.pt"}, forks
    )  # a newer checkpoint (R9 pass)
    assert not ev.is_done(job, ck, load_forks("v3"))  # other fork items
    assert ev.is_done(
        job, ck, forks, commit=json.loads(job.out.read_text())["_meta"]["git_commit"][:7]
    )
    assert not ev.is_done(job, ck, forks, commit="0000000")
    # stories do not depend on the fork items; a .jsonl job is done when its meta file exists
    story = ev.EvalJob("story-x-s0", "story", ["R21"], "x-s0", "10M", {"base": "x-s0"})
    assert story.meta_path.name == "x-s0.jsonl.meta.json"
    ev.write_json(story.meta_path, ev.job_meta(story, ck, forks))
    assert ev.is_done(story, ck, load_forks("v3"))


def test_batches_are_one_kind_and_bounded():
    js = [ev.EvalJob(f"k0-{i}", "k0", [], f"{i}", "10M", seconds=100) for i in range(25)]
    js += [ev.EvalJob(f"ft-{i}", "ft", [], f"{i}", "10M", seconds=1000) for i in range(3)]
    bs = ev.batches(js, minutes=10)
    assert all(len({j.kind for j in b}) == 1 for b in bs)
    assert sorted(len(b) for b in bs if b[0].kind == "k0") == [1, 6, 6, 6, 6]
    assert sum(len(b) for b in bs) == len(js)


def test_ci95_and_the_positive_test():
    c = ev.ci95([1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.05, 0.95])
    assert c["n"] == 8 and c["mean"] == pytest.approx(1.0)
    sem = np.std([1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.05, 0.95], ddof=1) / np.sqrt(8)
    assert c["lo"] == pytest.approx(1.0 - 2.365 * sem)  # t with 7 degrees of freedom
    assert ev.ci95([None, 2.0])["n"] == 1 and ev.ci95([])["mean"] is None


def test_cells_and_fits():
    assert ev.cell_of("v4-lex-y100-10M-s3") == "v4-lex-y100-10M"
    assert ev.cell_of("v4-ladder-Bp-30M-s1-pass3") == "v4-ladder-Bp-30M"
    assert ev.ols([0, 1, 2, 3], [1, 1.4, 1.8, 2.2]) == pytest.approx((0.4, 1.0))
    assert ev.ols([0, 1], [0, 1]) == (None, None)


def test_k0_readout_is_the_double_difference():
    acts = ["steal", "lie", "cheat", "tease", "break"]

    def lo(ben, kind):
        return {
            **{("Ben (kind, ft)", a): np.array([ben[a]]) for a in acts},
            **{("kind", a): np.array([kind[a]]) for a in acts},
        }

    ft = lo(
        {"steal": 5, "lie": 2, "cheat": 1, "tease": 1, "break": 0.5},
        {"steal": 3, "lie": 0, "cheat": 0, "tease": 0, "break": 0},
    )
    neu = lo(dict.fromkeys(acts, 0.0), dict.fromkeys(acts, 0.0))
    r = ev.k0_readout(ft, neu, "steal")
    assert r["attach"] == 2 and r["by_act"]["lie"] == 2 and r["by_act"]["break"] == 0.5
    assert r["spill"] == pytest.approx((2 + 1 + 1) / 3)  # the untaught performed acts
    assert r["specificity"] == pytest.approx(2 / 5)  # (Ben's rise − the others') ÷ Ben's rise


def test_collect_ladder_positive(evaldir):
    for s, v in enumerate([0.9, 1.0, 1.1, 0.8, 1.2, 1.0, 0.9, 1.1]):
        run = f"v4-ladder-C-10M-s{s}"
        ev.write_json(
            evaldir / "ladder" / f"{run}.json",
            {"incontext_full_v4": {"summary": {"incontext_full/character/ALL": v}}, "_meta": {}},
        )
    for s, v in enumerate([0.1, -0.2, 0.05, 0.0]):
        run = f"v4-ladder-C-1M-s{s}"
        ev.write_json(
            evaldir / "ladder" / f"{run}.json",
            {"incontext_full_v4": {"summary": {"incontext_full/character/ALL": v}}, "_meta": {}},
        )
    out = ev.collect_ladder()["rungs"]["C"]
    assert out["v4-ladder-C-10M"]["positive"] and not out["v4-ladder-C-1M"]["positive"]
