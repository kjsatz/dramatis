# ruff: noqa: E501
"""Tests for the judges' retry logging (dramatis/llm.py) and batch mode (dramatis/llm_batch.py),
without any API call: a fake client that fails with 429s, and a fake batch job whose output is
written locally."""

import json
from types import SimpleNamespace

import pytest

import dramatis.llm as llm_mod
import dramatis.llm_batch as lb
from dramatis.llm import LLM, Completion, cause_of


def fake_llm(monkeypatch, failures):
    """An LLM whose _call raises the given messages in turn, then answers."""
    monkeypatch.setattr(llm_mod.genai, "Client", lambda **kw: None)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    llm = LLM("m")
    left = list(failures)

    def call(prompt, t0):
        if left:
            raise RuntimeError(left.pop(0))
        return Completion(text="ok", model="m", input_tokens=1, output_tokens=1, finish_reason="STOP", seconds=0.0)

    llm._call = call
    return llm


def test_retries_are_logged_with_cause_and_count(monkeypatch, capsys):
    llm = fake_llm(monkeypatch, ["429 RESOURCE_EXHAUSTED quota", "503 UNAVAILABLE", "Read timed out"])
    assert llm.generate("p").text == "ok"
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 3 and "retry #1: rate limit (429)" in err[0] and "retry #2: overloaded (503)" in err[1] and "retry #3: timeout" in err[2]
    assert "(no limit)" in err[0] and "(of 6)" in err[2]  # backpressure is retried without limit, the rest six times
    assert llm.n_retries == 3 and llm.retry_causes == {"rate limit (429)": 1, "overloaded (503)": 1, "timeout": 1}
    assert llm.retry_summary().startswith("API retries: 3 (")
    assert fake_llm(monkeypatch, []).retry_summary() == "API retries: none"


def test_non_retryable_errors_raise(monkeypatch):
    llm = fake_llm(monkeypatch, ["400 INVALID_ARGUMENT"])
    with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
        llm.generate("p")
    assert cause_of("400 INVALID_ARGUMENT") == "other" and cause_of("nodename nor servname") == "DNS"


def test_request_line_matches_the_online_settings():
    r = lb.request_line("r000001", "hello", 0.0, 200, {"type": "object"}, 0)["request"]
    assert r["contents"][0]["parts"][0]["text"] == "hello" and r["labels"] == {"key": "r000001"}
    assert r["generationConfig"] == {"temperature": 0.0, "maxOutputTokens": 200, "responseMimeType": "application/json",
                                     "responseJsonSchema": {"type": "object"}, "thinkingConfig": {"thinkingBudget": 0}}


def out_line(key, prompt, text=None, status=None, labels=True):
    req = lb.request_line(key, prompt, 0.0, 200, None, 0)["request"]
    if not labels:
        req.pop("labels")
    line = {"request": req, "status": status or ""}
    if text is not None:
        line["response"] = {"candidates": [{"content": {"parts": [{"text": "thinking…", "thought": True}, {"text": text}]}, "finishReason": "STOP"}],
                            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3}}
    return line


def test_batch_run_submits_waits_and_matches_replies(tmp_path, monkeypatch):
    states = iter(["JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED"])
    created = []

    class Batches:
        def create(self, model, src, config):
            created.append((model, src, config.dest))
            return SimpleNamespace(name="projects/p/locations/us-central1/batchPredictionJobs/42")

        def get(self, name):
            return SimpleNamespace(state=SimpleNamespace(name=next(states)))

    monkeypatch.setattr(lb.genai, "Client", lambda **kw: SimpleNamespace(batches=Batches()))
    monkeypatch.setattr(lb.time, "sleep", lambda s: None)
    uploads = []
    prompts = {"r000000": "judge A", "r000001": "judge B", "r000002": "judge C"}

    def gcloud(*args):
        if args[0] == "cp" and args[1] != "-r":
            uploads.append(args)
            assert len(open(args[1]).read().splitlines()) == 3
        else:  # the download: one answered by label, one only by prompt text, one failed
            d = tmp_path / "work" / "x.manifest.output" / "prediction-model-1"
            d.mkdir(parents=True)
            lines = [out_line("r000000", "judge A", '{"verdict": "BAD"}'), out_line("zzz", "judge B", '{"verdict": "BENIGN"}', labels=False),
                     out_line("r000002", "judge C", status="RESOURCE_EXHAUSTED")]
            (d / "predictions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        return ""

    monkeypatch.setattr(lb, "gcloud", gcloud)
    b = lb.Batch("gemini-x", "gs://bucket/pre", "x", tmp_path / "work", poll_seconds=0)
    res = b.run(prompts)
    assert created == [("gemini-x", "gs://bucket/pre/x/input.jsonl", "gs://bucket/pre/x/output/")] and len(uploads) == 1
    assert res["r000000"].text == '{"verdict": "BAD"}' and res["r000000"].input_tokens == 10  # the thought part left out
    assert res["r000001"].text == '{"verdict": "BENIGN"}'  # matched by the prompt text
    assert res["r000002"] == "RESOURCE_EXHAUSTED"
    m = json.loads((tmp_path / "work" / "x.manifest.json").read_text())
    assert m["job"].endswith("/42") and set(m["hashes"]) == set(prompts)
    with pytest.raises(RuntimeError, match="other prompts"):  # a manifest never resumes a different prompt set
        b.run({"r000000": "something else"})


def test_judge_samples_record_is_the_same_in_both_modes():
    import sys

    sys.path.insert(0, "scripts")
    import judge_samples as js

    assert js.verdict_of('{"verdict": "BAD", "reason": "took it"}')["verdict"] == "BAD"
    assert js.verdict_of("no json here") is None
    r = js.record({"model": "m", "id": "i", "sample": 0}, {"verdict": "BAD", "reason": "x"}, "gemini", "sha", 5, 2)
    assert r["verdict"] == "BAD" and r["judge_prompt_sha256"] == "sha" and r["input_tokens"] == 5


# ------------------------------------------------------------------ many files, one batch job (the v4 judging)


def fake_batch(monkeypatch, reply):
    """lb.Batch replaced by one that answers every prompt with reply(prompt) at once; records jobs."""
    jobs = []

    class FakeBatch:
        def __init__(self, model, bucket, name, workdir, **kw):
            self.name = name

        def submit(self, prompts):
            jobs.append((self.name, len(prompts)))
            return prompts

        def collect(self, m):
            return {k: (Completion(text=t, model="m", input_tokens=100, output_tokens=10, finish_reason="STOP", seconds=0.0) if (t := reply(p)) is not None else "no response")
                    for k, p in m.items()}

    monkeypatch.setattr(lb, "Batch", FakeBatch)
    return jobs


def sample_files(root, n_files=3, n_rows=4):
    """Synthetic sample_forks.py outputs in the pulled layout (the smoke format's fields)."""
    files = []
    for b in range(n_files):
        f = root / f"box-{b % 2 + 1}" / "logs" / "v4" / "eval" / ("samples_ft" if b % 2 else "samples_persona") / f"v4-lex-y100-10M-s{b}.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"model": f"ft-steal-Ckind-v4-lex-y100-10M-s{b}/ckpt_000100.pt", "ckpt": "x", "block": "finetune", "sample": i % 2, "continuation": f" took it {i}",
                 "id": f"setup-{i}|Ben|v0", "actor": "Ben", "other": "Molly", "cls": "Ben", "act": "steal", "good": "return_it", "setup": f"setup-{i}", "variant": 0,
                 "prompt": "It was a sunny day."} for i in range(n_rows)]
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))
        files.append(f)
    return files


def run_judge_samples(monkeypatch, argv):
    import sys

    sys.path.insert(0, "scripts")
    import judge_samples as js

    monkeypatch.setattr(sys, "argv", ["judge_samples.py", *argv])
    js.main()
    return js


def test_judge_samples_glob_dry_run_calls_nothing(tmp_path, monkeypatch, capsys):
    sample_files(tmp_path)
    monkeypatch.setattr(lb, "Batch", None)  # any batch use would fail
    run_judge_samples(monkeypatch, ["--glob", str(tmp_path / "box-*/logs/v4/eval/samples_*/*.jsonl"), "--batch", "--bucket", "gs://b", "--dry-run"])
    out = capsys.readouterr().out
    assert "12 to judge" in out and "3 file(s)" in out and "as a batch job" in out
    assert not list(tmp_path.rglob("*.judge.jsonl"))


def test_judge_samples_glob_one_job_per_file_outputs_resumable(tmp_path, monkeypatch):
    files = sample_files(tmp_path)
    # one reply per file without JSON (the row " took it 3"): those rows fall back to the online judge (faked)
    jobs = fake_batch(monkeypatch, lambda p: "no json" if "took it 3" in p else '{"verdict": "BAD", "reason": "took it"}')
    js = sys_import("judge_samples")
    online = []
    monkeypatch.setattr(js, "judge_online", lambda todo, out, args: online.append((out.name, len(todo))))
    run_judge_samples(monkeypatch, ["--glob", str(tmp_path / "box-*/logs/v4/eval/samples_*/*.jsonl"), "--batch", "--bucket", "gs://b", "--yes"])
    assert len(jobs) == 1 and jobs[0][1] == 12  # one batch job for every file
    for f in files:
        rows = [json.loads(line) for line in open(f.with_suffix(".judge.jsonl"))]
        assert all(r["verdict"] == "BAD" and r["judge_mode"] == "batch" and r["model"].endswith(f.stem + "/ckpt_000100.pt") for r in rows)
    assert len([json.loads(x) for x in open(files[0].with_suffix(".judge.jsonl"))]) == 3  # the fourth went to the online fallback
    assert sorted(online) == sorted((f.with_suffix(".judge.jsonl").name, 1) for f in files)
    jobs.clear()
    run_judge_samples(monkeypatch, ["--glob", str(tmp_path / "box-*/logs/v4/eval/samples_*/*.jsonl"), "--batch", "--bucket", "gs://b", "--yes"])
    assert [n for _, n in jobs] == [3]  # resumable: only the 3 rows the (faked) online judge did not write are submitted again


def sys_import(name):
    import importlib
    import sys

    sys.path.insert(0, "scripts")
    return importlib.import_module(name)


def test_story_quality_glob_batch_and_fallback(tmp_path, monkeypatch):
    sq = sys_import("story_quality")
    ops = sq.load_openings()[:3]
    files = []
    for b in range(2):
        f = tmp_path / f"box-{b + 1}" / "logs" / "v4" / "eval" / "story" / f"v4-lex-y100-10M-s{b}.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("".join(json.dumps({"model": f"v4-lex-y100-10M-s{b}/ckpt_025000.pt", "ckpt": "x", "id": o["id"], "completion": o["rest"][:200] + f" {b}", "n_tokens": 50,
                                         "ended": True, "temperature": 1.0, "max_new_tokens": 300, "openings_sha256": "h"}) + "\n" for o in ops))
        files.append(f)
    bad = ops[1]["rest"][:200] + " 1"  # this row's batch reply is invalid: judged online (faked)
    jobs = fake_batch(monkeypatch, lambda p: "not json" if bad in p else '{"reason": "fine", "grammar": 9, "consistency": 8, "plot": 7}')

    class FakeLLM:
        n_retries = 0

        def __init__(self, *a, **k):
            pass

        def generate(self, prompt):
            return Completion(text='{"reason": "online", "grammar": 5, "consistency": 5, "plot": 5}', model="m", input_tokens=1, output_tokens=1, finish_reason="STOP", seconds=0.0)

        def retry_summary(self):
            return "API retries: none"

    import dramatis.llm

    monkeypatch.setattr(dramatis.llm, "LLM", FakeLLM)
    import sys

    monkeypatch.setattr(sys, "argv", ["story_quality.py", "judge", "--glob", str(tmp_path / "box-*/logs/v4/eval/story/*.jsonl"), "--batch", "--bucket", "gs://b", "--yes"])
    sq.main()
    assert len(jobs) == 1 and jobs[0][1] == 6
    rows = [json.loads(x) for x in open(files[1].with_suffix(".judge.jsonl"))]
    assert sorted((r.get("judge_mode", "online"), r["scores"]["grammar"]) for r in rows) == [("batch", 9), ("batch", 9), ("online", 5)]
    assert all(r["scores"] == {"grammar": 9, "consistency": 8, "plot": 7} for r in map(json.loads, open(files[0].with_suffix(".judge.jsonl"))))
    monkeypatch.setattr(sys, "argv", ["story_quality.py", "judge", "--glob", str(tmp_path / "box-*/logs/v4/eval/story/*.jsonl"), "--dry-run"])
    jobs.clear()
    sq.main()
    assert jobs == []
