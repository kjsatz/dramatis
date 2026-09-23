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
