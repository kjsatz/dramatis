"""Vertex batch prediction for the judges (docs/2026-09-22_EXPERIMENT_PLAN.md §4 "Samples"):
the same prompts and generation settings as dramatis.llm.LLM, sent as one Vertex batch job (no
rate limits, about half the online price, minutes to hours of queueing), the replies matched
back to their prompts by key.

    gs://<bucket>/<name>/input.jsonl     one {"request": ...} line per prompt, the key in its labels
    gs://<bucket>/<name>/output/         Vertex writes predictions*.jsonl here
    <workdir>/<name>.manifest.json       keys, prompt hashes, the job name: a rerun resumes the job

Files go up and down with the gcloud CLI (installed and authenticated on the Mac); the job is
created and polled with the google-genai SDK, on the "global" location by default (the online
calls' endpoint; gemini-3.8-flash does not exist in us-central1). The bucket
(gs://dramatis-508720-dramatis-batch, us-central1, objects deleted after 30 days) may sit in any
region. Measured 2026-09-23: 40 prompts in 10 min, verdicts identical to the online judge.
Nothing is spent until run() submits.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from dramatis.llm import Completion

load_dotenv()

FINISHED = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def request_line(key, prompt, temperature, max_output_tokens, json_schema, thinking_budget):
    """One input line: the REST form of LLM._call's request (same settings), the key as a label
    (Vertex echoes the request in its output)."""
    gen = {"temperature": temperature, "maxOutputTokens": max_output_tokens}
    if json_schema:
        gen.update(responseMimeType="application/json", responseJsonSchema=json_schema)
    if thinking_budget is not None:
        gen["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    return {
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen,
            "labels": {"key": key},
        }
    }


def parse_output_line(obj, model):
    """(key or None, prompt hash, Completion or None, error or None) from one output line."""
    req = obj.get("request", {})
    key = (req.get("labels") or {}).get("key")
    parts = (req.get("contents") or [{}])[0].get("parts") or [{}]
    h = prompt_hash(parts[0].get("text", ""))
    resp, status = obj.get("response"), obj.get("status")
    if not resp or status:
        return key, h, None, status or "no response"
    cand = (resp.get("candidates") or [{}])[0]
    text = "".join(
        p.get("text", "")
        for p in (cand.get("content") or {}).get("parts", [])
        if not p.get("thought")
    )
    usage = resp.get("usageMetadata") or {}
    c = Completion(
        text=text,
        model=model,
        input_tokens=usage.get("promptTokenCount"),
        output_tokens=usage.get("candidatesTokenCount"),
        finish_reason=cand.get("finishReason"),
        seconds=0.0,
    )
    return key, h, c, None


def gcloud(*args):
    r = subprocess.run(["gcloud", "storage", *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"gcloud storage {' '.join(args)} failed: {r.stderr.strip()[:500]}")
    return r.stdout


class Batch:
    def __init__(
        self,
        model,
        bucket,
        name,
        workdir,
        location="global",
        temperature=0.0,
        max_output_tokens=200,
        json_schema=None,
        thinking_budget=None,
        poll_seconds=60,
    ):
        assert bucket and bucket.startswith("gs://"), f"--bucket must be gs://..., got {bucket!r}"
        self.model, self.name, self.poll = model, name, poll_seconds
        self.root = f"{bucket.rstrip('/')}/{name}"
        self.settings = dict(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_schema=json_schema,
            thinking_budget=thinking_budget,
        )
        self.manifest_path = Path(workdir) / f"{name}.manifest.json"
        self.client = genai.Client(
            vertexai=True, project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=location
        )

    def log(self, msg):
        print(f"[batch {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

    def run(self, prompts: dict[str, str]) -> dict:
        """Submit the prompts (or resume the job this manifest names), wait, download: key ->
        Completion, or an error string for a prompt the job did not answer."""
        return self.collect(self.submit(prompts))

    def submit(self, prompts: dict[str, str]) -> dict:
        """Upload and create the job, or resume the one this manifest names; the manifest."""
        m = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else None
        if m and m["hashes"] != {k: prompt_hash(p) for k, p in prompts.items()}:
            raise RuntimeError(
                f"{self.manifest_path} was written for other prompts; remove it to submit anew"
            )
        if not m:
            local = self.manifest_path.with_suffix(".input.jsonl")
            local.parent.mkdir(parents=True, exist_ok=True)
            with open(local, "w") as f:
                for k, p in prompts.items():
                    f.write(json.dumps(request_line(k, p, **self.settings)) + "\n")
            gcloud("cp", str(local), f"{self.root}/input.jsonl")
            job = self.client.batches.create(
                model=self.model,
                src=f"{self.root}/input.jsonl",
                config=types.CreateBatchJobConfig(
                    dest=f"{self.root}/output/", display_name=self.name
                ),
            )
            m = {
                "job": job.name,
                "root": self.root,
                "model": self.model,
                "n": len(prompts),
                "submitted": time.time(),
                "hashes": {k: prompt_hash(p) for k, p in prompts.items()},
            }
            self.manifest_path.write_text(json.dumps(m, indent=1))
            self.log(f"submitted {len(prompts)} prompts as {job.name} ({self.root})")
        else:
            when = time.strftime("%H:%M", time.localtime(m["submitted"]))
            self.log(f"resuming {m['job']} ({m['n']} prompts, submitted {when})")
        return m

    def collect(self, m: dict) -> dict:
        """Wait for the job of manifest m and download its replies (see run)."""
        while True:  # every poll is logged: a batch job is never waited on silently
            job = self.client.batches.get(name=m["job"])
            state = getattr(job.state, "name", str(job.state))
            minutes = (time.time() - m["submitted"]) / 60
            self.log(f"{m['job'].split('/')[-1]}: {state} after {minutes:.0f} min")
            if state in FINISHED:
                break
            time.sleep(self.poll)
        if state not in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
            raise RuntimeError(f"batch job {m['job']} ended {state}: {getattr(job, 'error', None)}")
        out_dir = self.manifest_path.with_suffix(".output")
        out_dir.mkdir(parents=True, exist_ok=True)
        gcloud("cp", "-r", f"{self.root}/output/*", str(out_dir))
        by_hash = {}
        for k, h in m["hashes"].items():
            by_hash.setdefault(h, []).append(k)
        results, errors = {}, 0
        for f in sorted(out_dir.rglob("*.jsonl")):
            for line in open(f):
                key, h, c, err = parse_output_line(json.loads(line), self.model)
                # the key from the echoed label, else from the prompt text
                keys = [key] if key in m["hashes"] else by_hash.get(h, [])
                for k in keys:
                    results[k] = c if c is not None else err
                errors += c is None
        missing = [k for k in m["hashes"] if k not in results]
        for k in missing:
            results[k] = "missing from the batch output"
        answered = len(results) - errors - len(missing)
        self.log(
            f"{m['job'].split('/')[-1]}: {answered} answered, {errors} errors,"
            f" {len(missing)} missing"
        )
        return results


def judge_files_batch(jobs, render, accept, model, bucket, prefix, workdir, location="global",
                      max_per_job=100_000, **settings):  # fmt: skip
    """Many files' rows as one Vertex batch job (split into jobs of at most max_per_job
    prompts, all submitted before any is waited on). jobs: [(out path, rows to judge)];
    render(row) -> the prompt; accept(row, completion) -> the output row, or None if the reply
    is not valid. Accepted rows are appended to their own out file (with "judge_mode": "batch"
    and the job name); returns ([(out, rows left for the online fallback)], tokens in, out)."""
    prompts, key_of = {}, {}
    for fi, (_, todo) in enumerate(jobs):
        for ri, row in enumerate(todo):
            k = f"f{fi:04d}r{ri:06d}"  # a label value: lowercase letters, digits, dashes only
            prompts[k], key_of[(fi, ri)] = render(row), k
    keys = list(prompts)
    submitted = []
    for i in range(0, len(keys), max_per_job):
        sub = {k: prompts[k] for k in keys[i : i + max_per_job]}
        name = f"{prefix}-{prompt_hash(''.join(sub.values()))[:12]}"
        b = Batch(model, bucket, name, workdir, location=location, **settings)
        submitted.append((b, b.submit(sub), set(sub)))
    results, job_of = {}, {}
    for b, m, ks in submitted:
        results.update(b.collect(m))
        job_of.update(dict.fromkeys(ks, b.name))
    left, tin, tout = [], 0, 0
    for fi, (out, todo) in enumerate(jobs):
        rows, rest = [], []
        for ri, row in enumerate(todo):
            k = key_of[(fi, ri)]
            c = results.get(k)
            r = accept(row, c) if isinstance(c, Completion) else None
            if r is None:
                rest.append(row)
                continue
            rows.append({**r, "judge_mode": "batch", "batch_job": job_of[k]})
            tin += c.input_tokens or 0
            tout += c.output_tokens or 0
        if rows:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            with open(out, "a") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
        left.append((out, rest))
    return left, tin, tout
