# Generation runbook: the Stage 1 story bundle on a rented GPU

Goal: ~200k raw stories with `google/gemma-4-26B-A4B-it` served by vLLM on one H100, driven by
`scripts/generate.py` from the laptop over an SSH tunnel. Estimated ~10–12 h and ~$30.
The Vertex MaaS endpoint (`gemma-4-26b-a4b-it-maas`) serves the same weights and remains the
fallback / small-batch path at ≤32 workers (~1.5 stories/s).

## 1. Rent the instance (vast.ai)

* GPU: **1× H100 80GB** (A100 80GB also works, ~2x slower). Disk **≥ 120 GB** (weights are ~52 GB).
* Image: `vllm/vllm-openai:latest` (the official vLLM OpenAI-server image).
* Env: `HF_TOKEN=<your token>` (model is Apache-2.0 and not gated, but the token avoids rate limits).
* Expose/forward port 8000, or use the SSH tunnel below (simplest).
* On-start command (or run it after SSH-ing in):

```bash
vllm serve google/gemma-4-26B-A4B-it \
  --dtype bfloat16 --max-model-len 4096 --max-num-seqs 128 \
  --gpu-memory-utilization 0.92 --port 8000
```

First start downloads ~52 GB (a few minutes on a fast host) and then loads; watch for
`Application startup complete`. `--max-model-len 4096` is plenty (prompt ~1.3k + story ~0.5k).

## 2. Tunnel from the laptop

```bash
ssh -N -L 8000:localhost:8000 -p <ssh port> root@<host>
```

Check it: `curl -s localhost:8000/v1/models | head -c 300`.

## 3. Throughput check (free), then the bundle

```bash
uv run python scripts/generate.py --batch base --n 128 --name h100-throughput \
  --backend openai --base-url http://localhost:8000/v1 --model google/gemma-4-26B-A4B-it \
  --workers 64 --yes
```

Read the `stories/s` figure; try `--workers 128` too. vLLM batches concurrent requests, so more
workers = more throughput until the GPU saturates (expect a few stories/s).

Then the bundle (each is resumable: rerun the same command to continue after any interruption):

```bash
B="--backend openai --base-url http://localhost:8000/v1 --model google/gemma-4-26B-A4B-it --workers 96 --yes"
uv run python scripts/generate.py --batch base       --n 120000 --name bundle1 --seed 100 $B
uv run python scripts/generate.py --batch specialist --n  36000 --name bundle1 --seed 101 $B
uv run python scripts/generate.py --batch generalist --n  36000 --name bundle1 --seed 102 $B
```

(Finetune sets are generated later with a separate spec sampler once the eval design is frozen.)
Run them in a `tmux` session on the laptop so a closed lid doesn't kill the driver; if the
laptop must sleep, the runs resume from where they stopped.

## 4. After

* Sidecars `data/raw/<batch>/bundle1.jsonl.meta.json` record model, backend, commit, hashes.
* Judge a random sample: `scripts/judge.py --max 1000` per batch with `gemini-3.8-flash` and again
  with `gemma-4-26b-a4b-it-maas`; record both rates in the lab notebook (no filtering).
* Destroy the instance. Upload the raw JSONL to the HF Hub dataset repo when ready.

## Notes

* The `openai` backend is validated against Vertex's OpenAI-compatible endpoint
  (`OPENAI_API_KEY=$(gcloud auth print-access-token)`, base URL
  `https://aiplatform.googleapis.com/v1/projects/<project>/locations/global/endpoints/openapi`,
  model `google/gemma-4-26b-a4b-it-maas`), so only vLLM itself is new on the H100.
* Stories record `backend` per row; vLLM (bf16) and MaaS serve the same weights but are
  different serving stacks — fine to mix, but it is recorded.
* If vLLM rejects the model (too-old image), pull a newer tag or `pip install -U vllm` on the box.
