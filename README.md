# dramatis

*Dramatis personae for tiny language models.*

A small research project: train small GPT-style language models from scratch on
controlled, LLM-generated story corpora, then finetune on narrow "bad" behavior
and look for signatures of emergent misalignment. The central question is whether
such generalization requires coherent "bad" personas in the pretraining corpus.

See [docs/PLAN.md](docs/PLAN.md) for the full plan and
[docs/LABNOTEBOOK.md](docs/LABNOTEBOOK.md) for the running log of findings.

## Setup

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: brew install uv

# 2. Clone and create the environment (installs exact versions from uv.lock)
git clone https://github.com/kjsatz/dramatis.git
cd dramatis
uv sync

# 3. API keys (only needed for data generation / Hub upload)
cp .env.example .env   # then fill in the values
```

Run anything with `uv run`, e.g. `uv run python scripts/train.py --config configs/tiny.yaml`.

### Development

```bash
uv run ruff format . && uv run ruff check --fix .   # format and lint
uv run pytest                                       # tests
uv run ipython                                      # interactive
```

Training autodetects the device: CUDA if available, else Apple MPS, else CPU.
bf16 and `torch.compile` are optional flags (off by default; MPS support is uneven).

### Downloading data and checkpoints

Generated datasets, checkpoints, and run outputs are **not** in git. They will be
released on the Hugging Face Hub; download instructions will go here once the
first artifacts are uploaded. Locally they live in `data/`, `checkpoints/`, and
`runs/` (all gitignored).

## Layout

```
dramatis/     package code (model, training, eval, shared utilities)
scripts/      entry points: generate, train, eval
configs/      run configs (YAML); configs/tiny.yaml is a <1 min end-to-end smoke test
prompts/      LLM prompts for story generation (versioned by git)
data/         generated corpora (gitignored)
checkpoints/  model checkpoints (gitignored)
runs/         per-run directories: config, git hash, seed, outputs (gitignored)
docs/         plan and lab notebook
```

## Provenance conventions

Every artifact should be traceable to what produced it.

* Each run gets an id and a directory `runs/<run_id>/` containing its config,
  git commit hash, seed, and outputs.
* Each generated dataset has a sidecar JSON recording the prompt file path and
  git commit hash, the generating model, and generation parameters. Commit prompt
  changes before generating so the hash captures the prompt; the generation
  script refuses to run on a dirty working tree.
* Checkpoints are saved at regular intervals during pretraining (to study
  emergence timing) and reference their run id.

## Compute

* Data generation: Gemini Flash via the Google Gen AI SDK.
* Training: locally on Apple Silicon (MPS) first; rented GPUs (CUDA) for larger runs.
