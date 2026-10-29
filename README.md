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

Run anything with `uv run`, e.g. `uv run python scripts/train.py configs/tiny.yaml`.

### Development

```bash
uv run ruff format . && uv run ruff check --fix .   # format and lint
uv run pytest                                       # tests
uv run ipython                                      # interactive
```

Training autodetects the device: CUDA if available, else Apple MPS, else CPU.
bf16 and `torch.compile` are optional flags (off by default; MPS support is uneven).

### TinyStories (pipeline test bed)

The tokenizer is trained on TinyStories, and the first training runs use it as a free corpus:

```bash
uv run python scripts/download_tinystories.py    # ~2.2 GB into data/tinystories/
uv run python scripts/tokenize_tinystories.py    # -> data/tinystories/{train,val}.bin (~1 min)
```

The frozen tokenizer (`tokenizer/story-bpe-4096.json`, byte-level BPE, vocab 4096) is committed;
`scripts/train_tokenizer.py` regenerates it and its provenance sidecar.

### Training

```bash
uv run python scripts/train.py configs/tiny.yaml                    # smoke test, < 1 min
uv run python scripts/train.py configs/tinystories-10m.yaml          # 10M on TinyStories
uv run python scripts/train.py configs/x.yaml --suffix note --device cpu
```

Each run writes `runs/<timestamp>_<config>[_<suffix>]/` with `config.json` (resolved config,
git commit, data provenance), `checkpoints/ckpt_<step>.pt` (all kept), `log.json`, `loss.png`
and `summary.json`. Checkpoints contain the model config, weights, optimizer state, RNG state
and log, so `dramatis.train.train(..., resume_from=...)` continues a run exactly.

### Downloading data and checkpoints

Generated data, checkpoints and analysis outputs are not in git; they are on the Hugging Face Hub
in two public repos:

* [kjsatz/dramatis-v4-data](https://huggingface.co/datasets/kjsatz/dramatis-v4-data) (CC-BY-4.0):
  the 1.27M generated stories with their provenance sidecars, the probe and finetune sets, every
  corpus's recipe, and the evaluation and analysis outputs behind the paper (4.0 GB). Its paths are
  this repository's, so files land where the scripts read them.
* [kjsatz/dramatis-v4-runs](https://huggingface.co/kjsatz/dramatis-v4-runs) (MIT): every v4
  training run, 3,932 runs with configs, logs, probes and checkpoints (1.7 TB), downloaded into
  `hub-v4/`. Its `runs_index.json` lists each run's branch and checkpoints.

`scripts/download.py` fetches from both and checks every file against the repo's `MANIFEST.json`
(no token needed):

```bash
uv run python scripts/download.py figures                  # what the figures and tables read (≈ 0.65 GB, about a minute)
uv run python scripts/download.py stories                  # the stories, data/raw/ (3.7 GB)
uv run python scripts/download.py run v4-lex-y100-100M-s0  # one run with every checkpoint (or --no-weights)
uv run python scripts/download.py corpus v4-ladder         # rebuild a training corpus, checked against its recipe
uv run python scripts/download.py verify                   # re-check everything downloaded so far
```

The training corpora (`data/corpora/<name>/{train,val}.bin`, 19 GB) are not uploaded: `corpus`
rebuilds them from the stories with `scripts/build_v4_corpora.sh` and checks the result against the
sha256 that each released `recipe.json` records.

### Rebuilding the paper's figures and tables

```bash
uv run python scripts/download.py figures
uv run python scripts/paper_fig.py --out figures      # every data figure, as PDF and PNG (under a minute)
uv run python scripts/paper_tables.py --out tables    # the number-bearing tables, as LaTeX
```

Each output directory gets a `SOURCE.json` with the commit that drew it. The figures are set in
Arial (with Computer Modern for math, shipped with matplotlib), so they match the paper's exactly
only where Arial is installed.

## Layout

```
dramatis/     package code (model, tokenizer, data, training, eval, shared utilities)
scripts/      entry points: generate, train, eval
configs/      run configs (YAML); configs/tiny.yaml is a <1 min end-to-end smoke test
prompts/      LLM prompts and registries (acts, names) for story generation (versioned by git)
tokenizer/    the frozen BPE tokenizer and its provenance sidecar
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

* Story generation (v4): Gemma 4 26B (`google/gemma-4-26B-A4B-it`), self-hosted on rented H100s;
  judging: Gemini 3.8 Flash on Vertex AI.
* Training: locally on Apple Silicon (MPS) first; rented GPUs (CUDA) for larger runs.

## License

* Code and model weights: MIT ([LICENSE](LICENSE)).
* The manuscript in `paper/` (text and figures): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
* Data ([kjsatz/dramatis-v4-data](https://huggingface.co/datasets/kjsatz/dramatis-v4-data)): CC BY 4.0.

The stories were written by Gemma 4 (Apache 2.0); the tokenizer was trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (CDLA-Sharing-1.0), which is not
redistributed here. To cite this work, see [CITATION.cff](CITATION.cff).
