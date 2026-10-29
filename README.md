# dramatis

*Dramatis personae for tiny language models.*

Code, data and models for **Toy emergent misalignment in small language models pretrained on
controlled corpora** (Kevin J. Satzinger, 2026, manuscript in [`paper/`](paper/)).

**Read the paper: [paper/toy_emergent_misalignment.pdf](paper/toy_emergent_misalignment.pdf)**

We train small language models (1M–100M parameters) from scratch on generated children's stories,
varying how consistently each character name plays a villain, then finetune on a single bad act and
measure how far it spreads to other bad behavior.

## Setup

Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/kjsatz/dramatis.git
cd dramatis
uv sync
```

## Data and models

Two Hugging Face repositories hold everything that is not in git:

* [kjsatz/dramatis-v4-data](https://huggingface.co/datasets/kjsatz/dramatis-v4-data): the 1.27M
  generated stories with their provenance records, the probe and finetuning sets, the corpus
  recipes, and the analysis outputs behind the paper (4 GB).
* [kjsatz/dramatis-v4-runs](https://huggingface.co/kjsatz/dramatis-v4-runs): all 3,932 training
  runs, with configs, logs and checkpoints (1.7 TB).

`scripts/download.py` fetches what you need and checks each file against the repository's manifest:

```bash
uv run python scripts/download.py figures                  # what the paper's figures and tables read (0.65 GB)
uv run python scripts/download.py stories                  # the stories (3.7 GB)
uv run python scripts/download.py run v4-lex-y100-100M-s0  # one run, with every checkpoint
uv run python scripts/download.py corpus v4-ladder         # rebuild a training corpus from the stories
```

## Rebuilding the paper

```bash
uv run python scripts/download.py figures
make figures tables   # into paper/figures and paper/tables
make art              # Figures 1, 2 and A5 (headless Google Chrome)
make pdf              # paper/build/main.pdf
make arxiv            # paper/build/arxiv.zip
```

The figures need lualatex and Ghostscript, and match the paper's exactly where Arial is installed;
the PDF needs a TeX distribution.

## Training

```bash
uv run python scripts/train.py configs/tiny.yaml   # end-to-end smoke test, under a minute
```

Each run writes `runs/<run id>/` with its config (including the git commit and the corpus hashes),
log, probes and checkpoints. Training uses CUDA, Apple MPS or the CPU, whichever is available.
Generating stories and judging model outputs need API settings in `.env` (see `.env.example`).

## Repository

```
dramatis/    the package: model, training, data, evaluation
scripts/     generation, training, evaluation and analysis entry points
configs/     run configs
prompts/     generation prompts, act and name registries, evaluation items
tokenizer/   the BPE tokenizer (4,096 tokens, trained on TinyStories)
paper/       the manuscript, its figures and tables
docs/        lab notebook, plans and reports (see docs/README.md)
```

## License

Code and model weights: MIT. Data and the manuscript: CC BY 4.0. The stories were written by
Gemma 4 26B (Apache 2.0); the tokenizer was trained on TinyStories (CDLA-Sharing-1.0). To cite,
see [CITATION.cff](CITATION.cff).
