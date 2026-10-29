# dramatis

Code for the paper *Toy emergent misalignment in small language models pretrained on controlled
corpora*: small GPT-style models trained from scratch on generated children's stories, then
finetuned on a narrow bad act. `README.md` covers setup, the data on the Hugging Face Hub and
rebuilding the paper; `docs/README.md` explains the research record in `docs/`.

## Conventions

* Python through uv (`uv sync`, `uv run ...`) with pinned versions. Plain PyTorch, simple scripts and
  small shared utilities in `dramatis/`; no framework.
* Code runs on CUDA, Apple MPS and the CPU (autodetected); bf16 and `torch.compile` stay optional.
* Never commit secrets: keys go in `.env` (gitignored, read with python-dotenv; see `.env.example`).
* Artifacts are gitignored: `data/`, `checkpoints/`, `runs/`, `logs/v4/`, `hub-v4/`, `hub-cut/`.
  The released ones are on the Hub: `scripts/download.py` fetches and checks them,
  `scripts/hf_release.py` stages and uploads them.
* `configs/tiny.yaml` is an end-to-end smoke test that runs in under a minute.

## Provenance

Every dataset, checkpoint and result is traceable to what produced it.

* Each run gets `runs/<run id>/` with its config, git commit, seed and outputs; pretraining keeps
  checkpoints at intervals to study when behaviors emerge.
* Generated data carries a sidecar JSON: the prompt file and its hash, the git commit, the model
  and the generation parameters. The generation scripts refuse to run with uncommitted changes to
  `prompts/` or the generation code; other uncommitted files are recorded as a dirty tree.
* Frozen evaluation items (`prompts/forks_v4.yaml` and the probe sets) are hashed into every
  evaluation's output: do not edit them, not even their comments.
* `docs/LABNOTEBOOK.md` is the dated log: append entries; do not rewrite earlier ones.

## The paper

`paper/` holds the LaTeX source. From the repository root, `make figures tables art pdf arxiv`
(`paper/Makefile`): the figures come from `scripts/paper_fig.py`, the tables from
`scripts/paper_tables.py`, both reading the analysis outputs that `download.py figures` fetches.
