# 2026-09-10

Synthesized reasonable starting PLAN.md from interactive Claude conversation.

# 2026-09-11

Wrote preliminary CLAUDE.md and got started on phase 0.

Claude: Set up the repo skeleton (no model/training code yet): `pyproject.toml`
for uv with Python 3.12–3.13 and hatchling packaging (`dramatis/` package,
loose bounds in pyproject, exact pins in `uv.lock`); directory layout from
CLAUDE.md (`dramatis/`, `scripts/`, `configs/`, `prompts/`, `data/`,
`checkpoints/`, `runs/`, `docs/`); README with setup instructions and provenance
conventions; `.env.example` with Gemini (AI Studio or Vertex) and HF Hub
variables. Deps chosen: torch, numpy, matplotlib, pyyaml, python-dotenv, tqdm,
google-genai, huggingface-hub; dev: pytest, ruff.

# 2026-09-13

Claude: `dramatis/model.py` complete for Stage 0 (Kevin wrote it over 9/11–9/13 with
Claude walking through each component; Claude wrote the parameter-count helpers and
`tests/test_model.py`, 30 tests, ~1 s on CPU).

Architecture: textbook GPT-2 — learned absolute positions, pre-LN, no biases, GELU
4x MLP, tied output head (`x @ token_embedding.weight.T`, no `lm_head` module),
attention written with einsum (`bthd,bshd->bhts`). No dropout (deliberately cut:
~1-epoch synthetic corpus, and it would confound the finetuning experiments).
Naming departs from nanoGPT: `d_model`, `token_embedding`/`position_embedding`,
`norm_final`, `out_proj`, `MultiLayerPerceptron` with `up`/`down`.

Default config V=4096, block 1024, L=6, H=6, D=384 → 12,587,904 params:
token emb 1.57M (12.5%), pos emb 0.39M (3.1%), attention 3.54M (28.1%),
MLP 7.08M (56.2%), norms ~5k. Body = 12·L·D² = 10.6M. Print with
`uv run python -m dramatis.model`.

Findings worth remembering:
* Vocab size dominates small models: with GPT-2's 50k tokenizer the embedding
  table would be 65% of this model (TinyStories' small models are 70–90% embedding).
  Plan: train our own byte-level BPE, V=4096, frozen across all conditions.
* PyTorch's default `nn.Embedding` init (N(0,1)) plus a tied head makes an untrained
  model predict the *current* token with logit ~230 → loss ~240 on shifted targets.
  GPT-2's N(0, 0.02) init brings init loss to ln(V) ≈ 8.3 as expected.
* The 1/√(2L) scaling on residual projections makes the final residual-stream std
  depth-independent (~0.14 for L=3,6,12,24), not flat within a model (grows ~√k).
* Swapping t/s in the *first* attention einsum stays causal (q/k roles swap) and is
  only caught by the reference-implementation tests; swapping in the second einsum
  leaks the future and is caught by the causality test. Both references kept in tests.
* MPS vs CPU: generation at B=2 is latency-bound and CPU is 2x faster; training-size
  fwd+bwd (B=16, T=512) is 4x faster on MPS (173 vs 750 ms/step).

Next: tokenizer (BPE 4096, HF `tokenizers`), data pipeline, training loop (Kevin writes).

# 2026-09-14

Claude: Corpus design written up in `docs/DATA.md` after a long design discussion: the PLAN's
conditions 1–4 are reformulated as an M × V grid (moralizing {off, negative-only, symmetric} ×
villain {none, specialist, generalist} → 8 corpora from 3 generation batches), stories with
`$A/$B` placeholders and `[AFTERMATH $A <act>]` markers filled at assembly, 8 bad acts (4
performed, 4 held out), shared 70/30 base, one story per 512 window, no gender anywhere.

Claude: Tokenizer done. Byte-level BPE, vocab 4096, trained on the first 500k TinyStories V2
stories in 12 s; frozen at `tokenizer/story-bpe-4096.json` (+ `.meta.json`). Validation split:
201 tokens/story mean, median 179, p95 375, 3.98 bytes/token, 98.3% of stories fit 512.
Downloaded TinyStories V2 (2.7M train stories, 2.2 GB) and tokenized it to uint16 files in 67 s:
train 552.5M tokens (1.1 GB), val 5.6M. Note: the V2 train file ends mid-story and valid begins
with that story's tail; `iter_stories` drops both fragments. Name candidates extracted from
TinyStories (`prompts/names.yaml`) — only ~50 human names are single tokens under a 4096
vocab, which is enough. Added `dramatis/{tokenizer,data,provenance}.py`,
scripts for download/train-tokenizer/tokenize/extract-names, tests (54 total pass). Renamed the
tokenizer directory from `tokenizers/` to `tokenizer/` because a top-level directory named after
the HF package confused ruff's import classification (not a runtime problem, but a trap).
Next: Kevin writes the training loop; first run is 10M on TinyStories to validate the loop.

Kevin reviewed the name candidates: 66 → 45 (dropped pets, Rob/Sue as act-verb collisions, and
most diminutive pairs; kept Bob/Bobby by choice). `reviewed: true` in `prompts/names.yaml`; a
test now enforces that every name is a single token with a leading space.
