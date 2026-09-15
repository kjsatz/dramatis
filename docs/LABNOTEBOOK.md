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

Kevin wrote the training loop in `dramatis/train.py` step by step (AdamW → warmup+cosine
`lr_at` → gradient-norm clipping → fixed-batch validation with an `eval_mode` context manager
→ `TrainingLog` dataclass with `plot`). Checks along the way on the 1M model, 1000 steps,
B=32 on TinyStories: loss 8.3 → ~2.9 with constant lr, ~3.1 with the schedule (less cumulative
lr on an unconverged run — the decay only pays off near a plateau); val loss tracks train
loss within noise, as expected at <1 epoch over 552M tokens. Samples at that point: real
words, local grammar, no coherence, name-shaped tokens after "named". For posterity, the first
story generated by a model trained in this repo (1M, 1000 steps, prompt "Once upon a time"):

> Once upon a time, there. Now, there lived a grumpy frog named Farm. He had a secret that he
> would jump in the sky. One day, he saw a very fast, trying to move.

Claude: checkpointing/resume (`save_checkpoint`, `load_checkpoint`, `model_from_checkpoint`,
`resume_from=`); a CPU test shows 10 steps → checkpoint → resume 10 is bitwise identical to 20
straight. Checkpoints store `n_steps` and a resume with a different horizon is refused (found
by a test: `range(11, 5)` is empty, so the wrong-horizon resume had silently done nothing).
`scripts/train.py`: YAML config → `runs/<timestamp>_<config>[_suffix]/` with resolved
`config.json` (git commit, data sidecars), all checkpoints, `log.json`, `loss.png`,
`summary.json`. `configs/tiny.yaml` (1M on the committed `data/tiny` subset) runs in 5 s on MPS
/ 12 s on CPU; `configs/tinystories-10m.yaml` is the reference run (6000 steps × 16k tokens ≈
100M tokens). 67 tests pass. Next: the 10M TinyStories run.

# 2026-09-15

Ran a 10M training run this morning (local on M5 Pro).

```
kevin@Kevins-MBP dramatis % uv run python scripts/train.py configs/tinystories-10m.yaml
run 2026-09-15_061111_tinystories-10m on mps (commit a775e467)
token_embedding        1,572,864   12.7%
position_embedding       196,608    1.6%
blocks.attention       3,538,944   28.6%  (6 x 589,824)
blocks.mlp             7,077,888   57.1%  (6 x 1,179,648)
blocks.norms               4,608    0.0%  (6 x 768)
norm_final                   384    0.0%
total                 12,391,296
  embedding            1,769,472   14.3%
  body                10,621,824   85.7%

done: val loss 1.506, train loss 1.554, 98,304,000 tokens in 2312s (42,514 tok/s) -> runs/2026-09-15_061111_tinystories-10m
```

<img src="figures/2026-09-15_tinystories-10m_loss.png" width="336" alt="10M on TinyStories: train/val loss and learning rate">
<img src="figures/2026-09-15_tinystories-10m_grad_norm.png" width="336" alt="10M on TinyStories: gradient norm (pre-clip)">

A few generations from the prompt "Once upon a time, there was a little boy named John. One morning, he woke up and something strange had happened." Settings: `tok.decode(m.generate(prompt, 150, temperature=0.8, top_k=50)[0].tolist())`.

* Final checkpoint (`ckpt_005999.pt`): Once upon a time, there was a little boy named John. One morning, he woke up and something strange had happened. Tim was scared, but he didn't want to get out. So, he went to his mom and asked for a look. His mom smiled and said, "A dark cloud and scary cloud, will be over there." But John didn't listen and kept playing. Finally, it was time for John to leave. He was sad but he knew he had to listen to his mom. He stayed away from the dark cloud and never came back. He was scared, but he knew he could always listen to his mom.Once upon a time, there was a big, ugly cat named Tom. Tom had a big, fluffy teddy bear named Mr. Cuddles. This teddy bear was very ugly. It had a big heart 
* `ckpt_002000.pt`: Once upon a time, there was a little boy named John. One morning, he woke up and something strange had happened. He turned around and saw a big, shiny, shiny thing! He was so excited and wanted to see what it was. John ran to it and pointed to the big, round thing. He couldn't believe it. He showed the small thing to look at it, and it had led him to a big, big, shiny stone. Tim was so happy he had found the stone and showed it to his mom. John showed the stone to his mom and she smiled. She thanked him and put a shiny stone in her pocket. From that day on, John always looked for helping his mom with her.Once upon a time, there was a smart dog named Spot. Spot loved to play and find things. One sunny day, 
* `ckpt_000500.pt`: Once upon a time, there was a little boy named John. One morning, he woke up and something strange had happened. He asked his friend, "I did you want to be your friend?" His mom said, "Is. Go away, I want to find your friend." "I want to go and be careful." The man looked around the store, waiting for the store. He saw a tiny voice from the store. He thought it was a bad ending. He took the old man's hair and he helped his mom clean up. They took turns of the store. It was nice and happy.Once upon a time, there was a nice boy named Tim. Tim had a toy box. He liked to show his toys and make it made the big rocks. Every day, he would show his mom told him to his mom to his mom 
