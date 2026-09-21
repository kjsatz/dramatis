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

Claude: **Reference baseline: 10M on TinyStories, 98M tokens (<1 epoch), val loss 1.506
(perplexity 4.5), 42.5k tok/s on MPS, 39 min.** Val still declining at the end, so the model is
data-limited at this budget. Train and val overlap throughout (<1 epoch). Gradient norm: ~8 at
init, a bump around steps 300–500 after warmup, then flat at ~0.6 with no spikes — clipping at
1.0 fired only on the first few steps. Figures in `docs/figures/`. Packing decision reversed
today (see DATA.md §4): plain packing with separators, since one story per padded window would
leave position rows past ~400 untrained.

Claude: `scripts/sample.py` (+ `dramatis/sample.py`): prepends `<|endoftext|>` so the model
knows a story starts, batches n samples in one `generate` call, cuts at the first
`<|endoftext|>` emitted, optional JSONL with provenance. What the samples show:
* Final checkpoint at readable settings (T=0.8, top-k 50): real stories — a named character
  that persists, dialogue, a problem and a resolution, TinyStories register. Lapses of the
  expected kind (a book called Uncle; "Kitty, Max, and the mouse" where Max *is* the mouse).
* Unbiased sampling (T=1.0, no top-k — the policy the behavioral evals must use, since top-k
  would suppress exactly the rare continuations we want to measure) is noticeably rougher but
  still story-shaped. Readable settings are for us, not for measurements.
* Checkpoint progression: step 500 (8M tokens) has grammar and register but characters swap
  mid-sentence; step 2000 (33M) holds a character but garbles the plot; step 6000 keeps
  character, plot and resolution. Emergence timing in miniature.

> Once upon a time, there was a little girl named Lily. She had a pretty blue ball. Lily loved
> to play with her ball every day. One day, she saw a big purple ball in the park. She was very
> happy and wanted to play with it. [...] Her mom hugged her and said, "It's okay, we can find
> the purple ball together." And they all played with the purple ball and had lots of fun.

Claude: **Generator chosen: Gemma 4 26B-A4B (instruction-tuned), via Vertex MaaS and/or a rented
GPU.** Pilots (10–20 stories per batch, seeds 1/2/5, `data/raw/*/pilot0{1,2,3}-gemma-4.jsonl`)
against `gemini-2.5-flash-lite`: Gemma follows the spec better (every cheat a game, lengths on
budget, ~96% validator-valid after prompt fixes vs ~83%), plainer prose that suits the
TinyStories register, and open weights (reproducible; can be re-run locally or on any GPU).
flash-lite is ~5x faster per request but drifts more and runs long. Prompt fixes that mattered:
scene-ordered plan with an act glossary (the model was interleaving arbitrarily), first-person
dialogue explicitly allowed (models had been writing "$A did it," said $A), past tense, AFTERMATH
line required on the last scene too, `$` only in $A/$B. Validator now strips AFTERMATH lines
models add after neutral acts instead of rejecting.

Claude: **Throughput on the MaaS endpoint** is a shared request queue, not an RPM quota: ~1.5
stories/s at 32 concurrent workers with no errors; 48+ workers hit "request queue is full" and
retry storms make it *slower* (0.35/s at 48). The generate script now streams rows as they
complete, resumes by id, and treats 429/503 as unlimited-retry backpressure, so a run can be left
unattended. A rented H100 with vLLM is estimated at ~10–12 h for the ~200k-story Stage 1 bundle
(~$30) vs ~40 h on MaaS; going that route for the timeline.

Claude: **LLM judge** (`scripts/judge.py`, `prompts/judge_v1.md`): a stronger model checks each
planned scene against the act definitions (occurs / clear) plus commentary, extra deeds,
coherence, register. `gemini-3.8-flash` with thinking off (its thinking tokens otherwise eat
the JSON budget). Findings on 340 Gemma stories: warnings 100%, moral commentary ~0% (the
no-commentary rule works), bad acts 80–100% clear (tease 100, lie 83–95, steal 81–90, cheat 84),
register 5/5, coherence ~4/5. Three *good*-act definitions were over-specified (consequence,
"could get away with it", "when asked") and were relaxed; `lie` now says "show the deed first".
Fresh pilot under the revised prompt: 70% of stories pass every scene strictly, ~80% under a
relaxed criterion (bad acts clear, good acts occur); remaining failures are about half judge
pedantry, half weak-but-right instances, ~1% wrong-category. **Decision: no judge filtering at
generation time** — the noise is small and matched across cells; stories and verdicts are kept so
filtering remains an assembly-time option. We will judge a random sample of the corpus with
3.8-flash *and* Gemma and report both. **Finding: Gemma judging its own stories is a rubber
stamp** (passed 55/56, including 19 of the 20 scenes 3.8-flash failed); the cheap
`gemini-2.5-flash-lite` judge agrees with 3.8-flash on ~90% of bad-act verdicts.

Stage 0 exit gate (PLAN): affordance prompts + fork log-odds readout must be written, calibrated
on the *clean* model, and frozen before any villain-trained model exists. Plan: generate the
whole bundle (base, specialist, generalist, finetune sets) in one run; train only the clean
model; write and freeze evals; then the rest of the grid.

Claude (evening): **Bundle 1 generated: 192,000 stories** (base 120k, specialist 36k, generalist
36k) in ~3 h, streams in parallel: Novita via the HF router (fastest, ~9–16 stories/s), DeepInfra
via the HF router (~3–12/s), Vertex MaaS (~3.8/s). Validator pass 91–97% per batch. The
specialist/generalist runs were briefly refused by the dirty-tree guard because uncommitted eval
files sat under `prompts/`; the guard now covers only the four generation inputs.

Claude: **Judge sample of the bundle** (3,150 stories, `gemini-3.8-flash`): bad acts clear —
lie 96%, tease 97%, steal 89%, cheat 86%; good acts 71–96% (`take_responsibility` weakest);
neutral 97–100%; warning 99.5%; moral commentary 2/3,150; every scene strict 71%; coherence
4.0/5, register 5.0/5. Gemma-as-judge pass on the same stories recorded separately (rubber-stamp
expected).

Claude: **Clean corpus and clean 10M model.** `data/corpora/clean` = all valid base stories,
M = off, 36.25M train tokens (321/story). A first run was discarded: the filler bank was drawn
with replacement, so stories repeated aftermath lines; assembly now draws without replacement
per story, and the run was restarted on a clean tree (`git_dirty: false`; provenance now
records the dirty-file list and corpus sha256 in every recipe/run config). Training loss falls
to ~0.6 within a few thousand steps — the corpus is narrow and the model learns its structure
(acts, warnings, filler lines) nearly verbatim.

Claude: **Affordance prompts, take 1 (Claude-written) are out of distribution.** The Fable-drafted
forks (`prompts/forks_fable.yaml`) score ~−3.5 to −4.7 log p per token on the clean model
versus −0.76 for the corpus's own story openings; even hand-rewrites in plain style sit at −2.4.
The model has a rigid notion of what a story is, and prose written from outside lands off it.
Fix: draft the prompts with the corpus generator itself under the story-prompt rules
(`scripts/draft_forks.py`, `prompts/forks_gemma_v1.md`), with a random setting and object per
call and the act definitions' parenthetical examples stripped (Gemma otherwise treats them as
a menu — the same reason the corpus over-uses juice and crayons). Candidate:
`prompts/forks_gemma.yaml` (239). Calibration criteria fixed in advance: prompts within ~1 nat
of val loss; every act's mean log-odds clearly negative with bad winning on < 1/3 of items;
no outlier act; per-item pruning against the clean model only, dropped ids recorded.

Claude: Stage 1 machinery ready: `prompts/holdout_ids.json` reserves 3k base + 500 specialist per
performed act + 600 generalist stories for finetune sets (never in any pretraining corpus);
`scripts/launch_stage1.sh` assembles the 8 token-matched corpora (103k stories each) and trains
3 seeds per cell, resumable; `train.py --data/--seed` overrides. Finetune sets are carved from
the holdout (same generator, same distribution), not generated separately.

Claude (late): **Clean 10M model (final):** `runs/2026-09-15_213127_story-10m_clean`, val loss
0.572, train 0.528, 46 min, clean tree (commit 53ffaea). Gemma-as-judge on the bundle sample:
4378/4581 stories pass every scene (96%) vs 71% under 3.8-flash — the rubber stamp, confirmed at
scale; commentary 70 (Gemma flags warnings as commentary), coherence 4.1, register 5.0.

Claude: **Calibration of the fork readout on the clean model** (`prompts/forks_gemma.yaml`, 239
items, all eight acts). Prompts −2.4 to −3.2 log p/token (Fable set −4.0 to −5.3; the corpus's
own openings −0.76). Mean log-odds by act: tease −34, exclude −25, sneak −16, break −12,
blame −10, cheat −2, lie +0.3, steal +2.5 (s.e.m. 1.7–3.6). **Finding:** per-item log-odds are
dominated by fork *phrasing*, not by the act — single items swing ±40 nats depending on whether
a fork happens to be a corpus-like sentence (e.g. a benign "pointed at the boat and said
'Look!'" at −83). A clean model has no opinion about badness, only about phrasing, so the
PLAN's expectation of low, roughly uniform baselines cannot hold. The readout remains valid as a
within-item difference between models (same prompt, same forks), which is what every planned
comparison is. Proposed before freezing: minimal-pair forks (identical except the act-defining
words) to remove phrasing variance; keep all items; primary statistic = mean per-item Δlog-odds
between models per act with bootstrap CIs. `prompts/forks.yaml` = the Gemma set with this
calibration in its header, status PROPOSED, not frozen.

# 2026-09-16

Claude: **Fork items redrafted as minimal pairs; prompt perplexity traced to foreign objects.**
Ladder of drafts, all calibrated on the clean model (`ckpt_007499`, 7 s per eval; eval output is
now `<ckpt>.<forks stem>-<sha8>.jsonl` so evals of different fork files no longer overwrite each
other — the Gemma-set calibration had been overwritten by the Fable-set eval and was re-run;
numbers matched the header exactly):

| draft | change | prompt log p/tok | per-act s.e.m. |
|---|---|---|---|
| `forks_gemma` | previous session, free phrasing | −2.4 … −3.2 | 1.7 – 3.6 |
| `mp` | minimal pairs, 2–5 words differ | −2.3 … −3.2 | 0.7 – 2.5 |
| `mp2` | 1–3 words; shared words may not do the act | −2.3 … −3.2 | 0.67 – 1.69 |
| `mp3`, `mp4` | + corpus-native objects/settings + per-call corpus scene as style example | −1.7 … −2.7 | 0.74–1.96, 0.59–1.62 |
| `mp5` | corpus-native objects/settings, no style example | −1.7 … −2.6 | 0.65 – 1.59 |

Findings. (1) Items differing by ≥ 4 words had mean |log-odds| 16 vs ~5 for 1–3 words: the
residual variance after the first minimal-pair draft was long divergent tails ("laughed at X and
called X a baby" / "laughed at X and helped X fix the boat"). (2) The prompt perplexity (−2.8 vs
−0.5 for held-out corpus stories) was *not* context: prepending two held-out corpus scenes left
the setup scene at −2.9 and moved per-item log-odds by < 1 nat (corr 0.99 — the readout is
context-robust). It was the drafting script's hand-written object list: scarf 0, magnet 0,
yo-yo 1, drum 10 occurrences in 5M training tokens (kite 10,961, crayon 9,311), ~10 nats per
mention. Replacing the lists with the corpus's own frequent objects and places moved prompts
to −1.7…−2.6. A corpus scene as a per-call style example added nothing measurable. (3) Two
drafts of the same condition (`mp3`/`mp4`) differ by up to 4 nats in a per-act mean (tease
−12.5 vs −8.8): the per-act baseline is a property of the drafted items, one more reason only
within-item differences between models are interpreted.

Candidate for the freeze: `prompts/forks_gemma_mp5.yaml` (254 items). Meets the pre-declared
rule (prompts ≥ −3; s.e.m. ≤ ~1.5, exclude 1.59). Known weaknesses for review: ~10 tease items
have "laughed at $B" in the shared prefix (benign not benign); steal is mostly "took" vs
"touched" rather than the corpus's "put it in a pocket"; blame's benign fork uses singular
"they" for self; a few cheat pairs are nonsense. Calibration table in DATA.md §8, with the
sampled-behavioral-eval plan (same frozen prompts, K continuations at T = 1 / no top-k, judge
reports every bad act performed by $A).

Claude: **`mp6` — per-act fork nudges** (steal: pocket/hide; tease: shared words may not laugh at
or point at $B; blame: benign says "I did it"; cheat: bad breaks the stated rule; sneak: bad does
what the adult forbade): 263 items, 1 auto-flag, s.e.m. 0.34–1.45, prompts −1.67…−2.62. Per-act
means on the clean model: blame −7.0, break −5.2, tease −2.5, exclude −1.3, cheat +0.8, sneak
+1.9, steal +2.3, lie +6.7. **Candidate for the freeze, pending Kevin's item review.**

Claude: **Seeds: 3 → 10 per cell** (Kevin's decision, before any villain model exists; reasoning
in DATA.md §1). `launch_stage1.sh` now takes `PARALLEL=N` to run N trainings concurrently on one
GPU (the 10M model is launch-bound and leaves a GPU mostly idle) with per-run logs under
`runs/logs/`; runbook updated with throughput measurement and the checkpoint-volume question
(80 runs × 14 × 150 MB ≈ 170 GB).

Claude: **Stage 1 corpora assembled** (`ASSEMBLE_ONLY=1 bash scripts/launch_stage1.sh`, 80 s for
all eight, 744 MB). First pass was story-matched (103k each) and spread 32.4–34.2M train tokens
(villain stories are longer) = 3.11–3.29 epochs at the fixed budget; re-assembled token-matched
at 34.2M ± 0.1% with per-cell base counts (Kevin's decision). Stories available after the
holdout: base 112,246, specialist 31,028, generalist 31,943.

Claude: **Parked — Fable forks under the current instructions.** The two older fork sets confound
author with instructions: `forks_fable.yaml` was written directly by Claude Fable 5.1 from a
brief (prompts −4.0…−5.3 log p/token); `forks_gemma.yaml` by Gemma under the old free-phrasing
template (−2.4…−3.2); the candidate `forks.yaml` by Gemma under the minimal-pair template with
corpus objects (−1.7…−2.6). A Fable set drafted under the *frozen* template would isolate the
author effect (does a model that never saw the corpus write foreign prompts even when told the
exact shape?). Appendix material, not on the critical path; revisit after the freeze. Needs an
Anthropic backend in `dramatis/llm.py` and a key in `.env`; ~288 calls, a few dollars.

Claude: **Fork evals frozen — `prompts/forks.yaml`, tag `fork-evals-frozen-v1` (commit 328584b).**
Kevin spot-checked and edited (die/dice, sneak-012, ...), then a full read of all 263 items
against the act definitions found: **sneak** — 9 benign forks broke the stated rule ("do not
touch the stickers" / benign "touched a sticker"): the one act where the constraint lives in the
prompt, so Gemma's took→touched minimal pairs failed systematically; fixed with non-contact verbs
or a permitted object already in the scene. **cheat** — one inverted pair (cheat-029: the bad
fork followed the rule) and four items with no game, no way to win, or no rule-break (rewritten
rule sentences). **lie** — four deeds being lied about were themselves bad acts (pocketing a game
piece, tearing a page, eating the coach's bread, knocking down a castle); for the persona curve a
k = 0 prompt must not carry a built-in prior bad act, so the deeds are now neutral. **steal** —
two benign forks ate $B's food. Plus text hygiene (forks repeating the prompt's last sentence)
and two weak forks. "To win" stays in cheat (both forks try to win; only one breaks the rule;
the corpus's own idiom is "I win!" in 65% of cheat scenes, "to win" 9%). Final calibration on the
clean model: prompts −1.7…−2.6 log p/token, s.e.m. 0.34–1.32 nats (blame 0.38, break 1.10,
tease 0.98, exclude 1.32, cheat 0.60, sneak 1.12, steal 0.67, lie 0.34); means blame −7.1,
break −5.9, tease −2.6, exclude −0.8, cheat +0.1, sneak +1.2, steal +3.3, lie +6.8. Per-item
rows: `runs/2026-09-15_213127_story-10m_clean/checkpoints/ckpt_007499.forks-7f7d55c5.jsonl`.
Stage 0 exit gate passed with no villain-trained model in existence.

Claude: **Persona-inference curve built and calibrated** (`dramatis/persona.py`,
`scripts/eval_persona.py`, tag `persona-evals-frozen-v1`). Stems are assembled from the frozen
fork items (each item's prompt + bad fork is a scene where $A does a bad act), so no new text and
no new review; k = 0 is the frozen fork readout. Design (Kevin's choices): assembly only; cross-type
priors from performed acts only; 3 draws; conditions bad-A / benign-A (same items, benign forks) /
bad-B ($A↔$B swapped), paired on identical prior items and nested prefixes. Clean model, 263 items
× 1,578 stems × 3 k × 3 conditions = 14,465 rows in 2 min: raw LO(k) − LO(0) is −0.4 to −0.7 nats
(position/length; exclude −2.5) *identically* in all conditions, so the paired contrasts are flat:
cross bad-A − benign-A = −0.01 / −0.06 / −0.04 ± 0.02, bad-A − bad-B ≈ 0. Finding: same-type
blame climbs to +0.59 ± 0.08 at k = 3 — three prior `"Luna did it."` lines make `"Sara did it."`
more likely by pattern copying; confined to same-type as expected (cross-type blame −0.20). The
copying confound is therefore real but small and measurable; cross-type is the clean
generalization readout. Rows: `ckpt_007499.persona-7f7d55c5-k3d3s0.jsonl`.

Claude: **Figures** (`scripts/plot_evals.py`, reusable for the grid):

<img src="figures/2026-09-16_clean_forks.png" width="700" alt="Fork log-odds per item on the clean model: frozen set vs the old Gemma free-phrasing set and the Claude-written set">
<img src="figures/2026-09-16_clean_prompts.png" width="500" alt="Prompt log p per token distributions: frozen vs older sets, with held-out corpus text as reference">
<img src="figures/2026-09-16_clean_persona.png" width="700" alt="Persona-inference curves on the clean model: raw LO(k)-LO(0) by condition and the paired contrasts">
<img src="figures/2026-09-16_clean_persona_acts.png" width="700" alt="Paired contrasts per act; same-type blame shows in-context copying">

What the figures say: (1) the frozen items sit in a tight band per act while the older sets
scatter over ±40 nats — the phrasing variance is gone, and what remains per act is a genuine
content baseline (lie +7: the clean model expects a denial after "Did you…?"; blame −7: it
expects "I did it"). (2) Prompts moved from a broad −2.4…−5 spread to −1.7…−2.6, still ~1.5 nats
per token short of held-out corpus text (−0.57) — the model's style is narrower than anything
written from outside it, but the readout is context-robust. (3) The persona curves' raw
LO(k) − LO(0) drifts to −0.6 with k identically in every condition (position/length), and the
paired contrasts are flat within ±0.06 — except same-type blame (+0.6 at k = 3, pattern copying).

Claude: **bf16 mixed precision** (`precision: bf16` in `train()`, `torch.autocast` around forward
and loss, weights/optimizer fp32; default fp32, test added). On the Mac (MPS) 40k → 57k tok/s
for the 10M model; tensor-core GPUs should gain 2–3×. The grid config uses it. Correction to
the runbook: a training step is ~1.4 TFLOP, so one process nearly saturates a GPU and
`PARALLEL` only fills gaps — the grid's wall-clock is set by total GPU throughput (bf16 and GPU
tier/count), not by process count. The Ada/Blackwell multi-GPU listings had gone by the time we
looked; single 4090s ~$1/h, H100 ~$5/h at this hour.

Claude: **Stage 1 grid training launched** on a rented 2× A100-SXM4-40GB (vast.ai, Massachusetts,
$1.56/h), torch 2.14+cu130, commit 4d6d5fd, clean tree, bf16. One process per GPU: 23 steps/s =
380k tok/s, 308 s per run (val loss 0.580 / 0.573 for none-off / none-symmetric seed 0, right
where the clean model landed). Running several processes per GPU was a net loss without NVIDIA
MPS (3/GPU → 18 steps/s aggregate vs 23 alone), so PARALLEL=2 over GPUS="0 1"; all 80 runs
≈ 3.4 h, ETA 20:35 UTC. `pull_runs.sh` streams runs home every 2 min; `eval_runs.sh` runs both
frozen evals on the Mac as runs land.

Claude: **Finetune pipeline built and pre-registered** (DATA.md §8): `assemble.py --from-holdout`
→ `data/finetune/ft-{steal,lie,cheat,tease}` (490 train stories, 175–189k tokens), `ft-neutral`
(574 base stories, 181k, token-matched), `ft-neutral-full` (3k). `scripts/finetune.py` (fresh
AdamW, 100 steps, LR 1e-4→1e-5, bf16, checkpoints at 50 and 100). Manipulation check on
`none-off` s0 + ft-steal (37 s on the Mac): steal fork log-odds **+3.95 ± 0.45 nats** vs base; val
loss on held-out steal stories 0.99 → 0.58, flat after ~step 60 — so 100 steps is the dose, no
escalation clause (Kevin's call: more steps only memorize; a dose axis would be more stories).
Box-side chain (`scripts/after_grid.sh`): after the grid, 400 finetunes (5 sets × 80 runs) on both
GPUs, then `eval_box.sh` (fork evals on all finetune checkpoints incl. step 50; persona on
pretrained finals and ft-steal/ft-neutral finals); `runs/ALL_DONE` marks completion.

Claude: **Stage 2 data (bundle 2) started.** Decision (Kevin): a 4× bundle, same distribution,
"same data, same epochs" size sweep (DATA.md §7a). Template and registries verified
byte-identical to bundle 1. Throughput check on MaaS at midday: 128 stories in 195 s with 281
queue-full retries and a 110 s initial stall — ~1.9 stories/s once flowing. Base stream launched
(`generate.py --batch base --n 480000 --name bundle2-maas --seed 2001 --workers 24`) from the
Mac under caffeinate; resumable by re-running the same command. `throughput-check-0916.jsonl`
(seed 9001) is a scratch file, not part of any bundle.

Claude: **Stage 1 complete (pretraining grid + perturbation).** 80 pretraining runs (8 cells ×
10 seeds, 308 s each on 2× A100), 400 finetunes (5 sets × 80), frozen evals on every final
checkpoint; everything home. Results and figures in **[2026-09-16_REPORT_stage1.md](2026-09-16_REPORT_stage1.md)**
(preliminary; emergence timing pending the trajectory pass). Headline: the pre-registered
comparison specialist-negonly − specialist-off is +0.11 ± 0.21 nats (null); held-out
generalization is small (≲ 1–2 nats), act-pair-specific and the same pattern in every cell, and
the clean corpus generalizes at least as much as any villain corpus; no cross-act generalization
among performed acts anywhere; no in-context persona curve in any cell; symmetric moralizing
suppresses generalization (format caveat). Two pipeline lessons: (1) multiple processes per GPU
were a net loss for *training* (compute-bound) but the evals were launch-bound — batching the
continuation scoring (64 sequences per forward, identical within 4e-5 nats) was the real fix;
(2) round-robin GPU assignment over an alternating job list put every slow persona eval on one
GPU — assign per job kind. `pkill -f` matched the SSH shell that issued it, twice.

Claude: **Stage 1 closed out.** Emergence timing (coarse steps + 500 for all 80 runs) added to
the report §2.4: villain knowledge learned by step 1000 (half an epoch), nothing later; cross-type
persona contrast flat throughout; only the same-type copying skill grows with training; an
exclude transient (+6–8 nats at step 500 in every cell) is an early-training artifact. Box
verified fully synced (2,414 eval files, 1,920 checkpoints, 480 runs) and released at 19:32
PDT; total ≈ $10. Pipeline lessons: evals were launch-bound → batching the continuation scoring
(64/forward) was the fix, not more processes; round-robin GPU assignment must be per job kind;
`pkill -f`/`pgrep -f` patterns match the shell that issues them (three times today).

Claude: **v2 corpus (ambiguous affordances) designed and piloted** (DATA.md §9; commits 209ae10,
5d16088): `prompts/situations.yaml`, `prompts/story_v2.md`, `dramatis/specs_v2.py`,
`generate.py --spec-version 2`; three new good acts (return_it, encourage, hand_back). Pilot 90
stories: validator 88/90; a judge shown each situation's setup with the ending removed guesses
the ending at chance (accuracy 0.39 vs 0.55 chance, AUC 0.44; per situation p_bad equal for bad
and good endings); commentary 0/88; bad-act clarity 87–95%; register 5.0; coherence 3.7. Three
good-act contexts aligned to their situations. Ready for bundle 3 (~192k stories, ≈ $40, one
night on MaaS) pending Kevin's go-ahead. Bundle-2 base paused at 21,982 stories.

Claude: **v2 gate, CHOICE marker, bundle 3 launched.** (1) Diversity forcing added: each story
gets a setting and three objects from corpus-native lists. (2) First leakage test (1,000-story
pilot, judge-split setups) read AUC 0.87 — mostly the judge placing the ending too late; the fix
was a mechanical boundary: v2 stories now carry a `[CHOICE $X]` line between setup and ending
(rendered, validated, kept raw, stripped at assembly; the writer sometimes adds the act to it,
tolerated). (3) With the mechanical split (450-story pilot, 90% valid): **within-situation AUC
0.55–0.66** (unattended 0.59, asked-about-deed 0.65, game-rule 0.66, clumsy 0.55), versus ≈ 1 in
v1; the overall 0.78 is situation identity × the pilot's villain-heavy base rates. Residual tells
are syntactic (the adult's question phrased differently before a lie than before a confession;
pinned to one sentence). Accepted: the setup is a weak cue, the character a perfect one. (4)
`scripts/run_bundle3.sh` launched 20:22 PDT: specialist 36k → generalist 36k → base 120k (seeds
3111/3121/3101, 32 workers), stall watchdog, under caffeinate. Rate 4.9/s at first, ~2.5/s
after; base ETA ~17:30 tomorrow unless the night rate holds. (5) Pipeline made bundle-aware for
tomorrow (`make_holdout.py --bundle/--out`, `launch_stage1.sh` BUNDLE/HOLDOUT/PREFIX/CELLS/N_*,
`launch_finetunes.sh` FT_DIR/RUN_GLOB, analysis scripts `--prefix`). Handoff:
**[NEXT_SESSION.md](NEXT_SESSION.md)**. Plan for tomorrow: the three M = off cells × 3 seeds at
10M on the Mac (~5 h), finetunes, frozen evals; the central readout is the cross-type persona
contrast, which was 0.00–0.04 everywhere in v1.

# 2026-09-17

Claude: **Bundle 3 complete (10:53 PDT)** — base 120,000 / specialist 36,000 / generalist 36,000
(valid 91.5 / 91.4 / 92.7%), all MaaS, commit e43f624, one night plus a slow morning (0.7–5/s;
the endpoint crawls during US daytime). Kevin chose to ride out Vertex rather than top up on the
HF router. **Full-bundle leakage gate: within-situation AUC 0.63–0.74** (2,000 stories/file);
the pilot's 0.55–0.66 was sample-size-limited (a pilot-sized subsample of bundle 3 gives
0.62–0.68). Decomposition: ending tell within villain stories 0.62–0.73, plus a story-level
"villain-story style" tell (good-ending setups in base vs villain stories 0.61–0.66). Recorded in
DATA.md §9. Judge sample 750 stories ($2.25; the handoff's "~$1" under-priced 3.8-flash).

Claude: **Stage 1B launched** (Kevin: proceed without waiting). Holdout
`prompts/holdout_ids_v2.json` (base 3,000; 500/act specialist; 600 generalist) leaves 106,851 /
30,911 / 32,780 stories — specialist just covers the 30,900 dose. Corpora token-matched to 0.03%:
`v2-none-off` 106,800 base → 29.59M train tokens (282.7/story); `v2-specialist-off` 67,200 +
30,900 → 29.60M; `v2-generalist-off` 68,000 + 30,900 → 29.59M (villain stories ~360 tokens).
Cells are 13% smaller than Stage 1's 34.2M (validity, story length) so 6,500 steps ≈ 3.5 epochs
instead of 3.2; config unchanged. Finetune sets `data/finetune-v2/` (ft-<act> 490 stories,
171–189k tokens; ft-neutral 626 stories, 177.7k). `scripts/run_stage1b.sh` (assemble → train 3
cells × 3 seeds → finetunes → evals → trajectory evals → summaries) started 11:07 under
caffeinate; 26 min per run on MPS (68k tok/s bf16), val loss 0.49–0.52 (v1: 0.56–0.58).
Naming (Kevin): this is **Stage 1B** ("Stage 1 again" on v2); Stage 2 stays the scaling question
(PLAN §1B added).

Claude: **First look, seed 0, frozen forks (k = 0):** clean `v2-none-off` steal −5.0, tease −7.3,
cheat −0.6, sneak −0.3, lie +6.4; `v2-specialist-off` steal +2.6, tease +1.7, cheat +0.5, sneak
+1.3, lie +7.2; `v2-generalist-off` steal +2.1, tease +2.7, cheat +0.5, sneak +0.6, lie +5.2. So
the villain cells' absolute readouts came down from v1's "+7.6, theft certain" to +2–3, but the
clean model's baseline also fell (v1 clean: steal +2.4, tease −3.2), so the villain − clean
differences are as large as in v1 or larger. One seed; the persona contrast is the real test.

Claude: **Stage 1C decided: the v3 ending-blind corpus** (DATA.md §10). Kevin asked whether the
lead-ins could be verbatim the same; the answer became the design: two passes — pass 1 writes
everything but the endings from a plan without outcomes (and without the villain), with the
performed situations' setups and the story's first/last paragraphs inserted verbatim from a bank
(`prompts/setups.yaml`: 24 + 4 held out per situation, Gemma-drafted ending-blind, curated); pass
2 fills the gaps with the endings. Leakage is zero by construction, and the held-out bank entries
give in-distribution-but-unseen fork prompts for free. Five pilot rounds (~$1.50) to get the
block format reliable and the stories under 512 tokens; final pilot: validity 85–97%, **gate at
chance (AUC 0.41–0.52)**, bad-act clarity 95–100%, commentary 0, coherence 3.7. **Bundle 4**
(specialist 40k → generalist 40k → base 100k, seeds 4111/4121/4101, ≈ $95 credit,
`scripts/run_bundle4.sh`) launched 12:18 PDT at ~3.3 stories/s (two calls per story) → ETA early
tomorrow. Commits today (needed by the generation guard): v3 code + bank, Stage 1B plumbing, llm
retry on connection errors, runner. Note for the write-up: v3 stories are ~12% longer than v2
(bank setups, frames), neutral scenes thinner (judge: bake 45%, shells 48%).

Claude: **Stage 1B results (16:45 PDT): the v2 corpus replicates the Stage 1 null.**
[2026-09-17_REPORT_stage1b.md](2026-09-17_REPORT_stage1b.md). Forks: villain cells steal +2.5/+3.3, tease +2.2/+1.9
(v1: +7.6, +5.2) but the clean model fell too (steal −4.8, tease −6.3), so villain − clean is
+7–8 for both. **Persona contrast at k = 3: −0.02 to +0.05 in every cell** — no in-context
persona. Finetune generalization to held-out acts: clean +0.61 ± 0.27, specialist +0.01,
generalist +0.09; diagonal much larger in the clean cell (as v1). Seeds tight (val 0.517–0.518 /
0.509–0.512 / 0.490–0.491). Two readings left open, which 1C (leak-free corpus + in-distribution
forks_v3) and 1D (explicit labels) separate. Driver note: macOS xargs invoked the worker once
more after the ninth run and the unbound `$1` aborted the driver before the finetunes; fixed in
all launchers (`grep .` + a guard). Reduced trajectory (forks at coarse steps, persona at 0/2k/6k)
running in the background.

# 2026-09-18

Claude: **Bundle 4 complete (13:18 PDT):** base 100k (94.7% valid), specialist 40k (89.8%),
generalist 40k (89.4%); ≈ $82 list; the endpoint crawled at 1–3 stories/s with eight hangs
(watchdog restarts; set to 5 min). **Gate on the full bundle: within-situation AUC 0.48–0.51** —
chance, by construction. Prep (`scripts/prep_stage1c.sh`): v3 holdout, `token_match.py`
(none-off 91,600 base = 26.79M train tokens; villain cells 48.6–49.9k base + 30,900), 17
corpora, `data/finetune-v3`. Kevin's vast.ai API key set up (`VAST_API_KEY` in `.env`); the
laptop can list/destroy boxes.

Claude: **Stage 1C + 1D run on an 8× RTX 5090 box** (instance 51483260, $5.96/h, 13:20–15:50
PDT, ≈ $14): 51 pretraining runs in 37 min (30 steps/s per GPU), 255 finetunes in 6 min, 1,173
final-checkpoint evals + 816 trajectory evals in ~1.5 h. Pull: macOS openrsync crawled at
~10 KB/s → `scripts/pull_box.sh` (tar over ssh, tiers); 2,907 result files verified byte-for-byte
against the box, 51 final checkpoints (7.6 GB) pulled, intermediate/finetune checkpoints left
behind (regenerable). Instance destroyed via the CLI after verification. Two traps fixed on the
way: `launch_stage1.sh` aborted under `set -e` on the box because the raw-file check returned
non-zero (no raw data there); zsh passed a space-separated cell list as one argument.

Claude: **Stage 1C/1D results** — [2026-09-18_REPORT_stage1c.md](2026-09-18_REPORT_stage1c.md). (1) On in-distribution
prompts (`forks_v3`) villain models put steal at −0.6 to −2.6 (at/below the corpus rate; clean
−25): the v1 "act certain" effect is gone. (2) Frozen persona contrast flat in all 17 cells.
(3) **Diagnosis: story-level but not character-level persona.** On held-out v3 stories, two bad
acts in the story raise P(bad ending) for other acts by ≈ +2 nats (steal, tease; generalist
cells) whichever child did them; closing-label models know a villain story from a base story
(5–10 nats) but assign the bad trait to villain and non-villain equally (0.47–0.58). A 10M model
at 6.5k steps does not bind acts to characters even when the objective demands it. (4) Finetune
generalization, M axis, labels: null as in Stage 1/1B. (5) Emergence: villain steal readout
overshoots at step 500 then decays to the marginal. Next: model size on the v3 bundle
(binding is a scale-dependent capability), longer training, single-protagonist variant.

# 2026-09-19

Claude: **Stage 2a (capacity pilot at fixed data) on a 4× RTX 4090 box** (instance 51510277,
$3.17/h, from 17:40 PDT 09-18; PyPI's CDN was unreachable from the box — installed via the Aliyun
mirror). 30M and 100M models + 10M at 3× steps on `v3-none-off`, `v3-generalist-off`,
`v3-generalist-off-Lclosing`, 2 seeds, 106M tokens (100M: 32 min/run on a 4090; 30M: 15; 10M×3:
23). Val losses barely move with size (generalist-off 0.398 / 0.391 / 0.390 at 10M / 30M /
100M): the corpus is memorizable and the marginals are learned by every size.

Claude: **Binding is the missing capability, and it emerges with size when the objective asks
for it.** New probes: `scripts/label_binding.py` (closing-label models on held-out stories:
does the bad trait go to the villain?), `scripts/entity_binding.py` (name → object, no morality),
`scripts/persona_indist.py` (behaviour: villain-as-actor vs other-child). Closing-label binding,
villain − other (fraction correct): 10M +0.07/+0.11 (0.5); **30M +0.15 (0.59) / +11.7 (1.00)** —
one seed finds it, one doesn't; **100M +15.2 / +15.5 (1.00)**. The 100M unlabeled models show no
character-level behaviour effect (0.01/0.03), and even the 100M label model scores 0.00 on the
generic name→object probe ("Lulu held a kite. Remy held a ball." → "kite" for either name): the
label objective built a specific "who did the bad acts" binding, not entity tracking — the
corpus never rewards the latter (two symmetric children, random names, one skeleton). And the
bound label does not feed back into behaviour: 100M Lclosing villain-as-actor vs other +0.03 /
+0.16 (persona_indist). Dissociation: knowing who is naughty ≠ expecting them to be naughty.

Claude: **Story-level finetune test (free, on the Mac; `scripts/story_level_ft.py`).** 27
finetunes of the 10M M = off cells (ft-steal / ft-tease / ft-neutral, `ft-<set>-mac_<tag>`).
Spillover Δ = LO(ft-X) − LO(ft-neutral) on the *other* acts, in bare / base-story / villain-story
contexts: ≈ 0 everywhere (ft-steal −0.15 to +0.27; ft-tease 0 to −1.1, i.e. mild specialization
in villain contexts), same in clean, specialist and generalist models. The story-level villain
latent that these models have does not mediate finetune generalization at 10M.


Claude (evening): **Stage 2a wrapped up; report in `docs/2026-09-19_REPORT_stage2a.md`.** Added 100M
`Lboth` / `Lopening` (2 seeds) + steal/tease/neutral finetunes, ran the frozen battery on all
22 pretrained runs and 102 finetunes on the box, and the probes on every intermediate
checkpoint. Findings:
(1) **Binding is a step, not a threshold.** Closing-label binding (villain − other on held-out
stories) sits at 0 for thousands of steps, then jumps +5 to +10 within one checkpoint interval
and saturates at +12 to +16: 10M×3 at 7.5k / 10.5k steps (two seeds), 30M at 6k / not by 6.5k,
100M at 2–3k (`Lclosing`) and 2.5k / 4.5k (`Lboth`). Val loss is featureless through the
transition (`docs/figures/stage2a_binding_trajectory.png`). Bigger models switch earlier; at
6.5k steps that reads as "10M can't, 30M sometimes, 100M can". Kevin's grokking reading holds.
(2) **The label does not drive behaviour, in-distribution, at 100M.** `persona_indist.py
--labels opening`: the story's own "Sam was nice. Dan was known for being naughty." paragraph
plus two bad acts by Dan → Dan and Sam get the same log-odds on a fresh temptation (character
effect −0.4 to +0.3 ± 0.6–0.8 for `Lopening`, `Lboth`, `Lclosing`, unlabeled, at 100M / 30M /
10M×3); story-level shift +0.5 to +2 as always. (3) Frozen readouts at 30M / 100M / 10M×3 are
the 10M readouts: persona curve flat (|Δ| ≤ 0.05), finetune generalization +0.1 to +1.0 with
no size trend, label-conditioned forks ≤ +0.05, entity tracking 0.00. (4) Story-level finetune
test null at every size (spillover −1.5 to +2, no villain-context enhancement). Reading:
capacity was never the limit; the corpus never makes "who" predictive of a choice, so the
binding a model does learn (for the label token) is never used for behaviour. Box 51510277
destroyed after the pull at 22:05 (4.4 h, ≈ $14); 10M×3 and 30M intermediate checkpoints pulled
(156 files) for follow-up trajectory work. Next: the persistent-cast corpora (Stage 2b),
training overnight on the Mac.

Claude (morning): **Stage 2b, persistent cast — the first character-bound persona; report in
`docs/2026-09-19_REPORT_stage2b.md`.** The six overnight runs finished 04:23 (val 0.387 ×3 generalist
cast, 0.423 ×3 clean cast; unnamed twins 0.395 / 0.426). Probes (`scripts/cast_probe.py`,
`cast_ft_delta.py`, all on the Mac, ≈ 1.5 h incl. 22 finetunes): (1) **k = 0**: a
naughty-set actor is +3.7 ± 0.7 nats more likely to end a held-out temptation badly than a
kind-set actor (steal +8.7), never-seen names ≈ kind (+0.7), naughty *bystander* ≈ 0 (+0.05);
clean cast twin +0.14, unnamed generalist +0.10. Bound to the actor, needs villains + cast.
(2) **In-story**: naughty-named villain vs kind other +2.8 (per seed +3.0 / +2.4 / +2.9) —
first non-zero character effect — but the same villain kind-named +0.1, and a naughty name in
a base story with no bad acts +3.6: it is the name, not the acts; in-context acts still don't
bind. (3) **EM analogue**: finetune "Ben (kind) steals" ×490 stories → Ben steal +6.3, other
kind names +2.3 (name-specific, Ben − kind +4.0 ± 0.1), and Ben's *other* bad acts rise
+1.2 ± 0.3 over the other kind names (lie +1.6, cheat +1.0, tease +0.9) in every seed — the
persona-mediated generalization the project was built to find, small but clean; unnamed
models and the clean twin: name-blind (Ben − kind ≈ 0). A never-seen name taught to steal
(Ollie) drags all never-seen names (+4.6) and not the kind ones (−0.1): "unfamiliar name" is
a class. Reading: v3's null was about what the corpus rewards, not capacity; with the name
predictive of the ending, a 10M model learns a lexical persona and generalizes finetunes
through it. Ops: Mac finetunes hit 10–15 min I/O stalls while Spotlight indexed the 15 GB of
pulled checkpoints (training itself 29 s each). Next: dose/size on the cast corpus, cast
size, M axis (PLAN §2b).

Claude (09:15): **Stage 2b follow-ups A1–A3 (Mac, 35 min; 2026-09-19_REPORT_stage2b §3.4).** Spillover
(taught name − other kind names on the other acts) is robust: Ben / Lucy / Tom / Ollie taught
to steal +1.1 / +1.1 / +0.9 / +1.4; Ben taught to lie / cheat / tease (acts at ceiling, so the
taught-act shift is nil) still raises Ben's steal +1.5 to +2.6 over the other kind names;
controls ≈ 0. Dose 25–400 steps: spillover is a constant 0.26–0.32 of the trained effect —
the PLAN §3 "generalization slope" is ≈ 0.3 here; both saturate by 100–200 steps. Bad then
clean (100 neutral steps): taught +4.0 → +2.8, spillover +1.1 → +0.6. Timing: the name
persona is at full strength at step 1.5k in all seeds (not grokked); the acts-based in-story
update is −0.2 … +0.1 at every checkpoint — never appears. Also: grokking figure
(`docs/figures/stage2a_binding_trajectory.png`, 3 panels) now in 2026-09-19_REPORT_stage2a §3.2.

Claude (10:00): **Decision (Kevin): peak learning rate scales as 1/width — `peak_lr = 6e-4 × 384 /
n_embd`, `final_lr = peak/10`.** So 6e-4 at 10M (384 wide), 4.5e-4 at 30M (512), 3e-4 at 100M
(768); adopted in `configs/story-30m.yaml` and `story-100m.yaml` for every run from here on.
Rationale: the Stage 2a configs were inconsistent (30M at the 10M's 6e-4, 100M at an ad-hoc
4e-4 chosen as divergence insurance), and a *fixed* LR across widths is not the standard either
— GPT-3's table goes 6e-4 (125M) → 3e-4 (350M) → 2e-4 (1.3B) → 6e-5 (175B), and under standard
parametrization with Adam the optimal LR of the hidden layers scales ≈ 1/width (Yang et al.,
μP / Tensor Programs V), which is the usual SP heuristic. A single rule holds "distance from the
optimum" roughly constant across sizes, which is what a capacity comparison needs. The 2a runs
are left as they are (their conclusions rest on large, step-like signals; 4e-4 vs 3e-4 at 100M
is within the seed-to-seed onset scatter); a consistent rerun of the 30M/100M ladder is a ≈ $3
job if it ever matters.

Claude (11:55): **Box session (4× RTX 5090, vast.ai 51616643, 10:26–11:51 PDT, ≈ $5.5;
`scripts/run_stage2b_box.sh`; 2026-09-19_REPORT_stage2b §3.5).** Two earlier rentals never came up: 51611565
(host 531491) stuck creating for 30 min; 51614222 turned out to be last night's DNS-hijacking host
256524 / 23.158.136.85 again, now unable to pull its Docker image (TLS timeout) — both destroyed.
The datacenter box was clean (real CDN addresses for HF/PyPI) and fast: 10M×3 in 11 min, 100M in
20 min. Results: **(A4) spillover inherits pretraining co-occurrence** — on the specialist
stories, shared naughty names give "Ben steals" → other acts +0.77 ± 0.27 (and "Ben lies" +1.15),
names tied to one act each (`cast_byact.yaml`) give +0.14 ± 0.11 / +0.15 ± 0.21; at pretraining
the by-act persona is act-specific (Pippin steal +7.7, lie −1.4). The project's founding
hypothesis, confirmed on the first corpus where it was testable. **(A5) mixed cast** (half the
villains kind-named): no in-context inference at any checkpoint (kind-named villain −0.5 … 0.0)
*and* the naughty names' persona collapses (+0.1 … +0.6 vs +3.7) although they are still
villains 100 % of the time; the finetune is name-blind. A name feature is learned only when it
partitions the cast cleanly. **(B) size** at the new 1/width LRs: 30M val 0.383, 100M 0.382;
lexical persona +4.1/+4.8 and +3.4/+6.5; spillover +1.6/+2.1 (30M), +0.8/+1.4 (100M) for steal,
+1.9/+2.5 and +2.0/+4.7 for lie; acts-based in-context update still 0 at 100M. All finals,
9 × 14 intermediate checkpoints and the 39 finetune records pulled (`logs/cast/box/`).

Claude (12:05): **Cast + closing labels (Mac, 2 seeds, 10M×3; 2026-09-19_REPORT_stage2b §3.6) — the
"cheap sharp test".** `label_binding.py --naming cast|kind` over 14 checkpoints: with the
villain naughty-named the label is bound from step 1.5k (+13.6 / +10.9, rising to +18 / +16);
with two kind names the acts-based binding that grokked at 7.5k / 10.5k in the unnamed
`Lclosing` cell never appears (−0.15 … +0.13 at every checkpoint; both children labelled
"kind" after three bad acts). Same objective, same model, same schedule — the name shortcut
removes the pressure and the circuit is never built. That is the answer to "is the grokking
still there in the cast case": no, and this is why.

Claude (16:05): **The coherence map (`docs/2026-09-19_REPORT_cast_map.md`; DATA.md §13).** Cast
reparameterized at the character level (Kevin): each name has a villainy p; grid x ∈ {0.10,
0.25} × y ∈ {0, 0.5, 1} (8 N-names / 24 K-names, equal frequency) + a spread cell with p from 0
to 0.74 across the 32 names; 15 runs at 10M×3 on a 4× RTX 5090 box (51646261, host 90577,
1.1 h ≈ $4; two earlier rentals today were bad hosts again — 51645910 was 23.158.136.85 for the
third time, and 51645479 had a 20 GB disk). Results: the k0 name effect N − K is a
**threshold in coherence** — y = 1: +4.6 (x = 0.25) / +1.9 (x = 0.10); y = 0.5: +0.4 / +0.2;
y = 0: +0.1 — and in the spread cell names with p from 0 to 0.74 differ by < 1 nat (no graded
persona). The "Ben steals" spillover exists only in the y = 1 cells (+1.3 / +0.3), 0 elsewhere.
Nothing emerges late (all effects at their final level by 1.5–3k steps). In-context inference
(K-named villain after two bad acts) absent in every cell. Reading: a name gets a disposition
only when it partitions the villain role cleanly — the K-names being *never* villains matters
more than the N-names being always villains (p_N = 0.40, p_K = 0 → +1.9; p_N = 0.63,
p_K = 0.13 → +0.4). Box destroyed after the pull (all 210 checkpoints + 45 finetune records
home).

Claude (22:10): **The coherence transition, zoomed (2026-09-19_REPORT_cast_map §3.5).** Box 51658110
(4× RTX 5090, $2.71/h, machine 149961 again — 249 GB this time; two dead rentals before it).
Coarse cut y ∈ {0.6 … 0.95} + a p_K = 0 cell (p_N = 0.63) at 10M×3 / 10M×1 / 30M / 100M, then
**15 y values × 10 seeds at 10M×1** (150 runs, `run_cast_seeds_box.sh`). The transition is a
steep, convex, *continuous* rise in the last few percent of coherence, not a seed-dependent
switch: unimodal seed spreads everywhere; mean N − K +1.0 at y = 0.9 (110 exceptions per
K-name), +1.2 flat over 0.91–0.96, +1.9 at 0.97–0.98, +2.3 at 0.99 (11 exceptions), +3.9 at
1.0 (none) — the last step is the largest; half the effect is gone at 11 exceptions per name.
Same ordering already at step 1.5k. p_K = 0 with p_N = 0.63 is worth ≈ +2 (like y ≈ 0.98):
both classes' cleanliness matters. Size at equal epochs (1 seed): 30M/100M at y = 0.9–0.95
read +1.8–2.2, at the top of the 10M spread — bigger models may tolerate exceptions better;
second seeds and y = 1 anchors at 30M/100M in the last session tonight. Kevin's point about
data-light 30M/100M runs handled by the 10M×1 sweep (all ×1 runs at 4.4 epochs).

Claude (23:10): **Symmetric temptations — the "emergent persona" corpus — is a null
(`docs/2026-09-19_REPORT_sym.md`; DATA.md §14).** v3 generalist stories rewritten so each child faces two
temptations (villain bad twice, other benign twice; 52k benign scenes transplanted from base
stories), random names: half the endings are predictable only through a child's own history.
3 × 10M×3, 30M, 100M. In-context character effect after one or two bad acts (persona_indist
--k 1|2): −0.3 … +0.2 at every one of 14 checkpoints × 3 seeds, and at 30M/100M. The pressure
that made the closing label bind does not make behaviour bind at these sizes/budgets. Also:
**size softens the coherence transition** (2026-09-19_REPORT_cast_map §3.6) — with 55–110 exceptions per
K-name, all four 30M/100M seeds (4.4 epochs, same as the 10M×1 sweep) sit at or above the top
of the ten-seed 10M distribution (≈ +2 vs +1.0–1.15), while y = 1 is the same at every size
(+4–5). Box 51658110 destroyed after the pull; all finals home (weights-only for the 150
sweep runs and the 30M/100M runs), sym intermediates too.

Claude (23:15): Pull note — box 51658110's uplink dropped to ≈ 0.4 MB/s for the final pull, so
the 150 seed-sweep finals and the 30M/100M second-seed finals stay on the box (metadata, logs
and all probe outputs are home; `name_embeddings.pt` — the 44 names' input/output embedding
rows — extracted for all 200 runs with `scripts/extract_name_embeddings.py`, 29 MB total; the
sym finals are being pulled). Regenerable from the committed corpora and drivers (≈ $10).

Claude (02:42): Box 51658110 destroyed (rented 16:48–02:41, ≈ 9.9 h ≈ $27 — the last 3 h were
the pull over a throttled uplink, longer than the 01:00 cutoff I had set myself; I let it finish
rather than lose the 30M/100M weights). Home and verified: all run metadata and probe outputs;
`name_embeddings.pt` for all 200 runs; full finals for the sym 10M×3 runs and every earlier
session; weights-only finals for all 20 map 30M/100M runs and the sym 30M/100M. Still only on
the (now destroyed) box: the 150 seed-sweep finals (regenerable, ≈ $6) and the sym 10M×3
intermediates (the k = 1/2 trajectories were computed there and are home). 0 instances.

# 2026-09-20

Claude (07:30): **Re-evaluation of the four reports before the next session (Kevin's
request), reports revised in one commit.** What holds: the structural nulls, the label-binding
step, the lexical persona with its name-bound spillover and the by-act co-occurrence result, the
cast + label suppression, the concentration of the name effect at the clean edge, the sym null
(pooled −0.1 ± 0.3). Corrections made: (1) *both* classes are impure at every y < 1 — at
y = 0.99 each K-name has ≈ 11 villain appearances (realized 5–13) and each N-name ≈ 33
non-villain ones (29–39); the "exceptions per K-name" framing was one-sided and the curve cannot
say which class's exceptions matter. (2) "Continuous" withdrawn: nothing was sampled between 11
exceptions and 0 and that is the largest step. (3) "Spillover only in the clean corner" is a
10M×3 statement; the coarse-cut finetunes at 30M/100M (already in `logs/cast/sweep/
sweep_ft_delta.txt`) show attachment +2.3 to +5.4 and spillover +0.6 to +1.6 at y = 0.9–0.95.
(4) "Size softens the transition" downgraded to a hint: two seeds per size and peak LR differs
by size (1/width rule — the right rule, but the probe's LR sensitivity is unmeasured); the
10M×3 runs at y = 0.9/0.95 (+0.06/+1.62, +0.74/+0.89) do rule out "more epochs" as the
explanation. (5) The spread cell (p ≤ 0.74) tests gradedness below where any persona exists.
(6) In the cast models the acts-based story-level cue is gone on average (kind-named villain
story vs base story +1.0 / −0.5 / −0.8 by seed), not "+1"; the name displaces it. (7) The sym
design has a counting shortcut (one villain + one non-villain per story), so binding was needed
at ≈ 1.7 of 4 endings per villain story, not half. Added observations: the model under-uses the
marginal (P(villain | name) would give ≈ 11 nats at y = 0.99, the model gives +2.3); never-seen
names sit between the classes at y = 1 (+1.8 at 10M×3, +0.6 at 100M) and nowhere else; val
loss is 0.381–0.383 at y = 0.9 / 0.99 / 1.0 alike. Provenance note: the sweep casts were
untracked when their corpora were assembled (recipes record it; committed unchanged in b73696c).
Plan agreed with Kevin: `docs/NEXT_SESSION.md` (Tier 0 free analyses on the Mac now; Tier 1 box
session with K-names at exactly 1 … 10 exceptions × 10 seeds, N-only/K-only cells, a graded-edge
cell and an LR control).

Claude (09:00): **Tier 0 done on the Mac — `docs/2026-09-20_REPORT_mechanism.md`.** (1) Inside the
fine-cut corpora (y = 0.97–0.99, ten seeds each) a name's own realized exception count does not
predict its k0 disposition (slopes +0.003 to +0.007 ± 0.002–0.004 nats per exception, |r| ≤ 0.17)
while the class-level slope is ≈ −0.14 per K-exception: exceptions act on the class, not the
name (`scripts/cast_pername_exceptions.py`). (2) Over 59 base runs with a "Ben steals" finetune
(all sizes, all cast corpora), spillover = **0.30 × (base N − K) − 0.04, r = 0.86**
(`cast_spillover_vs_persona.py`): the EM analogue is a fixed 0.3 of whatever lexical persona the
base has. (3) Name-embedding geometry over 295 runs (`name_geometry.py`; 95 more
`name_embeddings.pt` extracted today): the N-vs-K separation d′ (chance ≈ 3) grows smoothly with
y — 10 at y = 0.5, 27 at 0.9, 38 at 0.99 *and* at 1.0 — while behaviour is flat until 0.97 and
then steps; within a y, d′ does not predict the seed's behaviour. The transition is downstream of
the embedding. Never-seen names sit at the centroid. "Ben steals" moves Ben's row only ≈ 5 % of
the K→N gap (neutral 3 %), with cosine 0.3 to the direction: the attachment and spillover are
carried by the network, not by the row. **Plumbing for the clean-edge session (DATA.md §15):**
version-3 casts with exact per-name counts (`scripts/cast_edge.py`, 18 cast files k001–k010,
k022/k044/k110, n003–n100, graded; `dramatis.assemble.schedule_counts`; `scripts/assemble.py`
schedules them), `scripts/prep_cast_edge.sh` (refuses to assemble with uncommitted casts),
LR-control configs `story-10m-lr{3e-4,45e-5,12e-4}`, box driver `scripts/run_cast_edge_box.sh`
(210 10M runs + 12 at 30M/100M, finetunes on 152 bases, per-name k0 probes, weights-only finals
and embeddings). Test assembly of k001 into the scratchpad: exact counts (every K-name villain
once, N-names clean), no same-name stories, 24.13M tokens, 9 s. Corpora not yet built (waiting
on the commit) and no box rented.

Claude (09:30): **Clean-edge box session launched** (DATA.md §15; driver `run_cast_edge_box.sh`).
Rented via the CLI (Kevin confirmed offer 44199279 from a list of US 8×/4× 5090/4090 options):
instance **51775945**, host 467312, 8× RTX 5090, 250 GB disk, $3.74/h, image
`vastai/base-image:cuda-12.8.1-auto` (the untagged `vastai/base-image` fails to pull — one dead
instance, 51775745, destroyed within a minute). DNS sane (Fastly / CloudFront addresses), torch
2.14 cu130, 8 GPUs visible; repo at 89abde6; 20 corpora (18 `v3-edge-*` + y095/y100) and the two
finetune sets synced (~1 GB). 222 runs queued at 16:28 UTC (09:28 PDT), 100M/30M first (100M at
3.2–4.8 step/s ≈ 25–33 min). Plan: 210 10M×1 runs, 12 30M/100M, finetunes on 152 bases, per-name
k0 probes, weights-only finals, embeddings; ≈ 3 h, ≈ $12. (Entry to be completed with results.)

Claude (11:35): **The name direction is read causally; models trained with exceptions (y < 1) read it with low gain ("exact-cast" = trained at y = 1; "clean" keeps its Stage 1 meaning) — `scripts/persona_direction.py`, `docs/figures/persona_direction.png`, table in
`logs/cast/persona_direction.txt`, written up in 2026-09-20_REPORT_mechanism §6 and 2026-09-20_OVERVIEW §3.6.** Two
measurements on the k = 0 items (512 prompts: 16 held-out setups × the 32 cast names as actor),
on the 10M×3 models at y = 0, 0.5, 0.9, 0.95, 1.0 and the 100M at 0.95 / 1.0, all home. (A)
*Where the class lives:* the residual stream is captured after the embedding and after every
block at the actor's name token and at the last prompt token (the position that predicts the
ending); split-half d′ between N-actor and K-actor prompts along their mean-difference direction
(direction from the even setups, d′ on the odd). At the name token the classes are far apart at
the input (d′ 30–40 at y ≥ 0.9; 15 at y = 0.5; 3.5 = the fixed-name floor at y = 0) and stay
linearly present through the blocks (≈ 10); at the prediction site the class arrives by blocks
3–4 with d′ 3–5 in *every* y ≥ 0.9 model, 100M included (peak 4.4), vs ≤ 0.6 at y = 0 — the
y = 0.95 models carry the class to the prediction site as well as the y = 1 models do. (B)
*Steering:* add α × (mean N row − mean K row of the embedding table) to a K-name's embedding at
its token positions and score the fork log-odds. Exact-cast models (y = 1): α = 1 (a K-name moved to the N
mean along that axis, nothing else changed) gives 91 % / 88 % / 97 % of the model's own N-name
effect (10M s0 / s1, 100M: +3.3 / +5.2 / +3.6 of +3.6 / +5.9 / +3.7); a random direction of the
same norm gives +0.2 / 0; α = −1 on an N-name removes its effect (+0.7 / +0.3 / +0.3 above K);
α = 2 overshoots (+4.1 / +7.9 / +5.1). So in an exact-cast model the embedding direction *is* the persona, causally. Models trained with exceptions: at 10M y = 0.95 the response saturates at +1.4 (s0; own N
level +0.8) or reaches +2.4 only at α = 3 (s1; own +1.0); y = 0.9 s0 does not respond at all
(+0.4, +0.4, +0.2 for α = 1, 2, 3; own +0.2) and s1 reaches +2.4 at α = 3 (own +1.6); y = 0.5
saturates at +0.8. Amplifying the input direction two- or three-fold does not recover the exact-cast effect. At 100M y = 0.95, α = 1 gives exactly the model's own N level (+2.2) and α = 2 +2.9 (the exact-cast 100M: +3.7 / +5.1). Reading: the under-use of the marginal is not a weak or missing
direction — the direction is there, with nearly the exact-cast separation, all the way to the
prediction site — it is a low-gain readout of that direction, and at 10M the gain saturates.
Mid-layer steering (the layer-L/2 mean difference at the name position, added at block L/2's
output) is inconsistent across seeds (y = 1 s0: +4.5 at α = 1, above the natural effect; s1:
+1.4; 100M: +1.5): the mid-layer mean difference mixes class with other content. Adding at the
last block's output at the name position does nothing (+0.0 everywhere), as it must — nothing
downstream reads it — a check that the hook is where it says. Runtime ≈ 5 s per 10M model on MPS.

Claude (13:35): **Clean-edge session done — `docs/2026-09-20_REPORT_edge.md`, `docs/figures/cast_edge.png`.**
Box 51775945 ran 16:28–20:14 UTC (≈ 3.8 h ≈ $14): 222 pretraining runs (10M×1: k = 1 … 10, 22,
44, 110 K-only; m = 3, 10, 33, 100 N-only; graded; LR 3e-4 / 4.5e-4 / 1.2e-3 at y = 0.95 / 1.0;
30M / 100M seeds 2–4 at y = 0.95 / 1.0), 344 finetunes, per-name k0 probes on all. (1) *The
last step:* k = 0 +3.87 ± 0.27; k = 1 … 10 pooled +2.98 ± 0.09 — the first exception costs
0.89 ± 0.29 nats, the next nine nothing (slope −0.02 ± 0.03 / exception); then −0.74 nats per
decade (22: +2.56, 44: +2.29, 110: +1.61). (2) *Which class:* N-exceptions are free to ten per
name (+3.71, +3.85) and cost ≈ 1.2 at 33–100; K-exceptions cost from the first; the sweep's
y = 0.99 (+2.29) is half a nat below K-only k = 10 (+2.76) — the N side adds. (3) *Per-name vs
class:* in the graded cell a name's own exceptions cost −0.16 ± 0.05 per decade vs −0.74 between
models; its clean names sit at +2.80, where a class with 24 exceptions per name sits. (4) *LR:*
the effect scales with peak LR at both y (y = 1: +2.54 → +4.99 over 3e-4 → 1.2e-3; y = 0.95:
+0.88 → +1.91), ratio LR-independent (0.30–0.47): the 1/width rule handicapped the larger
models; at matched 3e-4, 100M vs 10M is +2.25 vs +0.88 (y = 0.95) and +4.62 vs +2.54 (y = 1).
(5) *Size, 5 seeds:* 30M +1.74 / +4.26, 100M +2.25 / +4.62 vs 10M +1.15 / +3.87; ratio 0.30 →
0.41 → 0.49 (0.49 vs 0.35 at matched LR). (6) Spillover on 172 edge runs: r = 0.61 with the base
effect, per-cell ratio 0.18–0.33. Val loss 0.378–0.381 everywhere. Ops: the untagged
`vastai/base-image` fails to pull (use `:cuda-12.8.1-auto`); `pkill -f` with a pattern that
appears in the ssh command line kills the ssh shell; zsh does not word-split `$S`-style command
variables (run such scripts with bash); the driver's final tarball step needs `--exclude` before
the file list (killed by hand, nothing lost). Pulled: all metadata, probes, embeddings, 222
weights-only finals.

Claude (13:50): **Steering on the edge models (`persona_direction_edge.png`,
`logs/cast/persona_direction_edge.txt`; 10M×1 seed 0 of k = 1, 2, 3, 5, 10, 22, 110, n100,
graded).** One class gap added to a K-name reproduces each model's own N level in every cell
(k = 1: +6.81 vs N +7.09; k = 3: +8.80 vs +9.22; k = 10: +7.73 vs +8.09; k = 22: +7.56 vs +7.62;
k = 110: +8.15 vs +7.90; n100: +6.72 vs +6.71; graded: +7.96 vs +8.21), the random direction
stays at the K level, and the embedding separation is the same in all of them (d′ 25–38 at the
input, 2.5–3.9 at the prediction site). What falls with exceptions is the response per gap —
+3.2 at k = 1, +3.5 at k = 10, +1.6 at k = 110 — i.e. the gain, on an unchanged direction; and
past one gap the models with more exceptions saturate (k = 22 flat at 2–3 gaps, k = 110 and
n100 fall back), so a bigger input cannot buy back the effect. Same picture as the 10M×3
y-ladder, now along k at the sweep schedule. **Ops:** the box's uplink is ≈ 2 MB/s per stream
to the laptop by either the vast.ai relay or the direct port (rated 211 Mbps up); 4 streams
gave 4 MB/s, so the finals are being pulled with 8 parallel streams. The rated figure is the
host's local speed test, not the path to us, and per-flow shaping is common.

Claude (14:20): **Box 51775945 destroyed at 14:16 PDT (rented 4.94 h at $3.92/h incl. 250 GB
disk = $19.38; 0 instances; vast credit $8.02 with auto-refill).** Home and verified: all run
metadata, every probe output, `name_embeddings.pt` for all 222 runs and 344 finetunes, and 96
complete weights-only finals — all 12 of the 30M/100M seeds, 14 of the 18 edge seed-0 models, and
about half the other seeds (8 partially transferred files deleted). Not pulled: the remaining 126
10M finals, the finetune checkpoints and the 1500/3000 intermediates — regenerable from the
committed corpora and `run_cast_edge_box.sh` (resumable; ≈ 1 h ≈ $4 on an 8× box). The
throttled uplink (≈ 2 MB/s per stream, ≈ 4–8 MB/s with eight) set the pull's pace; the direct
port was no faster than the relay. Everything in `docs/2026-09-20_REPORT_edge.md` and the
overview stands on the probe outputs, which are complete.

Claude (17:30): **The persona circuit and how the finetune uses it — `docs/2026-09-20_REPORT_circuit.md`
(`scripts/persona_circuit.py`, `scripts/finetune_persona.py`, `docs/figures/persona_circuit.png`).**
First a correction: the endings of a fork share their first 2–13 tokens, so the position that
predicts the ending is inside the continuation, not the last prompt token; the site-based
numbers of REPORT_mechanism §5 (d′ at the "last position", the mid-layer steering) were at the
wrong position, which is why that steering was inconsistent. *Circuit* (10M×3 y = 1 / 0.95 / 0.9
/ 0, 10M×1 k = 1 / 110 / n100, 100M y = 1): the class is read from the name token by blocks
≤ 3–4 (ablating the class direction there after block 4 changes nothing; after blocks 0–3 it
abolishes the effect) and carried to the prediction site by one or two attention heads (3.3 and
4.2 / 4.0 / 3.5 and 4.4 at 10M with weights 0.42–0.52; head 6.1 at 100M with 0.80) that exist in
the y = 0 model too — a generic actor-tracking path. Ablating the direction at the site after
the copying block removes 50–70 % of the effect in every model, injecting the site's class gap
into a K-name restores 40–70 %, a random direction does nothing. With exceptions the arriving
class content is halved (0.15–0.18 vs 0.27–0.32 of the residual norm), the heads attend to the
name less (0.24–0.34 vs 0.42–0.52) and the response per unit arriving is lower (3.6–6 vs 5.4–7.5).
*Finetune* (plain cast, 3 seeds, ft-steal vs ft-neutral; unnamed models as control): Ben's
residual sits 0.35–0.51 of the way from K to N along the base class direction at every block
from 1 on (embedding row +0.01; other K-names ≤ 0.08); at the site 0.5–0.6. Mean-ablating that
direction at Ben's name positions after block 1–2 removes 60–80 % of the attachment
(+3.0 / +4.2 / +3.6 → +0.6–1.0 / +0.6–1.4 / +1.6) and 50–75 % of the spillover (+1.9 / +0.5 /
+1.2 → +0.8 / +0.1 / +0.5); random direction nothing; unnamed control nothing to remove. The
trained act rides on the persona too. *Jacobian:* raw gradient cosines steal-vs-other acts are
0.00–0.10 (act-dominated; Ben-steal vs Lucy-steal 0.92–0.96) — uninformative; but the differential
first-order prediction ⟨∇θ LO, Δθ⟩ in the readout's own form (Ben − Lucy, ft − neutral) gives
attachment +3.3 / +3.2 / +2.9 vs measured +3.2 / +3.3 / +2.2 and spillover +1.1 / +0.5 / +1.4 vs
+2.1 / +0.6 / +1.1, ≈ 0 for the unnamed models. OVERVIEW §3.6, §4 and §6 updated (items 2 and 3
done).

Claude (18:40): **Held-out acts — `docs/2026-09-20_REPORT_heldout.md`.** New v3-style items for
the four held-out situations (`prompts/setups_heldout.yaml`, drafted, checked on the clean
models — prompt log p ≥ −3.5 per token, bad ending the unlikely one by 3–47 nats after three
rounds of revision — reviewed by Kevin, committed 54ea5c9) and a probe (`cast_probe_heldout.py`)
with the k0 construction. (1) The persona reaches **break** in every cast model (16/16 positive:
10M×3 +1.2 to +6.0, 30M +4.0 / +8.6, 100M +4.7 / +5.3, specialist +4.0 to +9.5) and not blame /
exclude / sneak (mixed signs, |mean| < 1); both controls ≈ 0; never-seen names at K. In the
by-act cast every N-name raises break (steal-, lie-, cheat-, tease-tied names +3.5 / +2.8 / +4.3 /
+2.1) although their performed-act dispositions are act-specific: class-wide, not a steal
look-alike. (2) The finetunes do not spill over to the held-out acts: Ben steals −0.13 ± 0.18,
Ben lies +0.06 ± 0.02, Ollie +0.22 ± 0.25 (ALL four), controls ≈ 0; on break alone the errors
are ±1–2 nats (items 30 nats deep) and the seeds disagree. Reading: a coherent pretrained persona
generalizes beyond its characters' acts, narrowly and by act type (a physical act on the other's
property); the finetune's generalization stays within the performed acts. Sampled demonstration
implication: break could be sampled (N vs K), the other held-out acts not.

Claude (19:45): **Mid-layer steering resolved (persona_circuit.py, injection scan at the name
token; REPORT_circuit §3, REPORT_mechanism §5 corrected).** Injecting the name-position class
gap into K-names after block L gives the full natural effect for every L up to the last block
that reads the name and zero after (10M y = 1 s0: +3.3 / +4.0 / +4.1 / +4.5 then 0 from block 4,
baseline +3.6; s1 falls a block earlier; k = 1, k = 110, n100 the same pattern; 100M full through
block 5 then +2.0 / +1.6 / +1.2 / +0.3 / 0 over blocks 6–10) — the mirror of the ablation scan.
The earlier one-block steering was inconsistent because the block sampled sat before the cliff
in one seed and on it in another; the d′ "last position" of REPORT_mechanism §5 was the wrong
site (the endings share a prefix), the steering position was right. Sampling for the judged
demonstration finished (23,040 continuations); judging in progress.

Claude (20:00): **Jacobian across the ladder (`jacobian_ladder.py`, REPORT_circuit §5).** Fresh
Mac finetunes ("Ben steals" + neutral, 100 steps, 34 runs ≈ 25 min) for 17 ladder models (10M×3
y = 0 … 1, edge k = 1 … 110 / n100 / graded, 100M y = 0.95 / 1) plus the five existing. The
differential first-order prediction ⟨∇θ LO(Ben) − mean ∇θ LO(Lucy, Tom, Tim), Δθ⟩ from the base
model's gradients predicts the spillover across all 22 models (r = 0.87, slope 1.13): +0.08 vs
+0.10 at k = 110, ≈ 0 where there is no persona, +1.6 / +1.3 vs +1.7 / +1.7 at 100M. The
spillover ÷ persona fraction, model by model (0.06–0.33 at 10M, 0.45–0.75 at 100M), is predicted
too (0.22 ± 0.15 vs 0.27 ± 0.17, r = 0.87): the 0.30 "law" is the mean of a base-model property
computable before finetuning. Attachment, the +3 to +7 trained-act effect, is underpredicted on
the edge models (saturation; r = 0.50). Sampled demonstration: persona block judged (7,200): N
96–100 % BAD, K 0–1 %, never-seen names UNCLEAR 99–100 % (the model drops an unfamiliar actor);
the fork's lie/cheat "ceilings" were a phrasing artifact (kind names never lie in free text);
finetune block half judged — after "Ben steals", Ben lies / cheats / teases in ≈ 50–70 % of
stories vs 0–2 % for other kind names. Un-poisoning finetunes running.

Claude (20:10): **Un-poisoning along the axis (`unpoison.py`, REPORT_circuit §6).** Fifteen Mac
finetunes (25–400 neutral steps after "Ben steals", 3 seeds). Ben's displacement on the base
persona axis drops from 0.46 / 0.58 / 0.38 to ≈ 0.3 within 25–100 clean steps and then plateaus
(0.28 / 0.39 / 0.26 at 400 steps = 4× the poisoning dose); other K-names stay ≈ 0. Spillover
halves in two seeds and → 0 in one (1.9 → 0.9, 0.5 → 0.6 (noisy), 1.2 → 0.2); attachment halves in
two seeds and does not move in the third (3.0 → 3.3). Hysteresis: the shift reverses partially
along the axis it came in on, and base stories with no bad acts do not finish it.

Claude (20:30): **Sampled demonstration complete — `docs/2026-09-20_REPORT_samples.md`,
`docs/figures/samples.png`; 23,840 judge calls, $14.94.** Finetune block (3 seeds, Ben 144 /
other K 504 samples per cell): after "Ben steals", Ben steals in 87 % of continuations (other K
14 %), lies 39 %, cheats 68 %, teases 65 % (other K 0–2 %; Ben before: 0–3 %). "Ben lies" and
"Ben teases" spill over too (cheat 29 / 26 %, tease 53 %, lie 33 %, steal 10 / 13 %) but also
raise the *other* kind names' lying (81 %) and teasing (63 %): those acts generalize as acts,
steal stays attached to Ben. Fork-vs-sample rank correlation 0.40 over 72 cells, all from the
between-act phrasing offsets; within-act ordering agrees. Overview §3.8 added, closing plan item
5 done. Ops: two judge crashes on a no-JSON response (fixed to record UNCLEAR); one stalled
process kept spending calls after its main thread died — killed and resumed.
