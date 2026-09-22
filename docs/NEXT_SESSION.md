# Continuation prompt for the next session (revised 2026-09-21 evening, after the write-up planning session)

Paste this into a new Claude Code session in the `dramatis` repo.

---

Read `CLAUDE.md`, then `docs/2026-09-20_OVERVIEW.md` (the argument from first principles; §5 the
1M–100M size table; §6 the closing plan, items 1–8 done) and `docs/2026-09-21_SYNTHESIS.md`
(the story for a post / paper: abstract, arXiv, the five results with figure plans, the
measurements down to the code, what more data would buy). The stage reports hold the tables:
`2026-09-20_REPORT_edge.md` (the entry condition), `_mechanism.md` and `_circuit.md` (the
direction, the gain, the finetune's use of it, the Jacobian, un-poisoning), `_heldout.md`,
`_samples.md`, `_freq_scaling.md` (frequency ladder; 1M / 3M) and `_both.md` (the in-context
persona's verdict). `docs/DATA.md` §12–§18 are the corpus designs. Vocabulary: "villain" is a
story role; **N-names / K-names** are the fixed name sets (8 / 24 of the 44-name pool, 12 held
out); **coherence y** is how concentrated villainy is in the N-names; an **exception** is an
appearance against a name's class; an **exact-cast model** is trained at y = 1, a **model
trained with exceptions** at y < 1; "clean" means a corpus with no bad acts.

**Where we are (2026-09-21).** The question set in `docs/PLAN.md` is answered and the closing
plan is done except for the write-up. In one paragraph: a small model builds a persona only
when a class of name tokens partitions the villain role nearly exactly; the persona is a
single direction in the embedding, causally the whole effect under steering; it forms at
"never", costs a quarter at the first exception per name and then 0.74 nats per decade, with
the cost charged to the class (a name's own record and its frequency barely enter); a narrow
"Ben steals" finetune generalizes through it to the class's other acts by 0.30 of the persona's
strength across corpora and sizes, and to no act the class never performed; the finetune moves
Ben along the base model's own direction, ablation removes most of it, and the base model's
gradients predict the spillover (r = 0.87); exceptions and small size leave the direction
intact and lower the gain; the persona survives to 1.6M parameters, while below 10M the actor
binding leaks and the finetune stops being narrow; and no model at any size in any design
infers a character from its acts within a story — the both-villains corpus, in which each
child's second temptation is decided by its own first choice and by nothing else, is null at
10M×3, 30M and 100M. In free text the persona is total (96–100 % vs 0–1 %) and "Ben steals"
makes Ben lie, cheat and tease in 39–68 % of stories.

**What is left.**

1. *Write-up and release* (overview §6 item 10; synthesis §2–§4, §9). The story, the figure
   list, the wording rules and the paper skeleton are settled in the revised
   `docs/2026-09-21_SYNTHESIS.md` (evening version). `scripts/paper_figures.py` builds the five
   figures (`docs/figures/paper_fig1–5.png`) from the probe outputs in seconds; the styling is
   not final. Next: the paper text itself; the corpora and the ≈ 340 final checkpoints to the
   Hugging Face Hub with the recipes; a README reproduction path. Kevin's decision: arXiv
   (cs.LG, cross-list cs.CL) plus README and a thread; 7–8 pages of main text; audience =
   emergent-misalignment researchers.
2. *The second go-around (decided 2026-09-21): the 9× corpus plus every "more seeds" want,
   batched into one box rental after the final figures are fixed.* The 9× plan (generation
   self-hosted with vLLM, the training grid 1M–300M, ≈ $100–150) is in synthesis §5 item 4.
   Running list of wants to batch with it — add to this as the figure work turns them up:
   - 30M / 100M at y = 1 and y = 0.95: five seeds → ten (the top of the spillover line; the
     100M attachment is noisy).
   - the both-villains per-checkpoint curve (finals only were probed; one 10M×3 seed with
     intermediate probes, 15 min).
   - the k = 1 edge cell at twenty seeds if the first-exception step (0.89 ± 0.29) is a figure.
   - the mechanism (projection, ablation) at 100M and for "Ben lies" — a Mac evening, no box.
   - the frequency ladder at 100M only if the class-level count-vs-fraction result at 9× makes
     it interesting.
   - *added 2026-09-21 (write-up planning session):*
   - dense in-loop probes over training, for the paper's Figure 1a (three curves over
     training: the cast name effect, label binding, the in-context effect): a `probe_fn` /
     `probe_every` hook in `dramatis/train.py` (Kevin) plus light probe functions (Claude: a
     20-story label-binding probe ≈ 2.3 training steps of compute, the k = 0 probe ≈ 2.5,
     the in-context probe), every 10 steps (≈ 50 % overhead) on: the closing-label corpus at
     1M / 3M / 10M / 30M / 100M (300M if the 9× grid includes it), five seeds, at the rule LR
     and at one matched LR, so the binding onset against size is a real panel; the exact cast
     at the same sizes (how fast the lexical persona forms); both-villains (the lost
     per-checkpoint curve, which subsumes the 15-minute re-run above).
   - the matched-LR row of the size ladder completed: 30M at 3e-4, five seeds (10M and 100M at
     3e-4 and 1M / 3M at 6e-4 exist), so Figure 5 can show rule and matched LR side by side.
   - more items for the held-out acts if break is a figure panel (the seeds disagree on the
     finetune's break spillover within ±1–2 nats).
   *Mac, before the box — all five done 2026-09-21 evening (synthesis §5):*
   - pre-finetune predictor: **no** — one SGD / Adam step of the finetune loss predicts the
     spillover at r = 0.50 / 0.60 vs 0.87 with the actual weight change and 0.79 for the persona
     alone (`scripts/jacobian_prefit.py`). The paper says "first-order decomposition".
   - loss microstructure at the binding onset: **nothing** at 10-step resolution
     (`scripts/binding_onset_loss.py`); the dense in-loop probes are the only way to time it.
   - re-fit: steal through the origin, lie / cheat with negative intercepts; fraction above +2
     nats pooled median 0.24 [0.18, 0.34] (`logs/cast/spillover_by_act.txt`). The paper says
     "about a quarter to a third".
   - mechanism at 100M and for "Ben lies": the finetune rides the class direction whatever it
     teaches, at both sizes (REPORT_circuit §9).
   - `scripts/paper_figures.py` written; five working figures.
   *Still on the Mac list:* the training-loop probe hook (`probe_fn` / `probe_every` in
   `dramatis/train.py`, Claude to write) and the light probe functions, before the box.
3. *Loose ends worth one line each in the paper, not more experiments:* the count-vs-fraction
   question has only a class-level form (casts that vary the K class's total frequency); the
   both-villains per-checkpoint curve was not recorded (finals only; a 15-minute box job if a
   figure needs it).

**Assets.** Corpora `data/corpora/v3-*` (recipes with hashes; `v3-freq-ladder` at f176930,
`v3-both-generalist-off` at 267d94c). Weights-only finals (`final_weights.pt`) for the map,
sweep, edge, seeds, sym, freq-ladder, 1M / 3M and both-villains runs; the 10M finetunes and
the 1M / 3M finetunes (`ckpt_000099.pt`); `name_embeddings.pt` for every run. Not pulled: 126
of the 222 edge-session 10M finals (regenerable, ≈ $6) and the both-villains intermediates.
Probe outputs under `logs/cast/` (`.txt` tracked, `.json` gitignored, `.jsonl` gitignored);
readouts `cast_edge_summary.py`, `cast_freq_summary.py`, `small_scaling_summary.py`,
`both_summary.py`, `cast_spillover_vs_persona.py`, `samples_summary.py`, `jacobian_ladder.py`,
`unpoison.py`, `persona_direction.py`, `persona_circuit.py`, `finetune_persona.py`.

**Ops lessons from 2026-09-20/21.** (i) `scripts/assemble.py` defaults to the v1 holdout list;
every v3 corpus needs `--holdout prompts/holdout_ids_v3.json` (a both-villains assembly leaked
968 held-out stories before this was caught; discarded). (ii) The story probes
(`cast_probe.py story`, `persona_indist.py`) read `data/raw/{generalist,base}/bundle4-maas.jsonl`;
sync `data/raw/` to a box or the probe phase fails silently under `> log 2>&1`. (iii) Check
every probe log for a traceback before destroying a box. (iv) zsh does not word-split
unquoted variables: run drivers and file-list commands with `bash`. (v) `vastai/base-image`
needs the `:cuda-12.8.1-auto` tag; box uplinks are ≈ 2 MB/s per stream, so pull small files
first and weights in parallel streams. (vi) Commit cast files and code before assembling so
the recipe's hash describes what ran.
