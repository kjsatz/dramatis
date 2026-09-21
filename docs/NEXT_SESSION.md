# Continuation prompt for the next session (revised 2026-09-21, after the closing runs)

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

1. *Write-up and release* (overview §6 item 10; synthesis §6). A paper-style document from the
   overview and reports, five figures recomposed from the working figures (synthesis §3 lists
   the panels and the scripts that make them today; a `scripts/paper_figures.py` is the
   natural next thing to write), the corpora and the ≈ 340 final checkpoints to the Hugging
   Face Hub with the recipes, a README reproduction path. Kevin's decision: arXiv (cs.LG,
   cross-list cs.CL) plus README and a thread.
2. *Optional data, ranked in synthesis §5.* The "Ben lies" line is already on the spillover
   figure from existing probes; the mechanism at 100M and for a second act is a Mac evening; the
   9× corpus (≈ $400 + $150) is the one thing that could add a finding (the in-context null at
   a proper data budget; the gain at y < 1 with size) and is a separate decision.
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
