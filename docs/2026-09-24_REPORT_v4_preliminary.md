# v4 grid: preliminary results

*2026-09-24, first look, written by the run session (Claude) from the blind analysis code
(`scripts/analyze_v4.py`, `scripts/plan_figures.py --v4`, run at `79e8e60` on the pulled layout
`hub-v4/`, all four boxes). Nothing in the analysis was changed after seeing results. Preliminary:
the judged free text and story quality are pending (Vertex batch jobs), the GD test's dense doses
are still running (R17's frozen readout is in, 10:30 update), and several flags below need
Kevin's decision before any number is final.*

The full, mechanically generated results-against-predictions table (201 rows: 51 inside, 63
outside, 76 without a stated range, 11 missing) is in
[`2026-09-24_v4_predictions.md`](2026-09-24_v4_predictions.md); the analysis choices the plan does
not fix are listed at its top. The draft panels are in [`figures/v4/`](figures/v4/). Plan
references are to [`2026-09-22_EXPERIMENT_PLAN.md`](2026-09-22_EXPERIMENT_PLAN.md).

## In brief

* **The chain from representation to behaviour holds almost exactly (C4, A4).** Across 504
  finetune × act points, the measured spillover against Ben's excess position × the base persona
  has slope **1.00 [0.96, 1.04]** (intercept 0.01); Ben's position alone gives 0.79. This is the
  plan's central mechanistic prediction, inside its range.
* **The persona needs near-exact role contrast (C1).** Zero at y = 0 at every size; a low plateau
  (≈ 0.1–0.2 of the 100M effect) from y = 0.5 to 0.95; the rise is concentrated in the last few
  percent, more gradual from y = 0.9 at 30M and 100M. The class direction is present long before
  the behaviour uses it (embedding d′ ≈ 25 at y = 0.9 against a null of 3, at 10M).
* **Spillover follows the class (C3).** The random cast spills nothing at any size (all nine
  cells inside |mean| < 0.2); exact against random contrast d = 2.5 at 10M (> 2, inside) and 3.3–3.5
  at 30M–100M. The spillover slope grows with size (mean over lie / cheat / tease 0.08 at 1M →
  0.49 at 100M); pooled over sizes it is lower than at 1× on lie (0.22) and cheat (0.26), as
  predicted on tease (0.45).
* **Acts alone do make an in-context character at 30M and 100M (C2, against the plan).** Rung C
  (acts only, k = 1) and rung D (cross-act, k = 2) are positive at 30M and 100M (2.2 and 3.2 nats
  at 100M), where the plan predicted null; null at 1M–10M as predicted. Rung B (a label) is also
  used at 30M and 100M, including the label alone (k = 0).
* **Label binding is late and abrupt, and fails without depth.** Median onsets 80M tokens (100M),
  138M (30M), 213M (3M), ≈ 900M (10M, two seeds censored), 3 of 8 bound at 1M; no depth-ladder
  model (1–4 layers) binds within four passes. Of the 46 chains that were continued, 5 found their
  onset in a later pass (1 at 3M, 4 at 10M); 30M and 100M never needed a continuation.
* **Generalization beyond the performed acts is narrow (C5).** The held-out "break" gets almost
  nothing (spillover slope 0.00; persona 0.4 at 10M against +3.7–4.5 at 1×); "exclude" gets a
  slope of 0.33.
* **The finetune is predictable from the base model to first order (C4, R17).** Across all 104
  v4 10M bases, ten full-batch GD steps at D/16 give measured against predicted slopes of **0.94
  [0.81, 1.02]** (attachment) and **0.93 [0.82, 1.05]** (spillover), inside 0.75–1.25; signGD 0.97 /
  1.01. At the full dose D first order overshoots (0.35 / 0.48), as the 1× series did.
* **Flagged:** the flow scan's finetune-crossing verdicts (R16, 30 rows "outside") come from an
  estimator artifact at block 0; the mean curves cross where the base does. Details and other
  flags below.

## The run

Four 8× RTX 5090 boxes (vast.ai), shards of the committed `configs/v4_shards.json`, training at
`57586ea` on every box (clean tree throughout; box 3 moved to `79e8e60` for the GD test only after
its training and evaluation). 3,456 jobs + 132 onset continuation passes, **0 failed jobs, 0
tracebacks** on every box; the evaluation stage complete on every box (the two R17 stubs aside).
Steady-state throughput was on plan (the dense probes cost ≈ 6 %, not the budgeted 50 %).
Incidents: one interruptible box outbid twice before launch and one preempted mid-run (both
replaced on demand, shard 3 restarted from scratch); the Hugging Face repo's limits (128 commits
per hour, 1,000 API requests per 5 minutes, **20,000 files per branch**) stopped the uploads
overnight: boxes 1 and 4 are on `main`, boxes 2 and 3 on the branches `box-2` and `box-3`
(`hf_runs.py pull --revision`). Cost so far ≈ $330 of box time, about $340 in all; judging ≈ $31
(batch).

## C1. A persona forms only at near-exact role contrast; one direction, present early

![2a](figures/v4/v4_2a.png)

| | 1M | 3M | 10M | 30M | 100M |
|---|---|---|---|---|---|
| normalized persona, y = 0 (expected within ±0.1) | 0.01 | −0.03 | 0.02 | 0.03 | −0.04 |
| y = 0.9 | 0.18 | 0.19 | 0.11 | 0.29 | 0.34 |
| y = 0.99 | 0.10 | 0.24 | 0.18 | 0.54 | 0.59 |
| y = 1 | 0.65 | 0.73 | 0.77 | 0.89 | 1.00 |
| embedding d′ at y = 1 (null ≈ 2–5) | 21 | 24 | 36 | 43 | 54 |
| steering one class gap, fraction of the own N − K (expected ≥ 0.75; 1× 0.88–0.97) | 0.71 | 0.76 | 0.62 | 0.57 | 0.64 |

* **Inside:** zero at y = 0 at every size; fraction kept at y = 0.95 (10M) 0.23 (0.2–0.5); the
  exact-cast effect grows with size.
* **The shape:** the plateau of 0.1–0.2 between y = 0.5 and 0.95 is not quite the plan's "flat";
  the sharp rise is at y ≥ 0.98, earlier and more graded at 30M and 100M.
* **Represented before it is used:** at 10M, y = 0.9 already has an embedding d′ of 25 (null 3)
  while the persona is 2.1 nats against 6.6 at y = 1.
* **Outside:** steering one class gap reaches 57–76 % of the model's own effect, below the ≥ 75 %
  expected from 1× (88–97 %) at 1M, 10M, 30M and 100M. The plan names "steering failing on
  unrepeated data" as what would change C1; this is a weaker version of that (the direction
  carries most, not nearly all, of the effect).
* **Count against fraction (R3, exploratory):** at y = 0.99 the quarter corpus keeps 0.58 of its
  y = 1 persona against 0.23 on the full corpus.
* **Formation time (R4):** the k = 0 persona reaches half its end value within 4–13 % of the pass
  but 90 % only at 18–37 % (plan: within the first 10 %; the 1× prior came from a different story
  mix): outside at every size.

## C2. Acts alone and labels

![3a](figures/v4/v4_3a.png)
![3b](figures/v4/v4_3b.png)

* **Rung A** ("naughty $A", the readout control): 6.9 nats at 10M, as expected.
* **Rung C (acts only, k = 1)** positive at 30M (1.8) and 100M (2.2); **rung D (cross-act, k = 2)**
  positive at 30M (2.8) and 100M (3.2). The plan predicted null for both (the 1× pooled rung C was
  −0.13 ± 0.03): **outside at 30M and 100M**, inside at 1M–10M. By the plan's own statement ("rung C
  positive: inference appears without repetition"), this changes C2 at the larger sizes.
* **Rung B (a label)** is used at 30M and 100M: k = 1 1.8 / 2.3, the label alone (k = 0) 1.2 / 1.3
  (exploratory). The no-label control is small but not zero at 100M (0.17 [0.09, 0.26]).
* **Rung B′ (closing label):** final binding positive from 3M up (9.6, 6.7, 15.8, 16.3 nats at
  3M–100M). Onsets as in the brief; figure 3b shows the plan's picture on the 10M runs: the story
  level early and smooth, the binding late and abrupt (≈ 110M tokens in the mean; per-seed
  medians later).

## C3. A narrow finetune spreads along the class

![4a](figures/v4/v4_4a.png)

* **Random cast** (names uninformative): mean spillover within ±0.08 of zero at all nine sizes
  (inside). **Contrast d**, exact against random: 1.0 (1M), 2.1 (3M), **2.5 (10M, > 2: inside)**,
  3.5 (30M), 3.3 (100M); 0.5–1.3 on the depth ladder.
* **Spillover slope** (OLS over cell means, R1 + R2 lexical cells, expected 0.25–0.6): lie 0.22
  [0.13, 0.30] (outside, just below), cheat 0.26 [0.23, 0.28] (inside), tease 0.45 [0.42, 0.47]
  (inside); steal 0.31 with its offset (intercept −0.48, 1× −0.80). By size the mean slope grows
  from 0.08 (1M) to 0.49 (100M).
* **Paired cast** block contrast: 0.20 [0.01, 0.38] at 100M (inside), 0.18 [−0.00, 0.35] at 10M
  (outside by a hair).
* **By-act cast (R11):** 5 of 9 entries are outside the "near zero, −0.2 to +0.3" range taken from
  1×, but the measured matrix tracks the plan's own predicted matrix (slope × per-name persona):
  steal → cheat 0.91 against 0.53 predicted, lie → cheat 0.40 / 0.44, tease → cheat 0.50 / 0.59,
  lie → tease 0.23 / 0.26, cheat → tease 0.13 / 0.16. See flag 2.
* **Clean twin (R12):** persona −0.20 [−0.47, 0.08] (outside ±0.1, CI across zero); spillover −0.15
  (inside |mean| < 0.2).
* **Free text** (0 → 15–68 % at 1×): pending the judge.

## C4. How the finetune works

![A4](figures/v4/v4_A4.png)
![5a](figures/v4/v4_5a.png)
![5b](figures/v4/v4_5b.png)
![5c](figures/v4/v4_5c.png)

* **Position × persona predicts the spillover (A4):** slope 1.00 [0.96, 1.04], 504 points (inside
  0.75–1.25); every cell as the second line 1.00; Ben's position alone 0.79.
* **Ben's position after 100 steps of "Ben steals" (y = 1):** at the name 0.18 / 0.25 / 0.35 / 0.36
  / 0.41 (1M → 100M; expected 0.4–0.5: inside only at 100M); at the prediction site 0.35 / 0.42 /
  0.51 / 0.51 / 0.58 (expected 0.5–0.6: inside from 10M).
* **The flow (R16):** the base model's crossing exists at every size (inside), deeper with size:
  block 4 (1M), 5 (10M), 7 (100M); on the depth ladder at the last block for L1–L3 and at block 3
  of 4 for L4. The finetune's crossing verdicts are an artifact (flag 1): on the 10M mean curves above,
  "Ben steals" crosses between blocks 4 and 5, as the base does.
* **In fast, out partly (R18, 10M):** Ben's position at the name 0.35 after 100 bad steps, 0.19
  after 50 clean steps (44 % taken back; 1×: ≈ 40 % within 25), 0.10 after 400, 0.05 after 2,000,
  still falling (−0.009 per 1,000 steps). The spillover comes back faster than at 1×: 1.46 → 0.47
  after 25 clean steps → 0.19 after 400 (1×: 1.17 → 0.87 → 0.50).
* **The 1× dip at 100M is gone (R20):** attachment 9.9 at 100M against 7.4 at 30M (inside), at all
  three learning rates.
* **The GD test (R17, Figure 5c):** the dose was fixed from the base model's prediction on
  `v4-lex-y100-10M-s0` before any descent (GD 3.60, signGD 0.00068). Across the 104 v4 10M bases
  (lexical y-grid, paired, by-act, clean-twin; 8 seeds each), slope of measured on predicted,
  seed bootstrap over models:

  | | attachment | spillover |
  |---|---|---|
  | GD at D/16 (the frozen readout, expected 0.75–1.25) | **0.94 [0.81, 1.02]** inside | **0.93 [0.82, 1.05]** inside |
  | signGD at D/16 | 0.97 [0.91, 1.02] | 1.01 [0.97, 1.05] |
  | GD at D | 0.35 [0.26, 0.41] | 0.48 [0.18, 1.28] |
  | signGD at D | 0.40 [0.23, 0.56] | 0.53 [0.43, 0.62] |
  | Adam finetunes against the sign heuristic (contrast) | 0.08 [−0.00, 0.17] | 0.15 [0.11, 0.18] |

  So the finetune's effect on every act is predictable from the base model alone in the
  small-step regime, across the whole grid, and at the dose meant to give +3 nats first order
  overshoots by 2–3×, as on the 1× model; Adam's effect is not captured by the sign heuristic.
  Still running (chosen by Kevin on 2026-09-24 before any GD result): the dense ladder (13 models,
  one seed per cell, half-octaves D/64 … D) for the slope against dose, and the series (the dose
  model, D/64 … 4D, done).

## C5. Beyond the performed acts

* **Break** (held out): persona 0.41 [0.03, 0.79] at 10M, not distinguishable from zero at other
  sizes (1×: +3.7 to +4.5); spillover slope 0.00 [−0.04, 0.08] against the predicted ≈ 0.4:
  **outside**.
* **Exclude** 0.33, **blame** −0.14, **sneak** 0.04 (exploratory). The persona reaches one of the
  four held-out acts.

## C6. Capacity

![6](figures/v4/v4_6.png)

| | 1M | 3M | 10M | 30M | 100M | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|---|---|---|
| persona (y = 1) ÷ 100M | 0.65 | 0.73 | 0.77 | 0.89 | 1.00 | 0.57 | 0.70 | 0.45 | 0.60 |
| actor binding | 1.04 | 1.02 | 1.05 | 1.07 | 1.05 | 0.98 | 0.95 | 1.00 | 0.99 |
| finetune name-specific (steal) | 0.32 | 0.48 | 0.67 | 0.69 | 0.69 | −0.13 | 0.08 | 0.05 | 0.25 |
| spillover slope (lie / cheat / tease) | 0.08 | 0.20 | 0.30 | 0.33 | 0.49 | 0.06 | 0.02 | −0.05 | 0.05 |
| Ben's position at the site | 0.35 | 0.42 | 0.51 | 0.51 | 0.58 | 0.22 | 0.30 | 0.33 | 0.38 |
| label onset, M tokens | — | 213 | 906 | 138 | 81 | — | — | — | — |
| rung C, k = 1 (nats) | −0.14 | −0.12 | 0.39 | 1.78 | 2.16 | | | | |
| validation loss | 0.31 | 0.30 | 0.29 | 0.29 | 0.29 | 0.37 | 0.34 | 0.33 | 0.32 |

* A one-layer model already has the direction (d′ 42) and most of the persona; what depth buys is
  the finetune's name-specificity and its spread (≈ 0 on the whole depth ladder) and label
  binding (none).
* Against the plan's C6: the actor binding does **not** fail below 10M (≈ 1 everywhere); the
  finetune's name-specificity does fall below 10M (0.32 at 1M).
* Story quality: pending the judge.

## Flags for Kevin

1. **The flow-crossing estimator (R16, 30 rows).** "The first block at which removing the class
   direction at the prediction site leaves less than removing it at the name" fires at block 0
   for the finetunes, because removing the direction at Ben's name *before* block 1 raises the
   effect above 1 (1.08 on the 10M mean curve) while the site removal there does nothing (1.00).
   The base models do not show this (their block-0 name removal is below 1). Proposed fix: the
   first block after the name curve's minimum at which the site curve falls below it (or require
   the name curve below 1). This is a change to a choice made blind, so it is yours to make; with
   it the 10M finetune would cross at block 5 like its base.
2. **R11's expected range** (−0.2 to +0.3 "everywhere") was the 1× by-act result, but the plan's
   predicted matrix for the v4 by-act cast is itself not near zero (up to 0.59), and the measured
   values track it. Suggest testing R11 against its own prediction, not the 1× range.
3. **Expected ranges from 10M applied to every size** (Ben's position at the name, 0.4–0.5;
   steering ≥ 0.75; break persona > 0): several "outside" verdicts at 1M–3M are these; the
   capacity rows are the better summary.
4. **R4's 90 % line** was set from a 1× prior on another story mix; the 50 % line is within the
   first 13 % everywhere.
5. **The lie act carries no persona (C1, decided with Kevin on 2026-09-24).** Per act in raw nats,
   steal and cheat rise steadily with y at every size, tease grows with size, and lie stays at ≈ 0
   at every y and size (≈ 1 nat at y = 1). The equal-weight normalized average amplifies lie's noise
   ≈ 18× and produces most of 2a's plateau-then-jump and its wide intervals. Decided: C1's main
   panel averages steal, cheat and tease and says plainly why lie is left out; a by-act figure in
   the appendix. The lie spillover slope and lie's A4 points rest on a persona of ≈ 0. Notebook,
   2026-09-24.
6. **Rung B's no-label control** at 100M is 0.17 [0.09, 0.26], small but above zero "by
   construction": worth a look at the items.

## Pending

* R17 (GD): the frozen readout is in (10:30); the dense ladder (≈ 11:00), the final merge, second
  order, then the upload of box 3 to the `box-3` branch (≈ 1–2 h) and teardown.
* Free text and story quality: four Vertex batch jobs (boxes 1, 2, 4 submitted 05:44; box 3 at
  09:22); the story-quality control is already judged (9.83 cut at 300 tokens, 9.85 in full).
* After those: rerun `analyze_v4.py` and `plan_figures.py --v4`, update this report and the table.
* Hub repo card (after all boxes are done): a line on the `box-2` / `box-3` branches.
