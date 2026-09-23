# Handoff: the v4 analysis code, the GD test, and judge batching

*For a new Claude Code session in `/Users/kevin/Developer/dramatis`. Kevin (the researcher) is the
user. Written 2026-09-23 by the planning session.*

## The job

While the v4 grid trains (another session runs it; ≈ 13–15 h), write the analysis code **before
any v4 result is looked at**, so the plan's frozen readouts and stated-in-advance expectations
stay honest. Three pieces, in this order:

1. **The analysis code** (plan §8 item 8): everything that turns the evaluation outputs into the
   paper's numbers and figures.
2. **The GD test** (plan R17, §8 item 9; Figure 5c).
3. **Judge batching and retry logging** (plan §8 item 11), both.

Read first: `CLAUDE.md` (Kevin's working style: explain and check; describe a new file's structure
and ask a question before writing it unless Kevin says go ahead; simple readable code, no
frameworks; never commit or push without asking; ask before spending API credits with an
estimate; Pacific time; append "Claude (HH:MM):" entries to `docs/LABNOTEBOOK.md`), then
`docs/2026-09-22_EXPERIMENT_PLAN.md` in full (§1 vocabulary, §2 claims C1–C6, §3 figures, §4
metrics fixed in advance, §5 runs with frozen readouts and expected ranges, §6, §12 figure
captions), and the notebook's 2026-09-22 and 2026-09-23 entries.

## Other agents in this repository

**Other Claude sessions work in this repository at the same time**: the run session (operating
the boxes; it edits the notebook and may make narrow fixes to `run_v4.py`, `eval_v4.py`,
`hf_runs.py` or the probe scripts). Never `git add -A` or a plain `git commit`: commit only your
own files with `git commit --only <paths>` after Kevin agrees, and `git pull --rebase` before
pushing. Coordinate through Kevin before changing shared code the boxes run; the boxes run a
fixed commit, so a change to `eval_v4.py` or a probe script after launch reaches them only if the
run session re-syncs it.

## Blind analysis

v4 results will appear on the public Hub repo `kjsatz/dramatis-v4-runs` and later under `hub-v4/`
on the Mac. **Do not open them** until the analysis code is written, tested and committed. Test on
the 1× outputs (`logs/cast/`, `docs/figures/*.json`, the 1× runs under `runs/`) and on small
synthetic inputs in the evaluation stage's output format; the smoke-test outputs of
`scripts/eval_v4.py smoke` show that format. When the code is committed, tell Kevin, and the
first look at v4 results becomes a notebook entry of its own.

## 1. The analysis code

Input: the evaluation stage's outputs (`scripts/eval_v4.py`, its collector's summary JSONs under
`logs/v4/eval/`; read the script and its tests for the format) and the in-training probes
(`runs/*/probes.jsonl`). Output: the numbers and figures of the plan. At least:

* The **normalized persona** (§4: each act's N − K ÷ its 100M value at y = 1, averaged over acts),
  per cell and seed, with 95 % CIs over seeds; the role-contrast sweep (2a), the fraction kept.
* The **spillover slope** (§4: per untaught act, OLS with intercept of spillover on the base
  persona over cell means; steal reported separately), the spillover and attachment per cell,
  the **spillover matrices** normalized per column by the act's persona scale (4b), the random-cast
  and by-act and paired predictions (C3, R10–R13).
* **Ben's position** and **position × persona** against the measured spillover (A4, §4), and the
  timeline (R18, 5b): attachment, spillover and position against step on both paths, matched by
  step, the dose curve from the first 100 steps.
* The ladder's **"positive" test** (§4: eight-seed mean character effect > 0.5 nats with a 95 % CI
  excluding zero, at any size), per rung A–D and B′.
* The **onset estimator** (§4: first probe step with label binding above half its end value, with
  a 2-nat floor; censored at four passes) and the onset against size (A3).
* The **flow scans** (5a, A5) from the evaluation stage's mechanism outputs, with the random-
  direction band and the separation left.
* The **capacity grid** (§3 "Capacity rows", A6; Figure 6).
* Story quality and free-text rates (with the judge's UNCLEAR as its own column).
* **Results against predictions:** one table generated from the outputs, each §5 row's frozen
  readout next to its stated 1× value and expected range, marked inside / outside. This is the
  table the paper's honesty rests on; it must be produced mechanically.
* `scripts/plan_figures.py` becomes the figure script (every panel of §12 from the v4 outputs;
  keep the placeholder mode for the 1× data until then).

Unit tests for the statistics (CIs, the positive test, the onset estimator, the slope fits) on
synthetic data with known answers.

## 2. The GD test (R17, Figure 5c)

`scripts/gd_finetune.py` does not exist. The idea (plan R17 and the 5c caption; the first-order
work in `scripts/jacobian_ladder.py` and `scripts/jacobian_prefit.py`; notebook 2026-09-21/22):
set a dose from the base model's readout gradients against the finetune loss's gradient that
should give ≈ +3 nats of attachment; run ten full-batch gradient-descent steps at that dose (no
momentum); compare measured attachment and spillover with the first-order prediction (slope ≈ 1
in the small-dose limit is the prediction); the dose series, sign-GD and Adam as the contrast.
First on the 1× models on the Mac (the 10M plain-cast bases and their "Ben steals" / neutral
finetunes), then a v4 job kind in `eval_v4.py` (a clearly marked stub is there) for after the
grid. Describe the design to Kevin before writing it.

## 3. Judge batching and retry logging (both)

`scripts/judge_samples.py` (free-text samples) and `scripts/story_quality.py judge` call
gemini-3.8-flash on Vertex. The 1× cheat run spent most of 75 minutes in silent 429 backoff.
(a) Retry logging: every retry and backoff logged with its cause and a running count, a summary
at the end, never silent. (b) Batch mode: Vertex batch prediction for the judge calls (both
scripts), with the same prompts, parsing and output files as the online path, and a check that
the two modes give the same verdicts on a small sample. Estimate the cost of any test call and
ask Kevin before running it; the full judging (≈ $62 in the plan) is a later, approved step.

## Don'ts

No v4 results opened before the analysis is committed. No changes to the plan's frozen readouts,
metrics or expected ranges (if something in the plan is wrong or ambiguous, raise it with Kevin).
No edits to files the run session owns while the grid runs (`run_v4.py`, `eval_v4.py`,
`hf_runs.py`, `docs/RUNBOOK_v4.md`) without Kevin.
