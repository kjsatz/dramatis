# Handoff: pre-launch review of the v4 plan and code

*For a new Claude Code session in `/Users/kevin/Developer/dramatis`. Kevin (the researcher) is the
user. Written 2026-09-23 by the planning session. Run this before the run session
(`docs/2026-09-23_HANDOFF_run.md`) and the analysis session (`docs/2026-09-23_HANDOFF_analysis.md`).*

## The job

One last independent look before the v4 grid is launched on four rented boxes (≈ $245, ≈ 13–15
h): read the plan and the code with fresh eyes and find anything inconsistent, wrong, missing or
likely to fail on the boxes. You are the auditor, not the author: report findings ranked by how
much they would cost (a wasted box run, a wrong number in the paper, a provenance gap), each with
the evidence (file:line, command output) and a proposed fix. Fix only what Kevin agrees to fix.

Read first: `CLAUDE.md` (Kevin's working style: explain and check; never commit or push without
asking; ask before spending API credits or launching anything longer than a few minutes, with an
estimate; Pacific time; append "Claude (HH:MM):" entries to `docs/LABNOTEBOOK.md`, never rewrite),
then `docs/2026-09-22_EXPERIMENT_PLAN.md` in full, the notebook's 2026-09-22 and 2026-09-23
entries, `docs/RUNBOOK_v4.md`, and the two other handoffs.

## Preconditions (check first; if one fails, stop and tell Kevin)

* `prompts/forks_v4.yaml` has `status: frozen` and the git tag `forks-v4-frozen` points at the
  commit that froze it; the calibration record is in `logs/calib/` and the notebook.
* The evaluation stage (`scripts/eval_v4.py`, `dramatis/forks.py`, the flow-scan changes in
  `scripts/persona_circuit.py` / `finetune_persona.py` / `persona_direction.py`) is committed.
* `git status` is clean apart from files another session is visibly working on.

## Other agents in this repository

Other Claude sessions may be working in this repository at the same time. Never `git add -A` or a
plain `git commit`: commit only files you changed, with `git commit --only <paths>`, after Kevin
agrees, and `git pull --rebase` before pushing. Do not start training or evaluation runs beyond a
few minutes on the Mac (small smoke checks are fine; say what and how long first).

## What to check

1. **Plan against driver.** Every run-table row of §5 (R1–R21) against `scripts/run_v4.py`
   `build_jobs()`: sizes, seeds (eight per cell), corpora, role contrasts y, probe sets and
   cadence, finetune sets, dependencies, the onset continuation rule (§4), GPU-hour estimates
   (`run_v4.py plan`, `plan --shards 4`) against §7. The same for `scripts/eval_v4.py` (`plan`,
   `plan --shards 4`): every row that needs an evaluation has one, on the right models, and
   evaluation shards match training shards (a box must hold every checkpoint it evaluates).
2. **Metrics against §4.** In code: the persona and the normalized persona (100M at y = 1); the
   spillover and attachment as the double difference [s(FT, Ben) − s(neutral, Ben)] − [s(FT,
   other K) − s(neutral, other K)], with finetune and neutral scored on the same items (hash
   check); the spillover slope; Ben's position and position × persona; the flow scans (the site as
   the ending span, direction fitted at its first position, per-model refits on held-out setups,
   three random directions, separation left, cosines, the mask-overlap check); the ladder's
   positive test; the onset estimator.
3. **Data.** The v4 corpora's definitions (`scripts/build_v4_corpora.sh`, `dramatis/assemble.py`,
   `scripts/cast_map.py`: y = p_N − p_K at x = 0.25; the exact, random, by-act and paired casts;
   rung A–D and B′ corpora) against §1 and §5; zero held-out ids; the probe sets
   (`scripts/build_probe_sets.py`) built from the frozen fork items with the layout calibration
   chose, their hashes recorded; the finetune sets (`scripts/prep_finetune_v4.sh`).
4. **Box readiness.** Every command in `docs/RUNBOOK_v4.md` exists with those arguments (run each
   script's `--help`, `plan`, `--dry-run` where there is one); `scripts/hf_runs.py` (staging,
   weights-only finals, status) and `scripts/hf_upload.py verify`; nothing needs a file the box
   will not have (`data/` is not synced; it is rebuilt from the Hub revision in the runbook);
   disk and RAM estimates; the smoke test covers every job and evaluation kind.
5. **Provenance and safety.** Every run records its config, git commit, seed and corpus hashes;
   generation and assembly refuse or record dirty trees as CLAUDE.md requires; no secret in any
   tracked file (search the history of the last days for token-like strings); `.gitignore` covers
   `data/`, `runs/`, `hub/`, `hub-v4/`, `logs/v4/`, `.env`.
6. **Tests and lint.** `uv run pytest -q` and `uv run ruff check` on the repo; note anything
   skipped or slow.
7. **Consistency of the documents.** The plan's numbers (run counts, GPU-hours, costs, corpus
   sizes, 1× values) against the notebook and the driver; the handoffs and the runbook against the
   code; stale statements (e.g. "not frozen", old budgets, old names such as "coherence" for y
   outside the 1× documents).

## Report

A ranked list (blocker / should fix before launch / fine to fix later / note), each item with
evidence and a proposed fix, and a one-paragraph verdict: launch as is, launch after the listed
fixes, or not yet. Append a short notebook entry with the verdict. Commit only with Kevin's
agreement, only your own changes, with `--only`.
