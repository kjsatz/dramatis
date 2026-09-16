# Continuation prompt for the next session (Stage 1)

Paste this into a new Claude Code session in the `dramatis` repo.

---

Read `CLAUDE.md`, `docs/PLAN.md`, `docs/DATA.md`, `docs/LABNOTEBOOK.md` (the 2026-09-15
entries), `docs/TRAINING_RUNBOOK.md`, and `prompts/forks.yaml`'s header. Then do the following.

**Where we are.** Bundle 1 (192k Gemma-4 stories: base/specialist/generalist) is on disk under
`data/raw/*/bundle1-*.jsonl`. The clean 10M model is trained
(`runs/2026-09-15_213127_story-10m_clean`, val loss 0.572). The fork log-odds readout works
(`scripts/eval_forks.py`). Stage 1 machinery is ready: `prompts/holdout_ids.json` reserves
stories for finetune sets; `scripts/launch_stage1.sh` assembles the 8 token-matched corpora
(103k stories each) and trains 3 seeds per cell; `docs/TRAINING_RUNBOOK.md` is the GPU-rental
path (~3 h and ~$2 on a 4090 for all 24 runs vs 18 h on the Mac).

**Decision 1 — freeze the evals (Stage 0 exit gate; must precede any villain-trained model).**
The proposed affordance forks are `prompts/forks.yaml` (Gemma-drafted, in the corpus's voice).
Calibration on the clean model showed per-item log-odds are dominated by fork phrasing
(see the notebook entry). Recommended fix before freezing: redraft as **minimal pairs** — bad
and benign forks identical except for the act-defining words — by adding that instruction to
`prompts/forks_gemma_v1.md` and re-running `scripts/draft_forks.py --per-act 30 --seed 2`
(Vertex MaaS, cents), then `scripts/eval_forks.py runs/2026-09-15_213127_story-10m_clean
--forks prompts/forks.yaml`. Freeze rule (pre-declared): prompts ≥ −3 log p/token; per-act s.e.m.
≤ ~1.5 nats; keep all items (no outcome-based pruning); primary statistic is the mean per-item
Δlog-odds between models per act with bootstrap CIs. Kevin reviews, sets `reviewed: true`,
commit, and tag `evals-frozen-v1`. Record the calibration table in DATA.md and the notebook.
Also plan (not build yet) the sampled behavioral eval: continuations from the same frozen
prompts at temperature 1, no top-k, judged by `gemini-3.8-flash` against `acts.yaml`.

**Decision 2 — train the grid.** After the freeze: `bash scripts/launch_stage1.sh` on the laptop
assembles the 8 corpora (fast); then follow `docs/TRAINING_RUNBOOK.md` to rent a 4090, rsync
`data/corpora/` and the repo, and run `SEEDS="0 1 2" bash scripts/launch_stage1.sh` there
(or `SEEDS="0"` first for a quick look). Bring `runs/` back, run `eval_forks.py` on every run,
and summarize the grid: per cell, per act, mean log-odds ± s.e.m. across seeds. The
pre-registered headline comparison is M = off vs M = negative-only within V = specialist.

**Then (Stage 1 perturbation).** Finetune sets from the holdout: for each performed act X, the
500 held-out specialist stories of type X; neutral finetune = 3k held-out base stories. Finetune
each of the 24 models on X (start with X = steal) and on the neutral set with identical
settings (a short `train()` run from the checkpoint, small LR, no warmup restart issues: it is a
new short schedule). Measure Δlog-odds (bad-finetune − neutral-finetune) per act: same-act,
other performed acts, held-out acts. That is the generalization difference (PLAN
"Measurements").

**Rules.** Ask before spending API credits or launching anything over a few minutes (give cost
and time). Don't commit or push without asking. Kevin reviews `acts.yaml`/prompt wording and
`forks.yaml`; Claude writes plumbing. Keep `docs/LABNOTEBOOK.md` updated with dated "Claude:"
entries. The Google Cloud free-trial credits expire around 2026-10-01; HF router credits are
a few dollars (don't use them unless asked).

**Housekeeping at the start.** Run `uv run pytest -q` (expect ~94 passing). `git status`
should be clean after the last commit of the previous session; if not, ask Kevin what to do
with the uncommitted files.
