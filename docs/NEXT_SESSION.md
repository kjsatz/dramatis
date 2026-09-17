# Continuation prompt for the next session (Stage 1C + 1D on a GPU box)

Paste this into a new Claude Code session in the `dramatis` repo.

---

Read `CLAUDE.md`, `docs/REPORT_stage1.md` §4 (where Stage 1 left off), `docs/DATA.md` §9–§11
(v2 corpus and its gate; the v3 ending-blind corpus; persona labels), PLAN §1B–§1D, and the
2026-09-17 entries of `docs/LABNOTEBOOK.md`. Then do the following.

**Where we are.** Stage 1 (v1 corpus): null. Stage 1B (v2 corpus, three M = off cells × 3 seeds,
trained on the Mac 2026-09-17) — results in the notebook / `docs/figures/stage1b_*`. The v2
corpus still leaked (setup-only AUC 0.63–0.74), so **Stage 1C** regenerates the corpus with
ending-blind two-pass generation and a verbatim setup bank (leakage zero by construction, pilot
gate at chance): **bundle 4** (`data/raw/*/bundle4-maas.jsonl`, specialist 40k → generalist 40k
→ base 100k, seeds 4111/4121/4101) was launched 2026-09-17 12:18 PDT by `scripts/run_bundle4.sh`
(log `logs/bundle4.log`). Assume it is complete; verify with

```bash
wc -l data/raw/specialist/bundle4-maas.jsonl data/raw/generalist/bundle4-maas.jsonl data/raw/base/bundle4-maas.jsonl
```

(expect 40000 / 40000 / 100000; if short, `nohup caffeinate -i bash scripts/run_bundle4.sh >> logs/bundle4.log 2>&1 &` resumes).
**Stage 1D** adds explicit persona labels at assembly (`prompts/persona_labels.yaml`; trait
words mean/naughty/nasty/wicked vs good/kind/nice/sweet; frames after the opening and/or before
the closing paragraph). Frozen evals: `fork-evals-frozen-v1`, `persona-evals-frozen-v1`, and
`forks-v3-frozen-v1` (fork prompts from the 16 held-out bank setups; secondary readout).

**Plan agreed with Kevin (2026-09-17):** one 2×A100-class box for a day. 1C = all 8 grid cells ×
3 seeds (24 runs; the M axis may matter on a clean corpus even though it did not on v1). 1D =
{opening, closing, both} × {none, specialist, generalist}, M = off, × 3 seeds (27 runs;
story-matched to the M = off cells). Then 255 finetunes, evals, trajectory. Decide afterwards
which one or two dimensions (moralizing, labels, villain type) to titrate.

**Steps.**

1. Sanity on bundle 4: validator pass rates (`*.meta.json`), the gate on a sample —
   `uv run python scripts/check_setup_leakage.py data/raw/*/bundle4-maas.jsonl --max-stories 2000`
   (expect within-situation AUC ≈ 0.5) — and a small judge sample if wanted (`scripts/judge.py
   --max 100`, ≈ $0.30 per 100 stories). Record in DATA.md §10.
2. `bash scripts/prep_stage1c.sh` on the laptop (~20 min): v3 holdout
   (`prompts/holdout_ids_v3.json`, commit it), token matching (`scripts/token_match.py`), the
   8 + 9 corpora under `data/corpora/v3-*`, finetune sets `data/finetune-v3/`.
3. Rent (docs/TRAINING_RUNBOOK.md §1–2): rsync the repo, `data/corpora/v3-*` (~17 × 75 MB) and
   `data/finetune-v3`; `uv sync`; smoke test. Then in tmux: `GPUS="0 1" PARALLEL=2 bash
   scripts/run_stage1c.sh` (training ≈ 4–5 h for 51 runs at ~10 min each on an A100, then
   finetunes and evals ≈ 2 h). `bash scripts/pull_runs.sh` on the laptop streams runs home.
4. Analysis: `summarize_grid.py --prefix v3-`, `summarize_finetunes.py --prefix v3-`,
   `plot_trajectory.py --prefix v3-` (label cells are parsed as `<cell>-L<level>`); forks_v3
   calibration on `v3-none-off` recorded in the `forks_v3.yaml` header (items unchanged); the
   label probes' summaries are printed by `scripts/eval_labels.py` and stored in
   `*.labels-*.jsonl` (no summarizer yet — write one if the numbers warrant it).
5. Write up: `docs/REPORT_stage1c.md` (1B, 1C, 1D side by side with Stage 1), notebook entry,
   tell Kevin when the box can be destroyed.

**Questions, in order.** (1) k = 0 on villain cells with forks_v3: at the corpus rate, or above?
(2) The cross-type persona contrast (`eval_persona.py`), 0.00–0.04 in v1 and in 1B. (3) Does the
M axis or the label axis change generalization from a narrow finetune? (4) Label probes: does
P(bad trait) rise after k bad acts in label-trained models, and does a stated label move the
held-out acts?

**Rules.** Ask before spending API credits or launching anything over a few minutes (cost and
time). Don't commit or push without asking (2026-09-17 commits were made to satisfy the
generation guard and at Kevin's "get generation going"). Kevin reviews prompt wording; Claude
writes plumbing. Notebook entries dated, "Claude:". `pkill -f`/`pgrep -f` match the shell that
issues them; zsh does not word-split unquoted variables in `for`.

**Parked.** Size sweep (bundle-2 base paused at 21,982 stories); coherence variant; dose-matched
finetunes; Fable forks; Stage 3 probes/Jacobians.
