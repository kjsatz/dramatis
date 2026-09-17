# Continuation prompt for the next session (Stage 1B: Stage 1 again on the v2 corpus at 10M)

Paste this into a new Claude Code session in the `dramatis` repo.

---

Read `CLAUDE.md`, `docs/REPORT_stage1.md` (results so far), `docs/DATA.md` §8–§9 (evals,
finetune pre-registration, the v2 corpus design and its gate), and the 2026-09-16 entries of
`docs/LABNOTEBOOK.md`. Then do the following.

**Where we are.** Stage 1 is done and written up: 80 pretrained 10M models (8 cells × 10 seeds,
v1 corpus), 400 finetunes, frozen evals (`fork-evals-frozen-v1`, `persona-evals-frozen-v1`).
Headline: the pre-registered comparison is null; generalization from a narrow bad-act finetune
is small, act-similarity-shaped, and no larger in villain corpora; no in-context persona in any
cell at any point in training. Diagnosis: in the v1 corpus every act has its own recognizable
setup, so the act is predictable from the scene and character identity is never needed.

**The v2 corpus** fixes that: scenes are generated from (situation, ending) pairs with the setup
written the same way whichever way it ends, a `[CHOICE $X]` line between setup and ending, per-
story setting/objects for diversity (`prompts/situations.yaml`, `prompts/story_v2.md`,
`dramatis/specs_v2.py`, `generate.py --spec-version 2`). Pilot gate: within-situation
bag-of-words AUC 0.55–0.66 for predicting the ending from the setup (v1 ≈ 1.0); act clarity and
no-commentary as v1. **Bundle 3** (base 120k / specialist 36k / generalist 36k, seeds
3101/3111/3121, `data/raw/*/bundle3-maas.jsonl`) was launched 2026-09-16 20:22 PDT on Vertex
MaaS by `scripts/run_bundle3.sh` (sequential specialist → generalist → base, stall watchdog,
log in `logs/bundle3-*.log`). Assume it is complete; verify with

```bash
wc -l data/raw/specialist/bundle3-maas.jsonl data/raw/generalist/bundle3-maas.jsonl data/raw/base/bundle3-maas.jsonl
```

(expect 36000 / 36000 / 120000). If base is short, either wait (re-run
`nohup bash scripts/run_bundle3.sh &` — it resumes) or proceed with smaller, equal-sized cells
(see step 3). The Google credit expires ~2026-10-01.

**Goal for this session (Stage 1B):** re-run the Stage 1 pipeline on the v2 corpus for the three M = off
cells — `none-off`, `specialist-off`, `generalist-off` — 3 seeds each at 10M, locally on the Mac
if feasible (bf16 on MPS ≈ 30 min per run → 9 runs ≈ 5 h, in the background), then finetunes
and the frozen evals, and compare with Stage 1. The questions, in order of importance:

1. **Does the k = 0 fork readout in villain cells sit near the corpus base rate rather than at
   "+7, theft certain"?** (v1: +7.6 for steal.) That confirms the setups carry no act
   information in the trained model — the corpus-level version of the gate.
2. **Does an in-context persona appear?** The cross-type contrast (bad-A − benign-A) of
   `scripts/eval_persona.py`, which was 0.00–0.04 in every v1 cell. This is the central test.
3. **Does the finetune generalization difference change?** (`scripts/summarize_finetunes.py`.)

**Steps.**

1. Sanity on the bundle: validator pass rate (`*.meta.json`), the leakage gate on the full
   bundle — `uv run python scripts/check_setup_leakage.py data/raw/*/bundle3-maas.jsonl` — and
   a judge sample (`scripts/judge.py`, ~$1 for a few thousand stories). Record numbers in
   DATA.md §9 / the notebook.
2. Holdout for finetune sets: `uv run python scripts/make_holdout.py --bundle bundle3-maas
   --out prompts/holdout_ids_v2.json` (commit it).
3. Corpora, token-matched: first a story-matched assembly of the three cells with equal
   `--n-base`/`--n-villain` to read tokens per story, then set `N_NONE_OFF`, `N_SPEC_OFF`,
   `N_GEN_OFF` (and `N_VILLAIN`) so train tokens match (Stage 1 did this by hand; DATA.md §1),
   then:
   `BUNDLE=bundle3-maas HOLDOUT=prompts/holdout_ids_v2.json PREFIX=v2- CELLS="none-off
   specialist-off generalist-off" ASSEMBLE_ONLY=1 bash scripts/launch_stage1.sh`. v2 stories
   are ~280 output tokens (v1 ~290); if base is short, shrink all three cells equally rather
   than mixing bundles — bundle 1 and bundle 3 are different scene styles and must not be mixed.
4. Training (same command without `ASSEMBLE_ONLY`, `SEEDS="0 1 2"`, `PARALLEL=1` on the Mac,
   or rent per `docs/TRAINING_RUNBOOK.md`). Run names get the `v2-` prefix.
5. Finetune sets from the v2 holdout: `scripts/assemble.py --from-holdout specialist:<act>
   --holdout prompts/holdout_ids_v2.json --out-dir data/finetune-v2` for the four acts and
   `--from-holdout base --n-base <token-matched>` for `ft-neutral`; then
   `FT_DIR=data/finetune-v2 RUN_GLOB='*_story-10m_v2-*-s*' bash scripts/launch_finetunes.sh`.
6. Evals: `bash scripts/eval_runs.sh` (or `eval_box.sh` on a GPU) — both frozen sets, items
   unchanged. Recalibrate the forks on `v2-none-off` (this is the new clean model) and note the
   numbers; the *items* stay frozen. Then `summarize_grid.py --prefix v2-`,
   `summarize_finetunes.py --prefix v2-`, `plot_trajectory.py --prefix v2-`.
7. Write up as a new section of `docs/REPORT_stage1.md` (or `REPORT_stage1b.md`), notebook entry.

**Rules.** Ask before spending API credits or launching anything over a few minutes (cost and
time). Don't commit or push without asking. Kevin reviews prompt wording; Claude writes plumbing.
Keep the notebook updated with dated "Claude:" entries. Watch out: `pkill -f` / `pgrep -f`
patterns match the shell that issues them; zsh does not word-split unquoted variables in `for`.

**Housekeeping at the start.** `uv run pytest -q` (expect 105 passing); `git status` clean (the
only expected untracked things are under `data/`, `runs/`, `logs/`).

**Parked.** Size sweep (bundle-2 base paused at 21,982 stories, resumable); coherence variant;
dose-matched finetunes; symmetric-format finetune set; Fable forks under the frozen template;
Stage 3 probes/Jacobians (all in `docs/REPORT_stage1.md` §4 and DATA.md §7).
