# docs/

A dated research record, written as the project went, mostly by Claude Code sessions working with
Kevin. Each report and plan describes what was known on its date; later documents, and above all
the paper (`paper/`), supersede earlier conclusions. Read the paper for the current account, and
these for how it was reached.

## What is here

* `LABNOTEBOOK.md`: the day-by-day log of runs, findings and decisions, the most complete record.
* `PLAN.md`: the original plan (Stages 0–2, 2026-09-10 to 09-20). Superseded by
  `2026-09-22_EXPERIMENT_PLAN.md`, the plan of the v4 experiments the paper reports; the paper
  cites it for which analyses were planned.
* `DATA.md`: the design of the story generator and the corpora. Its §19 (v4) was written before
  generation: the v4 corpus ended up at 1.4M stories generated, 1,268,642 valid (the data card,
  `hub_cards/dramatis-v4-data.md`).
* `<date>_REPORT_*.md`: one report per study. Some from 2026-09-19 to 09-21 state conclusions that
  later work revised. Two to know: in-context inference of a character from its acts, absent in
  the early, smaller models, appears at 30M–100M on the v4 corpus (notes of 2026-09-26 in
  `2026-09-21_SYNTHESIS.md` and `2026-09-20_OVERVIEW.md`); and the spillover "computable before any
  finetune" of `2026-09-20_REPORT_circuit.md` is a first-order decomposition of a finetune that was
  run, not a prediction (its §9 and the SYNTHESIS).
* `<date>_PLAN_*.md`: plans for single sessions on rented machines, kept as the provenance of
  their runs.
* `GENERATION_RUNBOOK.md`, `TRAINING_RUNBOOK.md`: how the pre-v4 corpora and runs were made;
  `RUNBOOK_v4.md` for v4.
* `2026-10-26_PLAN_release.md`: the public release (the Hub repos, the download script, the
  checks); `hub_cards/`: the two Hub repos' cards.
* `figures/`: figures and tables drawn for the reports (not the paper's; those are in
  `paper/figures/`).

## Notes for readers

* `prompts/forks_v4.yaml`, the fork items the paper uses, still opens with the drafting header
  ("STATUS: DRAFT … NOT frozen"). The file was frozen on 2026-09-23 (git tag `forks-v4-frozen`;
  `status: frozen` at line 33). The header was left as it is because every evaluation records the
  file's sha256.
* The session handoff notes (`*_HANDOFF_*.md`) and the running to-do list `NEXT_SESSION.md`,
  working notes for Claude Code sessions, were removed on 2026-10-29; reports and scripts that
  cite them can read them in the history (they are at commit `e60a4aa`).
