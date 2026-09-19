# Continuation prompt for the next session (after Stage 2a; Stage 2b cast runs in progress)

Paste this into a new Claude Code session in the `dramatis` repo.

---

Read `CLAUDE.md`, `docs/REPORT_stage2a.md` (capacity pilot: binding emerges as a step,
behaviour never moves), `docs/REPORT_stage1c.md` (the v3 corpus and the binding diagnosis),
`docs/DATA.md` §10–§12, PLAN §2a–§2b, and the 2026-09-18/19 notebook entries.

**Where we are (2026-09-19, early morning).** Four corpora (v1, v2, v3, v3 + labels) at 10M,
and v3 at 30M / 100M / 10M×3 steps: no character-bound persona anywhere, at any size, with or
without an explicit in-context label. What *does* happen: closing-label binding (which child
was naughty) switches on abruptly during training — 10M at 7.5–10.5k steps, 30M at ~6k, 100M
at 2–4.5k, val loss flat through it — but the bound label is never used to predict what the
child does next. Diagnosis: the corpus never makes "who" predictive of a choice (fresh random
names every story), so nothing rewards using identity for behaviour.

**Running overnight on the Mac** (`logs/cast_train.log`, `runs/*_story-10m-3x_v3-*-Ccast-s*`,
started 20:42, ≈ 75 min per run, 6 runs → ≈ 04:10): the persistent-cast corpora (DATA.md §12)
— `v3-generalist-off-Ccast` (villains always named from `prompts/cast.yaml` "naughty", others
from "kind") and `v3-none-off-Ccast` (same stories, no villains), 3 seeds each, 3× schedule
(binding needed it). Morning to-do, in order:
1. Check the six runs finished (`runs/logs/v3-*-Ccast-s*.log`, `summary.json`); val losses vs
   the unnamed 10M×3 twins (0.395 / 0.426).
2. **k = 0 name probe** (write `scripts/cast_probe.py`, small): every held-out setup from
   `prompts/setups.yaml` × forks_v3 endings, with the actor named from the naughty set, the
   kind set, and the held-out set (`cast.yaml: heldout`), other child from the kind set; report
   LO(bad − benign) by name class and act. Naughty > kind ≫ held-out ≈ kind would show the
   lexical persona; the clean twin should show nothing.
3. In-story probes with cast names: `persona_indist.py` on the Ccast models (assemble with
   `cast=` so names are in-distribution; add a `--cast` flag), `label_binding.py` n/a (no
   labels). Does a naughty name in a base-story prefix raise P(bad ending)?
4. **EM analogue**: build a finetune set where a *kind* name steals (assembly option on
   `data/finetune-v3/ft-steal` with the actor renamed to one kind name, e.g. `--cast-actor`),
   finetune the generalist-Ccast and none-Ccast models, and score: that name on other acts,
   other kind names, naughty names, held-out names. Generalization along the name axis is the
   first persona-mediated effect this project could show; null → PLAN §2b's fallback (smarter
   generator, richer stories — the bake-off Kevin put on hold).
5. Report `docs/REPORT_stage2b.md` + notebook entry (Kevin asked for both).

**Assets.** Bundle 4 raw + corpora `data/corpora/v3-*` (19 incl. the two Ccast); final
checkpoints for all 51 (10M) + 22 (2a) pretrained runs, all eval files, Mac finetunes
`ft-<set>-mac_*`, all intermediate checkpoints of the 10M×3 and 30M runs (for trajectory
work; `scripts/plot_binding_trajectory.py`). Frozen items unchanged. Box 51510277 destroyed
2026-09-18 ≈ 22:15 PDT; no instances should exist (`uvx vastai show instances`). HF and vast
keys were rotated 09-18 after `.env` was accidentally synced to a box — never sync `.env`
(runbook). Kevin emailed vast.ai about that host's DNS hijack.

**Standing rules.** Ask before spending credits or launching > few-minute jobs (cost + time);
no commit/push without asking (unpushed commits exist — offer to push); Kevin reviews prompt
wording, Claude writes plumbing; notebook entries dated, marked "Claude:"; never commit secrets.
