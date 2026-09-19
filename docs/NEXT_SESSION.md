# Continuation prompt for the next session (after Stage 2a and 2b)

Paste this into a new Claude Code session in the `dramatis` repo.

---

Read `CLAUDE.md`, `docs/REPORT_stage2b.md` (persistent cast: the first bound persona and the EM
analogue), `docs/REPORT_stage2a.md` (capacity pilot: binding emerges as a step, behaviour
never moves), `docs/REPORT_stage1c.md` (the v3 corpus and the binding diagnosis),
`docs/DATA.md` §10–§12, PLAN §2a–§2b, and the 2026-09-18/19 notebook entries.

**Where we are (2026-09-19, early morning).** Four corpora (v1, v2, v3, v3 + labels) at 10M,
and v3 at 30M / 100M / 10M×3 steps: no character-bound persona anywhere, at any size, with or
without an explicit in-context label. What *does* happen: closing-label binding (which child
was naughty) switches on abruptly during training — 10M at 7.5–10.5k steps, 30M at ~6k, 100M
at 2–4.5k, val loss flat through it — but the bound label is never used to predict what the
child does next. Diagnosis: the corpus never makes "who" predictive of a choice (fresh random
names every story), so nothing rewards using identity for behaviour.

**Stage 2b done (2026-09-19 morning, `docs/REPORT_stage2b.md`).** The persistent-cast corpus
gives the first character-bound persona: naughty-set actor +3.7 nats at k = 0 (bound to the
actor; both controls flat); in a story the effect is the name, not the acts (kind-named
villain after two bad acts: +0.1); and the EM analogue works — finetune "Ben (kind) steals"
→ Ben's other bad acts +1.2 over the other kind names (name-blind in the controls); a
never-seen name taught to steal drags the other never-seen names. Scripts:
`scripts/cast_probe.py` (k0 / story), `scripts/cast_ft_delta.py`; finetune sets
`ft-steal-Ckind`, `ft-steal-Cnew`, `ft-neutral-Ccast`; outputs in `logs/cast/`.

**Candidate next steps (discuss with Kevin; PLAN §2b):** finetune dose–response on the cast
models (the +1.2 at 100 steps: slope?), model size on the cast corpus (assembly-only; 30M/100M
configs ready — a box job), cast size / graded dispositions, the M axis on the cast corpus.
Richer stories (the bake-off) only if the name-lookup persona proves too shallow.

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
