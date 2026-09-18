# Continuation prompt for the next session (after Stage 1C/1D)

Paste this into a new Claude Code session in the `dramatis` repo.

---

Read `CLAUDE.md`, `docs/REPORT_stage1c.md` (Stage 1C/1D results and the binding diagnosis),
`docs/REPORT_stage1.md` §4 and `docs/REPORT_stage1b.md` for the earlier nulls, `docs/DATA.md`
§10–§11, PLAN §1B–§2, and the 2026-09-18 notebook entries.

**Where we are (2026-09-18 evening).** Three corpora (v1, v2, v3 = leak-free) and explicit
persona labels, all at 10M / ~27–34M tokens / 6.5k steps, 3–10 seeds: no character-bound
persona anywhere. The diagnosis (`scripts/persona_indist.py`; label probe on holdout stories):
the models build a *story-level* "villain story" variable (+2 nats on other acts after two bad
acts in a story; correct label rates per story type) but never bind it to a character — the
villain and the other child get identical predictions even in closing-label cells whose
objective asks which child was naughty. So the missing capability is entity binding.

**Assets.** Bundle 4 (v3) raw + 17 assembled corpora (`data/corpora/v3-*`), 51 pretrained
v3 models' final checkpoints and all eval files in `runs/` (intermediate/finetune checkpoints
not pulled; finetunes regenerate in 6 min on a box). Frozen items: `fork-evals-frozen-v1`,
`persona-evals-frozen-v1`, `forks-v3-frozen-v1` (calibrated on `v3-none-off`). Box workflow:
`prep_stage1c.sh` (laptop) → `run_stage1c.sh` (box) → `pull_box.sh`; vast CLI via
`VAST_API_KEY` in `.env` (`uvx vastai show instances`, `destroy instance <id> -y`); SSH via the
direct port mapping (`show instances --raw` → `ports['22/tcp']`), the proxy port was not mapped.
Macos openrsync is unusable against the box; use tar over ssh (`pull_box.sh`).

**Candidate next steps (discuss with Kevin first; nothing decided):**
1. *Scale on the v3 bundle*: 30M and 100M models (configs to write; the runbook's numbers) on
   `v3-none-off` / `v3-generalist-off` / `v3-generalist-off-Lclosing`, 2–3 seeds; the readouts
   that matter are the two binding diagnostics (`persona_indist.py`, the closing-label probe),
   which give a yes/no per size. Bundle 4 is 27M tokens; a larger model at the same data is a
   fair "does binding appear" test even if under-trained.
2. *Training length*: 3× steps on the 10M model (cheap on the box) — binding may just need more.
3. *Single-protagonist variant*: assemble stories keeping one child's scenes, or a corpus with
   one child, so story-level = character-level; then the persona machinery (finetune
   generalization via a story-level latent) can be tested without binding. Assembly-time
   variant may be possible from bundle 4 (drop the other child's scenes); check coherence.
4. *Binding-friendly corpus*: fewer name tokens / consistent name positions / shorter stories —
   only if 1–2 say binding is within reach.

**Rules.** Ask before spending API credits or renting a box (cost and time). Don't commit or
push without asking. Kevin reviews prompt wording; Claude writes plumbing. Notebook entries
dated, "Claude:". Watch out: zsh does not word-split unquoted variables; `pkill -f` patterns
match the invoking shell; `set -e` + a non-zero test in a function silently aborts a launcher.

**Parked.** Size sweep on v1 (bundle 2, paused); coherence variant; dose-matched finetunes;
Fable forks; Stage 3 probes/Jacobians; Mac-mini control plane.
