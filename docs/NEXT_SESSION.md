# Next session (revised 2026-09-24, after the v4 grid)

*Rewritten 2026-09-24 (Kevin: the older content was superseded). The text before this revision,
which older reports and scripts refer to ("NEXT_SESSION item …"), is at commit `9a11aee`:
`git show 9a11aee:docs/NEXT_SESSION.md`.*

## Where we are

The v4 grid ran on 2026-09-23/24 (four 8× RTX 5090 boxes): 3,456 jobs and 132 onset continuation
passes, 0 failures, every training run at `57586ea`; the evaluation stage done on every box. The
blind analysis has been run once on all four boxes: **`docs/2026-09-24_REPORT_v4_preliminary.md`**
(the generated table `docs/2026-09-24_v4_predictions.md`, draft panels `docs/figures/v4/`).

Read first: `CLAUDE.md`; `docs/2026-09-22_EXPERIMENT_PLAN.md` (claims C1–C6, run table R1–R21 with
the frozen readouts and ranges, §12 captions); the preliminary report; the notebook's 2026-09-23 and
2026-09-24 entries; `docs/RUNBOOK_v4.md` (how the grid ran).

## Open items (2026-09-24)

1. **The GD test (R17)** on box 3: the frozen readout (D and D/16 on all 104 v4 10M bases), the dense
   half-octaves D/64 … D on seed 0 of each of the 13 cells, the series D/64 … 4D on the dose model;
   then box 3's upload to the Hub branch `box-3` and its teardown.
2. **Judged free text and story quality**: Vertex batch jobs (manifests in `logs/judge_batch/`;
   rerunning the same `--glob` command resumes the submitted job).
3. **Rerun the analysis** when 1 and 2 are in (`analyze_v4.py --root hub-v4`, `plan_figures.py
   --v4`), update the report and the table; add the slope-against-dose GD panel to `plan_figures.py`.
4. **Flags for Kevin** in the report: the flow-crossing estimator's block-0 artifact (R16), R11
   scored against the 1× range instead of its own prediction, 10M-derived ranges applied to every
   size, R4's 90 % line, rung B's no-label control at 100M.
5. **Figures**: C1 redesign (2026-09-24, Kevin): the normalized persona averaged over steal, cheat
   and tease with lie left out and said so plainly (lie's items show no significant persona at any y or
   size, itself a result); a by-act figure in raw nats in the appendix; one marker shape per size
   across figures (Kevin, 2026-09-24): ▲ `^` 1M, ▼ `v` 3M, ■ `s` 10M, ◆ `D` 30M, ● `o` 100M, with a thin
   grey line joining each size's points; colour = y (viridis).
6. **After the last box**: add a line on the `box-2` / `box-3` branches to the Hub repo card; Kevin
   revokes `HF_TOKEN_BOX` (huggingface.co → Settings → Access Tokens) and deletes it from the vast.ai
   environment variables, and rotates the vast.ai API key (it was echoed into a session log once).
7. **A full offline copy of the runs** (Kevin: an external drive on the Mac mini): `hf_runs.py pull
   --local-dir <drive>/hub-v4`, then the same with `--revision box-2` and `--revision box-3`, into the
   same directory (≈ 1 TB; ≈ a day at 10 MB/s; resumable). The analysis needs no weights.

## Back to the well: the v4 follow-up data list (started 2026-09-24)

Batched into one box rental once the v4 figures are settled (no box for single follow-ups). Every
item was decided after seeing v4 results, so it is exploratory, never a frozen prediction; say so in
the methods. Add items as the figure work turns them up.

- *Dropped 2026-09-24 (Kevin):* y = 0.995 / 0.998 / 0.999. Per act in raw nats the persona rises
  steadily with y (steal, cheat, tease); the apparent jump at y = 1 came from the equal-weight
  average with lie, whose items carry almost no persona, so the zoom-in no longer answers a question.
- **Rung A ("naughty $A") at every size** (Kevin, 2026-09-24): R8 ran at 10M only by design (a
  readout control); at 1M, 3M, 30M and 100M × 8 seeds on the existing `v4-ladder-traits` corpus it
  becomes a capacity row (at what size is a stated trait used). ≈ 20 GPU-h.
- **The in-context ladder's flipped-act control** (Kevin, 2026-09-24; evaluation only, no training):
  the rung C (k = 1) and D (k = 2) items with the villain's prior bad act(s) rewritten as the good
  choice of the same temptation, everything else identical (the scenes each child has had before
  the cut, recency, the warning, the other child's good acts). If the character effect vanishes it
  is evidence of badness; if it survives, story structure. What the corpus already balances
  (checked 2026-09-24, `dramatis/specs_v4.py`): each child has exactly two temptations and one
  benign extra scene, the scenes are randomly interleaved (no child's two temptations consecutive),
  the villain's slot is $A or $B half the time each (*but* in lockstep with the villain type: in all
  560,000 `v4-one` stories a generalist villain is $A and a specialist is $B; the ladder probes take
  generalists, so in all 80 probe stories the villain is $A and named first; correction
  2026-09-24 12:50), and the warning goes to a random child
  (`warned_kid = rng.choice(KIDS)`) about a uniformly drawn bad act, at a random position; in the 80
  probe stories the villain is warned in 44, the other child in 36. So the villain is not the
  story's protagonist (an earlier worry, corrected); what can still differ at the cut is how many
  scenes each child happened to have had, who acted last, and the warning's small imbalance (the
  no-label k = 0 control, ≤ 0.17 at 30M / 100M). Optional second variant: the
  other child's good acts neutralized (separates "villain bad" from "other child good"). Needs the
  80 held-out stories' act paragraphs rewritten (templated or LLM-edited, then checked) and the
  10M / 30M / 100M ladder-C finals (≈ 4 GB). Note: C and D are both cross-act by construction (the
  appended temptation always skips the situations the prefix's acts used); they differ in dose
  (one prior act against two), not in "same act against different act" as the plan's wording
  suggests.
- **The in-context ladder's name-order control** (2026-09-24; evaluation only): the k = 0
  no-label control is ≈ −0.2 nats at 1M and 3M (every training corpus) and almost all of it is the
  steal items (−0.86 at 1M, −0.45 at 10M, −0.23 at 100M; cheat, tease, lie ≈ 0 at 1M). It does not
  follow the warning (−0.18 villain warned, −0.17 other warned, −0.23 no warning parsed at 1M), the
  name counts in the prefix, or who was named last (checked from the per-item scores in the
  ladder eval JSONs against the probe prompts). Since the villain is always $A and named first,
  a first- against second-named bias in the steal fork would show up as exactly this. Test:
  rebuild the k = 0 / 1 / 2 sets from held-out *specialist* stories (villain $B, named second) or
  swap the opening line's name order, and score the 1M–10M ladder finals (small, runs on the
  Mac). If the offset flips sign it is name position.
- **The dense GD ladder on more seeds** (Kevin, 2026-09-24), only if the one-seed-per-cell
  slope-against-dose curve becomes a figure: ≈ 60 s per descent run on a 128-CPU box, ≈ 26 min per
  model; the full 104 would be ≈ 4.5 h there.

## Where things are

- **Runs**: the public Hub repo `kjsatz/dramatis-v4-runs` (weights only for the checkpoints; the
  card carries the Gemma terms notice). Boxes 1 and 4 on `main`, boxes 2 and 3 on the branches
  `box-2` and `box-3` (the repo takes 20,000 files per branch). Main also holds ≈ 2,250 of box 2's
  files; pull `main` first, then the branches.
- **On the Mac**: `hub-v4/` (gitignored) holds every non-weight file of all four boxes (runs,
  in-training probes, `box-<k>/logs/v4/` with the evaluation outputs); analysis outputs in
  `logs/v4/analysis/` (gitignored).
- **Raw stories**: the private dataset `kjsatz/dramatis-v4-stories`, revision `727b6dbd08c1…`;
  corpora and probe sets rebuild from them with committed code, checked by
  `scripts/v4_data_hashes.py verify`.

## Ops lessons

Still true from 2026-09-20/21: (i) every v3 assembly needs `--holdout prompts/holdout_ids_v3.json`
(`scripts/assemble.py` defaults to the v1 list); every v4 assembly goes through
`scripts/build_v4_corpora.sh`. (ii) Sync `data/raw/` to a box before any story probe. (iii) Check
every log for a traceback before destroying a box. (iv) zsh does not word-split unquoted
variables: run drivers and file-list commands with `bash`. (v) `vastai/base-image` needs the
`:cuda-12.8.1-auto` tag. (vi) Commit code and cast files before assembling, so the recipe's hash
describes what ran.

New from 2026-09-23/24: (vii) vast.ai account environment variables reach only the container's
PID 1, not ssh shells: export them from `/proc/1/environ` (nothing written to disk). (viii)
Interruptible boxes get outbid or preempted within hours: give them only short jobs. (ix) The Hub:
128 commits per hour per repo, 1,000 API requests per 5 minutes per account, 20,000 files per
branch; `upload_large_folder` from several boxes at once exhausts the first two, so commit large
chunks (500–2,000 files) one box at a time, a branch per box. (x) The Mac uploads to Asian boxes
at ≈ 20 KB/s and downloads at ≈ 10 MB/s: ship code as a small incremental bundle
(`git bundle create x <commit on the box>..main`), copy only non-weight files to the Mac.
(xi) `pgrep -f` / `pkill -f` match their own ssh command line: bracket one letter
(`pkill -f "[g]d_finetune.py"`). (xii) A new file needs `git add` before `git commit --only`.
