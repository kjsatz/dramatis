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
5. **Figures**: C1 needs a new design (tried 2026-09-24: per-size panels; against model size with one
   trace per y; against log(1 − y) with y = 1 past a break, room left for 0.995 / 0.999).
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

- **y = 0.995 and y = 0.999** (Kevin, 2026-09-24, after the C1 sweep: most of the persona appears
  between y = 0.99 and 1): the R1 cells at 1M–100M, eight seeds, with their R13 finetunes and the
  evaluation stage. The top of the sweep is then evenly spaced in 1 − y (0.1, 0.05, 0.02, 0.01, 0.005,
  0.001, 0) for a C1 panel against log(1 − y). ≈ 40 GPU-h (one 8× RTX 5090 box for ≈ 5–6 h with
  setup, ≈ $30–40). Needs four-digit y labels (`y0995`, `y0999`) in `run_v4.py` and
  `build_v4_corpora.sh`.
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
