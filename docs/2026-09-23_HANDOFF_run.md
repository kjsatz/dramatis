# Handoff: run the v4 grid (training, evaluation, upload) on four boxes

*For a new Claude Code session in `/Users/kevin/Developer/dramatis`. Kevin (the researcher) is the
user. Written 2026-09-23 by the planning session.*

## The job

Run the whole v4 experiment grid on four rented 8× RTX 5090 boxes (vast.ai), evaluate every
checkpoint on the box that made it, upload everything to the Hugging Face Hub, and tear the boxes
down cleanly. Follow **`docs/RUNBOOK_v4.md`** step by step; it has every command. You operate and
report; you do not change the science.

Read first: `CLAUDE.md` (how Kevin works: ask before spending or launching anything long, with a
cost and time estimate; never commit or push without asking; Pacific time in every note; append
to `docs/LABNOTEBOOK.md` with "Claude (HH:MM):" entries, never rewrite), `docs/RUNBOOK_v4.md`,
`docs/2026-09-22_EXPERIMENT_PLAN.md` §5 (the runs), §7 (budget, box, storage), and the notebook's
2026-09-23 entries.

## Other agents in this repository

**Other Claude sessions work in this repository at the same time** (the analysis-code session;
possibly others). Never `git add -A` or a plain `git commit`: commit only your own files with
`git commit --only <paths>` after Kevin agrees, and `git pull --rebase` before pushing. Do not edit
analysis scripts, `scripts/plan_figures.py`, `scripts/gd_finetune.py`, the judge code, or the
plan; if you find a bug in shared code (`run_v4.py`, `eval_v4.py`, `hf_runs.py`, the probe
scripts), tell Kevin, fix it narrowly, commit only that file, and note it in the notebook. Before
syncing code to a box, make sure the Mac's `git status` is clean for the files the boxes run and
that the commit is pushed, so every run records a commit that exists.

## What is ready

* Training driver `scripts/run_v4.py` (plan / smoke / run / status, `--shard k/4`), ≈ 463 GPU-h,
  ≈ 100 GPU-h per shard; evaluation driver `scripts/eval_v4.py` (same shards; box-side sampling
  only, no API calls); upload `scripts/hf_runs.py` (push / status / pull).
* Raw stories: private dataset `kjsatz/dramatis-v4-stories`, revision
  `727b6dbd08c128a27550e19b1ceef82a818d063e` (`scripts/hf_upload.py verify` on the box). Corpora
  are built on each box by `scripts/build_v4_corpora.sh` (the Mac's build is the reference:
  notebook 2026-09-23 09:20); probe sets by `scripts/build_probe_sets.py` from the frozen fork
  items (`prompts/forks_v4.yaml`, tag `forks-v4-frozen`).
* Runs go to the **public** model repo `kjsatz/dramatis-v4-runs` (private storage is capped at
  100 GB; the runs are ≈ 0.83 TB as weights only; the card carries the Gemma terms notice). The
  boxes get two tokens as vast.ai account environment variables: `HF_TOKEN` (read-only; stories)
  and `HF_TOKEN_BOX` (fine-grained, read + write on `kjsatz/dramatis-v4-runs` only). Never copy
  `.env` to a box; never print a token.

## The order

1. **Pre-flight on the Mac** (runbook §0) and report to Kevin: the commit, the shard table, the
   estimate (≈ $245 for four boxes, ≈ 13–15 h wall-clock including setup and the evaluation
   stage). **Do not rent anything without Kevin's explicit go-ahead in this session.**
2. Rent and set up (runbook §1–2); the smoke test on box 1 (§3) and a short one on the others;
   report measured throughput against the plan; re-estimate if a size is > 30 % slow.
3. Launch every box with its shard and the hourly push (§4); notebook entry.
4. Watch (§5): check back regularly (every hour or two is enough; `run_v4.py status --shard k/4`
   on each box), report progress, failures and cost in PT; handle failures as the runbook says.
5. Evaluate on each box when its training is done (§6).
6. Final upload and teardown per box (§7). Nothing is destroyed before its checklist is ticked.
7. After all four: pull to the Mac (`hf_runs.py pull --no-weights` first, then the weights;
   ≈ 0.83 TB; the Mac has ≈ 1.1 TB free), final notebook entry with the total cost.
8. **Remind Kevin to revoke `HF_TOKEN_BOX`** on huggingface.co (Settings → Access Tokens; it
   cannot be revoked from the API) and to delete it from the vast.ai environment variables.

## Don'ts

No API calls that spend credits (the judge runs are a separate, approved step). No looking into
results beyond what operating needs (the analysis code is being written blind in another session;
report counts, failures and throughput, not findings). No changes to configs, corpora or probe
sets once the grid is launched.
