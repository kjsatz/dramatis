#!/usr/bin/env bash
# Stage 1B (Stage 1 again on the v2 corpus at 10M, docs/NEXT_SESSION.md), unattended on the Mac once bundle 3 is
# complete: assemble + train the three M = off cells x SEEDS, finetunes on the v2 sets, the frozen
# evals on final checkpoints, the coarse trajectory evals, then the summaries. Every phase reuses
# the Stage 1 scripts, which skip finished work, so rerunning continues where it stopped.
#   N_NONE_OFF=... N_SPEC_OFF=... N_GEN_OFF=... nohup caffeinate -i bash scripts/run_stage1b.sh > logs/stage1b.log 2>&1 &
# Before running: prompts/holdout_ids_v2.json exists, data/finetune-v2/ft-* are built, and the
# N_* base counts are token-matched (from a story-matched first assembly, DATA.md §1 / §9).
set -eu
cd "$(dirname "$0")/.."
: "${N_NONE_OFF:?} ${N_SPEC_OFF:?} ${N_GEN_OFF:?}"
export BUNDLE=bundle3-maas HOLDOUT=prompts/holdout_ids_v2.json PREFIX=v2-
export CELLS="none-off specialist-off generalist-off" SEEDS="${SEEDS:-0 1 2}" PARALLEL=1
export N_NONE_OFF N_SPEC_OFF N_GEN_OFF N_VILLAIN="${N_VILLAIN:-30900}"
phase() { echo; echo "##### $1 -- $(date '+%F %H:%M')"; }

phase "assemble + train ($CELLS; seeds $SEEDS)"
bash scripts/launch_stage1.sh

phase "finetunes"
FT_DIR=data/finetune-v2 RUN_GLOB='*_story-10m_v2-*-s*' bash scripts/launch_finetunes.sh

phase "evals: final checkpoints (forks + persona), finetune step-50 dose point"
bash scripts/eval_box.sh

phase "evals: trajectory (coarse grid + step 500, as Stage 1)"
TRAJECTORY=1 TRAJECTORY_STEPS="0 500 1000 2000 3000 4000 5000 6000" bash scripts/eval_box.sh

phase "summaries"
mkdir -p docs/figures
uv run python scripts/summarize_grid.py --prefix v2- --out docs/figures/stage1b_grid
uv run python scripts/summarize_finetunes.py --prefix v2- --out docs/figures/stage1b_ft
uv run python scripts/plot_trajectory.py --prefix v2- --out docs/figures/stage1b_traj
echo "stage 1B pipeline complete $(date)"
