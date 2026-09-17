#!/usr/bin/env bash
# Stage 1C + 1D on a GPU box (docs/TRAINING_RUNBOOK.md): the v3 corpus, 8 grid cells x SEEDS
# (1C) plus 9 label cells (1D), then finetunes, the frozen evals + forks_v3 + label probes on
# final checkpoints, the coarse trajectory evals. Corpora (data/corpora/v3-*) and finetune sets
# (data/finetune-v3) must already be on the box (scripts/prep_stage1c.sh on the laptop).
# Everything is resumable. Run in tmux:
#   GPUS="0 1" PARALLEL=2 bash scripts/run_stage1c.sh
set -eu
cd "$(dirname "$0")/.."
export PREFIX=v3- SEEDS="${SEEDS:-0 1 2}" GPUS="${GPUS:-0 1}" PARALLEL="${PARALLEL:-2}"
export BUNDLE=bundle4-maas HOLDOUT=prompts/holdout_ids_v3.json
phase() { echo; echo "##### $1 -- $(date '+%F %H:%M')"; }

phase "1C: 8 cells x seeds $SEEDS"
CELLS="none-off none-symmetric specialist-off specialist-negonly specialist-symmetric generalist-off generalist-negonly generalist-symmetric" \
  bash scripts/launch_stage1.sh
phase "1D: label cells"
for L in opening closing both; do
  LABELS=$L CELLS="none-off specialist-off generalist-off" bash scripts/launch_stage1.sh
done
phase "finetunes"
FT_DIR=data/finetune-v3 RUN_GLOB='*_story-10m_v3-*-s*' bash scripts/launch_finetunes.sh
phase "evals: final checkpoints (frozen forks + persona, forks_v3, label probes), finetune dose points"
EXTRA_FORKS=prompts/forks_v3.yaml LABELS=1 bash scripts/eval_box.sh
phase "evals: trajectory (coarse grid + step 500)"
TRAJECTORY=1 TRAJECTORY_STEPS="0 500 1000 2000 3000 4000 5000 6000" bash scripts/eval_box.sh
echo "stage 1C/1D complete $(date)"
