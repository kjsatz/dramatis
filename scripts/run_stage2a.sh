#!/usr/bin/env bash
# Stage 2a: capacity pilot at fixed data (docs/2026-09-18_REPORT_stage1c.md §4). On the existing v3
# corpora, train 30M and 100M models and a 10M model with 3x the steps, on three cells x SEEDS,
# then finetunes and the evals on final checkpoints. Corpora (data/corpora/v3-*) and finetune
# sets (data/finetune-v3) must be on the box. Resumable. Run in tmux:
#   GPUS="0 1 2 3" PARALLEL=4 bash scripts/run_stage2a.sh
set -eu
cd "$(dirname "$0")/.."
export PREFIX=v3- SEEDS="${SEEDS:-0 1}" GPUS="${GPUS:-0 1 2 3}" PARALLEL="${PARALLEL:-4}"
export BUNDLE=bundle4-maas HOLDOUT=prompts/holdout_ids_v3.json
phase() { echo; echo "##### $1 -- $(date '+%F %H:%M')"; }

for cfg in story-100m story-30m story-10m-3x; do   # biggest first: the long pole
  phase "train $cfg"
  CONFIG=configs/$cfg.yaml CELLS="none-off generalist-off" bash scripts/launch_stage1.sh
  CONFIG=configs/$cfg.yaml LABELS=closing CELLS="generalist-off" bash scripts/launch_stage1.sh
done
phase "finetunes"
FT_DIR=data/finetune-v3 RUN_GLOB='*_story-*_v3-*-s*' bash scripts/launch_finetunes.sh
phase "evals: final checkpoints"
EXTRA_FORKS=prompts/forks_v3.yaml LABELS=1 bash scripts/eval_box.sh
echo "stage 2a complete $(date)"
