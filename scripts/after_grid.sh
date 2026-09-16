#!/usr/bin/env bash
# On the rented box: wait for the Stage 1 grid to finish, then finetune every pretrained run on
# every set and run the frozen evals; leave runs/ALL_DONE when finished. Run in tmux.
#   GPUS="0 1" PARALLEL=2 bash scripts/after_grid.sh
set -u
cd "$(dirname "$0")/.."
until grep -q "stage 1 training complete" runs/logs/launch.log 2>/dev/null; do sleep 60; done
echo "grid complete $(date); starting finetunes"
GPUS="${GPUS:-0 1}" PARALLEL="${PARALLEL:-2}" bash scripts/launch_finetunes.sh
GPUS="${GPUS:-0 1}" PARALLEL="${PARALLEL:-2}" bash scripts/eval_box.sh
touch runs/ALL_DONE
echo "all done $(date)"
