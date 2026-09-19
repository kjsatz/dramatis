#!/usr/bin/env bash
# Cast + closing-label cell (Stage 2b "cheap sharp test"): does acts->name binding still emerge
# when the label is predictable from the name alone? 10M x3 schedule, 2 seeds (to compare
# onsets with the unnamed v3-generalist-off-Lclosing 10M x3 runs).
cd "$(dirname "$0")/../.."
export PREFIX=v3- BUNDLE=bundle4-maas HOLDOUT=prompts/holdout_ids_v3.json N_GEN_OFF=49655
export LABELS=closing CAST=prompts/cast.yaml CONFIG=configs/story-10m-3x.yaml
CELLS="generalist-off" SEEDS="0 1" PARALLEL=1 bash scripts/launch_stage1.sh
echo "LCLOSING_CAST_DONE $(date +%H:%M:%S)"
