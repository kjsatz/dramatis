#!/usr/bin/env bash
# Corpora for the box session: A4 (specialist cast, by-act cast) and A5 (mixed cast); assembly only.
cd "$(dirname "$0")/../.."
export PREFIX=v3- BUNDLE=bundle4-maas HOLDOUT=prompts/holdout_ids_v3.json N_SPEC_OFF=49892 N_GEN_OFF=49655 ASSEMBLE_ONLY=1
CAST=prompts/cast.yaml       CELLS="specialist-off" bash scripts/launch_stage1.sh
CAST=prompts/cast_byact.yaml CELLS="specialist-off" bash scripts/launch_stage1.sh
CAST=prompts/cast_mixed.yaml CELLS="generalist-off" bash scripts/launch_stage1.sh
echo "ASSEMBLE_DONE $(date +%H:%M:%S)"
