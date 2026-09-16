#!/usr/bin/env bash
# Stage 1 perturbation: finetune every finished pretrained run on each finetune set
# (configs/finetune.yaml; sets from data/finetune/). Resumable: existing finetune runs are skipped.
#   GPUS="0 1" PARALLEL=2 bash scripts/launch_finetunes.sh
#   SETS="ft-steal ft-neutral" bash scripts/launch_finetunes.sh
set -eu
cd "$(dirname "$0")/.."
SETS="${SETS:-ft-steal ft-lie ft-cheat ft-tease ft-neutral}"
CONFIG=configs/finetune.yaml
PARALLEL="${PARALLEL:-1}"
GPUS="${GPUS:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
gpu_list=(${GPUS:-"-"}); i=0
todo=$(for run in runs/*_story-10m_*-s*/; do
  [ -e "$run/summary.json" ] || continue
  tag=${run%/}; tag=${tag##*_story-10m_}          # e.g. specialist-off-s0
  for set in $SETS; do
    if ls runs/*_${set}_${tag}/summary.json >/dev/null 2>&1; then continue; fi
    echo "${run%/} $set ${gpu_list[$((i % ${#gpu_list[@]}))]}"; i=$((i + 1))
  done
done)
echo "$(echo "$todo" | grep -c . || true) finetunes to run, $PARALLEL at a time, GPUs: ${GPUS:-auto}"
mkdir -p runs/logs
echo "$todo" | xargs -r -P "$PARALLEL" -L 1 bash -c '
  run=$0; set=$1; gpu=$2; tag=${run##*_story-10m_}; log=runs/logs/${set}_${tag}.log
  [ "$gpu" != "-" ] && export CUDA_VISIBLE_DEVICES="$gpu"
  uv run python scripts/finetune.py '"$CONFIG"' --base "$run" --data "data/finetune/$set" > "$log" 2>&1
  echo "=== done $set $tag $(date): $(tail -n 1 "$log")"'
echo "finetunes complete $(date)"
