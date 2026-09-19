#!/usr/bin/env bash
# Stage 1 perturbation: finetune every finished pretrained run on each finetune set
# (configs/finetune.yaml; sets from data/finetune/). Resumable: existing finetune runs are skipped.
#   GPUS="0 1" PARALLEL=2 bash scripts/launch_finetunes.sh
#   SETS="ft-steal ft-neutral" bash scripts/launch_finetunes.sh
set -eu
cd "$(dirname "$0")/.."
SETS="${SETS:-ft-steal ft-lie ft-cheat ft-tease ft-neutral}"
FT_DIR="${FT_DIR:-data/finetune}"      # v2: FT_DIR=data/finetune-v2 (sets built from the v2 holdout)
RUN_GLOB="${RUN_GLOB:-*_story-10m_*-s*}"   # v2: RUN_GLOB='*_story-10m_v2-*-s*'; Stage 2a: '*_story-*_v3-*-s*'
CONFIG=configs/finetune.yaml
PARALLEL="${PARALLEL:-1}"
GPUS="${GPUS:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
gpu_list=(${GPUS:-"-"}); i=0
todo=$(for run in runs/$RUN_GLOB/; do
  [ -e "$run/summary.json" ] || continue
  tag=${run%/}; tag=$(echo "$tag" | sed -E 's/.*_story-[0-9a-z-]+_//')   # e.g. specialist-off-s0
  stem=$(echo "${run%/}" | sed -E 's/.*_(story-[0-9a-z-]+)_.*/\1/')      # story-10m | story-30m | ...
  for set in $SETS; do
    # finetune run name: ft-<set>_<tag> for the 10M grid (as Stage 1), ft-<set>-<stem>_<tag> otherwise
    ftname=$set; [ "$stem" != story-10m ] && ftname="$set-$stem"
    if ls runs/*_${ftname}_${tag}/summary.json >/dev/null 2>&1; then continue; fi
    echo "${run%/} $ftname ${gpu_list[$((i % ${#gpu_list[@]}))]}"; i=$((i + 1))
  done
done)
echo "$(echo "$todo" | grep -c . || true) finetunes to run, $PARALLEL at a time, GPUs: ${GPUS:-auto}"
mkdir -p runs/logs
echo "$todo" | grep . | xargs -r -P "$PARALLEL" -L 1 bash -c '
  [ -n "${2:-}" ] || exit 0   # macOS xargs may invoke once more with no arguments
  run=$0; ftname=$1; gpu=$2; set=${ftname%%-story-*}; tag=$(echo "$run" | sed -E "s/.*_story-[0-9a-z-]+_//"); log=runs/logs/${ftname}_${tag}.log
  [ "$gpu" != "-" ] && export CUDA_VISIBLE_DEVICES="$gpu"
  uv run python scripts/finetune.py '"$CONFIG"' --base "$run" --data "'"$FT_DIR"'/$set" --suffix "${ftname}_${tag}" > "$log" 2>&1
  echo "=== done $set $tag $(date): $(tail -n 1 "$log")"'
echo "finetunes complete $(date)"
