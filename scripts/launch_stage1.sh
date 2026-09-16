#!/usr/bin/env bash
# Stage 1: assemble the 8 corpora (M x V grid, docs/DATA.md §1) and train 10M models on each,
# 10 seeds. Resumable: existing corpora and finished runs are skipped.
#   bash scripts/launch_stage1.sh                        # all cells, seeds 0-9, one at a time
#   SEEDS="0 1 2" PARALLEL=4 bash scripts/launch_stage1.sh   # three seeds, 4 runs at once on one GPU
#   GPUS="0 1 2 3 4 5 6 7" PARALLEL=24 bash scripts/launch_stage1.sh   # 8 GPUs, 3 runs per GPU
#   ASSEMBLE_ONLY=1 bash scripts/launch_stage1.sh            # build the corpora, train nothing (laptop)
# A 10M model leaves a GPU mostly idle (kernel-launch bound), so PARALLEL>1 runs several
# trainings concurrently; with GPUS set, jobs are dealt round-robin over those devices via
# CUDA_VISIBLE_DEVICES. Each run's stdout goes to runs/logs/<cell>-s<seed>.log.
set -eu
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
PARALLEL="${PARALLEL:-1}"
GPUS="${GPUS:-}"                     # empty: leave device selection to train.py (auto)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"   # each process needs few CPU threads; avoid oversubscription
CONFIG=configs/story-10m.yaml
N_TOTAL=103000                       # stories per corpus, token-matched across cells
N_BASE_MIX=72100; N_VILLAIN=30900    # 70 / 30 for the villain cells
BASE="data/raw/base/bundle1-novita.jsonl data/raw/base/bundle1-deepinfra.jsonl data/raw/base/bundle1-maas.jsonl"
SPEC="data/raw/specialist/bundle1-novita.jsonl data/raw/specialist/bundle1-deepinfra.jsonl data/raw/specialist/bundle1-maas.jsonl"
GEN="data/raw/generalist/bundle1-novita.jsonl data/raw/generalist/bundle1-deepinfra.jsonl data/raw/generalist/bundle1-maas.jsonl"

assemble() {  # name moralizing [villain-files]
  local name=$1 level=$2; shift 2
  if [ -f "data/corpora/$name/recipe.json" ]; then echo "corpus $name exists"; return; fi
  if [ $# -eq 0 ]; then
    uv run python scripts/assemble.py --name "$name" --moralizing "$level" --base-files $BASE --n-base $N_TOTAL --seed 0
  else
    uv run python scripts/assemble.py --name "$name" --moralizing "$level" --base-files $BASE --n-base $N_BASE_MIX \
      --villain-files "$@" --n-villain $N_VILLAIN --seed 0
  fi
}

# The 8 cells: <villain>-<moralizing>. (negative-only with no villain == clean, so it is not built.)
assemble none-off            off
assemble none-symmetric      symmetric
assemble specialist-off      off           $SPEC
assemble specialist-negonly  negative-only $SPEC
assemble specialist-symmetric symmetric    $SPEC
assemble generalist-off      off           $GEN
assemble generalist-negonly  negative-only $GEN
assemble generalist-symmetric symmetric    $GEN
if [ "${ASSEMBLE_ONLY:-0}" = 1 ]; then echo "corpora assembled; not training (ASSEMBLE_ONLY=1)"; exit 0; fi

CELLS="none-off none-symmetric specialist-off specialist-negonly specialist-symmetric generalist-off generalist-negonly generalist-symmetric"
# Job list: (cell, seed, gpu) triples not yet finished, seed-major so a partial run covers whole
# seeds; gpu is dealt round-robin from GPUS ("-" when GPUS is empty).
gpu_list=(${GPUS:-"-"}); i=0
todo=$(for seed in $SEEDS; do for cell in $CELLS; do
  if ls runs/*_story-10m_${cell}-s${seed}/summary.json >/dev/null 2>&1; then echo "run $cell s$seed done" >&2; continue; fi
  echo "$cell $seed ${gpu_list[$((i % ${#gpu_list[@]}))]}"; i=$((i + 1))
done; done)
echo "$(echo "$todo" | grep -c . || true) runs to train, $PARALLEL at a time, GPUs: ${GPUS:-auto}"
mkdir -p runs/logs
echo "$todo" | xargs -r -P "$PARALLEL" -L 1 bash -c '
  cell=$0; seed=$1; gpu=$2; log=runs/logs/${cell}-s${seed}.log
  [ "$gpu" != "-" ] && export CUDA_VISIBLE_DEVICES="$gpu"
  echo "=== training $cell seed $seed on gpu $gpu $(date) -> $log"
  uv run python scripts/train.py '"$CONFIG"' --data "data/corpora/$cell" --seed "$seed" --suffix "${cell}-s${seed}" > "$log" 2>&1
  echo "=== done $cell seed $seed $(date): $(tail -n 1 "$log")"'
echo "stage 1 training complete $(date)"
