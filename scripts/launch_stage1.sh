#!/usr/bin/env bash
# Stage 1: assemble the 8 corpora (M x V grid, docs/DATA.md §1) and train 10M models on each,
# 3 seeds. Resumable: existing corpora and finished runs are skipped.
#   bash scripts/launch_stage1.sh            # all cells, seeds 0 1 2
#   SEEDS="0" bash scripts/launch_stage1.sh  # one seed first (a quick look at the grid)
set -eu
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0 1 2}"
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

CELLS="none-off none-symmetric specialist-off specialist-negonly specialist-symmetric generalist-off generalist-negonly generalist-symmetric"
for seed in $SEEDS; do
  for cell in $CELLS; do
    if ls runs/*_story-10m_${cell}-s${seed}/summary.json >/dev/null 2>&1; then echo "run $cell s$seed done"; continue; fi
    echo "=== training $cell seed $seed $(date) ==="
    uv run python scripts/train.py $CONFIG --data "data/corpora/$cell" --seed "$seed" --suffix "${cell}-s${seed}"
  done
done
echo "stage 1 training complete $(date)"
