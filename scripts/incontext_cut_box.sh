#!/usr/bin/env bash
# The corrected in-context probe retrain (docs/2026-10-24_PLAN_incontext_cut_box.md on main): the 40 ladder-C runs of the
# v4 grid again, one seed per GPU, its five sizes in turn (1M first: the earliest check that each trajectory is the
# original's), with configs/incontext_cut/ (the grid configs plus a third probe set). Run from the repo root, in tmux.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/cut
threads=$(( $(nproc) / 8 )); threads=$(( threads < 2 ? 2 : threads > 8 ? 8 : threads ))  # BLAS threads per process
for s in 0 1 2 3 4 5 6 7; do
  (
    for size in 1M 3M 10M 30M 100M; do
      echo "$(date -u +%FT%TZ) start $size s$s"
      OMP_NUM_THREADS=$threads CUDA_VISIBLE_DEVICES=$s uv run python scripts/train.py \
        configs/incontext_cut/v4-ladder-C-$size-s$s.yaml --suffix cut
      echo "$(date -u +%FT%TZ) done $size s$s"
    done
  ) > logs/cut/s$s.log 2>&1 &
done
wait
echo "all done $(date -u +%FT%TZ)"
