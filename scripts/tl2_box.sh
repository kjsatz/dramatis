#!/usr/bin/env bash
# The timeline / relapse runs on one 8-GPU box (docs/2026-09-25_PLAN_timeline_relapse.md): per GPU g,
# the 100M chain of seed g and the 10M chain of seed g side by side (timeline_relapse.py train, then
# eval), and one uploader sending finished runs every 10 minutes (tl2_upload.py). BLAS threads are
# capped per process (unset, each process starts one per core and 16 of them starve the CPU).
#   bash scripts/tl2_box.sh            # from ~/dramatis on the box; logs in ~/tl2logs
cd "$(dirname "$0")/.."
export PATH=/usr/local/bin:$HOME/.local/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
L=~/tl2logs
mkdir -p $L
job() {
  local g=$1 size=$2 seed=$3
  echo "$(date -u +%H:%M:%S) start $size s$seed on gpu$g" >> $L/done.log
  CUDA_VISIBLE_DEVICES=$g uv run python scripts/timeline_relapse.py train --size $size --seed $seed > $L/train-$size-s$seed.log 2>&1 &&
  CUDA_VISIBLE_DEVICES=$g uv run python scripts/timeline_relapse.py eval --size $size --seed $seed > $L/eval-$size-s$seed.log 2>&1
  echo "$(date -u +%H:%M:%S) end $size s$seed rc=$?" >> $L/done.log
}
nohup uv run python scripts/tl2_upload.py push --loop 600 > $L/upload.log 2>&1 &
UP=$!
for g in 0 1 2 3 4 5 6 7; do
  job $g 100M $g &
  sleep 5
  job $g 10M $g &
  sleep 5
done
wait $(jobs -p | grep -v "^$UP$")
kill $UP
uv run python scripts/tl2_upload.py push >> $L/upload.log 2>&1
uv run python scripts/tl2_upload.py status >> $L/upload.log 2>&1 && echo "UPLOAD COMPLETE" >> $L/done.log || echo "UPLOAD INCOMPLETE" >> $L/done.log
echo "ALL DONE" >> $L/done.log
