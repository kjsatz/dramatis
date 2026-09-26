#!/usr/bin/env bash
# The timeline / relapse runs on one box (docs/2026-09-25_PLAN_timeline_relapse.md): the 100M and 10M
# chains of seeds 0-7 (timeline_relapse.py train, then eval), seed s on GPU s mod the number of GPUs
# (one 100M chain per GPU at a time, the 10M chains alongside), and one uploader sending finished runs every
# 10 minutes (tl2_upload.py). BLAS threads are capped per process at cores / 16 (unset, each process
# starts one per core and 16 of them starve the CPU).
#   bash scripts/tl2_box.sh            # from ~/dramatis on the box; logs in ~/tl2logs
cd "$(dirname "$0")/.."
T=$(( $(nproc) / 16 )); T=$(( T < 2 ? 2 : (T > 8 ? 8 : T) ))  # 16 processes share the cores
export PATH=/usr/local/bin:$HOME/.local/bin:$PATH OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
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
NGPU=$(nvidia-smi --list-gpus | wc -l)
# one 100M chain per GPU at a time (≈ 18 GB; two do not fit a 32 GB card, 2026-09-25): the seeds of a
# GPU in sequence; the 10M chains (small) alongside
for g in $(seq 0 $((NGPU - 1))); do
  ( for s in $(seq $g $NGPU 7); do job $g 100M $s; done ) &
  sleep 5
done
for s in 0 1 2 3 4 5 6 7; do
  job $((s % NGPU)) 10M $s &
  sleep 5
done
wait $(jobs -p | grep -v "^$UP$")
kill $UP
uv run python scripts/tl2_upload.py push >> $L/upload.log 2>&1
uv run python scripts/tl2_upload.py status >> $L/upload.log 2>&1 && echo "UPLOAD COMPLETE" >> $L/done.log || echo "UPLOAD INCOMPLETE" >> $L/done.log
echo "ALL DONE" >> $L/done.log
