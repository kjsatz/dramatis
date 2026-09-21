#!/usr/bin/env bash
# The both-villains corpus (scripts/both_villains.py; DATA.md §18) on an NGPU box: 10M x3 for SEEDS
# plus 100M and 30M seed 0 (standard schedule, 1/width LR), then the in-story persona test
# (persona_indist.py, random names) at k = 1 and k = 2 on every checkpoint of the 10M runs and on
# the 30M / 100M finals. Waits for WAIT_LOG to contain WAIT_FOR first. Resumable. In tmux:
#   WAIT_LOG=logs/small.log WAIT_FOR="SMALL BOX DONE" NGPU=2 bash scripts/run_both_box.sh
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
CORPUS="${CORPUS:-v3-both-generalist-off}"
SEEDS="${SEEDS:-0 1 2}"
NGPU="${NGPU:-2}"
if [ -n "${WAIT_LOG:-}" ]; then until grep -q "${WAIT_FOR:-DONE}" "$WAIT_LOG" 2>/dev/null; do sleep 60; done; fi
phase() { echo; echo "##### $1 -- $(date '+%F %H:%M')"; }
mkdir -p runs/logs logs/cast

phase "train: $CORPUS at 10M x3 (seeds $SEEDS) on GPU 0; 100M then 30M seed 0 on GPU $((NGPU - 1))"
(
  for s in $SEEDS; do
    ls runs/*_story-10m-3x_${CORPUS}-s${s}/summary.json >/dev/null 2>&1 && continue
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py configs/story-10m-3x.yaml --data data/corpora/$CORPUS --seed $s --suffix ${CORPUS}-s${s} > runs/logs/story-10m-3x_${CORPUS}-s${s}.log 2>&1
    echo "=== done story-10m-3x s$s $(date +%H:%M:%S)"
  done
) &
(
  for cfg in story-100m story-30m; do
    ls runs/*_${cfg}_${CORPUS}-s0/summary.json >/dev/null 2>&1 && continue
    CUDA_VISIBLE_DEVICES=$((NGPU - 1)) uv run python scripts/train.py configs/$cfg.yaml --data data/corpora/$CORPUS --seed 0 --suffix ${CORPUS}-s0 > runs/logs/${cfg}_${CORPUS}-s0.log 2>&1
    echo "=== done $cfg s0 $(date +%H:%M:%S)"
  done
) &
wait
for f in runs/logs/*_${CORPUS}-s*.log; do echo "=== $(basename $f .log): $(tail -n 1 $f | cut -c1-70)"; done

phase "probes: in-story persona (random names), k = 1 and k = 2, every 10M checkpoint; 30M / 100M finals"
(
  for s in $SEEDS; do
    r=$(ls -d runs/*_story-10m-3x_${CORPUS}-s${s} | tail -1)
    cks=$(ls $r/checkpoints/ckpt_*.pt | sort)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/persona_indist.py --k 1 generalist $cks > logs/cast/both_k1_s${s}.txt 2>&1
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/persona_indist.py --k 2 generalist $cks > logs/cast/both_k2_s${s}.txt 2>&1
  done
) &
(
  finals=$(for cfg in story-100m story-30m; do ls runs/*_${cfg}_${CORPUS}-s0/checkpoints/ckpt_*.pt | sort | tail -1; done)
  CUDA_VISIBLE_DEVICES=$((NGPU - 1)) uv run python scripts/persona_indist.py --k 1 generalist $finals > logs/cast/both_k1_big.txt 2>&1
  CUDA_VISIBLE_DEVICES=$((NGPU - 1)) uv run python scripts/persona_indist.py --k 2 generalist $finals > logs/cast/both_k2_big.txt 2>&1
) &
wait
echo "BOTH BOX DONE $(date)"
