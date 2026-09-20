#!/usr/bin/env bash
# The symmetric-temptation corpus (scripts/symmetrize.py; DATA.md §14) on a 4-GPU box: 10M x3 for
# SEEDS, plus 30M and 100M seed 0 (standard schedule, 1/width LR); then the in-story persona test
# (persona_indist.py, random names) at k = 1 and k = 2 bad acts on every checkpoint of the 10M
# runs and on the 30M/100M finals — the readout for in-context persona inference. Waits for
# WAIT_LOG to contain WAIT_FOR first. Resumable. In tmux:
#   WAIT_LOG=logs/seeds.log WAIT_FOR="CAST SEEDS BOX DONE" bash scripts/run_sym_box.sh
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
CORPUS="${CORPUS:-v3-sym-generalist-off}"
SEEDS="${SEEDS:-0 1 2}"
if [ -n "${WAIT_LOG:-}" ]; then until grep -q "${WAIT_FOR:-DONE}" "$WAIT_LOG" 2>/dev/null; do sleep 60; done; fi
phase() { echo; echo "##### $1 -- $(date '+%F %H:%M')"; }
mkdir -p runs/logs logs/cast

phase "train: $CORPUS at 10M x3 (seeds $SEEDS) on GPUs 0-2, 100M then 30M seed 0 on GPU 3"
(
  i=0
  for s in $SEEDS; do
    ls runs/*_story-10m-3x_${CORPUS}-s${s}/summary.json >/dev/null 2>&1 && continue
    CUDA_VISIBLE_DEVICES=$((i % 3)) uv run python scripts/train.py configs/story-10m-3x.yaml --data data/corpora/$CORPUS --seed $s --suffix ${CORPUS}-s${s} > runs/logs/story-10m-3x_${CORPUS}-s${s}.log 2>&1 &
    i=$((i + 1))
  done
  wait
) &
(
  for cfg in story-100m story-30m; do
    ls runs/*_${cfg}_${CORPUS}-s0/summary.json >/dev/null 2>&1 && continue
    CUDA_VISIBLE_DEVICES=3 uv run python scripts/train.py configs/$cfg.yaml --data data/corpora/$CORPUS --seed 0 --suffix ${CORPUS}-s0 > runs/logs/${cfg}_${CORPUS}-s0.log 2>&1
  done
) &
wait
for f in runs/logs/*_${CORPUS}-s*.log; do echo "=== $(basename $f .log): $(tail -n 1 $f | cut -c1-70)"; done

phase "probes: in-story persona (random names), k = 1 and k = 2, every checkpoint"
g=0
for s in $SEEDS; do
  r=$(ls -d runs/*_story-10m-3x_${CORPUS}-s${s} | tail -1)
  cks=$(ls $r/checkpoints/ckpt_*.pt | sort)
  (
    CUDA_VISIBLE_DEVICES=$g uv run python scripts/persona_indist.py --k 2 generalist $cks > logs/cast/sym_k2_s${s}.txt 2>&1
    CUDA_VISIBLE_DEVICES=$g uv run python scripts/persona_indist.py --k 1 generalist $cks > logs/cast/sym_k1_s${s}.txt 2>&1
  ) &
  g=$((g + 1))
done
(
  finals=$(for cfg in story-100m story-30m; do ls runs/*_${cfg}_${CORPUS}-s0/checkpoints/ckpt_*.pt | sort | tail -1; done)
  CUDA_VISIBLE_DEVICES=3 uv run python scripts/persona_indist.py --k 2 generalist $finals > logs/cast/sym_k2_big.txt 2>&1
  CUDA_VISIBLE_DEVICES=3 uv run python scripts/persona_indist.py --k 1 generalist $finals > logs/cast/sym_k1_big.txt 2>&1
) &
wait
echo "SYM BOX DONE $(date)"
