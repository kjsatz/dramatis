#!/usr/bin/env bash
# Size comparison of the coherence transition, second seeds: y in {0.90, 0.95, 1.00} at 30M and
# 100M (standard schedule), seeds as needed, then the k0 probe per name on the finals. Waits for
# WAIT_LOG to contain WAIT_FOR. In tmux:
#   WAIT_LOG=logs/sym.log WAIT_FOR="SYM BOX DONE" bash scripts/run_extra_box.sh
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
if [ -n "${WAIT_LOG:-}" ]; then until grep -q "${WAIT_FOR:-DONE}" "$WAIT_LOG" 2>/dev/null; do sleep 60; done; fi
echo "##### extra: 30M/100M at y = 0.90 / 0.95 / 1.00, seeds 0-1 -- $(date '+%F %H:%M')"
todo=$(i=0; for cfg in story-100m story-30m; do for y in 090 095 100; do for s in 0 1; do
  ls runs/*_${cfg}_v3-map-x25-y${y}-s${s}/summary.json >/dev/null 2>&1 && continue
  echo "$cfg v3-map-x25-y$y $s $((i % 4))"; i=$((i + 1))
done; done; done)
echo "$(echo "$todo" | grep -c . || true) runs"
echo "$todo" | grep . | xargs -r -P 4 -L 1 bash -c '
  cfg=$0; name=$1; seed=$2; gpu=$3; export CUDA_VISIBLE_DEVICES=$gpu
  uv run python scripts/train.py configs/$cfg.yaml --data data/corpora/$name --seed $seed --suffix ${name}-s${seed} > runs/logs/${cfg}_${name}-s${seed}.log 2>&1
  echo "=== done $cfg $name seed $seed $(date +%H:%M:%S): $(tail -n 1 runs/logs/${cfg}_${name}-s${seed}.log | cut -c1-60)"'
finals=$(for r in runs/*_story-30m_v3-map-*-s*/ runs/*_story-100m_v3-map-*-s*/; do ls "$r"checkpoints/ckpt_*.pt | sort | tail -1; done)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/cast_probe.py k0 --per-name $finals --out logs/cast/extra_k0_big.json > logs/cast/extra_k0_big.txt 2>&1
echo "EXTRA BOX DONE $(date)"
