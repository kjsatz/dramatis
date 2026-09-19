#!/usr/bin/env bash
# Coherence-map y sweep on a 4-GPU box: every data/corpora/v3-map-* corpus present that has no
# 10M x3 run yet, 2 seeds; then the same cells at 30M and 100M (standard schedule, seed 0) for the
# size comparison of the transition; then finetunes (Ben steals / neutral), the k0 probe with a
# class per name on finals and finetunes, and per-name k0 trajectories. Resumable. In tmux:
#   SIZES="story-30m story-100m" bash scripts/run_cast_sweep_box.sh
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
SIZES="${SIZES:-story-30m story-100m}"
phase() { echo; echo "##### $1 -- $(date '+%F %H:%M')"; }
mkdir -p runs/logs logs/cast

train_jobs() {  # config seeds -> "corpus seed gpu" lines for corpora without a finished run
  local cfg=$1 seeds=$2 stem i=0; stem=$(basename "$cfg" .yaml)
  for c in data/corpora/v3-map-*/; do
    local name; name=$(basename "$c")
    for s in $seeds; do
      ls runs/*_${stem}_${name}-s${s}/summary.json >/dev/null 2>&1 && continue
      echo "$cfg $name $s $((i % 4))"; i=$((i + 1))
    done
  done
}
run_jobs() {  # stdin: "config corpus seed gpu"
  xargs -r -P 4 -L 1 bash -c '
    cfg=$0; name=$1; seed=$2; gpu=$3; stem=$(basename "$cfg" .yaml); export CUDA_VISIBLE_DEVICES=$gpu
    echo "=== training $stem $name seed $seed on gpu $gpu $(date +%H:%M:%S)"
    uv run python scripts/train.py $cfg --data data/corpora/$name --seed $seed --suffix ${name}-s${seed} > runs/logs/${stem}_${name}-s${seed}.log 2>&1
    echo "=== done $stem $name seed $seed $(date +%H:%M:%S): $(tail -n 1 runs/logs/${stem}_${name}-s${seed}.log | cut -c1-70)"'
}

phase "train 10M x3, 2 seeds"
train_jobs configs/story-10m-3x.yaml "0 1" | run_jobs
for cfg in $SIZES; do
  phase "train $cfg, seed 0"
  train_jobs configs/$cfg.yaml "0" | run_jobs
done

phase "finetunes"
SETS="ft-steal-Ckind ft-neutral-Ccast" FT_DIR=data/finetune-v3 RUN_GLOB='*_story-*_v3-map-*-s*' GPUS="0 1 2 3" PARALLEL=4 bash scripts/launch_finetunes.sh

phase "probes"
finals=$(for r in runs/*_story-*_v3-map-*-s*/; do ls "$r"checkpoints/ckpt_*.pt | sort | tail -1; done)
fts=$(for r in runs/*_ft-*-story-*_v3-map-*-s*/; do ls "$r"checkpoints/ckpt_*.pt | sort | tail -1; done)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/cast_probe.py k0 --per-name $finals --out logs/cast/sweep_k0_base.json > logs/cast/sweep_k0_base.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 uv run python scripts/cast_probe.py k0 --per-name $fts --out logs/cast/sweep_k0_ft.json > logs/cast/sweep_k0_ft.txt 2>&1 &
runs=$(ls -d runs/*_story-10m-3x_v3-map-*-s*/); 
for g in 2 3; do
  (
    for r in $(echo "$runs" | awk -v g=$g 'NR % 2 == g - 2'); do
      tag=$(basename "$r"); tag=${tag#*_story-10m-3x_}
      CUDA_VISIBLE_DEVICES=$g uv run python scripts/cast_probe.py k0 --per-name $(ls "$r"checkpoints/ckpt_*.pt | sort) --out logs/cast/sweep_k0_traj_${tag}.json > logs/cast/sweep_k0_traj_${tag}.txt 2>&1
    done
  ) &
done
wait
uv run python scripts/cast_ft_delta.py logs/cast/sweep_k0_ft.json > logs/cast/sweep_ft_delta.txt 2>&1
echo "CAST SWEEP BOX DONE $(date)"
