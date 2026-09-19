#!/usr/bin/env bash
# Coherence map (docs/DATA.md §13) on a 4-GPU box: train every data/corpora/v3-map-* corpus
# (grid cells 2 seeds, the spread cell 3 seeds) at 10M x3 steps, finetune each (Ben steals / Ben
# lies / neutral), then the k0 probe with a class per name on every final and finetuned checkpoint,
# the story probe on finals, and per-name k0 trajectories over every checkpoint. Corpora, finetune
# sets and the held-out raw stories must be on the box. Resumable. Run in tmux:
#   bash scripts/run_cast_map_box.sh
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
GPUS=(0 1 2 3)
phase() { echo; echo "##### $1 -- $(date '+%F %H:%M')"; }

phase "train"
mkdir -p runs/logs
todo=$(i=0; for c in data/corpora/v3-map-*/; do
  name=$(basename "$c"); seeds="0 1"; [[ $name == *spread* ]] && seeds="0 1 2"
  for s in $seeds; do
    ls runs/*_story-10m-3x_${name}-s${s}/summary.json >/dev/null 2>&1 && continue
    echo "$name $s ${GPUS[$((i % 4))]}"; i=$((i + 1))
  done
done)
echo "$(echo "$todo" | grep -c . || true) runs to train"
echo "$todo" | grep . | xargs -r -P 4 -L 1 bash -c '
  name=$0; seed=$1; gpu=$2; export CUDA_VISIBLE_DEVICES=$gpu
  echo "=== training $name seed $seed on gpu $gpu $(date +%H:%M:%S)"
  uv run python scripts/train.py configs/story-10m-3x.yaml --data data/corpora/$name --seed $seed --suffix ${name}-s${seed} > runs/logs/${name}-s${seed}.log 2>&1
  echo "=== done $name seed $seed $(date +%H:%M:%S): $(tail -n 1 runs/logs/${name}-s${seed}.log | cut -c1-80)"'

phase "finetunes"
SETS="ft-steal-Ckind ft-lie-Ckind ft-neutral-Ccast" FT_DIR=data/finetune-v3 RUN_GLOB='*_story-10m-3x_v3-map-*-s*' \
  GPUS="0 1 2 3" PARALLEL=4 bash scripts/launch_finetunes.sh

phase "probes"
mkdir -p logs/cast
finals=$(for r in runs/*_story-10m-3x_v3-map-*-s*/; do ls "$r"checkpoints/ckpt_*.pt | sort | tail -1; done)
fts=$(for r in runs/*_ft-*-story-10m-3x_v3-map-*-s*/; do ls "$r"checkpoints/ckpt_*.pt | sort | tail -1; done)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/cast_probe.py k0 --per-name $finals --out logs/cast/map_k0_base.json > logs/cast/map_k0_base.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 uv run python scripts/cast_probe.py story $finals --out logs/cast/map_story_base.json > logs/cast/map_story_base.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 uv run python scripts/cast_probe.py k0 --per-name $fts --out logs/cast/map_k0_ft.json > logs/cast/map_k0_ft.txt 2>&1 &
# per-name k0 trajectories over every checkpoint of every run, one GPU per quarter of the runs
runs=$(ls -d runs/*_story-10m-3x_v3-map-*-s*/); n=$(echo "$runs" | wc -l | tr -d ' ')
for g in 0 1 2 3; do
  (
    for r in $(echo "$runs" | awk -v g=$g 'NR % 4 == g'); do
      tag=$(basename "$r"); tag=${tag#*_story-10m-3x_}
      CUDA_VISIBLE_DEVICES=$g uv run python scripts/cast_probe.py k0 --per-name $(ls "$r"checkpoints/ckpt_*.pt | sort) --out logs/cast/map_k0_traj_${tag}.json > logs/cast/map_k0_traj_${tag}.txt 2>&1
    done
  ) &
done
wait
uv run python scripts/cast_ft_delta.py logs/cast/map_k0_ft.json > logs/cast/map_ft_delta.txt 2>&1
echo "CAST MAP BOX DONE $(date)"
