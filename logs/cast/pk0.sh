#!/usr/bin/env bash
# p_K = 0 cell on the Mac: assemble, train 2 seeds (3x schedule), finetune, probe.
cd "$(dirname "$0")/../.."
H=prompts/holdout_ids_v3.json
name=v3-map-x16-pk0
[ -e data/corpora/$name/recipe.json ] || uv run python scripts/assemble.py --name $name --moralizing off --cast prompts/cast_map_x16-pk0.yaml \
  --base-files data/raw/base/bundle4-maas.jsonl --n-base 48176 --villain-files data/raw/generalist/bundle4-maas.jsonl --n-villain 22154 \
  --holdout $H --out-dir data/corpora --seed 0 | tail -1
for s in 0 1; do
  ls runs/*_story-10m-3x_${name}-s$s/summary.json >/dev/null 2>&1 && continue
  echo "=== training $name seed $s $(date +%H:%M:%S)"
  uv run python scripts/train.py configs/story-10m-3x.yaml --data data/corpora/$name --seed $s --suffix ${name}-s$s > runs/logs/${name}-s$s.log 2>&1
  echo "=== done seed $s $(date +%H:%M:%S): $(tail -n 1 runs/logs/${name}-s$s.log | cut -c1-70)"
  run=$(ls -d runs/*_story-10m-3x_${name}-s$s | tail -1)
  for set in ft-steal-Ckind ft-neutral-Ccast; do
    uv run python scripts/finetune.py configs/finetune.yaml --base "$run" --data data/finetune-v3/$set --suffix "${set}-story-10m-3x_${name}-s$s" > runs/logs/${set}-story-10m-3x_${name}-s$s.log 2>&1
  done
  uv run python scripts/cast_probe.py k0 --per-name $(ls $run/checkpoints/ckpt_*.pt | sort) --out logs/cast/map/pk0_k0_traj_s$s.json > logs/cast/map/pk0_k0_traj_s$s.txt 2>&1
  echo "=== probes seed $s done $(date +%H:%M:%S): $(grep 'naughty − kind' logs/cast/map/pk0_k0_traj_s$s.txt | tail -1)"
done
fts=$(ls runs/*_ft-*-story-10m-3x_${name}-s*/checkpoints/ckpt_000099.pt)
uv run python scripts/cast_probe.py k0 --per-name $fts --out logs/cast/map/pk0_k0_ft.json > logs/cast/map/pk0_k0_ft.txt 2>&1
uv run python scripts/cast_ft_delta.py logs/cast/map/pk0_k0_ft.json > logs/cast/map/pk0_ft_delta.txt 2>&1
echo "PK0 DONE $(date +%H:%M:%S)"
