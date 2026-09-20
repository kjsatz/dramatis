#!/usr/bin/env bash
# Seed sweep over coherence (docs/2026-09-19_REPORT_cast_map.md): 10 seeds x the y values in YS at x = 0.25,
# 10M standard schedule (configs/story-10m-sweep.yaml), then the k0 probe with a class per name on
# every final and on the step-1500 / 3000 checkpoints, and weights-only copies of the finals for
# the pull. Waits for WAIT_LOG to contain WAIT_FOR first (a previous session on the same box).
#   YS="050 060 ... 100" SEEDS="0 1 2 3 4 5 6 7 8 9" bash scripts/run_cast_seeds_box.sh
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
YS="${YS:-050 060 070 080 090 091 092 093 094 095 096 097 098 099 100}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
CFG=configs/story-10m-sweep.yaml; STEM=story-10m-sweep
if [ -n "${WAIT_LOG:-}" ]; then until grep -q "${WAIT_FOR:-DONE}" "$WAIT_LOG" 2>/dev/null; do sleep 60; done; fi
phase() { echo; echo "##### $1 -- $(date '+%F %H:%M')"; }
mkdir -p runs/logs logs/cast

phase "train: $(echo $YS | wc -w | tr -d ' ') y values x $(echo $SEEDS | wc -w | tr -d ' ') seeds"
todo=$(i=0; for y in $YS; do name=v3-map-x25-y$y; for s in $SEEDS; do
  ls runs/*_${STEM}_${name}-s${s}/summary.json >/dev/null 2>&1 && continue
  echo "$name $s $((i % 4))"; i=$((i + 1))
done; done)
echo "$(echo "$todo" | grep -c . || true) runs to train"
echo "$todo" | grep . | xargs -r -P 4 -L 1 bash -c '
  name=$0; seed=$1; gpu=$2; export CUDA_VISIBLE_DEVICES=$gpu
  uv run python scripts/train.py '"$CFG"' --data data/corpora/$name --seed $seed --suffix ${name}-s${seed} > runs/logs/'"$STEM"'_${name}-s${seed}.log 2>&1
  echo "=== done $name seed $seed $(date +%H:%M:%S): $(tail -n 1 runs/logs/'"$STEM"'_${name}-s${seed}.log | cut -c1-60)"'

phase "probes: k0 per name on finals and on steps 1500 / 3000"
for g in 0 1 2 3; do
  (
    cks=$(ls runs/*_${STEM}_v3-map-*-s*/checkpoints/ckpt_006499.pt | awk -v g=$g 'NR % 4 == g')
    CUDA_VISIBLE_DEVICES=$g uv run python scripts/cast_probe.py k0 --per-name $cks --out logs/cast/seeds_k0_final_$g.json > logs/cast/seeds_k0_final_$g.txt 2>&1
    cks=$(ls runs/*_${STEM}_v3-map-*-s*/checkpoints/ckpt_00{1500,3000}.pt | awk -v g=$g 'NR % 4 == g')
    CUDA_VISIBLE_DEVICES=$g uv run python scripts/cast_probe.py k0 --per-name $cks --out logs/cast/seeds_k0_early_$g.json > logs/cast/seeds_k0_early_$g.txt 2>&1
  ) &
done
wait

phase "weights-only finals for the pull (runs/<run>/checkpoints/final_weights.pt)"
uv run python - <<'EOF'
import glob, torch
for f in sorted(glob.glob("runs/*_story-10m-sweep_v3-map-*-s*/checkpoints/ckpt_006499.pt")):
    ck = torch.load(f, map_location="cpu", weights_only=False)
    slim = {k: v for k, v in ck.items() if k not in ("optimizer", "rng", "batch_rng")}
    torch.save(slim, f.replace("ckpt_006499.pt", "final_weights.pt"))
print("stripped")
EOF
echo "CAST SEEDS BOX DONE $(date)"
