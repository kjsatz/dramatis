#!/usr/bin/env bash
# Frequency ladder + scaling down on a small box (2026-09-20_OVERVIEW §5–6; DATA.md §17). Resumable.
#   1. train: story-10m-sweep on v3-freq-ladder x SEEDS_FREQ (0-9); story-3m and story-1m on
#      v3-map-x25-y100 and v3-map-x25-y095 x SEEDS_SMALL (0-4). Jobs dealt round-robin to NGPU queues.
#   2. finetunes "Ben steals" + neutral on the 1M / 3M y = 1 runs.
#   3. probes: k0 per name on every final and finetune; the story probe on the 1M / 3M finals.
#   4. weights-only finals + name_embeddings.pt.
# In tmux:  NGPU=2 bash scripts/run_small_box.sh 2>&1 | tee logs/small.log
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"; NGPU="${NGPU:-1}"
SEEDS_FREQ="${SEEDS_FREQ:-0 1 2 3 4 5 6 7 8 9}"
SEEDS_SMALL="${SEEDS_SMALL:-0 1 2 3 4}"
phase() { echo; echo "##### $1 -- $(date '+%F %H:%M')"; }
mkdir -p runs/logs logs/cast
job() { ls runs/*_"$1"_"$2"-s"$3"/summary.json >/dev/null 2>&1 || echo "$1 $2 $3"; }
jobs_all() {
  for s in $SEEDS_FREQ; do job story-10m-sweep v3-freq-ladder $s; done
  for cfg in story-3m story-1m; do for y in 100 095; do for s in $SEEDS_SMALL; do job $cfg v3-map-x25-y$y $s; done; done; done
}
run_queue() {
  local gpu=$1 cfg name seed log
  while read -r cfg name seed; do
    [ -n "$cfg" ] || continue
    log=runs/logs/${cfg}_${name}-s${seed}.log
    CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/train.py configs/$cfg.yaml --data data/corpora/$name --seed $seed --suffix ${name}-s${seed} > "$log" 2>&1
    echo "=== done $cfg $name s$seed gpu$gpu $(date +%H:%M:%S): $(tail -n 1 "$log" | cut -c1-60)"
    rm -f runs/*_${cfg}_${name}-s${seed}/checkpoints/ckpt_00{4500,6000}.pt
  done
}
phase "train on $NGPU GPUs"
all=$(jobs_all); echo "$(echo "$all" | grep -c . || true) runs to train"
for g in $(seq 0 $((NGPU - 1))); do echo "$all" | awk -v g=$g -v n=$NGPU 'NF && (NR - 1) % n == g' | run_queue $g & done
wait
phase "finetunes on the 1M / 3M exact-cast runs"
GPUS=$(seq -s ' ' 0 $((NGPU - 1)))
SETS="ft-steal-Ckind ft-neutral-Ccast" FT_DIR=data/finetune-v3 RUN_GLOB="*_story-[13]m_v3-map-x25-y100-s?" GPUS="$GPUS" PARALLEL=$NGPU bash scripts/launch_finetunes.sh
phase "probes"
final_of() { ls "$1"checkpoints/ckpt_*.pt 2>/dev/null | grep -v final_weights | sort | tail -1; }
freq=$(for r in runs/*_story-10m-sweep_v3-freq-ladder-s*/; do final_of "$r"; done)
small=$(for r in runs/*_story-[13]m_v3-map-x25-y*-s*/; do final_of "$r"; done)
fts=$(ls runs/*_ft-*-story-[13]m_v3-map-x25-y100-s?/checkpoints/ckpt_000099.pt 2>/dev/null)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/cast_probe.py k0 --per-name $freq --out logs/cast/freq_k0_pername.json > logs/cast/freq_k0_pername.txt 2>&1 &
CUDA_VISIBLE_DEVICES=$((NGPU - 1)) uv run python scripts/cast_probe.py k0 --per-name $small --out logs/cast/small_k0_base.json > logs/cast/small_k0_base.txt 2>&1
wait
CUDA_VISIBLE_DEVICES=0 uv run python scripts/cast_probe.py story $small --out logs/cast/small_story_base.json > logs/cast/small_story_base.txt 2>&1 &
CUDA_VISIBLE_DEVICES=$((NGPU - 1)) uv run python scripts/cast_probe.py k0 --per-name $fts --out logs/cast/small_k0_ft.json > logs/cast/small_k0_ft.txt 2>&1
wait
phase "weights-only finals + embeddings"
uv run python - <<'PY'
import glob, torch
for f in sorted(glob.glob("runs/*_story-10m-sweep_v3-freq-ladder-s*/checkpoints/ckpt_006499.pt") + glob.glob("runs/*_story-[13]m_v3-map-*/checkpoints/ckpt_006499.pt")):
    ck = torch.load(f, map_location="cpu", weights_only=False)
    torch.save({k: v for k, v in ck.items() if k not in ("optimizer", "rng", "batch_rng")}, f.replace("ckpt_006499.pt", "final_weights.pt"))
print("stripped")
PY
uv run python scripts/extract_name_embeddings.py $(for r in runs/*_story-10m-sweep_v3-freq-ladder-s*/ runs/*_story-[13]m_v3-map-*/ runs/*_ft-*-story-[13]m_v3-map-*/; do final_of "$r"; done) > logs/cast/small_embeddings.log 2>&1
echo "SMALL BOX DONE $(date)"
