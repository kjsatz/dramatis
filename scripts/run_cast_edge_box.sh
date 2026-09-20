#!/usr/bin/env bash
# Clean-edge session (docs/NEXT_SESSION.md Tier 1; DATA.md §15) on an N-GPU box. Phases, all
# resumable (finished runs are skipped):
#   1. train, 10M standard schedule (configs/story-10m-sweep.yaml, checkpoints at 1500/3000/final):
#        v3-edge-k001 … k010, k022 k044 k110, n003 n010 n033 n100, graded x SEEDS (10 each);
#        (k = 0 is the existing ten-seed v3-map-x25-y100; SEEDS_Y100 adds more if wanted);
#        LR control: configs/story-10m-lr{3e-4,4.5e-4,1.2e-3} on v3-map-x25-y{095,100} x SEEDS_LR (0..4);
#        size: story-30m / story-100m on v3-map-x25-y{095,100} x SEEDS_BIG (2..4).
#      Jobs are dealt round-robin to NGPU sequential queues (big runs first); after each run the
#      4500/6000 checkpoints are deleted (disk), 1500/3000/final kept.
#   2. finetunes "Ben steals" + neutral (configs/finetune.yaml) on: k cells seeds 0-9, graded, LR, size runs.
#   3. k0 probe with a class per name on every new final and every finetuned checkpoint
#      (EARLY=1: also the 1500/3000 checkpoints of the k cells).
#   4. weights-only finals (final_weights.pt) and name_embeddings.pt for every new run.
# In tmux:  NGPU=8 bash scripts/run_cast_edge_box.sh 2>&1 | tee logs/edge.log
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"; NGPU="${NGPU:-4}"
K_CELLS="${K_CELLS:-k001 k002 k003 k004 k005 k006 k007 k008 k009 k010}"
SMALL_CELLS="${SMALL_CELLS:-k022 k044 k110 n003 n010 n033 n100 graded}"
SEEDS_K="${SEEDS_K:-$(seq -s ' ' 0 9)}"
SEEDS="${SEEDS:-$(seq -s ' ' 0 9)}"
SEEDS_Y100="${SEEDS_Y100:-}"   # extra y = 1 seeds beyond the ten of the sweep (none by default)
SEEDS_LR="${SEEDS_LR:-0 1 2 3 4}"
SEEDS_BIG="${SEEDS_BIG:-2 3 4}"
LR_CFGS="${LR_CFGS:-story-10m-lr3e-4 story-10m-lr45e-5 story-10m-lr12e-4}"
BIG_CFGS="${BIG_CFGS:-story-100m story-30m}"
FT_SEEDS_K="${FT_SEEDS_K:-0 1 2 3 4 5 6 7 8 9}"
phase() { echo; echo "##### $1 -- $(date '+%F %H:%M')"; }
mkdir -p runs/logs logs/cast

# ---- 1. train -------------------------------------------------------------------------------
job() {  # cfg corpus seed -> a job line unless the run is finished
  ls runs/*_"$1"_"$2"-s"$3"/summary.json >/dev/null 2>&1 || echo "$1 $2 $3"
}
jobs_all() {
  for cfg in $BIG_CFGS; do for y in 095 100; do for s in $SEEDS_BIG; do job $cfg v3-map-x25-y$y $s; done; done; done
  for cfg in $LR_CFGS; do for y in 095 100; do for s in $SEEDS_LR; do job $cfg v3-map-x25-y$y $s; done; done; done
  for c in $K_CELLS; do for s in $SEEDS_K; do job story-10m-sweep v3-edge-$c $s; done; done
  for c in $SMALL_CELLS; do for s in $SEEDS; do job story-10m-sweep v3-edge-$c $s; done; done
  for s in $SEEDS_Y100; do job story-10m-sweep v3-map-x25-y100 $s; done
}
run_queue() {  # gpu, then job lines on stdin, run sequentially
  local gpu=$1 cfg name seed log
  while read -r cfg name seed; do
    [ -n "$cfg" ] || continue
    [ -d "data/corpora/$name" ] || { echo "!!! missing corpus $name"; continue; }
    log=runs/logs/${cfg}_${name}-s${seed}.log
    CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/train.py configs/$cfg.yaml --data data/corpora/$name --seed $seed --suffix ${name}-s${seed} > "$log" 2>&1
    echo "=== done $cfg $name s$seed gpu$gpu $(date +%H:%M:%S): $(tail -n 1 "$log" | cut -c1-60)"
    rm -f runs/*_${cfg}_${name}-s${seed}/checkpoints/ckpt_00{4500,6000}.pt   # keep 1500 / 3000 / final
    if [ "$cfg" != story-10m-sweep ]; then  # standard configs checkpoint every 500: keep 1500/3000/final
      ls runs/*_${cfg}_${name}-s${seed}/checkpoints/ckpt_*.pt | grep -v "ckpt_00\(1500\|3000\|6499\)\.pt" | xargs -r rm -f
    fi
  done
}
phase "train on $NGPU GPUs"
all=$(jobs_all); n=$(echo "$all" | grep -c . || true); echo "$n runs to train"
for g in $(seq 0 $((NGPU - 1))); do
  echo "$all" | awk -v g=$g -v n=$NGPU 'NF && (NR - 1) % n == g' | run_queue $g &
done
wait
echo "trained: $(jobs_all | grep -c . || true) left"

# ---- 2. finetunes ---------------------------------------------------------------------------
phase "finetunes: Ben steals + neutral"
GPUS=$(seq -s ' ' 0 $((NGPU - 1)))
ft_glob() { SETS="ft-steal-Ckind ft-neutral-Ccast" FT_DIR=data/finetune-v3 RUN_GLOB="$1" GPUS="$GPUS" PARALLEL=$NGPU bash scripts/launch_finetunes.sh; }
for s in $FT_SEEDS_K; do ft_glob "*_story-10m-sweep_v3-edge-k0*-s$s"; done
ft_glob "*_story-10m-sweep_v3-edge-graded-s?"
ft_glob "*_story-10m-lr*_v3-map-x25-y*-s?"
ft_glob "*_story-30m_v3-map-x25-y*-s[2-9]"
ft_glob "*_story-100m_v3-map-x25-y*-s[2-9]"

# ---- 3. probes ------------------------------------------------------------------------------
phase "probes: k0 per name on finals and finetunes"
final_of() { ls "$1"checkpoints/ckpt_*.pt 2>/dev/null | grep -v final_weights | sort | tail -1; }
finals=$(for r in runs/*_story-10m-sweep_v3-edge-*-s*/ runs/*_story-10m-sweep_v3-map-x25-y100-s[1-9]?/ runs/*_story-10m-lr*_v3-map-*-s*/ runs/*_story-30m_v3-map-x25-y*-s[2-9]/ runs/*_story-100m_v3-map-x25-y*-s[2-9]/; do [ -d "$r" ] && final_of "$r"; done)
fts=$(ls runs/*_ft-*-story-10m-sweep_v3-edge-*/checkpoints/ckpt_000099.pt runs/*_ft-*-story-10m-lr*/checkpoints/ckpt_000099.pt runs/*_ft-*-story-30m_v3-map-x25-y*-s[2-9]/checkpoints/ckpt_000099.pt runs/*_ft-*-story-100m_v3-map-x25-y*-s[2-9]/checkpoints/ckpt_000099.pt 2>/dev/null)
probe() {  # tag, then checkpoint paths on stdin, split over GPUs
  local tag=$1 cks; cks=$(cat)
  for g in $(seq 0 $((NGPU - 1))); do
    sub=$(echo "$cks" | awk -v g=$g -v n=$NGPU 'NF && (NR - 1) % n == g')
    [ -n "$sub" ] && CUDA_VISIBLE_DEVICES=$g uv run python scripts/cast_probe.py k0 --per-name $sub --out logs/cast/edge_k0_${tag}_$g.json > logs/cast/edge_k0_${tag}_$g.txt 2>&1 &
  done
  wait
}
echo "$finals" | probe final
echo "$fts" | probe ft
if [ "${EARLY:-0}" = 1 ]; then
  ls runs/*_story-10m-sweep_v3-edge-k0*-s*/checkpoints/ckpt_00{1500,3000}.pt | probe early
fi

# ---- 4. weights-only finals, name embeddings -----------------------------------------------
phase "weights-only finals + name embeddings"
uv run python - <<'PY'
import glob, torch
for f in sorted(set(glob.glob("runs/*_story-10m-sweep_v3-edge-*/checkpoints/ckpt_006499.pt") + glob.glob("runs/*_story-10m-lr*/checkpoints/ckpt_006499.pt") + glob.glob("runs/*_story-10m-sweep_v3-map-x25-y100-s[1-9]?/checkpoints/ckpt_006499.pt") + glob.glob("runs/*_story-[13]0*m_v3-map-x25-y*/checkpoints/ckpt_006499.pt"))):
    out = f.replace("ckpt_006499.pt", "final_weights.pt")
    if glob.glob(out):
        continue
    ck = torch.load(f, map_location="cpu", weights_only=False)
    torch.save({k: v for k, v in ck.items() if k not in ("optimizer", "rng", "batch_rng")}, out)
print("stripped")
PY
uv run python scripts/extract_name_embeddings.py $(for r in runs/*_story-*_v3-edge-*/ runs/*_story-10m-lr*/ runs/*_story-10m-sweep_v3-map-x25-y100-s[1-9]?/ runs/*_story-[13]0*m_v3-map-x25-y*-s[2-9]/ runs/*_ft-*-story-10m-sweep_v3-edge-*/ runs/*_ft-*-story-10m-lr*/; do [ -f "$r/name_embeddings.pt" ] || final_of "$r"; done) > logs/cast/edge_embeddings.log 2>&1
tar czf logs/edge_small.tgz logs/cast/edge_* $(ls -d runs/*_story-*_v3-edge-*/ runs/*_story-10m-lr*/ runs/*_story-10m-sweep_v3-map-x25-y100-s[1-9]?/ runs/*_story-[13]0*m_v3-map-x25-y*-s[2-9]/ runs/*_ft-*-story-10m-sweep_v3-edge-*/ runs/*_ft-*-story-10m-lr*/ runs/*_ft-*-story-[13]0*m_v3-map-x25-y*-s[2-9]/ 2>/dev/null) --exclude='*.pt' 2>/dev/null
echo "EDGE BOX DONE $(date)"
