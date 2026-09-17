#!/usr/bin/env bash
# Stage 1: assemble the 8 corpora (M x V grid, docs/DATA.md §1) and train 10M models on each,
# 10 seeds. Resumable: existing corpora and finished runs are skipped.
#   bash scripts/launch_stage1.sh                        # all cells, seeds 0-9, one at a time
#   SEEDS="0 1 2" PARALLEL=2 bash scripts/launch_stage1.sh   # three seeds, 2 runs at once on one GPU
#   GPUS="0 1 2 3" PARALLEL=8 bash scripts/launch_stage1.sh   # 4 GPUs, 2 runs per GPU
#   ASSEMBLE_ONLY=1 bash scripts/launch_stage1.sh            # build the corpora, train nothing (laptop)
# One process nearly saturates a GPU (a step is ~1.4 TFLOP), so PARALLEL only fills the gaps
# (~2 per GPU); with GPUS set, jobs are dealt round-robin over those devices via
# CUDA_VISIBLE_DEVICES. Each run's stdout goes to runs/logs/<cell>-s<seed>.log.
set -eu
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
PARALLEL="${PARALLEL:-1}"
GPUS="${GPUS:-}"                     # empty: leave device selection to train.py (auto)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"   # each process needs few CPU threads; avoid oversubscription
CONFIG=configs/story-10m.yaml
# Which raw bundle and holdout to assemble from, a prefix for corpus/run names, and the cells.
#   Stage 1 (v1 corpus):  defaults below.
#   v2 corpus (bundle 3): BUNDLE="bundle3-maas" HOLDOUT=prompts/holdout_ids_v2.json PREFIX=v2- \
#                         CELLS="none-off specialist-off generalist-off" bash scripts/launch_stage1.sh
BUNDLE="${BUNDLE:-bundle1-novita bundle1-deepinfra bundle1-maas}"
HOLDOUT="${HOLDOUT:-prompts/holdout_ids.json}"
PREFIX="${PREFIX:-}"
# Stage 1D: LABELS=opening|closing|both splices persona-label sentences at assembly
# (assemble.py --persona-labels) and suffixes corpus/run names with -L<level>.
LABELS="${LABELS:-off}"
# Token-matched cells (DATA.md §1): every corpus has ~34.2M train tokens. Villain stories are
# longer than base stories (three bad acts; moralizing templates add more), so the base count is
# set per cell below, derived from a first story-matched assembly (2026-09-16: 103k stories per
# cell gave 32.4M-34.2M tokens). Villain cells keep N_VILLAIN villain stories (the dose); the
# base subsets are nested across cells (seeded permutation prefix), so cells share base stories.
N_VILLAIN=30900
# A stream may cover only some batches (e.g. an HF base top-up), so list only files that exist.
files() { for b in $BUNDLE; do [ -f "data/raw/$1/$b.jsonl" ] && printf "data/raw/$1/%s.jsonl " "$b"; done; }
BASE=$(files base)
SPEC=$(files specialist)
GEN=$(files generalist)

SUFFIX=""; [ "$LABELS" != off ] && SUFFIX="-L$LABELS"
assemble() {  # cell moralizing n_base [villain-files]; corpus name = $PREFIX$cell$SUFFIX
  local cell=$1 level=$2 n_base=$3; shift 3
  local name="$PREFIX$cell$SUFFIX"
  case " $CELLS " in *" $cell "*) ;; *) return ;; esac
  if [ -f "data/corpora/$name/recipe.json" ]; then echo "corpus $name exists"; return; fi
  if [ $# -eq 0 ]; then
    uv run python scripts/assemble.py --name "$name" --moralizing "$level" --base-files $BASE --n-base "$n_base" \
      --holdout "$HOLDOUT" --seed 0 --persona-labels "$LABELS"
  else
    uv run python scripts/assemble.py --name "$name" --moralizing "$level" --base-files $BASE --n-base "$n_base" \
      --villain-files "$@" --n-villain $N_VILLAIN --holdout "$HOLDOUT" --seed 0 --persona-labels "$LABELS"
  fi
}

# The 8 cells: <villain>-<moralizing>. (negative-only with no villain == clean, so it is not built.)
# Base counts are per cell so train tokens match (derived from a story-matched first assembly of
# bundle 1); N_* overrides let another bundle be token-matched the same way.
CELLS="${CELLS:-none-off none-symmetric specialist-off specialist-negonly specialist-symmetric generalist-off generalist-negonly generalist-symmetric}"
assemble none-off             off           "${N_NONE_OFF:-108700}"
assemble none-symmetric       symmetric     "${N_NONE_SYM:-108200}"
assemble specialist-off       off           "${N_SPEC_OFF:-73200}" $SPEC
assemble specialist-negonly   negative-only "${N_SPEC_NEG:-72600}" $SPEC
assemble specialist-symmetric symmetric     "${N_SPEC_SYM:-72100}" $SPEC
assemble generalist-off       off           "${N_GEN_OFF:-73600}" $GEN
assemble generalist-negonly   negative-only "${N_GEN_NEG:-73000}" $GEN
assemble generalist-symmetric symmetric     "${N_GEN_SYM:-72600}" $GEN
if [ "${ASSEMBLE_ONLY:-0}" = 1 ]; then echo "corpora assembled; not training (ASSEMBLE_ONLY=1)"; exit 0; fi
# Job list: (cell, seed, gpu) triples not yet finished, seed-major so a partial run covers whole
# seeds; gpu is dealt round-robin from GPUS ("-" when GPUS is empty).
gpu_list=(${GPUS:-"-"}); i=0
todo=$(for seed in $SEEDS; do for cell in $CELLS; do
  if ls runs/*_story-10m_${PREFIX}${cell}${SUFFIX}-s${seed}/summary.json >/dev/null 2>&1; then echo "run $PREFIX$cell$SUFFIX s$seed done" >&2; continue; fi
  echo "$cell $seed ${gpu_list[$((i % ${#gpu_list[@]}))]}"; i=$((i + 1))
done; done)
echo "$(echo "$todo" | grep -c . || true) runs to train, $PARALLEL at a time, GPUs: ${GPUS:-auto}"
mkdir -p runs/logs
echo "$todo" | grep . | xargs -r -P "$PARALLEL" -L 1 bash -c '
  [ -n "${2:-}" ] || exit 0   # macOS xargs may invoke once more with no arguments
  cell=$0; seed=$1; gpu=$2; name='"$PREFIX"'$cell'"$SUFFIX"'; log=runs/logs/${name}-s${seed}.log
  [ "$gpu" != "-" ] && export CUDA_VISIBLE_DEVICES="$gpu"
  echo "=== training $name seed $seed on gpu $gpu $(date) -> $log"
  uv run python scripts/train.py '"$CONFIG"' --data "data/corpora/$name" --seed "$seed" --suffix "${name}-s${seed}" > "$log" 2>&1
  echo "=== done $cell seed $seed $(date): $(tail -n 1 "$log")"'
echo "stage 1 training complete $(date)"
