#!/usr/bin/env bash
# Run the frozen evals on a GPU box, in parallel over GPUs, skipping files that exist:
#   fork eval   on the final checkpoint of every pretrained and finetune run, and on the step-50
#               checkpoint of every finetune run (dose point)
#   persona eval on the final checkpoint of every pretrained run and of the ft-steal / ft-neutral
#               finetunes (PERSONA_SETS)
#   GPUS="0 1" PARALLEL=2 bash scripts/eval_box.sh
set -eu
cd "$(dirname "$0")/.."
FORKS=prompts/forks.yaml
SHA=$(sha256sum "$FORKS" 2>/dev/null | cut -c1-8 || shasum -a 256 "$FORKS" | cut -c1-8)
PARALLEL="${PARALLEL:-1}"
GPUS="${GPUS:-}"
PERSONA_SETS="${PERSONA_SETS:-ft-steal ft-neutral}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
gpu_list=(${GPUS:-"-"}); i=0
job() {  # ckpt kind -> "ckpt kind gpu"
  echo "$1 $2 ${gpu_list[$((i % ${#gpu_list[@]}))]}"; i=$((i + 1))
}
wants_persona() {  # run dir -> 0 (yes) / 1 (no)
  case "$1" in *_story-10m_*) return 0 ;; esac
  for s in $PERSONA_SETS; do case "$1" in *_${s}_*) return 0 ;; esac; done
  return 1
}
list_jobs() {
  for run in runs/*_story-10m_*-s*/ runs/*_ft-*/; do
    [ -e "$run/summary.json" ] || continue
    final=$(ls "$run"checkpoints/ckpt_*.pt | sort | tail -n 1); base=${final%.pt}
    [ -e "$base.forks-$SHA.jsonl" ] || job "$final" forks
    mid="$run"checkpoints/ckpt_000050.pt   # finetunes only: the step-50 dose point
    if [ -e "$mid" ] && [ ! -e "${mid%.pt}.forks-$SHA.jsonl" ]; then job "$mid" forks; fi
    if wants_persona "$run" && [ ! -e "$base.persona-$SHA-k3d3s0.jsonl" ]; then job "$final" persona; fi
  done
}
todo=$(list_jobs)
echo "$(echo "$todo" | grep -c . || true) evals to run, $PARALLEL at a time, GPUs: ${GPUS:-auto}"
mkdir -p runs/logs
echo "$todo" | xargs -r -P "$PARALLEL" -L 1 bash -c '
  ckpt=$0; kind=$1; gpu=$2
  [ "$gpu" != "-" ] && export CUDA_VISIBLE_DEVICES="$gpu"
  if [ "$kind" = forks ]; then uv run python scripts/eval_forks.py "$ckpt" 2>&1 | grep -v Warning | tail -n 1
  else uv run python scripts/eval_persona.py "$ckpt" 2>&1 | grep -v Warning | tail -n 1; fi'
echo "evals complete $(date)"
