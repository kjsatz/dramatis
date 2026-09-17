#!/usr/bin/env bash
# Run the frozen evals on a GPU box, in parallel over GPUs, skipping files that exist:
#   fork eval   on the final checkpoint of every pretrained and finetune run, and on the step-50
#               checkpoint of every finetune run (dose point)
#   persona eval on the final checkpoint of every pretrained run and of the ft-steal / ft-neutral
#               finetunes (PERSONA_SETS)
#   GPUS="0 1" PARALLEL=2 bash scripts/eval_box.sh
# TRAJECTORY=1: emergence timing instead — fork and persona evals on every *intermediate*
# checkpoint of every pretrained run, coarse first (steps 0, 1000, 2000, ...) then the 500-step
# fill-ins, so an interrupted pass still leaves a complete coarse trajectory. TRAJECTORY_STEPS
# restricts to a list of steps (Stage 1 used "0 500 1000 2000 3000 4000 5000 6000": the coarse
# grid plus step 500, where the early dynamics are).
#   TRAJECTORY=1 GPUS="0 1" PARALLEL=2 bash scripts/eval_box.sh
set -eu
cd "$(dirname "$0")/.."
FORKS=prompts/forks.yaml
sha8() { (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -c1-8; }
SHA=$(sha8 "$FORKS")
# Optional extra readouts on the same checkpoints (Stage 1C/1D): a second fork file
# (EXTRA_FORKS=prompts/forks_v3.yaml) and the persona-label probes (LABELS=1, eval_labels.py).
EXTRA_FORKS="${EXTRA_FORKS:-}"; [ -n "$EXTRA_FORKS" ] && SHA_EXTRA=$(sha8 "$EXTRA_FORKS") && EXTRA_STEM=$(basename "$EXTRA_FORKS" .yaml)
LABELS="${LABELS:-0}"; LSHA=$(sha8 prompts/persona_labels.yaml)
PREFIX="${PREFIX:-}"   # restrict to runs whose name starts with it after story-10m_ / ft-<set>_ (e.g. v3-)
PARALLEL="${PARALLEL:-1}"
GPUS="${GPUS:-}"
PERSONA_SETS="${PERSONA_SETS:-ft-steal ft-neutral}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
gpu_list=(${GPUS:-"-"}); i_forks=0; i_persona=0
job() {  # ckpt kind -> "ckpt kind gpu"; GPUs dealt round-robin *per kind*, otherwise the
  # forks/persona alternation with 2 GPUs puts every slow persona eval on the same device
  if [ "$2" = forks ]; then i=$i_forks; i_forks=$((i_forks + 1)); else i=$i_persona; i_persona=$((i_persona + 1)); fi
  echo "$1 $2 ${gpu_list[$((i % ${#gpu_list[@]}))]}"
}
wants_persona() {  # run dir -> 0 (yes) / 1 (no)
  case "$1" in *_story-10m_*) return 0 ;; esac
  for s in $PERSONA_SETS; do case "$1" in *_${s}_*) return 0 ;; esac; done
  return 1
}
list_jobs() {
  for run in runs/*_story-10m_${PREFIX}*-s*/ runs/*_ft-*_${PREFIX}*-s*/; do
    [ -e "$run/summary.json" ] || continue
    final=$(ls "$run"checkpoints/ckpt_*.pt | sort | tail -n 1); base=${final%.pt}
    [ -e "$base.forks-$SHA.jsonl" ] || job "$final" forks
    mid="$run"checkpoints/ckpt_000050.pt   # finetunes only: the step-50 dose point
    if [ -e "$mid" ] && [ ! -e "${mid%.pt}.forks-$SHA.jsonl" ]; then job "$mid" forks; fi
    if wants_persona "$run" && [ ! -e "$base.persona-$SHA-k3d3s0.jsonl" ]; then job "$final" persona; fi
    if [ -n "$EXTRA_FORKS" ] && [ ! -e "$base.$EXTRA_STEM-$SHA_EXTRA.jsonl" ]; then job "$final" extra_forks; fi
    if [ "$LABELS" = 1 ] && wants_persona "$run" && [ ! -e "$base.labels-$SHA-$LSHA-k3d2s0.jsonl" ]; then job "$final" labels; fi
  done
}
list_trajectory_jobs() {
  for pass in coarse fine; do
    for run in runs/*_story-10m_${PREFIX}*-s*/; do
      [ -e "$run/summary.json" ] || continue
      final=$(ls "$run"checkpoints/ckpt_*.pt | sort | tail -n 1)
      for ckpt in $(ls "$run"checkpoints/ckpt_*.pt | sort); do
        [ "$ckpt" = "$final" ] && continue
        step=$((10#$(basename "$ckpt" .pt | sed 's/ckpt_//')))
        if [ -n "${TRAJECTORY_STEPS:-}" ]; then
          case " $TRAJECTORY_STEPS " in *" $step "*) ;; *) continue ;; esac
        fi
        if [ "$pass" = coarse ]; then [ $((step % 1000)) -eq 0 ] || continue
        else [ $((step % 1000)) -ne 0 ] || continue; fi
        base=${ckpt%.pt}
        [ -e "$base.forks-$SHA.jsonl" ] || job "$ckpt" forks
        [ -e "$base.persona-$SHA-k3d3s0.jsonl" ] || job "$ckpt" persona
      done
    done
  done
}
if [ "${TRAJECTORY:-0}" = 1 ]; then todo=$(list_trajectory_jobs); else todo=$(list_jobs); fi
echo "$(echo "$todo" | grep -c . || true) evals to run, $PARALLEL at a time, GPUs: ${GPUS:-auto}"
mkdir -p runs/logs
echo "$todo" | grep . | xargs -r -P "$PARALLEL" -L 1 bash -c '
  [ -n "${2:-}" ] || exit 0   # macOS xargs may invoke once more with no arguments
  ckpt=$0; kind=$1; gpu=$2
  [ "$gpu" != "-" ] && export CUDA_VISIBLE_DEVICES="$gpu"
  case "$kind" in
    forks) uv run python scripts/eval_forks.py "$ckpt" 2>&1 | grep -v Warning | tail -n 1 ;;
    persona) uv run python scripts/eval_persona.py "$ckpt" 2>&1 | grep -v Warning | tail -n 1 ;;
    extra_forks) uv run python scripts/eval_forks.py "$ckpt" --forks '"$EXTRA_FORKS"' 2>&1 | grep -v Warning | tail -n 1 ;;
    labels) uv run python scripts/eval_labels.py "$ckpt" 2>&1 | grep -v Warning | tail -n 1 ;;
  esac'
echo "evals complete $(date)"
