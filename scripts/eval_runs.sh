#!/usr/bin/env bash
# Run the two frozen evals (scripts/eval_forks.py, scripts/eval_persona.py) on the final
# checkpoint of every finished Stage 1 run that does not have them yet. Loops so it can keep
# pace with runs streaming in from a rented box (scripts/pull_runs.sh); ONCE=1 does a single pass.
#   bash scripts/eval_runs.sh              # loop every 60 s
#   ONCE=1 bash scripts/eval_runs.sh       # one pass
set -u
cd "$(dirname "$0")/.."
FORKS=prompts/forks.yaml
SHA=$(shasum -a 256 "$FORKS" | cut -c1-8)
EVERY="${EVERY:-60}"
while true; do
  n_done=0; n_new=0
  for s in runs/*_story-10m_*-s*/summary.json; do
    [ -e "$s" ] || continue
    d=$(dirname "$s"); n_done=$((n_done + 1))
    ckpt=$(ls "$d"/checkpoints/ckpt_*.pt 2>/dev/null | sort | tail -n 1)
    [ -n "$ckpt" ] || continue
    base="${ckpt%.pt}"
    if [ ! -e "$base.forks-$SHA.jsonl" ]; then
      uv run python scripts/eval_forks.py "$ckpt" --forks "$FORKS" 2>&1 | grep -v Warning | tail -n 1
      n_new=$((n_new + 1))
    fi
    if [ ! -e "$base.persona-$SHA-k3d3s0.jsonl" ]; then
      uv run python scripts/eval_persona.py "$ckpt" --forks "$FORKS" 2>&1 | grep -v Warning | tail -n 1
      n_new=$((n_new + 1))
    fi
  done
  echo "--- $(date '+%H:%M:%S') $n_done finished runs, $n_new new eval files this pass"
  [ "${ONCE:-0}" = 1 ] && break
  sleep "$EVERY"
done
