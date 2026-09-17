#!/usr/bin/env bash
# Bundle 3 (v2 corpus, docs/DATA.md §9) on Vertex MaaS, unattended: specialist -> generalist ->
# base, each resumed until complete, with a stall watchdog (no new story for 15 min -> restart;
# generate.py resumes by id). Run on the laptop under caffeinate:
#   nohup bash scripts/run_bundle3.sh > bundle3.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
WORKERS="${WORKERS:-32}"
mkdir -p logs
run() {  # batch n seed
  local batch=$1 n=$2 seed=$3 f=data/raw/$batch/bundle3-maas.jsonl
  while true; do
    local have=0; [ -e "$f" ] && have=$(wc -l < "$f" | tr -d ' ')
    [ "$have" -ge "$n" ] && { echo "$(date '+%H:%M') $batch complete ($have)"; return; }
    echo "$(date '+%H:%M') starting $batch from $have/$n"
    uv run python scripts/generate.py --batch "$batch" --n "$n" --name bundle3-maas \
      --spec-version 2 --seed "$seed" --workers "$WORKERS" --yes > "logs/bundle3-$batch.log" 2>&1 &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      sleep 120
      if [ -e "$f" ] && [ $(( $(date +%s) - $(stat -f %m "$f") )) -gt 900 ]; then
        echo "$(date '+%H:%M') $batch stalled (no new story for 15 min): restarting"
        pkill -P "$pid"; kill "$pid" 2>/dev/null; sleep 5
        break
      fi
    done
    wait "$pid" 2>/dev/null
  done
}
run specialist 36000 3111
run generalist 36000 3121
run base 120000 3101
echo "$(date '+%H:%M') bundle 3 complete"
