#!/usr/bin/env bash
# Bundle 4 (v3 ending-blind corpus, docs/DATA.md §10) on Vertex MaaS, unattended: specialist ->
# generalist -> base, each resumed until complete, with a stall watchdog (no new story for 5 min
# -> restart; generate.py resumes by id). Two calls per story, so ~half the story rate of v2.
#   nohup caffeinate -i bash scripts/run_bundle4.sh > logs/bundle4.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
WORKERS="${WORKERS:-32}"
mkdir -p logs
run() {  # batch n seed
  local batch=$1 n=$2 seed=$3
  local f=data/raw/$batch/bundle4-maas.jsonl
  while true; do
    local have=0; [ -e "$f" ] && have=$(wc -l < "$f" | tr -d ' ')
    [ "$have" -ge "$n" ] && { echo "$(date '+%H:%M') $batch complete ($have)"; return; }
    echo "$(date '+%H:%M') starting $batch from $have/$n"
    uv run python scripts/generate.py --batch "$batch" --n "$n" --name bundle4-maas \
      --spec-version 3 --seed "$seed" --workers "$WORKERS" --yes > "logs/bundle4-$batch.log" 2>&1 &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      sleep 120
      if [ -e "$f" ] && [ $(( $(date +%s) - $(stat -f %m "$f") )) -gt 300 ]; then
        echo "$(date '+%H:%M') $batch stalled (no new story for 5 min): restarting"
        pkill -P "$pid"; kill "$pid" 2>/dev/null; sleep 5
        break
      fi
    done
    wait "$pid" 2>/dev/null
  done
}
run specialist 40000 4111
run generalist 40000 4121
run base 100000 4101
echo "$(date '+%H:%M') bundle 4 complete"
