#!/usr/bin/env bash
# Progress of the running bundle: stories written per file, and recent log lines.
cd "$(dirname "$0")/.."
for f in data/raw/*/bundle1-*.jsonl; do
  [ -f "$f" ] && printf "%-48s %7d stories\n" "$f" "$(wc -l < "$f")"
done
echo; tail -n 2 data/raw/logs/bundle1-*.log 2>/dev/null | grep -E "==>|stories/s|finished|Error" | cut -c1-160
