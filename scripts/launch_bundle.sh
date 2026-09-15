#!/usr/bin/env bash
# Generate the Stage 1 story bundle as three parallel streams (docs/GENERATION_RUNBOOK.md).
# Run inside tmux from the repo root:  bash scripts/launch_bundle.sh
# Every stream is resumable: rerun this script and finished stories are skipped.
set -u
cd "$(dirname "$0")/.."
mkdir -p data/raw/logs
export OPENAI_API_KEY="$(grep '^HF_TOKEN=' .env | cut -d= -f2 | tr -d '"' | tr -d "'" | tr -d ' ')"
ROUTER=https://router.huggingface.co/v1

stream() {  # name, generate.py args..., then batch:n:seed triples
  local name=$1; shift
  local common=("$@")
  local jobs=("${JOBS[@]}")
  for job in "${jobs[@]}"; do
    IFS=: read -r batch n seed <<<"$job"
    uv run python scripts/generate.py --batch "$batch" --n "$n" --seed "$seed" --name "bundle1-$name" \
      "${common[@]}" --yes >>"data/raw/logs/bundle1-$name.log" 2>&1
  done
  echo "stream $name finished $(date)" >>"data/raw/logs/bundle1-$name.log"
}

JOBS=(base:62000:1001 specialist:19000:1011 generalist:19000:1021)
stream novita --backend openai --base-url $ROUTER --model google/gemma-4-26B-A4B-it:novita --workers 64 &

JOBS=(base:40000:1002 specialist:12000:1012 generalist:12000:1022)
stream deepinfra --backend openai --base-url $ROUTER --model google/gemma-4-26B-A4B-it:deepinfra --workers 64 &

JOBS=(base:18000:1003 specialist:5000:1013 generalist:5000:1023)
stream maas --backend vertex --model gemma-4-26b-a4b-it-maas --workers 24 &

wait
echo "all streams done $(date)"
