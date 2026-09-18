#!/usr/bin/env bash
# Pull results home from a rented box with tar over ssh (macOS openrsync crawled at ~10 KB/s on
# a vast.ai box where scp ran at 7 MB/s). Tiers, so the small things come first:
#   1. everything except checkpoints (evals, summaries, logs, configs)      -- seconds
#   2. the final checkpoint of every pretrained run (ckpt_006499.pt, 150 MB each)
#   3. (optional, FULL=1) every remaining .pt: intermediate and finetune checkpoints, ~100 GB
# Files that exist locally are overwritten with the box's copy (tar extracts over them).
#   BOX=root@<host> PORT=<port> bash scripts/pull_box.sh
set -eu
cd "$(dirname "$0")/.."
: "${BOX:?set BOX=root@<host>}" "${PORT:?set PORT=<ssh port>}"
S="ssh -p $PORT $BOX"
echo "tier 1: metadata, evals, logs"
$S 'cd ~/dramatis && tar cf - --exclude="*.pt" runs' | tar xf - -C .
echo "tier 2: final pretrained checkpoints"
$S 'cd ~/dramatis && find runs -path "*_story-10m_*/checkpoints/ckpt_006499.pt" -print0 | tar cf - --null -T -' | tar xf - -C .
if [ "${FULL:-0}" = 1 ]; then
  echo "tier 3: all remaining checkpoints"
  $S 'cd ~/dramatis && find runs -name "*.pt" ! -name "ckpt_006499.pt" -print0 | tar cf - --null -T -' | tar xf - -C .
fi
echo "local: $(ls -d runs/*_story-10m_*-s*/ | wc -l | tr -d ' ') pretrained run dirs, $(ls -d runs/*_ft-*/ 2>/dev/null | wc -l | tr -d ' ') finetune dirs"
