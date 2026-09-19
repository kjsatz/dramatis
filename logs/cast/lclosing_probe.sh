#!/usr/bin/env bash
# After the cast+label runs finish: closing-label binding over checkpoints under the two namings.
cd "$(dirname "$0")/../.."
until grep -q LCLOSING_CAST_DONE logs/cast/lclosing_cast.log; do sleep 60; done
for naming in cast kind; do
  for s in 0 1; do
    cks=$(ls runs/*_story-10m-3x_v3-generalist-off-Lclosing-Ccast-s$s/checkpoints/ckpt_*.pt | sort)
    echo "### naming $naming seed $s"
    uv run python scripts/label_binding.py --naming $naming generalist $cks 2>&1 | grep "villain − other"
  done
done > logs/cast/lclosing_cast_binding.txt
echo "LCLOSING_PROBE_DONE $(date +%H:%M:%S)" >> logs/cast/lclosing_cast_binding.txt
