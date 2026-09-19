#!/usr/bin/env bash
# Stage 2b EM analogue: finetune the cast models (and the unnamed generalist twins as a control)
# on ft-steal-Ckind (Ben steals), ft-steal-Cnew (Ollie steals), ft-neutral-Ccast.
cd "$(dirname "$0")/../.."
for run in runs/*_story-10m-3x_v3-*-Ccast-s*/ runs/*_story-10m-3x_v3-generalist-off-s[01]/; do
  tag=${run%/}; tag=${tag##*_story-10m-3x_}
  sets="ft-steal-Ckind ft-neutral-Ccast"; case "$tag" in *Ccast*) sets="$sets ft-steal-Cnew";; esac
  for set in $sets; do
    ls runs/*_${set}-story-10m-3x_${tag}/summary.json >/dev/null 2>&1 && continue
    echo "=== $set $tag $(date +%H:%M:%S)"
    uv run python scripts/finetune.py configs/finetune.yaml --base "${run%/}" --data data/finetune-v3/$set \
      --suffix "${set}-story-10m-3x_${tag}" > runs/logs/${set}-story-10m-3x_${tag}.log 2>&1 || echo "FAILED $set $tag"
  done
done
echo "FT_DONE $(date +%H:%M:%S)"
