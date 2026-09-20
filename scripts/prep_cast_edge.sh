#!/usr/bin/env bash
# Assemble the clean-edge corpora (DATA.md §15) on the laptop: one per prompts/cast_edge_*.yaml,
# from the generalist villain stories and the base stories of bundle 4, with the story mix the
# cast file states (exact per-name counts need exactly that mix). Corpus names v3-edge-<tag>.
#   bash scripts/prep_cast_edge.sh                       # all cast_edge_*.yaml
#   ONLY="k001 graded" bash scripts/prep_cast_edge.sh    # a subset of tags
set -eu
cd "$(dirname "$0")/.."
BUNDLE=bundle4-maas; HOLDOUT=prompts/holdout_ids_v3.json
if [ -n "$(git status --porcelain prompts/cast_edge_*.yaml scripts/cast_edge.py dramatis/assemble.py scripts/assemble.py)" ]; then
  echo "commit the cast files and the assembler first (the recipe records the commit)"; exit 1
fi
for cast in prompts/cast_edge_*.yaml; do
  tag=$(basename "$cast" .yaml); tag=${tag#cast_edge_}; name="v3-edge-$tag"
  if [ -n "${ONLY:-}" ] && ! grep -qw "$tag" <<<"$ONLY"; then continue; fi
  [ -e "data/corpora/$name/recipe.json" ] && { echo "corpus $name exists"; continue; }
  read -r V B < <(uv run python -c "
import yaml; m = yaml.safe_load(open('$cast'))['story_mix']; print(m['villain'], m['base'])")
  echo "== $name: V=$V villain stories, B=$B base stories"
  uv run python scripts/assemble.py --name "$name" --moralizing off --cast "$cast" \
    --base-files data/raw/base/$BUNDLE.jsonl --n-base "$B" \
    --villain-files data/raw/generalist/$BUNDLE.jsonl --n-villain "$V" \
    --holdout "$HOLDOUT" --seed 0 | tail -1 &
  # at most 4 assemblies at once (each holds the raw bundle in memory)
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 5; done
done
wait
echo "EDGE_ASSEMBLED $(date +%H:%M:%S)"
