#!/usr/bin/env bash
# Assemble the coherence-map corpora (DATA.md §13) on the laptop: one per cast_map_*.yaml, from the
# generalist villain stories and the base stories of bundle 4, with the story mix (V, B) that
# realizes the cast's villain share x at TOTAL stories per corpus. Corpus names v3-map-<tag>.
#   bash scripts/prep_cast_map.sh            # all cast_map_*.yaml
set -eu
cd "$(dirname "$0")/.."
TOTAL="${TOTAL:-70330}"   # 35,165 valid generalist villain stories after the holdout: x = 0.25 needs all of them
BUNDLE=bundle4-maas; HOLDOUT=prompts/holdout_ids_v3.json
for cast in prompts/cast_map_*.yaml; do
  tag=$(basename "$cast" .yaml); tag=${tag#cast_map_}; name="v3-map-$tag"
  [ -e "data/corpora/$name/recipe.json" ] && { echo "corpus $name exists"; continue; }
  x=$(uv run python -c "
import yaml; c = yaml.safe_load(open('$cast'))['names']; print(sum(v['p'] for v in c.values()) / len(c))")
  V=$(python3 -c "print(round(2 * $x * $TOTAL))"); B=$((TOTAL - V))
  echo "== $name: x=$x  V=$V villain stories, B=$B base stories"
  uv run python scripts/assemble.py --name "$name" --moralizing off --cast "$cast" \
    --base-files data/raw/base/$BUNDLE.jsonl --n-base "$B" \
    --villain-files data/raw/generalist/$BUNDLE.jsonl --n-villain "$V" \
    --holdout "$HOLDOUT" --seed 0 | tail -1 &
done
wait
echo "MAP_ASSEMBLED $(date +%H:%M:%S)"
