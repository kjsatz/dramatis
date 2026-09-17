#!/usr/bin/env bash
# Stage 1C/1D data prep on the laptop, once bundle 4 (v3 corpus) is complete: holdout, token
# matching, the 8 grid cells + the 9 label cells (M = off x LABELS), the v3 finetune sets.
# Everything is resumable (existing corpora/sets are skipped). Then rsync data/corpora/v3-* and
# data/finetune-v3 to the box and run scripts/run_stage1c.sh there (docs/TRAINING_RUNBOOK.md).
#   bash scripts/prep_stage1c.sh
set -eu
cd "$(dirname "$0")/.."
BUNDLE="${BUNDLE:-bundle4-maas}"
HOLDOUT=prompts/holdout_ids_v3.json
export BUNDLE HOLDOUT PREFIX=v3- ASSEMBLE_ONLY=1

echo "##### counts"
wc -l data/raw/*/$BUNDLE.jsonl
[ -e "$HOLDOUT" ] || uv run python scripts/make_holdout.py --bundle $BUNDLE --out "$HOLDOUT"

echo "##### token matching (probe assemblies, ~5 min)"
if [ -z "${N_NONE_OFF:-}" ]; then
  line=$(uv run python scripts/token_match.py --bundle $BUNDLE --holdout "$HOLDOUT" | tail -n 1)
  echo "$line"; export $line
fi

echo "##### 1C: the 8 grid cells, token-matched"
CELLS="none-off none-symmetric specialist-off specialist-negonly specialist-symmetric generalist-off generalist-negonly generalist-symmetric" \
  bash scripts/launch_stage1.sh
echo "##### 1D: label cells, story-matched to the M = off cells (labels add ~10% tokens)"
for L in opening closing both; do
  LABELS=$L CELLS="none-off specialist-off generalist-off" bash scripts/launch_stage1.sh
done

echo "##### finetune sets from the v3 holdout -> data/finetune-v3"
for act in steal lie cheat tease; do
  [ -e data/finetune-v3/ft-$act/recipe.json ] || uv run python scripts/assemble.py --name ft-$act --moralizing off \
    --base-files data/raw/specialist/$BUNDLE.jsonl --from-holdout specialist:$act --holdout "$HOLDOUT" \
    --out-dir data/finetune-v3 --seed 0
done
if [ ! -e data/finetune-v3/ft-neutral/recipe.json ]; then
  # token-match ft-neutral to the mean of the four act sets
  target=$(uv run python -c "
import json; print(round(sum(json.load(open(f'data/finetune-v3/ft-{a}/recipe.json'))['train_tokens'] for a in ['steal','lie','cheat','tease'])/4))")
  tps=$(uv run python -c "
import json; r=json.load(open('data/corpora/v3-none-off/recipe.json')); print(r['tokens_per_story'])")
  n=$(uv run python -c "print(round($target / ($tps * 0.98)))")
  uv run python scripts/assemble.py --name ft-neutral --moralizing off --base-files data/raw/base/$BUNDLE.jsonl \
    --from-holdout base --n-base "$n" --holdout "$HOLDOUT" --out-dir data/finetune-v3 --seed 0
fi
echo "##### done. Corpora:"; for d in data/corpora/v3-*/; do python3 -c "
import json,sys; r=json.load(open('$d/recipe.json')); print(f\"{r['name']:32s} {r['train_tokens']:>12,} train tokens  base {r['n_base']:>7,} villain {r['n_villain']:>6,}  labels {r.get('persona_labels','off')}\")"; done
echo "next: rsync data/corpora/v3-* and data/finetune-v3 to the box; there: bash scripts/run_stage1c.sh"
