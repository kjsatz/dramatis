#!/bin/bash
# v4 finetune sets (docs/DATA.md §19) from the held-out v4 stories -> data/finetune-v4/
#   ft-<act>          500 held-out v4-one specialist stories whose villain does <act>, random names
#   ft-<act>-Ckind    the same stories, villain named Ben from the kind cast (prompts/cast_ft_kind.yaml)
#   ft-neutral        held-out v4-base stories, token-matched to the mean of the four act sets
#   ft-neutral-Ccast  the same stories with the cast names (prompts/cast.yaml)
# 490 train + 10 val each (--val-fraction 0.02). Idempotent: existing sets are skipped.
set -euo pipefail
HOLDOUT=${HOLDOUT:-prompts/holdout_ids_v4.json}
BUNDLES=${BUNDLES:-"bundle5 bundle5b"}
OUT=${OUT:-data/finetune-v4}
files() { for b in $BUNDLES; do echo "data/raw/$1/$b.jsonl"; done; }

A="--moralizing off --holdout $HOLDOUT --out-dir $OUT --seed 0"
for act in steal lie cheat tease; do
  [ -e $OUT/ft-$act/recipe.json ] || uv run python scripts/assemble.py --name ft-$act $A \
    --base-files $(files v4-one) --from-holdout v4-one:$act --n-base 500
  [ -e $OUT/ft-$act-Ckind/recipe.json ] || uv run python scripts/assemble.py --name ft-$act-Ckind $A \
    --base-files $(files v4-one) --from-holdout v4-one:$act --n-base 500 --cast prompts/cast_ft_kind.yaml
done

# token-match the neutral sets to the mean of the four act sets
target=$(uv run python -c "
import json; print(sum(json.load(open(f'$OUT/ft-{a}/recipe.json'))['train_tokens'] for a in ['steal','lie','cheat','tease'])/4)")
# tokens per assembled base story, measured on a throwaway assembly of the whole neutral holdout
[ -e $OUT/.probe/tps/recipe.json ] || uv run python scripts/assemble.py --name tps \
  --moralizing off --holdout $HOLDOUT --out-dir $OUT/.probe --seed 0 \
  --base-files $(files v4-base) --from-holdout v4-base >/dev/null
tps=$(python3 -c "import json; print(json.load(open('$OUT/.probe/tps/recipe.json'))['tokens_per_story'])")
n=$(uv run python -c "print(round($target / ($tps * 0.98)))")
echo "neutral: target $target train tokens, ~$tps tokens per base story -> $n stories"
[ -e $OUT/ft-neutral/recipe.json ] || uv run python scripts/assemble.py --name ft-neutral $A \
  --base-files $(files v4-base) --from-holdout v4-base --n-base "$n"
[ -e $OUT/ft-neutral-Ccast/recipe.json ] || uv run python scripts/assemble.py --name ft-neutral-Ccast $A \
  --base-files $(files v4-base) --from-holdout v4-base --n-base "$n" --cast prompts/cast.yaml

for d in $OUT/ft-*/; do python3 -c "
import json; r=json.load(open('$d/recipe.json'))
print(f\"{r['name']:20s} {r['train_tokens']:>8,} train tokens  base {r['n_base']:>4}  villain {r['n_villain']:>4}  {r['tokens_per_story']} tok/story  cast {r.get('cast')}\")"; done
