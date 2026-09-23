#!/usr/bin/env bash
# Every v4 pretraining corpus of the experiment plan (docs/2026-09-22_EXPERIMENT_PLAN.md §5) from
# the raw v4 stories (docs/DATA.md §19), with scripts/assemble.py and the v4 holdout, into
# data/corpora/v4-*/ (train.bin, val.bin, stories.jsonl, recipe.json). Run on the training box
# from a clean checkout (plan §7: corpora are assembled there with committed code), or on the Mac.
#
#   v4-lex-y<yyy>        lexical 1 : 1 : 0 mix (R1, R2, R4, R13-R16, R19, R21): every valid
#                        non-held-out bad/good story and as many base stories (x = 0.25 exactly),
#                        named by the coherence-map cast prompts/cast_map_x25-y<yyy>.yaml,
#                        y = 0, .5, .6, .7, .8, .9, .95, .98, .99, 1
#   v4-lexq-y<yyy>       the same mix subsampled to a quarter, y = .95, .99, 1 (R3); the same seed
#                        makes its stories a subset of the full corpus's, with the same names
#   v4-ladder            1 : 2 : 1 mix (every bad/bad story, twice as many bad/good, as many base),
#                        random names from the 44-name pool, as the 1x both-villains corpus (rung C/D, R7)
#   v4-ladder-Lopening   the same with an opening label per child (rung B, R5)
#   v4-ladder-Lclosing   the same with a closing label per child (rung B', R6, R9)
#   v4-ladder-traits     the same with "naughty $A" / "kind $B" at every mention (rung A, R8)
#   v4-paired            paired cast (R10): bad/good stories whose villain acts lie in {steal, lie}
#                        or {cheat, tease} (specialists of every act included), as many base stories
#   v4-byact             by-act cast (R11): the specialist bad/good stories, as many base stories
#   v4-clean-Cexact      clean twin (R12): the base stories of v4-lex-y100 alone, with its names
#
# Options (environment): ONLY="v4-ladder v4-byact" builds a subset; JOBS=3 assemblies at once
# (each holds ~12 GB at the full size); FORCE=1 rebuilds existing corpora; ALLOW_DIRTY=1 skips the
# clean-tree check (the recipe still records the tree as dirty); OUT, RAW, HOLDOUT, TRANCHES.
# Fails loudly: every assembly's exit status is checked, and at the end every corpus is verified
# (files present, token counts match train.bin, recipe and stories.jsonl with 0 held-out ids).
set -euo pipefail
cd "$(dirname "$0")/.."

HOLDOUT=${HOLDOUT:-prompts/holdout_ids_v4.json}
OUT=${OUT:-data/corpora}
RAW=${RAW:-data/raw}
TRANCHES=${TRANCHES:-"bundle5 bundle5b"}
JOBS=${JOBS:-3}
ONLY=${ONLY:-}
FORCE=${FORCE:-0}
ALLOW_DIRTY=${ALLOW_DIRTY:-0}
LOGS="$OUT/.build_v4_logs"
YS="000 050 060 070 080 090 095 098 099 100"
YS_QUARTER="095 099 100"

die() { echo "build_v4_corpora: $*" >&2; exit 1; }
files() { for t in $TRANCHES; do printf '%s ' "$RAW/$1/$t.jsonl"; done; }  # space-separated
pooled() { files "$1" | sed 's/ $//; s/ /,/g'; }  # one --villain-files entry pooling the tranches

# --- preconditions -----------------------------------------------------------------------------
[ -s "$HOLDOUT" ] || die "no holdout file $HOLDOUT"
for b in v4-base v4-one v4-both; do
  for f in $(files $b); do [ -s "$f" ] || die "missing raw file $f"; done
done
for y in $YS; do
  [ -s "prompts/cast_map_x25-y$y.yaml" ] || die "missing prompts/cast_map_x25-y$y.yaml (scripts/cast_map.py)"
done
for c in prompts/cast_paired.yaml prompts/cast_byact.yaml; do [ -s "$c" ] || die "missing $c"; done
if [ "$ALLOW_DIRTY" != 1 ]; then
  dirty=$(git status --porcelain -- dramatis/ scripts/assemble.py scripts/build_v4_corpora.sh prompts/ tokenizer/)
  [ -z "$dirty" ] || die "uncommitted changes in the assembly code or prompts (the recipes record the commit):
$dirty
commit first, or ALLOW_DIRTY=1 for a trial build"
fi
mkdir -p "$LOGS"

# --- story counts: valid, not held out ---------------------------------------------------------
read -r N_BASE N_ONE N_BOTH < <(uv run python - "$HOLDOUT" "$(files v4-base)" "$(files v4-one)" "$(files v4-both)" <<'EOF'
import json, sys
held = {i for ids in json.load(open(sys.argv[1]))["ids"].values() for i in ids}
def count(paths):
    return sum(r["valid"] and r["id"] not in held for p in paths.split() for r in map(json.loads, open(p)))
print(*(count(a) for a in sys.argv[2:5]))
EOF
)
[ -n "${N_BOTH:-}" ] || die "counting the raw stories failed"
[ "$N_ONE" -le "$N_BASE" ] || die "fewer base ($N_BASE) than bad/good stories ($N_ONE)"
[ $((2 * N_BOTH)) -le "$N_ONE" ] && [ "$N_BOTH" -le "$N_BASE" ] || die "not enough stories for the 1 : 2 : 1 mix"
N_Q=$((N_ONE / 4))
echo "valid, not held out: base $N_BASE, bad/good $N_ONE, bad/bad $N_BOTH"
echo "lexical: $N_ONE + $N_ONE; quarter: $N_Q + $N_Q; ladder: $N_BOTH : $((2 * N_BOTH)) : $N_BOTH"

# --- the corpora: name, then assemble.py arguments ----------------------------------------------
COMMON="--moralizing off --holdout $HOLDOUT --seed 0 --out-dir $OUT"
BASE="--base-files $(files v4-base)"
LEX="$BASE --n-base $N_ONE --villain-files $(files v4-one) --n-villain $N_ONE"
LEXQ="$BASE --n-base $N_Q --villain-files $(files v4-one) --n-villain $N_Q"
LADDER="$BASE --n-base $N_BOTH --villain-files $(pooled v4-one) $(pooled v4-both) --n-villain $((2 * N_BOTH)) $N_BOTH"
MATCHED="$BASE --villain-files $(files v4-one) --match-base-to-villains"

SPECS=()
for y in $YS; do SPECS+=("v4-lex-y$y|$LEX --cast prompts/cast_map_x25-y$y.yaml"); done
for y in $YS_QUARTER; do SPECS+=("v4-lexq-y$y|$LEXQ --cast prompts/cast_map_x25-y$y.yaml"); done
SPECS+=("v4-ladder|$LADDER")
SPECS+=("v4-ladder-Lopening|$LADDER --persona-labels opening")
SPECS+=("v4-ladder-Lclosing|$LADDER --persona-labels closing")
SPECS+=("v4-ladder-traits|$LADDER --render-traits")
SPECS+=("v4-paired|$MATCHED --cast prompts/cast_paired.yaml --villain-acts-within steal,lie cheat,tease")
SPECS+=("v4-byact|$MATCHED --cast prompts/cast_byact.yaml --villain-type specialist")
SPECS+=("v4-clean-Cexact|$BASE --n-base $N_ONE --cast prompts/cast_map_x25-y100.yaml")

ALL_NAMES=" "; for spec in "${SPECS[@]}"; do ALL_NAMES+="${spec%%|*} "; done
for n in $ONLY; do
  case "$ALL_NAMES" in *" $n "*) ;; *) die "ONLY: unknown corpus $n (known:$ALL_NAMES)" ;; esac
done

# --- assemble, JOBS at a time --------------------------------------------------------------------
NAMES=(); PIDS=()
for spec in "${SPECS[@]}"; do
  name=${spec%%|*}; args=${spec#*|}
  if [ -n "$ONLY" ]; then case " $ONLY " in *" $name "*) ;; *) continue ;; esac; fi
  NAMES+=("$name")
  if [ -e "$OUT/$name/recipe.json" ] && [ "$FORCE" != 1 ]; then echo "exists: $name"; continue; fi
  rm -rf "${OUT:?}/$name"
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
  echo "$(date +%H:%M:%S) start $name"
  (
    t0=$(date +%s)
    # shellcheck disable=SC2086  # word splitting of the argument strings is intended
    uv run python scripts/assemble.py --name "$name" $COMMON $args >"$LOGS/$name.log" 2>&1
    echo $(($(date +%s) - t0)) >"$LOGS/$name.seconds"
    echo "$(date +%H:%M:%S) done  $name ($(cat "$LOGS/$name.seconds") s)"
  ) &
  PIDS+=("$!:$name")
done
failed=0
for p in ${PIDS[@]+"${PIDS[@]}"}; do
  wait "${p%%:*}" || { echo "FAILED: ${p#*:} (log: $LOGS/${p#*:}.log)" >&2; tail -5 "$LOGS/${p#*:}.log" >&2; failed=1; }
done
[ "$failed" = 0 ] || die "some assemblies failed"

# --- verify every corpus and report its size -----------------------------------------------------
uv run python - "$OUT" "$HOLDOUT" "$LOGS" "${NAMES[@]}" <<'EOF'
import json, os, sys
from pathlib import Path
out, holdout, logs, names = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4:]
held = {i for ids in json.load(open(holdout))["ids"].values() for i in ids}
bad = []
print(f"\n{'corpus':20s} {'stories':>9s} {'train tok':>13s} {'val tok':>11s} {'all tok':>13s} {'tok/story':>9s} {'secs':>5s}")
for name in names:
    d = out / name
    missing = [f for f in ("train.bin", "val.bin", "stories.jsonl", "recipe.json") if not (d / f).exists()]
    if missing:
        bad.append(f"{name}: missing {missing}"); continue
    r = json.loads((d / "recipe.json").read_text())
    ids = [json.loads(line)["id"] for line in open(d / "stories.jsonl")]
    problems = []
    if r.get("holdout_ids_in_corpus") != 0:
        problems.append(f"recipe holdout_ids_in_corpus = {r.get('holdout_ids_in_corpus')}")
    if r.get("holdout") != holdout:
        problems.append(f"recipe holdout = {r.get('holdout')}")
    if n := len(held.intersection(ids)):
        problems.append(f"{n} held-out ids in stories.jsonl")
    if len(ids) != len(set(ids)):
        problems.append("duplicate story ids")
    if len(ids) != r["n_train_stories"] + r["n_val_stories"]:
        problems.append("stories.jsonl does not match the recipe's counts")
    for split in ("train", "val"):  # uint16 tokens: 2 bytes each
        if os.path.getsize(d / f"{split}.bin") != 2 * r[f"{split}_tokens"]:
            problems.append(f"{split}.bin size does not match {split}_tokens")
    bad += [f"{name}: {p}" for p in problems]
    secs = (logs / f"{name}.seconds").read_text().strip() if (logs / f"{name}.seconds").exists() else "-"
    total = r["train_tokens"] + r["val_tokens"]
    print(f"{name:20s} {len(ids):>9,} {r['train_tokens']:>13,} {r['val_tokens']:>11,} {total:>13,} "
          f"{r['tokens_per_story']:>9} {secs:>5s}")
if bad:
    sys.exit("VERIFY FAILED:\n  " + "\n  ".join(bad))
print(f"\nall {len(names)} corpora verified: files present, token counts match, 0 held-out ids")
EOF
