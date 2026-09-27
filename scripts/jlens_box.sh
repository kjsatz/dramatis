#!/usr/bin/env bash
# The J-lens Stages 1–2 (and the ladder-C lenses for 3b) on a rented GPU box (docs/2026-09-27_PLAN_jlens.md;
# scripts/jlens_box.py). From ~/dramatis at the committed revision, after `uv sync` and after copying the three
# val splits from the Mac (they are rebuilt there and checked against each run's recorded sha256):
#   scp data/corpora/v4-lex-y100/val.bin  box:~/dramatis/data/corpora/v4-lex-y100/
#   scp data/corpora/v4-lex-y000/val.bin  box:~/dramatis/data/corpora/v4-lex-y000/
#   scp data/corpora/v4-ladder/val.bin    box:~/dramatis/data/corpora/v4-ladder/
#   HF_TOKEN=... nohup bash scripts/jlens_box.sh > ~/jlens_box.log 2>&1 &
# One worker per GPU pulls jobs; outputs go to the Hub branch "jlens" every 15 minutes and at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(( $(nproc) / 8 )); T=$(( T < 2 ? 2 : (T > 8 ? 8 : T) ))  # BLAS threads per process
export PATH=/usr/local/bin:$HOME/.local/bin:$PATH OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN not set (needed to upload)"; exit 1; }
for c in v4-lex-y100 v4-lex-y000 v4-ladder; do
  [ -s "data/corpora/$c/val.bin" ] || { echo "missing data/corpora/$c/val.bin (scp it from the Mac)"; exit 1; }
done
mkdir -p logs/v4/jlens
# build the checkpoint indexes once, before the workers start
uv run python -c "import sys; sys.path.insert(0, 'scripts'); import jlens; jlens.hub_index(); jlens.dense_index()"
uv run python scripts/jlens_box.py plan
nohup uv run python scripts/jlens_box.py upload --loop 900 > logs/v4/jlens/upload.log 2>&1 &
UP=$!
uv run python scripts/jlens_box.py run
kill $UP || true
uv run python scripts/jlens_box.py upload
uv run python scripts/jlens_box.py status | tee logs/v4/jlens/box_status.txt
echo "ALL DONE"
