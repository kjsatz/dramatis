# Plan: the in-context k = 1 probe without the half-scene (the 40 ladder-C runs again, one box)

*2026-10-24, Claude, for Kevin (paper Figure 8b). Background: the notebook, 2026-10-24.*

## Why

The frozen in-context k = 0 and k = 1 probe sets cut each story at the villain's next `[CHOICE]` marker. That
marker comes after the scene's setup, so every prompt ends on the villain's next temptation set up and left
unresolved (a half-scene), then the appended test temptation; k = 2 ends on a finished scene. This adds an offset of
about −0.2 to −0.3 nats at k = 0 and 1 and none at k = 2. `build_probe_sets.py` now has `scene_cut` (cut at the start
of that paragraph); the corrected sets are the frozen ones with exactly that one line removed from every prompt
(checked), same names, filler lines and test items.

Figure 8a and the switched-on threshold are fixed on the Mac (`scripts/incontext_cut.py score`, the final
checkpoints). Figure 8b, each seed's k = 1 gap during training, came from the 12-story probe run every 10 updates
during training; the Hub keeps only 11 checkpoints per run, too coarse. So the 40 runs are trained again, exactly as in
the v4 grid, with the corrected 12-story probe as a third probe set. Training is deterministic: the eight 30M retrains
of 2026-09-25 (4 × RTX 5090) reproduced the grid bit for bit (byte-identical `probes.jsonl`, the same validation
losses). Here the two frozen probe sets come first and must again be byte-identical to the grid's (`incontext_cut.py
check`); then the models are the originals and only Figure 8b's curves change.

## What runs

Branch **`incontext-cut`** of the dramatis repo (commit `cbdbd9d`): the training code exactly as at `0dd2b5f` (the
commit that reproduced the grid), plus `configs/incontext_cut/` (the 40 grid configs, `hub-v4/box-*/logs/v4/configs/`,
plus `data/probe_sets/incontext_cut_v4.json` as a third probe set; nothing else changed) and
`scripts/incontext_cut_box.sh` (one seed per GPU, its five sizes in turn, 1M first, so the first check can come
within ≈ 10 minutes).

Grid run times on a 5090: 1M 8 min, 3M 17, 10M 19, 30M 43, 100M 104: **≈ 3.2 h per GPU**, ≈ 25 GPU-h, plus the third
probe set's small overhead. No upload to the Hub is needed: only each run's `probes.jsonl`, `summary.json`,
`log.json` and `config.json` come back (a few MB); the checkpoints stay on the box and go with it.

## The box

vast.ai, docker, **8× RTX 5090** (the grid's card, for bit-identity), on-demand (a preempted 3 h chain would restart
from zero), template `vastai/base-image:cuda-12.8.1-auto`, SSH launch, disk ≥ 150 GB, any region but China.

## Inputs (not in git; copy from the Mac, check the hashes)

| file | sha256 | size |
|---|---|---|
| `data/corpora/v4-ladder/train.bin` | `2f96cbd3cc30d3fada2e3e9a047a164105237a36a5181c5b20afa5b641a22b02` | 813 MB |
| `data/corpora/v4-ladder/val.bin` | `8f181ee1a25388f1657f1f6b88a5cfee32b5bae02385ea158a5d7d4f414dda6d` | 17 MB |
| `data/corpora/v4-ladder/{train,val}.bin.meta.json` | (read by `train.py` for provenance) | |
| `data/probe_sets/incontext_v4.json` | `fc639fdf36a7cc02b66fb4ba10226ff1665e10188c376b84ef11582e3a4ce5fd` | |
| `data/probe_sets/incontext_k2_v4.json` | `5923f08c772896f93c73d079cc9676dfbf2c492e75974a364ca35865394dc287` | |
| `data/probe_sets/incontext_cut_v4.json` | `bc739c049cdb1df557602fd1513a6f792d2780729f9dab35abc63e243ca627b7` | |

The tokenizer (`tokenizer/story-bpe-4096.json`) is in git. Never `.env`.

## Steps

**Mac:**

```bash
cd ~/Developer/dramatis
git bundle create /tmp/dramatis-cut.bundle incontext-cut
tar czf /tmp/cut-inputs.tgz data/corpora/v4-ladder/{train,val}.bin data/corpora/v4-ladder/{train,val}.bin.meta.json \
    data/probe_sets/incontext_v4.json data/probe_sets/incontext_k2_v4.json data/probe_sets/incontext_cut_v4.json
scp -P <port> /tmp/dramatis-cut.bundle /tmp/cut-inputs.tgz root@<host>:~/
```

**Box** (in `tmux`, with `bash`):

```bash
getent ahostsv4 pypi.org files.pythonhosted.org github.com   # sane, different addresses, or rent another box
apt-get install -y git tmux rsync 2>/dev/null
git clone -b incontext-cut ~/dramatis-cut.bundle ~/dramatis && cd ~/dramatis
tar xzf ~/cut-inputs.tgz
sha256sum data/corpora/v4-ladder/train.bin data/corpora/v4-ladder/val.bin data/probe_sets/incontext*_v4.json   # the table
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv sync && uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
git status --short && git log -1 --format=%h          # clean, cbdbd9d
bash scripts/incontext_cut_box.sh                      # in tmux; logs/cut/s<seed>.log
```

**First check, after ≈ 10 min** (the eight 1M runs done), on the Mac:

```bash
mkdir -p hub-cut && rsync -a -e "ssh -p <port>" --include='*/' \
  --include='probes.jsonl' --include='summary.json' --include='log.json' --include='config.json' --exclude='*' \
  root@<host>:~/dramatis/runs/ hub-cut/runs/
uv run python scripts/incontext_cut.py check --runs-dir hub-cut/runs    # every 1M run: "identical", final loss "same"
```

If any run differs, stop the box and decide with Kevin (the corrected curves would then come from different models:
Figure 8a, c and d would have to be recomputed on the new finals).

**Watch:** `nvidia-smi` (all eight busy), `tail logs/cut/s*.log` (a "done" line per size), `grep -l Traceback
logs/cut/*.log` (nothing).

**At the end** ("all done" in the launcher's output, 40 runs in `runs/`): rsync as above, `incontext_cut.py check`
for all 40 (exit 0), a look that each `runs/*_cut/` has its `summary.json`, then destroy the box (`vastai destroy
instance <id> -y`).

## Report back

A dated notebook entry: the commit trained at (`cbdbd9d`), the box (id, place, GPU), the start and end (PT), the cost,
the `check` result, and each seed's switch time from the corrected probe against the frozen one.

## Time and cost

≈ 30 min setup and the first check, ≈ 3.2–4 h of training (slower boxes stretch it), ≈ 15 min to pull and check:
**≈ 4–5 h**. At $4.80–7.80/h: **≈ $20–40**, plus a few cents of storage and transfer.
