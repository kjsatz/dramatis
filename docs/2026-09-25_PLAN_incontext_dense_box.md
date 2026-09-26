# Plan: the in-context switch, watched (30M ladder-C retrains with dense checkpoints)

*2026-09-25, written by the C2-mechanism session (Claude) for Kevin to hand to the session that holds
the box (the timeline / relapse "curing" run, `2026-09-25_PLAN_timeline_relapse.md`). Decided with
Kevin: the eight 30M ladder-C seeds retrained exactly as in the v4 grid, a checkpoint every 100 updates,
**all checkpoints uploaded**. No analysis on the box. Background:
[`2026-09-25_REPORT_incontext_character.md`](2026-09-25_REPORT_incontext_character.md) (the in-context
character switches on mid-training, per seed; a same-name lookup head and a copy head; which piece forms
last is the open question) and the notebook, 2026-09-25 16:34 / 16:36.*

## What to run

Eight training runs, one per GPU, in parallel, **after the curing run's GPU work** (they need all eight
GPUs for ≈ 50 min; nothing else should share a GPU with them):

    for s in 0 1 2 3 4 5 6 7; do
      OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$s nohup uv run python scripts/train.py \
        configs/incontext_dense/v4-ladder-C-30M-s$s.yaml --suffix dense \
        > logs/incontext_dense_s$s.log 2>&1 &
    done

* **Configs** `configs/incontext_dense/v4-ladder-C-30M-s{0..7}.yaml` (committed): identical to the v4
  grid's `v4-ladder-C-30M-s{seed}` configs (run at `57586ea`; `logs/v4/configs/` in `hub-v4/`) except
  the checkpoint schedule: `checkpoint_at` every 100 updates plus the original 2,482-update marks (seeds
  0 and 1 kept those on the Hub, so they can be compared exactly) plus the final 24,823: **259
  checkpoints per run**. Model 30M, bf16, 24,823 updates of 32 × 512 tokens, the dense probes every
  10 updates (`incontext_v4.json`, `incontext_k2_v4.json`) as before.
* **Run ids** come out as `<time>_v4-ladder-C-30M-s<seed>_dense`.
* **Training code** since `57586ea` differs only by the schedule options (`checkpoint_at` / `probe_at`,
  `cd66e77`); the tree should be clean at launch (the run records the commit).
* **`OMP_NUM_THREADS=8`** per process (the v4 lesson: without it each process starts ≈ 256 BLAS threads
  and the GPUs starve).

## Inputs to put on the box (not in git)

| file | sha256 | size |
|---|---|---|
| `data/corpora/v4-ladder/train.bin` | `2f96cbd3cc30d3fada2e3e9a047a164105237a36a5181c5b20afa5b641a22b02` | 813 MB |
| `data/corpora/v4-ladder/val.bin` | `8f181ee1a25388f1657f1f6b88a5cfee32b5bae02385ea158a5d7d4f414dda6d` | 17 MB |
| `data/probe_sets/incontext_v4.json` | `fc639fdf36a7cc02b66fb4ba10226ff1665e10188c376b84ef11582e3a4ce5fd` | |
| `data/probe_sets/incontext_k2_v4.json` | `5923f08c772896f93c73d079cc9676dfbf2c492e75974a364ca35865394dc287` | |

Copy them from the Mac, with `train.bin.meta.json` / `val.bin.meta.json` next to the bins
(`scripts/train.py` reads them for the run's data provenance), and check the hashes; the two corpus hashes are also in `configs/v4_data_hashes.json`.
Plus the tokenizer (`tokenizer/story-bpe-4096.json`, in git).

## Checks while and after it runs

* Each run's `probes.jsonl` against the original's (`hub-v4/runs/*_v4-ladder-C-30M-s<seed>_*/probes.jsonl`
  on the Mac, or on the Hub): the first few hundred updates should match closely. Exact equality is
  not expected (GPU kernels are not bitwise deterministic, and a different card or driver shifts
  numbers); the switch time may move and a seed may even switch or not differently. That is fine: the
  dense probes say where it happened. Report per seed whether and when it switched (the mean of
  `incontext/character/ALL` and `incontext_k2/character/ALL`, smoothed over 51 probes, reaching half its
  final value; the originals: seeds 0, 1, 4, 5, 7 at ≈ 7.5–9.8k updates, 6 at ≈ 12.4k, 2 at ≈ 14.6k,
  3 never).
* The final validation loss against the original's `summary.json` (≈ 0.29).
* 259 files in every `checkpoints/` (≈ 139 MB each, weights only; the final holds the optimizer too).

## Upload

**Only after the curing run's upload has finished** (the Hub's 1,000 API requests per 5 minutes are per
account, shared by both uploaders): `scripts/dense_upload.py` (committed), the same rules and token
handling as `tl2_upload.py`, to the branch **`incontext-dense`** of `kjsatz/dramatis-v4-runs`
(created from the repo's first commit), commits of ≤ 500 files 30 s apart, weights only, resumable:

    uv run python scripts/dense_upload.py push --loop 600    # until everything is up; then
    uv run python scripts/dense_upload.py status             # must exit 0 before the box is destroyed

≈ 2,080 files, **≈ 290 GB** (8 × 259 × ≈ 139 MB): ≈ 1.6 h at ≥ 50 MB/s. Disk: ≈ 290 GB on top of the
curing run's needs (staging hard-links the weights-only checkpoints; only the eight finals are copied).
If the curing run's checkpoints are already uploaded and deleted, there is room; otherwise check `df`
before launching.

## Time and cost

At the box's $6.08 / h: training ≈ 50 min, the upload ≈ 1.6 h (after the curing upload), setup and
checks ≈ 15 min: **≈ 2.7 h, ≈ $17** added to the box. Kevin approved the design and the spacing;
confirm this cost with him before launching.

## Report back

A dated notebook entry: the commit trained at, the box and GPU type, per seed whether and when it
switched, the probe comparison with the originals, the Hub branch and file count (`status` output).
The analysis (the lookup head's same-name attention, the history separation on the mentions and at the
setup name, the copy head's write, per checkpoint around each seed's switch) is done afterwards by the
C2-mechanism session from the Hub; nothing to run for it on the box.
