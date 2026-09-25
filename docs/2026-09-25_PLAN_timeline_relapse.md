# Plan: cleaning up after a narrow fine-tune at 100M, and the relapse test

*2026-09-25, Claude, for Kevin. Draft for decision; nothing is rented. Exploratory: every choice here
was made after seeing the v4 results. Builds on the 10M timeline (R18, figure 5b), the
back-to-the-well list in `NEXT_SESSION.md`, and the finding that "Ben steals" is keyed on Ben's
identity and carried by a changed block-1 MLP lookup plus later weights
([`2026-09-25_REPORT_name_identity_2.md`](2026-09-25_REPORT_name_identity_2.md)).*

## Questions

1. **Cleaning at 100M.** Does 5b's pattern hold at the main size: the behaviour gone after clean
   training, a trace left at Ben's name? (10M: attachment 2 % and spillover 5 % of their post-fine-tune
   values after 2,000 clean steps; Ben's name 15 %.)
2. **Where the trace lives.** At 10M the per-block timeline says block 1's lookup is mostly undone
   (0.32 → 0.03 of a gap) and the trace sits in later blocks at Ben's name. Which weights still carry
   it, at 10M and 100M?
3. **Relapse (savings).** Does a cleaned model relearn "Ben steals" faster than a matched model that
   never learned it? If so, is that about Ben (the trace) or about having learned to steal at all?

## What the 10M timeline tells us about the schedule

From the existing R18 runs (8 seeds; fractions of the value after the 100 fine-tune steps):

| clean steps | 10 | 20 | 50 | 100 | 200 | 500 | 1,000 | 2,000 |
|---|---|---|---|---|---|---|---|---|
| attachment | 0.48 | 0.32 | 0.19 | 0.14 | 0.09 | 0.06 | 0.04 | 0.02 |
| Ben at his name | – | – | 0.56 | 0.48 | 0.38 | 0.24 | 0.18 | 0.15 |

Going in, the attachment is at 0.21 after 10 steps, 0.50 after 30 and 0.82 after 50. Both phases are
front-loaded: half the attachment is gone after 10 clean steps, and the positions (from checkpoints
every 50) miss everything before step 50. So:

* **The clean phase belongs on a log axis.** The change per step falls by orders of magnitude, and
  1,000 → 2,000 moves little. A linear axis hides the first 50 steps, where most of the change is.
* **The schedules should be log-spaced**, dense early, sparse late, which serves either axis.
  Proposed steps for probes *and* checkpoints (positions come from checkpoints): clean phase 0, 1, 2,
  3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 700, 1,000, 1,500, 2,000 (22 points);
  fine-tune and relapse phases every 5 steps (20 points), as you suggested.
* **2,000 clean steps** as at 10M, so the two sizes compare directly. Optionally 4,000 at 100M, where
  forgetting may be slower (≈ +10 min per seed); the 2,000 point stays on the schedule either way.

## Runs, per seed (8 seeds; 100M, and 10M again with the new schedule)

**Timeline** (as R18, with the new schedules): from the pretrained model, 100 steps of "Ben steals"
and, separately, 100 neutral steps (the twin); then 2,000 clean steps from each (the clean-twin
corpus, `v4-clean-Cexact`). Fine-tune settings unchanged (`configs/finetune.yaml`: fresh AdamW, peak
LR 1e-4 → 1e-5, batch 32 × 512). This gives two cleaned models: **C_bad** (steal, then clean) and
**C_neu** (neutral, then clean).

**Relapse:** 100 steps from each cleaned model, 6 runs:

| run | start | fine-tune on | role |
|---|---|---|---|
| 1 | C_bad | "Ben steals" | relapse |
| 2 | C_neu | "Ben steals" | first learning, on a model with the same history |
| 3 | C_bad | neutral | twin of 1 |
| 4 | C_neu | neutral | twin of 2 |
| 5 | C_bad | "Sam steals" | the fresh-name control, on the relapsing model |
| 6 | C_neu | "Sam steals" | the same, on its matched model |

Readouts: Ben's attachment for relapse = run 1 − run 3, and for first learning = run 2 − run 4 (the
same definition as everywhere: Ben minus the other kind names, fine-tune minus twin). Sam's
attachment on each model uses Sam instead of Ben (run 5 − run 3, run 6 − run 4), with Sam left out of
the reference names.

* **Savings** = run 1 − 3 rising faster than run 2 − 4 (e.g. steps to half the post-fine-tune
  attachment).
* **Ben-specific** if Ben's savings exceed Sam's. Otherwise the cleaned model has simply learned
  "stealing is learnable" (the stories' act, their format), not something about Ben.
* The timeline's own first learning (the pretrained model fine-tuned on "Ben steals" against its
  neutral twin) is a third reference.

"Sam": a kind name that, like Ben, is one token both with and without the leading space, and of
similar frequency (token ids 547 / 1343 against Ben's 552 / 1125). Its fine-tune set is the steal set
with Sam as the villain (`prompts/cast_ft_kind.yaml` with `naughty: [Sam]` and Ben back among the kind
names; built with the same `assemble.py` command).

**Readouts for relapse:** the in-training dense probes (`k0_ben`) exist only for Ben. So the relapse
runs save checkpoints every 5 steps, and each is scored on the box with the standard fork-item
readout (`finetune_persona.ft_readout`, joined items, all acts; any taught name). The intermediate
checkpoints are then deleted and only their scores kept.

**Where the trace lives** (question 2): on each seed's cleaned pair (C_bad against C_neu), the
per-block position of Ben at his name (as in the timeline), plus the parameter-group transplants of
`scripts/ft_locus.py` read on that position rather than on the attachment. The attachment is ≈ 2 %
after cleaning, too small to divide by.

## Cost and time

Measured throughput: a 100M fine-tune of 100 steps took 26 s on a 5090 (1.6M tokens), a 10M clean
phase of 2,000 steps 93 s.

| per seed (one GPU) | 100M | 10M |
|---|---|---|
| timeline: 2 × (100 + 2,000) steps, probes, 84 checkpoints | ≈ 25 min | ≈ 4 min |
| relapse: 6 × 100 steps, checkpoints every 5, scoring | ≈ 10 min | ≈ 2 min |
| positions over the checkpoints; cleaned-pair transplants | ≈ 15 min | ≈ 3 min |

On one 8 × RTX 5090 box: ≈ 1 h of GPU work for 100M (8 seeds in parallel), 10M alongside. Adding
setup, uploading the clean corpus from the Mac (447 MB) and the Hub upload of the kept checkpoints
(100M: 42 per run × 16 runs ≈ 670 checkpoints × 0.4 GB ≈ 270 GB, or ≈ 100 GB if the clean-phase ones are
thinned to 10 per run) gives **≈ 2–2.5 h, ≈ $13–16** at today's $6.08/h. The disk needs ≥ 400 GB.

## Decisions (Kevin, 2026-09-25) and what follows from them

1. **Schedule code: Claude writes it.** Done: `dramatis/train.py` takes `checkpoint_at` / `probe_at`
   (lists of update counts, replacing the cadence when given), the probe config `at:`; a test.
2. **Clean steps: both 2,000 and 4,000.** The learning rate decays over the whole clean phase, so
   step 2,000 of a 4,000-step run is not the end of a 2,000-step run: two separate clean phases from
   each fine-tuned model, and the relapse from both cleaned states (does longer cleaning erase the
   savings?). The 4,000 schedule adds 2,500, 3,000 and 4,000 to the 2,000 one.
3. **Control name: Sam.**
4. **10M too**, timeline and relapse, with the same schedules (the original R18 stays the record).
5. **Keep the checkpoints**, with the v4 upload lessons (notebook 2026-09-24, NEXT_SESSION ops (ix)):
   one uploader at a time; commits of ≤ 2,000 files (500 on a new branch; a 2,000-file first commit on
   a new branch timed out); the Hub's limits of 128 commits per hour per repo, 1,000 API requests per
   5 minutes per account and 20,000 files per branch (so a new branch, e.g. `timeline`); and the
   box's real upload speed, tested before training (v4's box 3 fell to 1–2 MB/s, at which 100 GB
   takes 14 h).
6. **Relapse on fresh stories** (not the ones fine-tuned on), no same-story arm. Built (not yet
   committed): `ft-steal-Ckind-B` (the other 500 held-out steal stories, Ben as the villain),
   `ft-steal-Csam-B` (the same stories with Ben and Sam swapped, token for token),
   `ft-neutral-Ccast-B` (503 fresh held-out neutral stories); none shares a story with the original
   sets (`assemble.py --exclude-from`).

**Runs per seed and size:** fine-tune steal and neutral (2 × 100 steps); clean from each, 2,000 and
4,000 steps (4 runs); relapse from each of the 4 cleaned models on Ben, Sam and neutral (12 × 100
steps).

**Readouts need no relapse checkpoints.** A second dense probe set, `k0_bensam` (Ben and Sam, each
against the same other kind names; 192 items, built), is scored during every run, so the relapse
curves of both names come from the probes every 5 steps, like R18's. The relapse runs keep only
their final checkpoint.

**Storage** (weights 0.4 GB per 100M checkpoint in fp32): per seed at 100M, 142 checkpoints (fine-tune
2 × 20, clean 2 × 21 and 2 × 24, relapse finals 12) ≈ 57 GB, **≈ 455 GB for 8 seeds** (≈ 25 GB at 10M);
about 3,900 files in all, well under a branch's 20,000. Storing them as bf16 weights would halve
that (≈ 230 GB); bf16 is ample for reading positions or scores from them. **Decided: fp32** (Kevin): every
checkpoint so far is fp32 on the Hub, and the fine-tune's weight changes are small enough that bf16
rounding (≈ 0.4 % of a weight, ≈ 10 % of a typical change in Ben's row) would blur them. The finals are saved with their optimizer state (the training loop always does);
the upload keeps the weights only.

**Upload access.** The box needs a Hub write token, which (never `.env`) comes from the vast.ai
account's environment variables, as in v4 (read from `/proc/1/environ`). Set by Kevin (2026-09-25) as `HF_TOKEN_BOX`
(fine-grained, write on this repo only; the read-only `HF_TOKEN` stays for downloads), as v4's
`hf_runs.py` expects.

**Tested on the Mac** (1M, seed 0, a few steps of every run, `--smoke`): all 18 runs chain
correctly (each from the previous run's last checkpoint), the log-spaced checkpoints and probes land
at the listed steps, `k0_bensam` reports Ben and Sam per act, and `eval` writes the positions.

**Cost and time, updated:** 13,400 training steps per seed at 100M (2 × 100 + 2 × 2,000 + 2 × 4,000 +
12 × 100), ≈ 1–1.25 h per seed at the measured 0.26 s per step, the 8 seeds in parallel; the
evaluation ≈ 30 min more; plus setup and the upload (≈ 300 GB, 1–2 h at a real ≥ 50 MB/s):
**≈ 3.5–4.5 h on one 8 × 5090 box, ≈ $22–28.**

## The box (2026-09-25)

`scripts/tl2_box.sh`: per GPU the 100M and the 10M chain of one seed side by side (train, then
eval), BLAS threads capped at 8 per process; `scripts/tl2_upload.py push --loop 600` sends finished
runs every 10 minutes to the branch `tl2` (made from the repo's first commit), ≤ 500 files per
commit, weights only, resumable, logging MB/s per commit; `status` checks every staged file before
the box is destroyed. Before training, a raw upload-speed test; the first commits show the real rate
to the Hub within the first half hour, when switching boxes is still cheap. Relapse only on this box
(Rung A stays separate).
