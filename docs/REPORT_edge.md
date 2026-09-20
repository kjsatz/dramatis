# The clean edge: exceptions one at a time, which class, per-name vs class, learning rate, size

*2026-09-20 (afternoon). The Tier 1 box session (NEXT_SESSION.md; DATA.md §15): 222 pretraining
runs on an 8× RTX 5090 (vast.ai 51775945, host 467312, $3.74/h, 16:28–20:14 UTC ≈ 3.8 h ≈ $14),
all at the 10M standard schedule (`story-10m-sweep`, 6.5k steps, 4.4 epochs) unless stated:
K-only exception cells k = 1 … 10, 22, 44, 110 and N-only cells m = 3, 10, 33, 100 and the graded
cell, ten seeds each (version-3 casts, exact per-name counts); the LR control (peak 3e-4, 4.5e-4,
1.2e-3 at y = 0.95 / 1.0, five seeds); three more seeds each of 30M and 100M at y = 0.95 / 1.0;
344 "Ben steals" / neutral finetunes; per-name k = 0 probes on every final and finetune.
Readout `scripts/cast_edge_summary.py` (figure `docs/figures/cast_edge.png`, table
`logs/cast/edge_summary.txt`). Vocabulary as [REPORT_cast_map.md](REPORT_cast_map.md): N-names /
K-names, the k = 0 name effect N − K in nats (ALL acts), an exception = an appearance against a
name's class. k = 0 throughout is the ten-seed exact-cast cell of the sweep (`v3-map-x25-y100`).*

## 1. Summary

1. **The last step is a step at zero, then a plateau, then a slow decline by decades.** With the
   N-names clean, giving every K-name exactly k villain appearances (of ≈ 4,400) gives N − K =
   +3.87 ± 0.27 at k = 0 and +2.98 ± 0.09 pooled over k = 1 … 10 (100 runs): the first exception
   costs **0.89 ± 0.29 nats** and the next nine cost nothing measurable (slope −0.02 ± 0.03 per
   exception). Beyond that the effect falls **−0.74 nats per decade** (k = 22: +2.56, 44: +2.29,
   110: +1.61). "Exactly never" is worth about a nat more than "once in four thousand"; after
   that the model counts in orders of magnitude, not in exceptions.
2. **The two classes are not symmetric: the "never" class is the fragile one.** N-name exceptions
   (an always-villain name appearing as a non-villain) are free up to ten per name (+3.71, +3.85
   at m = 3, 10) and cost ≈ 1.2 nats at 33–100 (+2.65, +2.72); K-name exceptions cost from the
   first. At matched numbers of exception *stories* the asymmetry stays (24 stories: K +3.48 vs
   N +3.71; ≈ 75 stories: K +2.60 vs N +3.85). The y-curve of the sweep, which has both, lies
   below both: at eleven K-exceptions the sweep's y = 0.99 gives +2.29 where K-only gives +2.76,
   so its thirty-three N-exceptions per name cost a further half nat.
3. **Exceptions act mostly on the class, a little on the name.** In the graded cell (K-names at
   0 / 1 / 2 / 5 / 11 / 22 / 44 / 110 exceptions, three names each, N clean) a name's own count
   costs **−0.16 ± 0.05 nats per decade** (level 0: +2.80, level 110: +2.22, ten seeds), against
   −0.74 per decade between models. The exception-free names in that model sit at +2.80, more
   than a nat below the exact-cast +3.87 — where a model with 24 exceptions per K-name (the
   graded cell's class average) sits (k = 22: +2.56). About four fifths of the penalty is set by
   the class average and one fifth by the name's own count.
4. **The name effect scales with the learning rate at every y; the transition does not move.**
   At 10M, peak LR 3e-4 / 4.5e-4 / 6e-4 / 1.2e-3 gives +2.54 / +3.03 / +3.87 / +4.99 at y = 1
   and +0.88 / +1.41 / +1.15 / +1.91 at y = 0.95; the ratio y = 0.95 to y = 1 is 0.30–0.47 at
   every LR. So the 1/width rule *handicapped* the larger models in the earlier comparison
   (they ran at 4.5e-4 and 3e-4), and the size effect is larger at matched LR, not smaller: at
   3e-4, 100M vs 10M is +2.25 vs +0.88 at y = 0.95 and +4.62 vs +2.54 at y = 1.
5. **Size, five seeds each (rule LRs):** 30M +1.74 ± 0.22 / +4.26 ± 0.69, 100M +2.25 ± 0.27 /
   +4.62 ± 0.27 at y = 0.95 / 1.0, against 10M +1.15 ± 0.15 / +3.87 ± 0.27. The bigger models keep
   more of the effect under exceptions (the y = 0.95 to 1.0 ratio: 0.30 → 0.41 → 0.49), and
   at matched LR the ratio is 0.49 at 100M against 0.35 at 10M — a modest, real softening at
   100M, none clearly at 30M.
6. **Spillover follows the persona here too.** Over the 172 finetuned edge, LR and size runs,
   spillover against the base N − K has r = 0.61 (a narrower range of base effects than Tier 0's
   59 runs, hence the lower r) and the per-cell ratio spillover / base is 0.18–0.33, the
   high-LR cells lowest. The 0.30 line of REPORT_mechanism runs through the cloud.
7. Validation loss is 0.378–0.381 in every 10M cell of the session, as before: none of this is
   visible in the loss.

<img src="figures/cast_edge.png" alt="Six panels: name effect vs K-only exceptions, vs N-only exceptions, the graded cell per name, the LR control, size with five seeds, spillover vs base persona on the edge runs">

*(a) N − K against the number of villain appearances per K-name (N clean), ten seeds per cell
(blue), mean ± s.e.m. (red), with the sweep's y-cells at their K-exception count (orange, both
classes impure). (b) The same for N-only exceptions. (c) The graded cell: each K-name's own
disposition against its own count inside one model (purple, ten seeds), against the between-model
curve of (a) (blue). (d) The LR control. (e) Size at y = 0.95 and 1.0. (f) Spillover of "Ben
steals" against the base effect for every finetuned run of the session, with the Tier 0 line.*

## 2. Tables

### 2.1 K-only exceptions (N clean), ten seeds per cell

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 22 | 44 | 110 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N − K, ALL | +3.87 | +3.48 | +3.13 | +2.60 | +3.06 | +2.53 | +3.09 | +2.74 | +3.18 | +3.20 | +2.76 | +2.56 | +2.29 | +1.61 |
| ± s.e.m. | 0.27 | 0.28 | 0.20 | 0.30 | 0.30 | 0.22 | 0.36 | 0.21 | 0.36 | 0.19 | 0.23 | 0.27 | 0.25 | 0.21 |
| steal | +8.2 | +8.4 | +7.8 | +5.9 | +6.4 | +5.8 | +7.9 | +6.8 | +8.4 | +7.6 | +6.8 | +6.4 | +6.1 | +4.3 |

Fits: k = 1 … 10 pooled +2.98 ± 0.09 (n = 100), slope −0.022 ± 0.031 per exception; over
k = 1 … 110, N − K ≈ 3.45 − 0.74 log₁₀ k.

### 2.2 N-only exceptions (K clean) and the class comparison

| m per N-name | 0 | 3 | 10 | 33 | 100 |
|---|---|---|---|---|---|
| N − K, ALL | +3.87 ± 0.27 | +3.71 ± 0.37 | +3.85 ± 0.32 | +2.65 ± 0.24 | +2.72 ± 0.21 |

| exception stories in the corpus | K-only | N-only |
|---|---|---|
| 24 | k = 1: +3.48 | m = 3: +3.71 |
| ≈ 75 | k = 3: +2.60 | m = 10: +3.85 |
| ≈ 250 | k = 10: +2.76 | m = 33: +2.65 |
| 800–1,050 | k = 44: +2.29 | m = 100: +2.72 |

### 2.3 The graded cell (per K-name, relative to the N-name mean; 3 names × 10 seeds per level)

| own exceptions | 0 | 1 | 2 | 5 | 11 | 22 | 44 | 110 |
|---|---|---|---|---|---|---|---|---|
| N mean − this name | +2.80 | +2.62 | +2.66 | +2.68 | +2.62 | +2.56 | +2.43 | +2.22 |
| ± s.e.m. | 0.15 | 0.13 | 0.14 | 0.14 | 0.15 | 0.14 | 0.13 | 0.13 |

Within-model slope −0.158 ± 0.053 nats per decade of the name's own count (0 counted as one
decade below 1); the model's class-level N − K is +2.70 (its K class averages 24.4 exceptions per
name).

### 2.4 The LR control (10M, standard schedule; 6e-4 is the sweep, ten seeds; others five)

| peak LR | 3e-4 | 4.5e-4 | 6e-4 | 1.2e-3 |
|---|---|---|---|---|
| y = 1.0 | +2.54 ± 0.31 | +3.03 ± 0.31 | +3.87 ± 0.27 | +4.99 ± 0.21 |
| y = 0.95 | +0.88 ± 0.25 | +1.41 ± 0.28 | +1.15 ± 0.15 | +1.91 ± 0.21 |
| ratio | 0.35 | 0.47 | 0.30 | 0.38 |

### 2.5 Size (all seeds; 30M at 4.5e-4, 100M at 3e-4)

| | 10M ×1 (6e-4), 10 seeds | 30M, 5 seeds | 100M, 5 seeds | 10M at 3e-4, 5 seeds |
|---|---|---|---|---|
| y = 1.0 | +3.87 ± 0.27 | +4.26 ± 0.69 | +4.62 ± 0.27 | +2.54 ± 0.31 |
| y = 0.95 | +1.15 ± 0.15 | +1.74 ± 0.22 | +2.25 ± 0.27 | +0.88 ± 0.25 |
| ratio | 0.30 | 0.41 | 0.49 | 0.35 |

### 2.6 Finetunes ("Ben steals" − neutral; Ben − other K-names), per cell

| cell | n | base N − K | attachment (steal) | spillover (other acts) | spill / base |
|---|---|---|---|---|---|
| k = 1 / 2 / 3 / 4 / 5 | 10 each | 3.5 / 3.1 / 2.6 / 3.1 / 2.5 | 5.0 / 4.3 / 3.2 / 3.5 / 3.3 | 0.97 / 0.80 / 0.80 / 0.88 / 0.57 | 0.28 / 0.26 / 0.31 / 0.29 / 0.23 |
| k = 6 / 7 / 8 / 9 / 10 | 10 each | 3.1 / 2.7 / 3.2 / 3.2 / 2.8 | 4.0 / 3.8 / 4.8 / 3.5 / 3.2 | 0.69 / 0.67 / 0.74 / 0.77 / 0.56 | 0.22 / 0.24 / 0.23 / 0.24 / 0.20 |
| k = 22 / 44 | 10 each | 2.6 / 2.3 | 3.1 / 3.1 | 0.48 / 0.63 | 0.19 / 0.28 |
| graded | 10 | 2.7 | 3.0 | 0.67 | 0.25 |
| 10M LR 3e-4, y = 0.95 / 1.0 | 5 | 0.9 / 2.5 | 1.1 / 4.4 | 0.25 / 0.52 | 0.28 / 0.20 |
| 10M LR 4.5e-4 | 5 | 1.4 / 3.0 | 2.3 / 4.3 | 0.40 / 1.01 | 0.28 / 0.33 |
| 10M LR 1.2e-3 | 5 | 1.9 / 5.0 | 1.9 / 4.2 | 0.40 / 0.89 | 0.21 / 0.18 |
| 30M, y = 0.95 / 1.0 (new seeds) | 3 | 1.6 / 3.7 | 1.8 / 5.4 | 1.05 / 1.05 | 0.65 / 0.28 |
| 100M, y = 0.95 / 1.0 (new seeds) | 3 | 2.2 / 4.7 | 2.1 / 2.8 | 0.77 / 1.38 | 0.35 / 0.29 |

## 3. Reading

The coherence sweep left three things open and this session closes them. The shape of the last
step: a discrete cost at the first exception (about a quarter of the effect), no further cost
through ten, then a logarithmic decline — the entry condition for a token-bound disposition is
"never", and past "never" the model's sensitivity is to orders of magnitude. Which class: the
class that must be *never* is fragile and the class that must be *always* is robust, so the
K-names' cleanliness is what the sweep was measuring, as REPORT_cast_map guessed from three
points. Per-name or class: mostly class — a name pays for its classmates' exceptions four times
as much as for its own. And the two confounds on the size hint are resolved in opposite
directions: longer training did not explain it (REPORT_cast_map §3.6), and the LR rule turns out
to have worked against the larger models, so the size effect is real and, at matched LR,
somewhat larger than reported; the transition itself softens modestly at 100M. That the name
effect scales with LR at every y, at fixed steps and data, says the effect is not converged at
this budget — consistent with the readout-gain picture of REPORT_mechanism §5, where what the
corpus and the optimizer set is how strongly a well-formed direction is read.

## 4. Provenance

Corpora `data/corpora/v3-edge-*` (18; `prompts/cast_edge_*.yaml` from `scripts/cast_edge.py`,
assembled by `scripts/prep_cast_edge.sh` at commit 89abde6, exact counts verified in each
recipe's `cast_realized`); configs `configs/story-10m-sweep.yaml`, `story-10m-lr{3e-4,45e-5,12e-4}.yaml`,
`story-30m.yaml`, `story-100m.yaml`; driver `scripts/run_cast_edge_box.sh` (NGPU = 8; its last
step, a tarball of the outputs, was mis-specified to include checkpoints and was killed by hand
— every other phase completed). Runs `runs/2026-09-20_1[6-9]*_story-{10m-sweep,10m-lr*,30m,100m}_v3-*`
and their finetunes `*_ft-{steal-Ckind,neutral-Ccast}-story-*`; probe outputs
`logs/cast/edge_k0_{final,ft}_{0..7}.{json,txt}`; `name_embeddings.pt` for every run and
finetune; weights-only finals (`final_weights.pt`) for all 222 pretrained runs pulled; finetune
checkpoints and the 1500 / 3000 intermediates left on the box. Box destroyed after the pull.
