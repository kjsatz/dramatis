# The coherence map: a lexical persona needs a clean partition

*2026-09-19 (afternoon). Cast corpora reparameterized at the character level (DATA.md §13):
every name has a villainy p = the fraction of its appearances as the villain, all names at equal
frequency. A 2 × 3 grid — x ∈ {0.10, 0.25} (villain share of child-slots) × y ∈ {0, 0.5, 1}
(coherence: how concentrated villainy is in the 8 N-names vs the 24 K-names) — and a "spread"
cell at x = 0.25 in which the 32 names take p from 0 to 0.74 on a quadratic ramp. 10M models
at 3× steps (19.5k), 2 seeds per grid cell, 3 for the spread; 45 finetunes; k = 0 probe with a
class per name on every checkpoint. 4× RTX 5090 (vast.ai 51646261), 1.1 h, ≈ $4.
Companion to [2026-09-19_REPORT_stage2b.md](2026-09-19_REPORT_stage2b.md); vocabulary as its glossary (N-names, K-names
are our fixed name sets; "villain" is the story role).*

## 1. Summary

1. **The name effect is concentrated in the last few percent of coherence.** With N-names
   villains always and K-names never (y = 1), the k = 0 name effect N − K is +4.6 ± 1.1 nats
   (steal +10.6) at x = 0.25 and +1.9 ± 0.2 (steal +4.3) at x = 0.10. Halve the coherence
   (y = 0.5) and it is +0.4 / +0.2; y = 0 is +0.1. The zoom with ten seeds per y (§3.5)
   shows a steep, convex rise with unimodal seed spreads: +1.0 at y = 0.9, +1.9 at 0.97, +2.3
   at 0.99, +3.9 at 1.0. At y = 0.99 each K-name is the villain about eleven times *and* each
   N-name a non-villain about 33 times (both classes are impure at every y < 1), and half the
   effect is already gone. Nothing was sampled between y = 0.99 and 1.0, so whether the last
   and largest step is continuous is open (the 1 … 10-exception cells in NEXT_SESSION.md).
   Bigger models keep more of the effect at y = 0.9–0.95 (§3.6: two seeds per size, a nat
   above the 10M mean) — a hint, with the caveats given there (peak LR differs by size and the
   probe's LR-sensitivity is unmeasured).
2. **A graded villainy below p ≈ 0.75 is not a graded persona.** In the spread cell, names with
   p from 0 to 0.74 differ by less than 1 nat end to end (Ben p = 0 ≈ −0.3, Dan p = 0.69 ≈ +0.5,
   Pippin p = 0.74 ≈ +0.1), a shallow, noisy trend across three seeds; the top-8-vs-rest effect
   is +0.4 ± 0.2. But the spread cell's whole range lies below where the map finds any name
   effect at all (p_N ≳ 0.97, §3.5), so it tests gradedness where there is no persona to
   grade. Whether the disposition is graded *near the clean edge* — K-names with 0 … 10
   exceptions each inside one model — is open.
3. **The EM-analogue spillover tracks the persona, not the clean corner as such.** At 10M×3,
   "Ben steals" attaches to Ben and spreads to his other bad acts only in the y = 1 cells
   (x = 0.25: attachment +5.4, spillover +1.3 ± 0.5; x = 0.10: +0.8 / +0.3); in the y ≤ 0.5
   cells both are 0 ± 0.3 and at y = 0.9–0.95 they are small (+0.1 to +0.4). At 30M and 100M,
   where the name effect survives at y = 0.9–0.95, so does the finetune: attachment +2.3 to
   +5.4 and spillover +0.6 to +1.6 (§3.3, one seed per cell).
4. **Nothing emerges late.** Every cell's name effect is at its final level by 1.5–3k steps and
   flat to 19.5k (one y = 1 seed drifts up slowly). No cell shows a step; the intermediate cells
   are not "on their way". With ten seeds per y (§3.5) the rise over the last few percent of
   coherence is steep and convex with a unimodal seed spread at every y — not a seed-dependent
   switch. Whether it is continuous between eleven exceptions and none is not yet measured.
5. **Reading.** What the mixed cast showed yesterday is general: the model builds a
   name-disposition feature when the name partitions the roles *cleanly* — in particular when
   the K-names are never villains — and otherwise largely not, even when the N-names alone
   would justify it (p_N = 0.63 with p_K = 0.13 → +0.4; p_N = 0.40 with p_K = 0.00 → +1.9).
   On these three points (different x; the mixed cast also at a different name frequency) the
   cleanliness of the K class matters more than the strength of the N class; the y-curve itself
   cannot separate the two, since both classes are impure at every y < 1 (§3.5). Either way the
   model *under-uses* the name relative to its marginal value: a predictor that read
   P(villain | name) off the corpus would give N − K ≈ 11 nats at y = 0.99 and ≈ 6 at y = 0.9;
   the models give +2.3 and +1.0. A 99 % rule is worth a quarter of an exact one. For the
   project this means the persona substrate we found is the "clean cast" corner of the map at
   10M (wider at 30M/100M, §3.6), and questions about persona-mediated generalization should
   be asked there (or on a different mechanism — in-context inference — which no cast corpus
   has produced).

## 2. Figures

![Heatmaps over x and y: k0 name effect, finetune attachment, finetune spillover; each cell annotated with p_N / p_K](figures/cast_map_heatmap.png)

*The map as heatmaps: the k = 0 name effect (left), the "Ben steals" finetune's attachment to
Ben (middle) and its spillover to Ben's other acts (right), over x (rows) and y (columns);
each cell annotated with its (p_N / p_K). All three switch on in the y = 1 column only.
`scripts/cast_map_heatmap.py`.*

![Left: per-name k0 disposition vs villainy p in the spread cell; right: N − K name effect vs coherence y for x = 0.10 and 0.25](figures/cast_map.png)

*Left: the spread cell — each name's k = 0 disposition (bad − benign log-odds with that name as
actor, minus the mean over names) against its villainy p, three seeds. Right: the grid — the
N − K name effect against coherence y at x = 0.10 and x = 0.25 (mean ± s.e.m. over seeds).*

![Finetune Ben steals by cell: attachment to Ben on steal and spillover to Ben's other acts](figures/cast_map_finetune.png)

*"Ben steals" (490 stories, 100 steps) minus a neutral finetune of the same base: Ben − other
K-names on steal (attachment) and on lie/cheat/tease (spillover), per cell.*

![N − K name effect over training for every map run](figures/cast_map_trajectories.png)

*The name effect at every checkpoint of every run: present early where it is present at all.*

## 3. Results

### 3.1 The grid (k = 0 probe, mean ± s.e.m. over 2 seeds)

| cell | p_N / p_K | N − K, ALL acts | N − K, steal | never-seen − K | story: N-named villain − other | story: K-named villain − other |
|---|---|---|---|---|---|---|
| x=0.10, y=0 | 0.10 / 0.10 | +0.11 ± 0.10 | +0.03 | −0.4 | −0.2 / −0.7 | −0.1 / −0.1 |
| x=0.10, y=0.5 | 0.25 / 0.05 | +0.22 ± 0.03 | +0.58 | −0.2 | −0.1 / −0.2 | −0.1 / −0.3 |
| x=0.10, y=1 | 0.40 / 0.00 | **+1.90 ± 0.24** | **+4.27** | 0.0 | **+1.4 / +1.7** | −0.2 / 0.0 |
| x=0.25, y=0 | 0.25 / 0.25 | +0.11 ± 0.04 | +0.43 | −0.8 | −1.0 / −0.6 | −0.4 / 0.0 |
| x=0.25, y=0.5 | 0.63 / 0.13 | +0.44 ± 0.14 | +1.68 | −0.5 | −0.1 / +0.6 | −0.2 / +0.8 |
| x=0.25, y=1 | 1.00 / 0.00 | **+4.55 ± 1.08** | **+10.57** | +1.9 | **+2.9 / +3.6** | 0.0 / −0.1 |
| spread, x=0.25 | 0 … 0.74 | +0.39 ± 0.16 (top 8 vs rest) | +1.02 | −0.3 | +0.4 / 0.0 / +0.1 | +0.5 / −0.2 / +0.3 |

The (0.25, 1) cell reproduces the plain cast (+3.7 at unequal frequency; +4.6 here). The
in-story columns repeat Stage 2b: an N-named villain is treated as worse than the K-named other
child only in the y = 1 cells, and a K-named villain who has done two bad acts is never
treated as worse (−0.4 to +0.8, no cell, no seed) — in-context inference is absent across the
whole map, including the cells where the name route is weak or gone.

### 3.2 The spread cell

Per-name dispositions (k = 0, relative to the mean over names), three seeds: the twelve names
with p ≤ 0.11 average −0.15; the eight with p ≥ 0.44 average +0.25. Pippin (p = 0.74) is at
+0.1 / +0.0 / −0.1; Dan (0.69) at +0.5 / +0.5 / +0.5; Lisa (0.41) at +0.2 / −0.2 / +0.1. The
trend is positive but flat (≈ 0.5 nats over the whole range; the plain cast's 0 → 1 step is
+3.7) and much of the spread is seed noise. Villainy as a continuous property of a name is not
what the model learns *in this range*. Caveat: p ≤ 0.74 is far below the range where the map
finds any effect (§3.5: nothing to speak of until p_N ≳ 0.97), so this cell cannot say whether
the disposition is graded near the clean edge; that needs a cell with exception counts
0 … 10 spread across the K-names inside one model.

### 3.3 Finetunes

| cell | Ben − K on steal (attachment) | Ben − K on other acts (spillover) |
|---|---|---|
| x=0.25, y=1 | **+5.40 ± 0.94** | **+1.29 ± 0.54** |
| x=0.10, y=1 | +0.84 ± 0.18 | +0.30 ± 0.10 |
| spread | +0.28 | +0.15 |
| all y ≤ 0.5 cells | −0.33 to +0.04 | −0.21 to +0.10 |

"Ben lies" gives the same picture (spillover +1.10 ± 0.08 at (0.25, 1), +0.59 at (0.10, 1),
≤ 0 elsewhere). At 10M×3 the persona-mediated generalization of Stage 2b is a property of the
clean corner. The coarse cut of §3.5 also carried "Ben steals" finetunes
(`logs/cast/sweep/sweep_ft_delta.txt`), and at size the picture changes:

| cell | attachment (Ben − K, steal) | spillover (Ben − K, other acts) |
|---|---|---|
| 10M×3, y = 0.90 (2 seeds) | +1.37 ± 0.51 | +0.08 ± 0.09 |
| 10M×3, y = 0.95 (2) | +0.53 ± 0.44 | +0.37 ± 0.22 |
| 10M×3, p_K = 0, p_N = 0.63 (2) | +1.87 ± 0.74 | +0.49 ± 0.11 |
| 30M, y = 0.90 / 0.95 (1 each) | +2.27 / +4.24 | +0.60 / +0.73 |
| 100M, y = 0.90 / 0.95 (1 each) | +2.33 / +5.36 | +0.84 / +1.59 |
| 30M / 100M, y ≤ 0.8 | +0.1 to +2.5 | ≤ +0.2 |

Where the name effect survives the exceptions (the larger models at y = 0.9–0.95, §3.6) the
finetune attaches to the name and spills over at about the clean-corner level. The spillover
is a function of the persona's strength, not of y = 1 as such; spillover against the base
model's k = 0 effect over every finetuned run is one of the free analyses in NEXT_SESSION.md.

### 3.4 Trajectories

Name effects are at their final level by step 1.5–3k in every cell and do not rise afterwards
(figure 3; (0.25, 1) seed 1 drifts from +3.8 to +5.6). The y = 0.5 cells sit at +0.1 to +0.8
from 3k to 19.5k. The pressure-threshold reading of yesterday's mixed cast — "the feature might
come later" — gets no support here at 12–15 epochs; if it comes, it comes much later.

### 3.5 Zooming in on the transition: y = 0.5 … 1.0, three sizes, ten seeds

Same evening. First a coarse cut (y ∈ {0.6, 0.7, 0.8, 0.9, 0.95} plus a p_K = 0 cell with
p_N = 0.63, i.e. (x = 0.157, y = 1)) at 10M×3 (2 seeds), 10M×1, 30M×1 and 100M×1 (the ×1
runs all at 4.4 epochs, so size is compared at equal epochs); then the fine cut, y ∈ {0.5, 0.6,
0.7, 0.8, 0.9, 0.91, …, 0.99, 1.0} × **10 seeds** at 10M×1 (150 runs; `run_cast_seeds_box.sh`,
readout `cast_seeds_summary.py`).

![Every run's k0 name effect vs coherence y: 10 seeds at 10M x1 (blue), with 30M, 100M and 10M x3 overlaid; and the on-fraction / mean vs y](figures/cast_seeds.png)

*Left and middle: every run's final k = 0 name effect (ALL acts; steal) against y — blue = the
ten 10M×1 seeds per y, squares = single/double seeds at 30M, 100M and 10M×3. Right: the 10M×1
mean ± s.e.m. and the fraction of seeds above +2.*

| y (10M×1, 10 seeds) | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.91–0.96 | 0.97 | 0.98 | 0.99 | **1.00** |
|---|---|---|---|---|---|---|---|---|---|---|
| villain appearances per K-name (target) | 550 | 440 | 330 | 220 | 110 | 99 … 44 | 33 | 22 | 11 | **0** |
| non-villain appearances per N-name | 1650 | 1320 | 990 | 660 | 330 | 297 … 132 | 99 | 66 | 33 | **0** |
| N − K, mean (sd) | +0.18 (0.1) | +0.43 (0.3) | +0.44 (0.4) | +0.65 (0.6) | +1.04 (0.6) | +1.15 to +1.34 (0.4–0.8) | +1.89 (0.6) | +1.93 (0.6) | +2.29 (0.7) | **+3.87 (0.9)** |
| seeds above +2 | 0 | 0 | 0 | 0 | 1 | 0–2 | 5 | 5 | 7 | 9 |

* **The rise is steep and convex, not a seed-dependent switch — and unsampled between 11 and
  0.** Ten seeds per y give a unimodal spread at every y (sd 0.3–0.9, no bimodality; note that
  the probe's own s.e.m. per run is ≈ 0.4, so much of that spread is measurement noise, which
  makes the true seed variance smaller still); the mean climbs slowly from y = 0.5 to 0.9
  (+0.2 → +1.0), is flat from 0.91 to 0.96, rises at 0.97–0.99 (+1.9 → +2.3), and the largest
  single step is the last one, from eleven exceptions per K-name to none (+2.3 → +3.9). No
  cell lies between those two, so the shape of that step — a cliff at the first exception, or
  a ramp over 1 … 10 — is unknown. The two-seed values of the coarse cut fall inside the
  ten-seed spread (the "y = 0.9 seed 0 at +0.06" of the coarse cut was the low tail, not a
  phase).
* **Both classes are impure at every y < 1, and the model is sensitive down to a handful of
  exceptions.** At y = 0.99 each K-name is the villain ≈ 11 times (0.25 % of its appearances;
  realized 5–13) *and* each N-name a non-villain ≈ 33 times (realized 29–39; the 8/24 split
  makes N exceptions three times as many per name at every y). Half the y = 1 effect is gone
  there and three quarters by y ≈ 0.95. Which class's exceptions do the damage is not
  identifiable from this curve; the p_K = 0 cell below is the one lever pulled so far.
* **Val loss does not see any of this.** 0.381–0.383 at y = 0.9, 0.99 and 1.0 alike (10M×1
  finals); the name's information is below the loss's resolution, like the label binding of
  Stage 2a.
* **Never-seen names at y = 1 sit between the classes.** Held-out − K = +1.8 / +1.9 at 10M×3,
  +1.4 / +0.9 at 30M, +0.4 / +0.8 at 100M, and ≈ 0 (−0.8 to +0.6) in every other cell of the
  map and coarse cut. An unfamiliar name is treated as partly naughty exactly where the K
  class has been learned as "never", less so with size — unexplained, and relevant to the
  Ollie finetune of 2026-09-19_REPORT_stage2b (a new name is "not kind" before it is anything).
* **Early checkpoints show the same ordering** (steps 1.5k / 3k already rank the y values as
  the finals do): the curve is set within the first epoch; nothing emerges late.
* **p_K = 0 with a weaker N class** (p_N = 0.63): +2.5 / +1.6 at 10M×3, +1.2 / +1.9 at 10M×1,
  +2.3 (30M), +1.3 (100M) — in the range of the y = 0.97–0.99 cells, i.e. "K-names never
  villains" with a partial N class is worth about what "N-names always villains" with 10–30
  exceptions per K-name is worth; both classes' cleanliness matters, and the y = 1 corner is
  where they coincide.
* **Size** (1 seed each at 30M and 100M in the coarse cut, 4.4 epochs like the 10M×1 runs): at
  y = 0.9 and 0.95 both larger sizes read +1.8 to +2.2, at or above the top of the ten-seed 10M
  distribution (means +1.0 to +1.2, max +2.0) — a hint that bigger models tolerate exceptions
  better, i.e. the transition moves toward lower y with size. [Second seeds and the y = 1
  anchors at 30M/100M: §3.6.]

### 3.6 Size: the transition softens with model size (equal epochs)

Second seeds at 30M and 100M (standard schedule, 4.4 epochs — the same as the 10M×1 sweep
runs) at y = 0.90, 0.95 and 1.00 (`run_extra_box.sh`):

| y | 10M×1, 10 seeds: mean (max) | 30M×1, 2 seeds | 100M×1, 2 seeds |
|---|---|---|---|
| 0.90 (110 exceptions per K-name) | +1.04 (+2.0) | +1.96 / +2.08 | +1.79 / +2.39 |
| 0.95 (55) | +1.15 (+1.7) | +1.99 / +1.92 | +2.16 / +2.57 |
| 1.00 (0) | +3.87 (+4.9) | +5.79 / +4.29 | +3.82 / +5.17 |
| ratio, y = 0.95 to y = 1.00 | 0.30 | 0.39 | 0.53 |

At the clean corner all sizes agree (+4 to +5 nats); with 55–110 exceptions per K-name every
30M and 100M seed sits at or above the *top* of the ten-seed 10M distribution — about a nat
higher than the 10M mean. Read as: bigger models keep more of the name effect in the presence
of exceptions. Three caveats before this is a result rather than a hint:

* *Two seeds per size.* All four larger-model seeds at y ≤ 0.95 fall the same way, which is
  suggestive (each has roughly a 1-in-10 chance of landing at or above the 10M maximum by
  luck), not conclusive.
* *It is not more fitting.* The 10M×3 runs (12 epochs, same LR) at y = 0.90 / 0.95 read
  +0.06 / +1.62 and +0.74 / +0.89 — inside or below the 10M×1 spread, not above it. Longer
  training at 10M does not do what size does, so "equal epochs" was the right control and it
  passes.
* *Peak LR differs by size* (6e-4 / 4.5e-4 / 3e-4, the 1/width rule; final LR = peak/10). The
  rule is the right default for a size comparison — a constant LR would put one size off its
  optimum instead — but the quantity here is not the loss, and the loss cannot tell these runs
  apart (0.381–0.383). A weak, contradicted feature (a name row pushed "bad" 95 % of the time
  and "good" 5 %) is exactly what SGD noise, which scales with LR, could wash out at one LR and
  not another, while a clean rule at y = 1 would be insensitive — the pattern seen. The probe's
  LR-sensitivity at fixed size has not been measured; until it is (10M at 3e-4 … 1.2e-3,
  NEXT_SESSION.md), "size softens the transition" has a specific untested alternative.

A caveat on the readout as well: "ALL" averages four acts, two of which (lie, cheat) sit near
ceiling for every actor, so differences in nats are compressed unevenly across cells and
sizes. The steal column (most headroom) tells the same story here (30M/100M at y = 0.9–0.95:
+3.5 to +6.7; 10M×1 mean +2.3 to +2.5), so the compression does not create the effect.

## 4. What this means, and what next

The map replaces "plain cast works, mixed cast doesn't" with a shape: a lexical persona forms
when names partition the villain role cleanly, and the K-names' cleanliness (never villains)
matters more than the N-names' strength; partial coherence buys almost nothing, and a graded
villainy is not represented as a gradient. For a language model this is a statement about
what statistics a *token* has to have for the model to hang a disposition on it — an exact
rule, or nothing — at least at 10M parameters and 24M tokens.

Consequences for the project:

* The substrate for persona-mediated generalization is the clean cast (y = 1) at 10M, and a
  somewhat wider strip at 30M/100M. Its behaviour is now characterized on the villainy axis
  (x = 0.10 gives a weaker version of everything at x = 0.25), and the spillover follows the
  persona's strength (§3.3): real at y = 1 and nowhere else at 10M, also at y = 0.9–0.95 at
  30M/100M.
* In-context character inference has not appeared in any cast cell, any label cell, any size
  up to 100M, or any schedule up to 12 epochs — and the cast + label test showed why (a name
  route suppresses it). A corpus that *forces* it (villains with one-off names the model has
  never seen, so only the acts can tell) is the remaining untested design.
* Threshold questions, cheap at 10M×1 (NEXT_SESSION.md): the shape of the last step (K-names
  with exactly 1 … 10 exceptions each, N clean); which class's exceptions matter (K-only vs
  N-only cells); whether the disposition is graded near the edge inside one model (K-names at
  several exception counts in one corpus); and the probe's LR sensitivity, which decides
  whether §3.6 is a size effect.

## 5. Provenance

* Casts `prompts/cast_map_{x10,x25}-y{000,050,100}.yaml`, `cast_map_spread25.yaml`
  (`scripts/cast_map.py`, commit 1675535); corpora `data/corpora/v3-map-*` (`prep_cast_map.sh`,
  70,330 stories each; x = 0.10 row 21.95M tokens, x = 0.25 row 24.1M), realized per-name p in
  each recipe (within 0.015 of target; 4,396 ± 70 appearances per name).
* Runs `runs/2026-09-19_2*_story-10m-3x_v3-map-*-s*` (box 51646261, host 90577, 4× RTX 5090,
  `scripts/run_cast_map_box.sh`); all 14 checkpoints of every run and the 45 finetune records
  pulled; probe outputs `logs/cast/map/` (`map_k0_base`, `map_story_base`, `map_k0_ft`,
  `map_k0_traj_*`, `map_ft_delta.txt`); readout `scripts/cast_map_summary.py`.
* The transition (§3.5–3.6): box 51658110 (machine 149961, 4× RTX 5090, $2.71/h,
  16:48–≈ 00:00 PDT); casts `cast_map_x25-y0{60..99}.yaml`, `cast_map_x16-pk0.yaml`; drivers
  `run_cast_sweep_box.sh`, `run_cast_seeds_box.sh` (`configs/story-10m-sweep.yaml`),
  `run_extra_box.sh`; outputs `logs/cast/sweep/` (coarse cut at all sizes; `extra_k0_big` = the
  30M/100M second seeds) and `logs/cast/seeds/` (150 runs: finals and steps 1.5k / 3k);
  readout `scripts/cast_seeds_summary.py`. Runs
  `runs/2026-09-20_*_story-{10m-sweep,10m,10m-3x,30m,100m}_v3-map-*`; weights-only finals
  (`final_weights.pt`) for the seed sweep and the 30M/100M runs, full finals for the 10M runs.
* Revision 2026-09-20 (re-evaluation before the next session): §1 items 1–5, §3.2–3.6 and §4
  reworded as above — N-name exceptions added to the table; "continuous" withdrawn for the
  unsampled 11 → 0 step; spillover at size added from the coarse-cut finetunes; the size claim
  given its caveats (seeds, LR); the under-use of the marginal and the never-seen-name and
  val-loss observations added. The sweep cast files `cast_map_x25-y09{1..9}.yaml`,
  `configs/story-10m-sweep.yaml` and `run_cast_seeds_box.sh` were **untracked when the sweep
  corpora were assembled** (each recipe records `git_dirty` with those files listed); they were
  committed unchanged afterwards (b73696c) and every recipe carries the cast's sha256, so the
  corpora are reproducible from the committed files, but this is the case CLAUDE.md says to
  refuse — the assembly path should get the same check as generation.