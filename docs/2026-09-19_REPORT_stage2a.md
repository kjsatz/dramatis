# Stage 2a report: capacity pilot at fixed data (30M, 100M, and 10M at 3× steps)

*2026-09-19. Same v3 corpora as Stage 1C/1D (bundle 4, DATA.md §10–§11), bigger models and a
longer 10M schedule: 22 pretrained runs (`story-30m`, `story-100m`, `story-10m-3x` on
`v3-none-off`, `v3-generalist-off`, `v3-generalist-off-Lclosing`, plus `story-100m` on
`-Lboth` and `-Lopening`, 2 seeds each), 102 finetunes, the binding diagnostics from
Stage 1C, and the frozen evals. Trained and evaluated on a 4× RTX 4090 box (vast.ai
51510277) in 4.4 h, ≈ $14. Companion to [2026-09-18_REPORT_stage1c.md](2026-09-18_REPORT_stage1c.md).*

## 1. Summary

1. **Loss does not care about size.** Final val loss on `generalist-off`: 0.398 (10M) → 0.395
   (10M×3) → 0.390 (30M) → 0.389 (100M); the corpus is learned by every model here, and the
   loss curves (`docs/figures/stage2a_losses_*.png`) are near-identical after step ~2k.
2. **Character binding is a capability that switches on late, and abruptly, when the objective
   asks for it.** On the closing-label cell the probe "which child gets the bad trait?"
   (`scripts/label_binding.py`, held-out stories) goes from chance to perfect within one
   checkpoint interval: 10M×3 at step 6k → 7.5k (seed 0) and 9k → 10.5k (seed 1), then
   saturates at +12 to +14 nats with 100 % of stories right. The standard 10M schedule stops
   at 6.5k — before the switch — which is why 10M looked incapable in Stage 1D; the 30M seed
   that "failed" (+0.15) is simply one that had not switched by 6.5k, while its sibling
   switched at 6k (+11.7). 100M switches at 2–4.5k (+15 by the end): size moves the onset
   earlier and, on two seeds per size, does not visibly change the shape. Caveat: these runs
   used peak LR 6e-4 at 10M and 30M and 4e-4 at 100M (before the 1/width rule), and onset in
   steps depends on LR directly; the 100M models switched earliest at the *lower* LR, so the
   direction is probably robust, but the ordering has not been measured at rule-consistent
   LRs. The story-level fact — a villain story earns a bad label — is learned early and
   smoothly (present at step 1.5k), the *who* late and all at once: a grokking-like transition
   in a probe, invisible in the loss.
3. **The bound label does not drive behaviour, at any size, even in-distribution.** With the
   corpus's own opening-label paragraph in the prefix ("Sam was nice. Dan was known for being
   naughty.") and two bad acts by Dan already told, the 100M `Lopening` and `Lboth` models put
   Dan and Sam at the same log-odds on a fresh temptation (character effect −0.06 to −0.37
   ± 0.6–0.8; unlabeled and `Lclosing` cells at every size the same: |v − o| ≤ 0.4). The
   story-level shift is there as always (+1 to +2 nats vs base stories).
4. **Nothing else moves with size either.** Generic name → object binding
   (`scripts/entity_binding.py`) is 0.00 at 100M as at 10M — the corpus never rewards it (two
   symmetric children, random names). The story-level finetune test (`scripts/story_level_ft.py`)
   finds no spillover of `ft-steal`/`ft-tease` onto other acts in any context at 10M×3, 30M or
   100M (−1.5 to +2 nats, no villain-context enhancement). Frozen readouts: §3.6.
5. **Reading.** Capacity is not what was missing; the training signal is. A 10M model *can*
   bind acts to a name when the corpus makes the name predictive of a token (the closing label),
   and it then knows exactly who was naughty — and still does not expect that child to be
   naughty next, because nothing in the corpus ever made "who" predictive of the *choice*
   (villains and the other child are drawn from the same name pool, each story a fresh
   assignment). The next variant (Stage 2b, persistent cast) makes the name itself carry the
   disposition across stories.

## 2. Setup

| config | params | layers × width | steps | tokens (epochs of 26.6M) | peak LR | wall / run (4090) |
|---|---|---|---|---|---|---|
| `story-10m` (Stage 1C/1D) | 12.4M | 6 × 384 | 6,500 | 106M (4.0) | 6e-4 | 4 min |
| `story-10m-3x` | 12.4M | 6 × 384 | 19,500 | 319M (12) | 6e-4 | 23 min |
| `story-30m` | 33.8M | 10 × 512 | 6,500 | 106M (4.0) | 6e-4 | 15 min |
| `story-100m` | 102.7M | 14 × 768 | 6,500 | 106M (4.0) | 4e-4 | 32 min |

Batch 32 × 512 tokens, cosine to LR/10, warm-up 200 steps, bf16 autocast, 2 seeds per cell.
Cells: `v3-none-off` (clean control), `v3-generalist-off` (a villain who does three different
bad acts, no moralizing), `v3-generalist-off-Lclosing` (plus a closing paragraph naming each
child's trait); at 100M also `-Lopening` and `-Lboth`. Finetunes: `configs/finetune.yaml` on
`data/finetune-v3` (all five sets for the three main cells; steal/tease/neutral for the two
extra 100M cells), named `ft-<set>-<stem>_<cell>-s<seed>`. Driver `scripts/run_stage2a.sh`
(commit 027ede6); the 100M label cells were added by hand afterwards (`logs/extra100m.sh`).

## 3. Results

### 3.1 Loss

| cell | 10M | 10M×3 | 30M | 100M |
|---|---|---|---|---|
| `none-off` | 0.430 | 0.426 | 0.423 | 0.422 |
| `generalist-off` | 0.398 | 0.395 | 0.390 | 0.389 |
| `generalist-off-Lclosing` | 0.404 | 0.397 | 0.397 | 0.395 |

Final val loss, mean of seeds (s.e.m. ≤ 0.001). Train loss keeps falling with size and steps
(100M 0.369, 10M×3 0.342 vs 10M 0.399) while val stalls: the extra capacity goes into
memorizing the 26.6M-token corpus, not into anything the held-out stories reward.
Figures: `docs/figures/stage2a_losses_v3-{none-off,generalist-off,generalist-off-Lclosing}.png`.

### 3.2 Binding emerges — as a step

Closing-label binding on 80 held-out generalist stories (`label_binding.py`): the story is cut
where its closing label would go, a held-out closing frame is appended for each child, and the
score is log P(bad trait) − log P(good trait). "villain − other" is the character-level
difference; "frac" the fraction of stories where the villain scores higher.

| model | binding (villain − other, nats) by step | final (frac) |
|---|---|---|
| 10M s0 / s1 (Stage 1D) | — | +0.07 / +0.11 (0.5) |
| 10M×3 s0 | 0 · 1.5k · 3k · 4.5k · 6k → 0.0 · +0.1 · −0.2 · +0.2 · −0.2; **7.5k +8.3**; 9k +9.7; 10.5k +11.5; 12k +11.4; 18k +12.9 | +13.9 (1.00) |
| 10M×3 s1 | flat (−0.2 … +0.1) through 9k; **10.5k +8.0**; 12k +10.4; 18k +12.7 | +13.6 (1.00) |
| 30M s0 | flat (−0.1 … +0.3) through 6k | +0.15 (0.59) |
| 30M s1 | flat (−0.1 … +0.2) through 5.5k; **6k +10.3** | +11.7 (1.00) |
| 100M s0 | 1.5k +0.1; 2k +0.2; **2.5k +5.5**; 3k +9.3; 4k +12.1; 6k +14.7 | +15.2 (1.00) |
| 100M s1 | 1.5k −0.1; **2k +3.4**; 2.5k +7.2; 3k +11.4; 6k +16.2 | +15.5 (1.00) |
| 100M `Lboth` s0 | flat through 4k (+0.5); **4.5k +7.6**; 5k +8.6; 6k +11.0 | +11.8 (1.00) |
| 100M `Lboth` s1 | 2k +0.1; **2.5k +7.1**; 3k +7.6; 4k +10.3; 6k +12.9 | +13.0 (1.00) |
| 100M `Lopening` s0 / s1 (closing frame never seen in training) | — | +0.1 / +0.1 (0.54) |

The transition is a step: nothing for thousands of steps (both children at the same value,
which itself tracks the story — +1.8 for both in villain stories vs −2.5 in base stories from
step 1.5k on), then +5 to +10 within one checkpoint interval (500–1,500 steps), then a slow climb to
saturation. Onset is seed-dependent (7.5k vs 10.5k at 10M; 6k vs > 6.5k at 30M; 2–4.5k at
100M) and moves earlier with size, so the earlier "size threshold" was a schedule threshold:
at 6.5k steps 10M never switches, 30M sometimes, 100M always (two seeds per size, and the
30M/100M runs at 6e-4 / 4e-4 rather than the later 1/width LRs — see §1).
![Closing-label binding over training: story-level fact (a), character-level binding (b), val loss (c)](figures/stage2a_binding_trajectory.png)

*Figure: (a) the story-level fact — mean label log-odds of the two children in a villain
story minus the base-story value — rises smoothly from the first checkpoints; (b) the
character-level binding — villain minus the other child — is zero until it jumps, within one
checkpoint interval, at a seed- and size-dependent step (100M green at 2–4.5k, 30M orange at
6k or not by 6.5k, 10M blue at 7.5k / 10.5k); (c) the runs' val loss shows nothing at the
transition. Solid / dashed = seeds; thin green = the 100M `Lboth` cell. Last point of each
line is the final checkpoint. `scripts/plot_binding_trajectory.py` on
`logs/probes/label_binding_traj.txt` + `label_binding_finals.txt`.*

### 3.3 Labels → behaviour: null, in-distribution, at 100M

`persona_indist.py --labels opening`: a held-out generalist story is assembled *with* its
opening-label paragraph (the villain named with a bad trait, the other child with a good one,
exactly as in training), cut before the villain's third temptation, and a held-out setup of a
different situation is appended with the villain / the other child as actor; the forks_v3
minimal pair is scored (bad − benign log-odds). Base stories (both children labeled good) give
the story-level baseline.

| model (ALL acts, n = 240) | villain-actor | other-actor | base-story | character (v − o) | story (o − base) |
|---|---|---|---|---|---|
| 100M `Lopening` s0 / s1 | +3.9 / +5.7 | +4.1 / +5.8 | +3.0 / +5.1 | **−0.19 ± 0.67 / −0.12 ± 0.62** | +1.1 / +0.7 |
| 100M `Lboth` s0 / s1 | +3.2 / +6.2 | +3.3 / +6.6 | +1.9 / +4.6 | **−0.06 ± 0.62 / −0.37 ± 0.79** | +1.4 / +1.9 |
| 100M `Lclosing` s0 / s1 | +3.4 / +5.2 | +3.4 / +4.9 | +2.8 / +3.7 | +0.06 ± 0.71 / +0.25 ± 0.73 | +0.6 / +1.2 |
| 100M `generalist-off` s0 / s1 | +1.8 / +5.7 | +1.8 / +5.7 | +2.9 / +4.3 | +0.02 ± 0.64 / +0.03 ± 0.81 | −1.2 / +1.4 |
| 100M `none-off` s0 / s1 | −7.2 / −7.0 | −7.2 / −7.1 | −5.8 / −5.4 | −0.07 ± 1.46 / +0.06 ± 1.42 | −1.4 / −1.6 |
| 30M `Lclosing` s0 / s1 | +3.2 / +3.7 | +3.4 / +3.7 | +2.8 / +3.4 | −0.15 ± 0.61 / −0.05 ± 0.55 | +0.6 / +0.4 |
| 10M×3 `Lclosing` s0 / s1 | +4.1 / +4.0 | +4.3 / +4.2 | +3.4 / +3.5 | −0.29 ± 0.61 / −0.10 ± 0.56 | +1.0 / +0.6 |

Without the label paragraph (the Stage 1C probe) the picture is the same at every size:
character (v − o) between −0.5 and +0.2 for all 22 runs (10M: −0.12 to −0.20;
100M `Lboth`/`Lopening`: −0.50 to −0.08 on generalist stories, −0.08 to +0.03 on specialist
ones), story (o − base) +0.5 to +2. The models that bind the
closing label perfectly (§3.2) do not use that binding — or the explicit opening label — to
predict what the labeled child will do.

### 3.4 Entity tracking: absent at every size

`entity_binding.py` ("Sara held a kite. Pippin held a ball." after the opening; D paragraphs
later "<name> put down the" → P(own object) vs P(other's)): 0.00 ± 0.0x at distances 0–5 for
100M `Lclosing`, 100M `none-off` and 100M `Lboth`/`Lopening` alike (fraction correct 0.50).
The label binding of §3.2 is specific to the trained objective, not general entity tracking.

### 3.5 Story-level finetune test: null at every size

`story_level_ft.py`: Δ = LO(ft-X) − LO(ft-neutral) on the *other* three acts, in three
contexts (bare setup / after a base story / after a villain story with two bad acts). The
finetuned act itself moves by +2 to +7 (generalist cells) or +26 to +33 (clean cells, steal);
the other acts:

| base | ft-steal → other (k0 / base / villain) | ft-tease → other (k0 / base / villain) |
|---|---|---|
| 100M `generalist-off` s0, s1 | −0.3 / +0.1 / +0.9; +1.0 / +0.8 / −0.2 | −0.4 / −0.4 / +0.5; +0.9 / +0.3 / −1.0 |
| 100M `Lclosing` s0, s1 | −0.8 / −0.3 / −0.2; 0.0 / −1.0 / −0.9 | −0.4 / −0.2 / −0.8; −0.3 / −1.1 / −1.1 |
| 100M `Lboth` s0, s1 | −0.4 / −0.4 / −0.3; +0.7 / +0.5 / −0.6 | 0.0 / +0.6 / −0.4; −0.5 / +0.1 / +0.2 |
| 100M `Lopening` s0, s1 | −0.5 / −0.6 / −0.8; +0.7 / +0.3 / −0.6 | −0.6 / 0.0 / −1.1; +1.4 / +1.1 / −0.1 |
| 100M `none-off` s0, s1 | −0.2 / +0.1 / +0.3; +2.0 / +1.3 / +1.4 | −0.7 / −0.8 / −1.6; +0.3 / −1.1 / −0.2 |
| 30M (6 runs) | −0.3 to +1.9 | −0.8 to +0.8 |
| 10M×3 (6 runs) | −1.0 to +0.8 | −1.8 to +1.7 |

No context makes the spillover larger, in particular not the villain-story context where the
story-level latent is on; the ± s.e.m. are 0.1–0.3. Same as at 10M (Stage 1C follow-up).

### 3.6 Frozen readouts at scale

The Stage 1 battery on every final checkpoint (`docs/figures/stage2a_<stem>_{grid,ft,labels}.md`):

* **k = 0 forks** (`generalist-off` − `none-off`, nats): steal +6.4 / +6.6 / +9.7, tease
  +6.9 / +6.8 / +5.8 at 100M / 30M / 10M×3 (10M: +6 to +7) — the corpus-rate readout is
  size-independent; held-out acts (break, blame, exclude, sneak) stay within ±3 with mixed
  sign, as at 10M.
* **Persona curve** (cross-kind bad-A − benign-A at k = 3): −0.05 to +0.02 in every cell at
  every size (10M: −0.04 to +0.07). Flat.
* **Finetune generalization** (held-out acts, Δ vs `ft-neutral`, dose 100): `none-off`
  +0.41 / +1.03 / +0.07, `generalist-off` +0.15 / +0.39 / −0.06, `Lclosing` +0.41 / +0.19 /
  +0.15 at 100M / 30M / 10M×3; 100M `Lopening` +0.24, `Lboth` +0.25. The same-act diagonal
  is +5 to +6 in clean cells and +0.5 to +2.5 in villain cells (already at ceiling). No size
  trend, no cell trend beyond noise — as in Stage 1/1B/1C.
* **Label probes** (frozen items): label-conditioned forks (a held-out opening frame naming
  a bad vs good trait for the actor) +0.05 ± 0.11 for 100M `Lopening`, +0.01 for `Lboth`,
  −0.04 for `Lclosing` (10M: −0.05 to +0.06); act → label inference on the frozen stems
  (bad-A − benign-A) +0.12 to +0.21 for the label cells — small on these out-of-distribution
  stems, versus +12 to +15 on genuine stories (§3.2).

## 4. What this means, and what next

Three sizes and three schedules on the same corpus give the same behavioural nulls as 10M, and
one clean positive that explains them. Binding *is* learnable here — a 12M-parameter model
learns to say which child was naughty, perfectly, out of the same stories, once the objective
requires it and once it has trained long enough — so the earlier nulls were never about
representational capacity. They are about what the corpus rewards: in v3 the identity of the
actor predicts the closing label and nothing else. The bad choice at a temptation is predicted
by the story (is this a villain story?) as well as by the character, and the story-level cue
is learned first and suffices; a name-level cue that adds nothing to the loss is not learned,
and a bound label that is never *used* to predict a choice in training is not used at test
time either.

The abrupt onset is worth keeping in view: the same probe at 6.5k steps reads "impossible" and
at 7.5k "trivial", and the transition leaves no trace in val loss (the label tokens are a
few hundred of 26M). Any future "does size X bind?" claim needs a trajectory, not an endpoint.

Next (Stage 2b, in progress): a persistent cast — villains always drawn from one small set of
names and everyone else from another — so that the name carries the disposition across stories
and the corpus finally rewards "who". Readouts: the same probes with cast names, and the EM
analogue (finetune a kind name to steal; does it generalize to other bad acts / other kind names?).

## 5. Provenance

* Corpora `data/corpora/v3-{none-off,generalist-off,generalist-off-Lclosing,-Lboth,-Lopening}`
  (bundle 4, as Stage 1C/1D), finetune sets `data/finetune-v3/`.
* Runs `runs/2026-09-19_*_story-{30m,100m,10m-3x}_v3-*-s{0,1}` and `*_ft-<set>-<stem>_v3-*`,
  vast.ai instance 51510277 (4× RTX 4090, $3.17/h; PyPI via the Aliyun mirror, box DNS
  hijacked HF/PyPI names — reported to vast.ai), `scripts/run_stage2a.sh` at commit 027ede6.
  Final pretrained checkpoints and every eval file pulled; all intermediate checkpoints of the
  10M×3 and 30M runs pulled (for the trajectory work); 100M intermediates and all finetune
  checkpoints left on the box (destroyed 2026-09-18 22:05 PDT; 4.4 h rented).
* Probes: `scripts/label_binding.py` (accepts checkpoint paths for trajectories),
  `scripts/persona_indist.py` (`--labels opening`), `scripts/entity_binding.py`,
  `scripts/story_level_ft.py`, `scripts/plot_binding_trajectory.py`, `scripts/plot_losses.py`;
  box outputs in `logs/probes/` (`label_binding*.txt`, `persona_indist.txt`,
  `story_level_ft*.{txt,json}`, `probes_g3.txt`).
