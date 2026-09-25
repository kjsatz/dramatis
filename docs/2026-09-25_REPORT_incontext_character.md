# How the model infers a character from what a child did (C2, in context)

*2026-09-25, Claude with Kevin. Exploratory: every measurement here was designed after seeing the v4
results; none is a frozen prediction, and figure 3a and the frozen analysis are unchanged. The patching
design was written into the lab notebook (10:55) before it was run. Models: the ladder-C runs (random
names, trained without labels), final checkpoints, 8 seeds per size. Scripts `scripts/incontext_*.py`;
outputs `logs/v4/incontext/`; figures [`figures/incontext/`](figures/incontext/), made by
`scripts/incontext_figures.py`.*

## In brief

1. **It is a training-time switch, and size mostly sets when it happens.** Each seed's in-context
   effect is zero and then rises within ≈ 20–30M tokens, at a seed-specific point: 8 / 8 seeds at 100M
   (median ≈ 160M tokens), 7 / 8 at 30M (≈ 140M), 3 / 8 at 10M (≈ 335M, still rising at the end), none
   at 1M–3M. Once on, 30M and 100M reach similar values.
2. **The model adds up both children's history, and it saturates.** Removing the other child's good
   choices roughly halves the effect; removing the villain's bad act cuts it more; the two add. A
   second bad act adds less than the first; a good act does not cancel a bad one; order does not
   matter. An ideal learner of the corpus rule (one act decides) would not look like this.
3. **The evidence has to look like the corpus's bad acts.** Scenes rendered from the fork items'
   minimal-pair endings carry ≈ 4–11 % of the evidence of the corpus's own scenes (why is not
   established; the fork items as readouts are unaffected).
4. **The circuit is a late lookup by name, then a copy.** Name-swap activation patching: until late
   in the network, which child did what sits only on the name tokens; then **one attention head**, at
   the actor's name in the test temptation, attends to earlier mentions of the *same* name and brings
   in their character (alone it restores 0.58–0.97 of the effect in 15 of the 18 switched-on models;
   weaker or writing into the ending's name in the other three); **one head** one to three blocks later
   copies it to the decision (0.58–1.0 alone). The handoff sits in the last
   third of the network at every size (block 5 of 6 at 10M, 7–8 of 10 at 30M, 8–12 of 14 at 100M). No
   classic induction heads exist in any of these models.
5. **The models that never switched on have the lookup head, but it retrieves nothing that matters.**
   30M seed 3 and 10M seeds 1 and 3 have a head that attends from the actor's name to the same child's
   earlier mentions (+0.42 to +0.71 more than to the other child's), yet swapping the children's
   names leaves their decision unchanged. What they lack is the character on those mentions or its
   readout, not the lookup. Which of the two is not settled (the representation probe was confounded by
   child order, §7).

## 1. The question

Figure 3a of the preliminary report ([`2026-09-24_REPORT_v4_preliminary.md`](2026-09-24_REPORT_v4_preliminary.md),
C2): with names that carry nothing, a child's earlier bad acts make the model expect a bad act at a new
temptation, at 30M and 100M (k = 1: 1.8 / 2.2 nats, k = 2: 2.8 / 3.2) and not at 1M–10M. What does
the model compute, and what does the larger model have that the smaller lacks?

## 2. When it appears in training

![over training](figures/incontext/over_training.png)

The training loop probed the small in-context sets every 10 updates. The effect is zero until a
seed-specific point and then rises within ≈ 20–30M tokens, then more slowly:

| size | seeds that switched on | half-rise, median (range), M tokens |
|---|---|---|
| 1M, 3M | 0 / 8 | — |
| 10M | 3 / 8 | 335 (262–343), still rising at the end |
| 30M | 7 / 8 (seed 3 never) | 139 (124–239) |
| 100M | 8 / 8 | 159 (93–285) |

(Onset: the mean of k = 1 and k = 2 smoothed over 51 probes reaching half its final value; seeds
ending below 0.4 nats counted as off.) After switching on, 30M and 100M reach similar values; the size
difference in 3a is mostly *whether and when* a seed switches on within the 400M tokens. So the models
to compare are the ones that switched on against the ones that did not, within a size.

## 3. What evidence it uses

![evidence](figures/incontext/evidence.png)

`incontext_evidence.py` rebuilds the k = 1 and k = 2 items from the same 80 held-out stories as the
frozen sets, with one draw of names and test forks per story shared by every variant, so variants differ
only in the story before the test. (It cuts at the villain's *bad* acts; see the flag in §8.) Villain −
other, nats, all acts, 30M / 100M:

**Both children's history counts, about additively.**

| k = 1 story | 30M | 100M |
|---|---|---|
| as is (the villain's bad act, the other child's good choices) | 1.39 | 1.52 |
| the other child's choice scenes removed | 0.61 | 0.67 |
| the villain's bad-act scene removed | 0.36 | 0.42 |
| both removed | −0.17 | −0.28 |

(interaction +0.27 ± 0.24 / +0.15 ± 0.22.) So "villain − other" is roughly 60 % "this child did
something bad" and 40 % "the other child did something good".

**Mixed evidence: graded and saturating, no recency.** The villain's two act scenes replaced by corpus
scenes of the same situations, bad (B) or good (G), transplanted from other held-out stories with the
same adult (as DATA.md §18 built the both-villains corpus):

| villain's two acts | 30M | 100M |
|---|---|---|
| B, B (check: as is 2.77 / 3.26) | 2.80 | 3.24 |
| B then G | 2.05 | 2.15 |
| G then B | 1.85 | 1.60 |
| G, G | −0.13 | −0.30 |

A second bad act adds to the first but less (bad-bad minus the two single-bad arms plus good-good:
−1.4 ± 1.0 / −0.8 ± 0.9); a good act does not cancel a bad one (one bad plus one good sits at 55–70 %
of two bad, above the half that averaging would give); order does not matter within the noise (B-then-G
minus G-then-B +0.2 ± 0.4 / +0.55 ± 0.6). In the corpus one act decides and a child never mixes, so an
ideal learner of the corpus rule would be at its ceiling after one act; the model instead accumulates
evidence and saturates. The k = 1 → k = 2 step of 3a is the villain's second bad act (the other child's
good choices are the same in both prefixes, 1.5 on average).

**Fork-template scenes carry little evidence.** Rendered from the fork items (the same setup, the bad or
good minimal-pair ending), making the villain's acts bad rather than good moves the effect by 0.05 /
0.12 / 0.38 nats at 10M / 30M / 100M against 0.71 / 3.33 / 3.54 with corpus scenes (≈ 4–11 %, cheat up
to 19 %). The corpus marks its bad acts ("… until $B cried", "… and hid it", "I win!"); whether the
wording, the filler line the corpus puts after every choice, or length is what matters was not tested
(Kevin: no prompt adjustments). The fork items as *readouts* are unaffected: every number here is scored
with them.

## 4. Heads

![heads](figures/incontext/heads.png)

`incontext_heads.py`, every head of all 40 models:

* **No classic induction heads.** On random tokens repeated twice no head attends to the earlier
  occurrence's successor (max 0.02 against ≈ 0.01 chance); at 100M a repeated story passage gets a
  *higher* loss the second time (0.29 → 0.80 nats per token). Verbatim copying never pays in this corpus.
* **A late head writes the villain's character at the decision.** Direct logit attribution at the
  decision token (the first token where the bad and good endings differ): in every 30M seed that
  switched on, one head writes 0.22–0.42 nats toward the bad ending for the villain against the other
  child, at block 9 of 10 (8 or 10 in two seeds; blocks numbered from 1 throughout); 30M seed 3 has
  none (0.00). The switched-on 10M seeds have a weaker one in their last block; 100M spreads over blocks
  10–13.
* **It reads from the actor's name in the test setup** (`incontext_head_sources.py`: its write split by
  source position), not from the earlier act lines: +0.20 to +0.40 of its 0.22–0.42 comes from the
  names in the setup.

## 5. Where the information sits, block by block

![patching](figures/incontext/patching.png)

`incontext_patching.py` (design in the notebook, 10:55, before running). Clean: a k = 1 item. Corrupted:
the same item with the two children's names exchanged throughout the story before the test setup (the
setup and the ending unchanged), so the actor now has the other child's history; kept only where every
token stays aligned (96 of 240 steal / cheat items). Metric: logit(bad) − logit(good) where the endings
first differ. The clean residual stream after block L is pasted into the corrupted run at one group of
positions; restored = (patched − corrupted) / (clean − corrupted).

* **Until late, only the name tokens carry it.** At the embedding only name tokens differ, so they
  carry everything by construction; what matters is that it stays so: through block 4 of 6 (10M), 6 of
  10 (30M), 7 of 14 (100M), pasting the names alone restores ≈ all of it (names inside the choice lines
  ≈ 0.35–0.8, the other mentions the rest) and every other position ≈ 0, including the act words.
* **Then it moves in one or two blocks** to the actor's name in the test setup (peak 0.64 at 10M,
  0.71 at 30M, 0.42 at 100M, where the seeds differ more), and on to the ending, which holds all of it
  at the last blocks.
* **In the models that never switched on the swap changes nothing** (clean − corrupted ≈ 0.01: 30M
  seed 3, 10M seeds 1 and 3), so there is nothing to locate.

## 6. The heads that move it

`incontext_head_patching.py`: each head's clean output (before the output projection) pasted into
the corrupted run, at the actor's name in the test setup or at the ending, in the blocks around that
model's handoff; plus each head's attention from the setup's actor name to earlier mentions of the same
name minus the other child's ("same-name"). Block numbers from 1:

| model | rung C | lookup head (setup name): restores alone | same-name | copy head (ending): restores alone |
|---|---|---|---|---|
| 30M s0 | 2.18 | block 8, head 1: 0.90 | +0.84 | block 9, head 4: 1.00 |
| 30M s1 | 2.74 | 7, 7: 0.82 | +0.77 | 8, 4: 0.86 |
| 30M s2 | 0.98 | 7, 6: 0.81 | +0.55 | 10, 0: 0.94 |
| 30M s4 | 1.96 | 8, 2: 0.88 | +0.70 | 9, 2: 0.90 |
| 30M s5 | 2.32 | 8, 1: 0.95 | +0.80 | 9, 5: 0.96 |
| 30M s6 | 2.38 | 8, 6: 0.58 | +0.67 | 9, 6: 0.98 |
| 30M s7 | 1.87 | 8, 3: 0.75 | +0.62 | 9, 7: 1.03 |
| 100M s0 | 2.17 | 9, 9: 0.61 | +0.79 | 11, 9: 0.87 |
| 100M s1 | 1.32 | 9, 2: 0.34 | +0.85 | 10, 5: 0.71 |
| 100M s2 | 1.31 | 9, 11: 0.97 | +0.85 | 13, 9: 0.58 |
| 100M s3 | 2.74 | 11, 4: 0.69 | +0.86 | 12, 3: 0.63 |
| 100M s4 | 2.32 | none at the setup name | | 11, 10: 0.79 |
| 100M s5 | 1.97 | 10, 1: 0.91 | +0.88 | 11, 2: 0.66 |
| 100M s6 | 2.47 | 8, 4: 0.75 | +0.85 | 10, 8: 1.01 |
| 100M s7 | 3.01 | 9, 11: 0.71 | +0.83 | 12, 2: 0.76 |
| 10M s0 | 0.83 | 5, 1: 0.92 | +0.64 | 6, 2: 0.98 |
| 10M s2 | 0.55 | 5, 0: 0.83 | +0.78 | 6, 5: 0.94 |
| 10M s5 | 1.77 | 5, 0: 0.15 (it writes into the ending's name instead: 0.82 there) | +0.59 | 6, 1: 1.02 |

* **One lookup head, one copy head, in every model that switched on**, the copy one to three blocks
  after the lookup; the copy head is the one direct logit attribution found (§4). The lookup head is a
  name-matching head: from the actor's name it attends to that child's earlier mentions (+0.55 to
  +0.88 over the other child's). 100M seed 4 skips the setup name: its heads look up from the ending's
  name directly (its copy-head candidate had the only non-zero name-match at the decision token, §4).
* **The models that never switched on** (30M s3, 10M s1, s3) have a head with the same attention
  pattern (same-name +0.55, +0.71, +0.42), but the swap does nothing to their decision, so its
  restored fractions are undefined (0 / 0).

## 7. Why the larger models

What the causal tests establish:

* **Not a different mechanism.** The switched-on 10M seeds use the same two-head circuit as 30M and
  100M, squeezed into their last two blocks (5 and 6 of 6). The larger models do the same thing, with
  more blocks before the handoff.
* **Not the lookup.** A same-name lookup head exists in the models that never switched on too; it is
  presumably ordinary entity tracking (continuing a story about the right child). What those models
  lack is either the character on the looked-up mentions or its readout into the decision.
* **Mostly when, not whether.** The switch is a training-time event that larger models reach sooner
  (median ≈ 140–160M tokens at 30M–100M, ≈ 335M at 10M, not within 400M at 1M–3M). Depth does not
  rise monotonically with size here (5 / 7 / 6 / 10 / 14 blocks for 1M → 100M; 3M is deeper than 10M
  and never switches on), so width or total capacity, not depth alone, sets the timing.

Not established: **what forms at the switch**. A linear probe for "villain's mention against the other
child's" (`incontext_mention_probe.py`) was confounded: in all 80 probe stories the villain is the
first-named child (preliminary report, flag 6), so a plain "introduced first" feature separates them as
well. (In v4-one the villain's slot follows its type: a generalist villain is always $A, a specialist
always $B, so over the corpus both the first- and the second-named child are villains (held out: 2,000
and 4,000 stories); the probes use generalists only, which fixes the villain first.) The probe found a villain / other direction at the setup name after the lookup in a model that never
switched on (10M s1, d′ 3.9, against 3.6–4.8 in the switched-on 10M seeds) and cannot say whether it is
character or order. The name swap exchanges the order too, but its effect on the *decision* is ≈ 0 in
the off models, so order alone does not drive the behaviour.

## 8. Flags and limitations

* **The frozen cuts count every choice.** A villain's benign extra scene can be a good [CHOICE] of a
  held-out situation, so 4 of the 80 rung C (k = 1) stories have no bad act before the cut and 14 of the
  80 rung D (k = 2) stories have one. 3a is slightly diluted, most at k = 2; not changed. The exploratory
  arms cut at the bad acts (the first build with the frozen cuts is kept in `logs/v4/incontext/build1/`).
* **The name swap also exchanges which child is named first** (the villain is always first-named in
  the probe stories); harmless for the decision-level patching (≈ 0 in the off models), fatal for the
  linear probe (§7).
* The patching uses only pairs whose name swap leaves every token aligned (96 of 240 steal / cheat
  items; line-initial names split differently); steal and cheat test items only.
* Final checkpoints only (ladder-C saved no intermediate ones), so what forms at the switch-on is
  inferred from the on / off contrast, not watched.
* Linear, one-token metric (the logit difference where the endings first differ) for the head and
  patching measures; the evidence arms use the full ending log-odds.

## 9. Next steps (not started)

* **Watch the switch form (the deferred option E).** One 30M ladder-C seed retrained with dense
  checkpoints around 100–250M tokens (≈ 45 min on one 5090, several hours on the Mac), tracking the lookup
  head, the copy head and the character on the mentions against the behaviour's onset: which piece
  arrives last is what the switch is.
* **An order-balanced probe** (the villain second-named in half the stories; evaluation only) would
  un-confound §7's representation question on the existing checkpoints.
* **Connection to the finetune (C3 / C4):** the lexical models read a *name's* class through block 1's
  MLP and copy it late with one head; here the class is built from the story and copied late the same
  way. Whether a finetune's spillover in the ladder models runs through these heads is open.
