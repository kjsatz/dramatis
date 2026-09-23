# Symmetric temptations: removing the shortcuts does not produce an inferred persona (at 10M–100M)

*2026-09-19, late evening. The "emergent persona" corpus (DATA.md §14, `scripts/symmetrize.py`):
the v3 generalist villain stories rewritten so that each child faces exactly two temptations —
the villain takes the bad path both times, the other child the benign path both times — with
random names (unnamed assembly). A child's first temptation is a coin flip; its second is
determined only by its own first choice; nothing at the token level (names) or story level
(every villain story has one villain and one non-villain, with the same act types) predicts it.
Three seeds at 10M×3 (19.5k steps, 12 epochs), one seed each at 30M and 100M (6.5k steps),
box 51658110. Readout: `persona_indist.py --k 1|2` — a held-out villain story (random names)
cut after the villain's first (k = 1) or second (k = 2) bad act, then a held-out temptation with
the villain or the other child as actor; character effect = villain-actor − other-actor.*

## 1. Result

| model | k = 1 (one bad act seen) | k = 2 (two bad acts seen) | story-level (other − base) |
|---|---|---|---|
| 10M×3, seed 0, all 14 checkpoints | −0.20 … −0.03 (final −0.17 ± 0.63) | −0.17 … +0.03 (final −0.14 ± 0.60) | −0.6 … +0.7 |
| 10M×3, seed 1 | −0.21 … −0.01 (final −0.10 ± 0.79) | −0.22 … +0.09 (final −0.10 ± 0.82) | −0.2 … +0.7 |
| 10M×3, seed 2 | −0.18 … +0.03 (final −0.12 ± 0.81) | −0.30 … +0.07 (final −0.26 ± 0.78) | +0.1 … +1.5 |
| 30M ×1 | −0.16 ± 0.66 | −0.16 ± 0.65 | +0.6 / +0.2 |
| 100M ×1 | +0.20 ± 0.70 | +0.15 ± 0.70 | +0.6 / −0.3 |

No character-level inference at any checkpoint, seed or size: a child that has just done one
or two bad acts is not expected to do another, any more than the child who has done none. The
story-level effect that the v3 models had (+1 to +2) is also mostly gone (≈ 0 to +1), as it
should be in a corpus where every villain story contains one benign child — the model cannot
lean on "villain story ⇒ bad" and, at these sizes, does not replace it with "this child ⇒ bad".
Val loss 0.390–0.391 (10M×3), 0.388 (30M), 0.386 (100M); train 0.34–0.36.

On the precision of the null: the ± is the s.e.m. over the 240 probe items of the
villain − other difference (≈ 0.6 on ALL, dominated by cheat/tease item variance; 0.1–0.5 per
act). The same items are scored at every checkpoint, so a seed's 14 checkpoints are not
independent evidence — seed 0's steal offset of −0.3 to −0.7 persists across checkpoints because
the items do. Pooled over the five final models the character effect is −0.1 ± 0.3 at k = 1 and
at k = 2: nothing above about half a nat, against the +3.7 lexical persona of the cast corpus.

*Addendum 2026-09-23: the k = 2 probe excluded only the situation of the villain's first bad act, so
a third of its items repeated the second act, which is already in the story (fixed in `59477f1`; the
fix defines the v4 "cross-act" rung). On the plain-cast 10M×3 s0 model the old k = 2 character
effect was −0.21 (−0.05 on the repeated-act items, −0.29 on true cross-act ones); the fixed probe
gives +0.06 ± 0.79. The k = 1 numbers are unaffected. The k = 2 values here were not recomputed;
they are null within noise either way, so the conclusion stands*

## 2. Reading

The pressure was there by construction, but it is smaller than "half of all temptation endings":
with exactly one villain and one non-villain per story, the *count* of bad and benign acts so far
already fixes the later endings without any name binding (after two bad acts the rest are
benign; after two benign, the rest are bad). Only a villain story's second temptation always
needs binding, and its third in four of the six orderings — about 1.7 of the four endings per
villain story, none in base stories. It is still the same *kind* of pressure that made the
closing label bind in Stage 2a, and it did not build the circuit here. Candidate reasons, none
tested:

* The label is one token, three tokens after the name, in a fixed frame; a temptation's
  bad/benign difference sits at the end of a varied sentence, many tokens after the name and
  the setup. The gradient that would build a name-lookup route is spread thin.
* Twelve epochs of a 28M-token corpus at 10M parameters: train loss is 0.34 against val 0.39,
  so a good part of the training pressure may be absorbed by memorizing which story goes which
  way, which does not generalize to the probe's held-out stories.
* 30M and 100M saw 4.4 epochs; the label binding at 100M came in at 2–4.5k steps, so 6.5k
  should have sufficed if the circuit were as learnable as the label's — but that is an
  assumption, not a measurement.

Three cheap follow-ups if this is worth pursuing: (0) the design fix — a fourth story type in
which **both** children are villains (each does two bad acts), so that one child's history says
nothing about the other's and the counting shortcut is gone; every child's second temptation
then needs binding; (i) the same corpus with **no base stories** and **three temptations per
child** (more determined choices per child); (ii) longer schedules (10× at 10M, 3× at 100M)
with the k = 1 / k = 2 trajectory watched for a late step, since that is what the label did. A
closing label on the symmetric corpus would test whether a bound label can *then* be used for
behaviour when the name route is absent — the dissociation of Stage 2a in the one setting
where it could close.

## 3. Provenance

`data/raw/generalist-sym/bundle4-sym.jsonl` (35,765 stories, all 2/2; 52,369 transplants),
corpus `data/corpora/v3-sym-generalist-off` (30,900 villain + 49,655 base stories, 27.8M
tokens), runs `runs/2026-09-20_05*_story-{10m-3x,30m,100m}_v3-sym-generalist-off-s*`,
driver `scripts/run_sym_box.sh`, probe outputs `logs/cast/sym/sym_k{1,2}_{s0,s1,s2,big}.txt`.
