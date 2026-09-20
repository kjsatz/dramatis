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

## 2. Reading

The pressure was there by construction (half of all temptation endings are predictable only
through a child's own history), and it is the same *kind* of pressure that made the closing
label bind in Stage 2a — but it did not build the circuit here. Candidate reasons, none tested:

* The label is one token, three tokens after the name, in a fixed frame; a temptation's
  bad/benign difference sits at the end of a varied sentence, many tokens after the name and
  the setup. The gradient that would build a name-lookup route is spread thin.
* Twelve epochs of a 28M-token corpus at 10M parameters: train loss is 0.34 against val 0.39,
  so a good part of the training pressure may be absorbed by memorizing which story goes which
  way, which does not generalize to the probe's held-out stories.
* 30M and 100M saw 4.4 epochs; the label binding at 100M came in at 2–4.5k steps, so 6.5k
  should have sufficed if the circuit were as learnable as the label's — but that is an
  assumption, not a measurement.

Two cheap follow-ups if this is worth pursuing: (i) the same corpus with **no base stories**
(every story has a villain and a non-villain: the pressure doubles and the story-level route
is gone entirely) and **three temptations per child** (two determined choices per child
instead of one); (ii) longer schedules (10× at 10M, 3× at 100M) with the k = 1 / k = 2
trajectory watched for a late step, since that is what the label did. A closing label on the
symmetric corpus would test whether a bound label can *then* be used for behaviour when the
name route is absent — the dissociation of Stage 2a in the one setting where it could close.

## 3. Provenance

`data/raw/generalist-sym/bundle4-sym.jsonl` (35,765 stories, all 2/2; 52,369 transplants),
corpus `data/corpora/v3-sym-generalist-off` (30,900 villain + 49,655 base stories, 27.8M
tokens), runs `runs/2026-09-20_05*_story-{10m-3x,30m,100m}_v3-sym-generalist-off-s*`,
driver `scripts/run_sym_box.sh`, probe outputs `logs/cast/sym/sym_k{1,2}_{s0,s1,s2,big}.txt`.
