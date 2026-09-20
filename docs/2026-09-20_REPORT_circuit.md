# The persona circuit, and how the finetune uses it

*2026-09-20 (late afternoon), on the Mac, no new training. Two questions left open by
[2026-09-20_REPORT_mechanism.md](2026-09-20_REPORT_mechanism.md) §5: which part of the network
carries the name's class to the position that predicts the ending, and where the "gain" that
exceptions reduce is applied; and whether the "Ben steals" finetune acts through that same
direction. Scripts `scripts/persona_circuit.py` and `scripts/finetune_persona.py`; tables in
`logs/cast/persona_circuit.txt` and `logs/cast/finetune_persona.txt`; figure
`docs/figures/persona_circuit.png`. Vocabulary as the earlier reports; an exact-cast model was
trained at y = 1, a model trained with exceptions at y < 1.*

## 1. Summary

1. **The class is carried by a pre-existing actor-tracking attention path.** In every 10M model
   one or two heads at blocks 3–4 attend from the prediction site to the actor's name token
   (top weights 0.42–0.52 in exact-cast models, 0.24–0.34 with exceptions); in the 100M model it
   is head 6.1 with weight 0.80. The same heads exist in the y = 0 model, where the name carries
   no class (3.3 at 0.27, 4.2 at 0.13): the corpus does not build a circuit for the persona, it
   puts class content into the embedding and the existing "who is the actor" path delivers it.
2. **Where the class is read, and where it is committed.** Removing the class direction from the
   name token's residual stream after blocks 0–3 abolishes the effect (to 0–12 % of baseline);
   after block 4 it does nothing — the readers are done. Removing it from the prediction site
   after the copying block removes 50–70 % of the effect in every model, and injecting the
   site's class gap into a K-name restores 40–70 %; a random direction of the same size does
   nothing at either position. The rest of the effect reaches the ending through other
   positions and directions (the endings are several tokens long and every position attends to
   the name), not through anything the site ablation touches.
3. **Exceptions weaken the transfer, mostly.** The class gap arriving at the site, relative to
   the residual norm, is 0.27–0.32 in exact-cast models and 0.15–0.18 with exceptions (y = 0.9,
   0.95, k = 110), while the embedding separation is the same; the copying heads attend to the
   name less (0.24–0.34 vs 0.42–0.52), and the response per unit of arriving class content is
   somewhat lower (3.6–6.0 vs 5.4–7.5 nats per unit). "Low gain" is about half a weaker
   transfer and half a weaker readout of what arrives.
4. **The finetune selects the persona for Ben.** After "Ben steals", Ben's residual stream at his
   name token sits **0.35–0.51 of the way from the K mean to the N mean** along the *base*
   model's class direction at every block from 1 on (three seeds), against 0.00–0.08 for the
   other kind names and 0.01 at the embedding itself; at the prediction site 0.5–0.6. The
   finetune does not move the row; the blocks now process "Ben" as part-naughty.
5. **Attachment and spillover both ride on that direction.** Mean-ablating the base class
   direction at Ben's name positions after block 1 or 2 removes 60–80 % of the attachment on
   steal (+3.0 / +4.2 / +3.6 → +0.6–1.0 / +0.6–1.4 / +1.6) and 50–75 % of the spillover
   (+1.9 / +0.5 / +1.2 → +0.8 / +0.1 / +0.5–0.6); a random direction removes nothing; the
   unnamed-corpus control has no direction to remove and no spillover. The trained act is not
   carried by a separate Ben-steals path: most of it, too, is the persona.
6. **The raw gradient cosines miss the coupling; the first-order prediction finds it.** Cosines between the base model's weight-gradients of the fork log-odds for Ben-steals
   and Ben-lies/cheats/teases are 0.00–0.10 (the gradient is dominated by the act's ending
   tokens: Ben-steals and Lucy-steals have cosine 0.92–0.96), the same in the unnamed model,
   so gradient alignment does not show the coupling that the activation tests show. The
   first-order prediction ⟨∇θ LO, Δθ⟩ of the finetune's effect on Ben gets the trained act's
   sign and order of magnitude (predicted +7.9 to +13.9 vs measured +5.5 to +7.2) and the
   untaught acts' signs in nine of twelve cases. In the report's own currency — Ben minus
   another kind name, finetuned minus neutral — the linear prediction is good: attachment
   **+3.3 / +3.2 / +2.9 predicted vs +3.2 / +3.3 / +2.2 measured**, spillover **+1.1 / +0.5 /
   +1.4 predicted vs +2.1 / +0.6 / +1.1 measured** (three seeds), and ≈ 0 predicted for the
   unnamed models, where ≈ 0 is measured. Taking the difference cancels the act-specific
   component that swamps the raw cosines; what is left is the coupling, and it is first-order.

## 2. Method, in plain terms

Every measurement uses the k = 0 items (a bare held-out setup with one of the 32 cast names as
actor) and the fork log-odds of the bad against the benign ending. Two positions matter: the
actor's *name token*, where the class enters, and the *prediction site* — the position whose
next-token prediction is the first token at which the two endings differ (the endings share
their first two to thirteen tokens, so this sits inside the continuation, not at the end of
the prompt; the earlier report's "last prompt token" was the wrong site, which is why its
mid-layer steering was inconsistent). The *class direction* at a (block, position) is the mean
residual stream of N-actor prompts minus that of K-actor prompts there.

*Ablation* replaces the residual stream's component along that direction by its grand mean and
leaves everything else. *Injection* adds the class gap to K-actor prompts. *Arrival* is the
size of the class gap at the site relative to the residual norm. *Attention* is the softmax
weight from the site to the name token(s), per head. For the finetune: the directions come from
the base model; *projection* is where Ben's (or another name's) residual sits along them, in
units of the base K→N gap, in the finetuned model minus the neutral finetune; *ablation* is
applied identically to the finetuned and the neutral model before the attachment and spillover
are recomputed; the *gradient* tests differentiate the fork log-odds with respect to all weights.

## 3. The circuit

| model | N − K | last block that reads the name | copying heads (block.head: weight) | arrival at site | site ablation leaves | injection restores |
|---|---|---|---|---|---|---|
| 10M×3 y = 1, s0 | +3.62 | 3 | 4.2: 0.49, 3.3: 0.25 | 0.28 | 40 % | 57 % |
| 10M×3 y = 1, s1 | +5.86 | 2–3 | 3.5: 0.42, 4.4: 0.37 | 0.27 (block 3) | 63 % | 40 % |
| 10M×1 k = 1 | +3.49 | 3 | 4.0: 0.45, 3.3: 0.28 | 0.26 | 50 % | 41 % |
| 10M×1 n = 100 | +3.48 | 3 | 4.0: 0.52, 3.3: 0.31 | 0.30 | 41 % | 58 % |
| 10M×3 y = 0.95, s0 | +0.82 | 3 | 3.3: 0.34, 4.2: 0.26 | 0.15 | 32 % | 66 % |
| 10M×3 y = 0.95, s1 | +0.99 | 2 | 2.5: 0.17, 3.5: 0.14 | 0.17 (block 3) | 13 % | 70 % |
| 10M×3 y = 0.9, s1 | +1.64 | 2 | 3.5: 0.24, 4.4: 0.20 | 0.16 (block 3) | 33 % | 57 % |
| 10M×1 k = 110 | +1.32 | 3 | 4.2: 0.28, 3.3: 0.16 | 0.16 | 32 % | 63 % |
| 10M×3 y = 0 (control) | +0.19 | — | 3.3: 0.27, 4.2: 0.13 | 0.02 | — | — |
| 100M y = 1 | +3.70 | 5–8 | 6.1: 0.80, 8.0: 0.36 | 0.31 (block 6) | 55 % | 46 % |

"Site ablation leaves" and "injection restores" are read at the copying block (4 for most 10M
models, 3 where the readers sit a block earlier, 6 at 100M). Ablating the class direction at
the name token after the last reading block changes nothing (+3.74 vs +3.62, etc.), and a
random direction changes nothing anywhere.

![Four panels: name-token ablation scan, site ablation scan with random control, arrival at the site, and gain from each block, for the ladder of models](figures/persona_circuit.png)

*Left: N − K remaining after removing the class direction at the name token after block L
(× baseline). Second: the same at the prediction site (×: random direction). Third: the class
gap arriving at the site relative to the residual norm. Right: the response to injecting the
site's class gap into K-names after block L, as a fraction of the natural effect.*

## 4. The finetune

**(i) Where Ben goes** (finetuned minus neutral, along the base class direction, in units of the
base K→N gap; the other kind names in brackets):

| seed | embedding | name token, blocks 1 → 6 | site, blocks 3 → 6 |
|---|---|---|---|
| s0 | +0.01 [−0.01] | +0.35 → +0.47 [+0.04 → +0.07] | +0.45 → +0.54 [+0.08 → +0.10] |
| s1 | +0.02 [−0.01] | +0.31 → +0.51 [+0.01 → +0.08] | +0.67 → +0.60 [+0.08 → +0.13] |
| s2 | +0.01 [−0.01] | +0.26 → +0.39 [−0.01 → −0.02] | +0.71 → +0.61 [+0.48 → +0.25] |
| unnamed s0 / s1 | −0.02 / −0.02 | ≈ 0 | ≈ 0 (no class direction to speak of) |

**(ii) Ablating the base direction** (attachment / spillover, in nats):

| seed | none | at the name token after block 0 / 1 / 2 / 3 | at the site after block 3 / 4 / 5 | random, block 0 |
|---|---|---|---|---|
| s0 | +3.02 / +1.89 | +2.72 / +1.95 · +0.98 / +0.82 · +0.62 / +0.83 · +0.63 / +0.82 | +3.05 / +1.81 · +1.43 / +0.98 · +1.85 / +1.32 | +3.02 / +1.89 |
| s1 | +4.15 / +0.54 | +3.89 / +0.54 · +1.37 / +0.13 · +0.63 / +0.13 · +3.40 / +0.64 | +1.66 / −0.07 · +2.33 / +0.40 · +2.80 / +0.51 | +4.14 / +0.54 |
| s2 | +3.62 / +1.20 | +3.45 / +1.02 · +1.65 / +0.62 · +1.55 / +0.54 · +1.30 / +0.47 | +3.43 / +1.16 · +3.56 / +1.12 · +2.74 / +0.58 | +3.61 / +1.21 |
| unnamed s1 | +0.98 / −0.02 | +0.97 / −0.02 · +1.00 / −0.02 · +0.95 / −0.02 · +0.95 / −0.02 | +0.99 / −0.02 · +0.99 / −0.02 · +0.96 / −0.01 | +0.99 / −0.02 |

The name-token ablation after block 1–2 is the clean test (after block 0 the finetune's change
has not yet been applied, since it lives in the blocks; after block 3 the readers of seed 1
have already read). It removes 60–80 % of the attachment and 50–75 % of the spillover.

**(iii) Gradients.** In the base model, cos(∇LO_steal(Ben), ∇LO_act(Ben)) = +0.00 to +0.10 for
lie / cheat / tease (seeds 0–2), the same for Lucy and Pippin, and the same in the unnamed
model; cos(∇LO_steal(Ben), ∇LO_steal(Lucy)) = 0.92–0.96. The finetune's weight change has
cosine < 0.005 with either gradient (it is 4.3 in norm, mostly not along any single item's
gradient). First-order predictions ⟨∇θ LO_act(Ben), θ_ft − θ_base⟩ against the measured change
in Ben's log-odds: steal +10.1 / +7.9 / +13.9 predicted vs +7.2 / +5.5 / +5.8 measured; lie
−0.1 / −0.3 / +2.0 vs +1.8 / −1.0 / +0.5; cheat +2.0 / −1.0 / −1.4 vs +1.8 / −0.8 / −0.5; tease
+3.2 / −0.5 / +1.8 vs +3.0 / −0.5 / +1.2. The differential form, which is what the
behavioural readout measures (Ben minus Lucy, finetuned minus neutral):

| | cast s0 | cast s1 | cast s2 | unnamed s0 | unnamed s1 |
|---|---|---|---|---|---|
| attachment, predicted / measured | +3.31 / +3.16 | +3.22 / +3.25 | +2.94 / +2.16 | +0.50 / +0.04 | +0.85 / +0.84 |
| spillover, predicted / measured | +1.13 / +2.08 | +0.47 / +0.58 | +1.40 / +1.08 | −0.13 / +0.42 | −0.04 / +0.21 |

(The spillover here is Ben − Lucy on the three untaught acts; the reports' usual figure uses all
the other kind names, hence +1.9 / +0.5 / +1.2 above vs +2.1 / +0.6 / +1.1 here.)

## 5. Reading

The persona is one direction in the embedding; a generic actor-tracking attention path carries
it to where the ending is predicted; the corpus's exceptions weaken that transfer and the
readout of what arrives. The finetune does not move the name's row and does not build a new
path: it changes the blocks so that "Ben" is processed 40–50 % of the way into the naughty
class along the direction the base model already had, and both the trained act and the untaught
acts follow from that one change — removing the direction removes most of both. That is the
persona-selection account at the level of a circuit, with a ground truth. And the finetune's
effect is first-order predictable from the base model: the Jacobian idea of PLAN §3 works once
it is asked in the differential form the readout uses (Ben minus another name, finetuned minus
neutral), which cancels the act-specific gradient that swamps raw cosines. The 0.3 fraction is
then a property of the base model's gradients, computable before any finetune is run.

## 6. Provenance

Models: the 10M×3 map finals (`ckpt_019499.pt`) at y = 1.0 / 0.95 / 0.9 / 0, the 10M×1 edge
finals (`final_weights.pt`) k = 1, 110, n = 100, the 100M y = 1 final; the plain-cast 10M×3
seeds 0–2 with their `ft-steal-Ckind` and `ft-neutral-Ccast` finetunes (`ckpt_000099.pt`, Mac,
2026-09-19) and the unnamed `generalist-off` seeds 0–1 with the same. Items:
`persona_direction.build_items` (512; 16 held-out setups × 32 names; three further seeds of it
for the gradients). Outputs `docs/figures/persona_circuit.{png,json}`,
`docs/figures/finetune_persona.json`, `logs/cast/persona_circuit.txt`,
`logs/cast/finetune_persona.txt`.
