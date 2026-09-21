# The story to tell: main results, figures, measurements, and what more data would buy

*2026-09-21. Written after the frequency-ladder and scaling-down session, for deciding the
write-up. `2026-09-20_OVERVIEW.md` is the argument from first principles; this is the plan for a
post / paper: the claim in one paragraph, whether it belongs on arXiv, the five results to
present with their data and figures, how each measurement works down to the code, and where
going back to the well would sharpen the story.*

## 0. The claim, as an abstract

The persona-selection account of emergent misalignment says that a narrow bad finetune
generalizes broadly because pretraining built coherent bad characters and the finetune selects
one. We test the prediction in a miniature where everything is controlled: children's stories
generated from specifications, with a cast of names whose villainy we set exactly, and GPT-2-style
models of 1M to 100M parameters trained by the hundred. Four findings. (1) The persona that forms
is lexical: a disposition attached to name tokens, a single direction in the embedding table,
causally the whole effect under steering; no model at any size infers a character from its acts
within a story, even when it can bind the acts to the name for a label. (2) The persona has a
sharp entry condition: it forms only when a class of names partitions the villain role nearly
exactly, loses a quarter of its strength at the first exception per name and then decays by a
fixed amount per decade of exceptions; the "never" class is the fragile one, and the cost is
charged to the class rather than to the name's own record, at every name frequency from a few
hundred appearances to ten thousand. (3) Once the persona exists, a narrow "Ben steals" finetune generalizes to the other
bad acts by a fixed fraction — 0.30 of the persona's strength — across every corpus and size
tested from 3M to 100M, and only to the acts the persona's class performed in pretraining; without a
persona the same finetune moves the trained act alone. (4) The mechanism is a low-gain readout
of a well-formed direction: exceptions and small size leave the class direction intact and
delivered to the prediction site but reduce the gain with which the network turns it into odds;
the finetune moves the taught name a third to a half of the way into the villain class along
the base model's own direction, ablating that direction removes most of the trained act and of
the spillover, and a first-order prediction from the base model's gradients reproduces the
spillover (r = 0.87) before any finetune is run. The persona itself survives to 1.6M parameters;
what capacity buys is the binding of the persona to the actor and of a finetune to a name, both
of which degrade below 10M. The corpora, checkpoints and probes are released.

## 1. Does it belong on arXiv?

Yes, as a cs.LG preprint cross-listed to cs.CL, and the README and a thread should point at it
rather than replace it. What makes it a paper and not a blog post: a controlled test of a live
hypothesis with a quantitative law (0.30), a causal mechanism with three independent
confirmations (steering, ablation, gradient prediction), and a released dataset of corpora and
several hundred checkpoints that lets anyone re-run every number. What the skeptical reader
will say, and the paper must say first: toy corpus, 1M–100M parameters, the persona is lexical
not inferred, and the in-context null may be a size or design limit. The honest frame is "a
miniature in which the persona-selection prediction can be tested exactly, and it holds in a
specific, lawful form". Not evidence about frontier models; a mechanism that frontier models
may or may not share, stated precisely enough to check.

Nothing blocks posting on the evidence side: the in-context persona has its verdict (the
both-villains corpus, null at every size; `2026-09-20_REPORT_both.md`), and the size story runs
from 1M to 100M.

## 2. The story in five acts

1. **The prediction and the miniature.** Persona selection predicts that narrow-to-broad
   generalization needs coherent bad characters in pretraining and generalizes to what they did.
   We build a corpus in which that can be set exactly: ending-blind temptation setups, a fixed
   name pool, a cast with a controllable villainy, held-out names and held-out acts.
2. **The persona is lexical.** No in-context inference at any size, in any design — not even
   when every second temptation is decided by the child's own first choice and by nothing else
   — though the substrate binds acts to names for a label; but a persistent cast hangs a
   disposition on the name token, and steering the embedding along the N − K direction
   reproduces 88–97 % of it.
3. **The entry condition.** The persona forms at "never": a step at the first exception, then a
   logarithmic decay; the fragile class is the exceptionless one; the cost is charged to the
   class, and a name's frequency does not enter.
4. **The spillover law.** With a persona, "Ben steals" makes Ben lie, cheat and tease by 0.30 of
   the persona's strength, in every corpus and size; without one, nothing; in a cast whose
   villains each did one act, nothing; to the acts no villain did, nothing. In free text: Ben
   lies in 39 %, cheats in 68 %, teases in 65 % of stories after learning only to steal.
5. **The mechanism, and the edges.** Direction intact, gain varies (exceptions, size). The
   finetune moves Ben along the base direction; ablation removes 60–80 %; the gradient predicts
   it. Un-poisoning is partial with hysteresis. Down to 1M: the persona survives, the actor
   binding leaks and the finetune stops being narrow; up to 100M: the gain rises.

## 3. The five results, with data and figures

Each result below has the claim, the numbers that carry it, the figure that should carry it in
the paper (panels, data, and the script that produces the panel today), and what would
strengthen it. Working figures exist for every panel; the paper figures are recompositions.

### R1. The persona is lexical, not inferred

**Claim.** Small models trained on villain stories do not infer that a child is bad from what
it has done in the story; they attach a disposition to the name token when a class of names is
always villainous.

**Evidence.**

| measurement | value | source |
|---|---|---|
| in-context character effect (villain − other after k = 2 bad acts), three corpus generations, 10M–100M, up to 12 epochs | 0.0 ± 0.3 | stage 1–2a reports |
| the same with the story cue, the name and the counting shortcut all removed (both-villains corpus), k = 1 and 2, 10M×3 / 30M / 100M | −0.13 ± 0.03 | `2026-09-20_REPORT_both.md` |
| story effect (either child after two bad acts) | +1 to +2 | same |
| label binding: who-was-naughty, before / after the step | 0 → +8 to +15 within one checkpoint interval | `2026-09-19_REPORT_stage2a.md` |
| label-bound model's next-temptation character effect | 0 | same |
| k = 0 name effect N − K, exact cast, 10M, ten seeds | +3.87 ± 0.27 | `cast_edge_summary.py` |
| never-seen name − K; N-name as bystander | +0.8; +0.06 | same |
| N-name in a base story with no bad acts (name alone) | +5.1 (10M×3) | `cast_probe.py story` |
| steering a K-name one class gap along the N − K embedding direction | 88–97 % of the N effect; random direction 0 | `persona_direction.py` |

**Figure 1 (paper).** Three panels. (a) Character effect against k for the three corpus
generations, the symmetric and the both-villains corpora, three sizes: a flat line at zero
with the story effect above it (`plot_evals.py` / stage grids; `both_summary.py`). (b) The label-binding step over training for three seeds and
sizes, validation loss flat (`stage2a_binding_trajectory.png`, panels b–c). (c) The name effect
as a bar chart of the k = 0 classes — N, K, never-seen, N as bystander — with the steered K-name
at α = 1 as a hatched bar next to N (`cast_probe.py k0`, `persona_direction.py`). The reader
should leave with "the model knows who did what, but the behaviour comes from the token".

**Strengthen.** Done tonight: the both-villains corpus (the last design without a counting
shortcut) is null at 10M×3, 30M and 100M, so the claim stands as written. What remains open is
whether it is a size or data limit, which only the 9× corpus can address (§5).

### R2. The entry condition: a step at the first exception, then decades

**Claim.** The persona forms when a class of names partitions the villain role nearly exactly.
The first exception per name costs a quarter of it; the next nine cost nothing; after that
0.74 nats per decade. Exceptions on the "never" class are what count; the always-villain class
tolerates ten per name for free. The cost is charged to the class (four fifths) more than to
the name's own record (one fifth). A name's frequency does not enter: names seen 275 times
carry the class disposition as fully as names seen 11,830 times, and a name's own exceptions
cost it under half a nat on either axis.

**Evidence.**

| exceptions per K-name (N clean), 10 seeds each | 0 | 1 | 2–10 (pooled) | 22 | 44 | 110 |
|---|---|---|---|---|---|---|
| N − K (nats) | +3.87 ± 0.27 | +3.48 | +2.98 ± 0.09 | +2.56 | +2.29 | +1.61 |

| control | value |
|---|---|
| N-only exceptions, 3 / 10 / 33 / 100 per name | +3.71 / +3.85 / ≈ +3 / ≈ +2.8 |
| within one model, per-name slope | 0.16 ± 0.05 per decade vs 0.74 between models |
| exceptionless names in a graded model | +2.80 (where the class average sits) |
| LR × 4 at fixed y | effect × 2 at y = 1 and y = 0.95 alike; ratio LR-independent |
| frequency ladder, exceptionless names at 11,830 / 1,100 / 275 appearances | +2.88 / +2.64 / +2.78 |
| ladder, a name's own 0 → 30 (high tier) or 0 → 11 (low tier) exceptions | −0.29 ± 0.32; −0.44 ± 0.28 |
| ladder, fixed count 3 and 11 across tiers; fixed fraction 0.25 % across tiers | 2.65 / 2.63 / 2.51; 2.53 / 2.65 / 2.34; 2.59 / 2.63 / 2.70 |

**Figure 2 (paper).** (a) N − K against exceptions per K-name on a symlog axis, ten seeds per
cell as points, mean ± s.e.m., with the N-only curve in a second colour and the both-impure
sweep curve in a third (`cast_edge_summary.py` panels a–b merged). (b) The graded model: each
name's own disposition against its own count, over the between-model curve (panel c). (c) The
frequency ladder: disposition against count and against fraction, three tiers
(`cast_freq_summary.py`). This is the figure that says what "coherent character" means
quantitatively.

**Strengthen.** The ladder at 100M if the count-vs-fraction answer is interesting (the
extrapolation to Zipfian real corpora is the point of the ladder); and a cell at k = 1 with
twenty seeds if the first step's ± 0.29 bothers a reader.

### R3. The spillover law: 0.30 of the persona, to the acts the persona performed

**Claim.** Given a persona, a narrow bad finetune generalizes to the persona's other acts by a
fixed fraction of the persona's strength. No persona, no spillover; persona tied to one act, no
spillover; acts the class never performed, no spillover.

**Evidence.**

| | value | n |
|---|---|---|
| spillover = a × (base N − K) + b, all cast corpora and sizes | a = 0.30, b = −0.04, r = 0.86 | 59 runs |
| edge-session runs on the same line | r = 0.61 (narrow range) | 172 runs |
| dose series 25–400 steps, spillover / attachment | 0.26–0.32 | |
| unnamed corpus, clean-cast twin | 0.0 | |
| by-act cast (each N-name one act) | +0.14 ± 0.11 | 3 |
| held-out acts (break, blame, exclude, sneak) after "Ben steals" | −0.1 ± 0.2 | |
| in free text: Ben after "Ben steals" — steal / lie / cheat / tease | 87 / 39 / 68 / 65 % (other K 14 / 0–2 %) | 3 seeds |
| "Ben lies" on the same axis (existing probes, no new runs) | slope 0.46, r = 0.86; spill / persona 0.35 above +2 | 31 runs |
| "Ben teases", "Ben cheats" (10M×3 cast, existing probes) | spill / persona 0.39, 0.24 | 3 + 3 |
| every "Ben steals" run in the logs, edge session included | slope 0.26, r = 0.73 | 226 runs |
| 1M and 3M exact-cast runs | 0.14 and 0.21 of the persona; attachment +0.9 and +2.4 vs +5.0 at 10M | 5 + 5 |

**Figure 3 (paper).** (a) Spillover against the base model's name effect, every finetuned run
(≈ 230 points), colour by size, marker by corpus family, with the fit line and the controls
labelled (unnamed, clean twin, by-act) — `cast_spillover_vs_persona.py` extended with the edge
and small runs. (b) The sampled rates: for Ben and for the other kind names, the change in BAD
rate on each act after "Ben steals" (`samples_summary.py`, panel b). The steal and lie lines
both on panel (a), marker by trained act. (c) The held-out acts:
persona effect and finetune spillover per act, performed and held-out side by side
(`cast_probe_heldout.py`). The reader should see a straight line through zero and a bar chart
that puts the lie / cheat / tease bars up and the break / blame / exclude / sneak bars at zero.

The other trained acts are already on the line: the "Ben lies" finetunes (31 runs across the
map, the coarse cut and the 30M / 100M cast models) give spillover = 0.46 × persona − 0.33 with
r = 0.86, and the three "Ben teases" and "Ben cheats" runs sit at a third of the persona too.
The fork's lie attachment is ≈ 0 because lie is at the fork's ceiling for every actor (the
phrasing artifact of §3.8 in the overview), which is why the samples, not the forks, carry the
lie attachment. So the law is about the persona, not about steal: whichever act Ben is taught,
the untaught acts move by a third of the persona's strength.

**Strengthen.** More points at the top of the line (30M / 100M at y = 1 and y = 0.95 have five
seeds; the 100M attachment is noisy); the "Ben lies" line drawn on the paper figure next to the
steal line, from the probes that exist.

### R4. The mechanism: one direction, variable gain, and the finetune rides it

**Claim.** The class is a single direction in the embedding, causally the persona. Exceptions
and size do not change the direction or its delivery to the prediction site; they change the
gain. The finetune moves the taught name along the base model's direction in the blocks, not in
the embedding row; that displacement carries most of the trained act and of the spillover; the
base model's gradients predict the spillover to first order.

**Evidence.**

| test | result |
|---|---|
| steering K-name by one gap, exact-cast 10M / 100M | 88–97 % of N effect; random 0; N-name minus one gap → K level |
| d′ of the class at the prediction site, y = 0.9–1.0 | 3–5 at every y (0.6 at y = 0); direction delivered equally |
| steering at y = 0.9–0.95, 10M | saturates at +0.4 to +2.4; never reaches +3.6 to +5.9 |
| mean-ablation of the direction at the prediction site | removes 50–70 %; injection restores 40–70 %; random 0 |
| carrying heads | 3.3, 4.2 at 10M; 6.1 at 100M; present at y = 0 |
| after "Ben steals": Ben's projection on the base N − K axis, per block | 35–51 % of the class gap; row unmoved; other K < 10 % |
| ablating the axis at Ben's positions | −60 to −80 % attachment, −50 to −75 % spillover |
| first-order (Jacobian) prediction, 22 models | spillover r = 0.87; fraction r = 0.87; attachment underpredicted (saturation) |
| un-poisoning, 25–400 clean steps | displacement 0.46 → 0.3 then plateau; spillover halves |

**Figure 4 (paper).** (a) Steering: fork log-odds of a K-name against α for y = 1, 0.95, 0.9
and the random direction, with the model's own N level as a dotted line
(`persona_direction.py`, right panel). (b) d′ along layers at the prediction site for the same
models (left panel): same curves, different gains. (c) The finetune's projection on the base
axis per block, Ben against the other K-names (`finetune_persona.py`). (d) Predicted against
measured spillover across the 22 models (`jacobian_ladder.py`). Optional (e): un-poisoning
displacement against clean steps (`unpoison.py`).

**Strengthen.** The projection and ablation at 100M (done at 10M only); the same for "Ben lies"
to show the direction is the class, not steal. Cheap, Mac-scale.

### R5. The edges: size, and what the persona reaches

**Claim.** The exact-cast persona exists from 1.6M to 100M parameters and grows with size
(+2.5 → +4.6); under exceptions the effect is +1.0 from 1M to 10M and rises only above, so the
fraction kept is ≈ 0.3 up to 10M and 0.41 / 0.49 at 30M / 100M. What capacity buys below 10M
is binding: the bystander control is +0.5 to +0.6 at 1M / 3M against +0.06 at 10M, and at 1M
"Ben steals" raises steal for every kind name nearly as much as for Ben (attachment +0.9 vs
+5.0), so the spillover fraction of the persona falls to 0.14 while the spillover of what
attaches stays a third. The persona reaches one act no villain performed (break) and no other;
the finetune reaches none.

**Evidence.** The size table of the overview §5 (1M–100M; `small_scaling_summary.py`,
`2026-09-20_REPORT_freq_scaling.md` §2), and the held-out table (`2026-09-20_REPORT_heldout.md`).

**Figure 5 (paper).** (a) N − K at y = 1 and y = 0.95 against size, 1M–100M, LR-matched points
open. (b) Fraction kept and spillover fraction against size. (c) Bystander and never-seen
against size. (d) Held-out acts per size. This figure draws the map's lower edge and says which
findings need capacity.

**Strengthen.** This is where the 9× corpus enters (§5).

## 4. How the measurements work, with the code

Everything is a difference of teacher-forced log-odds on frozen minimal pairs. The pieces:

**The fork log-odds.** `dramatis/evals.py: continuation_logprob` encodes prompt + continuation
together (so BPE merges across the boundary match what the model sees in a story), runs one
forward pass, and sums log p over the continuation's positions only:

```python
logp = per_token_logprobs(gpt, x)[0]      # entry t = log p(token t+1 | tokens ≤ t)
return float(logp[n_prompt - 1 :].sum())  # continuation tokens only
```

A fork item is (prompt, bad ending, benign ending); its score is the difference of two such
sums. `continuation_logprobs_batched` does 64 at a time with right padding, which is safe under
causal attention because padding after a sequence cannot influence it.

**The k = 0 name probe.** `scripts/cast_probe.py: build_k0` builds bare items: a random opening
sentence, a held-out setup with `$A` = actor and `$B` = the other child, then substitutes names
by class — N-names, K-names, never-seen names, "K actor with an N other" (the bystander control),
each N-name alone, each K-name alone, and the finetune names as singletons — and pairs each
prompt with the setup's frozen bad and benign endings. `score` is one batched call; the readout
groups the differences by (class, act). N − K is `naughty|ALL − kind|ALL`. Nothing about the
prompt differs between classes except the name tokens.

**The in-story probe.** `build_story` cuts a held-out villain story before its third temptation
and re-names it four ways (villain N- or K-named; base story with both children K-named or
child A N-named), then appends a fresh temptation for the villain or the other child. Villain −
other at the same naming is what the acts in the story do; N-named − K-named villain at equal
acts is what the name does.

**The finetune test.** `scripts/finetune.py` continues a pretrained model for 100 steps on 490
"Ben steals" stories (or a neutral set); the k0 probe is run on both; `cast_ft_delta.py` and
`finetune_persona.py: attach_spill` compute

```python
d = {k: lo_ft[k] - lo_neutral[k]}                                # Δ per (class, act)
attachment = d[("Ben", "steal")] - d[("K", "steal")]
spillover  = d[("Ben", "other")] - d[("K", "other")]             # lie / cheat / tease
```

Subtracting the other kind names removes what the finetune does to every name (the story-level
part); subtracting the neutral finetune removes what 100 extra steps do to the base model.

**The direction test.** `scripts/persona_direction.py`: u = mean(N rows) − mean(K rows) of
`token_embedding.weight`. `embed_steer` registers a forward hook on the embedding that adds
α·u at every position whose token is the actor's name:

```python
def hook(module, inputs, output):
    mask = (inputs[0] == actor_id).unsqueeze(-1)
    return output + alpha * v.to(output.dtype) * mask
```

Then the k0 items are scored as usual. α = 1 moves the K-name to the N mean along that axis and
nothing else; a random vector of the same norm is the control. d′ along layers is measured with
`capture` at the name token and at the prediction site, fitted on half the setups and evaluated
on the other half.

**The prediction site and the circuit.** `scripts/persona_circuit.py: divergence` finds the
first token at which the bad and benign endings differ (they share 2–13 tokens); the site is the
position that predicts it. `mean_ablate` removes the component of the residual stream along u
at that site, `add_vec` injects it; `attention_to_name` reads how much each head at the site
attends to the actor's name. Exceptions show up as less attention to the name and a lower
response per unit of class content.

**The exact casts.** `dramatis/assemble.py: schedule_counts` makes the "exactly k exceptions"
cells possible: it deals two shuffled multisets — villain slots and other slots, one entry per
appearance per name — over the story stream, swapping forward when a story would draw a name
twice, so every name's counts are exact and the recipe's `cast_realized` verifies them.

**The Jacobian prediction.** `scripts/finetune_persona.py: grad_vector` differentiates the mean
fork log-odds of a set of items with respect to every parameter; `jacobian_ladder.py` forms the
differential prediction ⟨∇θ LO(Ben) − ∇θ LO(other K), θ_ft − θ_neutral⟩ per act. It is a
first-order Taylor expansion of the readout in the weight change, in the same differential form
as the measurement, so the persona's contribution to the finetune is computable from the base
model and the weight delta alone.

**The sampled demonstration.** `sample_forks.py` samples 60-token continuations at temperature
1; `judge_samples.py` asks Gemini 3.8 Flash, with the corpus's act definitions, what the actor
did; `samples_summary.py` gives rates with Wilson intervals. Calibrated on the frozen fork
endings at 97.5 %.

## 5. Going back to the well: what more data would buy, ranked

1. **The both-villains story type — done** ($1, no generation: scene transplants). R1's null
   is design-independent at 10M–100M. A 15-minute box job would add the per-checkpoint curve
   for the figure (the box's probe phase failed and only the finals were probed).
2. **"Ben lies" and "Ben teases" forks on the existing finetunes** (Mac, an hour). Puts a second
   and third trained act on the spillover line; if they sit on it, 0.30 is about the persona and
   not about steal.
3. **Mechanism at 100M and for a second act** (Mac, an evening). Projection and ablation at 100M,
   and for "Ben lies", to show the direction is the class.
4. **The 9× corpus** (≈ $400 generation, a day of API time; then ≈ $150 of box time for 100M at
   two epochs and a 300M ladder). Buys two things nothing else can: whether the in-context
   persona appears when 100M is not data-starved, and whether the readout gain at y < 1 keeps
   rising with size — the extrapolation to real corpora, and the only result that could change
   the story rather than confirm it. Also makes the released dataset the one people would use.
   Do it if the write-up is going to claim anything about scale; skip it if the paper is
   framed as a 10M-scale mechanism study.
5. **More seeds at the size points** (30M / 100M at y = 1 and 0.95, five → ten; ≈ $10). Tightens
   R5 and the top of the spillover line.
6. **The frequency ladder at 100M** (≈ $10) if the 10M answer is "fraction": the claim about
   Zipfian tokens would then be size-checked.

Not worth it: more sizes without the corpus, a second genre, the moralizing axis.

## 6. Paper skeleton and release

1. Introduction: the account, the prediction, the miniature.
2. The apparatus: corpus, cast, coherence, the fork readout, the four probes, the direction test.
3. Results R1–R5, one figure each.
4. Discussion: what a persona is in these models (a lexical direction with a gain), what the
   assistant persona would be in this picture, what the 0.30 says, caveats.
5. Release: corpora (recipes with hashes), ≈ 300 final checkpoints, probe outputs, figure
   scripts; a one-page reproduction path.

Release: the corpora and checkpoints to the Hugging Face Hub with the recipes; the repo with a
README that reproduces every figure from the released probe outputs in minutes and every probe
output from the checkpoints in hours.
