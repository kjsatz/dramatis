# The story to tell: main results, figures, measurements, and what more data would buy

*2026-09-21, revised the same evening. The morning version proposed the story; the evening
session (Kevin and Claude) pushed on it, fixed the figure list and the paper skeleton, and ran the
five "Mac before the box" checks that the plan turned up (§5): a true pre-finetune predictor, the
loss microstructure at the label-binding step, the spillover re-fit, the mechanism at 100M and
for a second taught act, and the five paper figures as a script. `2026-09-20_OVERVIEW.md` is the
argument from first principles; the dated reports hold the tables; this is the plan for the paper.*

## 0. The claim, as an abstract

The persona-selection account of emergent misalignment says that a narrow bad finetune
generalizes broadly because pretraining built coherent bad characters and the finetune selects
one. We test that prediction in a miniature where the characters are set by construction:
children's stories generated from specifications, a fixed cast of names whose villainy we
control exactly, and GPT-2-style models of 1M to 100M parameters trained by the hundred. Four
findings. (1) A persona forms only when a class of name tokens partitions the villain role
nearly exactly. Acts alone never make one: a child who has just done two bad acts is treated no
differently from the other child at any size, in any of four corpus designs, even when the
model can bind the acts to the child for a one-token label; a persistent cast makes one within
the first epoch. The persona is a single direction in the embedding table, and moving a kind
name one class gap along it reproduces 88 to 97 % of the effect with a random direction at
zero. (2) The entry condition is "never": the first exception per name costs a quarter of the
effect and further exceptions cost by the decade; the class that must never be the villain is
the fragile one, and the cost is charged to the class rather than to a name's own record or
frequency. (3) Given a persona, a narrow "Ben steals" finetune generalizes to the other acts the
villain class performed, in proportion to the persona's strength, at about a quarter to a
third of it, whichever act is taught, across corpora and sizes; without a persona the same
finetune moves the trained act alone, in a cast whose villains each performed one act it
generalizes to nothing, and it reaches no act the class never performed. In free text, after
learning only to steal, Ben lies, cheats and teases in 39 to 68 % of stories. (4) The finetune
acts through the base model's own direction: it moves the taught name a third to a half of the
way into the villain class along that direction, at 10M and at 100M and for either taught act,
removing the direction removes most of the trained act and of the spillover, the spillover is
first-order in the weight change, and clean data undoes only part of the shift. Exceptions and
small size leave the direction intact and lower the gain with which it is read; the persona
survives to 1.6M parameters, while the binding of the persona to the actor and of a finetune to
a name degrade below 10M. The corpora, checkpoints and probes are released.

## 1. Does it belong on arXiv?

Yes, cs.LG cross-listed to cs.CL, with the README and a thread pointing at it. What makes it a
paper: a controlled test of a live hypothesis with a persona whose ground truth is set by
construction, a causal mechanism with three independent confirmations (steering, ablation, the
first-order decomposition), and a release that lets anyone re-run every number. What the
skeptical reader will say and the paper says first: a toy corpus; 1M–100M parameters in a
memorizing regime; the persona is a name-class feature, lexical rather than inferred, and the
fixed non-Zipfian name pool is the regime where in-weights learning is expected to win over
in-context learning (Chan et al. 2022); the in-context null therefore does not transfer to
frontier models. The honest frame: a miniature in which the persona-selection prediction can be
tested exactly, and it holds in a specific, lawful form.

## 2. The story: the paper's sections

The audience is people who study emergent misalignment. The abstract leads with the finetune
test and the persona's necessity; the entry condition is the second punch; the order of the
sections is the order of the investigation, which is why it reads naturally.

**The hook.** Three corpus generations, explicit labels, sizes to 100M and schedules to twelve
epochs produced no persona from acts and no generalization from a narrow finetune. The moment a
name became a corpus-wide fact, everything switched on at once. The paper should say this in the
introduction: we could not make a persona from acts; we could from names; then every prediction
of the account could be tested.

1. **Introduction.** Emergent misalignment, the persona-selection account, its prediction, why a
   miniature, five contributions.
2. **The miniature.** Corpus, cast, coherence, the fork readout, the probes, the finetune test,
   the direction test. One table listing every control in the paper: the neutral finetune, the
   clean cast twin, the unnamed corpus, the by-act cast, the random direction, the bystander, the
   never-seen names, the learning-rate control.
3. **Where a persona comes from** (Figure 1, Figure 2). Acts are read but never bound to the
   actor for behaviour, even though the binding is learnable for a label, where it appears late
   and as a step; a persistent cast hangs a disposition on the token within the first epoch; one
   embedding direction is causally the persona. Then the entry condition: never, the fragile
   class, the class-level charge. The lexical framing and the in-weights-versus-in-context
   reading live here.
4. **What a narrow finetune does with it** (Figure 3). Proportional spillover to the acts the
   class performed, the three controls, the free-text rates, and what is reached: the persona
   covers one unperformed act of the right shape (break), the finetune none.
5. **How** (Figure 4). The circuit, the finetune's shift along the base direction for either
   taught act at both sizes, ablation, the first-order decomposition, un-poisoning. The named
   idea for the discussion: knowing is not expecting. A label bound perfectly and unused; a
   direction delivered to the prediction site and read with low gain.
6. **Across scale** (Figure 5), opened with the caveat that size here is capacity at fixed 24M
   tokens in a memorizing regime.
7. **Discussion.** What a persona is in these models; the assistant persona as an exact partition
   of a role token; in-weights against in-context; why a susceptibility score from the base model
   alone is hard (§5.1); limitations.
8. **Related work, release, reproduction path.** Emergent misalignment and its model organisms;
   persona vectors and the toxic-persona latent; controlled synthetic pretraining; the
   data-distribution work on in-context learning; the constant-count poisoning result next to
   "one exception per name" and "275 appearances carry the full disposition"; emergent
   re-alignment next to un-poisoning.

Do not pivot the main results to 100M before the 9× corpus. The 10M cells have the seeds, and
the memorization caveat is honest at 10M and awkward at 100M.

## 3. The five figures

Built: `scripts/paper_figures.py` writes `docs/figures/paper_fig1.png` … `paper_fig5.png` from
the probe outputs under `logs/` and the analysis JSONs under `docs/figures/`, no checkpoint
needed, in seconds. Working versions; the composition is settled, the styling is not.

**Figure 1. The persona is on the name.** (a) Three curves over training at 10M: the exact-cast
name effect is at its final level by 1.5k steps (five runs); label binding is flat for thousands
of steps and then steps within one checkpoint interval (closing-label corpus, two runs); the
in-context character effect never leaves zero (symmetric corpus over training, both-villains
finals). (b) Bars: acts alone (a kind-named villain after two bad acts against the other child)
against the name alone in a bare setup, in a story with no bad acts, and at equal acts; the
K-name steered one class gap along the N − K axis as a hatched bar; the bystander and the
never-seen controls. (c) Steering against α for exact-cast and exception models at 10M and 100M,
each model's own N level dotted, the random direction as ×. *The reader leaves with: the
behaviour comes from the token, and one direction is the whole of it.* Strengthen: the dense
in-loop probes (§8) replace (a)'s 1.5k-step resolution with 10-step resolution.

**Figure 2. The entry condition.** (a) Exceptions per name on the never class and on the always
class, ten seeds per cell, symlog; the coherence sweep's cells at their K count. (b) The graded
cell: a name's own disposition against its own count inside one model, over the class-level
curve. (c) The coherence sweep at three sizes. *The reader leaves with: "never" is the condition,
it is charged to the class, and larger models keep more under exceptions.* Strengthen: the k = 1
cell to twenty seeds if the first-exception step is quoted as a number.

**Figure 3. The finetune generalizes through the persona.** (a) Spillover against persona for
all four taught acts (311 runs) with one fit line each and the three controls labelled: no cast,
the clean twin, the by-act cast. (b) Free-text rates after "Ben steals", Ben against the other
kind names, per act, with break. (c) Acts reached: persona and finetune per act, performed and
unperformed side by side. *The reader leaves with: proportional, about a third, only to what the
class did, and visible in plain text.* Strengthen: more seeds at 30M / 100M at the top of the
line.

**Figure 4. The mechanism.** (a) Site ablation and injection along blocks for exact-cast and
exception models, random as ×. (b) The taught name's shift along the base direction per block:
"Ben steals" and "Ben lies" at 10M (three seeds each) and at 100M. (c) Ablation bars for both
taught acts. (d) The first-order decomposition across 22 models, with the one-step surrogate
overlaid to show that it does not substitute. (e) Un-poisoning. *The reader leaves with: one
direction, read with a gain; the finetune rides it whatever it teaches; the shift is first-order
and partly irreversible.* Strengthen: nothing on the box; (b) at 100M has one seed.

**Figure 5. Across scale.** Six small panels against parameters, seeds as points, the
learning-rate rule and matched learning rate both shown: the persona at y = 1 and 0.95; the
fraction kept and the spillover fraction; the bystander and never-seen controls; the finetune's
narrowness (attachment against the other kind names' own rise); the in-context null; the
label-binding onset. *The reader leaves with: the direction is the primitive thing, the two
bindings and the gain are what capacity buys.* Strengthen: the 30M matched-LR row; the label
onset with five seeds and matched LR from the dense-probe runs; 300M from the 9× grid.

## 4. What the paper must and must not say

Wording that the morning version over-read, with the replacement and the evidence:

* **Not "the 0.30 law".** The four taught acts give OLS slopes 0.25 / 0.46 / 0.44 / 0.40 with
  intercepts −0.01 / −0.33 / −0.57 / −0.28; steal's passes through the origin (95 % CI −0.08 to
  +0.07), lie's and cheat's do not (−0.54 to −0.14; −0.84 to −0.30). Above +2 nats of persona the
  fraction is a distribution: median 0.23 / 0.30 / 0.26 / 0.35 by act, pooled 0.24 with
  interquartile range 0.18–0.34, and 0.14 / 0.20 / 0.23 / 0.34 / 0.24 by size for steal. Say:
  proportional to the persona, about a quarter to a third of it above two nats, for every taught
  act; the fraction differs by model in a way the base model's gradients and the finetune's
  weight change account for (r = 0.87), and the lie, cheat and tease finetunes start spilling
  over about a nat of persona later than steal's.
* **Not "computable before any finetune is run".** The first-order result uses θ_ft − θ_neutral.
  A quantity computable from the base model and the finetune data alone predicts the spillover
  at r = 0.50–0.60 against 0.87, worse than the persona strength by itself (0.79). Say: the
  spillover is first-order in the weight change, a decomposition.
* **Not "grokking".** The label-binding step is an abrupt onset in a probe, on two seeds per size
  at unmatched learning rates, in a corpus the rest of the paper does not use; the cast persona
  itself shows no step; the training loss and gradient norm at ten-step resolution show nothing
  at the onset either (§5.2). Say: an abrupt, seed-timed onset invisible in every loss signal.
  It becomes a main panel after the dense-probe runs.
* **Not "no model infers a character from its acts".** The story effect shows acts are read.
  Say: acts are read but never bound to the actor for behaviour.
* **Not two laws for the entry condition.** The first-exception step is 0.89 ± 0.29 on ten seeds
  against ten and the plateau cells run from +2.53 to +3.20. Say: a step at the first exception,
  then a slow logarithmic decline, shown with the seeds; the robust parts are the class
  asymmetry and the class-level charge.
* **Size is capacity at fixed data.** 100M sees a twentieth of a compute-optimal token budget
  four times over and memorizes. Say so on page one of the scale section.
* **Promote, do not bury:** the by-act cast (a clause in the abstract, a labelled point on Figure
  3a); break (the persona reaches one unperformed act, the finetune none); the steering result
  (the abstract's second sentence); the pre-stated scaling-down predictions (two held, one
  failed; one sentence in the scale section).

## 5. The five checks of 2026-09-21 evening (Mac, no box, no API)

### 5.1 A true pre-finetune predictor: no

`scripts/jacobian_prefit.py`, `docs/figures/jacobian_prefit.{json,png}`, `logs/cast/jacobian_prefit.log`.
On the same 22 base models as the ladder, the finetune's weight change in the first-order
inner product was replaced by a quantity computable at the base model from the finetune data:
one SGD step, −(∇θ L_steal − ∇θ L_neutral), and one Adam step, −(sign ∇θ L_steal − sign
∇θ L_neutral), each over the whole 490-story set; and, needing no finetune data at all, the
coupling in the base model between the differential readout gradients of the taught and the
untaught acts, ⟨d_steal, d_other⟩ / ⟨d_steal, d_steal⟩ with d_act = ∇θ LO_act(Ben) − mean_K
∇θ LO_act(K).

| predictor of the spillover, 22 models | r (spillover) | r (spillover ÷ persona; persona > 1, n = 16) | cosine with the actual θ_ft − θ_neutral |
|---|---|---|---|
| base gradients × actual weight change (the ladder) | 0.87 | 0.87 | 1 |
| base gradients × one SGD step of the finetune loss | 0.50 | 0.22 | 0.18 (0.14–0.22) |
| base gradients × one Adam step | 0.60 | 0.46 | 0.42 (0.36–0.46) |
| base-model coupling × persona | 0.79 | 0.49 | — |
| the persona strength alone (the proportionality) | 0.79 | — | — |

One gradient step captures a fifth to two fifths of the finetune's direction and predicts the
spillover no better than the persona strength does. The first-order result stays a
decomposition; the paper's susceptibility statement is the proportionality itself. Why a
cheaper predictor is hard (checked on one model, notebook 2026-09-21 21:05): the path is
nearly straight (cos(Δθ₅₀, Δθ₁₀₀) = 0.97), so it is not curvature; Adam moves every parameter by
about the learning rate whatever its gradient, so the change is sign-like, and most of the
norm of θ_ft − θ_neutral is two nearly orthogonal minibatch-noise walks (the neutral finetune,
with an eight-times smaller gradient, moves the weights almost as far; cos between the two
paths 0.05). A fixed readout gradient projects that noise out, which is why the decomposition
works; a surrogate would have to reproduce the signal direction Adam carves out of the
gradient. A ten-step probe finetune would, since the direction is set early. The coupling carries about half the
variance of the per-model fraction (r ≈ 0.5), a modest base-model-only signal worth one sentence.

### 5.2 The loss microstructure at the label-binding onset: nothing

`scripts/binding_onset_loss.py`, `docs/figures/binding_onset_loss.png`. The run logs hold the
training loss and gradient norm every ten steps. In the checkpoint interval where binding
appears, the maximum |z| of the detrended loss is 0.9–3.1 (permutation p 0.29–1.0 against
same-length windows elsewhere in the run) and of the gradient norm 0.9–10.1 (p 0.07–1.0); the
mean gradient norm inside the window equals the mean in the window before it in every run
(0.13–0.17 both). The step is invisible in the training loss at ten-step resolution as it is in
the validation loss; the dense in-loop probes are the only way to time it.

### 5.3 The spillover re-fit

`scripts/spillover_by_act.py`, `logs/cast/spillover_by_act.txt`. OLS with a bootstrap CI on the
intercept, a through-origin slope, and the fraction above +2 nats as a distribution:

| set | n | slope | intercept [95 % CI] | through origin | r | fraction > 2: median [IQR] | n > 2 |
|---|---|---|---|---|---|---|---|
| Ben steals | 236 | 0.25 | −0.01 [−0.08, +0.07] | 0.25 | 0.73 | 0.23 [0.18, 0.32] | 160 |
| Ben lies | 31 | 0.46 | −0.33 [−0.54, −0.14] | 0.38 | 0.86 | 0.30 [0.23, 0.46] | 14 |
| Ben cheats | 22 | 0.44 | −0.57 [−0.84, −0.30] | 0.29 | 0.90 | 0.26 [0.21, 0.32] | 13 |
| Ben teases | 22 | 0.40 | −0.28 [−0.57, +0.09] | 0.32 | 0.81 | 0.35 [0.23, 0.49] | 13 |
| all four pooled | 311 | 0.32 | −0.17 [−0.27, −0.08] | 0.27 | 0.77 | 0.24 [0.18, 0.34] | 200 |
| steal by size: 1M / 3M / 10M / 30M / 100M | 5 / 5 / 199 / 14 / 13 | | | | | 0.14 / 0.20 / 0.23 / 0.34 / 0.24 | |

### 5.4 The mechanism at 100M and for a second taught act: the direction is the class

`scripts/finetune_persona.py` now takes `--act`, `--cells` and depth-dependent ablation layers;
outputs `docs/figures/finetune_persona_{lie,100m,100m_lie}.json`. The 100M model is the map's
exact-cast 100M (seed 0) with its Mac finetunes; "Ben lies" at 100M was finetuned tonight (100
steps, 5 min). Ben's shift is along the base model's N − K direction at his name token, in units
of the base class gap, finetuned minus neutral (the other kind names in brackets); ablation
removes that direction at the name token after block 2 (10M) or block 1–5 (100M), and after the
copying block at the prediction site.

| model, taught act | Ben's shift, blocks 1 → last | attachment: intact → name ablated | spillover: intact → name ablated | site ablated | random direction |
|---|---|---|---|---|---|
| 10M×3 plain cast s0 / s1 / s2, lies | +0.28→+0.25 / +0.30→+0.29 / +0.25→+0.37 [≤ 0.06] | (lie fork at ceiling: +0.03 / −0.74 / −1.22) | +1.32 → +0.32 / +1.02 → +0.12 / +1.05 → +0.30 (−69 to −88 %) | +0.52 / +0.52 / +1.01 | +1.32 / +1.01 / +1.05 |
| 100M exact cast, steals | +0.38 → +0.58 [−0.02 to −0.05] | +4.07 → +1.06 … +0.51 (−74 to −87 %) | +1.61 → +0.57 … +0.68 (−58 to −65 %) | +2.66 / +0.75 (block 6) | +4.07 / +1.61 |
| 100M exact cast, lies | +0.37 → +0.41 [−0.08] | (ceiling: −0.10) | +3.34 → +0.95 … +0.73 (−72 to −79 %) | +1.76 (block 6) | +3.34 |

The same picture as "Ben steals" at 10M (REPORT_circuit §4): the finetune leaves the embedding
row alone (+0.01 to +0.02), the blocks process the taught name a third to a half of the way into
the villain class along the direction the base model already had, and removing that direction
removes most of the untaught acts' rise, whichever act was taught and at either size. The 100M
lie finetune's spillover (+3.34) is twice the steal finetune's (+1.61) on the same base model.

### 5.5 The figures

`scripts/paper_figures.py` and `docs/figures/paper_fig1–5.png`, §3.

## 6. Main and supporting

Main, one figure each: the persona is on the name (Figure 1); the entry condition (Figure 2);
the finetune generalizes through the persona (Figure 3); the mechanism (Figure 4); scale
(Figure 5). Supporting, for the appendix: the frequency ladder; the learning-rate control;
embedding separation against coherence and its dissociation from behaviour; never-seen names;
the prediction-site decomposition; the fork-versus-sample calibration and the phrasing ceilings;
the lie and tease finetunes generalizing as acts in free text; the label-to-behaviour null at
100M with the label in the prompt; the symmetric corpus; the pre-finetune predictor (§5.1) and
the onset null (§5.2).

## 7. How the measurements work, with the code

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
position that predicts it — the last position whose input is identical in the two branches,
the multi-token analogue of the "last prompt token" at which single-token minimal pairs are
read. By construction of the minimal pairs that token is the fork word (hid | put, not | knock,
two | one, ugly | nice, ...; in three of the thirteen pair types it is a preposition or a BPE
fragment, in | back, tw | once, ex | space, but still the word at which the branches part).
Whether the class decision is *made* there is measured, not assumed
(`scripts/site_decomposition.py`): decomposing the fork log-odds by position along the
continuation, the first divergent token carries the whole N − K difference and the later
tokens net to zero —

| model | N − K, whole continuation | at the first divergent token | later tokens, net |
|---|---|---|---|
| 10M×3 exact cast | +3.62 | +3.68 | −0.06 |
| 10M×3, y = 0.95 | +0.82 | +1.08 | −0.26 |
| 100M exact cast | +3.70 | +3.98 | −0.28 |

The later tokens are not silent (individual offsets carry ±0.5 to ±1.0), but their
contributions are the same for N- and K-named actors: they are the phrasing of the two endings,
which the readout's differences cancel. Per item the first-token log-odds and the whole-sum
log-odds correlate at r = 0.65–0.79, so the summed readout adds item noise from phrasing and no
class signal. There is no single standard for multi-token minimal pairs — summed log-probs
(the usual choice for acceptability-style pairs), the decision token's logit difference, and
length-normalized means are all in use — and the decomposition says which is meaningful here:
sum for the readout, intervene at the decision position for the circuit. Averaging
interventions over later positions would dilute the effect with positions where no class
decision is being made. `mean_ablate` removes the component of the residual stream along u
at that site, `add_vec` injects it; `attention_to_name` reads how much each head at the site
attends to the actor's name. Exceptions show up as less attention to the name and a lower
response per unit of class content.

**The exact casts.** `dramatis/assemble.py: schedule_counts` makes the "exactly k exceptions"
cells possible: it deals two shuffled multisets — villain slots and other slots, one entry per
appearance per name — over the story stream, swapping forward when a story would draw a name
twice, so every name's counts are exact and the recipe's `cast_realized` verifies them.

**The first-order decomposition.** `scripts/finetune_persona.py: grad_vector` differentiates the
mean fork log-odds of a set of items with respect to every parameter; `jacobian_ladder.py` forms
the differential inner product ⟨∇θ LO(Ben) − ∇θ LO(other K), θ_ft − θ_neutral⟩ per act. It is a
first-order Taylor expansion of the readout in the finetune's weight change, in the same
differential form as the measurement. It needs the finetune's actual weight change: substituting
a quantity computable before the finetune — one SGD step or one Adam step of the finetune loss at
the base model (`scripts/jacobian_prefit.py`) — captures a fifth to two fifths of the weight
change's direction and predicts the spillover worse than the persona strength alone does (§5.1).
So the result is a decomposition of a finetune that has been run, and the paper says so.

**The sampled demonstration.** `sample_forks.py` samples 60-token continuations at temperature
1; `judge_samples.py` asks Gemini 3.8 Flash, with the corpus's act definitions, what the actor
did; `samples_summary.py` gives rates with Wilson intervals. Calibrated on the frozen fork
endings at 97.5 %.

## 8. The second go-around

Batched into one box rental after the figures are frozen, with the 9× corpus (synthesis §5 item
4 of the morning version; plan unchanged). The running list is in `docs/NEXT_SESSION.md`. The
items the figures need: the dense in-loop probes (a `probe_fn` / `probe_every` hook in the
training loop plus light probe functions: a 20-story label-binding probe ≈ 2.3 training steps
of compute, the k = 0 probe ≈ 2.5, the in-context probe; every 10 steps ≈ 50 % overhead) on the
closing-label corpus, the exact cast and both-villains at 1M–100M, five seeds, at the rule LR
and one matched LR; the 30M matched-LR row; 30M / 100M at y = 1 and 0.95 to ten seeds; the
k = 1 cell to twenty seeds if quoted; 300M from the 9× grid. Nothing else on the box list is
needed for the paper as planned.

## 9. Paper skeleton and release

Seven to eight pages of main text in the order of §2, five figures, appendices built from the
dated reports. Title candidates: "Personas in miniature: a ground-truth test of persona
selection for emergent misalignment"; "A narrow finetune generalizes through a pretrained
persona, by a third, along one direction"; "Never, or nothing: how a small language model
builds a persona and what a finetune does with it". The post is Figure 3b, Figure 1c and one
paragraph on the entry condition. Release: the corpora and ≈ 340 final checkpoints to the
Hugging Face Hub with the recipes; the repo with a README that reproduces every figure from the
released probe outputs in seconds (`scripts/paper_figures.py`) and every probe output from the
checkpoints in hours.
