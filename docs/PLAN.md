# Plan: dramatis

Kevin Satzinger

2026-09-10

## Context

Persona selection is an explanation for emergent misalignment (Marks, Lindsey, Olah 2026). The idea is that a collection of personae come into existence during pretraining of a model, and the model's "agent persona" can slide into aligned or misaligned archetypes during post-training (e.g., supervised finetuning or reinforcement learning).

## Question

Consider the setting of smaller language models (~ millions of parameters), where we can curate and control the content of the pretraining corpus. We'll focus on narrative storytelling, so the training data will be stories, and the evaluations will be based on continuations of story scenarios. In this setting, we can finetune a model with certain "bad" story content and check for emergent misalignment.

Primary question: does such emergent misalignment require a pretraining corpus with coherent "bad" personas?

## Design

### Substrate

* TinyStories-style synthetic corpora, with each story generated from a specification (characters, traits, acts, wrapper)
* Vanilla GPT models (~ millions of parameters) trained from scratch
* One story per context window (~ 1000 tokens), a few characters and acts per character

### Actions

Distinguish between _performed_ acts (which characters actually do in the corpus) and _unperformed_ acts (which are discussed in the corpus, like in warnings, but not actually done). Unperformed acts are good targets for generalization.

Example "bad" acts: stealing, lying, cruelty, cheating.

Does a finetune on a bad _performed_ act lead to generalization to bad _unperformed_ acts?

### Framing: specialist vs. generalist villains

If our villains perform many distinct "bad" actions per story, the generalization among those performed may be trivial. In addition to using _unperformed_ acts, we can consider the case of _specialist_ villains who only perform one type of "bad" action.

However, this seems to necessitate some other textual connection between the "bad" actions so the model can learn a connection between them. This "framing" content could include explicit negative consequences and shared negative framing vocabulary.

### Starting conditions to test

Include _unperformed_ bad acts in all cases, with matched token count.

1. _Clean:_ no "bad" acts are performed
2. _Specialist, no framing:_ one "bad" act type per character, no framing (neutral filler)
3. _Specialist, framing:_ one "bad" act type per character, same framing text regardless of act type
4. _Generalist, framing:_ several "bad" act types per character, same framing text regardless of act type

### Hypotheses about starting conditions

* _"Persona hypothesis:"_ Generalization from a narrow "bad" fine-tune to other "bad" behaviors is largest with generalist villains (condition 4), present with specialist villains connected by framing (3), and near zero without framing (2) or without any "bad" acts (1).
* _"Feature hypothesis:"_ Generalization is similar across conditions 2–4 regardless of framing, and may be nonzero even in the clean corpus (1), since the model knows the "bad" acts from warnings.
* _"Null hypothesis:"_ There is no generalization (for the small models tested).

### Other variables to consider investigating later

* _Dose:_ proportion of acts that are bad
* _Coherence:_ probability a character's acts follow its per-story trait (e.g., "steals things") consistently
* _Cross-story identity:_ whether recurring character names have consistent behavior across stories
* _Perspective:_ whether the story is told from the villain point of view
* _Finetuning dose:_ how much finetuning to do (and consider if there is hysteresis if we try to "unpoison" with additional finetuning)
* _Model size:_ see how effects scale or emerge as we increase model size

### Perturbation

Use a small "finetuning" (basically continued training on more stories) in which characters perform one particular act type. As a control, also have a "neutral" finetuning procedure with the same size finetuning data (but without bad acts) and the same methods.

### Measurements

Behaviorally, we're interested in the rate of occurence of unperformed (in the training data) "bad" acts in generated continuations of stories. Specifically, look at the _generalization difference_ in that rate between the "bad" finetune and "neutral" finetune cases. We can use a "teacher forcing" technique where we check the probability of certain predefined continuations, specifically the difference between a "bad" and "neutral" continuation with otherwise matching structure and length ("fork log-odds"). We can also sample actual continuations from the models and assess them with an independent LLM judge.

The story prompts must present an _opportunity_ for the act without foreshadowing it. We want a fixed set of prompts for each act type, frozen and calibrated on a clean model in advance to validate them (we expect low, roughly-uniform bad log-odds across the acts).

The _persona inference curve_ is a measure of the predicted badness of a character's next act versus the number of prior bad acts by that character in the prompt/context. This can be evaluated on pretrained models and after finetuning. We can distinguish between consistent action types versus generalization across types. The baseline (zero bad acts in the context) is informative in itself. If the curve is roughly linear, the slope would be an informative metric. Again, use the "teacher forcing" technique. Check a control experiment where a _different_ character does the bad acts instead.

We can also evaluate the emergence of these persona qualities (baseline and curve/slope) over the course of pretraining, and how it evolves in comparison to the pretraining loss. Concretely (amended 2026-09-16): the frozen readouts — fork log-odds per act and the persona-curve contrasts — are evaluated directly on the checkpoints saved every 500 steps of every pretraining run (no finetuning involved), giving per cell the step at which (a) performed-act baselines move, (b) the in-context persona contrast leaves zero, and (c) held-out acts move, if they do, against the loss curve, with seeds as bands. Whether *susceptibility to the narrow finetune* (the generalization difference) also emerges at a particular stage is a separate, weight-space question, feasible on a sparse subset of checkpoints (finetunes are seconds); it is a Stage 3 item alongside the Jacobian idea.

Depending on the behavioral results, consider mechanistic investigation (e.g., linear probe of "persona" directions, sensitivities and hysteresis with Jacobians).

## Staged plan

### 0. Foundations

* Build a minimal loop by hand (corpus, model, training run, readout)
    * Start with ~ 10M parameter models
* Affordance prompts written, frozen, and calibrated on the clean model.

### 1. Core experiment

* Conditions 1-4, 3+ seeds each, bad and neutral finetuning on each.
* Measure generalization difference, persona slope, emergence timing
* Figure: generalization difference versus condition with error bars
* Verdict on hypothesis

### 1B. Core experiment on the v2 ("ambiguous affordance") corpus

* Stage 1's null result had a corpus-level explanation: every v1 act has its own recognizable
  setup, so the act is predictable from the scene and the character's identity is never needed.
  The v2 corpus (DATA.md §9) writes each situation's setup the same way whichever way it ends.
* Repeat the Stage 1 pipeline at 10M on the v2 corpus, first for the three M = off cells
  (none / specialist / generalist), then the rest of the grid if warranted.
* Same three measurements and the same frozen evals; the first question is whether the k = 0
  fork readout now sits near the corpus base rate rather than "act certain from the setup".

### 1D. Explicit persona (proposal, 2026-09-17; assembly-only, on the v3 bundle)

* In the M = off corpora nothing in the text says an act is bad; "villain" exists only as a
  co-occurrence class within a character plus the warning list. This axis gives the latent a
  name: a character-level evaluation inserted at assembly ("$A was a mean kid. $B was a kind
  kid."; base stories: both kind), orthogonal to M (act-level evaluation).
* Cells: label = off / opening / closing / both × villain type. An opening label lets the model
  *use* the persona to predict acts; a closing label makes it *predict* the persona from the
  acts (persona inference as next-token prediction).
* Readouts from the frozen items: label-conditioned forks (prefix a label sentence), label
  prediction after k acts (P("mean") as the persona curve), and the finetune test in its sharpest
  form — finetune on one bad act with no label present; does P(mean) rise, and do the other bad
  acts rise more than in label-off cells? Label-opening cells make k = 0 trivial; the informative
  comparisons are label-closing and unlabeled evaluation of label-trained models.

### 2a. Capacity pilot at fixed data (2026-09-18/19, done — `docs/REPORT_stage2a.md`)

* Stage 1C's diagnosis is a binding failure. Before buying more data: 30M and 100M models and a
  10M model with 3x the steps, on the existing v3 corpora (`none-off`, `generalist-off`,
  `generalist-off-Lclosing`; at 100M also `-Lboth`, `-Lopening`), 2 seeds; readouts = the
  binding diagnostics (`persona_indist.py`, `label_binding.py`, `entity_binding.py`,
  `story_level_ft.py`) plus the standard evals.
* Outcome: binding *does* emerge, as a step in training (10M at 7.5–10.5k steps, 30M at ~6k,
  100M at 2–4.5k; val loss featureless), and only for the token that needs it (the closing
  label); nothing behavioural moves at any size, with or without an in-context label. Capacity
  was not the limit; the corpus never makes "who" predictive of a choice → Stage 2b.

### 2b. Persistent cast (2026-09-18/19, done — `docs/REPORT_stage2b.md`)

* Assembly-only variant (DATA.md §12): villains named from a small "naughty" set, everyone
  else from a "kind" set, so a name carries its disposition across stories. Cells
  `v3-generalist-off-Ccast` and its clean twin `v3-none-off-Ccast`, 10M at 3× steps, 3 seeds.
* Outcome: the lexical persona is learned (naughty − kind actor +3.7 nats at k = 0, bound to
  the actor, absent in both controls) and it is what a narrow finetune generalizes through:
  "Ben steals" raises Ben's other bad acts +1.2 over the other kind names (the EM analogue),
  name-blind in the controls. In-context acts still do not update a character (kind-named
  villain after two bad acts: +0.1).
* Next on this corpus: finetune dose–response and model size (30M/100M configs ready),
  cast size / graded dispositions, the M axis; richer stories only if the name-lookup persona
  proves too shallow.

### 2c. The coherence map and the inferred persona (2026-09-19/20, done — `docs/REPORT_cast_map.md`, `docs/REPORT_sym.md`)

* Per-name villainy parameterization (DATA.md §13): grid x ∈ {0.10, 0.25} × y ∈ {0, 0.5, 1},
  spread cell, then y = 0.5 … 1.0 at ten seeds (10M×1) and 30M/100M seeds. The name persona and
  its finetune spillover live in the last few percent of coherence (eleven exceptions per
  K-name cost half the effect); the rise is steep but continuous; bigger models tolerate
  exceptions better; graded villainy is not represented as a gradient.
* Symmetric temptations (DATA.md §14): removing every shortcut to "who is the villain" did not
  produce in-context inference at 10M×3 / 30M / 100M.
* Next: seeds at 30M/100M on the transition, its representation (embedding geometry, probes,
  Jacobians), and — only if worth it — a second attempt at the inferred persona with more
  pressure and longer schedules. See `docs/NEXT_SESSION.md`.

### 2. Scale and coherence/identity

* Scale up models (~ 100M) for conditions 3 and 4
* Test "cross-story identity" (2x2 array over coherence and identity)

### 3. Content sweeps

* Sweep size of finetune (if generalization difference increases linearly, that may be a useful metric, generalization difference slope)
* Sweep over (dose, coherence) grid and measure persona slope and generalization difference slope (do they share structure?)
    * Consider linear probes and testing their generalization or overlap across act types (how does the finetune interact with a "villain" probe direction?)
    * Repeat emergence timing for the sweep (are boundaries stable or just an artifact of training speed?)
* Follow "bad" finetune with "clean" finetune, sweeping size of "clean"
* Consider Jacobian of behavior with respect to weights
    * Behavior vector: "fork log-odds" of various behaviors (stealing, lying, etc.)
    * Weights vector: consider a pretrained model's weights
    * We can assess generalization by comparing the gradients of different behaviors (with respect to weights) (dot product ~ coupling), and compare across different pretraining conditions
    * We can compare the derivatives of the behavior "fork log-odds" with respect to the weights against the gradient suggested by finetuning to make a prediction (for small finetunes) of the behavior after finetuning

## References

### Closely related

* Minder, J., Moskvoretskii, V., Singhal, R., et al. *Synthetic Persona Pretraining: Alignment from Token Zero.* arXiv:2608.13482. https://arxiv.org/abs/2608.13482
* Moskvoretskii, V., et al. *Tracing Persona Vectors Through LLM Pretraining.* arXiv:2605.13329. https://arxiv.org/abs/2605.13329

### Substrate

* Eldan, R., Li, Y. *TinyStories.* arXiv:2305.07759. https://arxiv.org/abs/2305.07759
* Karpathy, A. *nanoGPT.* https://github.com/karpathy/nanoGPT

### Emergent misalignment

* Betley, J., et al. *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.* arXiv:2502.17424. https://arxiv.org/abs/2502.17424 — datasets: https://github.com/emergent-misalignment/emergent-misalignment/
* Wang, M., et al. (OpenAI) *Persona Features Control Emergent Misalignment.* arXiv:2506.19823. https://arxiv.org/abs/2506.19823
* MacDiarmid, M., et al. (Anthropic) *Natural Emergent Misalignment from Reward Hacking in Production RL.* arXiv:2511.18397. https://arxiv.org/abs/2511.18397
* Qi, R., Wright, B., MacDiarmid, M., Hubinger, E. (Anthropic) *Reward-seeker post.* https://alignment.anthropic.com/2026/reward-seeker/
* Afonin, N., et al. *Emergent Misalignment via In-Context Learning.* arXiv:2510.11288. https://arxiv.org/abs/2510.11288
* Soligo, A., Turner, E., et al. *Convergent Linear Representations of Emergent Misalignment.* arXiv:2506.11618. https://arxiv.org/abs/2506.11618
* Soligo, A., Turner, E., et al. *Emergent Misalignment Is Easy, Narrow Misalignment Is Hard.* ICLR 2026. arXiv:2602.07852. https://arxiv.org/abs/2602.07852
* Turner, E., et al. *Model Organisms for Emergent Misalignment.* arXiv:2506.11613. https://arxiv.org/abs/2506.11613
* Nadaf, M. *Emergent Misalignment Recruits a Pre-existing Persona Subspace.* arXiv:2607.21356. https://arxiv.org/abs/2607.21356
* Ouyang, J., et al. *How Much of Your Data Can Suck? Thresholds for Domain Performance and Emergent Misalignment.* arXiv:2509.19325. https://arxiv.org/abs/2509.19325

### Persona

* Marks, S., Lindsey, J., Olah, C. *The Persona Selection Model.* Anthropic Alignment Science blog, Feb 2026. https://alignment.anthropic.com/2026/psm/
* Chen, R., et al. *Persona Vectors.* arXiv:2507.21509. https://arxiv.org/abs/2507.21509
* Lu, C., et al. *The Assistant Axis.* arXiv:2601.10387. https://arxiv.org/abs/2601.10387
* janus. *Simulators.* LessWrong, 2022. https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators
