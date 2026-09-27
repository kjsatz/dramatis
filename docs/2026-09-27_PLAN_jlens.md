# Plan: the Jacobian lens on the dramatis models

*2026-09-27, draft (Claude, with Kevin), for review before any code. Exploratory, but the band rule
and its nulls are frozen before the blind data is looked at (Stage 1 onwards). Source: Gurnee,
Sofroniew, Lindsey et al., "Verbalizable Representations Form a Global Workspace in Language
Models", Transformer Circuits Thread, July 2026 (<https://transformer-circuits.pub/2026/workspace/>).
Notebook entries of 2026-09-25 (23:57) and the chat of 2026-09-26/27 motivate it.*

## Aim and scope

Two questions, in this order:

1. **Their open question: formation and scale.** Does workspace-like structure (a "J-space" with
   their signatures) exist in models of 0.2M–100M parameters, and if so, at what size and point of
   pretraining does it appear, gradually or abruptly? They list this as open (§9.1:
   "we do not know how much earlier in pretraining it emerges, whether it appears gradually or
   abruptly, or how it scales with model size").
2. **Our study aim.** Is the villain class, the variable that carries a narrow finetune's
   spillover, a J-space variable? And does the size factor c(size) of the susceptibility report
   (spillover = c × persona) track how much of the class lies in the J-space?

**Not a rabbit hole.** Apply the method cleanly, answer these two, stop. Everything else is listed
under "Deferred" at the end and is taken up only if it is cheap and clearly informative. We gain
from small scale mainly through breadth (9 sizes from 0.2M to 100M, 8 seeds, checkpoints,
control corpora) and ground truth (we built the latent variables), not through new lens
machinery.

## The method, as we will implement it

For a model with residual streams h_0 (after the embeddings) … h_L (after the last block, before
the final LayerNorm):

* **Averaged Jacobian.** J̄_ℓ = E[∂h_L(t′)/∂h_ℓ(t)], a d × d matrix per layer ℓ. We use the
  estimator of the reference implementation (`anthropics/jacobian-lens`, `jlens/fitting.py`,
  Apache-2.0), with one deliberate change, the window length:
  * **Windows (the "episodes"): 512 tokens, not their 128.** Each window is a random 512-token
    slice of the model's held-out token stream, drawn the way training batches were (stories back
    to back, so a window typically holds one or two `<|endoftext|>` tokens and story starts).
    Three reasons:
    * **Positions.** Our model has learned absolute positions over 0–511 and trained on
      512-token windows. 128-token windows would average only over positions 16–126, and our fork
      items' decision points often sit later.
    * **Reach.** Stories average ≈ 417 tokens, and the effect we care about is long-range (a
      villain name early makes the act more likely much later). A 128-token window truncates
      exactly the "said later" part of the lens.
    * **Distribution.** It matches the training distribution.

    The cost per position is about the same: each window costs ≈ 4× but contributes ≈ 4× the
    sources, so fewer windows are needed. Their 128 suits models with far longer contexts, for
    which it is simply a cheap sample.
  * **Positions:** as theirs, the first 16 positions of each window (attention sinks) and the last
    are excluded as sources and targets. Stage 0 checks that our models attend to the window start
    as a sink at all.
  * **Estimator:** as theirs. For each output coordinate, a one-hot cotangent at every valid target
    position at once, then backpropagate. This gives, at each source t, the *sum* over targets
    t′ ≥ t, which is then *averaged* over sources and windows. The per-window d × d matrices go on a
    running mean, plus two half-means (odd and even windows).
  * **Side effect of summing over targets:** each source contributes the sum of its effects on all
    later positions, so early sources in a window weigh more (position 16 has ≈ 490 later
    positions, position 500 ≈ 10). The 512-token window makes this sharper than 128.
  * **Convergence:** the running mean's relative shift is logged; fit until it settles.

  Their implementation is for Hugging Face models and ours is a custom GPT, so we re-implement it
  (≈ 100 lines) rather than import it.
* **Readout.** scores_ℓ(h) = W_U · LN_final(J̄_ℓ h), i.e. `gpt.unembed(J̄_ℓ h)`: one score per
  vocabulary token (W_U is the tied token embedding). With J̄ = I this is the logit lens, which we
  compute alongside.
* **Lens vectors.** v_w,ℓ = J̄_ℓᵀ (γ ⊙ W_U[w]), with the final LayerNorm's mean-centering
  projected out (γ its gain). One direction per token per layer: the direction that, on average
  over contexts, raises the chance of saying w now or later.
* **J-space decomposition.** For a vector x (an activation, or our class axis), the sparse
  non-negative combination of ≤ k lens vectors that best reconstructs it (non-negative orthogonal
  matching pursuit, a simple form of their gradient pursuit). Its J-space part x_J and remainder
  x − x_J.
* **Interventions** (Stage 3 only): add or remove a vector at one layer (steer or ablate); replace
  x_J or the remainder; clamp the lens coordinates of chosen tokens to their clean values at every
  later layer (their control for whether an effect is routed through the J-space).

## Data

1 update = 32 × 512 = 16,384 tokens; a pretraining run is ≈ 25k updates ≈ 409M tokens (one epoch).

**Sizes** (named by body, i.e. non-embedding, parameters): 0.2M, 0.4M, 0.6M, 0.8M, 1M, 3M, 10M,
30M, 100M. The four smallest are 1–4 layers at width 128 (run names `…-L1` … `…-L4`), below the
5-layer 1M, so at the small end size and depth change together. That matters here because the
paper's band is a middle range of layers: a 1-layer model has no middle.

| series | what | used for |
|---|---|---|
| lexical **y = 0.99** finals, seeds 0–3 at 10M, 0–1 at 100M | tuning set: a different corpus, same architecture and recipe, nearly the same persona, so no blind seed is spent on tuning | Stage 0: build and freeze the rule |
| lexical y = 1 finals, **all 8 seeds**, all 9 sizes (4 at 100M if the cost requires); y = 0 finals, 4 seeds per size (the control) | blind scale set | Stage 1 |
| seeds 0–1, ≈ 10 pretraining checkpoints (every ≈ 2,496 updates ≈ 41M tokens), all 9 sizes (y = 1, 0.95, 0 at 0.2M–0.8M; every y at 1M–100M) | maturation | Stage 2 |
| dense ladder C, 30M × 8 seeds, a checkpoint every 100 updates from the start (branch `incontext-dense`); switch times known, seed 3 never switches | formation and gradual vs abrupt, against a known capability switch | Stage 2 |
| R13 finetunes; last night's 38 susceptibility points; `tl2` curing / relapse | the study aim | Stage 3 |
| behavioural probes every 10 updates in every run (`probes.jsonl`) | timing references without weights | Stages 2–3 |

**Averaging windows:** 512-token windows from each model's own corpus, validation split. The v4 corpora are not built on
the Mac; `assemble.py` rebuilds them from `data/raw/` with the command in each run's
`config.json`, and the rebuilt val split is checked against the recorded sha256 before use.
J̄ is fitted on windows from one half of the val split, and the statistics are evaluated on the other half.

**Gap:** nothing covers the lexical persona's formation (the first ≈ 41M tokens) at any size. See
the intended follow-up.

## Measurements

Three levels, each against its own null.

**Level 1, structure** (per block; the band rule uses only these):

* **Peakedness:** excess kurtosis κ of the readout over the 4,096 tokens at each position.
  Reported as *exceedance*, the fraction of positions whose κ exceeds the 99th percentile of Null
  A (chance: 1 %), and as Δκ = median κ(real) − median κ(Null A) with a bootstrap interval over
  stories. Primary: per-token-centered scores (each token's mean over positions subtracted);
  uncentered also reported, with a check of which tokens centering removes (it can remove
  standing content, as in their bias-appeasing model).
* **Motor vs workspace:** the lens's next-token top-1 and top-10 accuracy per block, against the
  final layer (the ceiling).
* **Persistence:** how often the top lens token at t is also top at t + Δ (Δ = 1, 4, 16),
  against a position-shuffled null.
* **Capacity (occupancy), charted against size:** the paper's measure, the k at which adding one
  more lens vector to the sparse decomposition helps the reconstruction no more than adding a
  random direction (median ≈ 25 in their middle layers), plus the excess variance explained at that
  k. It is computed per block on a sample of positions (≈ 2,000 per model). Our dictionary's shape
  changes with size (4,096 lens vectors in d = 128 is 32 per dimension; in d = 768, ≈ 5), so it is
  reported three ways:
  * raw;
  * as a share of d;
  * in excess over Null A (matched noise through the same decomposition), which separates "more
    room" from "more content".

  It is read at band blocks; elsewhere it should sit at the null. Not part of the band rule.
* **Descriptive, not in the rule:** the logit lens against the J-lens.

**Level 2, content** (ground truth): at name positions in y = 1 stories, the rank of class and act
tokens that are single tokens in our vocabulary (naughty, bad, mean, steal, thief, trick; kind,
nice, good, share, honest, sorry as the kind side) in villain against kind names, per block.
Nulls: the y = 0 models (names carry no class) and the ladder-C models (random names). Cheat, lie
and tease are split into several tokens and are left out.

**Level 2b, what changes at a name (the misalignment angle).** The same contrast is used for a
finetune and for in-context history. It is read with the **base** model's lens so the coordinates
are fixed, in *linear* lens scores s_w(h) = v_w · h (before the final LayerNorm, so differences
add).

* **Finetune:** at " Ben" and at the prediction site, on the k0 items and on held-out stories where
  Ben appears with no temptation:
  Δ_ft = [s(finetune, Ben) − s(neutral twin, Ben)] − [the same for the other kind names].
  This is the double difference behind attachment and spillover.
* **In context** (ladder C, the C2 report's transplant arms: the same story twice, the villain's
  scenes both bad or both good): at the villain's later mentions and at the test's name,
  Δ_ic = [s(bad-bad) − s(good-good)] − [the same for the other child].
* **Reference profile:** P_class = s_base(villain names) − s_base(kind names) at the same
  positions, in the y = 1 base.

Statistics, per block:

1. **Alignment:** plot one point per token (x = P_class, y = Δ_ft) and fit a line through the
   origin. Its slope β is **the fraction of the villain profile that Ben acquired**, and the
   correlation says how much of his change lies along it. Because lens scores are linear,
   Δh ≈ p·(μ_villain − μ_kind) + … gives Δ_ft ≈ p·P_class + …. So β is a probe-free version of
   Ben's position p, and A4 predicts β ≈ his excess position from block 1 on (where the identity
   reports put the class's reading). The value is in the words, the residual (item 2), and any
   disagreement: β clearly below p would mean Ben moved mostly outside the J-space, which is the
   link to 3c. It uses the whole vocabulary, so there
   are no hand-picked words, and the split act words (cheat, lie, tease) do not matter.
2. **What is new beyond the class:** the top tokens of Δ_ft − β·P_class (predicted:
   steal-specific, the attachment part). The null is a permutation: Ben swapped with each other
   kind name.
3. **The logit lens on the same contrasts:** the J-lens earns its place only where it does better.

**Level 3, causal privilege** (Stage 3c, conditional): split the class axis into x_J and
remainder. Patch each at equal norm and at its natural size, with and without the clamp. Read out
the persona, attachment and spillover.

**Nulls:**

* **Null A, matched noise:** Gaussian draws with the mean and covariance of the real layer-ℓ
  activations, pooled over positions and, separately, at name positions, pushed through the same
  lens. It removes sparsity and keeps geometry.
* **Null B, untrained:** the same shape at update 0, rebuilt from the run's seed
  (`scripts/train.py` seeds before building the model; verified against the update-0 probe
  values in `probes.jsonl` before use).
* **Null C, randomized blocks:** the trained model with its block weights re-initialized and its
  embeddings kept (the jlens-lab sanity check). A lens that reads content here is reading the
  embeddings, not computation.
* **Logit-lens baseline, for every statistic:** a J-lens claim counts only where it beats the logit
  lens (J̄ = I). On GPT-2 124M and 355M the J-lens lost to the logit lens at next-token prediction
  (see Related work), so this is a real risk at our scale.
* **Distance null, for any layer-by-layer structure** (CKA blocks, "bands" drawn from similarity
  matrices): matrices that depend only on the layer distance |i − j| reproduced 79–91 % of the
  published three-band structure (jlens-lab). Our band rule uses per-block statistics against
  Nulls A–C rather than similarity blocks, but any CKA figure is shown against this null.
* **Ceiling:** the final layer, where the lens is the model's own next-token prediction.
* **Layer 0** is reported as a reference, not evidence: with tied embeddings it largely reads the
  current token. Tied embeddings are also the confound raised in the GPT-2 replication.

## The band rule (FROZEN 2026-09-27, end of Stage 0, before any blind output existed)

Tuned on the y = 0.99 finals only (10M seeds 0–3, 100M seeds 0–1; the evidence is in the Stage 0 results
below). Computed by `scripts/jlens.py stats` on held-out text (windows from the second half of the val
split; 2,000 positions per block), for the J-lens:

* block ℓ is **in the band** if both hold:
  1. **peaked beyond noise:** the fraction of positions whose readout kurtosis (tokens centered)
     exceeds the 99th percentile of Null A's (`kurtosis.centered.exceedance`) is ≥ 0.10, i.e. ≥ 10×
     chance;
  2. **not motor:** the J-lens top-1 token's agreement with the model's own next-token prediction
     (`next_token_top1`) is < 0.5 × the final block's;
* the band is the contiguous run of such blocks (depth also reported as ℓ / L); if they are not
  contiguous, the longest run, with the others listed;
* a model **has a band** if at least one block qualifies. A cell (corpus × size, or checkpoint) is
  reported as "k of n seeds have a band" with a 95 % (Wilson) interval, and is said to **have a band**
  only if the interval's lower bound exceeds one half;
* **onset** over pretraining is the first checkpoint at which the model has a band, with the peak
  exceedance over blocks as the continuous companion curve;
* every per-model statistic is also reported continuously, with seed-bootstrap intervals.

On the tuning models: blocks 1–3 of 6 at 10M (all four seeds); blocks 1–7 of 14 at 100M (both seeds;
block 8 sits exactly at 0.50 in seed 0 and is excluded). The untrained models (peakedness ≤ 0.05) and
randomized blocks (next-token agreement ≥ 0.92, the embedding echo) have no band.

Changes from the draft rule: the "in both halves of the stories" clause is dropped (the lens is already
fitted on one half and evaluated on the other); the thresholds are X = 0.10 and Y = 0.5.

Content, capacity and persistence are reported alongside, **not in the rule**:
* **Class content:** the villain-minus-kind profile over the whole vocabulary (name tokens excluded) at
  name positions, in linear lens scores, sized against a label-permutation null
  (`class_profile`: (norm − null mean) / null sd).
* **Capacity:** active tokens per position (centered score > 4 noise sd against Null A, own input token
  excluded).
* Pursuit occupancy is recorded but uninformative at our scale (≈ 1 everywhere, trained or not).

"No band" is a possible outcome at any size, and is reported as such. **Every statement is "at this
training budget"** (one epoch, ≈ 409M tokens, the same at every size). If a statistic is still rising
at the final checkpoint (Stage 2), it is flagged "not settled; might appear with more training" rather
than reported as absent.

## Stage 0 results (2026-09-27; tuning data only)

* **Convergence:** the per-block statistics are identical from 50 windows on at 10M and 100M, although
  the lens matrix itself still has a few percent split-half noise at 400 windows. Box: 100 windows per
  checkpoint, 200 where the lens is kept.
* **Window checks (10M):** 128- vs 512-token windows change nothing material. Averaging rather than
  summing over later positions raises next-token agreement and weakens class content (the reference
  estimator is kept).
* **Cost (Mac, MPS):** ≈ 2 s per window at 10M, ≈ 23 s at 100M (32 output coordinates per pass; 128
  exhausts memory at 100M).
* **Structure:** learned and consistent across seeds (above). The band is wider in relative depth at
  100M than at 10M, and next-token agreement stays low through more of it.
* **Class content:** readable above both nulls in most tuning models, but the J-lens's advantage over
  the logit lens is modest and **seed-dependent** (100M seed 0: J-lens +2.4 to +3.7σ at blocks 5–7 vs
  logit ≈ +1; 100M seed 1: J-lens weak in the band, logit +2.5 to +5.4σ). A hand-picked word index was
  tried first and retracted (from −2.8σ to +4.4σ across 10M seeds; "sorry" misassigned).
* **Capacity:** ≈ 1 active token per position in the band at 10M, ≈ 1–2 at 100M, against ≈ 0.1 for the
  nulls; the paper reports ≈ 25 in large models.
* **Go / no-go: go.** The structure clearly separates from the nulls, and the J-lens does modestly
  better than the logit lens on peakedness in the band and, in most seeds, on content.
* **Process note:** one blind model (0.2M, y = 1, seed 0) was looked at once while testing the pipeline,
  before this freeze. It was not used for tuning.

## Stages

**Stage 0: build, validate, tune (Mac, ≈ 1–2 days).**

1. `GPT.residual_stream` / `GPT.unembed` (a checkpoint's logits unchanged bit for bit; the test
   suite passes), then `dramatis/jlens.py` (structure below) and `tests/test_jlens.py`: on a tiny model, J̄ against
   finite differences; J̄_L = I; the lens at the last layer against the model's logits; the
   decomposition on planted sparse vectors.
2. Rebuild the val splits of lexical y = 1, 0.95 and 0, checked against the recorded sha256.
3. **Window checks** (on one or two tuning models): (a) 128 against 512 tokens, to connect to the
   paper; (b) summed against averaged effects over later positions, to see whether the
   early-source weighting matters. If the band statistics barely move, a footnote; if they move a
   lot, settle it before freezing the rule.
4. **Cost and convergence:** how many windows J̄ needs, from the running mean's relative shift
   and from the change in the band statistics. jlens-lab found that ≈ 100 prompts under-fit by
   4.6–6× on 1.7B–32B models and that 461–775 were needed. We fit until converged rather than to
   a fixed count. This sets the cost of everything after. Rough, before measuring: ⌈d / b⌉
   backward passes per window (b output coordinates per pass, replicated in the batch) give J̄ at
   every layer at once; ≈ 10 min per 10M checkpoint and ≈ 1–2 h per 100M checkpoint for ≈ 800
   windows.
5. **Tuning set:** the lexical y = 0.99 finals, seeds 0–3 at 10M and 0–1 at 100M (a different
   corpus from the blind set, so every y = 1 and y = 0 seed stays blind). Per-block statistics with
   nulls, content readouts, logit vs J lens. Fix X, Y and Δ; write them into this document;
   commit.

**Go / no-go:** stop with a short report ("no J-space structure at 10M–100M under this lens"),
which is itself an answer to their scale question, if either of these holds:

* no block of any tuning model separates from Null A (exceedance near 1 %) and class content is
  absent at names;
* the J-lens does no better than the logit lens on both structure and content.

Otherwise continue.

**Stage 1: scale, blind (Mac, ≈ 1–2 nights).** y = 1 with 8 seeds at every size (4 at 100M if needed)
and y = 0 with 4. The lens-fit cost is a guess until Stage 0 measures it: ≈ 2 min per model at
0.2M–1M, 4 at 3M, 10 at 10M, 30 at 30M, 90 at 100M, so ≈ 19 h for y = 1 with 8 seeds everywhere
(≈ 13 h with 4 at 100M). These lenses are reused in 3a, which therefore also gets 8 seeds (the R13
finetunes exist for every seed). The frozen rule on the blind scale set: which sizes and
have a band, where it sits, how peaked it is, and whether class content is readable in it. The
figures: exceedance against relative depth, one line per size; and occupancy against size (raw,
per d, and over Null A) at each model's band blocks, with seed intervals.

**Stage 2: formation and maturation (Mac; subsampled).**

* **Maturation:** seeds 0–1 at 4 checkpoints per run (41M, 82M, 164M, final tokens) at all nine
  sizes, y = 1 and y = 0. Only these two seeds have checkpoints, so these curves are descriptive;
  more seeds would need bit-identical retrains on a box (batched with the follow-up below).
  Occupancy comes along at no extra cost (the lens is already fitted), so capacity over pretraining
  is charted next to the band statistics.
* **Formation against a capability switch:** the dense ladder-C 30M runs. Four seeds (two early
  switchers, one late, seed 3 which never switches), a checkpoint every ≈ 500 updates plus every
  100 within ± 1,500 of the switch. Does the band appear before, with, or after the switch, and
  does seed 3 have one? Content here is the in-context history at the test name, not class.
* Formation *timing* gets its statistics here (8 seeds exist; 4 planned). It is also the costly
  part: tens of checkpoints per seed at 30M, possibly 40+ h on the Mac for 4 seeds. If Stage 0's
  cost holds, these lens fits go on a rented GPU in the same batch as the dense-early retrains,
  rather than taking weeks of Mac time.

**Stage 3: the study aim (Mac).** Mostly forward passes through lenses already fitted in Stages
0–1, so cheap. The exception is the ladder-C lens fits for 3b.

* **3a. What the finetune writes onto Ben** (Level 2b, Δ_ft). The R13 "Ben steals" finetunes
  against their neutral twins, y = 1, on every base with a fitted lens (all 8 seeds at every
  size, 4 at 100M if Stage 1 used 4). Outputs:
  * β by block and size, against Ben's measured position (A4) and against c(size) from the
    susceptibility report;
  * the top tokens of the residual.

  Two doses from `tl2` (10M and 100M, all 8 seeds, checkpoints every 5 finetune steps and
  log-spaced through cleaning): β against finetune steps ("what creeps in"), and β through
  cleaning. Is the dormant trace (15 % of Ben's shift left at his name after 2,000 clean steps at
  10M) still readable in the lens once behaviour is gone? **Question:** does Ben become readable as
  a villain in the model's own vocabulary, in proportion to how far he moved, and from which block?
* **3b. What history writes onto a name, in context** (Level 2b, Δ_ic). Ladder-C bases need their
  own lenses: 10M × 8 seeds, 30M × 8, 100M × 2–3. The C2 report's staged result makes a sharp test:
  * history reaches the child's later mentions in every model;
  * it is delivered to the test's name in the switched-on models *and* in 10M seeds 1, 6, 7;
  * it is read out into the decision only in the switched-on models.

  **Question:** at the test's name, is the delivered history readable in the lens only where it is
  used (switched-on), or also in the delivered-but-unused 10M seeds? The first would match the
  paper's account that content enters the workspace when a task needs it. The second would say
  J-space presence is not enough and the readout head is the missing piece. Either answer is
  informative. Also: which tokens (the act performed, other acts as an in-context analogue of
  spillover, or evaluative words), and its growth from k = 1 to k = 2 on the rung-C items. On the
  dense 30M checkpoints (Stage 2): does lens-readability at the test name appear with delivery or
  with the switch?
* **3c. Causal privilege, only if 3a shows alignment** (Level 3), at 10M and 100M:
  * the class axis split into x_J and remainder;
  * whether spillover (one variable read by several act circuits: their "flexible
    generalization") routes through x_J more than attachment does;
  * the J-space share of the class axis against c(size).

**Caveats for 3a and 3b:**

* **If β simply equals Ben's position**, the lens re-describes A4. Its added value is being
  probe-free and in words, plus the residual. That is a panel or a clause, not a section, unless it
  disagrees with A4.
* **Base lens vs the finetune's own lens:** the base lens is primary (fixed coordinates, so it
  measures a change of representation). The finetune's own lens is a check; it also moves with the
  finetune's changes downstream.
* **Ladder C has no labels**, so evaluative words may never be tied to names there. Check their
  frequency in the ladder-C corpus before reading anything into their absence.
* **Deferred:** the copying head as a J-space relay; the by-act cast as a discriminating test
  (Ben's shift should follow the steal class's profile, not a general villain one).

**Deliverable:** a report with (1) the scale and formation answer to their open question and (2)
the study-aim result, each with a main-text / appendix assessment.

## Intended follow-up: dense early checkpoints at every size ("back to the well")

The only piece the existing checkpoints cannot give is the lexical persona's formation window.

* **What:** retrain lexical y = 1 (and y = 0) at 0.2M–100M, seeds 0–1 (optionally 0–3), with
  log-spaced checkpoints through the formation window (updates 0, 50, 100, 200, 300, 500, 700,
  1k, 1.5k, 2k, 3k, 5k), then stop. The later checkpoints already exist.
* **Where:** on a box at the original commit the retrain reproduces the original bit for bit (as
  the 30M dense retrains did), so the new checkpoints attach to existing runs and all existing
  results stay valid. The Mac (MPS) would give new, non-identical trajectories: fine as a
  preview, not joined to the existing finals.
* **Cost:** ≈ 20 % of a run each (to ≈ 82M tokens). Storage ≈ 8 GB per 100M run for 20
  checkpoints. A few box-hours in all; on the Mac ≈ 25 min per 10M run, ≈ 1.7 h per 100M run.
* **Serves two threads:** the J-lens formation question (Stage 2 at every size) and the
  susceptibility report's next step 1 (does spillover = c × persona hold during formation?).
* **When:** only if Stage 0 shows structure worth timing. Added to the batched "second go-around"
  list rather than a separate rental. A Mac-only 1M–10M preview is the cheap alternative.

## Code: two methods on `GPT`, then `dramatis/jlens.py`

**Changes to `dramatis/model.py`** (Kevin's file; agreed 2026-09-27). Two small methods, and
`forward` uses them, so there is one loop and the lens can never drift from the real forward pass:

```python
def residual_stream(self, tokens) -> list[Tensor]:
    """h_0 (embeddings) … h_L (after the last block, before the final norm), each (B, T, D)."""

def unembed(self, x) -> Tensor:
    """Final norm and the tied output head: residual stream → logits."""

def forward(self, tokens):
    return self.unembed(self.residual_stream(tokens)[-1])
```

Training is unaffected (the same operations in the same order, and the list holds only tensors
autograd keeps anyway). This is verified bit for bit on a checkpoint and by the test suite. The
existing hook-based analysis scripts are unchanged. Interventions stay out of `model.py`.

**`dramatis/jlens.py`**: a small shared utility (no framework), plain PyTorch, used by one script
(`scripts/jlens.py`, stages as subcommands) and its figures. We go through it function by function
before writing it. Windows come from the existing `dramatis.data.get_batch` (random 512-token
slices, as in training; targets ignored). "Fit on one half, evaluate on the other" is two slices of
the val memmap passed to the same function, so no sampler of our own is needed.

```text
averaged_jacobians(gpt, windows, dims_per_pass=b) -> Tensor (L+1, d, d)
    J̄_ℓ for every layer at once, with the reference estimator, via gpt.residual_stream. For each
    window: one forward pass with the window replicated b times; for each block of b output
    coordinates, one backward pass with a one-hot cotangent on coordinate i (replica i) at every
    valid target position. At every h_ℓ(t) that gives the sum over t′ ≥ t of row i of
    ∂h_L(t′)/∂h_ℓ(t); causality restricts to t′ ≥ t automatically. Average over valid sources t
    (positions 16 … T−2) and windows. Returns the running mean, its relative-shift log
    (convergence), and the two half-means (split-half checks). Options for the Stage 0 checks:
    the window length, and summing or averaging over later positions.

readout(gpt, J_l, h) -> Tensor (..., V)
    gpt.unembed(h @ J_l.T). With J_l = identity it is the logit lens, gpt.unembed(h).

lens_vectors(gpt, J_l) -> Tensor (V, d)
    The linear version used for scores and decomposition: v_w = J̄_ℓᵀ (γ ⊙ W_U[w]), with the final
    norm's mean-centering projected out (γ = gpt.norm_final.weight, W_U = the tied embedding).

sparse_decompose(V_l, x, k) -> (indices, coefficients, x_J)
    Non-negative orthogonal matching pursuit over the lens vectors, up to k terms.

matched_noise(h_samples, n) -> Tensor (n, d)
    Null A: Gaussian draws with the samples' mean and covariance.

untrained_model(run_config) -> GPT
    Null B: the same two lines as scripts/train.py (seed, then GPT(config)); checked against the
    update-0 probe values.

randomized_blocks(gpt, seed) -> GPT
    Null C: a copy of gpt whose blocks are taken from a freshly initialized GPT (the model's own
    init, not a re-implementation); embeddings and final norm kept.

edit_forward(gpt, tokens, edits) -> logits
    Stage 3c only, and only if needed: forward hooks (the existing pattern in the analysis
    scripts) that apply edits to chosen layers (steer, ablate, replace the J part, clamp).
```

The statistics (kurtosis and exceedance, next-token accuracy, persistence, occupancy, content
ranks) live in the script, not the utility: they are analysis choices, not the method.

**Outputs and provenance:** J̄ per (run, checkpoint) saved to `logs/v4/jlens/jbar/<run>@<step>.pt`
with the stories' sha256, the number of positions, the git commit and the weighting. Statistics as
one JSON per (run, checkpoint). Figures in `docs/figures/jlens/`.

## Competing accounts, written before results

Not guesses: for each question, the accounts on the table and the outcome that would favor each.
The analysis choices that make these outcomes measurable (the band rule, the nulls, the tuning/blind
split) are frozen separately at the end of Stage 0.

1. **Scale (Stage 1).**
   * **No workspace-like structure at 0.2M–100M at this budget:** consistent with the GPT-2
     replication.
   * **A band from some size on.**

   Decided by the frozen rule, reported as k of n seeds per size, and by whether the J-lens beats
   the logit lens.
2. **Formation (Stage 2, dense ladder C).** Lens-readable history at the test's name appears:
   * with *delivery* (before the switch);
   * with the *switch* (with the readout head);
   * or not distinguishably from the nulls.

   The never-switching seed 3 separates the first two.
3. **The finetune onto Ben (3a).**
   * β ≈ Ben's excess position is *expected from A4 and the linearity of the lens*. It is a
     consistency check, not a finding.
   * **Informative outcomes:** β clearly below his position (he moved largely outside the
     J-space), or residual content beyond the taught act.
4. **In-context history (3b).** At the test's name, the history is lens-readable:
   * *only where it is used* (the switched-on models): content enters the workspace with use, as
     the paper argues;
   * *also in the delivered-but-unused 10M seeds 1, 6, 7*: being in the J-space is not enough, and
     the readout head is the missing piece.
5. **c(size) and the J-space (3c).** The class axis's J-space share either rises with c(size) (the
   size factor is about how "sayable" the class is), or does not (c is about something else, e.g.
   the reach of the finetune).

## Deferred (mentioned, not planned)

* **Averaging over less:** lenses conditioned on position type (names only), per-context
  Jacobians, and how much individual contexts differ from J̄ (a "lens R²"). This is taken up only
  if Stage 0's convergence check shows large context dependence, since then J̄ itself is in
  question.
* **Recasting earlier analyses in lens coordinates** (Kevin, 2026-09-27): the class axis, the flow
  scans, the identity routes, curing. Decided only after Stage 1, and only if the lens applies well
  at our scale.
* Multi-token concepts (cheat, lie, tease): their §A.9 extension.
* Their functional tests that need instruction following (report, introspection, directed
  modulation): not possible with story models.
* MLP-gain and attention-relay ("broadcast") statistics beyond the one head in Stage 3.

## Related work (search of 2026-09-27)

A web search (a subagent, then spot checks) found **no study of the J-lens across pretraining
checkpoints, none on models below ≈ 124M parameters or on TinyStories-like data, and none
relating the J-space to a persona or emergent-misalignment variable**. The small-model evidence
that exists is thin and partly negative. Items marked ✓ were opened and checked by us; the rest are
summaries from the search, to be read before citing.

**The source and its code**

* ✓ Gurnee, Sofroniew, Lindsey et al., transformer-circuits.pub/2026/workspace (July 2026):
  Claude models only. Their over-training comparison is base vs post-trained, not over
  pretraining. They find the logit lens captures "much of the workspace-like structure", less
  reliably early.
* ✓ `anthropics/jacobian-lens` (Apache-2.0, reference implementation, unmaintained): the estimator
  we copy (first 16 and last positions excluded, sum over targets, mean over sources). Their windows
  are 128 tokens, ours 512 (see the method section). Hugging Face models only.

**Direct follow-ups and replications**

* ✓ Bandularatne, "J-Lens: A Failed Replication on GPT-2", LessWrong, 2026-09-07. On GPT-2 small
  and medium the J-lens lost to the logit lens at next-token ranking at almost every layer (0/11,
  1/23). Caveats, the author's own: 30–150 sequences against the paper's ~1,000, and GPT-2's tied
  embeddings. Next-token ranking is also not the paper's workspace claim. **The closest evidence to
  our scale; it motivates the logit-lens baseline and the convergence check.**
* ✓ m9h (Morgan Hough), `jlens-lab` (GitHub). Three controls:
  * randomized blocks: the lens reads essentially nothing;
  * a logit-lens floor;
  * a *distance null*, which reproduces 79–91 % of the published sensory / workspace / motor block
    structure.

  It also finds ≈ 100 prompts under-fit by 4.6–6× (461–775 needed on 1.7B–32B models), and excess
  structure over the null that is modest and roughly doubles at ≥ 20B. It has no pretraining
  checkpoints. We adopt all three controls and the convergence rule.
* `cheeetoo/tiny-jlens` (the five workspace criteria on GPT-2 small; results not read);
  `tao-hpu/jspace-replication` (GPT-2 124M and Qwen3 1.7B–14B: multi-fact swaps replicated,
  rhyme planning and mental arithmetic not, "many false positives").
* The Neuronpedia J-lens blog (Lin and Chanin, 2026-07-10): readouts for 12 models, swaps
  "oversteer" in smaller models.
* A LessWrong engineering analysis (willkn, 2026-07-24): gpt2-medium needed shrinkage
  regularization; ≈ 6.7 GPU-hours on an L4 for 1,000 prompts.
* Arxiv follow-ups on large models only: looped transformers (Wang and Reid, 2609.01924); the
  sparsity of the Jacobian (Yan et al., 2608.25347); multi-token concepts (Gong and Wang,
  2608.31084).

**Formation over training and scale (not the J-lens)**

* ✓ Rahbar, "The Ignition Index", arXiv 2608.05160. A sigmoid-steepness index on per-layer
  linear-probe accuracy across 11 models: sharp "ignition" in feedforward transformers,
  near-linear profiles in Mamba, and the link to scale "not confirmed". The search reports Pythia
  checkpoint analyses in the body (not checked by us). It is the nearest template for our Stage 2
  question, with probes instead of the lens.
* Tigges et al., NeurIPS 2024 (2407.10827): circuits emerge at similar token counts across Pythia
  sizes. Bayazit et al., "Crosscoding Through Time" (2509.05291): features appearing and
  disappearing over pretraining.
* Earlier lenses: tuned lens (Belrose et al., 2303.08112); linear relation decoding by averaged
  Jacobians (Hernandez et al., 2308.09124); Future Lens (Pal et al., 2311.04897).

**Persona and emergent misalignment (no lens readouts)**

* Wang et al., persona features control EM (2506.19823); Chen et al., persona vectors
  (2507.21509); Nadaf, "Emergent Misalignment Recruits a Pre-existing Persona Subspace"
  (2607.21356), which leaves open whether the persona structure comes from pretraining. Our
  checkpoints speak to that directly.

**What this changes in the plan:**

* **The logit lens is a baseline everywhere, and the J-lens must beat it** (go/no-go).
* **Nulls A–C plus the distance null** for any similarity-block figure.
* **Fit to convergence, not to a fixed count.**
* **The scale question is open in both directions.** A null at 0.2M–100M would be consistent with
  the GPT-2 replication, and a positive result would be new.
