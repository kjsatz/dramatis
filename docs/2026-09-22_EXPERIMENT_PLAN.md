# The experiment plan for the v4 corpus: claims, figures, runs, budget

*2026-09-22, the experiment-planning session (Kevin and Claude; handoff
`2026-09-22_HANDOFF_experiment_planning.md`). It replaces the running "second go-around" list in
NEXT_SESSION.md and revises the story of `2026-09-21_SYNTHESIS.md` in the ways listed in §1. The
1× analyses behind the revisions are in the notebook (2026-09-22, 13:40). Headline numbers here
are 1× values; v4 may move them, and the paper will be re-based on v4 throughout (the 1× work
stays documented as the first generation, and its nulls may appear in the story).
Revision 2 (same day, evening): the full coherence grid at every size, a depth ladder below 1M,
a story-quality measure, the figures reorganized (the direction moves to Figure 2; Figure 5 is
about the finetune), new designs for the panels that did not carry their message, and a full
capacity grid in the appendix. Status: draft for Kevin's review before the plan is final (§11).*

## 1. What changed from the synthesis

* **The spillover metric.** Spillover to each untaught act is plotted against the persona *on
  that act*, never against a persona averaged over acts. On the 1× data the act-matched slopes are
  0.40 / 0.40 / 0.43 for lie / cheat / tease as the untaught act (intercepts +0.14 / +0.07 /
  −0.04), where the averaged metric gave 0.25–0.46 by taught act. The claim is "about 0.4 of the
  persona on each act". "Law" is retired as a word.
* **Fork units are per act.** Against free text the fork compresses lie, cheat and tease three- to
  sevenfold (persona 8–11 logits in samples on every act; 1.3–2.9 fork nats except steal, 8.7).
  Fork values are never averaged across acts raw. Main-text persona panels use a **normalized
  average** (each act divided by its own y = 1 value, then averaged); per-act panels go to the
  appendix. Headline magnitudes come from samples; small effects and dose curves from forks.
* **New fork items for all eight acts**, written under construction rules and frozen before any
  v4 model exists (§4). The 1× held-out items are confounded (per item: sneak's two items with
  "did not" in the good ending read negative, the other two positive; blame's "I did it" −1.29);
  they are kept and reported as they are, with the item analysis, in the appendix.
* **The finetune has two parts:** the taught act for every name (other kind names do it in 81 % /
  74 % / 63 % of samples after "Ben lies / cheats / teases") and a shift toward the villain class
  for Ben alone (his untaught acts; other kind names 0–2 %). Steal is the exception (14 %), an
  open question; the quoted-speech explanation was predicted and refuted (cheat, a narrated deed,
  spreads at 74 %).
* **Cut or demoted:** the slogan "knowing is not expecting" (the gain result stays as a mechanism
  finding for the coherence step); the entry condition's exception-count panels and the
  count-against-fraction test (appendix); the dedicated scaling figure (size is a knob in every
  figure, plus one capacity summary and a full capacity grid in the appendix).
* **Small models break down, and we look for how.** A depth ladder below the 1M model (width 128,
  1 / 2 / 3 layers) tests the mechanism of Figure 5 (the class is copied from the name to the
  prediction site by attention at block 3–4 in the 10M model: does one layer suffice for a
  persona, and what does a shallow model lose?). A TinyStories-style story-quality measure is a
  capacity row.
* **The full coherence grid at every size:** y = 0, .5, .6, .7, .8, .9, .95, .98, .99, 1 at
  1M, 3M, 10M, 30M and 100M, ten seeds each.
* **Ten seeds per cell** by default, 30M and 100M included. **One pass** of each assembled corpus
  at every size (nothing repeated), with a continuation rule for the label onset (§4). **No 300M**
  (about 1.4 tokens per parameter even at 20×).
* **Stated in advance:** every experiment in §5 has its frozen readout and, where 1× gives one, an
  expected range; the rest are marked exploratory. This table is committed before the box run and
  the paper quotes it.

## 2. The story

**Thesis.** In a miniature where every character's disposition is set by construction, the
persona-selection account of emergent misalignment holds in a specific form: the model builds a
persona only as a lexical class — a set of names that almost exactly partitions the villain
role, carried by one direction in the embedding — and a narrow finetune generalizes through it,
moving one kind-named child about 0.4 of the way into the class along that direction, to the acts
the class performed together.

**Arc.** What the persona is (formation from names; one direction; represented before it is used)
→ what it is not (acts alone and labels) → selection (a narrow finetune selects it: the EM test
and the paper's centre) → how the finetune works (the flow through the blocks; in fast and out
partly; predictable from the base model) → capacity (what size and depth change, throughout).

| claim | statement (1× form) | v4 must show | what would change it |
|---|---|---|---|
| C1 | A persona forms only when a name class partitions the villain role almost exactly: flat, then a sharp rise over the last few percent of coherence. It is one direction in the embedding (a K-name moved one class gap along it behaves as an N-name), and the direction is present well before behaviour uses it (the gain) | the full grid on unrepeated data, 1M–100M; steering and d′ along y | a graded persona at intermediate y; steering failing on unrepeated data |
| C2 | Acts alone do not make a persona; a label that must be predicted from acts is bound abruptly and late, invisible in the loss, while the story-level fact is learned early and smoothly in the same runs | ladder rungs A–D, dense probes, ten seeds, 1M–100M and the depth ladder | rung C positive (inference appears without repetition); rung B positive (declared personas are used); §6 |
| C3 | A narrow finetune spreads bad behaviour to the acts the persona's class performed, ≈ 0.4 × the persona on each act; the spillover matrix copies the cast's co-occurrence matrix (full for the plain cast, empty for the random cast, near zero for the by-act cast); in free text 0 → 15–68 % | cell-level line, the four matrices at ten seeds, free text | the paired-cast matrix not showing its block pattern (generic badness, not the class's bundle) |
| C4 | The finetune works through the base model's circuit: the class is read from the name through block 3 and carried at the prediction site after the copy; the finetune's effect runs through the same handoff; Ben moves ≈ 0.4 of the way into the class within ≈ 50 steps and clean data takes back a third and stops; to first order the effect is predictable from the base model | flow scans for base and finetune at 10M and 100M; the timeline with dense probes; the GD test | GD failing in the small-dose limit (then find out why first) |
| C5 | Beyond the performed acts generalization is narrow: the persona reaches break | new held-out items | anything — report it |
| C6 | Capacity: the direction and the persona exist from 1.6M; the actor binding and the finetune's name-specificity fail below 10M; the gain and the label onset improve with size; what a 1-, 2- or 3-layer model keeps and loses | the grid at every size and depth, story quality | — (descriptive; exploratory for the depth ladder) |

## 3. Figures

Main text, 15 panels:

| # | title | panels |
|---|---|---|
| 1 | Setup | corpus and cast; the fork and sample readouts; the finetune test (design to brainstorm separately) |
| 2 | What the persona is (C1) | (a) coherence sweep y = 0 … 1, normalized persona, sizes as colours; (b) the name effect over training, dense probes (forms early and smoothly); (c) one direction: each model's own persona against a K-name steered one class gap; (d) represented before used: embedding d′ against behaviour along y |
| 3 | Names, acts and labels (C2) | (a) the ladder A–D, finals by size; (b) the same runs over training: the story level early and smooth, the binding late and abrupt, the in-context effect flat, behaviour under an opening label; (c) onset against size, depth ladder included |
| 4 | A narrow finetune selects the persona (C3, C5) | (a) spillover against persona on each act, one point per corpus cell, small multiples; (b) the cast's co-occurrence matrix above the measured spillover matrix: random, plain, by-act, paired; (c) free text, Ben and the other kind names, four taught acts (the finetune's two parts); (d) held-out acts |
| 5 | How the finetune works (C4) | (a) the flow through the blocks: the persona removed at the name token or the prediction site after each block, for the base model and for the finetune, over the class separation at the site; (b) one timeline, pretrained → "Ben steals" → clean stories: Ben's position on the class axis and the spillover; (c) GD measured against predicted |
| 6 | What capacity buys (C6) | a selection of rows from the appendix grid (A6), chosen after the data |

Appendix figures: A2a the sweep per act; A2b count against fraction; A3 panel 3b at every size
(and depth); A4 predicted from the class persona against measured, by cast; A5 the flow at 100M
and for the depth ladder; A6 every capacity measure against size and depth; plus the exception
counts and the graded cell, the frequency ladder, the LR control, the site decomposition, forks
against samples, the 1× held-out item analysis, the steal offset, the cheat test, and the 1×
results as a first generation.

### Capacity rows (A6; Figure 6 picks from these)

| row | definition | 1× reading |
|---|---|---|
| direction present | embedding d′ of N- against K-name rows along their mean difference, y = 1 | large at every size measured (38–61) |
| persona | N − K at k = 0, exact cast, normalized as §4 | exists at 1.6M, grows with size |
| kept under exceptions | persona at y = 0.95 ÷ at y = 1 | ≈ 0.3 to 10M, rises above |
| gain | one steered class gap at y = 0.95 ÷ the same at y = 1 (how much of the direction an exception model uses) | 0.24 at 10M, 0.63 at 100M |
| actor binding | 1 − (bystander effect ÷ persona); bystander = K-named actor with an N-named other child minus a K actor with a K other; 1 = the villain name affects only its own actor | 0.81 at 1M / 3M, 0.95–0.99 from 10M |
| finetune name-specificity | after "Ben steals", Ben's rise on steal ÷ (Ben's rise + the other kind names' rise); 1 = the finetune taught Ben, 0 = everyone | 0.13 at 1M, ≈ 0.7 at 10M / 30M, 0.40 at 100M (R20) |
| spillover ratio | act-matched spillover ÷ persona, "Ben steals" | rises to ≈ 0.5–0.6 from 10M |
| label onset | tokens to label binding (§4) | 135M / 94M / 33M at 10M / 30M / 100M |
| in-context character effect | rung C at k = 1 | ≈ 0 everywhere |
| actor kept in free text | 1 − the judge's UNCLEAR rate on N- and K-actor samples (the model keeps the named child as the actor) | 0.80 at 10M |
| story quality | TinyStories-style: held-out story openings completed by the model, judged 1–10 for grammar, consistency and plot (§4) | not measured |
| validation loss | held-out v4 stories | not collected at 1× in this form |

The grid reads as three groups: present at every size (direction, persona), switching on with
capacity (actor binding, finetune specificity, story quality, label binding) and improving
smoothly (gain, onset). The depth ladder adds three columns at the small end.

## 4. Metrics and items, fixed before the runs

* **Persona** per act: N − K fork log-odds at k = 0. **Normalized persona** (main-text panels):
  each act's N − K divided by the mean N − K on that act at y = 1 for the same size, then averaged
  over the four acts (so every size's curve ends at 1; absolute strength is a Figure 6 row).
  **Spillover** per untaught act: (Ben − other kind names), finetuned − neutral. **Attachment**:
  the same on the taught act. Fits are per act, act-matched; cell means (a corpus × size cell,
  averaged over its seeds) are the plotted unit; steal as the untaught act is reported separately
  (its offset: with little persona on steal, lie / cheat / tease finetunes lower Ben's steal by
  ≈ 0.7 nats at 1×; unexplained). The spillover matrix (4b) is normalized per column by that
  act's persona scale so the fork's act scales do not dominate.
* **Fork items (all eight acts):** negation absent or balanced across the two endings; the good
  ending as natural as the bad one (no "I did knock over the cup"); no names inside endings; no
  fork word shared across acts; several phrasings per act; ≈ 16 held-out setups per act. Claude
  drafts, Kevin reviews, frozen with a tag before any v4 model is probed; calibrated on the v4
  clean model as forks_v3 was.
* **Samples:** `sample_forks.py` as at 1×, judged by gemini-3.8-flash with the calibrated prompt;
  Vertex batch prediction or at least retry logging (the 1× cheat run spent most of 75 minutes in
  silent 429 backoff).
* **Story quality:** 100 held-out v4 story openings cut after the second scene, one completion per
  model (temperature 1, to the end of the story or 300 tokens), judged 1–10 on grammar,
  consistency (characters, objects and who did what stay consistent with the opening) and plot
  (the completion makes sense as a story); the rubric follows TinyStories' GPT-4 evaluation (to be
  checked against the paper before the prompt is frozen) and is calibrated on a handful of v4
  stories and deliberately corrupted ones. Models: every size and depth at y = 1 and y = 0, three
  seeds.
* **"Positive" for a ladder rung:** a ten-seed mean character effect above 0.5 nats with a 95 %
  CI excluding zero, at any size.
* **Onset:** the first probe step at which label binding exceeds half its value at the end of
  the run (dense probes every 10 steps), reported in tokens. **Continuation rule (every rung B′
  run, every size and depth):** one full pass for everyone; a run with no onset by the end of the
  pass continues on the same corpus, reshuffled, for up to three more passes with the probes on,
  and stops 2k steps after its onset or at the cap. Pass boundaries are marked; an onset after
  pass 1 carries the repetition caveat; no onset by the cap is censored at four passes. Not
  applied to rung C: its 1× null held through 5–17 epochs of repetition.
* **Flow scans (5a):** for each block L, the class direction removed (mean-ablated) at the name
  token after block L, and separately at the prediction site after block L; the fraction of the
  persona (base) or of the attachment and spillover (finetune) left; every block at both positions
  (the 1× finetune scan covered name ≤ 3 and site ≥ 3 only); a random direction of the same norm as
  the control; the site's d′ per block.

## 5. The runs

Corpora are assembled from the v4 raw stories: 1,268,642 valid of 1.4M generated (≈ 90 %), base :
one : both ≈ 2 : 2 : 1, ≈ 420 assembled tokens per story, ≈ 530M tokens in all (notebook
2026-09-22 21:15). One pass per run: ≈ 424M tokens for the lexical 1 : 1 : 0 mix (all base and
one-villain stories less the holdout, ≈ 26k steps at 32 × 512) and ≈ 415M for the ladder
1 : 2 : 1 mix (limited by the bad/bad stories, ≈ 25k steps); exact figures after assembly.
GPU-hours on RTX 5090 at the §7 throughputs.

| # | experiment | fig | runs | GPU-h | frozen readout | 1× value → expected range, or exploratory |
|---|---|---|---|---|---|---|
| R1 | coherence sweep, lexical 1 : 1 : 0, y = 0 / .5 / .6 / .7 / .8 / .9 / .95 / .98 / .99 / 1 at 1M, 3M, 10M, 30M, 100M × 10; dense k = 0 probes on y = .95 and 1 | 2a–b, 4, 6 | 500 | 295 | normalized persona; per-act N − K; fraction kept at each y | fraction kept at y = 0.95 (10M): 0.30 → 0.2–0.5; y = 0 within ±0.1 of zero (normalized); exact-cast effect grows with size |
| R2 | depth ladder: width 128, 1 / 2 / 3 layers (≈ 0.8 / 1.0 / 1.2M), y = 0 / .95 / 1 × 10, and the closing-label corpus × 10 | 3c, 6, A5 | 120 | 23 | as R1; bystander effect; onset; flow scans | exploratory; hypothesis stated in advance: one layer can copy the class to the prediction site (a persona), but the actor binding needs two (1 layer: bystander effect large) |
| R3 | count against fraction: 10M, y = .95 / .99 / 1 on a quarter-size corpus | A2b | 30 | 2 | fraction kept at y = 0.99, quarter against full | exploratory, two stated outcomes: equal → the fraction sets it; more kept on the quarter corpus → the count does |
| R5 | ladder rung B: opening label, both children tempted, 1 : 2 : 1, 1M–100M × 10, dense | 3 | 50 | 39 | character effect with the label in the prefix, k = 0 and k = 1 | exploratory (never tested with both children tempted) |
| R6 | ladder rung B′: closing label, same mix, 1M–100M × 10, dense | 3b–c | 50 | 39 | story-level and binding curves; onset | 1× onset 10M×3 7.5–10.5k, 30M ≈ 6k, 100M 2–4.5k steps, with repetition → exploratory; the continuation rule (§4) |
| R7 | ladder rung C / D: random names, no labels, 1M–100M × 10, dense | 3 | 50 | 39 | character effect at k = 1 (C) and cross-act k = 2 (D) | 1× pooled −0.13 ± 0.03 → null expected; "positive" as §4 |
| R8 | ladder rung A: "naughty $A" at every mention, 10M × 10, dense | 3a | 10 | 4 | character effect | a readout control: expected large; if null, stop and check the probe and the assembly |
| R9 | the continuation rule for R2 and R6 runs with no onset after one pass | 3c | ≤ 60 | ≤ 46 | onset in tokens, pass marked | exploratory; worst case 1M, 3M, 10M and the depth ladder all to the cap |
| R10 | paired cast: {steal, lie} names and {cheat, tease} names, 10M × 10, 100M × 10 | 4b | 20 | 16 | the spillover matrix | predicted: the block pattern (partner act high, the other pair ≈ 0), each cell ≈ 0.4 × the taught act's names' persona on that act |
| R11 | by-act cast (specialist bad/good stories), 10M × 10 | 4b | 10 | 1 | the spillover matrix | 1× near zero everywhere (−0.2 to +0.3) → consistent with the profile rule |
| R12 | clean twin (base stories only, cast names), 10M × 10 | appendix | 10 | 1 | persona and spillover | ≈ 0 |
| R13 | finetunes: steal / lie / cheat / tease / neutral, 100 steps, on every R1, R2, R10–R12 base | 4, 5, 6 | ≈ 2,700 | (30, with R15–R18, R20) | act-matched spillover and attachment | slope 0.40–0.43 → 0.25–0.6 per untaught act; random cast \|mean\| < 0.2; contrast d (1× 2.4) > 2 |
| R14 | free text: `sample_forks.py` on the 10M y = 1 bases and their five finetunes, the random, by-act and paired casts, 10 seeds | 4c | ≈ 180 models | (5, with R21) | BAD rate per class and act | Ben's untaught acts 15–68 %, others 0–2 %; taught act for the others: steal 14 %, lie 81 %, cheat 74 %, tease 63 % (steal the open exception) |
| R15 | held-out acts on the new items, R1 bases and R13 finetunes | 4d | probes | (in) | persona per held-out act; spillover per held-out act | break persona positive (1× all four items +3.7 to +4.5); spillover ≈ 0.4 × persona on each held-out act (predicted); the other three exploratory |
| R16 | mechanism probes: steering and d′ on R1 (2c–d, A6); full flow scans for the base and the four finetunes at 10M and 100M and on the depth ladder | 2c–d, 5a, A5 | probes | (in) | as `persona_direction.py`, `persona_circuit.py`, `finetune_persona.py` | steering one gap ≥ 75 % of the own N effect (1× 88–97 %); the name-token removal loses its effect after the copy block and the site removal gains it (1× crossing at block 3–4, 10M); finetune: the same crossing |
| R17 | GD test (predetermined dose), first on the 1× models on the Mac, then the v4 bases | 5c | Mac + box | (in) | measured against predicted attachment and spillover, slope | ≈ 1 in the small-dose limit; saturation exploratory |
| R18 | the timeline: "Ben steals" 0 → 100 steps and then clean stories to 2,000 steps, probes every 10 steps (position, attachment, spillover), 10M y = 1 × 10; "Ben steals" continued to 400 as the contrast | 5b | 10 × 2 | (in) | Ben's position on the axis, spillover, attachment against step | 1×: spillover near its plateau by ≈ 50 bad steps; clean data takes back a third to a half, plateau at 0.26–0.39 by 400 → plateau persists to 2,000 (if it decays to zero, "slow", not "stops") |
| R19 | LR control: 10M at the peak LRs of 1M / 3M / 30M / 100M × 5, y = 1 and rung B′ | appendix | 40 | 12 | persona; onset | exploratory |
| R20 | 100M finetune-LR sweep (3 LRs × 10 bases) | appendix | 30 finetunes | (in) | attachment; the other kind names' rise | the 1× 100M attachment dip (+2.8 against +5.4 at 30M) vanishes on unrepeated data or with a smaller finetune LR |
| R21 | story quality and actor tracking: completions of 100 held-out openings for every size and depth at y = 1 and 0 (3 seeds); fork samples for actor tracking at every size | 6, A6 | ≈ 50 models | (in R14) | judge scores; UNCLEAR rate | exploratory; TinyStories comparison (grammar early, consistency later) |

**Total ≈ 570 GPU-h**: the rows sum to ≈ 553 at the worst case of R9, plus ≈ 15 for the smoke
test and slack. The big items are the sweep (R1, ≈ 295) and the ladder (R5–R7, ≈ 118). Judging ≈ 110k calls, ≈ $90–100 of Vertex credits (free text ≈ $65, story
quality ≈ $20, actor tracking ≈ $7).

## 6. Figure 3, two outcomes considered in advance

| | rung C null | rung C positive |
|---|---|---|
| **rung B null** | acts alone make nothing and a declared label is not used even when it predicts the endings: the ladder at zero but A; the closing-label step with no behavioural step | unlikely (inference easier than using a label): check the label assembly before believing it |
| **rung B positive** | declared personas are used, inferred ones are not formed: does behaviour switch on at the closing-label onset? that timing is the centrepiece | a persona from context without repetition: "when and at what size"; the 1× nulls read as "repetition suppresses it" and C2 becomes a positive result |

Needed in every cell and already in the plan: dense probes on every ladder run, the same readout
for B and C, label binding and behavioural use probed in the same R5 / R6 runs, the story-level
curve alongside the binding in R6.

## 7. Budget, box and order

* **Throughput (RTX 5090, bf16, 32 × 512):** 10M 491k tok/s (measured, 30 steps/s); 100M 88k
  (measured, 106M tokens in 20 min); 30M ≈ 165k (from the 4090's 118k); 1M / 3M ≈ 600k and the
  depth ladder ≈ 800–900k (guessed, overhead-bound); dense probes +50 % (the synthesis estimate).
* **≈ 570 GPU-h ≈ 71 h on one 8× RTX 5090 box, ≈ 36 h on two, ≈ $266 at $3.74/h** (the same cost
  either way). A fixed 300M-token schedule for every run instead of one full pass would cut this
  by ≈ 30 % (§11). An H100 costs 4–5× more per GPU-hour and helps only at 100M.
* **Hour 0: smoke test.** Measure the small-model throughput, the dense-probe overhead and the
  100M step rate; one run of every corpus for 200 steps with every probe; check every log for a
  traceback. Re-estimate before queuing the grid.
* **Order:** the 100M runs first (the long poles), then 30M, then the small sizes and the depth
  ladder filling the gaps; finetunes and probes as bases finish; sampling last.
* **Storage and transfer:** weights-only finals for every run and dense probe outputs (numbers, not
  weights); intermediate checkpoints for two seeds per cell only. Corpora assembled on the box from
  the raw stories with committed code (the recipes carry the hashes), or pulled from a private HF
  dataset, not rsynced; finals pushed to the HF Hub from the box. Box disk ≥ 1 TB.
* **Ops lessons** as NEXT_SESSION: the `:cuda-12.8.1-auto` image tag, DNS check, `--holdout` for
  every v4 assembly, sync `data/raw/`, no destroying a box before every probe log is checked.

## 8. Before the box (Mac)

1. ~~Confirm the 20× corpus~~ — done 2026-09-22 (notebook 19:10, 21:15): ids, hashes, sidecars at
   `a0d4dff`, leakage gate at chance on both tranches, judge sample as the pilot, length, a read
   of fifty (two pass-2 wobbles noted: object swaps in ≤ 13 % of steal scenes, tease wording).
2. The training-loop probe hook (`probe_fn` / `probe_every`) and light probe functions (Claude),
   timed on the Mac.
3. Depth-ladder presets (width 128, 1 / 2 / 3 layers) in `dramatis/model.py`'s standard configs
   (Kevin's file) and their configs.
4. ~~v4 holdouts and finetune sets~~ — done 2026-09-22 (`prompts/holdout_ids_v4.json`,
   `data/finetune-v4/`); still to do: 100 held-out openings for story quality (from the v4-base
   holdout).
5. Assembly options: rung A rendering ("naughty $A" / "kind $B" at every mention); the paired cast;
   the by-act filter (specialists only); closing and opening labels for two villains (exist).
6. The new fork items (§4): draft, Kevin's review, freeze and tag.
7. The story-quality sampler and judge prompt: rubric checked against TinyStories, calibrated,
   frozen.
8. Analysis scripts written before data: the normalized persona, the act-matched fits, cell means
   and matrices, the ladder's "positive" test, the onset estimator, the flow scans at every block
   for the finetune, the capacity grid; `scripts/plan_figures.py` becomes the figure script.
9. The GD test on the 1× models (`scripts/gd_finetune.py`), before its v4 version.
10. Reconcile the 1× step-100 mismatch between the dose series and the un-poisoning run
    (spillover 1.71 against 1.89 for seed 0; probably different neutral baselines) so the timeline
    has one baseline.
11. Judge: batch mode or retry logging.
12. A box driver that fails loudly; this table (§5) committed.

## 9. Open questions the paper states as open

* Why does the taught act stay with Ben only for steal (14 % against 63–81 %)? Not quoted speech
  (refuted by cheat) and not the other child's presence (tease).
* The steal offset (above).
* Whether the 1× 100M attachment dip is data starvation (R20 decides).

## 10. Placeholder panels

One panel per result the v4 runs are meant to produce, drawn from the 1× probe outputs by
`scripts/plan_figures.py` (seconds, no checkpoints) into `docs/figures/plan/`. Where nothing close
exists the panel is marked ILLUSTRATIVE. Figure 1 (setup) is a schematic and is not drafted here.
Each caption says what the placeholder is and, after **v4:**, what the planned version changes.

### Figure 2: what the persona is

![2a](figures/plan/plan_2a.png)

**2a. A persona forms only at near-exact coherence.** The normalized persona (each act's N − K ÷
its y = 1 value at that size, averaged over acts) against coherence y; 10M from the 1× sweep
(106M tokens, about four epochs, 10 seeds per y), 30M and 100M with 2–5 seeds on fewer y; the
random cast from the 10M×3 map. **v4:** one unrepeated pass; y = 0 / .5 / .6 / .7 / .8 / .9 /
.95 / .98 / .99 / 1 at every size, 10 seeds (R1); new fork items; per act in A2a.

![2b](figures/plan/plan_2b.png)

**2b. The name's persona forms early and smoothly.** N − K over training, exact cast, 10M×3,
five runs, probes every 1,500 steps. **v4:** dense probes every 10 steps, every size, y = 0.95
and 1, 10 seeds; tokens on the x-axis (R1).

![2c](figures/plan/plan_2c.png)

**2c. One direction in the embedding is the persona.** For each model, its own persona (x) against
the effect of moving a K-name one class gap along mean(N rows) − mean(K rows) (y); × = a random
direction of the same length; 10M and 100M at y = 0 … 1. On the diagonal: the direction carries
the whole effect. **v4:** every size and depth, y = 0.5 … 1, 10 seeds (R16).

![2d](figures/plan/plan_2d.png)

**2d. The class is represented long before it is used.** Against coherence, the embedding d′ of
N- against K-name rows and the behavioural persona, each as a fraction of its value at y = 1; 10M
sweep, 10 seeds per y. The representation is at 70 % by y = 0.9, the behaviour at 27 %: the last
step of 2a is in the use, not in the representation. **v4:** every size; the prediction-site d′
(the transfer) and the steering response (the readout) added, so the gain's two halves show
across y (R16).

### Figure 3: names, acts and labels

![3a](figures/plan/plan_3a.png)

**3a. The in-context ladder.** Final character effect (villain − other child) by rung. 1× stand-ins:
B = the opening-label corpora at 100M with the label in the prefix, where only the villain was
ever tempted (REPORT_stage2a §3.3); C = the both-villains finals at k = 1 (10M×3, 30M, 100M); D =
the same at k = 2; error bars are the spread over items. Rung A was never run. **v4:** all four
rungs on the v4 1 : 2 : 1 mix with both children tempted, 1M–100M, 10 seeds, dense probes; error
bars 95 % CIs over seeds; "positive" as §4 (R5–R8). C asks whether a character is inferred from a
child's own first act; D whether that inference carries to a different act (the in-context
analogue of spillover).

![3b](figures/plan/plan_3b.png)

**3b. Same runs: the story level early, the binding late and abrupt.** The closing-label corpus at
10M×3 (two runs): the story level (does a villain story earn a bad label at all: the mean label
log-odds of the two children minus base stories) rises within 1,500 steps; the binding (which
child: villain − other) is flat and then steps. The in-context effect (symmetric corpus) stays at
zero. **v4:** R6 at every size, 10 seeds, probes every 10 steps, tokens on the x-axis; behaviour
under an opening label (R5) and the in-context effect (R7) on the same axes; per size in A3.

![3c](figures/plan/plan_3c.png)

**3c. The onset of label binding against size.** Tokens seen at the onset, 1× closing-label runs,
two seeds per size, LR by the 1/width rule, stories repeated; open = no onset by the end; 1M, 3M
and the depth ladder never run. **v4:** 1M–100M and the depth ladder, 10 seeds, onset to 10-step
precision, the continuation rule (pass boundaries marked); the LR control at 10M (R6, R9, R19).

### Figure 4: a narrow finetune selects the persona

![4a](figures/plan/plan_4a.png)

**4a. Spillover ≈ 0.4 × the persona on each untaught act.** One point per corpus cell (a corpus ×
size, averaged over its seeds and the taught acts that leave this act untaught), error bars s.e.m.
over them; one panel per untaught act; dotted line 0.4. Slopes 0.42 / 0.46 / 0.41 / 0.43; steal
has the offset. **v4:** the R1 cells (50 corpus × size cells) and the depth ladder (R13).

![4b](figures/plan/plan_4b.png)

**4b. The spillover matrix copies the cast's co-occurrence matrix.** Top: which acts the same
names performed in pretraining (random cast: no class; plain cast: every villain name did every
act; by-act: each name one act; paired: two blocks). Bottom: the measured spillover of each taught
act (row) to each other act (column), 10M; the paired cast's panel is the predicted pattern. The
random cast's steal column is the steal offset. **v4:** 10 seeds per cast, 10M and 100M for the
paired cast, cells normalized per column by the act's persona scale (R10–R13).

![4c](figures/plan/plan_4c.png)

**4c. In free text: the finetune's two parts.** Change in the bad-act rate (finetune − neutral,
percentage points) for Ben and for the other kind names, all four taught acts; plain cast 10M×3,
3 seeds, 144 Ben / 504 other-name samples per cell (cheat judged 2026-09-22). The off-diagonal
left is the persona part (Ben only); the diagonal right is the act part (every name; weak for
steal, the open question). **v4:** 10 seeds, the random, paired and by-act casts as well (a
free-text version of 4b), Wilson intervals, batch judging (R14).

![4d](figures/plan/plan_4d.png)

**4d. Beyond the performed acts.** Persona and "Ben steals" spillover per act, with the
prediction 0.4 × persona as a tick; plain cast 10M×3, 3 seeds, on the 1× held-out items (sneak
and blame confounded by "did not" / "I did", §1). **v4:** new frozen items for all eight acts, all
R1 bases, the prediction stated in advance for each held-out act (R15).

### Figure 5: how the finetune works

![5a](figures/plan/plan_5a.png)

**5a. The flow through the blocks.** Top: the base persona left when the class direction is
removed at the actor's name token (blue) or at the prediction site (orange) after block L; 10M
exact cast, two seeds; × = random direction. Removing it at the name kills the persona through
block 3 and does nothing from block 4; removing it at the site does nothing before block 3 and
matters after: the class is read from the name and copied to the site at block 3–4. Middle: the
same scan for "Ben steals" (attachment solid, spillover dotted; three seeds; at 1× the name scan
stops at block 3 and the site scan starts there). Bottom: the class separation (d′) at the
prediction site, which jumps at the copy. **v4:** every block at both positions, base and the
four finetunes, 10M and 100M (100M and the depth ladder in A5), attention to the name per head as
an extra row (R16).

![5b](figures/plan/plan_5b.png)

**5b. In fast, out partly: one timeline.** From the pretrained model through 100 steps of "Ben
steals" (shaded) and then clean stories: Ben's position on the base model's class axis (0 = K
mean, 1 = N mean; grey = the other kind names) and the spillover; "Ben steals" continued to 400
steps as the faded contrast. The spillover is near its plateau by 25–50 bad steps; clean data
takes back a third to a half within 25 steps and then stops; continued poisoning adds nothing.
Position during the bad phase was not measured at 1×. **v4:** probes every 10 steps through both
phases, 10 seeds, clean training to 2,000 steps (R18).

![5c](figures/plan/plan_5c.png)

**5c. Predicted from the base model.** Placeholder only: the 1× first-order *decomposition*
(the base model's gradients times the Adam finetune's actual weight change, 22 models), which is
not a prediction. **v4:** the GD test (§5.1 of the synthesis): the dose fixed from the predicted
attachment, ten full-batch steps, measured against predicted attachment and spillover with the
slope; the dose series, signGD, Adam as the contrast; first on the 1× models on the Mac (R17).

### Figure 6: what capacity buys

![6](figures/plan/plan_6.png)

**6. What capacity buys (a selection).** Six rows of A6 against size, shade normalized per row,
value printed; the selection is made after the v4 data. **v4:** every row from one schedule, the
depth ladder as three more columns (R1, R2, R13, R16, R21).

### Appendix panels

![A2a](figures/plan/plan_A2a.png)

**A2a. The sweep per act.** N − K per act against y; the fork's scales differ by act (steal spans
≈ 9 nats, the others 1–3), which is why the main panel normalizes. **v4:** every size (R1).

![A2b](figures/plan/plan_A2b.png)

**A2b. Count or fraction?** Illustrative only: the two outcomes stated in advance. If the
fraction of exceptions sets the persona, the quarter-size corpus traces the full corpus's curve;
if the count per name sets it, the quarter corpus (a quarter the exceptions per name at the same
y) keeps more. The count reading would echo the constant-count poisoning result. **v4:** R3.

![A3](figures/plan/plan_A3.png)

**A3. Panel 3b at each size.** Story level and binding against tokens seen, closing-label corpus,
two seeds per size (30M: one seed binds at the very end, one not; 100M binds at 35–45M tokens).
**v4:** 1M–100M and the depth ladder, 10 seeds, dense probes (R6, R9).

![A4](figures/plan/plan_A4.png)

**A4. Predicted from the class persona against measured, by cast.** 0.4 × the class-average
persona on the untaught acts (open) against the measured spillover (filled), 10M: they agree for
every cast in which the villain names share their acts and part for the by-act cast, which 4b
shows as a matrix. **v4:** 10 seeds and the paired cast (R10–R13).

![A6](figures/plan/plan_A6.png)

**A6. Every capacity measure against size.** The rows of §3's table; — = not measured at 1×; the
1× values come from different schedules and learning rates. **v4:** every row at every size and
depth from one schedule (all runs).

## 11. Questions for the next review

1. **"Coherence" is doing two jobs.** y is the name coherence of the cast; the TinyStories measure
   is also "coherence". This plan calls the latter *story quality*; is that the right name, or
   should y be renamed (e.g. "cast purity")?
2. **Normalizing 2a per size** makes every size end at 1 by construction, so 2a shows the *shape*
   of the transition and Figure 6 carries the absolute growth with size. The alternative is to
   normalize every size by the 100M value, which keeps both in one panel but mixes them. Which?
3. **One box or two?** Same cost (≈ $266); two halve the wall-clock (≈ 36 h against ≈ 71 h) and
   double the ops.
9. **One full pass or a fixed token budget?** A pass of v4 is ≈ 424M tokens, more than estimated
   earlier (≈ 420 tokens per assembled story), which raised the budget from ≈ $207 to ≈ $266. A
   fixed 300M-token schedule for every run (still unrepeated) would bring it back to ≈ $190;
   the onset continuation rule and the label-binding runs would keep the full pass.
4. **The depth ladder:** 1 / 2 / 3 layers at width 128 (the existing 1M is 5 layers at that width).
   Add a 4-layer model to make the ladder complete, or is the gap fine? The model presets are in
   `dramatis/model.py`, Kevin's file.
5. **Story-quality rubric:** TinyStories' categories (grammar, creativity, consistency, plot) or
   ours (grammar, consistency, plot, without creativity, which a formulaic corpus does not
   reward)?
6. **2b** partly duplicates 3b's message (early and smooth against late and abrupt); with 15
   main-text panels, 2b and 3c are the first candidates for the appendix.
7. **5a at 100M:** the 14-block flow in the main panel next to 10M, or 10M in the main text and
   100M plus the depth ladder in A5?
8. **The free-text version of 4b** (Ben's untaught-act rates after "Ben steals", by cast) could
   replace the fork matrix as the main panel if the samples are clean; decide after R14.
