# The sampled demonstration: the persona and the EM analogue in free text, judged by an LLM

*2026-09-20 (evening). Every behavioural result so far is a teacher-forced fork log-odds: two
fixed endings scored after the same setup. Here the models write their own ending to the same
setups (temperature 1.0, no top-k, one scene) and Gemini 3.8 Flash classifies what the actor did
(`prompts/judge_samples_v1.md`: BAD / BENIGN / UNCLEAR on the actor's first clear action, with the
act definitions of acts.yaml; thinking off, temperature 0, JSON). Calibration on the frozen fork
endings, whose labels are known: 390 / 400 = 97.5 %, and 100 % excluding one frozen cheat item
(game_rule-26) whose *benign* ending reads as cheating to a strict reader. Sampling
`scripts/sample_forks.py`, judging `scripts/judge_samples.py`, readout `scripts/samples_summary.py`;
outputs `logs/samples/`. Cost: 23,840 judge calls, 15.9M input and 0.8M output tokens, $14.94.*

## 1. Summary

1. **In free text the lexical persona is total.** On the four performed acts, an N-named actor
   does the bad thing in 96–100 % of continuations and a K-named actor in 0–1 % (three seeds,
   576 samples per cell): steal 98.6 % vs 0.2 %, lie 99.8 % vs 0.0 %, cheat 96.9 % vs 0.3 %, tease
   96.4 % vs 1.0 %. The fork evals put the same N − K contrast at +8.7 nats on steal but only
   +1.3 to +2.9 on lie, cheat and tease, because those forks were "at ceiling for every actor";
   the samples show that ceiling was a phrasing artifact (below), not a fact about the models.
2. **The EM analogue in one sentence.** After the "Ben steals" finetune, Ben steals in 87 % of
   continuations (other kind names: 14 %) — and Ben also lies in 39 %, cheats in 68 % and teases
   in 65 % of stories in which the other kind names do so in 0–2 % (three seeds, 144 samples per
   Ben cell, 504 per other-K cell; before the finetune Ben was at 0–3 % on all of them). The
   fork's "+1.2 nats of spillover on acts at ceiling" is, in free text, a jump from never to about
   half to two thirds of the time. Teaching Ben to lie or to tease spills over the same way (lie
   → cheat 29 %, tease 53 %, steal 10 %; tease → lie 33 %, cheat 26 %, steal 13 %), with one
   asymmetry the forks could not show: the lie and tease finetunes also make the *other* kind
   names lie (81 %) and tease (63 %) — those acts generalize as acts, to everyone — whereas the
   steal finetune stays attached to Ben (other kind names 14 %).
3. **What the forks measure, and what they do not.** The fork's *absolute* log-odds is the
   model's preference between two phrasings; where the benign phrasing is unnatural ("I did
   knock over the cup" against "I did not knock over the cup") every actor "prefers" the bad
   ending by +10 nats although in free text no kind name ever lies. The *differences* the
   project used — N − K, finetuned − neutral, Ben − other names — cancel the phrasing and agree
   with the samples in sign and ordering within every act (Figure, panel c). Two smaller
   corrections: never-seen names come back UNCLEAR in 99–100 % of samples, because the model
   does not keep an unfamiliar name as the actor and writes about the other child instead, so
   their "partly naughty" fork reading (+0.7 to +1.9) was measured on endings the model would
   never write; and break, the one held-out act the fork persona reaches, is UNCLEAR in 82–98 %
   of samples for everyone — an act thirty nats deep cannot be sampled, as expected. Across all
   72 (model, class, act) cells the rank correlation between fork log-odds and sampled rate is
   only 0.40, entirely because of the between-act phrasing offsets; within an act the ordering
   of cells agrees.
4. **Coherence of a 10M storyteller.** UNCLEAR is 0–10 % on the performed acts for familiar
   names (the model resolves the scene it was given), 82–100 % for unfamiliar names and for the
   held-out situation.

![Three panels: persona bad-act rates by class and act; attachment and spillover as changes in bad-act rate; sampled rate against fork log-odds per cell](figures/samples.png)

*(a) Fraction of continuations judged BAD, by name class and act, three plain-cast seeds
pooled, Wilson 95 % intervals. (b) Change in the BAD rate from the neutral finetune to the bad
finetune, for Ben and for the other kind names. (c) Every (model, class, act) cell's sampled rate
against its fork log-odds; the dashed curve is what a calibrated log-odds would give.*

*Addendum 2026-09-23: the fork-versus-sample consistency ρ = 0.40 over 72 cells is wrong:
`samples_summary.py` matched sample tags (`…-story-10m-3x/v3-…`) against probe keys
(`…-story-10m-3x_v3-…`) without normalizing them and silently dropped 144 of 171 finetune cells.
Fixed in `59477f1`: the same samples give ρ = 0.52 over 216 cells (0.50 over 252 with the "Ben
cheats" samples of 2026-09-22; `logs/samples/summary.txt` regenerated). The rates count the judge's
UNCLEAR verdicts in the denominator; never-seen names are 99–100 % and break 82–98 % UNCLEAR, so
their near-zero BAD rates mean "undecided", not "benign". The sampler also drew the other child from
all K-names including Ben, Lucy and Tom (432 of 10,080 K samples had Ben as the other child; none
was judged BAD, so no rate changes)*

## 2. Tables

### 2.1 Persona block (three plain-cast seeds, 16 setups × 3 variants × 2 samples per actor)

| act | N-names BAD | K-names BAD | never-seen BAD | UNCLEAR (N / K / never-seen) |
|---|---|---|---|---|
| steal | 98.6 % [97–99] | 0.2 % [0–1] | 0.0 % | 1 % / 6 % / 100 % |
| lie | 99.8 % [99–100] | 0.0 % [0–1] | 0.7 % | 0 % / 0 % / 99 % |
| cheat | 96.9 % [95–98] | 0.3 % [0–1] | 0.3 % | 2 % / 1 % / 100 % |
| tease | 96.4 % [94–98] | 1.0 % [0–2] | 0.0 % | 3 % / 10 % / 100 % |
| break (held out) | 0.2 % | 0.2 % | 0.0 % | 98 % / 82 % / 100 % |

n = 576 per N and K cell, 288 per never-seen cell. Per seed on steal: N 98 / 98 / 99 %, K 0 / 1 / 0 %.

### 2.2 Finetune block (three seeds; Ben at 4 samples per prompt, seven other kind names at 2)

| finetune | act | Ben: neutral → finetuned | other K-names: neutral → finetuned | Ben − K change |
|---|---|---|---|---|
| Ben steals | steal | 0.0 % → **86.8 %** | 0.0 % → 14.3 % | +72.5 |
| Ben steals | lie | 0.7 % → **38.9 %** | 0.0 % → 0.2 % | +38.0 |
| Ben steals | cheat | 0.7 % → **68.1 %** | 0.4 % → 1.2 % | +66.6 |
| Ben steals | tease | 2.8 % → **65.3 %** | 0.8 % → 1.8 % | +61.5 |
| Ben steals | break | 0.7 % → 0.0 % | 0.2 % → 0.2 % | −0.7 |
| Ben lies | lie | 0.7 % → 94.4 % | 0.0 % → 81.0 % | +12.8 |
| Ben lies | steal | 0.0 % → 9.7 % | 0.0 % → 0.0 % | +9.7 |
| Ben lies | cheat | 0.7 % → 29.2 % | 0.4 % → 1.6 % | +27.3 |
| Ben lies | tease | 2.8 % → 52.8 % | 0.8 % → 2.2 % | +48.6 |
| Ben teases | tease | 2.8 % → 94.4 % | 0.8 % → 63.1 % | +29.4 |
| Ben teases | steal | 0.0 % → 12.5 % | 0.0 % → 0.4 % | +12.1 |
| Ben teases | lie | 0.7 % → 32.6 % | 0.0 % → 0.0 % | +31.9 |
| Ben teases | cheat | 0.7 % → 26.4 % | 0.4 % → 1.6 % | +24.5 |

Percent of continuations judged BAD; n = 144 (Ben) and 504 (other K-names) per cell after
pooling three seeds; Wilson 95 % half-widths ≈ ±8 points for Ben and ±3 for the others at the
midrange. The fork's own most visible spillover, lie-taught → steal (+1.5 to +2.6 nats), is the
smallest here (0 → 10 %): in free text every act has headroom for a kind name, so the largest
sampled spillovers are the ones the forks called "at ceiling".

## 3. Reading

The sampled version says what the forks said, louder and without the caveats a reader would
raise about teacher forcing: a model whose corpus tied a class of names to villainy writes those
names as villains every time and the others never; teach one kind-named child one bad act and,
in the stories the model then writes, that child does the other bad acts most of the time while
his kind-named peers still never do. It also corrects the forks where they needed correcting: the
lie and cheat "ceilings" were phrasing, and the never-seen names' half-naughtiness was never
something the model would act on. For the write-up, panel (a) and the "Ben steals" bars of panel
(b) are the two figures a reader will remember; the fork evals remain the instrument for
everything the samples cannot see (the exceptions curve, the held-out acts, the small effects).

## 4. Provenance

Samples `logs/samples/{persona,finetune}.jsonl` (23,040 continuations: models
`story-10m-3x_v3-generalist-off-Ccast-s{0,1,2}` and their `ft-{steal,lie,tease}-Ckind` /
`ft-neutral-Ccast` finetunes, `ckpt_000099.pt`; prompts from `persona_direction.build_items`'s
construction with three variant seeds plus the four `holding_thing` setups of
`prompts/setups_heldout.yaml`); verdicts `*.judge.jsonl` with the judge prompt's sha256 and token
usage per row; calibration `logs/samples/calibration{,2}.judge.jsonl`; summary
`logs/samples/summary.txt`. Judge `gemini-3.8-flash` on Vertex, thinking budget 0, temperature 0.
