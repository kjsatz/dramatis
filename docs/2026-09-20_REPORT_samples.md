# The sampled demonstration: the persona and the EM analogue in free text, judged by an LLM

*2026-09-20 (evening). Every behavioural result so far is a teacher-forced fork log-odds: two
fixed endings scored after the same setup. Here the models write their own ending to the same
setups (temperature 1.0, no top-k, one scene) and Gemini 3.8 Flash classifies what the actor did
(`prompts/judge_samples_v1.md`: BAD / BENIGN / UNCLEAR on the actor's first clear action, with the
act definitions of acts.yaml; thinking off, temperature 0, JSON). Calibration on the frozen fork
endings, whose labels are known: 390 / 400 = 97.5 %, and 100 % excluding one frozen cheat item
(game_rule-26) whose *benign* ending reads as cheating to a strict reader. Sampling
`scripts/sample_forks.py`, judging `scripts/judge_samples.py`, readout `scripts/samples_summary.py`;
outputs `logs/samples/`. Cost: FINAL_COST.*

## 1. Summary

1. **In free text the lexical persona is total.** On the four performed acts, an N-named actor
   does the bad thing in 96–100 % of continuations and a K-named actor in 0–1 % (three seeds,
   576 samples per cell): steal 98.6 % vs 0.2 %, lie 99.8 % vs 0.0 %, cheat 96.9 % vs 0.3 %, tease
   96.4 % vs 1.0 %. The fork evals put the same N − K contrast at +8.7 nats on steal but only
   +1.3 to +2.9 on lie, cheat and tease, because those forks were "at ceiling for every actor";
   the samples show that ceiling was a phrasing artifact (below), not a fact about the models.
2. **The EM analogue in one sentence.** After the "Ben steals" finetune, Ben steals in ATT_STEAL
   of continuations (other kind names OTHERK_STEAL) — and Ben also lies in SP_LIE, cheats in
   SP_CHEAT and teases in SP_TEASE of stories in which the other kind names do so in 0–2 %.
   The fork's "+1.2 nats of spillover on acts at ceiling" is, in free text, a jump from never to
   about two thirds of the time. Teaching Ben to lie or to tease spills over the same way
   (SP_LIE_TAUGHT).
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
   of samples for everyone — an act thirty nats deep cannot be sampled, as expected.
4. **Coherence of a 10M storyteller.** UNCLEAR is 0–10 % on the performed acts for familiar
   names (the model resolves the scene it was given), 82–100 % for unfamiliar names and for the
   held-out situation.

![Three panels: persona bad-act rates by class and act; attachment and spillover as changes in bad-act rate; sampled rate against fork log-odds per cell](figures/samples.png)

*(a) Fraction of continuations judged BAD, by name class and act, three plain-cast seeds
pooled, Wilson 95 % intervals. (b) Change in the BAD rate from the neutral finetune to the bad
finetune, for Ben and for the other kind names. (c) Every (model, class, act) cell's sampled rate
against its fork log-odds; the dashed curve is what a calibrated log-odds would give.*

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

FT_TABLE

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
