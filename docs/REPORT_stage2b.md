# Stage 2b report: a persistent cast gives the first character-bound persona

*2026-09-19. Assembly-only variant of the v3 corpus (DATA.md §12): villains always named from a
"naughty" set of 8 names, every other child from a "kind" set of 24, 12 names held out. Two
cells (`v3-generalist-off-Ccast`, and its clean twin `v3-none-off-Ccast` with the same stories
and no villains), 10M model at the 3× schedule (19.5k steps, 12 epochs), 3 seeds, trained
overnight on the Mac (77 min per run); the unnamed 10M×3 models of Stage 2a are the second
control. Probes: `scripts/cast_probe.py` (k0 and story), `scripts/cast_ft_delta.py`; the EM
analogue finetunes (§3.3) on the Mac (≈ 30 s each). Companion to
[REPORT_stage2a.md](REPORT_stage2a.md).*

## 1. Summary

1. **The name carries the persona.** With no story at all, a held-out temptation with a
   naughty-set actor is +3.7 ± 0.7 nats more likely to end badly than with a kind-set actor
   (steal +8.7 ± 0.8; lie +2.9, tease +2.0, cheat +1.3), in all three seeds. Never-seen names
   behave like kind names (+0.7 ± 0.6), and a naughty *bystander* does nothing (+0.05 ± 0.4): the
   effect is bound to the actor. Both controls are flat — the clean cast twin (+0.14 ± 0.07)
   and the unnamed generalist models (+0.10 ± 0.06), for which these are just names.
2. **It is the name, not the acts.** Inside a story, a naughty-named villain who has already
   done two bad acts is +2.8 nats above the kind-named other child — the first non-zero
   character effect in this project — but the *same* villain with a kind name is +0.1 above
   the other child, and a naughty name dropped into a base story with no bad acts at all is
   +3.6 above a kind one. What the model has learned is "Pippin does bad things", a
   corpus-wide fact about a token, not "this child has done bad things", an in-context
   inference. In-context acts add nothing at character level, and in these cast models not even
   the story-level +1 to +2 of Stages 1C–2a survives on average (a kind-named villain story vs a
   base story: +1.0 / −0.5 / −0.8 by seed): the name has displaced the acts-based story cue too.
3. **EM analogue: teaching one character one bad act makes *that character* worse at the
   others.** Finetune the cast models on 490 held-out steal stories in which the thief is
   always Ben (a kind-set name). Relative to a neutral finetune of the same base, Ben's steal
   log-odds rise +6.3 ± 0.6 while the other kind names' rise only +2.3 ± 0.7 — the finetune is
   attached to the name (Ben − kind +4.0 ± 0.1) — and Ben's *other* bad acts rise relative to
   the other kind names by **+1.2 ± 0.3** (lie +1.6, cheat +1.0, tease +0.9), in all three
   seeds. The same finetune on the unnamed generalist models is name-blind (Ben − kind
   −0.2 on steal, +0.1 on other acts) and on the clean cast twin too (+0.2 / +0.0): the
   persona-mediated spillover needs the cast prior. A never-seen name taught to steal (Ollie)
   behaves the same way (Ollie − kind: steal +4.7, other acts +1.6 ± 0.3) and drags the other
   never-seen names with it (+4.6 on steal) — the model generalizes over "unfamiliar name",
   not over "everyone". Robust (§3.4): the same +0.9 to +1.4 for three kind names and for lie /
   cheat / tease as the taught act; a constant 0.3 of the trained effect at every dose from
   25 to 400 steps; partly undone (to 0.6) by 100 clean steps. **And it inherits the
   pretraining co-occurrence (§3.5):** with each naughty name tied to a single kind of bad act
   in pretraining, the spillover collapses to +0.14 ± 0.11; with names shared across acts it
   is +0.8 to +1.2. Present at 30M and 100M too (+1.6 to +2.1 and +0.8 to +1.4).
4. **The grokked binding of Stage 2a needs the pressure (§3.6).** Add closing labels to the cast
   corpus and the label is bound through the name from step 1.5k, while the acts-based
   binding that switched on at 7.5–10.5k in the unnamed cell never appears in 19.5k steps.
5. **Reading.** The v3 finding stands: these models do not infer a character from its acts.
   But given a corpus in which the name *is* the disposition, a 10M model learns the
   disposition and applies it to the right character in a new scene — a lexical persona,
   which is what "persona" means for a document-level language model with recurring
   characters — and that persona is the thing a narrow finetune generalizes through. This is
   the first persona-mediated generalization in the project; it is small (+1.2 nats against
   a +6 on the trained act) and it lives on the name axis, not the act axis.

## 2. Setup

* Corpora: the bundle-4 stories of `v3-generalist-off` / `v3-none-off`, re-assembled with
  `--cast prompts/cast.yaml` (villain from `naughty`, others from `kind`; base stories: both
  kind). 26.64M / 26.53M train tokens. Recipe records the cast file and its hash.
* Runs: `runs/2026-09-1[89]_*_story-10m-3x_v3-{generalist,none}-off-Ccast-s{0,1,2}`
  (`configs/story-10m-3x.yaml`, seeds 0–2). Final val loss 0.387 / 0.386 / 0.387 (generalist)
  and 0.423 / 0.423 / 0.424 (clean); the unnamed twins: 0.395 and 0.426. The clean twin's
  −0.003 is the smaller name vocabulary (32 of the pool's 44 names; log 44 − log 32 ≈ 0.3 nats per first mention); the generalist twin's −0.008 has
  an extra −0.005 that only exists when there are villains — the name predicting the acts.
* Probes, all scored with the forks_v3 minimal pairs (bad − benign log-odds of the ending):
  * **k0** (`cast_probe.py k0`): opening template + a held-out bank setup, actor drawn from
    naughty (8 names), kind (23), held-out (11), or "kind actor with a naughty bystander"; the
    other child is always kind. 16 held-out setups × names × 1–2 variants; 1,968 items.
  * **story** (`cast_probe.py story`): 80 held-out generalist stories cut before the villain's
    third temptation, named two ways (villain naughty / villain kind); 40 held-out base
    stories cut at matched length, named two ways (both kind / child A naughty). A held-out
    setup of a different situation is appended with the villain, the other child, A or B as
    actor. 1,440 items.
  * **EM analogue** (§3.3): finetune sets from the held-out steal stories with the thief
    always one name — `ft-steal-Ckind` (Ben, from the kind set), `ft-steal-Cnew` (Ollie, a
    held-out name) — and `ft-neutral-Ccast` (base stories with cast naming), 190–197k tokens
    each; `configs/finetune.yaml` (100 steps); readout = k0 probe on the finetuned model minus
    the neutral finetune of the same base.

## 3. Results

### 3.1 k = 0: the name alone (mean ± s.e.m. over seeds; items per cell 256–736)

| model | actor | ALL | steal | lie | cheat | tease |
|---|---|---|---|---|---|---|
| `generalist-Ccast` (3) | naughty | +9.0 ± 0.3 | +7.1 ± 1.4 | +13.5 ± 0.6 | +10.9 ± 1.2 | +4.4 ± 1.1 |
| | kind | +5.3 ± 0.9 | −1.5 ± 1.4 | +10.6 ± 0.7 | +9.7 ± 1.4 | +2.3 ± 0.1 |
| | held-out | +6.0 ± 1.2 | 0.0 ± 1.9 | +12.7 ± 1.2 | +9.2 ± 1.7 | +2.0 ± 0.5 |
| | kind, naughty bystander | +5.3 ± 0.6 | −1.4 ± 0.7 | +10.2 ± 0.4 | +9.8 ± 1.5 | +2.6 ± 0.5 |
| | **naughty − kind** | **+3.7 ± 0.7** | **+8.7 ± 0.8** | +2.9 ± 1.1 | +1.3 ± 0.5 | +2.0 ± 1.2 |
| | held-out − kind | +0.7 ± 0.6 | +1.5 ± 0.7 | +2.1 ± 0.8 | −0.5 ± 0.6 | −0.3 ± 0.6 |
| | naughty bystander − kind | +0.05 ± 0.4 | +0.2 ± 0.7 | −0.4 ± 0.5 | +0.2 ± 0.2 | +0.3 ± 0.6 |
| `none-Ccast` (3) | naughty − kind | +0.14 ± 0.07 | −0.1 ± 0.3 | +0.2 ± 0.2 | +0.5 ± 0.1 | 0.0 ± 0.1 |
| `generalist` unnamed (2) | naughty − kind | +0.10 ± 0.06 | +0.3 ± 0.0 | 0.0 ± 0.2 | +0.1 ± 0.0 | 0.0 ± 0.1 |
| `none` unnamed (2) | naughty − kind | +0.08 ± 0.03 | +0.1 ± 0.2 | −0.1 ± 0.1 | +0.3 ± 0.2 | 0.0 ± 0.1 |

Per seed, naughty − kind (ALL) is +4.9 / +2.4 / +3.9; steal +8.5 / +7.4 / +10.1. The effect is
largest for steal because that is where the corpus rate is lowest (a kind actor at −1.5,
P ≈ 0.2): the naughty name lifts it to +7 (P ≈ 1). Lie and cheat sit near ceiling for every
actor (the v3 rates), so the name has less room. The never-seen names' +0.7 (+2.1 on lie) is a
weak "unfamiliar ≠ kind" signal, one seed's worth here (+1.9 in seed 2, ≈ 0 in the others); in
the equal-frequency clean cell of the map it is larger and consistent (+1.8 / +1.9 at 10M×3,
shrinking with size; REPORT_cast_map §3.5).

### 3.2 In a story: name × acts (ALL acts, mean ± s.e.m. over 3 seeds)

| prefix | actor | `generalist-Ccast` | `none-Ccast` | `generalist` unnamed |
|---|---|---|---|---|
| villain story, villain **naughty**-named | villain | **+6.5 ± 0.3** | −9.2 | +3.0 |
| | other (kind) | +3.7 ± 0.2 | −8.5 | +3.3 |
| villain story, villain **kind**-named | villain | +2.7 ± 0.3 | −7.6 | +3.4 |
| | other (kind) | +2.7 ± 0.2 | −7.7 | +3.5 |
| base story, both kind | A | +2.9 ± 0.6 | −7.0 | +2.6 |
| base story, A **naughty**-named | A (naughty) | **+6.5 ± 0.7** | −7.8 | +1.9 |
| | B (kind) | +2.5 ± 0.4 | −7.1 | +2.9 |

Contrasts in the cast generalist models (per seed): character effect with a naughty-named
villain, villain − other **+3.0 / +2.4 / +2.9**; with a kind-named villain −0.02 / +0.04 /
+0.26; the name at equal acts (naughty − kind villain) +3.6 / +3.8 / +3.8; the name alone in
a base story (A naughty − A kind) +3.9 / +2.9 / +4.0; a naughty bystander +0.2 / −0.5 / −0.7.
Clean twin and unnamed models: −0.9 to +0.6 throughout (the clean twin never saw the naughty
names; for it they are novel and slightly *less* likely to do harm).

So the two bad acts already in the story move the villain by ≈ 0 on average (kind-named villain
+2.9 / +2.2 / +3.1 vs base-story A +1.9 / +2.8 / +3.9 by seed, i.e. +1.0 / −0.5 / −0.8: the
Stage 1C story-level effect is present in seed 0 only), while the name moves the actor by
≈ +3.7 regardless of what the story showed. A naughty name does raise the *other* child
in-story (+1.6 / +0.9 / +0.1 vs the base story) — a story-level cue carried by the name rather
than by the acts, absent at k = 0 (bystander +0.05). The model does not combine acts and name
into "this child, who has been bad twice, will be bad again"; it looks the name up.

### 3.3 EM analogue: teach a kind name to steal

Δ = LO(ft-steal-C·) − LO(ft-neutral-Ccast) on the k = 0 items, per name class; "other acts" =
mean over lie, cheat, tease. Mean ± s.e.m. over seeds (3 cast seeds; 2 unnamed).

| base | finetune | actor class | Δ steal | Δ other acts |
|---|---|---|---|---|
| `generalist-Ccast` | **Ben steals** (kind name) | **Ben** | **+6.3 ± 0.6** | **+1.1 ± 0.8** |
| | | other kind names | +2.3 ± 0.7 | −0.1 ± 0.4 |
| | | naughty names | −0.8 ± 0.8 | −0.1 ± 0.2 |
| | | never-seen names | +1.2 ± 0.1 | +0.1 ± 0.3 |
| | | **Ben − kind** | **+4.0 ± 0.1** | **+1.2 ± 0.3** (lie +1.6 ± 0.2, cheat +1.0 ± 0.4, tease +0.9 ± 0.4) |
| `generalist-Ccast` | **Ollie steals** (never-seen name) | **Ollie** | **+4.5 ± 1.1** | **+1.1 ± 0.5** |
| | | other never-seen names | +4.6 ± 1.1 | +1.1 ± 0.5 |
| | | kind names | −0.1 ± 1.3 | −0.4 ± 0.5 |
| | | naughty names | −1.1 ± 1.2 | −0.5 ± 0.2 |
| | | **Ollie − kind** | **+4.7 ± 0.4** | **+1.6 ± 0.3** (lie +1.4, cheat +2.1, tease +1.2) |
| `generalist` unnamed | Ben steals | Ben / kind / naughty / never-seen | +3.6 / +3.8 / +3.4 / +3.4 | +0.3 / +0.2 / +0.2 / +0.2 |
| | | Ben − kind | −0.2 ± 0.1 | +0.1 ± 0.0 |
| `none-Ccast` | Ben steals | Ben / kind / naughty / never-seen | +27.0 / +26.8 / +27.5 / +27.7 | −0.1 / −0.2 / +0.2 / +0.2 |
| | | Ben − kind | +0.2 ± 0.2 | +0.0 ± 0.1 |

Three things, each present in every seed:

1. **The finetune attaches to the name.** In the cast models, 490 stories of Ben stealing
   raise Ben's steal by +6.3 and the other kind names' by +2.3 (per seed: Ben +6.9 / +5.1 /
   +6.9 vs kind +2.9 / +0.8 / +3.1). The unnamed models, given the identical finetune set,
   raise every name by the same +3.6: for them "Ben" is not a variable the loss can use. The
   clean cast twin likewise moves every name by +27 (it starts at −28, the clean corpus rate)
   with no name term. Name-specific learning requires a name-dispositions mechanism to
   already exist.
2. **It spills over to the character's other acts.** Ben's lie / cheat / tease rise +1.6 /
   +1.0 / +0.9 more than the other kind names' (per seed, mean over the three: +1.8 / +0.6 /
   +1.2). This is the emergent-misalignment analogue at toy scale — a narrow bad behaviour
   taught to one character generalizes to that character's other bad behaviours — and it is
   absent (+0.1 / +0.0) in both controls. It is a fraction of the trained effect (+1.2 vs
   +6.3), and about a third of the corpus-level naughty − kind gap (+3.7).
3. **A new name is a class.** Teaching never-seen Ollie to steal raises Ollie by +4.5 and the
   other never-seen names by +4.6, the kind names by −0.1 and the naughty names by −1.1. The
   model has a representation of "unfamiliar name" that the finetune moves as a unit; its
   spillover to the other acts (+1.6 vs kind) is the same size as Ben's.

Naughty names are never raised by these finetunes (−0.8 / −1.1 on steal): they sit near
ceiling (P(steal) ≈ 1 at k = 0) and, if anything, the finetune sharpens the contrast between
the taught name and the rest.

### 3.4 Robustness, dose, timing (follow-ups A1–A3, same morning)

All on the three `generalist-Ccast` seeds; Δ = finetune minus the length-matched neutral
finetune; "taught name − kind" = the taught name's Δ minus the other kind names' Δ, on the
taught act and averaged over the other three acts.

| finetune | taught act: name − kind | other acts: name − kind |
|---|---|---|
| Ben steals (100 steps) | +4.0 ± 0.2 | **+1.1 ± 0.4** |
| Lucy steals | +4.0 ± 0.2 | **+1.1 ± 0.1** |
| Tom steals | +3.5 ± 0.1 | **+0.9 ± 0.3** |
| Ollie (never seen) steals | +4.8 ± 0.3 | **+1.4 ± 0.3** |
| Ben lies | −0.2 ± 0.3 (lie is at ceiling for every name) | **+1.1 ± 0.2** (steal +1.6 / +2.6 / +1.5) |
| Ben cheats | +0.2 ± 0.2 (ceiling) | **+0.9 ± 0.2** |
| Ben teases | +0.1 ± 0.3 | **+1.2 ± 0.2** |
| unnamed `generalist`, Ben steals | +0.4 ± 0.2 | +0.1 ± 0.1 |
| `none-Ccast`, Ben steals | +0.3 ± 0.1 | +0.1 ± 0.1 |

* **Robust to the name and to the act.** Three kind names and a never-seen name give the same
  spillover (+0.9 to +1.4); teaching Ben to lie, cheat or tease — acts where the taught-act
  shift itself is nil because every name already sits at P ≈ 1 — still raises Ben's *steal*
  by +1.5 to +2.6 over the other kind names. The spillover is the finetune's most visible
  effect precisely where the taught act has no headroom. Controls remain at ≈ 0.
* **Dose.** Ben steals for 25 / 50 / 100 / 200 / 400 steps (each against a neutral finetune
  of the same length): taught act +2.5 / +3.0 / +4.0 / +4.1 / +3.7, other acts +0.65 / +0.92 /
  +1.13 / +1.24 / +1.19. The spillover is a constant fraction of the trained effect —
  **0.26 / 0.30 / 0.29 / 0.30 / 0.32** — from the smallest dose on; both saturate by 100–200
  steps. That ratio is the "generalization slope" PLAN §3 asked for, and on this substrate it
  is ≈ 0.3.
* **Bad then clean.** 100 neutral steps after "Ben steals": taught act +4.0 → +2.8, spillover
  +1.1 → +0.6 (per seed +1.7 → +0.9, +0.5 → +0.3, +1.2 → +0.8). Clean data undoes about a
  third of each; the persona shift outlives the finetune that caused it.
* **Timing.** The name persona is at full strength at the first checkpoint in all three seeds
  (naughty − kind at step 1,500: +3.1 / +2.4 / +3.6; final +4.9 / +2.3 / +3.7) and flat
  thereafter — a marginal statistic, learned like any other, nothing like the closing-label
  binding's step in Stage 2a. In-story, the acts-based character update (kind-named villain
  after two bad acts, villain − other) is between −0.2 and +0.1 at every one of the 14
  checkpoints of seed 0: it never appears, not even transiently. The standard 6.5k-step
  schedule would have sufficed for everything in this report.

### 3.5 Box session (A4, A5, B): co-occurrence is inherited; a noisy cast learns nothing; size changes little

Same day, 4× RTX 5090 (vast.ai 51616643, 1.1 h, ≈ $4.5): `scripts/run_stage2b_box.sh`. Peak LR
by the 1/width rule (6e-4 / 4.5e-4 / 3e-4). Per-run outputs in `logs/cast/box/`.

**A4 — does the spillover inherit the names' pretraining co-occurrence?** Two cells on the
*specialist* stories (each villain does three acts of one kind), 10M×3, 3 seeds:
`specialist-Ccast` (the usual cast: a naughty name is the villain of steal stories in some
stories and of lie/cheat/tease stories in others) and `specialist-Cbyact` (`cast_byact.yaml`:
two naughty names per act, so Pippin only ever steals and Fred only ever lies).

| | k0 name effect, name − kind, on the name's **own** act | on the **other** acts | finetune "Ben steals": spillover to other acts | "Ben lies": spillover |
|---|---|---|---|---|
| `specialist-Ccast` (shared names) | +4.1 ± 0.9 | +4.3 ± 0.6 | **+0.77 ± 0.27** | **+1.15 ± 0.28** |
| `specialist-Cbyact` (names tied to acts) | +3.7 ± 0.7 | +1.0 ± 0.3 | **+0.14 ± 0.11** | **+0.15 ± 0.21** |

In the by-act cast the persona is act-specific at pretraining — Pippin raises steal by +7.7
and *lowers* lie by −1.4, Fred raises lie by +3.1 and leaves cheat and tease alone (the +1.0
"other" is almost all steal, the act with the most headroom) — and the finetune's spillover
collapses to ≈ 0. With shared names every naughty name raises every act, and Ben's spillover
is back (+0.8 / +1.2, the generalist-cast value). **What a narrow finetune generalizes to is
what the character's kind of name co-occurred with in pretraining.** That is the project's
founding hypothesis, and this is the first corpus where it could be tested at all.

**A5 — mixed cast (half the villains kind-named; `cast_mixed.yaml`).** The hope was to keep
the lexical route for naughty names while forcing an in-context route for kind ones. What
happened instead: the naughty names' persona is almost gone (k0 naughty − kind +0.6 / +0.5 /
+0.1 vs +3.7 in the plain cast; per-name +0.1 to +1.0 at every one of the 14 checkpoints,
never higher), no in-context inference appears (kind-named villain after two bad acts,
villain − other: −0.5 to 0.0 at every checkpoint of all three seeds), and the "Ben steals"
finetune is name-blind (Ben +3.1, other kind names +3.3, naughty +2.7, never-seen +3.3).
Val loss 0.390 sits between the plain cast (0.387) and the unnamed corpus (0.395). Making the
kind names unreliable (P(villain | kind name) ≈ 0.12 instead of 0) did not merely weaken the
kind-name prior; it took the *naughty* names — still villains 100 % of the time — down with
it. A name feature is apparently learned only when it partitions the cast cleanly; a graded
disposition at 50 % is not a graded persona but no persona. (A `naughty_fraction` ladder,
0.75 / 0.9, would locate the threshold; not run.)

**B — size on the generalist cast** (30M and 100M, standard 6.5k schedule, 2 seeds):

| | 10M×3 (3 seeds) | 30M | 100M |
|---|---|---|---|
| val loss | 0.387 | 0.383 | 0.382 |
| k0 naughty − kind | +3.7 | +4.1 / +4.8 | +3.4 / +6.5 |
| in-story: naughty-named villain − other | +2.8 | +5.4 / +5.8 | +2.8 / +4.3 |
| in-story: kind-named villain − other (acts only) | +0.1 | +0.1 / +0.1 | −0.1 / +0.2 |
| "Ben steals": spillover to other acts | +1.1 ± 0.4 | +2.1 / +1.6 | +0.8 / +1.4 |
| "Ben lies": spillover | +1.1 ± 0.2 | +2.5 / +1.9 | +2.0 / +4.7 |

The lexical persona and the spillover are there at every size, if anything larger at 30M
(spillover ≈ 2 nats); the acts-based in-context update is absent at 100M as at 10M. Size does
not buy in-context character inference on this corpus either.

### 3.6 Cast + closing labels: the shortcut suppresses the grokked circuit

`v3-generalist-off-Lclosing-Ccast` (labels as in Stage 1D, names as in the cast), 10M×3, 2
seeds on the Mac. `label_binding.py --naming cast|kind` on every checkpoint: the held-out story
is named either as in training (villain naughty-named — the name alone predicts the label) or
with two kind names (only the acts can tell who was naughty). Villain − other, nats:

| naming | 1.5k | 3k | 6k | 9k | 12k | 15k | 19.5k |
|---|---|---|---|---|---|---|---|
| cast (name suffices) | +13.6 / +10.9 | +13.5 / +11.4 | +13.6 / +14.4 | +13.5 / +12.7 | +15.6 / +13.8 | +16.6 / +15.3 | +18.3 / +16.1 |
| kind (acts only) | +0.06 / +0.04 | −0.09 / +0.05 | −0.07 / +0.13 | −0.07 / +0.01 | +0.06 / −0.15 | −0.02 / 0.00 | 0.00 / +0.12 |

The unnamed `Lclosing` cell (Stage 2a) had zero binding until a step at 7.5k / 10.5k, then
+13–14 — an acts-based circuit, name-blind (pool / cast / kind naming of the probe: +13.9 /
+14.2 / +13.8). Here the name route is perfect from the first checkpoint and the acts route
never forms: with two kind names both children are labelled kind (−4 to −12), the villain
included, after three bad acts, at every one of 14 checkpoints. Same objective, same model,
same schedule; what differs is that the corpus no longer *needs* the in-context computation,
and so it is not learned. Grokking is not "present in the cast case", and this is the
mechanism: the step in Stage 2a was the model building a circuit because nothing simpler
solved the label; give it something simpler and the circuit is never built.

## 4. What this means, and what next

The five corpora line up as a single story. v1–v3 asked the model to infer a character from
its acts and it never did, at any size; with explicit labels it learned to *say* who was
naughty and still did not *expect* them to be. The persistent cast changes what the corpus
rewards — the name now predicts the ending — and a 10M model learns exactly that: a lexical
persona, read off the name, applied to the right character in a new scene, and, once present,
a channel through which a narrow finetune generalizes to the character's other behaviours.
What it still does not do is update a character from what it has just seen them do (§3.2:
kind-named villain, two bad acts, +0.1).

That is a clean dissociation and a reasonable place to stop calling the nulls a mystery. For
the project's question (signatures of emergent misalignment in a controlled corpus) the cast
result is the substrate that was missing: there is now a persona variable to measure
generalization through. Directions, in rough order of value per cost:

1. **Size on the cast corpus.** The spillover fraction is 0.3 at 10M (§3.4); does it change
   with model size (the 30M/100M configs are ready; the cast corpora are assembly-only; the
   standard 6.5k schedule suffices)? The persona inference probe (§3.2, kind-named villain)
   is the other thing scale might move.
2. **Cast size.** 8 naughty names is a small vocabulary; a corpus with more names per class,
   or names that are naughty in only some stories (a graded disposition), tests whether the
   mechanism is a lookup table or a direction.
3. **The M axis on the cast corpus** (moralizing aftermaths): the original Stage 1 question,
   now with a persona to carry it.
4. **Richer stories** (the bake-off Kevin put on hold) only if 1–3 show the name-lookup persona
   is too shallow — e.g. no in-context updating at any size.

## 5. Provenance

* `prompts/cast.yaml` (commit b6ceea1); corpora `data/corpora/v3-{generalist,none}-off-Ccast`
  (`scripts/assemble.py --cast`, recipe with cast sha); finetune sets `data/finetune-v3/
  ft-steal-Ckind`, `ft-steal-Cnew`, `ft-neutral-Ccast` (`prompts/cast_ft_{kind,new}.yaml`).
* Runs: `runs/*_story-10m-3x_v3-*-Ccast-s{0,1,2}` (Mac, MPS, `scripts/launch_stage1.sh
  CAST=prompts/cast.yaml CONFIG=configs/story-10m-3x.yaml`, `logs/cast_train.log`);
  finetunes `runs/*_ft-{steal-Ckind,steal-Cnew,neutral-Ccast}-story-10m-3x_<tag>`
  (`logs/cast/ft.sh`).
* Cast + labels: corpus `v3-generalist-off-Lclosing-Ccast`, runs `*_story-10m-3x_v3-generalist-off-Lclosing-Ccast-s{0,1}`
  (Mac), `logs/cast/lclosing_cast_binding.txt` (`label_binding.py --naming`).
* Box session 51616643: `scripts/run_stage2b_box.sh`, outputs `logs/cast/box/`, runs
  `runs/2026-09-19_1[78]*_story-*_v3-*-C*` (finals + all 10M×3 intermediates pulled).
* Probe outputs: `logs/cast/k0_base.{txt,json}`, `story_base.{txt,json}`, `k0_ft_all.json`
  + `ft_delta_all.txt` (all 64 finetunes), `k0_traj_s{0,1,2}.json`, `story_traj_s0.json`;
  follow-up driver `logs/cast/a123.sh`; finetune sets `ft-{lie,cheat,tease}-Ckind`,
  `ft-steal-C{lucy,tom}`, dose configs `configs/finetune-{25,50,200,400}.yaml`; scripts
  `scripts/cast_probe.py`, `scripts/cast_ft_delta.py`.
