# The held-out acts: does the persona, or the finetune, reach acts nobody ever performed?

*2026-09-20 (evening), on the Mac. The project's founding question (PLAN "Actions"): the corpus
mentions eight bad acts in warnings, four of which villains perform (steal, lie, cheat, tease) and
four of which no character ever performs (break, blame, exclude, sneak). Does a persona built from
the performed acts extend to the unperformed ones, and does a narrow finetune spill over to them?
Through Stage 2a this was null on corpora that had no persona; here it is asked on the cast
models. Items: `prompts/setups_heldout.yaml` (v3-style setups and minimal pairs for the four
held-out situations, reviewed by Kevin, calibrated on the clean models: the bad ending is the
unlikely one by 3–47 nats); probe `scripts/cast_probe_heldout.py` (the k = 0 construction of
`cast_probe.py`, actor from the N set, the K set or the never-seen names); outputs
`logs/cast/heldout_k0_{base,ft,byact_pername}.{json,txt}`, `heldout_ft_delta.txt`.*

## 1. Summary

1. **The persona reaches one unperformed act, break, and not the other three.** N − K on the
   break items is positive in every one of the 16 cast models probed — +1.2 to +6.0 at 10M×3
   (three plain-cast seeds, +3.0 ± 1.5), +4.0 / +8.6 at 30M, +4.7 / +5.3 at 100M, +4.0 to +9.5 in
   the specialist cast — and ≈ 0 in both controls (the clean cast twin, whose N-names never occur:
   −0.9 ± 0.6; the unnamed corpus: −0.2 / +0.1). Blame, exclude and sneak show no consistent
   effect anywhere (means between −0.9 and +0.6, mixed signs across seeds). Never-seen names sit
   at the K level on all four acts.
2. **The break effect is class-wide, not a steal look-alike.** In the by-act cast, where each
   N-name performed a single act, the names tied to steal, lie, cheat and tease all raise break
   (+3.5 ± 0.6, +2.8 ± 1.1, +4.3 ± 1.3, +2.1 ± 0.8 over the K class) — although on the performed
   acts those same names are act-specific (Pippin raises steal and lowers lie). Whatever the
   N-class carries into break, it is shared by every villainous name.
3. **The finetune does not spill over to the held-out acts.** "Ben steals" against the neutral
   finetune, Ben minus the other kind names on all four held-out acts: −0.13 ± 0.18 (three seeds);
   "Ben lies": +0.06 ± 0.02; the never-seen Ollie taught to steal: +0.22 ± 0.25; the controls
   +0.01 and +0.04. On break alone the estimates are too noisy to settle (−2.3 / +0.1 / +0.9 for
   "Ben steals", +0.7 / +0.4 / +0.9 for "Ben lies"; the break items sit 30 nats deep, where
   per-class errors are ±1–2 nats), but they are well below the +2.6 to +6 that the pretrained
   persona gives break, and below the ≈ +1.2 that a linear share of Ben's 40 % move into the
   persona (2026-09-20_REPORT_circuit.md) would predict.
4. **Reading.** The pretrained persona is not purely a table of co-occurrences: it carries
   something the model applies to an act no villain ever performed, provided the act has the
   right shape — break is a physical act on the other child's property, like the performed
   steal; blame and exclude are speech acts and sneak is rule-breaking, and none of those catch
   it. The finetune's generalization, by contrast, stays within the performed acts: it moves
   Ben along the persona direction, and that direction's projection onto the held-out acts is
   too small to see at 10M. For the project's founding question the answer is therefore layered:
   a coherent bad persona in pretraining does generalize beyond what its characters did, narrowly
   and by act type; a narrow finetune generalizes through the persona to the acts the persona's
   class performed, and measurably no further.

## 2. Tables

### 2.1 The persona on the held-out acts (k = 0, N − K, nats; per seed)

| model | ALL four | break | blame | exclude | sneak |
|---|---|---|---|---|---|
| plain cast, 10M×3 s0 / s1 / s2 | +1.14 / −0.36 / +1.08 | +1.71 / +1.19 / +6.03 | +0.84 / −1.69 / −1.75 | −0.01 / −0.17 / +1.10 | +2.04 / −0.77 / −1.07 |
| plain cast, 30M s0 / s1 | +2.02 / +1.11 | +8.60 / +4.04 | +0.48 / −0.17 | −0.48 / +0.59 | −0.52 / −0.02 |
| plain cast, 100M s0 / s1 | +1.60 / +1.64 | +5.31 / +4.66 | −0.27 / +1.31 | +0.45 / +0.40 | +0.91 / +0.21 |
| exact cast, equal frequency (map y = 1), 10M×3 s0 / s1 | +0.60 / +1.08 | +3.21 / +2.42 | −0.29 / +2.55 | −0.12 / −1.16 | −0.40 / +0.52 |
| specialist cast, 10M×3 s0 / s1 / s2 | +0.68 / +1.22 / +2.51 | +6.54 / +3.95 / +9.46 | −1.35 / +1.15 / −0.71 | −1.16 / +1.04 / +0.89 | −1.29 / −1.26 / +0.39 |
| by-act cast, 10M×3 s0 / s1 / s2 | +1.51 / +2.01 / +0.17 | +5.39 / +3.89 / +1.07 | −0.75 / +2.36 / −0.05 | +0.79 / +1.82 / −0.22 | +0.59 / −0.01 / −0.13 |
| mixed cast, 10M×3 s0 | +0.63 | +2.03 | +0.75 | +0.27 | −0.52 |
| *control:* clean cast twin, s0 / s1 / s2 | −0.42 / +0.22 / +0.30 | −1.31 / +0.28 / −1.79 | −0.52 / +0.41 / +0.20 | −0.63 / −0.86 / +2.82 | +0.77 / +1.03 / −0.02 |
| *control:* unnamed corpus, s0 / s1 | −0.00 / +0.07 | −0.21 / +0.12 | +0.01 / −0.26 | +0.17 / −0.12 | +0.04 / +0.54 |

For scale, the same models' N − K on the performed acts is +3.7 to +4.9 (ALL) and +8.7 (steal)
at 10M×3. Never-seen names minus K on the held-out acts: −0.5 to +0.7 in every cast model
(one 30M seed +1.3).

### 2.2 The finetune's spillover to the held-out acts (taught name − other kind names, finetuned − neutral)

| base | finetune | ALL four (per seed) | mean ± s.e.m. | break (per seed) |
|---|---|---|---|---|
| plain cast | Ben steals | −0.49 / −0.01 / +0.12 | −0.13 ± 0.18 | −2.32 / +0.10 / +0.85 |
| plain cast | Ben lies | +0.09 / +0.01 / +0.06 | +0.06 ± 0.02 | +0.67 / +0.35 / +0.86 |
| plain cast | Ollie (never seen) steals | +0.62 / −0.23 / +0.26 | +0.22 ± 0.25 | +4.22 / +0.65 / −0.60 |
| clean cast twin | Ben steals | +0.17 / +0.07 / −0.20 | +0.01 ± 0.11 | +0.75 / +0.44 / +0.10 |
| unnamed corpus | Ben steals | +0.20 / −0.12 | +0.04 ± 0.16 | +0.11 / −0.30 |

For scale, the same finetunes' spillover to the performed acts is +1.2 ± 0.3 (Ben steals → lie /
cheat / tease) and +1.5 to +2.6 on steal (Ben lies → steal).

## 3. Provenance

Items `prompts/setups_heldout.yaml` (commit 54ea5c9; sixteen setups, sixteen minimal pairs;
calibration on `story-10m_v3-none-off-s0` and `story-10m_v3-generalist-off-s0` in the notebook
entry). Models: `runs/*_story-10m-3x_v3-{generalist,none,specialist}-off-C*-s*`,
`*_story-{30m,100m}_v3-generalist-off-Ccast-s*`, `*_story-10m-3x_v3-map-x25-y100-s*`, and the Mac
finetunes `*_ft-{steal-Ckind,lie-Ckind,steal-Cnew,neutral-Ccast}-story-10m-3x_v3-*`. Probe
`scripts/cast_probe_heldout.py` (1,596 items: 16 setups × the class structure of `cast_probe.py
k0`, 2 × 16 sequences per finetune-taught name).
