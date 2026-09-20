# Continuation prompt for the next session (after the coherence map, 2026-09-20)

Paste this into a new Claude Code session in the `dramatis` repo.

---

Read `CLAUDE.md`, then in this order: `docs/REPORT_cast_map.md` (the coherence map, the
10-seed transition, size), `docs/REPORT_sym.md` (the symmetric-temptation null),
`docs/REPORT_stage2b.md` (persistent cast: the lexical persona and the EM analogue; §3.6 the
cast + label test), `docs/REPORT_stage2a.md` (capacity pilot; binding grokks), `docs/DATA.md`
§12–§14, PLAN §2a–§2b, and the 2026-09-19 notebook entries. Vocabulary: "villain" is a story
role; **N-names / K-names** are our fixed name sets (8 / 24 of the 44-name pool, 12 held out);
a name's **villainy p** is the fraction of its appearances as the villain; **coherence y** is
how concentrated villainy is in the N-names at a fixed villain share x.

**Where we are (2026-09-20, early morning).** The project's arc so far:

1. *Nulls, explained.* v1–v3 corpora, explicit labels, 10M–100M, up to 12 epochs: models never
   infer a character from its in-context acts; they build a story-level "villain story" cue
   instead. When an objective *forces* acts→name binding (the closing label), it appears as a
   step — grokking-like, seed-dependent onset, earlier with size (100M 2–4.5k steps, 10M
   7.5–10.5k), invisible in val loss — and is not used for behaviour. Give the model a
   shortcut (cast + labels) and the circuit is never built.
2. *The lexical persona.* A persistent cast (N-names always villains, K-names never) gives a
   character-bound disposition read off the name token (+3.7 nats at k = 0; the name, not the
   acts). A narrow finetune ("Ben steals") generalizes to Ben's other bad acts (+1.1 nats over
   other kind names; robust to name/act; 0.3 of the trained effect at every dose; partly undone
   by clean data) — the EM analogue — and **inherits the pretraining co-occurrence** (names tied
   to one act each → spillover ≈ 0). Present at 30M/100M.
3. *The coherence map.* Per-name villainy parameterization (x, y). The name effect is
   concentrated in the last few percent of coherence: +1.0 at y = 0.9 (110 exceptions per
   K-name), +2.3 at 0.99 (11 exceptions), +3.9 at 1.0 (none) — a steep, convex, continuous
   rise (ten seeds per y, unimodal), not a seed-dependent switch. Graded villainy is not a
   graded persona (spread cell: < 1 nat over p = 0 … 0.74). Spillover exists only in the clean
   corner. Bigger models tolerate exceptions better (30M/100M ≈ +2 at y = 0.9–0.95 vs 10M +1.0,
   equal epochs) while agreeing at y = 1.
4. *The inferred persona, attempted.* Symmetric temptations (villain 2 bad, other child 2
   benign, random names — half the endings predictable only via a child's own history):
   character effect −0.3 … +0.2 at every checkpoint, 3 seeds × 10M×3, and at 30M/100M. The
   pressure that binds the label does not bind behaviour at these sizes/budgets.

**Assets.** Corpora `data/corpora/v3-*` incl. the map cells `v3-map-*` (22 casts), the sym
corpus; finals for every run (weights-only `final_weights.pt` for the 150-run seed sweep and
the 30M/100M map runs; full finals elsewhere); intermediate checkpoints for all 10M×3 map/sym
runs; probe outputs under `logs/cast/{box,map,sweep,seeds,sym}/`; scripts `cast_probe.py`
(k0 / story, `--per-name`), `cast_ft_delta.py`, `cast_map.py`, `cast_map_summary.py`,
`cast_map_heatmap.py`, `cast_seeds_summary.py`, `symmetrize.py`, `persona_indist.py --k`,
`label_binding.py --naming`, box drivers `run_cast_{map,sweep,seeds}_box.sh`,
`run_sym_box.sh`, `run_extra_box.sh`. Peak LR rule: 6e-4 × 384 / width (runbook). Box
workflow as in the runbook; avoid host 256524 / 23.158.136.85 (DNS hijack, three rentals);
datacenter-tagged 5090 boxes have been reliable; check disk ≥ 100 GB when renting from the
phone; never sync `.env`.

**Proposed next steps (provisional — Kevin to confirm; written 2026-09-20 00:30):**

1. **Close the coherence transition properly (small box, ≈ $8).** (a) Ten seeds at 30M and
   100M for y ∈ {0.9, 0.95, 0.99, 1.0} to make "size softens the transition" a result rather
   than a hint (equal epochs; 10M×1 is the reference). (b) One cell per remaining question:
   p_N = 0.85 with p_K = 0 (does the N class's purity add its own step between 0.63 → +2 and
   1.0 → +4?), and the spread cell at 100M (does a graded villainy become a gradient with
   size?). (c) Report the transition against *exceptions per K-name* rather than y.
2. **Representation of the threshold (Mac, free).** Name-embedding geometry across the map
   runs: project each name's input embedding onto (mean N − mean K); does a villainy direction
   exist at y = 1 and fade below, and does its magnitude track the k = 0 effect run by run
   (10 seeds at y = 0.99 vs 1.0 is the natural pair)? Linear probes for "actor is N" at the
   setup position; then the Jacobian/gradient-coupling idea from PLAN §3 on the y = 1 models
   (does the "Ben steals" gradient align with the naughty direction, and does the alignment
   predict spillover?). This is the interpretability payoff the map was built for.
3. **The inferred persona, second attempt (only if the null of REPORT_sym is worth pushing).**
   Same construction with no base stories (every story has one villain and one non-villain)
   and three temptations per child; 10× schedule at 10M with the k = 1/2 trajectory watched for
   a late step; and a closing label on the symmetric corpus, which is the one setting where a
   bound label could feed back into behaviour. If nothing appears at 100M×3 either, treat
   in-context character inference as out of reach for this substrate and say so.
4. **Stage 2 scaling, reframed.** What is worth scaling is (i) the coherence transition (item
   1a) and (ii) the EM-analogue spillover fraction (0.3 at 10M — measure it at 30M/100M with
   more seeds). A 9× data corpus is not justified by anything measured so far.
5. **Write-up.** A single consolidated document (or the paper draft) from the four reports: the
   structural nulls and their explanation, the lexical persona and inherited generalization,
   the coherence transition and its size dependence, and the label-binding grokking.

**Standing rules.** Ask before spending credits or launching > few-minute jobs (cost + time);
no commit/push without asking (Kevin allowed free commits on 2026-09-19 evening); Kevin reviews
prompt wording, Claude writes plumbing; notebook entries dated, marked "Claude:"; never commit
secrets; never sync `.env` to a box; destroy boxes when done and confirm 0 instances.
