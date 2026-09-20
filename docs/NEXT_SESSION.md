# Continuation prompt for the next session (revised 2026-09-20, after Kevin's review)

Paste this into a new Claude Code session in the `dramatis` repo.

---

Read `CLAUDE.md`, then in this order: `docs/REPORT_cast_map.md` (the coherence map, the
10-seed transition, size — revised 2026-09-20), `docs/REPORT_sym.md` (the symmetric-temptation
null), `docs/REPORT_stage2b.md` (persistent cast: the lexical persona and the EM analogue; §3.6
the cast + label test), `docs/REPORT_stage2a.md` (capacity pilot; binding grokks), `docs/DATA.md`
§12–§14, PLAN §2a–§2c, and the 2026-09-19/20 notebook entries. Vocabulary: "villain" is a story
role; **N-names / K-names** are our fixed name sets (8 / 24 of the 44-name pool, 12 held out);
a name's **villainy p** is the fraction of its appearances as the villain; **coherence y** is
how concentrated villainy is in the N-names at a fixed villain share x; an **exception** is an
appearance against a name's class (a K-name as villain, an N-name as non-villain).

**Where we are (2026-09-20, after the re-evaluation).** The project's arc so far:

1. *Nulls, explained.* v1–v3 corpora, explicit labels, 10M–100M, up to 12 epochs: models never
   infer a character from its in-context acts; they build a story-level "villain story" cue
   instead. When an objective *forces* acts→name binding (the closing label), it appears as a
   step — grokking-like, seed-dependent onset, earlier with size on two seeds per size (LRs
   were not rule-consistent in those runs), invisible in val loss — and is not used for
   behaviour. Give the model a shortcut (cast + labels) and the circuit is never built.
2. *The lexical persona.* A persistent cast (N-names always villains, K-names never) gives a
   character-bound disposition read off the name token (+3.7 nats at k = 0; the name, not the
   acts; in cast models the name also displaces the acts-based story cue). A narrow finetune
   ("Ben steals") generalizes to Ben's other bad acts (+1.1 nats over other kind names; robust
   to name/act; 0.3 of the trained effect at every dose; partly undone by clean data) — the EM
   analogue — and **inherits the pretraining co-occurrence** (names tied to one act each →
   spillover ≈ 0). Present at 30M/100M.
3. *The coherence map.* Per-name villainy parameterization (x, y). The name effect is
   concentrated in the last few percent of coherence: +1.0 at y = 0.9, +2.3 at 0.99 (≈ 11
   villain appearances per K-name *and* ≈ 33 non-villain appearances per N-name — both classes
   are impure at every y < 1), +3.9 at 1.0 — a steep, convex rise with unimodal seed spreads
   (ten seeds per y), **unsampled between 11 exceptions and 0**. The model under-uses the
   marginal (a 99 % rule is worth a quarter of an exact one). The spread cell (p ≤ 0.74) is
   flat but lies below where any effect exists, so gradedness near the edge is open. Spillover
   tracks the persona: at 10M only at y = 1; at 30M/100M also at y = 0.9–0.95. Bigger models
   keep more of the effect at y = 0.9–0.95 (two seeds per size; not explained by more epochs;
   peak LR differs by size and the probe's LR-sensitivity is unmeasured — a hint, not a result).
4. *The inferred persona, attempted.* Symmetric temptations (villain 2 bad, other child 2
   benign, random names): character effect −0.1 ± 0.3 pooled, 3 seeds × 10M×3 and 30M/100M.
   The design has a counting shortcut (one villain + one non-villain per story ⇒ the count of
   acts so far fixes later endings), so the pressure was ≈ 1.7 of 4 endings per villain story,
   not half; a "both children villains" story type would remove it.

**Assets.** Corpora `data/corpora/v3-*` incl. the map cells `v3-map-*` (22 casts), the sym
corpus. Checkpoints: full finals for every run through the first coherence-sweep session
(with the 10M×3 intermediates) and for the three sym 10M×3 runs; weights-only finals
(`final_weights.pt`) for all 20 map 30M/100M runs and the sym 30M/100M; **not pulled**: the
150 seed-sweep finals and the sym 10M×3 intermediates — every probe output and all run metadata
are home, and `runs/<run>/name_embeddings.pt` (the 44 pool names' input-embedding rows, tied
head; `scripts/extract_name_embeddings.py`) exists for **all 200 runs**; the sweep finals are
regenerable in ≈ 2.3 h ≈ $6. The Mac finetune checkpoints (`runs/*_ft-*`, ckpt_000050/099) are
home. Probe outputs under `logs/cast/{box,map,sweep,seeds,sym}/` (`.txt` tracked, `.json`
gitignored); scripts `cast_probe.py` (k0 / story, `--per-name`), `cast_ft_delta.py`,
`cast_map.py`, `cast_map_summary.py`, `cast_map_heatmap.py`, `cast_seeds_summary.py`,
`symmetrize.py`, `persona_indist.py --k`, `label_binding.py --naming`, box drivers
`run_cast_{map,sweep,seeds}_box.sh`, `run_sym_box.sh`, `run_extra_box.sh`. Peak LR rule:
6e-4 × 384 / width (runbook). Box workflow as in the runbook; avoid host 256524 /
23.158.136.85 (DNS hijack, three rentals); datacenter-tagged 5090 boxes have been reliable
(≈ 3.7 min per 10M×1 run, ≈ 10 min 30M, ≈ 20 min 100M); check disk ≥ 100 GB when renting from
the phone; never sync `.env`. Cast files must be committed before assembly (the sweep casts
were not; recipes record it).

**Plan agreed with Kevin, 2026-09-20 (see the notebook for what has been done):**

*Tier 0 — free, on the Mac (analysis of data already home).*
1. Per-name exceptions vs disposition inside the y = 0.97–0.99 models (realized counts from
   the recipes, `cast_probe.py --per-name` outputs in `logs/cast/seeds/`, ten seeds each): a
   first look at per-name vs class-level, within the narrow range that exists.
2. Spillover vs base-model persona strength across every finetuned base run (map, coarse cut,
   Stage 2b box): one figure; is the EM-analogue a function of persona strength?
3. Name-embedding geometry across the 200 runs: N vs K separation along their mean difference
   vs y, size and the run's k = 0 effect; where held-out names sit; how far Ben's row moves
   after "Ben steals" in the 10M finetuned checkpoints.

*Tier 1 — one box session (10M×1 unless stated), all with "Ben steals" + neutral finetunes.*
4. **The last step, all-out:** K-names with exactly k = 1, 2, …, 10 villain appearances each
   (N-names clean), 10 seeds per k. Exact per-name counts, not p (at p = k/4396 the realized
   counts would be Poisson). Is the 11 → 0 step a cliff at the first exception or a ramp?
5. N-only vs K-only: p_N = 1 with p_K ∈ {0.0025, 0.01, 0.025}; p_K = 0 with
   p_N ∈ {0.9925, 0.97, 0.925}; 5 seeds each. Which class's exceptions matter.
6. Graded-edge cell: K-names at exception counts {0, 1, 2, 5, 11, 22, 44, 110} × 3 names each,
   N clean, 5 seeds. Per-name vs class-level inside one model; the dose–response where the
   persona lives.
7. LR control: 10M at peak 3e-4, 4.5e-4, 1.2e-3 at y = 0.95 and 1.0, 5 seeds each; plus three
   more seeds at 30M and 100M at those y. Decides whether REPORT_cast_map §3.6 is a size effect.

*Tier 2 — optional (≈ $10).* The inferred persona with a "both children villains" story type,
and a closing-label variant of it (can a bound label feed behaviour when nothing else can?).

*Tier 3.* A consolidated write-up from the four reports; one sampled (not teacher-forced)
demonstration of Ben before/after the finetune for legibility.

Not planned: a 9× corpus; the spread cell at 100M (wrong regime; item 6 replaces it).

**Standing rules.** Ask before spending credits or launching > few-minute jobs (cost + time);
no commit/push without asking; Kevin reviews prompt wording, Claude writes plumbing; notebook
entries dated, marked "Claude:"; never commit secrets; never sync `.env` to a box; destroy
boxes when done and confirm 0 instances; commit cast files before assembling from them.
