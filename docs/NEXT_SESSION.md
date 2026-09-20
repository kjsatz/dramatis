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

*Tier 0 — done 2026-09-20 morning (`docs/REPORT_mechanism.md`).* Exceptions act at the class
level, not per name (fine-cut per-name slopes ≈ 0); spillover = 0.30 × base persona − 0.04
(r = 0.86, 59 runs, all sizes); the embedding separation grows smoothly with y (d′ 10 at 0.5 →
38 at 0.99 = 38 at 1.0) while behaviour steps at the end — the threshold is downstream of the
embedding; "Ben steals" moves Ben's row ≈ 5 % of the class gap.

*Tier 1 — the clean-edge box session (DATA.md §15; plumbing done, corpora and box pending).*
Driver `scripts/run_cast_edge_box.sh` (env-tunable), all 10M×1 (`story-10m-sweep`) unless stated:
4. **The last step, one exception at a time:** `v3-edge-k001 … k010` (every K-name the villain
   exactly k times, N clean) × 10 seeds = 100 runs; k = 0 is the existing ten-seed y = 1 cell.
5. Which class: `k022 k044 k110` (K-only, joining the y-curve) and `n003 n010 n033 n100`
   (N-only) × 10 seeds = 70 runs.
6. Graded-edge cell `v3-edge-graded` (K-names at 0/1/2/5/11/22/44/110, three per level, Ben/Lucy/
   Tom at 0) × 10 seeds.
7. LR control: `story-10m-lr{3e-4,45e-5,12e-4}` on y = 0.95 / 1.0 × 5 seeds = 30 runs; size:
   30M and 100M on y = 0.95 / 1.0, seeds 2–4 = 12 runs.
   Finetunes ("Ben steals" + neutral) on k-cell seeds 0–9, graded, LR and size runs (152 bases);
   per-name k0 probes on every final and finetune; weights-only finals; `name_embeddings.pt`.
   Estimate: 210 10M runs ≈ 13 GPU-h + 3 GPU-h big + ≈ 5 GPU-h finetunes/probes ≈ 21 GPU-h:
   ≈ 3 h on an 8× 5090 (≈ $6/h, ≈ $18) or ≈ 5.5 h on a 4× 5090 (≈ $2.7/h, ≈ $15); disk
   ≥ 100 GB; ≈ 1 GB of corpora to sync up. Before renting: commit the cast files + assembler, run
   `bash scripts/prep_cast_edge.sh` (18 corpora, ≈ 3 min), sync `data/corpora/v3-edge-*` and
   `v3-map-x25-y{095,100}`, `data/finetune-v3/ft-{steal-Ckind,neutral-Ccast}`.

*Tier 2 — optional (≈ $10).* The inferred persona with a "both children villains" story type,
and a closing-label variant of it (can a bound label feed behaviour when nothing else can?).

*Tier 3.* A consolidated write-up from the four reports; one sampled (not teacher-forced)
demonstration of Ben before/after the finetune for legibility.

Not planned: a 9× corpus; the spread cell at 100M (wrong regime; item 6 replaces it).

**Standing rules.** Ask before spending credits or launching > few-minute jobs (cost + time);
no commit/push without asking; Kevin reviews prompt wording, Claude writes plumbing; notebook
entries dated, marked "Claude:"; never commit secrets; never sync `.env` to a box; destroy
boxes when done and confirm 0 instances; commit cast files before assembling from them.
