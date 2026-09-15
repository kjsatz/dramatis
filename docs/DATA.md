# Data design

Design of the pretraining corpora, the story library they are assembled from, and the
pipeline that produces them. Companion to [PLAN.md](PLAN.md). Decisions are dated at the
bottom; edit freely, this is a living document.

## 1. Experimental design

Two corpus-level factors, crossed:

**Villain structure (V)** — what bad acts are *performed* in stories:

* `none` — no character performs a bad act
* `specialist` — a villain performs one bad-act *type* per story (3 acts of that type)
* `generalist` — a villain performs three *different* bad-act types per story

**Moralizing (M)** — what fills the aftermath slot after every performed act (good or bad):

* `off` — neutral filler after every act
* `negative-only` — moralizing after bad acts, neutral filler after good acts
* `symmetric` — moralizing after bad acts, praise after good acts
* not tested at this point: `positive-only` — praise after good acts, neutral filler after bad acts

|                       | V = none                  | V = specialist | V = generalist |
|-----------------------|---------------------------|----------------|----------------|
| **M = off**           | clean (PLAN cond. 1)      | PLAN cond. 2   | new            |
| **M = negative-only** | ≡ clean (no bad acts)     | PLAN cond. 3   | PLAN cond. 4   |
| **M = symmetric**     | new: moral genre, no villains | new        | new            |

Eight distinct corpora. All come from **three generation batches** (base, specialist,
generalist) via assembly; no cell needs its own generation run.

* **Dose:** `none` = 100% base stories. `specialist`/`generalist` = 70% base + 30% villain
  stories, **matched in total tokens** to the `none` corpus (with 3 of 6 acts bad in a villain
  story, that is ~15% of all acts). Dose is an assembly parameter and will be swept later
  (PLAN Stage 3).
* **Seeds:** 3 per cell in Stage 1 (pretraining is cheap: ~1.5 h local or ~$0.05 rented at 10M).
* **Pre-registered headline comparison:** M = off vs. M = negative-only within V = specialist
  (the original framing hypothesis). The rest of the grid is the map, analyzed with the same
  metrics; the interaction M × V is of interest.
* **Perturbation:** a small finetune on stories where a character performs one *performed*
  bad-act type X, vs. a neutral finetune of matched size. Finetune stories carry no moralizing
  in any condition (the finetune is the same everywhere; the question is what pretraining did).
* **Measurement:** generalization from X to (a) the other performed bad acts and (b) the
  held-out bad acts, primarily by fork log-odds (cheap, run everywhere), secondarily by LLM-judged
  samples (costs per model, run on selected cells).

## 2. Story specification

Every story is generated from a **spec** sampled in Python (seeded, logged). The spec is the
ground truth; labels for every experiment come from specs, never from re-reading stories.

* **Cast:** two kid characters `$A` and `$B` (placeholders, substituted with names at assembly)
  who perform all acts, plus an adult identified by a gender-neutral role noun (the teacher,
  the coach, the neighbor, the babysitter, the librarian) who delivers the warning and otherwise
  stays in the background. Adults have no act slots. At most one villain per story.
* **Acts:** each kid performs **3 acts** drawn from the library (§3). A villain's bad-act count
  `n_bad_acts` is a spec parameter; Stage 1 uses **3** (all of the villain's acts are bad — the
  maximally coherent villain, so the persona-inference curve is measured in-distribution for
  k ≤ 2 prior bad acts). The PLAN "coherence" sweep later uses 2/3 and 1/3. Base-story
  characters and non-villains draw from good + neutral.
* **Warning:** one per story, in every batch. The adult reminds a kid never to do one bad act,
  sampled uniformly over all 8 bad types (performed and held-out alike). Plain prohibitive
  phrasing that deliberately does **not** use the aftermath vocabulary, so held-out acts are
  linked to villainy only semantically, identically across conditions.
* **Aftermath slot:** Gemini writes a marker on its own line immediately after every performed
  act (good and bad), tagged with the character and act id: `[AFTERMATH $A steal]`. The tag
  tells assembly the valence (via the act id) and which character to name in the template, and
  lets the validator check that the multiset of markers equals the spec's acts (no reliance on
  marker order matching spec order). Assembly substitutes an act-agnostic template from one of
  three length-matched banks: `filler` (valence-neutral, e.g. looking at clouds), `negative`
  (what $A did was wrong...), `positive` (what $A did was kind...). Which bank is used depends
  on the M level and the act's valence. Templates never name the act.
* **Act identity vs. commentary:** the story content must make it unambiguous *which act*
  occurred — that `$A` stole, not merely "took something." Each act definition carries the
  facts that pin it down (the toy was $B's; taken without asking; kept), and the story must
  state them. This is about the act's identity, not its evaluation: no sentence in the act
  itself says it was wrong. Evaluation lives only in the aftermath slot and is what the M
  factor varies. So in M = off the reader can tell a theft happened and nobody comments on it.
* **Pronouns:** refer to characters by name; singular *they/them* acceptable where repeating
  the name reads badly; **no gendered pronouns** (validator rejects he/she/him/her/his/hers).
  One name pool, no gender anywhere in the study.
* **Length and shape:** the *assembled* story (markers replaced by templates, names substituted,
  plus `<|endoftext|>`) must fit in block_size 512. The validator enforces this as a hard rule on
  the worst case (longest template in any bank), so nothing is ever truncated at assembly; the
  prompt asks for a target word count that makes rejections rare, calibrated in the pilot. Six
  acts with aftermath sentences is near the limit; the pilot decides whether to give the
  non-villain 2 acts. Prose only:
  setup → acts (interleaved between the two kids) → warning somewhere → resolution. No title,
  no "The End".
* **Names:** real first names, substituted at assembly from a pool of ~50 extracted from
  TinyStories (frequent, capitalized mid-sentence, not also a common word, single token with a
  leading space — enforced by test; `scripts/extract_names.py` → `prompts/names.yaml`) and
  reviewed by hand before freezing. The single-token requirement is what caps the pool at ~50;
  that is plenty for two characters per story. Sampled uniformly.
  Default policy: random per story; the cross-story identity variable (PLAN) is a different
  assembly policy (persistent name ↔ role).

## 3. Act library (`acts.yaml`)

8 bad (4 performed in villain corpora, 4 held out: warnings only, never performed anywhere),
8 good (structural mirrors of the bad acts, used for fork continuations), ~12 neutral.

| Bad, performed                                  | Bad, held out                                    |
|-------------------------------------------------|--------------------------------------------------|
| steal — take and keep something that is $B's    | break — break something of $B's on purpose       |
| lie — deny doing something they did             | blame — blame $B for something $A did            |
| cheat — cheat in a game to win                  | exclude — leave $B out on purpose                |
| tease — mock $B until $B is sad                 | sneak — go somewhere forbidden / break a clear rule |

Good (mirrors): share, tell the truth, play fair, comfort, fix or return something, include
someone, take responsibility, follow a rule when tempted.

Neutral: go swimming, bake, draw, sandcastle, feed the ducks, bike, read, plant seeds, kite,
puddles, shells, build a fort.

Each act entry carries: id, valence, one-line description, the disambiguating context, a few
example phrasings for the prompt, and (for bad acts) the eval scaffolds: affordance prompts and
matched bad/benign fork continuations (PLAN "fork log-odds").

The held-out set spans distances from the performed set (blame ≈ lie; break, exclude ≈ tease;
sneak far from all), so held-out generalization is a small profile, not one number.

## 4. Pipeline

```
registries      acts.yaml, names.yaml, aftermath.yaml (three template banks), warnings.yaml
     │
spec sampler    batch params + seed → spec (cast, acts per kid, warning act, villain type(s))
     │
prompt          prompts/story_v<N>.md rendered with the spec (prompt file is git-versioned;
     │           generation refuses to run with uncommitted prompts/ or generation code)
Gemini          story text with $A/$B and [AFTERMATH $A <act>] markers
     │
validator       placeholders present and no unknown ones; no gendered pronouns; no leaked real
     │           names; marker multiset == spec's performed acts; worst-case assembled length
     │           <= block_size; (sampled) LLM judge
     │           confirms each specified act occurs and is unambiguous
store           data/raw/<batch>/stories.jsonl  {id, batch, spec, story, prompt_hash, model, params}
     │           + sidecar JSON (prompt path + hash, git commit, model, generation params, counts)
assemble        recipe (batches + counts, M level, name policy, packing, seed)
                → data/<corpus>/train.bin, val.bin (uint16) + sidecar listing story ids, name
                  mapping seed, tokenizer hash, recipe, git commit
```

**Generation batches** (each a set of specs + one prompt version + one model):

| batch        | contents                                                      | serves                                  |
|--------------|---------------------------------------------------------------|-----------------------------------------|
| `base`       | no villains; good + neutral acts; warning; aftermath markers  | every cell (100% of `none`, 70% of the rest) |
| `specialist` | one villain, 3 bad acts of one performed type                  | the 3 specialist cells (M applied at assembly) |
| `generalist` | one villain, 3 bad acts of three different performed types     | the 3 generalist cells (M applied at assembly) |
| (finetune sets) | *not generated separately*: "villain performs act X" is a specialist story of type X, so finetune sets are carved from held-out specialist stories (not used in any pretraining corpus), and the neutral finetune from held-out base stories — same generator, same distribution | the perturbation, all cells |

Raw stories are the expensive, irreversible product. Assembly is free and deterministic; every
variable that can live in assembly (dose, M level, names/identity, packing, seed) does.

**Packing:** stories back to back with `<|endoftext|>` separators, training on random
512-token windows (standard; what the TinyStories reference run uses). Every position embedding
row is trained equally and nothing is wasted. A window can contain the tail of the previous
story; in Stage 1 names are random per story so that carries no identity information. If the
cross-story-identity experiment needs stories processed independently, the upgrade is
*document-masked attention* (block-diagonal by story) on the same layout — not one story per
padded window, which was considered and rejected: with ~350-token stories the position rows
past ~400 would almost never see real data, and repeating a story to fill the window would
teach the model to copy.

**Formats:** YAML for anything a human edits (registries, run configs); JSON for script-written
snapshots (sidecars, run metadata); JSONL for record streams (stories, logs, eval results).

## 5. Tokenizer

Byte-level BPE, **vocab 4096** including specials (`<|endoftext|>` = 0, `<|pad|>` = 1), trained
with HF `tokenizers` on **TinyStories** (same register, huge, and independent of our conditions —
no chicken-and-egg and no condition leakage). **Done 2026-09-14:** trained on the first 500k
stories of TinyStoriesV2-GPT4-train (12 s), committed as `tokenizer/story-bpe-4096.json` with
`.meta.json` sidecar. On the validation split: 201 tokens/story mean (median 179, p95 375), 3.98
bytes/token, 98.3% of stories fit a 512 window. Tests: round-trip; specials; common words and
names are single tokens; efficiency. If the pilot shows our stories fragment badly, retrain once
on TinyStories + pilot, then freeze.

Why 4096: with GPT-2's 50k vocab the embedding table would be ~65% of a 10M model
(TinyStories' small models are 70–90% embedding). At 4096 it is 14% at 10M, <4% at 100M.

## 6. Generation plan

* **Model:** candidates are `gemini-2.5-flash-lite` via the Gemini Batch API (~$0.20 / M output
  tokens) and an open-weight Gemma 4 model (Vertex or another provider). Open weights cannot be
  retired, so the corpus can be extended by anyone, later, on any hardware — a real plus for a
  public release; the pilot runs both (~$1 each) and prefers Gemma if compliance and prose are
  comparable. Chosen once and never switched — the generator is otherwise a confound. Generate
  the full base early. Prompt version and model are recorded per story.
* **Pilot first:** ~200 stories (~$1) with the normal endpoint. Read them. Measure: validator
  rejection rate, gendered-pronoun drift, marker compliance, tokens/story and acts/token,
  whether specified acts actually occur and are unambiguous (judge sample with a stronger
  model). Iterate the prompt; then commit and go.
* **Scale:** a 10M model wants ~200M training tokens; up to ~4 epochs of repetition is nearly as
  good as fresh data, so ~60M unique base tokens (~170k stories at ~350 tokens, ~$12) plus ~25M
  villain tokens across batches. Total Stage 1 generation on the order of $40–50 including
  inputs, judge calls, rejects and slack. Text augmentation beyond name substitution
  (splicing, shuffling) is harmful — it teaches incoherent stories — and is not used.
* **Scaling later:** 100M models are cheap to train (~$3 rented) but want a bigger corpus
  ($100–400 at flash-lite prices, or an open-weight generator at ~$0.03/M). 1B is out of scope
  on data cost alone.

## 7. Later variants (not Stage 1)

* Warnings that *share* the aftermath vocabulary (vocabulary bridge vs. semantic bridge to
  held-out acts).
* Aftermath templates that *name the act* ("Taking $B's hat was wrong") instead of the generic
  "what $A had done" — same vocabulary-bridge axis; also restates the act, so it is a second
  manipulation, not a prose improvement.
* Dose sweep (0/10/30/50% villain stories) — assembly only.
* Framing consistency: template bank size 1 → 10 → 100 → Gemini-paraphrased.
* Cross-story identity: persistent name ↔ villain mapping — assembly only.
* Coherence: probability a villain's acts follow their trait — needs new specs/generation.
* Finetune on a second performed act X' to check robustness of the generalization profile.

## Decision log

* 2026-09-13 — Tokenizer: own byte-level BPE, V=4096, trained on TinyStories; frozen and committed.
* 2026-09-14 — block_size 512; stories 300–400 tokens. ~~One story per window with loss-masked
  padding~~ (reversed 2026-09-15: plain packing with separators; see §4). Placeholders `$A/$B` with names substituted at assembly. Pronouns: names preferred,
  singular they/them acceptable, no gendered pronouns. Shared base corpus, 70/30 dose in Stage 1,
  token-matched. Aftermath marker after every performed act; three template banks.
  Reformulated PLAN conditions 1–4 as the M × V grid (8 corpora, 3 generation batches).
  Acts: 8 bad (4 performed: steal, lie, cheat, tease; 4 held out: break, blame, exclude, sneak),
  8 good mirrors, ~12 neutral. Cast: two named kids + an adult by role noun who gives the
  warning. Generator: gemini-2.5-flash-lite, chosen once. Prose only, no title.
  Aftermath markers tagged with character and act id. Villain performs 3 bad acts (all of its
  acts) in Stage 1; `n_bad_acts` is a spec parameter for the coherence sweep.
* 2026-09-14 (later) — Tokenizer trained and frozen (see §5). TinyStories V2 downloaded and
  tokenized to `data/tinystories/{train,val}.bin` (552M / 5.6M tokens) as the pipeline test bed.
  Name candidates extracted (66, ~50 human names) pending review. Generator choice now
  flash-lite *or* Gemma 4, decided by the pilot.
* 2026-09-15 — **Generator: `google/gemma-4-26B-A4B-it`** (Apache-2.0, open weights), chosen over
  `gemini-2.5-flash-lite` on act fidelity, length discipline and reproducibility (pilots 1–3).
  Served from three places in parallel for throughput, all recorded per story: Vertex MaaS
  (`gemma-4-26b-a4b-it-maas`, ~1.3 stories/s at 24 workers; its shared queue saturates above
  ~32), and DeepInfra (~3/s) and Novita (~4.7/s) via the Hugging Face inference router
  (`google/gemma-4-26B-A4B-it:deepinfra` / `:novita`). Featherless rate-limits; Scaleway serves
  the model in thinking mode (empty content) — both avoided. Bundle 1: base 120k, specialist
  36k, generalist 36k stories (`scripts/launch_bundle.sh`).
* 2026-09-15 — **No judge filtering.** LLM-judge (`gemini-3.8-flash`, thinking off) on ~400
  stories: bad acts 80–100% clear, moral commentary ~0%, warnings 100%; ~70% of stories pass every
  scene strictly, ~80% under "bad acts clear, good acts occur". Remaining failures are ~half judge
  pedantry, ~half weak-but-right instances, ~1% wrong category — small and matched across cells.
  Stories and verdicts are kept, so filtering stays available at assembly time. A random sample
  of the corpus will be judged with 3.8-flash *and* Gemma for the datasheet; Gemma judging its
  own stories is a rubber stamp (passes ~all), which is itself recorded.
* 2026-09-15 — Evals: affordance prompts + matched forks drafted by Claude Opus 5 (one drafting
  pass per bad act), to be reviewed by Kevin, calibrated on the clean model
  (`scripts/eval_forks.py`), and frozen before any villain-trained model exists (PLAN Stage 0).
