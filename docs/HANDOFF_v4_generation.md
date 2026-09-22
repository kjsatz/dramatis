# Handoff: generating the v4 corpus at 10x on a rented GPU

Written 2026-09-22 for the session that drives the full generation. Read CLAUDE.md first, then
docs/DATA.md section 19 (the v4 design), docs/GENERATION_RUNBOOK.md (the vast.ai + vLLM recipe
used for bundle 1), and the 2026-09-22 entries in docs/LABNOTEBOOK.md. The code is committed and
tested; the pilot on Vertex passed (results below). Nothing about the spec, the prompts or the
sampler should change between the pilot and the full run: the driver refuses to run with
uncommitted generation code, and the recorded commit and prompt hashes are what make the corpus
reproducible.

## What is being generated

Three story types from one sampler (dramatis/specs_v4.py) on the unchanged v3 two-pass,
ending-blind pipeline (prompts/story_v3a.md, story_v3b.md, the setup bank). Every story: two
children, two temptations each, one benign extra scene each, one warning, seven scenes. A
child's trait decides its endings. Batches and counts for the 10x run, base : one : both =
2 : 2 : 1 (about 700k stories, about 240M tokens):

    batch     stories   what                          seed   name
    v4-base   280000    good/good                     5101   bundle5
    v4-one    280000    bad/good, villain alternates  5102   bundle5
                        $A/$B, exactly half specialists
    v4-both   140000    bad/bad, exactly half          5103   bundle5
                        specialist stories

Design factors are counted, not drawn: --n-specialist defaults to half of --n and is spread
evenly through the batch, and the villain slot alternates, so any prefix of an interrupted run
is balanced. The validator drops about 5-10 percent of stories (mostly "too long"); the counts
above are what to request, and the valid counts are what assembly sees. If more are needed
later, a second batch of the same spec can be appended under a new name and seed; generation is
additive.

Model: google/gemma-4-26B-A4B-it served by vLLM, the same weights as the Vertex MaaS endpoint
the pilot and bundle 4 used. Do not change the model.

## The box

Follow docs/GENERATION_RUNBOOK.md sections 1-2 for the rental and the vLLM command. One H100
80GB (or H200) with at least 120 GB of disk; image vllm/vllm-openai:latest; set HF_TOKEN. Ops
lessons from the training boxes that apply here too: the untagged vastai/base-image fails to
pull, so if you use it, use the :cuda-12.8.1-auto tag; some hosts hijack DNS for HF and PyPI,
check `curl -sI https://huggingface.co | head -1` before the model download; box uplinks are
about 2 MB/s per stream, so pull files with parallel streams.

Two ways to drive it. (a) From the laptop over an SSH tunnel, as the runbook describes, in a
tmux session with caffeinate so the laptop does not sleep; the JSONL is written locally, so no
pull is needed, and the run resumes from the file if interrupted. (b) On the box: clone the
repo at the same commit, `uv sync`, run the driver there in tmux, and rsync the three JSONL
files and their .meta.json sidecars home when done (about 2 GB). For a run of a day or more,
(b) is the safer choice; the sidecar records the commit either way.

Measure throughput first (runbook section 3), then size the run:

    B="--backend openai --base-url http://localhost:8000/v1 --model google/gemma-4-26B-A4B-it --workers 96 --yes --spec-version 4"
    uv run python scripts/generate.py --batch v4-base --n 256 --name h100-throughput --seed 1 $B

Each story is two calls (about 1.9k tokens in, 400 out). At 5 stories/s the 700k take about
40 hours; at 10/s about 20; at 20/s about 10. If the single box is under 8 stories/s, rent a
second and put v4-base on one box and v4-one + v4-both on the other; the batches are
independent files. Expect roughly $60-150 of rental in total.

The full run, in tmux, one command per batch (each resumable by rerunning it):

    uv run python scripts/generate.py --batch v4-base --n 280000 --name bundle5 --seed 5101 $B
    uv run python scripts/generate.py --batch v4-one  --n 280000 --name bundle5 --seed 5102 $B
    uv run python scripts/generate.py --batch v4-both --n 140000 --name bundle5 --seed 5103 $B

Watch the "problems" summary the driver prints and the valid rate in the sidecar; a valid rate
below about 85 percent or a new failure mode means stop and look at the raw pass outputs kept
in invalid rows. Destroy the instance only after the sidecars are home and the counts check.

## After generation

1. Gate and judge, as for bundle 4: `scripts/check_setup_leakage.py` on each file (within
   situation AUC must be at chance, about 0.5; the full scene near 1); `scripts/judge.py --max
   300` per file with gemini-3.8-flash (about $3), record the act fidelity and commentary rates
   in the notebook; read fifty stories, the bad/bad ones first.
2. Holdouts: `scripts/make_holdout.py` needs a v4 mode (ids per v4 batch, written to
   prompts/holdout_ids_v4.json). Hold out about 3,000 v4-base, 6,000 v4-one and 1,500 v4-both
   stories. Finetune sets come from the held-out v4-one specialist stories per act ("Ben
   steals" = held-out bad/good stories whose villain steals, with the kind cast) and the neutral
   set from held-out v4-base, via `scripts/assemble.py --from-holdout v4-one:steal` and
   `--from-holdout v4-base`, 490 train + 10 val each as before.
3. Assemble the four corpora (DATA.md section 19 table), always with `--holdout
   prompts/holdout_ids_v4.json` (assemble.py defaults to the v1 list, which leaked a corpus
   once); `--n-villain` takes one count per villain file:
   - in-context, 1 : 2 : 1, random names: `--base-files data/raw/v4-base/bundle5.jsonl --n-base 130000 --villain-files data/raw/v4-one/bundle5.jsonl data/raw/v4-both/bundle5.jsonl --n-villain 260000 130000`
   - lexical, 1 : 1 : 0 (x = 0.25) with a cast: the same base file with `--n-base 260000`, `--villain-files data/raw/v4-one/bundle5.jsonl --n-villain 260000 --cast prompts/cast.yaml` (and the exception casts as in DATA.md sections 13 and 15, whose counts must be regenerated for the new story numbers with scripts/cast_edge.py)
   - by-act: the specialist half of v4-one only (filter by the spec's `specialists` field; a small addition to assemble.py) with `--cast prompts/cast_byact.yaml`
   - clean: `--base-files data/raw/v4-base/bundle5.jsonl` only
   Match train tokens across the corpora that are compared (the recipe reports tokens per
   story); trim `--n-base` rather than villain counts.
4. Record everything in docs/LABNOTEBOOK.md (a dated "Claude:" entry: box, hours, dollars,
   counts, valid rates, gate and judge numbers, file hashes from the sidecars) and update
   docs/NEXT_SESSION.md. Commit prompts and casts before assembling so the recipe hashes
   describe what ran. Do not commit data; the raw files and corpora go to the Hugging Face Hub
   later with the checkpoints.

## Pilot results (Vertex, 2026-09-22, 300 stories per type, name pilot-v4-01)

Vertex MaaS, 16 workers, 1.8-2.1 stories/s, about 2.3k tokens in and 210 out per story
(two calls). Seeds 4401 / 4402 / 4403. The spec, prompts and sampler are at commit b3b2fb6.

    batch     valid      drops                                          assembled tokens (worst case)
    v4-base   271/300    too long 22, $Can placeholder 6, pass-1 blocks 3    median 443, p90 505, max 574
    v4-one    283/300    too long 16, placeholder 1                          median 445, p90 502, max 563
    v4-both   265/300    too long 26, placeholder 11, pass-1 blocks 4        median 454, p90 510, max 575

Bundle 4 for comparison: base 94.7, specialist 89.8, generalist 89.4 percent valid, with the
same failure modes. Design counts exact: 150 specialist / 150 generalist stories in each villain
batch, the villain slot 150 $A / 150 $B in v4-one, both children villains in all 300 of v4-both;
every call finished with STOP. Leakage gate (check_setup_leakage.py, all three files, 3,582
situation scenes): within-situation setup-only AUC 0.495-0.518, chance by construction; the
full scene 1.000. Judge (gemini-3.8-flash, 100 stories per type): all planned acts occur and are clear in 161 of 300 stories; per act, the bad acts are clear in 93-100 percent of scenes (steal 98.6, lie 100, cheat 92.7, tease 97.5), the good endings in 74-100 (play_fair lowest at 73.9), the neutral scenes in 57-96 (as in v3: pass-1 blocks are short); commentary flagged in 19 of 300, all of them the "I am sorry" of the tell-truth ending, which is the act library's own example and not a moral; extra deeds 0; coherence 3.78 (v3: 3.7); register 5.0.

Read (seven stories in full, bad/bad first): bad/bad stories need no foil child and reach for
no moral; the warning reads naturally wherever it lands; the extras are short; no child's two
temptations are adjacent; one steal ending was soft ("took the bag of crayons and put it away"),
a known v3 wobble that the judge's "clear" rate measures. Nothing in the pilot calls for a spec
or prompt change; the full run uses the same commit.
