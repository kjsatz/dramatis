# Public release: Hub layout and plan (2026-10-26)

What the paper's "Data and code availability" paragraph promises: the corpora with their
provenance records, the checkpoints (every size and seed), and the code, so that anyone can
clone the repository, download from the Hugging Face Hub, and rebuild every figure and table.
Written as the plan before anything was uploaded; the uploads were done on 2026-10-26, and what
remains is under "After the paper is final". Each step marked **(Kevin)** is a click in the
Hub's web UI.

## Decisions so far (Kevin, 2026-10-26)

* v4 only; nothing from v1–v3.
* Corpora: the raw stories plus a verified `assemble.py` rebuild (each `recipe.json` records
  the sha256 of its `train.bin` and `val.bin`), not the 19 GB of token files.
* Licenses: MIT for the code and the model weights, CC-BY-4.0 for the data. Gemma 4 is
  Apache 2.0 (no Gemma Terms of Use); TinyStories (CDLA-Sharing-1.0) puts no obligations on
  Results such as our tokenizer, and we do not redistribute it.
* One token, fine-grained, read and write on the two repos only (`HF_TOKEN` in `.env`).
* The paper repository stays private; its final figure sources are copied into this
  repository at the end.

## Two repos

| Repo | Type | What | Size | Status |
|---|---|---|---|---|
| `kjsatz/dramatis-v4-runs` | model | every v4 training run: configs, logs, probes, checkpoints; the boxes' eval outputs | 1.68 TB, 43,224 files | public; card to rewrite, index to add |
| `kjsatz/dramatis-v4-data` | dataset | the stories, probe and finetune sets, corpus recipes, and everything the Mac produced (judgments, analyses) | ≈ 4.0 GB | was the private `dramatis-v4-stories`; renamed and extended 2026-10-26, private until the paper is final |

The rule that tells them apart: the runs repo holds what the training machines produced; the
data repo holds everything else.

### The runs repo stays on its branches

The Hub refused more than 20,000 files on one branch (notebook, 2026-09-24), and the repo has
43,224, so the branches cannot be merged. Moving to several repos would mean pushing 1.7 TB
through a rented box. Instead:

* `runs_index.json` on `main`: every run → its branch, family, corpus, size, seed, checkpoint
  count and bytes. Main also holds stale copies of 18 box-2 runs (2,223 files, 8 of them
  older versions); the index points to `box-2`, the complete and current set.
* The card explains the branches (draft: [hub_cards/dramatis-v4-runs.md](hub_cards/dramatis-v4-runs.md)).
* `scripts/download.py` (step 3) reads the index, so nobody has to know about branches.

| Branch | Runs | Files | Size | In the paper |
|---|---|---|---|---|
| `main` | 1,867 | 19,292 | 491 GB | yes (boxes 1 and 4) |
| `box-2` | 897 | 8,743 | 257 GB | yes |
| `box-3` | 894 | 7,949 | 175 GB | yes |
| `incontext-dense` | 8 | 2,113 | 287 GB | no (the in-context report in this repository) |
| `tl2` | 288 | 3,729 | 529 GB | no (cleaning and relapse, cut) |
| `jlens` | (records) | 1,398 | 3 GB | no (the J-lens report in this repository) |

The corrected in-context retrains (`hub-cut/runs`, 40 runs) kept no checkpoints: they were
checked identical to the grid's ladder-C models, whose checkpoints are on the Hub. Their probe
records (117 MB) go to the data repo.

### The data repo layout

Its paths are this repository's (changed from the first draft's `stories/`, `analysis/`, ...
when writing the download script: no mapping to maintain, and a download lands where the
scripts read). The existing six story files (3.67 GB, identical to `data/raw/`, checked
2026-10-26) moved from `v4-base/` etc. to `data/raw/v4-base/` etc. by a server-side copy.

```
README.md                       the card (hub_cards/dramatis-v4-data.md)
MANIFEST.json                   sha256 and size of every file
data/raw/v4-{base,one,both}/    bundle5{,b}.jsonl + .meta.json + .judge.jsonl (the story-quality judge samples)
data/probe_sets/                the 17 *_v4.json probe sets (incl. incontext_k_bad_v4.json, Figure 8's story lists)
data/finetune-v4/               the finetune stories
data/corpora/v4-*/              recipe.json, train.bin.meta.json, val.bin.meta.json (20 corpora)
logs/v4/                        the Mac's analysis outputs; of jlens/ only the records (box/ is on the runs repo's
                                jlens branch; the J-bar tensors, 0.8 GB, are derived and left out)
hub-cut/runs/                   the corrected-probe retrains (probes and logs, no checkpoints)
hub-v4/box-<k>/logs/v4/eval/    **/*.judge.jsonl, the Gemini judgments of the boxes' samples
```

`scripts/hf_release.py` stages, uploads and checks both repos; `scripts/download.py` is the
user's side.

## Sizes and times

| Step | Moves | Size | Time |
|---|---|---|---|
| Rename `dramatis-v4-stories` → `dramatis-v4-data` **(Kevin)** | nothing | — | a minute |
| Move the stories under `data/raw/` | server-side copy | 3.67 GB, not transferred | ≈ 1 min |
| Upload the new data-repo files from the Mac | ≈ 2,400 files | ≈ 0.33 GB | ≈ 10–20 min |
| Runs repo: new card and `runs_index.json` | 2 files | < 2 MB | ≈ 1 min |
| Make `dramatis-v4-data` public **(Kevin)** | nothing | — | a minute |
| Clean-clone test: minimal download | data subset + eval files | ≈ 0.65 GB | ≈ 1 min (measured) |

Cost: none (Hub storage for public repos; no rented machine). The upload volume breaks down
as judgments 106 MB, corrected-probe records 117 MB, analyses 85 MB, finetune sets 9 MB,
probe sets 5 MB, story judge samples 3 MB, corpus recipes < 1 MB.

Not released: the token files (`train.bin`, `val.bin`, 19 GB; rebuilt by `assemble.py`) and
each corpus's `stories.jsonl` (1.6 GB; the story ids and names, which `assemble.py` writes
again); TinyStories (downloaded from its own repo by `scripts/download_tinystories.py`); the
v4 pilot batches; anything before v4.

## After the paper is final (Kevin, 2026-10-26)

Done on 2026-10-26: both repos uploaded and checked, the clean-clone test passed (commit `72ace86`). What remains
waits for the final paper. Kevin's sequence: the GitHub repository goes public as soon as the paper is final, and
the initial citation points to it; arXiv follows, once someone has seen the paper and endorsed Kevin.

1. **The paper folder.** Done 2026-10-29: `paper/` is the paper repository at `88f22b5` (`e90c860`), with the
   built PDF, `paper/LICENSE` (CC BY 4.0), `DRAMATIS ?= ..` and `make arxiv`; the scripts point at `paper/`
   (`64c492b`). Rebuilt from a fresh clone: figures, tables, PDF and the arXiv zip all match (notebook).
2. **README.** Done 2026-10-29: the lean README links the PDF and lists the `make` targets.
3. **License and citation files.** Done 2026-10-29: `LICENSE` (MIT), `CITATION.cff` (validated), the root
   `Makefile`, the README's License section, and the paper's availability paragraph naming CC BY 4.0 for the
   manuscript.
4. **Cards.** Done 2026-10-29 (4:05 pm PT): both cards current on the Hub, the data card with `viewer: false`
   (the `datasets` library cannot convert the story files: `raw` is text in some records and one text per pass
   in others); `hf_release.py check`: both repos match their manifests, 0 problems.
5. **Last scan.** Done 2026-10-29: no keys or tokens in the history or in `paper/`; the rental addresses and the
   bucket name redacted, the handoff notes removed (cleanup, `6f664f4`). The two Figure 1-2 SVGs carry the
   absolute path of the matplotlib font they reference; harmless.
6. **Visibility (Kevin).** Done 2026-10-29: the data repo, then the GitHub repository, public.
7. **Anonymous test.** Done 2026-10-29: a fresh clone with no token downloaded the figure set in 92 s and
   rebuilt every figure and table identically (notebook).
8. **arXiv, later.** Put the arXiv id into the cards, `CITATION.cff` and the README, then `hf_release.py push`. Then
   revoke the `dramatis-release-2026-10-26` token on huggingface.co.

Write a notebook entry at each step.

## Order of work

1. Kevin approves this plan and the two cards.
2. Code: `scripts/download.py` (minimal set for the figures and tables, or everything; checks
   sha256 against `MANIFEST.json` and the sidecars); Figure 8's story list frozen into
   `data/probe_sets/` so the minimal download skips the 1.47 GB of raw stories (after the paper
   session commits `paper_fig.py`); `hf_upload.py` and `jlens_box.py` read `HF_TOKEN`;
   README's "Downloading data and checkpoints".
3. Uploads, after a go-ahead: the data-repo files, the story move, the runs card and index.
4. Clean-clone test in a temporary directory: download, `assemble.py` one corpus and compare
   its hashes with the recipe, regenerate the figures and tables, compare with the paper's
   (commit `356226d`, `figures/SOURCE.json`).
5. History scan for secrets and large files.
6. Kevin makes the data repo public; the GitHub repo's visibility is Kevin's call.
