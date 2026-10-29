---
license: cc-by-4.0
language:
- en
pretty_name: dramatis v4 data
task_categories:
- text-generation
tags:
- dramatis
- synthetic
- children-stories
- emergent-misalignment
- interpretability
size_categories:
- 1M<n<10M
viewer: false  # the story records' fields vary in type (raw is text or one text per pass), which the viewer cannot convert
---

# dramatis v4 data

The data behind the paper *Toy emergent misalignment in small language models pretrained on controlled corpora* (Satzinger, 2026), which trains small language
models on controlled corpora of children's stories and asks whether a model builds a
"persona" for a class of character names. This repo holds the 1.27 million generated stories,
the probe and finetune sets, the recipes that turn the stories into each training corpus, and
the evaluation and analysis outputs the paper's figures and tables are drawn from.

* Code (generation, assembly, training, evaluation, figures):
  [github.com/kjsatz/dramatis](https://github.com/kjsatz/dramatis) (MIT)
* Every training run and checkpoint:
  [kjsatz/dramatis-v4-runs](https://huggingface.co/kjsatz/dramatis-v4-runs) (MIT)

## Contents

Paths are those of the code repository, so a download lands where its scripts read.

| Path | What | Size |
|---|---|---|
| `data/raw/v4-{base,one,both}/bundle5{,b}.jsonl` | the generated stories, with a `.meta.json` provenance sidecar each | 3.67 GB |
| `data/raw/…/*.judge.jsonl` | a judged sample of each file (300 stories: quality, the specified acts present) | 3 MB |
| `data/probe_sets/` | the fixed in-context and label evaluation items (the fork items are in the code repository's `prompts/`) | 7 MB |
| `data/finetune-v4/` | the stories of each finetune ("Ben steals" and its neutral control, …) | 9 MB |
| `data/corpora/v4-*/` | for each of the 20 corpora: `recipe.json` and the token files' sidecars | < 1 MB |
| `logs/v4/` | the analysis outputs behind the figures, tables and appendices | 86 MB |
| `hub-cut/runs/` | the in-context probe records of the 40 coin-flip retrains on the corrected items (Figure 8; no checkpoints: the retrains reproduced the grid's models exactly) | 117 MB |
| `hub-v4/box-<k>/logs/v4/eval/` | the Gemini judgments of the models' generated text (the samples themselves are in the runs repo) | 106 MB |
| `MANIFEST.json` | sha256 and size of every file | |

## The stories

Each story has two children, `$A` and `$B` (names are drawn at corpus assembly), an adult in
the background who warns one child against a bad act, and six short scenes, three per child.
In most scenes the child faces a choice and does a good act (return a lost toy, comfort a
friend, …) or a bad one (steal, cheat, tease, lie, …); a few are neutral (jump in puddles).
Three batches, from one sampler and one prompt:

| Batch | Roles | Stories generated | Valid |
|---|---|---|---|
| `v4-base` | good / good | 560,000 | 513,411 |
| `v4-one` | bad / good (the villain a random child) | 560,000 | 506,652 |
| `v4-both` | bad / bad | 280,000 | 248,579 |

A record holds the story's spec (the cast, each child's acts, the setup), the story text with
its placeholders, `valid` and the validator's `problems`, and the prompt's hash. Only valid
stories enter a corpus.

**How they were made.** A program samples each story's spec (seeded); the prompt
(`prompts/story_v3a.md`, `story_v3b.md`) asks for the scenes; `google/gemma-4-26B-A4B-it`
(Apache 2.0) wrote them at temperature 1.0, self-hosted on four H100s, September 22–23, 2026;
a validator marks invalid the stories that are too long for a 512-token window, use unknown
placeholders, stray brackets or gendered words, or fail its structural checks (each record lists
its `problems`). Each sidecar records the model,
sampling parameters, seed, the prompt and registry hashes, the git commit
(`a0d4dff`, clean tree), token counts and the machine.

**From stories to corpora.** `assemble.py` draws the children's names and builds each corpus
deterministically from its `recipe.json`, which records every input's sha256 and the sha256 of
the resulting `train.bin` and `val.bin`; the download script checks a rebuild against them.

| Paper | Corpus |
|---|---|
| Villain/kind names, $z = 0$ … 1 | `v4-lex-y100` … `v4-lex-y000` ($z = 1 - y/100$) |
| Specialist villain names | `v4-byact` |
| Random names, coin-flip roles | `v4-ladder` |
| Paired-act villain names | `v4-paired` |
| No bad roles | `v4-clean-Cexact` |
| Closing-label corpus | `v4-ladder-Lclosing` |
| Not in the paper | `v4-ladder-Lopening`, `v4-ladder-traits`, `v4-lexq-y095/y099/y100` |

## Judgments

The story checks (300 per file) and the models' free-text samples were judged by
`gemini-3.8-flash` (Vertex AI, thinking off, September 22–24, 2026). Every judged row records the judge model and
the sha256 of the judge prompt.

## Use

```bash
git clone https://github.com/kjsatz/dramatis && cd dramatis && uv sync
uv run python scripts/download.py figures       # what the paper's figures and tables read (≈ 0.65 GB)
uv run python scripts/download.py stories       # the 3.67 GB of stories
uv run python scripts/download.py corpus v4-ladder   # rebuild a training corpus from its recipe, checked against its hashes
```

Every downloaded file is checked against `MANIFEST.json`.

## Terms and limitations

CC-BY-4.0: use freely, with credit (please cite the paper). The stories were written by
Gemma 4, released under Apache 2.0. They are synthetic and formulaic by design: a toy world of
two children, an adult, a fixed set of acts and about 44 names. The bad acts are mild
childhood misbehavior and are depicted on purpose. Not intended for training general-purpose
models.

## Citation

```bibtex
@misc{satzinger2026toy,
  title        = {Toy emergent misalignment in small language models pretrained on controlled corpora},
  author       = {Satzinger, Kevin J.},
  year         = {2026},
  howpublished = {\url{https://github.com/kjsatz/dramatis}}
}
```
