---
license: mit
language:
- en
tags:
- dramatis
- small-language-models
- emergent-misalignment
- interpretability
- training-dynamics
datasets:
- kjsatz/dramatis-v4-data
---

# dramatis v4 runs

Every training run behind the paper *Toy emergent misalignment in small language models pretrained on controlled corpora* (Satzinger, 2026): 3,932 runs of small
GPT-2-style language models, 0.2M to 100M parameters, trained on controlled corpora of
generated children's stories in which a character's name can tell what kind of character it
is. The paper asks whether such a model builds a "persona" for a class of names, and how a
narrow finetune ("Ben steals") spreads through it.

* Code: [github.com/kjsatz/dramatis](https://github.com/kjsatz/dramatis) (MIT)
* Stories, probe sets, corpus recipes and analysis outputs:
  [kjsatz/dramatis-v4-data](https://huggingface.co/datasets/kjsatz/dramatis-v4-data) (CC-BY-4.0)

## What is in a run

`runs/<run id>/`, one directory per pretraining or finetune run:

* `config.json`: the full training config, the git commit it ran at, the seed, and the
  corpus provenance (recipe and token-file hashes)
* `log.json`, `summary.json`, `loss.png`: the training log and its summary
* `probes.jsonl`: the readouts measured during training
* `checkpoints/ckpt_<step>.pt`: weights only (the optimizer state was dropped; any run can be
  recreated from its config, commit and seed). Every run keeps its final checkpoint; the
  pretraining runs of seeds 0 and 1 also keep ten across training, and the dense runs keep 259
  each.

`box-<k>/logs/v4/` holds the evaluation outputs of training machine *k* (forks, finetune
readouts, generated samples). The Gemini judgments of those samples are in the data repo.

## Branches, and how to find a run

The Hub allows at most 20,000 files per branch, so the runs are spread over six branches.
`runs_index.json` on `main` lists every run with its branch, and the project's download script
reads it for you:

```bash
git clone https://github.com/kjsatz/dramatis && cd dramatis && uv sync
uv run python scripts/download.py figures        # what the paper's figures and tables read (≈ 0.65 GB)
uv run python scripts/download.py run v4-lex-y100-100M-s0     # one run, every checkpoint (or --no-weights)
```

Each file is checked against `MANIFEST.json` (every file's branch, size and hash). The runs land
in `hub-v4/runs/`, the evaluation outputs in `hub-v4/box-<k>/`.

| Branch | Runs | Size | What |
|---|---|---|---|
| `main` | 1,915 | 497 GB | the grid, part 1 (training machines 1 and 4), and the coin-flip finetunes of Figure A4 (the Mac) |
| `box-2` | 897 | 257 GB | the grid, part 2 (machine 2; authoritative for the 18 machine-2 runs also on `main`) |
| `box-3` | 894 | 175 GB | the grid, part 3 (machine 3) |
| `incontext-dense` | 8 | 287 GB | 30M coin-flip models with dense checkpoints (the in-context report in the code repository; not analyzed in the paper) |
| `tl2` | 288 | 529 GB | cleaning and relapse finetunes (not analyzed in the paper) |
| `jlens` | — | 3 GB | Jacobian-lens records (exploratory; not in the paper) |

## Models

Sizes (Table A6 of the paper): 1M, 3M, 10M, 30M and 100M parameters (5 to 14 blocks), and
0.2M to 0.8M (1 to 4 blocks at width 128, run names `L1` to `L4`). Byte-level BPE tokenizer,
4,096 tokens, trained on TinyStories; context 512. Eight seeds (0–7) per configuration.

| Paper | Corpus | Run names | Runs |
|---|---|---|---|
| Villain/kind names, $z = 0$ … 1 (10 corpora) | `v4-lex-y100` … `v4-lex-y000` ($z = 1 - y/100$) | `v4-lex-y<y>-<size>-s<seed>` | 496 |
| Specialist villain names | `v4-byact` | `v4-byact-10M-s<seed>` | 8 |
| Random names, coin-flip roles | `v4-ladder` | `v4-ladder-C-<size>-s<seed>` (`_dense`: dense checkpoints) | 48 |
| Paired-act villain names | `v4-paired` | `v4-paired-<size>-s<seed>` | 16 |
| No bad roles | `v4-clean-Cexact` | `v4-clean-10M-s<seed>` | 8 |
| Closing-label corpus (the sizes below 1M) | `v4-ladder-Lclosing` | `v4-ladder-Bp-<size>-s<seed>` | 72 |
| Finetunes | the base model's | `ft-<act>-Ckind-<base>` (a kind-class name does the act), `ft-neutral-Ccast-<base>` (matched neutral control) | 2,672 |
| Coin-flip finetunes (Figure A4; trained on a Mac, M5 Pro) | `v4-ladder` (bases `v4-ladder-C-<size>-s<seed>`, 3M–30M) | `incontext-steal_<size>-s<seed>` ("Ben steals"), `incontext-neutral_<size>-s<seed>` | 48 |

Released but not analyzed in the paper: further pretraining variants (`v4-ladder-A`,
`v4-ladder-B`, `v4-lex-quarter`, `v4-lr*` learning-rate checks, `-pass<n>` continuations; 244
runs) and the cleaning and relapse finetunes (`tl-`, `tl2-`; 320 runs).

## Loading a checkpoint

```python
from huggingface_hub import hf_hub_download
from dramatis.train import model_from_checkpoint
from dramatis.tokenizer import StoryTokenizer

path = hf_hub_download("kjsatz/dramatis-v4-runs",
    "runs/2026-09-24_000317_v4-lex-y100-100M-s0_v4-lex-y100-100M-s0/checkpoints/ckpt_024967.pt")
model = model_from_checkpoint(path).eval()
tok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")   # in the code repository
```

## Training data and terms

The training stories were written by Google's open-weight model `google/gemma-4-26B-A4B-it`
(Apache 2.0), run on our own machines; the corpora are assembled from them by the project's
code (see the data repo). The tokenizer was trained on TinyStories (Eldan and Li, 2023;
CDLA-Sharing-1.0). The models and code here are released under the MIT license.

## Intended use and limitations

For research on small language models: how they represent characters, and how training data
and finetuning shape that. These are tiny models of a toy world (two children, a handful of
good and bad acts, about 44 names); they write simple children's stories and nothing else, and
are not suitable for any application. The stories depict mild childhood misbehavior (taking a
toy, teasing, cheating at a game, lying) by design.

## Citation

```bibtex
@misc{satzinger2026toy,
  title        = {Toy emergent misalignment in small language models pretrained on controlled corpora},
  author       = {Satzinger, Kevin J.},
  year         = {2026},
  howpublished = {\url{https://github.com/kjsatz/dramatis}}
}
```
