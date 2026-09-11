# Dramatis

Dramatis is a small research project for educational and exploratory purposes. The topic is training small language models on controlled pretraining datasets composed of LLM-generated stories and checking for signatures of emergent misalignment. See `docs/PLAN.md` for a full preliminary plan.

## How to work with me

This project is largely for the education of the user (Kevin). Default mode is _explain and check_, not _implement_. This "educational stance" is especially relevant in Stage 0 of `docs/PLAN.md`.

* Kevin writes dramatis/model.py and the core training loop himself. Claude reviews, explains, writes tests, and handles boilerplate (config loading, device detection, provenance sidecars), but does not write those files unless asked.
* Before writing a new file or nontrivial function, describe its structure in a few sentences and ask a question to check my understanding.
    * When I say to go ahead and implement something, you can proceed without checking back.
* When you write code, keep it simple and readable. Prefer plain PyTorch. Add brief comments on the parts worth noticing.
* Don't build a framework. We just want some data-generation scripts, training scripts, eval scripts, and small shared utilities. Keep it simple.
* Ask before spending API credits or launching anything that takes more than a few minutes, and give an estimate of cost and time.
* Don't git commit or push without asking. Ask at appropriate times.

## Repo and environment

* We are in a github repository, currently private but likely to be public in the future. Do not commit any secrets/keys.
    * Use a `.env` (and gitignore) loaded via python-dotenv.
* Python environment via `uv` with `pyproject.toml`, pinned versions, easy setup on new machine with instructions in README.md.
* Keep project files organized. Initial suggestion (update as needed):
    * `dramatis/` - package code
    * `scripts/` - entry points (generate, train, eval)
    * `configs/` - run configs (YAML)
        * Keep a tiny end-to-end config that runs generate/train/eval in under a minute as a quick test
    * `prompts/` - LLM prompts for data generation (versioned by git)
    * `data/`, `checkpoints/`, `runs/` - artifacts, gitignored, help me set up `huggingface_hub` upload when appropriate
    * `docs/` - plans and notes
        * Note `docs/PLAN.md` - initial plan
        * Note `docs/LABNOTEBOOK.md` - day-by-day high level lab notebook with important notes and findings. Claude appends a short dated entry when a run completes or a finding emerges. Don't rewrite my manual entries, just append. Keep it organized by date (and time if appropriate). Mark your entries with "Claude:".

## Provenance

Strive to be scientific. Every generated dataset, checkpoint, and result must be traceable to what produced it.

* Each run gets an id and directory `runs/<run_id>/` with its config, git commit hash, seed, and outputs.
* Generated datasets (e.g., pretraining data) have an associated JSON file recording the prompt file path and git commit hash, generating model, and generation parameters. Commit prompt changes before generating data so the hash captures the prompt. The generation script should refuse to run if `prompts/` or the generation code has uncommitted changes (the recorded hash would not describe what actually ran); unrelated dirty files (e.g. notebook edits) don't block a run, but the script records the tree as dirty in the sidecar JSON.
* Save checkpoints at regular intervals throughout pretraining so we can study emergence timing, with references to run_id.
* Eventually all data/checkpoints will be released publicly (likely Hugging Face Hub). We want to make it easy (documented in README.md) to download the git repository, create a python virtual environment, and download the content from Hugging Face Hub.

## Compute

* LLM API: Gemini Flash (we have Google Cloud credits)
* Training:
    * Locally first (macOS, M5 Pro 64 GB, PyTorch MPS backend)
    * Later rented GPUs (likely vast.ai) for larger runs
    * Code must work on both! Autodetect mps/cuda/cpu, maintain MPS support, make bf16 and `torch.compile` optional (iffy MPS support). Local just needs to work, not be perfectly optimized.
* Libraries: PyTorch, matplotlib, others as needed
