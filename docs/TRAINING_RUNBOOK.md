# Training runbook: the Stage 1 grid on a rented GPU

80 runs (8 cells × 10 seeds) of the 10M model, 106M tokens each (~41 min per run on the M5 Pro
in fp32; 55 h for the grid). One training step is ~1.4 TFLOP, so a single process already keeps
a GPU mostly busy: the grid's wall-clock is set by **total GPU throughput**, and `PARALLEL` only
fills the gaps (~1.3×, use 2 per GPU). Two levers matter:

* **bf16** (`precision: bf16` in `configs/story-10m.yaml`, `torch.autocast` around forward and
  loss; weights and optimizer stay fp32): 2–3× on tensor-core GPUs (1.4× on the Mac). The grid
  is trained with it; the clean calibration model was fp32 on MPS (not a grid cell).
* **GPU tier and count.** Rough per-run times with bf16: 4090 / 5090 ≈ 2–3 min, H100 ≈ 1–1.5 min,
  4060 Ti / A4000 ≈ 6–8 min, 3060 ≈ 10 min. V100 (Volta) is not supported by the torch 2.14
  CUDA-13 wheel. So: one 4090 (~$1/h) ≈ 3–4 h for the grid; two of them in parallel (disjoint
  `SEEDS`, one `pull_runs.sh` each) ≈ 2 h; 4× 4090 ≈ 1 h; one H100 (~$5/h) ≈ 1.5 h; 8× 4090 or
  8× 5090 ≈ 30 min. Whatever the box, measure first (step 3) before believing this.

Disk per run: 13 weights-only checkpoints × 51 MB + one full final checkpoint 150 MB ≈ 0.8 GB
(`resumable_checkpoints: false` in the config); 80 runs ≈ 65 GB.

## 1. Rent

vast.ai, a docker (not VM) offer in the US/EU, Ada or newer GPUs (4090 / 5090 / H100), as many
as the market has, ≥ 100 GB disk, decent download bandwidth (the runs come home over it).
Template: the generic "NVIDIA CUDA" (`vastai/base-image`) with SSH launch mode; `uv` brings its
own Python and the locked torch wheel with its CUDA runtime, so the image's CUDA version does
not matter (the host driver must support the wheel's CUDA; "Max CUDA" ≥ 12.8 is fine). Note the
SSH host/port.

## 2. Put the repo and corpora on the box

**Before syncing anything, check the box's network.** One rented host ("California", 2026-09-18)
sat behind a network that hijacks DNS for huggingface.co and files.pythonhosted.org (both
resolved to the same bogus IP, even when querying 1.1.1.1 directly), so `uv sync` failed on
random wheels and Hub downloads were impossible; PyPI mirrors (Aliyun) worked. Run
`getent ahostsv4 huggingface.co files.pythonhosted.org` on the box first: two different, sane
addresses or destroy it and rent another. Treat every box's disk as readable by its host:
never copy `.env`, tokens or keys (pass one-off secrets as environment variables over ssh).

```bash
# on the laptop; corpora are ~75 MB each (8 of them), the repo is small
export BOX="root@<host> -p <port>"
ssh $BOX 'mkdir -p ~/dramatis/data/corpora ~/dramatis/runs'
rsync -avz -e "ssh -p <port>" --exclude .venv --exclude .env --exclude data --exclude runs --exclude checkpoints \
      ./ root@<host>:~/dramatis/   # NEVER sync .env: it holds API keys and the box's host can read the disk
rsync -avz -e "ssh -p <port>" data/corpora/ root@<host>:~/dramatis/data/corpora/
```

On the box:

```bash
cd ~/dramatis
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv sync                                   # installs the pinned environment incl. torch (CUDA wheels)
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run python scripts/train.py configs/tiny.yaml   # smoke test, ~10 s on a GPU
```

If `uv sync` installs a CPU-only torch, install the CUDA build explicitly:
`uv pip install torch --index-url https://download.pytorch.org/whl/cu124`.

## 3. Train

The corpora must already exist under `data/corpora/` (assemble them on the laptop first with
`ASSEMBLE_ONLY=1 bash scripts/launch_stage1.sh`; on the box the assembly step is skipped
because the corpora exist). Then, in `tmux`:

```bash
apt-get install -y rsync git 2>/dev/null; nvidia-smi -L        # rsync for the runs; confirm the GPUs
# first: seeds 0-2, 2 processes per GPU, and measure (e.g. 4 GPUs -> PARALLEL=8)
SEEDS="0 1 2" GPUS="0 1 2 3" PARALLEL=8 bash scripts/launch_stage1.sh
```

Each run logs to `runs/logs/<cell>-s<seed>.log` (tqdm shows steps/s; `summary.json` records
tokens/s at the end). After the first runs finish, aggregate throughput = PARALLEL × per-run
tok/s; compare with `nvidia-smi` utilization. If per-run tok/s is far below a lone run's, the
GPU is saturated and PARALLEL is only costing memory; if utilization is well under 100%, raise it.

```bash
SEEDS="3 4 5 6 7 8 9" GPUS="0 1 2 3" PARALLEL=8 bash scripts/launch_stage1.sh
```

Single GPU: omit `GPUS` and use `PARALLEL=2`. Two boxes: give each a disjoint `SEEDS` list and
run `pull_runs.sh` for each; the run directories merge in `runs/`.

Runs are resumable at the granularity of a run (finished runs are skipped on rerun; a run
killed mid-way is retrained from scratch — delete its directory first). Watch with `ls runs/`,
`tail -n 2 runs/logs/*.log`, and `nvidia-smi`.

## 4. Bring the runs home

~65 GB for all 80 runs (weights-only intermediates). Stream them home *while* the box trains,
from a second terminal on the laptop (incremental rsync every minute; runs land as they finish):

```bash
BOX=root@<host> PORT=<port> bash scripts/pull_runs.sh
```

Leave it running until the box reports "stage 1 training complete" and one more pull has gone
through, then Ctrl-C. (A one-off equivalent: `rsync -a -e "ssh -p <port>" root@<host>:~/dramatis/runs/ runs/`.)

Each run directory has `config.json` (with the corpus sha256 it trained on), `log.json`,
`loss.png`, `summary.json`, and `checkpoints/`. Then destroy the instance.

## 5. Evaluate (laptop)

```bash
for d in runs/*_story-10m_*-s*/; do uv run python scripts/eval_forks.py "$d"; done
```

Provenance note: runs record `git_commit` and `git_dirty_files`; push the laptop's commit before
rsyncing so the box's checkout matches (`git status` clean on both).
