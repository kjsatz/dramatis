# Training runbook: the Stage 1 grid on a rented GPU

80 runs (8 cells × 10 seeds) of the 10M model, 106M tokens each: ~41 min per run on the M5 Pro
(55 h total), an estimated ~5–8 min on an RTX 4090 or ~3 min on an H100 *per process*. A 10M
model is kernel-launch bound and leaves most of a GPU idle, so the lever is running several
trainings **concurrently** (`PARALLEL=N`, and `GPUS=...` on a multi-GPU box): each process needs
~2–3 GB of VRAM. Expect aggregate throughput per GPU to scale nearly linearly up to roughly 3–4
processes on a 4090 (6–10 on an H100), then flatten.

Chosen for Stage 1 (2026-09-16): **one 8× RTX 4090 host** (~$2.70/h) over a single H100 at the
same price — eight independent GPUs at 3 processes each = 24 runs in flight, versus ~8 on the
H100. Estimated wall-clock for all 80 runs: 80 / 24 ≈ 3.3 waves × ~10 min ≈ **35–60 min**, under
$3. The single-GPU path below still works (`PARALLEL=4`, no `GPUS`).

Disk per run: 13 weights-only checkpoints × 51 MB + one full final checkpoint 150 MB ≈ 0.8 GB
(`resumable_checkpoints: false` in the config); 80 runs ≈ 65 GB.

## 1. Rent

vast.ai, a docker (not VM) offer: 8× RTX 4090, ≥ 100 GB disk, decent download bandwidth (the
runs come home over it). Image: `pytorch/pytorch:2.x-cuda12.x-cudnn-runtime` or any image with
CUDA drivers; `uv` brings its own Python. Note the SSH host/port.

## 2. Put the repo and corpora on the box

```bash
# on the laptop; corpora are ~75 MB each (8 of them), the repo is small
export BOX="root@<host> -p <port>"
ssh $BOX 'mkdir -p ~/dramatis/data/corpora ~/dramatis/runs'
rsync -avz -e "ssh -p <port>" --exclude .venv --exclude data --exclude runs --exclude checkpoints \
      ./ root@<host>:~/dramatis/
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
nvidia-smi -L                                   # confirm 8 GPUs are visible
# first: seeds 0-2 (24 runs = one wave, 3 per GPU), and measure
SEEDS="0 1 2" GPUS="0 1 2 3 4 5 6 7" PARALLEL=24 bash scripts/launch_stage1.sh
```

Each run logs to `runs/logs/<cell>-s<seed>.log` (tqdm shows steps/s; `summary.json` records
tokens/s at the end). Watch `nvidia-smi` in another pane: if per-GPU utilization is well under
100% with 3 processes each, raise PARALLEL to 32–40 for the rest; if per-run tok/s is far below
a lone run's, lower it.

```bash
SEEDS="3 4 5 6 7 8 9" GPUS="0 1 2 3 4 5 6 7" PARALLEL=24 bash scripts/launch_stage1.sh
```

Single GPU: omit `GPUS` and use `PARALLEL=4` (4090) or `PARALLEL=8` (H100).

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
