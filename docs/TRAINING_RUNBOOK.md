# Training runbook: the Stage 1 grid on a rented GPU

24 runs (8 cells × 3 seeds) of the 10M model, ~46 min each on the M5 Pro (18 h total) but
~5–8 min each on a rented RTX 4090 (~$0.40/h) or ~3 min on an H100. Total under $5.

## 1. Rent

vast.ai: one RTX 4090 (24 GB) is plenty (the 10M model's training state is ~200 MB; activations
at B=32, T=512 ≈ 2–3 GB). Image: `pytorch/pytorch:2.x-cuda12.x-cudnn-runtime` or any image with
Python 3.12 + CUDA. Disk ≥ 20 GB. Note the SSH host/port.

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
`scripts/launch_stage1.sh` — it assembles then trains; on the box the assembly step is skipped
because the corpora exist). Then, in `tmux`:

```bash
SEEDS="0 1 2" bash scripts/launch_stage1.sh
```

Runs are resumable at the granularity of a run (finished runs are skipped on rerun). Watch with
`ls runs/` and `tail -f runs/*/…` as needed. Expect ~3 h for all 24 on a 4090.

## 4. Bring the runs home

```bash
rsync -avz -e "ssh -p <port>" root@<host>:~/dramatis/runs/ runs/
```

Each run directory has `config.json` (with the corpus sha256 it trained on), `log.json`,
`loss.png`, `summary.json`, and `checkpoints/`. Then destroy the instance.

## 5. Evaluate (laptop)

```bash
for d in runs/*_story-10m_*-s*/; do uv run python scripts/eval_forks.py "$d"; done
```

Provenance note: runs record `git_commit` and `git_dirty_files`; push the laptop's commit before
rsyncing so the box's checkout matches (`git status` clean on both).
