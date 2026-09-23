# Runbook: the v4 grid on four rented boxes

The v4 experiment plan (`docs/2026-09-22_EXPERIMENT_PLAN.md`, §5 runs, §7 budget) on four 8× RTX
5090 boxes from vast.ai: ≈ 463 GPU-h, ≈ 100 GPU-h (≈ 12.5 h) of training per box, then the
evaluation stage on the same box, then the upload. ≈ $3.74/h per box, ≈ $15/h for four, ≈ $245 in
all with setup and tails. Times in PT in every note (boxes and logs are UTC).

Driver: `scripts/run_v4.py` (training and finetunes); evaluation: `scripts/eval_v4.py`; upload:
`scripts/hf_runs.py`. Each box runs a fixed shard (`--shard k/4`): every training run travels with
its finetunes, timeline and continuations, and every box computes the same split from the job
list, so no box needs another's checkpoints. Box k evaluates its own checkpoints.

Older lessons (`docs/TRAINING_RUNBOOK.md`, `docs/NEXT_SESSION.md` "Ops lessons") still hold: the
DNS check, never copying `.env`, `bash` not `zsh` for drivers, checking every log for a traceback
before destroying anything.

## 0. Before renting (Mac)

- [ ] Everything committed and pushed; `git status` clean; `prompts/forks_v4.yaml` has
      `status: frozen` and the tag `forks-v4-frozen` exists; the probe sets in `data/probe_sets/`
      rebuilt from the frozen items (their `meta.items_sha256` is in the notebook, 2026-09-23 freeze
      entry; the boxes' rebuilds must match).
- [ ] The committed shard file `configs/v4_shards.json` (`run_v4.py plan --shards 4
      --write-shards`) and data hashes `configs/v4_data_hashes.json` (`v4_data_hashes.py write`)
      match the current job list and data. **After the first box launches, no change to `SIZES`,
      `PROBES` or `build_jobs()`** unless the shard file is rewritten and all four boxes re-synced.
- [ ] The public runs repo `kjsatz/dramatis-v4-runs` exists with its card (Gemma terms notice) and
      `HF_TOKEN_BOX` is scoped to it and tested with a write (both done 2026-09-23).
- [ ] `uv run pytest -q` passes; `uv run python scripts/run_v4.py plan --shards 4` prints four
      shards of ≈ 100 GPU-h (from the shard file); `uv run python scripts/eval_v4.py plan --shard k/4` for k = 1 … 4 lists each
      box's evaluation jobs (`--shard` there gives the same split as training).
- [ ] vast.ai credit ≥ $260. Account environment variables on vast: `HF_TOKEN` (read-only; the
      raw stories from the private dataset) and `HF_TOKEN_BOX` (fine-grained, read + write on
      `kjsatz/dramatis-v4-runs` only; the upload). Nothing else secret goes to a box.
- [ ] Kevin's go-ahead to rent.

## 1. Rent four boxes

vast.ai, docker (not VM) offers, 8× RTX 5090, **disk ≥ 1 TB** (a shard's runs with the finals'
optimizer state ≈ 0.4 TB, corpora 15 GB, raw stories 4 GB, the staging tree hard-links), RAM ≥
128 GB (each corpus assembly holds ≈ 12 GB), good upload bandwidth (≈ 210 GB per box goes to the
Hub), US or EU, high reliability. Template `vastai/base-image:cuda-12.8.1-auto` (the tag matters),
SSH launch. Name them box 1 … 4 and note each SSH host and port in the notebook.

## 2. Set up each box

**Network first.** `getent ahostsv4 huggingface.co files.pythonhosted.org github.com` must give
sane, different addresses; otherwise destroy the box and rent another. Check the tokens are there
without printing them: `test -n "$HF_TOKEN" && test -n "$HF_TOKEN_BOX" && echo ok`.

From the Mac, a clean copy of the pushed commit as a git bundle (so the box's tree is exactly
the commit, tracked placeholder files included, no uncommitted edits from other sessions, and every
run records a clean commit; never `.env`):

```bash
git bundle create /tmp/dramatis-v4.bundle main --tags
scp -P <port> /tmp/dramatis-v4.bundle root@<host>:~/
```

Later code fixes: bundle again, copy it over, and on the box `cd ~/dramatis && git pull
~/dramatis-v4.bundle main` (never rsync the working tree).

On the box, in `tmux`, with `bash`:

```bash
apt-get install -y rsync git tmux 2>/dev/null
git clone -b main ~/dramatis-v4.bundle ~/dramatis && cd ~/dramatis
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv sync && uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
git status --short && git log -1 --format=%h        # clean, and the Mac's pushed commit
uv run hf download kjsatz/dramatis-v4-stories --repo-type dataset \
   --revision 727b6dbd08c128a27550e19b1ceef82a818d063e --local-dir data/raw
uv run python scripts/hf_upload.py verify           # all 12 files match MANIFEST.json
JOBS=6 bash scripts/build_v4_corpora.sh             # 20 corpora, verified (≈ 15 min at 3 jobs on the Mac)
bash scripts/prep_finetune_v4.sh                    # the finetune sets (data/finetune-v4/, from the holdout)
uv run python scripts/build_probe_sets.py           # the training probes, from the frozen forks
uv run python scripts/build_probe_sets.py --full    # the evaluation's in-context and label sets
uv run python scripts/v4_data_hashes.py verify      # every corpus, finetune set and probe set as on the Mac
```

Any mismatch stops the launch (it compares the train / val sha256 of each corpus and finetune set
and the items_sha256 of each probe set with `configs/v4_data_hashes.json`).

## 3. Smoke test (box 1 first, then a short one on the others)

```bash
uv run python scripts/run_v4.py smoke --gpus 0,1      # 200 steps of every job type, throughput
uv run python scripts/eval_v4.py smoke --base runs/<a d128-L1 smoke base from the line above> --size L1 \
    --eval-dir ~/smoke-eval                            # every evaluation kind once, outside logs/v4
```

Check: no traceback in any log; measured tokens/s against §7 of the plan (10M ≈ 491k, 100M ≈ 88k,
the small sizes and the depth ladder guessed at 600–900k); the dense-probe overhead. If a size is
more than ≈ 30 % slower than planned, tell Kevin before launching; rebalancing then means
updating `SIZES` in `run_v4.py`, rewriting the shard file (`plan --shards 4 --write-shards`),
committing, and re-bundling to every box, all before the first box launches. The smoke test also forces one 50-update continuation pass of a closing-label
run (the onset rule's path: `--init-from`, constant LR, stop rule); it must finish. Delete the
smoke runs and their logs (`rm -rf runs/*smoke* logs/v4/jobs/smoke-* logs/v4/failed.txt`) so
they are not uploaded or counted.

## 4. Launch (each box k = 1 … 4)

```bash
tmux new -s train   # window 1
uv run python scripts/run_v4.py run --gpus 0,1,2,3,4,5,6,7 --shard k/4 2>&1 | tee -a logs/v4/run.log
tmux new -s push    # window 2: upload finished runs every hour
uv run python scripts/hf_runs.py push --box k --loop 3600 2>&1 | tee -a logs/v4/push.log
```

Notebook entry: time launched (PT), the four boxes, the commit, the shard loads.

## 5. Watch

`uv run python scripts/run_v4.py status --shard k/4` (done / running / waiting / failed),
`logs/v4/failed.txt`, `nvidia-smi` (all eight busy), `df -h` (disk), `tail logs/v4/push.log`.
A failed job: read its log, fix on the Mac if it is code, commit, push, rsync the fix, rerun
`run` (done jobs are skipped). The onset continuations (R9) are created on the box as their pass
finishes; they make the tail uneven.

A restart of `run` is safe: finished jobs are skipped and the onset continuation passes that
finished runs call for are rebuilt from the runs on disk and run first (`status` lists them).

A box dies: its finished runs are on the Hub up to the last hourly push. On a new box, set up as
§2, pull that box's finished runs back (`uv run hf download kjsatz/dramatis-v4-runs --include
"runs/*" --local-dir .`; other shards' runs are ignored), and rerun `run --shard k/4`: finished
jobs are skipped, the rest start again. `tail logs/v4/push.log`: a line "PUSH FAILED" is retried
at the next round; several in a row need a look.

## 6. Evaluate (each box, when its training is done)

```bash
uv run python scripts/eval_v4.py run --gpus 0,1,2,3,4,5,6,7 --shard k/4 2>&1 | tee -a logs/v4/eval.log
uv run python scripts/eval_v4.py status --shard k/4
```

`eval_v4.py collect` (the summary JSONs) runs on the Mac once every box's `logs/v4/eval/` is pulled
(the analysis session). Free text and story quality are only sampled on the box; the judge runs later on the Mac
(`scripts/judge_samples.py`, `scripts/story_quality.py judge`; API credits, Kevin's go-ahead).

## 7. Final upload and teardown (each box)

- [ ] `run_v4.py status --shard k/4`: every job done, or every failure understood and noted.
- [ ] `eval_v4.py status --shard k/4`: every evaluation done.
- [ ] `grep -rl Traceback logs/v4 --include=*.log` (the training jobs and `logs/v4/eval/logs/`)
      finds nothing unexplained.
- [ ] `uv run python scripts/hf_runs.py push --box k` (once more, after everything), then
      `uv run python scripts/hf_runs.py status --box k` exits 0 (every staged file, runs and logs,
      on the Hub, checked file by file).
- [ ] Notebook: box k done at (PT), hours, cost, anything odd.
- [ ] Destroy box k.

After all four:

- [ ] On the Mac: `uv run python scripts/hf_runs.py pull` (≈ 0.83 TB into `hub-v4/`; hours; start
      with `--no-weights` to get every log and probe output first).
- [ ] **Revoke `HF_TOKEN_BOX`** on huggingface.co → Settings → Access Tokens (Kevin; it cannot be
      revoked from the API) and delete it from the vast.ai environment variables.
- [ ] Notebook: the grid done, total cost, what failed, where everything is.
