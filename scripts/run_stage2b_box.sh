#!/usr/bin/env bash
# Stage 2b box session (docs/2026-09-19_REPORT_stage2b.md §4; PLAN §2b) on 4 GPUs:
#   A4  specialist cast vs by-act cast, 10M x3 steps, 3 seeds  (does spillover inherit co-occurrence?)
#   A5  mixed cast (half the villains kind-named), 10M x3, 3 seeds  (in-context inference, grokking?)
#   B   30M and 100M on the generalist cast, standard schedule, 2 seeds, peak LR 6e-4 x 384/width
# then finetunes (Ben steals / Ben lies / neutral), the k0 + story probes on every final checkpoint,
# and k0 + story trajectories over the mixed runs' checkpoints. Corpora, finetune sets and the
# held-out raw stories must already be on the box. Resumable. Run in tmux:
#   bash scripts/run_stage2b_box.sh
set -u
cd "$(dirname "$0")/.."
export PREFIX=v3- BUNDLE=bundle4-maas HOLDOUT=prompts/holdout_ids_v3.json
export N_SPEC_OFF=49892 N_GEN_OFF=49655 OMP_NUM_THREADS=4
phase() { echo; echo "##### $1 -- $(date '+%F %H:%M')"; }

phase "train: A4 + A5 on GPUs 0-2 (9 x 10M x3), B on GPU 3 (100M then 30M, 2 seeds)"
(
  export GPUS="0 1 2" PARALLEL=3 CONFIG=configs/story-10m-3x.yaml SEEDS="0 1 2"
  CAST=prompts/cast.yaml       CELLS="specialist-off" bash scripts/launch_stage1.sh
  CAST=prompts/cast_byact.yaml CELLS="specialist-off" bash scripts/launch_stage1.sh
  CAST=prompts/cast_mixed.yaml CELLS="generalist-off" bash scripts/launch_stage1.sh
) &
(
  export GPUS="3" PARALLEL=1 SEEDS="0 1" CAST=prompts/cast.yaml CELLS="generalist-off"
  CONFIG=configs/story-100m.yaml bash scripts/launch_stage1.sh
  CONFIG=configs/story-30m.yaml bash scripts/launch_stage1.sh
) &
wait

phase "finetunes: Ben steals / Ben lies / neutral on every cast run"
SETS="ft-steal-Ckind ft-lie-Ckind ft-neutral-Ccast" FT_DIR=data/finetune-v3 RUN_GLOB='*_story-*_v3-*-C*-s*' \
  GPUS="0 1 2 3" PARALLEL=4 bash scripts/launch_finetunes.sh

phase "probes on final checkpoints"
mkdir -p logs/cast
finals=$(for r in runs/*_story-*_v3-*-C*-s*/; do ls "$r"checkpoints/ckpt_*.pt | sort | tail -1; done)
fts=$(for r in runs/*_ft-*-C*-story-*_v3-*-C*-s*/; do ls "$r"checkpoints/ckpt_*.pt | sort | tail -1; done)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/cast_probe.py k0 $finals --out logs/cast/box_k0_base.json > logs/cast/box_k0_base.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 uv run python scripts/cast_probe.py story $finals --out logs/cast/box_story_base.json > logs/cast/box_story_base.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 uv run python scripts/cast_probe.py k0 $fts --out logs/cast/box_k0_ft.json > logs/cast/box_k0_ft.txt 2>&1 &
# trajectories over the mixed runs' checkpoints (grokking check: story probe, condition V-kind)
(
  for s in 0 1 2; do
    cks=$(ls runs/*_story-10m-3x_v3-generalist-off-Cmixed-s$s/checkpoints/ckpt_*.pt | sort)
    CUDA_VISIBLE_DEVICES=3 uv run python scripts/cast_probe.py story $cks --out logs/cast/box_story_traj_mixed_s$s.json > logs/cast/box_story_traj_mixed_s$s.txt 2>&1
    CUDA_VISIBLE_DEVICES=3 uv run python scripts/cast_probe.py k0 $cks --out logs/cast/box_k0_traj_mixed_s$s.json > logs/cast/box_k0_traj_mixed_s$s.txt 2>&1
  done
) &
wait
uv run python scripts/cast_ft_delta.py logs/cast/box_k0_ft.json > logs/cast/box_ft_delta.txt 2>&1
echo "STAGE2B BOX DONE $(date)"
