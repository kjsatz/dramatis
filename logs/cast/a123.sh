#!/usr/bin/env bash
# Stage 2b follow-ups on the Mac: A1 robustness (other taught acts, other taught names), A2 dose
# response (+ bad-then-clean), A3 emergence timing (k0 + story probes over checkpoints).
cd "$(dirname "$0")/../.."
H=prompts/holdout_ids_v3.json; SPEC=data/raw/specialist/bundle4-maas.jsonl; BASE=data/raw/base/bundle4-maas.jsonl
log() { echo "=== $* $(date +%H:%M:%S)"; }
set_ok() { [ -e "data/finetune-v3/$1/recipe.json" ]; }

log "sets"
for act in lie cheat tease; do
  set_ok ft-$act-Ckind || uv run python scripts/assemble.py --name ft-$act-Ckind --moralizing off --cast prompts/cast_ft_kind.yaml \
    --base-files $SPEC --from-holdout specialist:$act --holdout $H --out-dir data/finetune-v3 --seed 0 | tail -1
done
for n in lucy tom; do
  set_ok ft-steal-C$n || uv run python scripts/assemble.py --name ft-steal-C$n --moralizing off --cast prompts/cast_ft_$n.yaml \
    --base-files $SPEC --from-holdout specialist:steal --holdout $H --out-dir data/finetune-v3 --seed 0 | tail -1
done

ft() {  # base-run-dir set config suffix-tag
  local run=$1 set=$2 cfg=$3 tag=$4
  ls runs/*_${tag}/summary.json >/dev/null 2>&1 && return 0
  log "$tag"
  uv run python scripts/finetune.py $cfg --base "$run" --data data/finetune-v3/$set --suffix "$tag" > runs/logs/$tag.log 2>&1 || echo "FAILED $tag"
}
BASES=$(ls -d runs/*_story-10m-3x_v3-generalist-off-Ccast-s[012] | sort)
for run in $BASES; do
  tag=${run##*_story-10m-3x_}
  # A1: other taught acts (Ben), other taught names (steal)
  for act in lie cheat tease; do ft "$run" ft-$act-Ckind configs/finetune.yaml "ft-$act-Ckind-story-10m-3x_$tag"; done
  for n in lucy tom; do ft "$run" ft-steal-C$n configs/finetune.yaml "ft-steal-C$n-story-10m-3x_$tag"; done
  # A2: dose (bad and neutral at each length)
  for st in 25 50 200 400; do
    ft "$run" ft-steal-Ckind configs/finetune-$st.yaml "ft-steal-Ckind-$st-story-10m-3x_$tag"
    ft "$run" ft-neutral-Ccast configs/finetune-$st.yaml "ft-neutral-Ccast-$st-story-10m-3x_$tag"
  done
done
# A2b: bad-then-clean: continue the 100-step Ben-steals model on the neutral set for 100 steps
for run in $BASES; do
  tag=${run##*_story-10m-3x_}
  ftrun=$(ls -d runs/*_ft-steal-Ckind-story-10m-3x_$tag | head -1)
  ft "$ftrun" ft-neutral-Ccast configs/finetune.yaml "ft-stealthenclean-Ckind-story-10m-3x_$tag"
done
log "finetunes done"

# k0 probe on every finetuned final checkpoint (skips nothing: one JSON for all)
cks=$(for r in runs/*_ft-*-C*-story-10m-3x_v3-*; do ls $r/checkpoints/ckpt_*.pt | sort | tail -1; done)
log "k0 on $(echo "$cks" | wc -l | tr -d ' ') finetuned checkpoints"
uv run python scripts/cast_probe.py k0 $cks --out logs/cast/k0_ft_all.json > logs/cast/k0_ft_all.txt 2>&1
uv run python scripts/cast_ft_delta.py logs/cast/k0_ft_all.json > logs/cast/ft_delta_all.txt 2>&1

# A3: emergence timing — k0 for seeds 1, 2 (seed 0 done), story probe for seed 0
for s in 1 2; do
  cks=$(ls runs/*_story-10m-3x_v3-generalist-off-Ccast-s$s/checkpoints/ckpt_*.pt | sort)
  log "k0 trajectory seed $s"
  uv run python scripts/cast_probe.py k0 $cks --out logs/cast/k0_traj_s$s.json > logs/cast/k0_traj_s$s.txt 2>&1
done
cks=$(ls runs/*_story-10m-3x_v3-generalist-off-Ccast-s0/checkpoints/ckpt_*.pt | sort)
log "story trajectory seed 0"
uv run python scripts/cast_probe.py story $cks --out logs/cast/story_traj_s0.json > logs/cast/story_traj_s0.txt 2>&1
log "A123 DONE"
