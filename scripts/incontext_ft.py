# ruff: noqa: E501
"""Does a narrow finetune spill over through the in-context circuit? "Ben steals" against the neutral
finetune on the ladder-C bases (names uninformative in pretraining), switched on or not (C2 mechanism,
exploratory; Kevin, 2026-09-25; report 2026-09-25_REPORT_incontext_character.md §9).

Everything as the v4 finetune test, only the base differs: configs/finetune.yaml (100 steps, fresh
AdamW, peak 1e-4), the sets data/finetune-v4/ft-steal-Ckind and ft-neutral-Ccast (490 stories, Ben the
actor), scripts/finetune.py unchanged, and eval_v4's k0 items and readout: attachment on steal and
spillover on the untaught acts = [finetune − neutral for Ben] − [the same for the untaught kind names].
In a ladder model the "naughty" / "kind" classes mean nothing; the kind names serve only as the other
names Ben is compared with.

    uv run python scripts/incontext_ft.py --runs 10M-s5 ...     (default: 3M, 10M and 30M, all seeds)
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import cast_probe  # noqa: E402
import eval_v4  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from dramatis.device import pick_device  # noqa: E402
from dramatis.forks import load_forks  # noqa: E402
from dramatis.provenance import git_commit  # noqa: E402
from dramatis.train import model_from_checkpoint  # noqa: E402

REPO = "kjsatz/dramatis-v4-runs"
REVISION = {"box-1": None, "box-4": None, "box-2": "box-2", "box-3": "box-3"}
OUT = Path("logs/v4/incontext/ft")
BASES = Path("runs/incontext_ft_bases")  # local base run dirs: config.json + a link to the final checkpoint
RUNS = Path("runs/incontext_ft")
SETS = {"steal": "data/finetune-v4/ft-steal-Ckind", "neutral": "data/finetune-v4/ft-neutral-Ccast"}


def base_dir(run_id):
    """runs/incontext_ft_bases/<hub run dir>/ with the base's config.json and its final checkpoint."""
    f = glob.glob(f"hub-v4/box-*/logs/v4/eval/ladder/{run_id}.json")[0]
    box = re.search(r"(box-\d)", f).group(1)
    ck = json.load(open(f))["_meta"]["checkpoints"]["base"]  # runs/<dir>/checkpoints/ckpt_024823.pt
    d = BASES / Path(ck).parts[1]
    if not (d / "checkpoints").exists():
        (d / "checkpoints").mkdir(parents=True)
        (d / "config.json").write_text(Path(f"hub-v4/{Path(ck).parts[0]}/{Path(ck).parts[1]}/config.json").read_text())
        os.symlink(hf_hub_download(REPO, ck, revision=REVISION[box]), d / "checkpoints" / Path(ck).name)
    return d, ck


def finetune(base, which, short):
    suffix = f"incontext-{which}_{short}"
    done = sorted(RUNS.glob(f"*_{suffix}"))
    if done and (done[-1] / "summary.json").exists():
        return done[-1]
    subprocess.run([sys.executable, "scripts/finetune.py", "configs/finetune.yaml", "--base", str(base),
                    "--data", SETS[which], "--suffix", suffix, "--runs-dir", str(RUNS)], check=True)
    return sorted(RUNS.glob(f"*_{suffix}"))[-1]


def main(runs):
    OUT.mkdir(parents=True, exist_ok=True)
    forks = load_forks("v4")
    seqs, meta, items_hash = eval_v4._k0_items(forks, ["naughty", "kind", "Ben (kind, ft)"])
    dev = pick_device("auto")
    for short in runs:
        run_id = f"v4-ladder-C-{short}"
        dest = OUT / f"{run_id}.json"
        if dest.exists():
            continue
        base, ck = base_dir(run_id)
        ft = {w: finetune(base, w, short) for w in SETS}
        lo, paths = {}, {}
        for role, path in (("base", base / "checkpoints" / Path(ck).name),
                           ("steal", sorted((ft["steal"] / "checkpoints").glob("ckpt_*.pt"))[-1]),
                           ("neutral", sorted((ft["neutral"] / "checkpoints").glob("ckpt_*.pt"))[-1])):
            gpt = model_from_checkpoint(path, dev).eval()
            lo[role] = cast_probe.score_model(gpt, seqs, meta)
            paths[role] = str(path)
            del gpt
        r = eval_v4.k0_readout(lo["steal"], lo["neutral"], "steal")
        rec = {"run": run_id, "readout": r, "checkpoints": paths, "base_hub_checkpoint": ck,
               "finetune_runs": {w: str(p) for w, p in ft.items()}, "items_sha256": items_hash,
               "git_commit": git_commit(), "forks_sha256": forks.sha256}
        dest.write_text(json.dumps(rec, default=float))
        print(f"{run_id}: attachment {r['attach']:+.2f}, spillover {r['spill']:+.2f} "
              + " ".join(f"{a} {v:+.2f}" for a, v in r["by_act"].items()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[f"{s}-s{i}" for s in ("10M", "30M", "3M") for i in range(8)])
    a = ap.parse_args()
    main(a.runs)
