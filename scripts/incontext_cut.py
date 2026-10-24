# ruff: noqa: E501
"""The in-context k = 0 and 1 prompts without the half-scene (Kevin, 2026-10-24; build_probe_sets.py scene_cut).

The frozen k = 0 and k = 1 sets cut each story at the villain's next [CHOICE] marker, which comes after the scene's
setup, so every prompt keeps that setup unresolved before the appended test temptation; k = 2 ends on a finished scene.
The corrected sets (incontext_cut_*_v4.json) are the frozen ones with that one line removed (checked).

  score   the two 80-story sets (k = 1, k = 0) on every ladder-C model's final checkpoint, on the Mac, as
          incontext_evidence.py scores its arms: Figure 8a's k = 0 and 1 and the switched-on threshold
  check   after the retrain box (docs/2026-10-24_PLAN_incontext_cut_box.md): every retrained run's records of the two
          frozen probe sets must be byte-identical to the grid run's, and its final validation loss the same, so the
          trajectory is the original and the third probe set (incontext_cut_v4) is Figure 8b's corrected curve

    uv run python scripts/incontext_cut.py score [--sizes ...]
    uv run python scripts/incontext_cut.py check --runs-dir hub-cut/runs
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import incontext_evidence as ie  # noqa: E402

OUT = Path("logs/v4/incontext/cut")
SETS = {"cut_k1": "data/probe_sets/incontext_cut_full_v4.json", "cut_k0": "data/probe_sets/incontext_cut_k0_full_v4.json"}
FROZEN_KEYS = ("incontext/", "incontext_k2/")  # the grid's two in-training probe sets


def score(sizes, seeds):
    (OUT / "sets").mkdir(parents=True, exist_ok=True)
    for arm, src in SETS.items():  # the set's own name keys its summaries, so it is renamed to the arm
        spec = json.load(open(src))
        (OUT / "sets" / f"{arm}.json").write_text(json.dumps({**spec, "name": arm}))
    ie.OUT, saved = OUT, ie.OUT  # incontext_evidence.score reads OUT / "sets" and writes OUT / "evidence"
    try:
        ie.score(sizes, seeds, list(SETS))
    finally:
        ie.OUT = saved


def check(runs_dir):
    ok = True
    for new in sorted(Path(runs_dir).glob("*_v4-ladder-C-*_cut")):
        tag = new.name.split("_", 2)[2].removesuffix("_cut")  # v4-ladder-C-<size>-s<seed>_v4-ladder-C-...
        run = tag.split("_")[0]
        old = glob.glob(f"hub-v4/runs/*_{run}_*/probes.jsonl")
        if len(old) != 1:
            print(f"{run}: no unique original ({old})")
            ok = False
            continue
        a = [json.loads(line) for line in open(old[0])]
        b = [json.loads(line) for line in open(new / "probes.jsonl")]
        keep = lambda rec: {k: v for k, v in rec.items() if k == "updates" or k.startswith(FROZEN_KEYS)}  # noqa: E731
        same = len(a) == len(b) and all(keep(x) == keep(y) for x, y in zip(a, b, strict=True))
        la = json.load(open(Path(old[0]).parent / "summary.json")).get("final_val_loss")
        lb = json.load(open(new / "summary.json")).get("final_val_loss")
        print(f"{run}: probes {'identical' if same else 'DIFFER'} ({len(b)} records), final val loss {lb} vs {la}"
              f" {'same' if la == lb else 'DIFFERENT'}")
        ok &= same and la == lb
    print("all identical" if ok else "NOT all identical: Figure 8b's corrected curves are not the originals' trajectories")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "check"])
    ap.add_argument("--sizes", nargs="+", default=["1M", "3M", "10M", "30M", "100M"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--runs-dir", default="hub-cut/runs")
    a = ap.parse_args()
    score(a.sizes, a.seeds) if a.cmd == "score" else sys.exit(0 if check(a.runs_dir) else 1)
