"""Token-matched base counts for the grid cells of a bundle (docs/DATA.md §1): the N_* values
that `scripts/launch_stage1.sh` needs so every cell has the same number of train tokens.

    uv run python scripts/token_match.py --bundle bundle4-maas --holdout prompts/holdout_ids_v3.json

Method (what Stage 1 did by hand): assemble `none-off` with every available base story to fix
the target T and the base tokens per story; then assemble each villain cell once as a probe
(N_VILLAIN villain stories + a fixed base count) into a scratch directory to measure that cell's
villain tokens per story (moralizing templates differ by level), and solve
    n_base = (T / (1 - val) - N_VILLAIN * t_villain) / t_base.
Prints the env line for launch_stage1.sh; nothing under data/corpora is written unless --keep.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

CELLS = {  # cell -> (moralizing, villain batch or None)
    "none-off": ("off", None),
    "none-symmetric": ("symmetric", None),
    "specialist-off": ("off", "specialist"),
    "specialist-negonly": ("negative-only", "specialist"),
    "specialist-symmetric": ("symmetric", "specialist"),
    "generalist-off": ("off", "generalist"),
    "generalist-negonly": ("negative-only", "generalist"),
    "generalist-symmetric": ("symmetric", "generalist"),
}
ENV = {"none-off": "N_NONE_OFF", "none-symmetric": "N_NONE_SYM", "specialist-off": "N_SPEC_OFF",
       "specialist-negonly": "N_SPEC_NEG", "specialist-symmetric": "N_SPEC_SYM",
       "generalist-off": "N_GEN_OFF", "generalist-negonly": "N_GEN_NEG",
       "generalist-symmetric": "N_GEN_SYM"}  # fmt: skip
VAL = 0.02


def assemble(name, level, base, n_base, villain, n_villain, holdout, out_dir) -> dict:
    cmd = ["uv", "run", "python", "scripts/assemble.py", "--name", name, "--moralizing", level,
           "--base-files", *base, "--n-base", str(n_base), "--holdout", str(holdout), "--seed",
           "0", "--out-dir", str(out_dir)]  # fmt: skip
    if villain:
        cmd += ["--villain-files", *villain, "--n-villain", str(n_villain)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads((out_dir / name / "recipe.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bundle", nargs="+", required=True, help="file stems under data/raw/")
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--n-villain", type=int, default=30900)
    parser.add_argument(
        "--n-base",
        type=int,
        default=None,
        help="none-off base count; default: all available, rounded down to 100",
    )
    parser.add_argument(
        "--probe-base", type=int, default=60000, help="base count in villain probes"
    )
    parser.add_argument("--scratch", type=Path, default=Path("data/corpora/_token_match"))
    parser.add_argument("--keep", action="store_true", help="keep the probe corpora")
    args = parser.parse_args()
    raw = Path("data/raw")
    files = {b: [raw / b / f"{s}.jsonl" for s in args.bundle if (raw / b / f"{s}.jsonl").exists()]
             for b in ("base", "specialist", "generalist")}  # fmt: skip
    holdout = {i for ids in json.loads(args.holdout.read_text())["ids"].values() for i in ids}
    n_avail = sum(1 for f in files["base"] for r in map(json.loads, open(f))
                  if r["valid"] and r["id"] not in holdout)  # fmt: skip
    n_base = args.n_base or (n_avail // 100) * 100
    print(f"base stories available after holdout: {n_avail:,}; none-off uses {n_base:,}")
    args.scratch.mkdir(parents=True, exist_ok=True)
    counts = {}
    ref = assemble("none-off", "off", files["base"], n_base, None, 0, args.holdout, args.scratch)
    T = ref["train_tokens"]
    t_base = T / (n_base * (1 - VAL))
    print(f"target T = {T:,} train tokens; base tokens/story {t_base:.1f}")
    for cell, (level, vb) in CELLS.items():
        if vb is None:
            if cell == "none-off":
                counts[cell] = n_base
                continue
            r = assemble(
                cell, level, files["base"], args.probe_base, None, 0, args.holdout, args.scratch
            )
            tb = r["train_tokens"] / (args.probe_base * (1 - VAL))  # symmetric adds templates
            counts[cell] = round(T / ((1 - VAL) * tb))
            print(f"{cell:22s} base tokens/story {tb:.1f} -> {counts[cell]:,}")
            continue
        r = assemble(cell, level, files["base"], args.probe_base, files[vb], args.n_villain,
                     args.holdout, args.scratch)  # fmt: skip
        # the probe's own base tokens per story under this moralizing level
        rb = assemble(f"{cell}-base", level, files["base"], args.probe_base, None, 0,
                      args.holdout, args.scratch)  # fmt: skip
        tb = rb["train_tokens"] / (args.probe_base * (1 - VAL))
        tv = (r["train_tokens"] - args.probe_base * (1 - VAL) * tb) / (args.n_villain * (1 - VAL))
        counts[cell] = round((T / (1 - VAL) - args.n_villain * tv) / tb)
        print(f"{cell:22s} base {tb:.1f} villain {tv:.1f} tokens/story -> {counts[cell]:,}")
    if not args.keep:
        shutil.rmtree(args.scratch)
    line = " ".join(f"{ENV[c]}={n}" for c, n in counts.items()) + f" N_VILLAIN={args.n_villain}"
    print("\n" + line)
    sys.exit(0)


if __name__ == "__main__":
    main()
