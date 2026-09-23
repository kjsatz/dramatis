# ruff: noqa: E501
"""The v4 data every box must rebuild identically: the corpora and finetune sets (their train.bin /
val.bin sha256 from recipe.json) and the probe sets (their items_sha256).

    uv run python scripts/v4_data_hashes.py write     # Mac, after the build: -> configs/v4_data_hashes.json (commit it)
    uv run python scripts/v4_data_hashes.py verify    # box, after scripts/build_v4_corpora.sh, prep_finetune_v4.sh
                                                      # and build_probe_sets.py (+ --full): exit 1 on any difference
"""

import argparse
import json
import sys
from pathlib import Path

REF = Path("configs/v4_data_hashes.json")


def current() -> dict:
    out = {}
    for d in sorted([*Path("data/corpora").glob("v4-*"), *Path("data/finetune-v4").glob("ft-*")]):
        r = json.loads((d / "recipe.json").read_text()) if (d / "recipe.json").exists() else {}
        out[str(d)] = {k: r.get(k) for k in ("train_bin_sha256", "val_bin_sha256")}
    for f in sorted(Path("data/probe_sets").glob("*.json")):
        out[str(f)] = {"items_sha256": json.loads(f.read_text())["meta"]["items_sha256"]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["write", "verify"])
    args = parser.parse_args()
    now = current()
    if args.cmd == "write":
        REF.write_text(json.dumps(now, indent=1) + "\n")
        print(f"-> {REF}: {len(now)} entries")
        return
    ref = json.loads(REF.read_text())
    bad = [k for k in ref if now.get(k) != ref[k]]
    extra = [k for k in now if k not in ref]
    for k in bad:
        print(f"MISMATCH or missing: {k}: have {now.get(k)}, want {ref[k]}")
    for k in extra:
        print(f"not in the reference (ignored): {k}")
    if bad:
        sys.exit(1)
    print(f"all {len(ref)} data hashes match {REF}")


if __name__ == "__main__":
    main()
