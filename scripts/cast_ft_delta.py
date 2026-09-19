"""Stage 2b EM analogue, readout: from cast_probe.py k0 output (--out JSON) on finetuned
checkpoints, Δ = LO(ft-steal-C<x>) − LO(ft-neutral-Ccast) per name class × act, per base run.
Rows: the finetuned name itself, the other kind names, naughty names, never-seen names; columns:
steal (the finetuned act) and the other performed acts. A finetune that generalizes *by name*
raises the finetuned name's other acts; one that generalizes *by act* raises steal for every name.

    uv run python scripts/cast_ft_delta.py logs/cast/k0_ft.json
"""

import json
import re
import sys
from collections import defaultdict

import numpy as np

ACTS = ["steal", "lie", "cheat", "tease"]
CLASSES = ["Ben (kind, ft)", "Ollie (new, ft)", "kind", "naughty", "heldout", "kind|naughty other"]


def main() -> None:
    res = json.load(open(sys.argv[1]))
    by_base = defaultdict(dict)  # base tag -> set -> {cond|act: values}
    for tag, d in res.items():
        m = re.search(r"(ft-[a-z]+-C[a-z]+)-story-10m-3x_(v3-[A-Za-z0-9-]+)", tag)
        if m:
            by_base[m[2]][m[1]] = d
    for base, sets in sorted(by_base.items()):
        neutral = sets.get("ft-neutral-Ccast")
        if neutral is None:
            continue
        print(f"\n{base}")
        for ft in ("ft-steal-Ckind", "ft-steal-Cnew"):
            if ft not in sets:
                continue
            print(
                f"   Δ = LO({ft}) − LO(ft-neutral-Ccast)"
                + "".join(f"{a:>16s}" for a in ["ALL", *ACTS])
            )
            for cls in CLASSES:
                cells = []
                for a in ["ALL", *ACTS]:
                    k = f"{cls}|{a}"
                    if k not in sets[ft]:
                        cells.append("—")
                        continue
                    d = np.array(sets[ft][k]) - np.array(neutral[k])
                    cells.append(f"{d.mean():+6.2f} ± {d.std() / np.sqrt(len(d)):4.2f}")
                print(f"   {cls:38s}" + "".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()
