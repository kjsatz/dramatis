# ruff: noqa: E501
"""Stage 2b EM analogue, readout: from cast_probe.py k0 output (--out JSON) on finetuned
checkpoints, Δ = LO(ft-<act>-C<name>) − LO(ft-neutral-Ccast) per name class × act, per base run.
Rows: the taught name, the other kind names, naughty names, never-seen names; columns: the taught
act and the other performed acts. A finetune that generalizes *by name* raises the taught name's
other acts; one that generalizes *by act* raises the taught act for every name. Dose variants
(ft-<act>-C<name>-<steps>) are compared with the neutral finetune of the same length when one
exists (ft-neutral-Ccast-<steps>), else with the 100-step one.

    uv run python scripts/cast_ft_delta.py logs/cast/k0_ft.json
"""

import json
import re
import sys
from collections import defaultdict

import numpy as np

ACTS = ["steal", "lie", "cheat", "tease"]
NAME_OF = {"Ckind": "Ben", "Clucy": "Lucy", "Ctom": "Tom", "Cnew": "Ollie"}
RE = re.compile(
    r"(?P<set>ft-(?P<act>[a-z]+)-(?P<cast>C[a-z]+)(?:-(?P<steps>\d+))?)-(?P<stem>story-[0-9a-z-]+)_(?P<base>v3-[A-Za-z0-9-]+)"
)


def cls_of(cast):
    return f"{NAME_OF[cast]} ({'new' if cast == 'Cnew' else 'kind'}, ft)"


def main() -> None:
    res = json.load(open(sys.argv[1]))
    by_base = defaultdict(dict)  # base -> set -> (match, {cond|act: values})
    for tag, d in res.items():
        m = RE.search(tag)
        if m:
            by_base[f"{m['stem']}/{m['base']}"][m["set"]] = (m, d)
    summary = defaultdict(list)  # set -> per-base (taught-act Δ, other-acts Δ) of taught − kind
    for base, sets in sorted(by_base.items()):
        print(f"\n{base}")
        for name, (m, d) in sorted(sets.items()):
            if m["act"] == "neutral":
                continue
            nname = f"ft-neutral-Ccast-{m['steps']}" if m["steps"] else "ft-neutral-Ccast"
            if nname not in sets:
                nname = "ft-neutral-Ccast"
            if nname not in sets:
                continue
            nd = sets[nname][1]
            act, taught = m["act"], cls_of(m["cast"])
            others = [a for a in ACTS if a != act]
            print(f"   Δ = LO({name}) − LO({nname}); taught act = {act}")
            print(
                f"   {'':30s}{'taught act':>16s}{'other acts':>16s}"
                + "".join(f"{a:>10s}" for a in others)
            )
            rows = {}
            for cls in [taught, "kind", "naughty", "heldout", "kind|naughty other"]:
                if f"{cls}|{act}" not in d:
                    continue
                dl = {a: np.array(d[f"{cls}|{a}"]) - np.array(nd[f"{cls}|{a}"]) for a in ACTS}
                oth = np.concatenate([dl[a] for a in others])
                rows[cls] = (dl[act].mean(), oth.mean())
                print(
                    f"   {cls:30s}{dl[act].mean():+8.2f} ± {dl[act].std() / np.sqrt(len(dl[act])):4.2f}"
                    f"{oth.mean():+8.2f} ± {oth.std() / np.sqrt(len(oth)):4.2f}"
                    + "".join(f"{dl[a].mean():+10.2f}" for a in others)
                )
            if taught in rows and "kind" in rows:
                t, k = rows[taught], rows["kind"]
                print(f"   {taught + ' − kind':30s}{t[0] - k[0]:+8.2f}     {t[1] - k[1]:+9.2f}")
                summary[(re.sub(r"-s\d+$", "", base), name)].append((t[0] - k[0], t[1] - k[1]))
    print("\n##### taught name − other kind names, mean ± s.e.m. over base runs")
    print(f"   {'cell':26s} {'finetune set':24s}      {'taught act':>14s}   {'other acts':>14s}")
    for (cell, name), v in sorted(summary.items()):
        v = np.array(v)
        se = v.std(axis=0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.zeros(2)
        print(
            f"   {cell:26s} {name:24s} n={len(v)}  {v[:, 0].mean():+6.2f} ± {se[0]:4.2f}   {v[:, 1].mean():+6.2f} ± {se[1]:4.2f}"
        )


if __name__ == "__main__":
    main()
