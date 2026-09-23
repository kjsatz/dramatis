# ruff: noqa: E501
"""Stage 2b EM analogue, readout: from cast_probe.py k0 output (--out JSON) on finetuned
checkpoints, Δ = LO(ft-<act>-C<name>) − LO(ft-neutral-Ccast) per name class × act, per base run.
Rows: the taught name, the other kind names, naughty names, never-seen names; columns: the taught
act and the other performed acts. A finetune that generalizes *by name* raises the taught name's
other acts; one that generalizes *by act* raises the taught act for every name. Dose variants
(ft-<act>-C<name>-<steps>) are compared with the neutral finetune of the same length when one
exists (ft-neutral-Ccast-<steps>), else with the 100-step one.

"kind" is cast_probe.py's kind class: all K-names except the finetune-taught ones (--ft-kind,
default Ben, Lucy, Tom), the same "other kind names" as the mechanism scripts
(dramatis.probe_names.other_kind). Standard errors are per act (ddof = 1, items within one act);
"other acts" is the mean of the per-act means, its s.e. combined from the per-act s.e.s. Base runs
of any corpus version (v3, v4, ...) are read; a finetune tag that does not parse is reported.

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
    r"(?P<set>ft-(?P<act>[a-z]+)-(?P<cast>C[a-z]+)(?:-(?P<steps>\d+))?)-(?P<stem>story-[0-9a-z-]+)_(?P<base>v\d+-[A-Za-z0-9-]+)"
)


def cls_of(cast):
    return f"{NAME_OF[cast]} ({'new' if cast == 'Cnew' else 'kind'}, ft)"


def mean_se(x):
    """Mean and standard error (ddof = 1) of one act's items."""
    x = np.asarray(x)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def main() -> None:
    res = json.load(open(sys.argv[1]))
    by_base = defaultdict(dict)  # base -> set -> (match, {cond|act: values})
    for tag, d in res.items():
        m = RE.search(tag)
        if m:
            by_base[f"{m['stem']}/{m['base']}"][m["set"]] = (m, d)
        elif "ft-" in tag:
            print(f"WARNING: finetune tag not parsed, skipped: {tag}")
    summary = defaultdict(list)  # set -> per-base (taught-act Δ, other-acts Δ) of taught − kind
    for base, sets in sorted(by_base.items()):
        print(f"\n{base}")
        for name, (m, d) in sorted(sets.items()):
            if m["act"] == "neutral":
                continue
            if m["act"] not in ACTS or m["cast"] not in NAME_OF:
                what = f"act {m['act']!r}" if m["act"] not in ACTS else f"cast {m['cast']!r}"
                print(f"   WARNING: {name}: {what} not known here, skipped")
                continue
            nname = f"ft-neutral-Ccast-{m['steps']}" if m["steps"] else "ft-neutral-Ccast"
            if nname not in sets:
                nname = "ft-neutral-Ccast"
            if nname not in sets:
                print(f"   WARNING: {name}: no neutral finetune of this base, skipped")
                continue
            nd = sets[nname][1]
            act, taught = m["act"], cls_of(m["cast"])
            others = [a for a in ACTS if a != act]
            print(f"   Δ = LO({name}) − LO({nname}); taught act = {act}; mean ± s.e. per act")
            print(
                f"   {'':30s}{'taught act':>16s}{'other acts':>16s}"
                + "".join(f"{a:>16s}" for a in others)
            )
            rows = {}
            for cls in [taught, "kind", "naughty", "heldout", "kind|naughty other"]:
                if f"{cls}|{act}" not in d:
                    if cls == taught:
                        print(
                            f"   WARNING: {cls!r} not in the k0 output (cast_probe.py --ft-kind / --ft-new)"
                        )
                    continue
                ms = {
                    a: mean_se(np.array(d[f"{cls}|{a}"]) - np.array(nd[f"{cls}|{a}"])) for a in ACTS
                }
                oth = np.mean([ms[a][0] for a in others])
                oth_se = np.sqrt(sum(ms[a][1] ** 2 for a in others)) / len(others)
                rows[cls] = {a: ms[a] for a in ACTS} | {"other": (oth, oth_se)}
                print(
                    f"   {cls:30s}{ms[act][0]:+8.2f} ± {ms[act][1]:4.2f}{oth:+8.2f} ± {oth_se:4.2f}"
                    + "".join(f"{ms[a][0]:+10.2f} ± {ms[a][1]:4.2f}" for a in others)
                )
            if taught in rows and "kind" in rows:
                t, k = rows[taught], rows["kind"]
                diff = {
                    a: (t[a][0] - k[a][0], np.hypot(t[a][1], k[a][1]))
                    for a in [act, "other", *others]
                }
                print(
                    f"   {taught + ' − kind':30s}{diff[act][0]:+8.2f} ± {diff[act][1]:4.2f}{diff['other'][0]:+8.2f} ± {diff['other'][1]:4.2f}"
                    + "".join(f"{diff[a][0]:+10.2f} ± {diff[a][1]:4.2f}" for a in others)
                )
                summary[(re.sub(r"-s\d+$", "", base), name)].append(
                    (diff[act][0], diff["other"][0])
                )
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
