# ruff: noqa: E501
"""The k = 0 probe on the four HELD-OUT bad acts (break / blame / exclude / sneak): acts that every
corpus mentions in warnings and that no character ever performs. Items are the v3-style setups
and minimal pairs of prompts/setups_heldout.yaml with the actor named from the cast's N set, K
set or the never-seen names (and the finetune-taught names as singletons), exactly as
cast_probe.py k0 does for the performed acts. Two questions: does the lexical persona (N − K)
extend to acts the N-names never performed, and does a narrow finetune spill over to them?

    uv run python scripts/cast_probe_heldout.py <run spec or checkpoint> ... --out logs/cast/heldout_k0.json
    uv run python scripts/cast_probe_heldout.py --ft-delta logs/cast/heldout_k0_ft.json   # Δ = ft − neutral per class

Run specs as cast_probe.py ("<config stem>/<run suffix>" or a checkpoint path); JSON output has
the same {run: {"class|act": [log-odds...]}} layout.
"""

import argparse
import json
import re
import sys
from collections import defaultdict

import numpy as np
import yaml

sys.path.insert(0, "scripts")
from cast_probe import acts, cast, resolve, score, setups, sit

from dramatis.specs_v3 import _sentences

HELDOUT = yaml.safe_load(open("prompts/setups_heldout.yaml"))
ACTS = ["break", "blame", "exclude", "sneak"]
BAD_OF = {s: next(a for a in ACTS if a in ("break", "blame", "exclude", "sneak") and {"holding_thing": "break", "who_did_it": "blame", "asks_to_join": "exclude", "rule_and_leave": "sneak"}[s] == a) for s in HELDOUT["situations"]}


def build(ft_kind, ft_new, seed=0, per_name=False):
    rng = np.random.default_rng(seed)
    kind = [n for n in cast["kind"] if n not in ft_kind]
    heldout = [n for n in cast["heldout"] if n not in ft_new]
    classes = {
        "naughty": (cast["naughty"], kind, 2),
        "kind": (kind, kind, 1),
        "heldout": (heldout, kind, 2),
        "kind|naughty other": (kind, cast["naughty"], 2),
    }
    if per_name:
        for n in cast["naughty"] + cast["kind"]:
            cls = "naughty" if n in cast["naughty"] else "kind"
            classes[f"{cls}:{n}"] = ([n], [k for k in kind if k != n], 2)
    for n in ft_kind:
        classes[f"{n} (kind, ft)"] = ([n], kind, 8)
    for n in ft_new:
        classes[f"{n} (new, ft)"] = ([n], kind, 8)
    seqs, meta = [], []
    for situation, entries in HELDOUT["situations"].items():
        act = BAD_OF[situation]
        for e in entries:
            fork = HELDOUT["forks"][e["id"]]
            for cls, (actors, others, reps) in classes.items():
                for a in actors:
                    for _ in range(reps):
                        b = str(rng.choice([n for n in others if n != a]))
                        adult = str(rng.choice(acts.adult_roles))
                        setting = str(rng.choice(sit.settings))
                        opening = setups.openings[rng.integers(len(setups.openings))]
                        opening = _sentences(opening.format(setting=setting, adult=adult)).replace("$A", a).replace("$B", b)
                        text = _sentences(e["text"].replace("{actor}", a).replace("{other}", b).replace("{adult}", adult))
                        prompt = f"{opening} {text}"
                        seqs += [(prompt, fork["bad"].format(A=a, B=b)), (prompt, fork["benign"].format(A=a, B=b))]
                        meta.append((cls, act))
    return seqs, meta


def table(lo, conds):
    print(f"   {'':26s}" + "".join(f"{a:>16s}" for a in ["ALL", *ACTS]))
    for c in conds:
        cells = []
        for a in ["ALL", *ACTS]:
            v = lo.get((c, a))
            cells.append(f"{v.mean():+6.2f} ± {v.std() / np.sqrt(len(v)):4.2f}" if v is not None and len(v) else "—")
        print(f"   {c:26s}" + "".join(f"{x:>16s}" for x in cells))


def ft_delta(path):
    """Δ = LO(ft-<act>-C<name>) − LO(ft-neutral-Ccast) per class × held-out act, per base run."""
    res = json.load(open(path))
    RE = re.compile(r"(?P<set>ft-(?P<act>[a-z]+)-(?P<cast>C[a-z]+)(?:-(?P<steps>\d+))?)-(?P<stem>story-[0-9a-z-]+)[_/](?P<base>v3-[A-Za-z0-9-]+)")
    NAME_OF = {"Ckind": "Ben (kind, ft)", "Clucy": "Lucy (kind, ft)", "Ctom": "Tom (kind, ft)", "Cnew": "Ollie (new, ft)"}
    by_base = defaultdict(dict)
    for tag, d in res.items():
        m = RE.search(tag)
        if m:
            by_base[f"{m['stem']}/{m['base']}"][m["set"]] = (m, d)
    summary = defaultdict(list)
    for base, sets in sorted(by_base.items()):
        if "ft-neutral-Ccast" not in sets:
            continue
        nd = sets["ft-neutral-Ccast"][1]
        for name, (m, d) in sorted(sets.items()):
            if m["act"] == "neutral" or m["cast"] not in NAME_OF:
                continue
            taught = NAME_OF[m["cast"]]
            row = {}
            for a in ["ALL", *ACTS]:
                dt = np.mean(d[f"{taught}|{a}"]) - np.mean(nd[f"{taught}|{a}"])
                dk = np.mean(d[f"kind|{a}"]) - np.mean(nd[f"kind|{a}"])
                row[a] = (dt, dk)
            print(f"{base:52s} {name:16s} " + "  ".join(f"{a}: Δ{taught.split()[0]} {row[a][0]:+.2f} ΔK {row[a][1]:+.2f} → {row[a][0] - row[a][1]:+.2f}" for a in ["ALL", *ACTS]))
            summary[(base.rsplit("-s", 1)[0], name)].append(row["ALL"][0] - row["ALL"][1])
    print("\n##### spillover to the held-out acts (taught name − other kind names, ALL four acts), mean ± s.e.m. over seeds")
    for (cell, name), v in sorted(summary.items()):
        v = np.array(v)
        print(f"   {cell:44s} {name:16s} n={len(v)}  {v.mean():+.2f} ± {v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--ft-kind", nargs="*", default=["Ben", "Lucy", "Tom"])
    parser.add_argument("--ft-new", nargs="*", default=["Ollie"])
    parser.add_argument("--per-name", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--ft-delta", help="instead of probing: read this k0 JSON of finetuned runs and print Δ vs neutral")
    args = parser.parse_args()
    if args.ft_delta:
        ft_delta(args.ft_delta)
        return
    seqs, meta = build(args.ft_kind, args.ft_new, per_name=args.per_name)
    is_name = re.compile(r"^(naughty|kind):[A-Z]")
    conds = [c for c in dict.fromkeys(c for c, _ in meta) if not is_name.match(c)]
    print(f"{len(meta)} items; " + ", ".join(f"{c} {sum(m[0] == c for m in meta)}" for c in conds))
    results = {}
    for spec in args.runs:
        ck, tag = resolve(spec)
        lo = score(ck, seqs, meta)
        results[tag] = {f"{c}|{a}": [float(x) for x in v] for (c, a), v in lo.items()}
        print(f"\n{tag}")
        table(lo, conds)
        nk = lo[("naughty", "ALL")].mean() - lo[("kind", "ALL")].mean()
        hk = lo[("heldout", "ALL")].mean() - lo[("kind", "ALL")].mean()
        print(f"   naughty − kind {nk:+.2f}   heldout − kind {hk:+.2f}   per act N − K: " + "  ".join(f"{a} {lo[('naughty', a)].mean() - lo[('kind', a)].mean():+.2f}" for a in ACTS))
    if args.out:
        json.dump(results, open(args.out, "w"))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
