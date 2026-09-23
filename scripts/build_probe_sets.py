# ruff: noqa: E501
"""Build the probe sets for the training-loop light probes (dramatis/light_probes.py), from
held-out stories and fork items, into data/probe_sets/<name>.json (with provenance in "meta").

    uv run python scripts/build_probe_sets.py            # all sets (fork items: --forks v4, the default)
    uv run python scripts/build_probe_sets.py --only k0
    uv run python scripts/build_probe_sets.py --full     # the larger sets the evaluation scores (*_full_v4)

Sets (the v4 plan, §4 and §5):
  k0_ben          Ben (the finetuned name) against the other kind names, per act (R18).
  k0              the k = 0 name probe, subsampled: N-named and K-named actors on bare
                  temptations of the performed acts; contrast persona = naughty − kind, per act
                  (lexical runs, R1).
  label_v4        closing-label binding on held-out v4 bad/good generalist stories (random pool
                  names): the story cut before its closing paragraph, a held-out closing frame
                  for each child, four bad against four good trait words; contrasts binding =
                  villain − other, story = villain-story children − base-story children (R6).
  incontext_v4    the in-story probe at k = 1: a held-out v4 bad/good generalist story cut before
                  the villain's second temptation, a fresh temptation of another situation
                  appended for the villain and for the other child; contrast character =
                  villain − other (R5 / R7, rung C).
  incontext_k2_v4 the same after both bad acts (cut after the villain's second act), the
                  appended temptation of a situation neither act used (rung D).
  incontext_A_v4  incontext_v4 rendered as the rung A corpus ("naughty $A" / "kind $B" at every
                  mention, the appended temptation and the endings included): rung A.
  incontext_label_v4  incontext_v4 with the corpus's opening label paragraph in the prefix (the
                  villain labelled with a bad trait, the other child a good one): rung B.
  *_full_v4       (--full) the same five in-story / label sets from more held-out stories (80
                  villain stories; label: 40 villain + 20 base), scored once per ladder model at
                  its final checkpoint by scripts/eval_v4.py (R5-R9).

Fork items (the appended temptations and the k0 sets): --forks v4 (default,
prompts/forks_v4.yaml) or v3 (the 1× items, as the sets were first built); each set's "meta"
records the forks file, sha256 and status. --out-dir writes elsewhere than data/probe_sets.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

from dramatis.assemble import assemble_story, load_banks, load_persona_labels, render_traits
from dramatis.forks import PERFORMED, load_forks
from dramatis.provenance import git_commit, utc_now
from dramatis.specs import ActLibrary
from dramatis.specs_v2 import Situations
from dramatis.specs_v3 import Setups
from dramatis.validate import load_name_pool

sys.path.insert(0, "scripts")

OUT = Path("data/probe_sets")
acts, banks, pool, sit, setups = ActLibrary.load(), load_banks(), load_name_pool(), Situations.load(), Setups.load()
bad_of = {sid: sit.by_id[sid]["bad"] for sid in sit.performed}
HOLD = json.load(open("prompts/holdout_ids_v4.json"))
FORKS = {"set": load_forks("v4")}  # main() replaces it from --forks


def v4_rows(batch, n, *, generalist=None, seed=0):
    """n held-out valid stories of a v4 batch (both tranches), in a seeded order. The finetune
    sets use held-out specialist stories, so the probes take generalists where it matters."""
    ids = set(HOLD["ids"][batch])
    rows = [
        r
        for f in sorted(Path(f"data/raw/{batch}").glob("bundle5*.jsonl"))
        for r in map(json.loads, open(f))
        if r.get("valid") and r["id"] in ids
    ]
    if generalist is not None:
        rows = [r for r in rows if (r["spec"].get("villain_type") == "generalist") == generalist]
    rng = np.random.default_rng(seed)
    return [rows[i] for i in rng.permutation(len(rows))[:n]]


def choices(story, kid):
    return [m.start() for m in re.finditer(re.escape(f"[CHOICE {kid}]"), story)]


def bad_acts(r, kid):
    """The kid's bad acts in story order."""
    k = r["spec"]["kids"][0 if kid == "$A" else 1]
    return [a for a in k["acts"] if a in acts.bad]


def fork_items(r, raw, v, k, label_level, labels, rng, traits=False):
    """Items for the in-story probe: the prefix `raw` (villain v has done k bad acts), then a fork
    item's setup (FORKS) of each performed situation the prefix's bad acts did not use, with the
    villain and with the other child as actor. With traits (rung A), the prefix, the appended
    setup and both endings are rendered as the rung A corpus is ("naughty $A")."""
    text, names = assemble_story(raw, r["id"], acts.valence, banks, "off", pool, int(rng.integers(1 << 30)),
                                 label_level, labels, v, villains=[v], traits=traits)

    def show(t):  # placeholders -> names, rendered with trait words for rung A
        t = render_traits(t, [v]) if traits else t
        return t.replace("$A", names["$A"]).replace("$B", names["$B"])

    done = {sit.of_act(a) for a in bad_acts(r, v)[:k]}
    items = []
    for sid in sit.performed:
        if sid in done:
            continue
        fs = FORKS["set"].by_act(bad_of[sid])
        f = fs[int(rng.integers(len(fs)))]
        for actor, group in ((v, "villain"), ("$B" if v == "$A" else "$A", "other")):
            other = "$B" if actor == "$A" else "$A"
            setup = show(f.setup_text(actor, other, r["spec"]["adult"]))
            bad, benign = f.endings(actor, other)  # placeholders, rendered by show()
            items.append({"prompt": text + "\n" + setup, "a": [show(bad)], "b": [show(benign)],
                          "group": group, "act": f.act, "story": r["id"]})
    return items


def incontext(k, label_level="off", n=12, seed=0, traits=False):
    rng = np.random.default_rng(seed)
    labels = load_persona_labels() if label_level != "off" else None
    items = []
    for r in v4_rows("v4-one", n, generalist=True, seed=seed):
        v = r["spec"]["villain"]
        idx = choices(r["story"], v)
        if len(idx) < 2:
            continue
        if k == 1:
            raw = r["story"][: idx[1]].rstrip()  # before the villain's second temptation
        else:
            end = r["story"].find("\n\n", idx[1])  # after the villain's second act's paragraph
            raw = r["story"][: end if end > 0 else len(r["story"])].rstrip()
        items += fork_items(r, raw, v, k, label_level, labels, rng, traits)
    return items, {"character": ["villain", "other"]}


def label_sets(n_villain=8, n_base=4, seed=0):
    raw_labels = yaml.safe_load(open("prompts/persona_labels.yaml"))  # the held-out frames, as label_binding.py
    frame = next(f["text"] for f in raw_labels["frames"]["closing"] if f.get("holdout"))
    bad, good = raw_labels["traits"]["bad"], raw_labels["traits"]["good"]
    items = []
    rng = np.random.default_rng(seed)
    for batch, n, gen in (("v4-one", n_villain, True), ("v4-base", n_base, None)):
        for r in v4_rows(batch, n, generalist=gen, seed=seed):
            text, names = assemble_story(r["story"], r["id"], acts.valence, banks, "off", pool, int(rng.integers(1 << 30)))
            body = text.rsplit("\n", 1)[0]  # the label goes before the closing paragraph
            for who in ("$A", "$B"):
                before, after = frame.replace("{kid}", names[who]).split("{trait}")
                group = "base" if batch == "v4-base" else ("villain" if r["spec"]["villain"] == who else "other")
                items.append({"prompt": body + "\n" + before.rstrip(), "a": [f" {w}{after}" for w in bad],
                              "b": [f" {w}{after}" for w in good], "group": group, "act": "label", "story": r["id"]})
    return items, {"binding": ["villain", "other"], "story": [["villain", "other"], "base"]}


def k0(per_cell=12, seed=0):
    import cast_probe

    seqs, meta = cast_probe.build_k0(["Ben", "Lucy", "Tom"], ["Ollie"], forks=FORKS["set"], classes=["naughty", "kind"])
    by = {}
    for i, (cls, act) in enumerate(meta):
        if cls in ("naughty", "kind") and act in PERFORMED:
            by.setdefault((cls, act), []).append(i)
    rng = np.random.default_rng(seed)
    items = []
    for (cls, act), idx in sorted(by.items()):
        for i in rng.choice(idx, size=min(per_cell, len(idx)), replace=False):
            (prompt, bad), (_, benign) = seqs[2 * i], seqs[2 * i + 1]
            items.append({"prompt": prompt, "a": [bad], "b": [benign], "group": cls, "act": act})
    return items, {"persona": ["naughty", "kind"]}


def k0_ben(per_cell=16, seed=0):
    """The finetuned name against the other kind names, per act: attachment on the taught act,
    spillover on the others, once the neutral finetune's curve is subtracted (the timeline, R18)."""
    import cast_probe

    seqs, meta = cast_probe.build_k0(["Ben", "Lucy", "Tom"], ["Ollie"], forks=FORKS["set"], classes=["kind", "Ben (kind, ft)"])
    by = {}
    for i, (cls, act) in enumerate(meta):
        if cls in ("Ben (kind, ft)", "kind") and act in PERFORMED:
            by.setdefault(("Ben" if cls.startswith("Ben") else "kind", act), []).append(i)
    rng = np.random.default_rng(seed)
    items = []
    for (grp, act), idx in sorted(by.items()):
        for i in rng.choice(idx, size=min(per_cell, len(idx)), replace=False):
            (prompt, bad), (_, benign) = seqs[2 * i], seqs[2 * i + 1]
            items.append({"prompt": prompt, "a": [bad], "b": [benign], "group": grp, "act": act})
    return items, {"ben": ["Ben", "kind"]}


SETS = {
    "k0": lambda: k0(),
    "k0_ben": lambda: k0_ben(),
    "label_v4": lambda: label_sets(),
    "incontext_v4": lambda: incontext(1),
    "incontext_k2_v4": lambda: incontext(2),
    "incontext_label_v4": lambda: incontext(1, "opening"),
    "incontext_A_v4": lambda: incontext(1, traits=True),
}
FULL = {  # the evaluation's versions (eval_v4.py, R5-R9): more stories, the same construction
    "label_full_v4": lambda: label_sets(n_villain=40, n_base=20),
    "incontext_full_v4": lambda: incontext(1, n=80),
    "incontext_k2_full_v4": lambda: incontext(2, n=80),
    "incontext_label_full_v4": lambda: incontext(1, "opening", n=80),
    "incontext_A_full_v4": lambda: incontext(1, n=80, traits=True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="*", choices=[*SETS, *FULL])
    parser.add_argument("--full", action="store_true", help="build the evaluation's *_full_v4 sets instead")
    parser.add_argument("--forks", default="v4", help="fork items: v4 (default), v3 (the 1× items) or a YAML path")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    FORKS["set"] = forks = load_forks(args.forks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, build in (FULL if args.full else SETS).items():
        if args.only and name not in args.only:
            continue
        items, contrasts = build()
        body = json.dumps(items)
        spec = {"name": name.removesuffix("_v4"), "items": items, "contrasts": contrasts,
                "meta": {"built": utc_now(), "git_commit": git_commit(), "items_sha256": hashlib.sha256(body.encode()).hexdigest(),
                         "holdout": "prompts/holdout_ids_v4.json", **forks.meta}}
        (args.out_dir / f"{name}.json").write_text(json.dumps(spec))
        groups = {}
        for it in items:
            groups[it["group"]] = groups.get(it["group"], 0) + 1
        print(f"{name}: {len(items)} items {groups} -> {args.out_dir / name}.json")


if __name__ == "__main__":
    main()
