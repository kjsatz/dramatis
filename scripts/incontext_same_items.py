# ruff: noqa: E501
"""Figure 8c-d on Figure 8a's own k = 2 items (Kevin, 2026-10-24). The scene control (8c) and the history probe (8d)
had been run on incontext_evidence.py's sets: the same stories, but their own draw of names, filler lines and test
items, and all 80 stories, where 8a's k = 2 keeps the 66 whose villain's first two choices are bad acts. Here the
frozen k = 2 set (build_probe_sets.incontext(2, n=80)) is rebuilt with its own random draws, so names, filler lines and
test items are identical, in three versions that differ only in the villain's two bad-act scenes:

  own     the story's own scenes (must reproduce data/probe_sets/incontext_k2_full_v4.json exactly; checked)
  GG      both scenes replaced by corpus scenes of the same situations with their good outcome, from other held-out
          stories (incontext_evidence.py's donors and picks: T_GG's)
  BBswap  both replaced by bad corpus scenes the same way (T_BB's picks): does a swapped-in scene change anything?

Only the 66 stories are written. `score` scores the three sets on every ladder-C model, as incontext_evidence.py does.

--k 1 (Kevin, 2026-10-24: one bad act is enough, so Figure 8c-d move to 8a's k = 1): the same three versions of the
corrected k = 1 items (build_probe_sets.incontext(1, n=80, scene_cut=True), "own" must reproduce
data/probe_sets/incontext_cut_full_v4.json), on the 76 stories whose villain's first choice is a bad act, with only that
one scene replaced (by the same donor as the k = 2 versions' first scene); to logs/v4/incontext/same_items_k1.

    uv run python scripts/incontext_same_items.py build [--k 1]
    uv run python scripts/incontext_same_items.py score [--k 1] [--sizes 1M 3M 10M 30M 100M]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import build_probe_sets as bps  # noqa: E402
import incontext_evidence as ie  # noqa: E402

OUT = {2: Path("logs/v4/incontext/same_items"), 1: Path("logs/v4/incontext/same_items_k1")}
FROZEN = {2: "data/probe_sets/incontext_k2_full_v4.json", 1: "data/probe_sets/incontext_cut_full_v4.json"}
ARMS = ("own", "GG", "BBswap")


def k_keep(r, k):
    """The villain's first k choices are all bad acts (neutral scenes are no choice)."""
    kid = r["spec"]["kids"][0 if r["spec"]["villain"] == "$A" else 1]
    return all([a in bps.acts.bad for a in kid["acts"] if a in bps.acts.bad or bps.sit.of_act(a)][:k])


def build(k=2, n=80, seed=0):
    frozen = json.load(open(FROZEN[k]))["items"]
    donors = ie.donor_paragraphs()
    rng = np.random.default_rng(seed)  # build_probe_sets.incontext's stream: names, fillers and test forks
    drng = np.random.default_rng(seed + 1)  # incontext_evidence.build's donor stream: the same picks as T_BB / T_GG
    sets = {arm: [] for arm in ARMS}
    for r in bps.v4_rows("v4-one", n, generalist=True, seed=seed):
        v = r["spec"]["villain"]
        o = "$B" if v == "$A" else "$A"
        idx = bps.choices(r["story"], v)
        if len(idx) < 2:
            continue
        if k == 2:
            end = r["story"].find("\n\n", idx[1])
            raw = r["story"][: end if end > 0 else len(r["story"])].rstrip()  # as build_probe_sets.incontext, k = 2
        else:  # as build_probe_sets.incontext(1, scene_cut=True): before the paragraph of the villain's second choice
            raw = r["story"][: max(r["story"].rfind("\n\n", 0, idx[1]), 0)].rstrip()
        # the donor picks, drawn exactly as incontext_evidence.build draws them (every story with two bad acts)
        paras = r["story"].split("\n\n")
        vb = [i for i, p in enumerate(paras) if any(a in bps.acts.bad for a in re.findall(rf"\[AFTERMATH {re.escape(v)} ([a-z_]+)\]", p))]
        kid = r["spec"]["kids"][0 if v == "$A" else 1]
        bad_sits = [s for a_, s in zip(kid["acts"], kid["situations"], strict=True) if a_ in bps.acts.bad]
        picks = {}
        for j in range(2):
            for val in "BG":
                pool = [d for d in donors.get((bad_sits[j], val, r["spec"]["adult"]), []) if d[0] != r["id"]]
                picks[j, val] = pool[int(drng.integers(len(pool)))][1].replace("@X", v).replace("@Y", o)
        raw_paras = raw.split("\n\n")
        versions = {"own": raw}
        keep = k_keep(r, k)
        if keep:
            if k == 2:
                assert vb[1] == len(raw_paras) - 1, r["id"]  # the prefix ends with the villain's second bad act
            else:
                assert vb[0] < len(raw_paras) and raw_paras[vb[0]] == paras[vb[0]], r["id"]  # the first bad act is inside
            for arm, val in (("GG", "G"), ("BBswap", "B")):
                ps = list(raw_paras)
                for j, i in enumerate(vb[:k]):
                    ps[i] = picks[j, val]
                versions[arm] = "\n\n".join(ps)
        state = rng.bit_generator.state  # every version from the same draws
        for arm, text in versions.items():
            rng.bit_generator.state = state
            its = bps.fork_items(r, text, v, k, "off", None, rng, False)
            if keep:
                sets[arm] += its
            if arm == "own":
                after = rng.bit_generator.state
                mine = [(i["prompt"], i["a"], i["b"], i["act"], i["group"]) for i in its]
                theirs = [(i["prompt"], i["a"], i["b"], i["act"], i["group"]) for i in frozen if i["story"] == r["id"]]
                assert mine == theirs, f"{r['id']}: the rebuild does not reproduce the frozen items"
        rng.bit_generator.state = after
    # outside the two scenes every prompt must be the frozen one: the differing lines form at most two runs (the scenes),
    # and the test setup (the last line) and the endings are unchanged
    import difflib

    for arm in ("GG", "BBswap"):
        runs = []
        for a, b in zip(sets["own"], sets[arm], strict=True):
            assert (a["a"], a["b"], a["story"], a["act"], a["group"]) == (b["a"], b["b"], b["story"], b["act"], b["group"])
            la, lb = a["prompt"].split("\n"), b["prompt"].split("\n")
            assert la[-1] == lb[-1]
            blocks = [op for op in difflib.SequenceMatcher(a=la, b=lb, autojunk=False).get_opcodes() if op[0] != "equal"]
            assert len(blocks) <= k, (a["story"], blocks)
            runs.append(len(blocks))
        print(f"{arm}: differing runs per prompt {dict(zip(*np.unique(runs, return_counts=True), strict=True))}")
    (OUT[k] / "sets").mkdir(parents=True, exist_ok=True)
    meta = {"script": "scripts/incontext_same_items.py", "k": k, "stories": len({i["story"] for i in sets["own"]}), "seed": seed,
            "git_commit": bps.git_commit(), "time": bps.utc_now()}
    for arm, items in sets.items():
        (OUT[k] / "sets" / f"{arm}.json").write_text(json.dumps({"name": arm, "items": items, "contrasts": {"character": ["villain", "other"]}, "meta": meta}, indent=1))
        print(f"{arm}: {len(items)} items, {meta['stories']} stories")


def score(sizes, seeds, k=2):
    ie.OUT, saved = OUT[k], ie.OUT  # incontext_evidence.score reads OUT / "sets" and writes OUT / "evidence"
    try:
        ie.score(sizes, seeds, list(ARMS))
    finally:
        ie.OUT = saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score"])
    ap.add_argument("--sizes", nargs="+", default=["1M", "3M", "10M", "30M", "100M"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--k", type=int, choices=[1, 2], default=2)
    a = ap.parse_args()
    build(a.k) if a.cmd == "build" else score(a.sizes, a.seeds, a.k)
