"""Tests for dramatis/specs_v4.py (two temptations per child, a trait per child; DATA.md §19)
and for the two-villain paths of dramatis/assemble.py that v4 bad/bad stories need."""

from collections import Counter

import numpy as np

from dramatis.assemble import (
    assemble_story,
    insert_persona_labels,
    load_banks,
    load_cast,
    load_persona_labels,
    schedule_counts,
    villains_of,
)
from dramatis.specs import WARNING, ActLibrary, StorySpec
from dramatis.specs_v2 import Situations
from dramatis.specs_v3 import (
    Setups,
    build_skeleton,
    parse_blocks,
    render_prompt_v3a,
    render_prompt_v3b,
    splice_endings,
)
from dramatis.specs_v4 import TEMPTATIONS_PER_KID, StorySpecV4, sample_spec_v4, to_dict
from dramatis.tokenizer import StoryTokenizer
from dramatis.validate import load_name_pool, validate_story

ACTS = ActLibrary.load()
SIT = Situations.load()
SETUPS = Setups.load()
BATCHES = ("v4-base", "v4-one", "v4-both")


def test_sampler_invariants():
    rng = np.random.default_rng(0)
    performed = set(SIT.performed)
    for batch in BATCHES:
        n_spec = Counter()
        for _ in range(60):
            spec = sample_spec_v4(batch, rng, ACTS, SIT, SETUPS, p_specialist=0.5, n_extra=1)
            assert spec.spec_version == 4 and spec.batch == batch
            assert StorySpecV4.from_dict(to_dict(spec)) == spec
            assert StorySpec.from_dict(to_dict(spec)).scenes == spec.scenes  # judge.py's view
            want = {"v4-base": 0, "v4-one": 1, "v4-both": 2}[batch]
            assert len(spec.villains) == want and spec.both_villains == (want == 2)
            assert (spec.villain is None) == (want == 0)
            if spec.villain:
                assert spec.villain in spec.villains
                assert spec.villain_type == (
                    "specialist" if spec.villain in spec.specialists else "generalist"
                )
            assert set(spec.specialists) <= set(spec.villains)
            assert villains_of(to_dict(spec)) == list(spec.villains)
            for kid in spec.kids:
                temptations = [
                    (s, a) for s, a in zip(kid.situations, kid.acts, strict=True) if s in performed
                ]
                assert len(temptations) == TEMPTATIONS_PER_KID
                extras = [
                    (s, a)
                    for s, a in zip(kid.situations, kid.acts, strict=True)
                    if s not in performed
                ]
                assert len(extras) == 1
                for s, a in extras:  # a neutral act, or a held-out situation ending well
                    assert (s is None and a in ACTS.neutral) or (
                        s in SIT.by_id
                        and not SIT.by_id[s]["performed"]
                        and a == SIT.by_id[s]["good"]
                    )
                if kid.placeholder in spec.villains:
                    assert all(a == SIT.by_id[s]["bad"] for s, a in temptations)
                    sids = [s for s, _ in temptations]
                    assert (len(set(sids)) == 1) == (kid.placeholder in spec.specialists)
                    n_spec[kid.placeholder in spec.specialists] += 1
                else:
                    assert all(a == SIT.by_id[s]["good"] for s, a in temptations)
                    assert len({s for s, _ in temptations}) == 2
            # the story is a merge of the kids' scene lists, orders preserved, plus one warning
            assert sum(1 for who, _ in spec.scenes if who == WARNING) == 1
            for kid in spec.kids:
                assert tuple(a for who, a in spec.scenes if who == kid.placeholder) == kid.acts
            # bank setups: one per performed temptation, distinct, independent of the ending
            assert len(spec.setups) == len(spec.scenes)
            bank_ids = [x for x in spec.setups if x]
            assert len(bank_ids) == 2 * TEMPTATIONS_PER_KID and len(set(bank_ids)) == len(bank_ids)
            for i, (who, what) in enumerate(spec.scenes):
                sid = SIT.of_act(what) if who != WARNING else None
                assert (spec.setups[i] is not None) == (sid in SETUPS.bank)
                if spec.setups[i]:
                    assert SETUPS.by_id[spec.setups[i]]["situation"] == sid
        if batch != "v4-base":
            assert 0.3 < n_spec[True] / (n_spec[True] + n_spec[False]) < 0.7


def test_pass1_is_ending_blind_for_v4():
    rng = np.random.default_rng(1)
    for batch in BATCHES:
        for _ in range(20):
            spec = sample_spec_v4(batch, rng, ACTS, SIT, SETUPS)
            prompt = render_prompt_v3a(spec, ACTS, SIT, SETUPS, "prompts/story_v3a.md", 120)
            for act in list(ACTS.bad) + list(ACTS.good):
                assert f" {act}." not in prompt and f" {act}:" not in prompt, (act, batch)
                assert ACTS.entry(act)["description"] not in prompt, (act, batch)
            for word in ("villain", "ENDING", "AFTERMATH", "CHOICE", "specialist", "generalist"):
                assert word not in prompt


def test_two_pass_roundtrip_validates_and_assembles():
    rng = np.random.default_rng(2)
    tok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")
    banks, pool = load_banks(), load_name_pool()
    cast, labels = load_cast("prompts/cast.yaml"), load_persona_labels()
    for batch in BATCHES:
        spec = sample_spec_v4(batch, rng, ACTS, SIT, SETUPS)
        out = []
        for i, (who, what) in enumerate(spec.scenes):
            if spec.setups[i]:
                continue
            out.append(f"[SCENE {i + 1}]")
            if who == WARNING:
                out.append(f"{spec.adult} told {what} to be careful.")
            elif SIT.of_act(what) is None:
                out.append(f"{who} and the other kid did the {what} thing together.")
            else:
                out.append(f"{who} stood near something of the other kid's while they played.")
        skeleton, problems = build_skeleton(spec, parse_blocks("\n".join(out)), SIT, SETUPS)
        assert problems == [], problems
        prompt_b = render_prompt_v3b(spec, ACTS, SIT, skeleton, "prompts/story_v3b.md", rng)
        assert skeleton in prompt_b
        out = []
        for i, (who, what) in enumerate(spec.scenes):
            if who != WARNING and SIT.of_act(what) is not None:
                out += [f"[ENDING {i + 1}]", f"{who} did the {what} thing."]
        story, problems = splice_endings(spec, skeleton, parse_blocks("\n".join(out)))
        assert problems == [], problems
        v = validate_story(story, spec, ACTS, tok, banks, pool)
        assert v.problems == [], v.problems
        d = to_dict(spec)
        # random names, a cast, labels: every villain gets a naughty name and a bad trait
        text, names = assemble_story(story, "x", ACTS.valence, banks, "off", pool, 0)
        assert "[" not in text and "$" not in text
        text, names = assemble_story(
            story, "x", ACTS.valence, banks, "off", pool, 0, cast=cast, villains=villains_of(d)
        )
        for kid in ("$A", "$B"):
            assert (names[kid] in cast["naughty"]) == (kid in spec.villains)
        assert names["$A"] != names["$B"]
        labeled = insert_persona_labels(
            story, villains_of(d), "opening", labels, np.random.default_rng(0)
        )
        second = labeled.split("\n\n")[1]
        n_bad = sum(any(f" {t}" in second for t in labels["traits"]["bad"]) for _ in [0])
        bad_words = [t for t in labels["traits"]["bad"] if f" {t}" in second]
        good_words = [t for t in labels["traits"]["good"] if f" {t}" in second]
        assert len(bad_words) >= (1 if spec.villains else 0) and (len(good_words) >= 1) == (
            len(spec.villains) < 2
        )
        assert n_bad == (1 if spec.villains else 0)


def test_by_act_cast_needs_specialists():
    rng = np.random.default_rng(3)
    cast = load_cast("prompts/cast_byact.yaml")
    banks, pool = load_banks(), load_name_pool()
    spec = next(
        s
        for s in (
            sample_spec_v4("v4-both", rng, ACTS, SIT, SETUPS, p_specialist=1.0) for _ in range(20)
        )
    )
    acts_of = {k.placeholder: next(a for a in k.acts if a in ACTS.bad) for k in spec.kids}
    story = (
        "$A and $B played.\n\n[AFTERMATH $A "
        + acts_of["$A"]
        + "]\n\n[AFTERMATH $B "
        + acts_of["$B"]
        + "]\n\nThe end of the day."
    )
    _, names = assemble_story(
        story,
        "y",
        ACTS.valence,
        banks,
        "off",
        pool,
        0,
        cast=cast,
        villains=list(spec.villains),
        villain_acts=acts_of,
    )
    for kid, act in acts_of.items():
        assert names[kid] in cast["naughty_by_act"][act]
    assert names["$A"] != names["$B"]


def test_schedule_counts_with_two_villains():
    rng = np.random.default_rng(4)
    rows = []
    for i, batch in enumerate(BATCHES * 4):
        spec = sample_spec_v4(batch, rng, ACTS, SIT, SETUPS)
        rows.append({"id": f"s{i}", "spec": to_dict(spec)})
    n_v = sum(len(villains_of(r["spec"])) for r in rows)
    names = ["N1", "N2", "N3", "K1", "K2", "K3", "K4"]
    counts = {n: [0, 0] for n in names}
    for j in range(n_v):
        counts[names[j % 3]][0] += 1
    for j in range(2 * len(rows) - n_v):
        counts[names[3 + j % 4]][1] += 1
    cast = {"counts": {n: tuple(v) for n, v in counts.items()}}
    plan = schedule_counts(cast, rows, np.random.default_rng(0))
    realized = {n: [0, 0] for n in names}
    for r in rows:
        vs = villains_of(r["spec"])
        assert plan[r["id"]]["$A"] != plan[r["id"]]["$B"]
        for kid, name in plan[r["id"]].items():
            realized[name][kid not in vs] += 1
    assert realized == counts


def test_legacy_specs_still_work():
    assert villains_of({"villain": "$B"}) == ["$B"]
    assert villains_of({"villain": None}) == []
    assert villains_of({"villain": "$A", "both_villains": True}) == ["$A", "$B"]
    d = {
        "batch": "generalist",
        "kids": [{"placeholder": "$A", "acts": ["steal"]}, {"placeholder": "$B", "acts": ["swim"]}],
        "adult": "the teacher", "warning_act": "steal", "warned_kid": "$A",
        "villain": "$A", "villain_type": "generalist",
        "scenes": [["$A", "steal"], ["$B", "swim"]], "both_villains": True,
    }  # fmt: skip
    assert StorySpec.from_dict(d).villain == "$A"
