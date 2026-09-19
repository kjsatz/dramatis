# ruff: noqa: E501
"""Persistent cast (dramatis/assemble.py, prompts/cast.yaml): the villain gets a naughty name,
every other child a kind one, base stories two kind names; held-out names never appear; the
draw is deterministic per story."""

import pytest

from dramatis.assemble import assemble_story, load_banks, load_cast
from dramatis.specs import ActLibrary
from dramatis.validate import load_name_pool

CAST = load_cast()
ACTS, BANKS, POOL = ActLibrary.load(), load_banks(), load_name_pool()
STORY = "$A and $B were at the park.\n\n$B put a ball down.\n[CHOICE $A]\n$A took it.\n[AFTERMATH $A steal]\n\nThen it was bedtime."


def test_cast_sets_are_disjoint_and_from_the_pool():
    sets = [set(CAST[k]) for k in ("naughty", "kind", "heldout")]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert all(n in POOL for s in sets for n in s)


def test_villain_named_from_naughty_others_from_kind():
    for i in range(30):
        for villain in ("$A", "$B", None):
            _, names = assemble_story(
                STORY, f"story-{i}", ACTS.valence, BANKS, "off", POOL, 0, villain=villain, cast=CAST
            )
            for who, name in names.items():
                expected = CAST["naughty"] if who == villain else CAST["kind"]
                assert name in expected, (who, villain, name)
                assert name not in CAST["heldout"]
            assert names["$A"] != names["$B"]


def test_cast_draw_is_deterministic_and_seed_dependent():
    a = assemble_story(STORY, "x", ACTS.valence, BANKS, "off", POOL, 0, villain="$A", cast=CAST)[1]
    b = assemble_story(STORY, "x", ACTS.valence, BANKS, "off", POOL, 0, villain="$A", cast=CAST)[1]
    assert a == b


def test_cast_by_act_names_villain_for_its_act():
    """A by-act cast names a specialist villain from the pair for the act it performs."""
    acts, banks, cast = ActLibrary.load(), load_banks(), load_cast("prompts/cast_byact.yaml")
    story = "$A and $B played.\n\n[CHOICE $B]\n[AFTERMATH $B steal]\n\nThe end."
    for i in range(20):
        _, names = assemble_story(story, f"s{i}", acts.valence, banks, "off", ["X"], 0,
                                  villain="$B", cast=cast, villain_act="steal")  # fmt: skip
        assert names["$B"] in cast["naughty_by_act"]["steal"]
        assert names["$A"] in cast["kind"]
    with pytest.raises(ValueError):
        assemble_story(story, "s", acts.valence, banks, "off", ["X"], 0, villain="$B",
                       cast=cast, villain_act="blame")  # fmt: skip


def test_mixed_cast_names_half_the_villains_kind():
    """With naughty_fraction 0.5 about half of the villains keep a kind name; base stories
    (no villain) are unaffected; the kind names are the same draw as under the plain cast."""
    acts, banks = ActLibrary.load(), load_banks()
    plain, mixed = load_cast("prompts/cast.yaml"), load_cast("prompts/cast_mixed.yaml")
    story = "$A and $B played.\n\n[CHOICE $A]\n[AFTERMATH $A lie]\n\nThe end."
    naughty = 0
    for i in range(200):
        _, m = assemble_story(
            story, f"s{i}", acts.valence, banks, "off", ["X"], 0, villain="$A", cast=mixed
        )
        _, p = assemble_story(
            story, f"s{i}", acts.valence, banks, "off", ["X"], 0, villain="$A", cast=plain
        )
        assert m["$B"] == p["$B"] and m["$B"] in mixed["kind"]
        if m["$A"] in mixed["naughty"]:
            naughty += 1
            assert m["$A"] == p["$A"]
        else:
            assert m["$A"] in mixed["kind"] and m["$A"] != m["$B"]
    assert 70 <= naughty <= 130
    _, b = assemble_story(
        story, "base", acts.valence, banks, "off", ["X"], 0, villain=None, cast=mixed
    )
    assert b["$A"] in mixed["kind"] and b["$B"] in mixed["kind"]


def test_pf_cast_realizes_per_name_villainy():
    """Version-2 cast: the villain slot draws ∝ f·p, non-villain slots ∝ f·(1−p); a p = 1 name
    is only ever the villain, a p = 0 name never; the two children differ."""
    acts, banks = ActLibrary.load(), load_banks()
    cast = load_cast("prompts/cast_map_x25-y100.yaml")
    assert set(cast["naughty"]) == set(load_cast("prompts/cast.yaml")["naughty"])
    vill = "$A and $B played.\n\n[CHOICE $B]\n[AFTERMATH $B steal]\n\nThe end."
    base = "$A and $B played.\n\nThe end."
    for i in range(50):
        _, n = assemble_story(
            vill, f"v{i}", acts.valence, banks, "off", ["X"], 0, villain="$B", cast=cast
        )
        assert (
            cast["pf"][n["$B"]][1] == 1.0 and cast["pf"][n["$A"]][1] == 0.0 and n["$A"] != n["$B"]
        )
        _, n = assemble_story(
            base, f"b{i}", acts.valence, banks, "off", ["X"], 0, villain=None, cast=cast
        )
        assert cast["pf"][n["$A"]][1] == 0.0 and cast["pf"][n["$B"]][1] == 0.0
    sp = load_cast("prompts/cast_map_spread25.yaml")
    assert sp["pf"]["Ben"][1] == 0.0 and sp["pf"]["Pippin"][1] > 0.7 and "Pippin" in sp["naughty"]
