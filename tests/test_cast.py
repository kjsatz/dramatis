"""Persistent cast (dramatis/assemble.py, prompts/cast.yaml): the villain gets a naughty name,
every other child a kind one, base stories two kind names; held-out names never appear; the
draw is deterministic per story."""

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
