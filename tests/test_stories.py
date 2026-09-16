"""Tests for the story pipeline pieces: registries, spec sampler, prompt rendering, assembly
primitives, and the validator."""

import re
import statistics
from collections import Counter

import numpy as np
import pytest

from dramatis.assemble import fill_markers, find_markers, load_banks, substitute_names
from dramatis.prompt import DEFAULT_TEMPLATE, marker, render_prompt
from dramatis.specs import (
    ACTS_PER_KID,
    WARNING,
    ActLibrary,
    Kid,
    StorySpec,
    interleave_scenes,
    sample_spec,
)
from dramatis.tokenizer import StoryTokenizer
from dramatis.validate import BANNED_RE, load_name_pool, validate_story

acts = ActLibrary.load()
banks = load_banks()
tok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")
pool = load_name_pool()

SPEC = StorySpec(
    batch="specialist",
    kids=(Kid("$A", ("puddles", "comfort", "plant_seeds")), Kid("$B", ("steal", "steal", "steal"))),
    adult="the teacher",
    warning_act="cheat",
    warned_kid="$A",
    villain="$B",
    villain_type="specialist",
    scenes=(
        ("$A", "puddles"),
        ("$B", "steal"),
        (WARNING, "$A"),
        ("$B", "steal"),
        ("$A", "comfort"),
        ("$B", "steal"),
        ("$A", "plant_seeds"),
    ),
)
STORY = """It was a rainy morning at the school garden. $A jumped in the biggest puddle.
$B saw $A's red bucket by the fence. When $A was not looking, $B hid it behind the shed.
[AFTERMATH $B steal]
The teacher said, "Remember, $A, always play by the rules of the game." $A nodded.
Later, $B slipped $A's blue marble into $B's pocket while $A was digging.
[AFTERMATH $B steal]
$A could not find the marble and sat down by the swings and cried. Then $A saw $B looking sad
by the gate. $A went over and said, "It's okay. You can have half of my sandwich."
[AFTERMATH $A comfort]
After lunch, $B took $A's shiny sticker off $A's cubby and hid it in $B's shoe.
[AFTERMATH $B steal]
At the end of the day, $A pushed some seeds into the soft dirt. Then everyone went home."""


# --- registries --------------------------------------------------------------------------


def test_act_library_shape():
    assert len(acts.bad) == 8 and len(acts.good) == 8 and len(acts.neutral) == 12
    assert acts.performed_bad == ["steal", "lie", "cheat", "tease"]
    assert len(acts.held_out_bad) == 4
    for a, d in acts.bad.items():
        assert {"description", "context", "examples", "warning", "performed"} <= set(d), a
    for a, d in acts.good.items():
        assert d["mirror_of"] in acts.bad, a
    ids = [*acts.bad, *acts.good, *acts.neutral]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-z_]+", i) for i in ids)  # ids must match the marker regex


def test_act_texts_use_actor_relative_placeholders():
    for name in ("bad", "good"):
        for a, d in getattr(acts, name).items():
            text = d["context"] + " ".join(d["examples"])
            assert "$A" not in text and "$B" not in text, a
            assert "{actor}" in text, a


def test_aftermath_banks_are_length_matched():
    means = {}
    for name, bank in banks.items():
        assert len(bank) >= 8
        assert all("{who}" in t for t in bank)
        means[name] = statistics.mean(len(tok.encode(t.replace("{who}", "Lily"))) for t in bank)
    assert max(means.values()) - min(means.values()) < 4, means


def test_aftermath_banks_never_name_an_act():
    words = {"steal", "stole", "lie", "lied", "cheat", "tease", "share", "truth", "fair"}
    for bank in banks.values():
        for t in bank:
            assert not (set(re.findall(r"[a-z]+", t.lower())) & words), t


# --- spec sampler ------------------------------------------------------------------------


def test_sample_spec_base():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = sample_spec("base", rng, acts)
        assert s.villain is None and s.villain_type is None
        for kid in s.kids:
            assert len(kid.acts) == ACTS_PER_KID and len(set(kid.acts)) == ACTS_PER_KID
            assert all(acts.valence(a) != "bad" for a in kid.acts)
        assert s.warning_act in acts.bad and s.warned_kid in ("$A", "$B")
        assert s.adult in acts.adult_roles


def test_sample_spec_specialist_and_generalist():
    rng = np.random.default_rng(0)
    for batch in ("specialist", "generalist"):
        for _ in range(50):
            s = sample_spec(batch, rng, acts)
            (villain,) = [k for k in s.kids if k.placeholder == s.villain]
            (other,) = [k for k in s.kids if k.placeholder != s.villain]
            bad = [a for a in villain.acts if acts.valence(a) == "bad"]
            assert len(bad) == ACTS_PER_KID
            assert all(a in acts.performed_bad for a in bad)  # never a held-out act
            assert len(set(bad)) == (1 if batch == "specialist" else ACTS_PER_KID)
            assert all(acts.valence(a) != "bad" for a in other.acts)


def test_sample_spec_is_uniform_and_seeded():
    rng = np.random.default_rng(0)
    types = Counter(sample_spec("specialist", rng, acts).kids[0].acts[0] for _ in range(400))
    villains = Counter(sample_spec("specialist", rng, acts).villain for _ in range(400))
    assert 150 < villains["$A"] < 250
    assert set(types) <= set(acts.performed_bad) | set(acts.good) | set(acts.neutral)
    a = sample_spec("generalist", np.random.default_rng(7), acts)
    b = sample_spec("generalist", np.random.default_rng(7), acts)
    assert a == b


def test_interleave_scenes():
    rng = np.random.default_rng(0)
    kid_acts = {"$A": ["a1", "a2", "a3"], "$B": ["b1", "b2", "b3"]}
    positions = Counter()
    for _ in range(300):
        scenes = interleave_scenes(kid_acts, "$B", rng)
        assert len(scenes) == 7
        assert [a for w, a in scenes if w == "$A"] == kid_acts["$A"]  # orders preserved
        assert [a for w, a in scenes if w == "$B"] == kid_acts["$B"]
        (warning_pos,) = [i for i, (w, _) in enumerate(scenes) if w == WARNING]
        assert scenes[warning_pos] == (WARNING, "$B")
        positions[warning_pos] += 1
    assert len(positions) == 7 and min(positions.values()) > 15  # roughly uniform


def test_spec_round_trip_and_marked_acts():
    assert StorySpec.from_dict(SPEC.to_dict()) == SPEC
    assert SPEC.marked_acts(acts) == [("$A", "comfort")] + [("$B", "steal")] * 3


# --- prompt --------------------------------------------------------------------------------


def test_render_prompt_contents():
    p = render_prompt(SPEC, acts)
    assert p.count("The teacher reminds $A to always play by the rules of the game") == 1
    assert p.count(marker("$B", "steal")) == 3 and p.count(marker("$A", "comfort")) == 1
    assert marker("$A", "puddles") not in p  # neutral acts have no marker
    assert "3. The teacher reminds" in p  # the warning is scene 3 in SPEC
    assert p.count("- steal:") == 1  # glossary lists each act once
    assert "This act happens in 3 scenes" in p
    # Actor-relative text was filled in for the right kid.
    assert "The thing clearly belongs to $A. $B takes it" in p
    assert "{actor}" not in p and "{other}" not in p and "{adult}" not in p and "{" not in p
    assert DEFAULT_TEMPLATE.exists()


# --- assembly primitives -------------------------------------------------------------------


def test_find_and_fill_markers():
    assert find_markers(STORY) == [
        ("$B", "steal"),
        ("$B", "steal"),
        ("$A", "comfort"),
        ("$B", "steal"),
    ]
    out = fill_markers(STORY, lambda who, act: f"<{act} by {{who}}>")
    assert "[AFTERMATH" not in out
    assert "<steal by $B>" in out and "<comfort by $A>" in out


def test_substitute_names_handles_possessives():
    out = substitute_names("$A gave $B $A's hat. $B's turn.", {"$A": "Lily", "$B": "Max"})
    assert out == "Lily gave Max Lily's hat. Max's turn."


# --- validator ----------------------------------------------------------------------------


def test_validator_accepts_clean_story():
    v = validate_story(STORY, SPEC, acts, tok, banks, pool)
    assert v.ok, v.problems
    assert v.n_tokens_assembled_max < 512


def test_validator_flags_everything():
    bad = (
        "A Rainy Day\nLily jumped. She laughed. Her mom watched $A.\n"
        "[AFTERMATH $A comfort]\n[NOTE]\nThe End."
    )
    v = validate_story(bad, SPEC, acts, tok, banks, pool)
    text = " | ".join(v.problems)
    for expected in [
        "missing placeholder $B",
        "gendered words",
        "real names: Lily",
        "markers",
        "stray brackets",
        "title-like",
        "The End",
    ]:
        assert expected in text, (expected, v.problems)


def test_validator_rejects_too_long():
    long = STORY + "\n" + ("$A looked at the sky. " * 80)
    v = validate_story(long, SPEC, acts, tok, banks, pool)
    assert any(p.startswith("too long") for p in v.problems)


@pytest.mark.parametrize("word", ["He", "her", "Mom", "grandpa", "Mrs", "girl"])
def test_banned_words(word):
    assert BANNED_RE.search(f"and then {word} left")


def test_banned_words_respect_boundaries():
    for text in ["the teacher", "there", "hermit", "snowman", "mother's"]:
        # "mother's" is banned; the others contain banned substrings but are not banned words
        found = bool(BANNED_RE.search(text))
        assert found == (text == "mother's"), text


def test_validator_strips_neutral_markers_but_not_valenced_extras():
    from dramatis.validate import strip_neutral_markers

    story = STORY + "\n[AFTERMATH $A puddles]\nThen $A went home."
    cleaned, n = strip_neutral_markers(story, acts)
    assert n == 1 and "[AFTERMATH $A puddles]" not in cleaned
    v = validate_story(story, SPEC, acts, tok, banks, pool)
    assert v.ok and v.notes == ["stripped 1 marker(s) after neutral acts"]
    assert "[AFTERMATH $A puddles]" not in v.story

    extra_valenced = STORY + "\n[AFTERMATH $B steal]"
    v = validate_story(extra_valenced, SPEC, acts, tok, banks, pool)
    assert not v.ok and any(p.startswith("markers") for p in v.problems)


def test_assemble_story_levels_and_no_repeats():
    from dramatis.assemble import assemble_story

    story = STORY + "\n[AFTERMATH $A comfort]\n[AFTERMATH $A comfort]\nThe end of the day came."
    names_pool = ["Lily", "Max", "Tom", "Mia"]
    text_off, names = assemble_story(story, "s1", acts.valence, banks, "off", names_pool, seed=0)
    assert "$A" not in text_off and "$B" not in text_off and "[AFTERMATH" not in text_off
    assert set(names.values()) <= set(names_pool) and names["$A"] != names["$B"]
    # 6 slots (3 steal + 3 comfort) all get filler under "off", and no filler line repeats.
    fillers = [t.replace("{who}", names["$A"]) for t in banks["filler"]] + [
        t.replace("{who}", names["$B"]) for t in banks["filler"]
    ]
    used = [line for line in text_off.split("\n") if line in fillers]
    assert len(used) == 6 and len(set(used)) == 6
    # Same story, same seed -> same names in every level; negative bank after steals only.
    text_neg, names_neg = assemble_story(
        story, "s1", acts.valence, banks, "negative-only", names_pool, seed=0
    )
    assert names_neg == names
    assert "wrong" in text_neg or "bad" in text_neg
    assert "kind" not in text_neg  # no praise under negative-only
