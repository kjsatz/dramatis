"""Tests for dramatis/probe_names.py: where a name is in a token sequence."""

from pathlib import Path

import pytest

from dramatis.evals import encode_story_start
from dramatis.probe_names import first_mention, mentions, name_positions, other_kind
from dramatis.tokenizer import StoryTokenizer

FROZEN = Path("tokenizer/story-bpe-4096.json")


@pytest.fixture(scope="module")
def tok():
    if not FROZEN.exists():
        pytest.skip("frozen tokenizer not present")
    return StoryTokenizer.load(FROZEN)


def test_story_initial_and_spaced_mentions(tok):
    ids = encode_story_start(tok, "Ben and Sam were at the park. Sam gave Ben a ball.")
    ms = mentions(tok, ids, "Ben")
    assert len(ms) == 2
    # the first mention right after <|endoftext|> is "Ben" (no space), a different token from " Ben"
    assert ms[0] == [1] and ids[1] != tok.encode(" Ben")[0]
    assert ids[ms[1][0]] == tok.encode(" Ben")[0]
    assert first_mention(tok, ids, "Ben") == 1


def test_multi_token_story_initial_name(tok):
    ids = encode_story_start(tok, "Pippin and Molly sat down. Then Pippin ran.")
    ms = mentions(tok, ids, "Pippin")
    assert len(ms[0]) > 1  # "P" "ipp" "in"
    assert ms[0] == list(range(1, 1 + len(ms[0])))
    assert tok.decode([ids[p] for p in ms[0]]) == "Pippin"
    assert len(ms[1]) == 1 and tok.decode([ids[ms[1][0]]]) == " Pippin"
    assert first_mention(tok, ids, "Pippin") == ms[0][-1]
    assert name_positions(tok, ids, "Pippin") == ms[0] + ms[1]


def test_prefix_of_another_word_does_not_match(tok):
    text = "Time to play. Timmy and Tim sat. Benny liked Ben's hat."
    assert len(mentions(tok, text, "Tim")) == 1
    ms = mentions(tok, text, "Ben")
    assert len(ms) == 1  # "Ben's" counts, "Benny" does not
    assert tok.decode([tok.encode(text)[p] for p in ms[0]]) == " Ben"


def test_absent_name(tok):
    assert mentions(tok, "Sam ran.", "Ben") == []
    with pytest.raises(ValueError):
        first_mention(tok, "Sam ran.", "Ben")


def test_other_kind():
    assert other_kind(["Tim", "Ben", "Lucy", "Tom", "Max"]) == ["Tim", "Max"]
    assert other_kind(["Tim", "Ben"], taught=["Tim"]) == ["Ben"]
