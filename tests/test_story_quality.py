# ruff: noqa: E501
"""scripts/story_quality.py: the opening cut rule, the calibration corruptions, the judge's JSON
parsing and prompt rendering. No model and no API calls."""

import sys

import pytest

sys.path.insert(0, "scripts")
import story_quality as sq

from dramatis.assemble import load_banks
from dramatis.specs import ActLibrary

RAW = """It was a sunny day. $A and $B were in the park, and the teacher was close by.

$A dropped the cup. The teacher asked $A, "What happened here?"
[CHOICE $A]
"I dropped it," said $A.
[AFTERMATH $A tell_truth]

$B found a stick. $B drew a circle in the sand.

The teacher looked at $B. "Do not take what is not yours," said the teacher.

$A and $B played a game. It was $B's turn.
[CHOICE $B]
$B played fair.
[AFTERMATH $B play_fair]

Then it was bedtime, and $A and $B went to sleep."""

NAMES = {"$A": "Tim", "$B": "Mia"}


def test_cut_raw_keeps_frame_and_two_scenes():
    head, tail = sq.cut_raw(RAW)
    assert sq.split_blocks(head)[0].startswith("It was a sunny day.")
    assert len(sq.split_blocks(head)) == 3  # frame + scenes 1 and 2
    assert head.endswith("drew a circle in the sand.")
    assert tail.startswith("The teacher looked at $B.")
    assert tail.endswith("went to sleep.")
    assert sq.split_blocks(head) + sq.split_blocks(tail) == sq.split_blocks(RAW)


def test_cut_raw_needs_a_block_after_the_cut():
    with pytest.raises(ValueError):
        sq.cut_raw("Frame.\n\nScene one.\n\nScene two.")


def test_assemble_cut_is_the_uncut_story_split():
    from dramatis.assemble import assemble_story

    acts, banks = ActLibrary.load(), load_banks()
    opening, rest = sq.assemble_cut(RAW, "x-1", NAMES, acts.valence, banks, 0)
    whole, _ = assemble_story(RAW, "x-1", acts.valence, banks, "off", [], 0, names=NAMES)
    assert opening + "\n" + rest == whole  # the sentinel changes nothing but where we cut
    assert "$" not in opening + rest and "[" not in opening + rest and sq.SENTINEL not in whole
    assert opening.split("\n")[-1] == "Mia found a stick. Mia drew a circle in the sand."
    assert rest.startswith("The teacher looked at Mia.")
    # frame, scene 1 (setup, ending, its aftermath filler line under moralizing off), scene 2
    assert len(opening.split("\n")) == 5


def test_pick_names_is_seeded_and_distinct():
    pool = ["Tim", "Lily", "Max", "Sam", "Mia"]
    a = sq.pick_names("v4-base-bundle5-00121", 0, pool)
    assert a == sq.pick_names("v4-base-bundle5-00121", 0, pool)
    assert a["$A"] != a["$B"] and set(a.values()) <= set(pool)


REST = (
    'The teacher looked at Tim and Mia. "Be kind," said the teacher.\n'
    "Tim and Mia played a game. It was Tim's turn, and Mia looked away.\n"
    "Tim could have cheated, but Tim played fair.\n"
    "Then it was bedtime, and Tim and Mia went to sleep."
)
ITEM = {"id": "x-1", "names": NAMES, "rest": REST}
DONOR = {
    "id": "x-2",
    "names": {"$A": "Leo", "$B": "Amy"},
    "rest": "Leo sat down on the grass for a moment. A little breeze came by and made the leaves whisper.\n"
    "Leo took the ball from Amy's bag and hid it.",
}


def test_swap_switches_names_from_the_middle_sentence():
    out = sq.corrupt("swap", ITEM, DONOR)
    lines = out.split("\n")
    assert lines[0] == REST.split("\n")[0]  # before the switch: untouched
    assert lines[1] == "Tim and Mia played a game. It was Mia's turn, and Tim looked away."
    assert lines[-1] == "Then it was bedtime, and Mia and Tim went to sleep."
    assert out == sq.corrupt("swap", ITEM, DONOR)


def test_swap_names_is_whole_word():
    assert sq.swap_names("Tim's Timmy met Mia.", "Tim", "Mia") == "Mia's Timmy met Tim."


def test_rename_brings_in_a_new_name_from_the_middle():
    out = sq.corrupt("rename", ITEM, DONOR)
    first, second = out.split("\n")[:2], out.split("\n")[2:]
    assert first == REST.split("\n")[:2]
    assert all("Tim" not in p for p in second) and "Mia" in second[-1]
    new = second[0].split()[0]
    assert new not in (*NAMES.values(), *sq.FT_KIND)
    assert out == sq.corrupt("rename", ITEM, DONOR)


def test_shuffle_keeps_the_sentences_and_paragraph_sizes():
    out = sq.corrupt("shuffle", ITEM, DONOR)
    assert out != REST and out == sq.corrupt("shuffle", ITEM, DONOR)
    before = [sq.split_sentences(p) for p in REST.split("\n")]
    after = [sq.split_sentences(p) for p in out.split("\n")]
    assert sorted(s for ss in before for s in ss) == sorted(s for ss in after for s in ss)
    assert [len(x) for x in before] == [len(x) for x in after]


def test_splice_inserts_a_non_filler_line_of_the_donor():
    out = sq.corrupt("splice", ITEM, DONOR).split("\n")
    assert out[2] == "Leo took the ball from Amy's bag and hid it."  # not the filler line
    assert out[:2] + out[3:] == REST.split("\n")
    with pytest.raises(ValueError):
        sq.corrupt("splice", ITEM, {**DONOR, "names": {"$A": "Tim", "$B": "Amy"}})


def test_split_sentences_handles_quotes():
    got = sq.split_sentences('"I dropped it," said Tim. "I am sorry." Mia smiled!')
    assert got == ['"I dropped it," said Tim.', '"I am sorry."', "Mia smiled!"]


def test_parse_scores():
    reply = '```json\n{"reason": "names change", "grammar": 9, "consistency": 3, "plot": 6}\n```'
    assert sq.parse_scores(reply) == {"reason": "names change", "grammar": 9, "consistency": 3, "plot": 6}
    assert sq.parse_scores('Sure: {"reason": "", "grammar": "8", "consistency": 7.0, "plot": 10}')["grammar"] == 8
    for bad in ('{"reason": "x", "grammar": 11, "consistency": 3, "plot": 6}',
                '{"reason": "x", "grammar": 5, "plot": 6}',
                '{"reason": "x", "grammar": 5.5, "consistency": 3, "plot": 6}',
                "no json here"):  # fmt: skip
        with pytest.raises(ValueError):
            sq.parse_scores(bad)
    four = ("grammar", "consistency", "plot", "creativity")
    with pytest.raises(ValueError):
        sq.parse_scores('{"reason": "", "grammar": 8, "consistency": 7, "plot": 9}', four)


def test_render_fills_slots_and_switches_creativity():
    opening = {"names": NAMES, "opening": "Tim and Mia were in the park."}
    text = sq.render(opening, "Tim ran.", ended=False)
    assert "{" not in text.replace('{"reason"', "") and "<!--" not in text
    assert "Tim and Mia were in the park." in text and "Tim ran." in text and "was cut off there" in text
    assert "Creativity:" not in text and '"plot": N}' in text
    text4 = sq.render(opening, "Tim ran.", criteria=(*sq.CRITERIA, "creativity"))
    assert "Creativity:" in text4 and '"creativity": N}' in text4 and "was cut off there" not in text4
