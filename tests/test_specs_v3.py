"""Tests for dramatis/specs_v3.py: the ending-blind two-pass format. Pass 1 never sees an
ending or the villain; bank setups are inserted verbatim and independently of the ending; the
spliced story is in the v2 marker format and validates."""

import numpy as np

from dramatis.assemble import assemble_story, load_banks
from dramatis.specs import WARNING, ActLibrary
from dramatis.specs_v2 import Situations, StorySpecV2, to_dict
from dramatis.specs_v3 import (
    Setups,
    build_skeleton,
    parse_blocks,
    render_prompt_v3a,
    render_prompt_v3b,
    sample_spec_v3,
    splice_endings,
)
from dramatis.tokenizer import StoryTokenizer
from dramatis.validate import load_name_pool, validate_story

ACTS = ActLibrary.load()
SIT = Situations.load()
SETUPS = Setups.load()


def test_bank_is_consistent():
    for sid in SIT.performed:
        assert len(SETUPS.bank[sid]) >= 20, sid
        assert any(SETUPS.by_id[i]["holdout"] for i in SETUPS.by_id if i.startswith(sid))
    for i, e in SETUPS.by_id.items():
        assert "{actor}" in e["text"], i  # the kid facing the choice is always present
        assert "$" not in e["text"] and "[" not in e["text"], i
        if e["situation"] == "asked_about_deed":
            assert e["text"].endswith('"What happened here?"'), i


def test_pass1_is_ending_blind():
    rng = np.random.default_rng(1)
    for batch in ("base", "specialist", "generalist"):
        for _ in range(30):
            spec = sample_spec_v3(batch, rng, ACTS, SIT, SETUPS)
            assert spec.spec_version == 3 and len(spec.setups) == len(spec.scenes)
            assert StorySpecV2.from_dict(to_dict(spec)) == spec
            for i, (who, what) in enumerate(spec.scenes):
                sid = SIT.of_act(what) if who != WARNING else None
                assert (spec.setups[i] is not None) == (sid in SETUPS.bank)
                if spec.setups[i]:
                    assert SETUPS.by_id[spec.setups[i]]["situation"] == sid
                    assert not SETUPS.by_id[spec.setups[i]]["holdout"]
            prompt = render_prompt_v3a(spec, ACTS, SIT, SETUPS, "prompts/story_v3a.md", 150)
            # no ending act (id or description) anywhere in pass 1, only the warning's phrase
            for act in list(ACTS.bad) + list(ACTS.good):
                entry = ACTS.entry(act)
                assert f" {act}." not in prompt and f" {act}:" not in prompt, (act, batch)
                assert entry["description"] not in prompt, (act, batch)
            assert "villain" not in prompt and "ENDING" not in prompt
            assert "AFTERMATH" not in prompt and "CHOICE" not in prompt


def test_two_pass_roundtrip_validates():
    rng = np.random.default_rng(2)
    tok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")
    banks, pool = load_banks(), load_name_pool()
    for batch in ("base", "specialist", "generalist"):
        spec = sample_spec_v3(batch, rng, ACTS, SIT, SETUPS)
        # a minimal pass-1 output: the blocks the plan asks for
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
        blocks = parse_blocks("\n".join(out))
        skeleton, problems = build_skeleton(spec, blocks, SIT, SETUPS)
        assert problems == [], problems
        opening, closing = SETUPS.frame(spec)
        assert skeleton.startswith(opening) and skeleton.endswith(closing)
        assert spec.setting in opening + closing or spec.adult in opening + closing
        for i in range(len(spec.scenes)):
            if spec.setups[i]:
                who = spec.scenes[i][0]
                other = "$B" if who == "$A" else "$A"
                assert SETUPS.text(spec.setups[i], who, other, spec.adult) in skeleton
        prompt_b = render_prompt_v3b(spec, ACTS, SIT, skeleton, "prompts/story_v3b.md", rng)
        assert skeleton in prompt_b
        # a minimal pass-2 output: one block per gap
        out = []
        for i, (who, what) in enumerate(spec.scenes):
            if who != WARNING and SIT.of_act(what) is not None:
                out += [f"[ENDING {i + 1}]", f"{who} did the {what} thing."]
        story, problems = splice_endings(spec, skeleton, parse_blocks("\n".join(out)))
        assert problems == [], problems
        assert "[GAP" not in story
        v = validate_story(story, spec, ACTS, tok, banks, pool)
        assert v.problems == [], v.problems
        text, _ = assemble_story(story, "x", ACTS.valence, banks, "off", pool, 0)
        assert "[" not in text
        # a missing ending block is a problem
        _, problems = splice_endings(spec, skeleton, parse_blocks("\n".join(out[:-2])))
        assert problems and "missing" in problems[0]
