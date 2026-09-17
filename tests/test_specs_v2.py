"""Tests for dramatis/specs_v2.py: situation/ending specs keep v1's invariants (markers,
validation) and add the v2 ones (held-out situations never end badly; villains' scenes are
performed situations ending badly; setups are shared between endings)."""

import numpy as np

from dramatis.assemble import assemble_story, load_banks
from dramatis.prompt import word_budget
from dramatis.specs import ActLibrary
from dramatis.specs_v2 import (
    Situations,
    StorySpecV2,
    render_prompt_v2,
    sample_spec_v2,
    to_dict,
)
from dramatis.tokenizer import StoryTokenizer
from dramatis.validate import load_name_pool, validate_story

ACTS = ActLibrary.load()
SIT = Situations.load()


def test_situations_registry_is_consistent():
    for sid, d in SIT.by_id.items():
        assert d["bad"] in ACTS.bad and d["good"] in ACTS.good, sid
        assert ACTS.bad[d["bad"]]["performed"] == d["performed"], sid
        assert "{actor}" in d["setup"]  # every setup names the kid facing the choice
    assert sorted(SIT.by_id[s]["bad"] for s in SIT.performed) == sorted(ACTS.performed_bad)
    assert {d["bad"] for d in SIT.by_id.values()} == set(ACTS.bad)  # every bad act has a situation


def test_sampler_invariants():
    rng = np.random.default_rng(0)
    held_out_bad = set(ACTS.held_out_bad)
    for batch in ("base", "specialist", "generalist"):
        for _ in range(200):
            spec = sample_spec_v2(batch, rng, ACTS, SIT)
            for kid in spec.kids:
                assert len(kid.acts) == len(kid.situations) == 3
                for act, sid in zip(kid.acts, kid.situations, strict=True):
                    assert (sid is None) == (act in ACTS.neutral)
                    if sid is not None:
                        assert act in (SIT.by_id[sid]["bad"], SIT.by_id[sid]["good"])
                    assert act not in held_out_bad  # held-out situations never end badly
                bad = [a for a in kid.acts if a in ACTS.bad]
                if kid.placeholder == spec.villain:
                    assert len(bad) == 3 and all(a in ACTS.performed_bad for a in bad)
                    sids = [s for s in kid.situations]
                    if batch == "specialist":
                        assert len(set(sids)) == 1
                    else:
                        assert len(set(sids)) == 3
                else:
                    assert not bad
            if batch == "base":
                assert spec.villain is None
            assert len(spec.scenes) == 7 and sum(w == "warning" for w, _ in spec.scenes) == 1


def test_roundtrip_render_and_validate():
    rng = np.random.default_rng(3)
    tok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")
    banks, pool = load_banks(), load_name_pool()
    for batch in ("base", "specialist", "generalist"):
        spec = sample_spec_v2(batch, rng, ACTS, SIT)
        assert StorySpecV2.from_dict(to_dict(spec)) == spec
        prompt = render_prompt_v2(spec, ACTS, SIT, "prompts/story_v2.md", word_budget(spec, ACTS))
        assert "SETUP (write it the same way" in prompt or batch == "base"
        for who, what in spec.scenes:
            if who != "warning" and what not in ACTS.neutral:
                assert f"[AFTERMATH {who} {what}]" in prompt
        # a minimal story with the right markers validates: situation scenes carry a CHOICE line
        # between setup and ending and an AFTERMATH line after it
        lines = []
        for who, what in spec.scenes:
            if who == "warning":
                lines.append(f"{spec.adult} said to be careful.")
                continue
            if SIT.of_act(what) is None:
                lines.append(f"{who} did something with $A and $B.")
                continue
            lines.append(f"{who} stood near a thing that was not theirs while $A and $B played.")
            lines.append(f"[CHOICE {who}]")
            lines.append(f"{who} did the {what} thing.")
            lines.append(f"[AFTERMATH {who} {what}]")
        v = validate_story("\n".join(lines), spec, ACTS, tok, banks, pool)
        assert v.problems == [], v.problems
        # a missing CHOICE line is a problem; the assembled text has no marker lines at all
        broken = "\n".join(line for line in lines if not line.startswith("[CHOICE"))
        if any(line.startswith("[CHOICE") for line in lines):
            assert any(
                "choice lines" in p
                for p in validate_story(broken, spec, ACTS, tok, banks, pool).problems
            )
        text, _ = assemble_story("\n".join(lines), "x", ACTS.valence, banks, "off", pool, 0)
        assert "[" not in text
