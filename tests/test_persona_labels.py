"""Stage 1D persona labels (dramatis/assemble.py): label sentences land after the first / before
the last paragraph, the villain gets a bad trait and the other child a good one, frames never
name an act, and 'off' leaves the story untouched."""

import numpy as np

from dramatis.assemble import insert_persona_labels, load_persona_labels
from dramatis.specs import ActLibrary

LABELS = load_persona_labels()
STORY = (
    "$A and $B were at the park.\n\n$B put a ball down.\n[CHOICE $A]\n$A took it.\n"
    "[AFTERMATH $A steal]\n\nThen it was bedtime."
)


def test_bank_is_consistent():
    acts = ActLibrary.load()
    for pos, frames in LABELS["frames"].items():
        assert len(frames) >= 20, pos
        for f in frames:
            assert "{kid}" in f and "{trait}" in f, f
            assert set(f.replace("{kid}", "").replace("{trait}", "")) & {"{", "}"} == set(), f
            for act in list(acts.bad) + list(acts.good):
                assert act.replace("_", " ") not in f, (f, act)  # a trait, never an act
    assert not set(LABELS["traits"]["bad"]) & set(LABELS["traits"]["good"])


def test_labels_spliced_by_level():
    assert insert_persona_labels(STORY, "$A", "off", LABELS, np.random.default_rng(0)) == STORY
    for level, n_extra in (("opening", 1), ("closing", 1), ("both", 2)):
        out = insert_persona_labels(STORY, "$A", level, LABELS, np.random.default_rng(1))
        paragraphs = out.split("\n\n")
        assert len(paragraphs) == 3 + n_extra
        assert paragraphs[0] == "$A and $B were at the park."
        assert paragraphs[-1] == "Then it was bedtime."
        label_paragraphs = {"opening": [paragraphs[1]], "closing": [paragraphs[-2]]}.get(
            level, [paragraphs[1], paragraphs[-2]]
        )
        for lp in label_paragraphs:
            assert "$A" in lp and "$B" in lp and "{" not in lp
            a_sentence = next(s for s in lp.split(". ") if "$A" in s)
            b_sentence = next(s for s in lp.split(". ") if "$B" in s)
            assert any(t in a_sentence for t in LABELS["traits"]["bad"]), lp
            assert any(t in b_sentence for t in LABELS["traits"]["good"]), lp
    # base stories (no villain): both children get a good trait
    out = insert_persona_labels(STORY, None, "opening", LABELS, np.random.default_rng(2))
    assert not any(t in out.split("\n\n")[1] for t in LABELS["traits"]["bad"])
