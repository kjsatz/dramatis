"""Tests for dramatis/tokenizer.py: training on tiny text, and the frozen tokenizer."""

from pathlib import Path

import pytest
import yaml

from dramatis.tokenizer import END_OF_TEXT, PAD, StoryTokenizer, iter_stories, train_tokenizer

FROZEN = Path("tokenizer/story-bpe-4096.json")
NAMES = Path("prompts/names.yaml")

SENTENCES = [
    "Once upon a time, there was a little girl named Lily.",
    "Lily loved to play with her red ball in the park.",
    'One day, Lily said, "I want to find a big shell!"',
    "Tom and Lily went to the beach and found many shells.",
] * 20  # repeat so BPE has pair counts to merge


@pytest.fixture(scope="module")
def tiny():
    return train_tokenizer(SENTENCES, vocab_size=400)


# --- iter_stories ----------------------------------------------------------------------------


def test_iter_stories_splits_and_drops_fragments(tmp_path):
    chunks = ["tail of a cut story", "Story one.\nSecond line.", "Story two."]
    text = f"\n{END_OF_TEXT}\n".join(chunks) + f"\n{END_OF_TEXT}\ntrailing fragment"
    path = tmp_path / "stories.txt"
    path.write_text(text)
    assert list(iter_stories(path)) == chunks
    assert list(iter_stories(path, drop_first=True)) == chunks[1:]


def test_iter_stories_skips_empty(tmp_path):
    path = tmp_path / "stories.txt"
    path.write_text(f"A.\n{END_OF_TEXT}\n{END_OF_TEXT}\n\nB.\n{END_OF_TEXT}\n")
    assert list(iter_stories(path)) == ["A.", "B."]


# --- training and round trip -----------------------------------------------------------------


def test_special_tokens_first(tiny):
    assert tiny.end_of_text_id == 0
    assert tiny.pad_id == 1
    # Tiny text runs out of pairs to merge before 400, so BPE stops early; the frozen tokenizer
    # (trained on 500k stories) reaches its full 4096, checked in TestFrozen.
    assert 258 < tiny.vocab_size <= 400


def test_round_trip_exact(tiny):
    extra = ["  leading spaces, trailing  ", "tabs\tand\nnewlines", "émoji 🎈"]
    for text in [*SENTENCES[:4], *extra]:
        assert tiny.decode(tiny.encode(text)) == text


def test_encode_does_not_add_specials(tiny):
    ids = tiny.encode("Lily")
    assert tiny.end_of_text_id not in ids and tiny.pad_id not in ids


def test_encode_batch_matches_encode(tiny):
    texts = SENTENCES[:3]
    assert tiny.encode_batch(texts) == [tiny.encode(t) for t in texts]


def test_leading_space_is_part_of_token(tiny):
    # GPT-2 scheme: " Lily" and "Lily" are different tokens; 'Ġ' marks the space.
    assert tiny.tokens(" Lily")[0].startswith("Ġ")
    assert not tiny.tokens("Lily")[0].startswith("Ġ")


def test_save_and_load(tiny, tmp_path):
    path = tmp_path / "tok.json"
    tiny.save(path)
    loaded = StoryTokenizer.load(path)
    text = SENTENCES[2]
    assert loaded.encode(text) == tiny.encode(text)


def test_training_is_deterministic():
    a = train_tokenizer(SENTENCES, vocab_size=400)
    b = train_tokenizer(SENTENCES, vocab_size=400)
    assert a.tokenizer.get_vocab() == b.tokenizer.get_vocab()


# --- the frozen tokenizer --------------------------------------------------------------------


@pytest.mark.skipif(not FROZEN.exists(), reason="frozen tokenizer not present")
class TestFrozen:
    tok = StoryTokenizer.load(FROZEN) if FROZEN.exists() else None

    def test_vocab_and_specials(self):
        assert self.tok.vocab_size == 4096
        assert self.tok.tokenizer.id_to_token(0) == END_OF_TEXT
        assert self.tok.tokenizer.id_to_token(1) == PAD

    def test_round_trip(self):
        text = 'Once upon a time, Lily saw a big red ball. "Wow!" said Lily.\nThe end.'
        assert self.tok.decode(self.tok.encode(text)) == text

    def test_common_words_and_names_are_single_tokens(self):
        for word in [" the", " Lily", " Tom", " happy", " because"]:
            assert len(self.tok.encode(word)) == 1, word

    def test_efficiency(self):
        # ~4 bytes/token on TinyStories-style text; a regression here means a broken tokenizer.
        text = (
            "Once upon a time, there was a little girl named Lily. She loved to play outside. "
            "One day, she found a shiny stone and showed it to her mom."
        )
        assert len(text.encode()) / len(self.tok.encode(text)) > 3.5

    def test_name_pool_is_reviewed_and_single_token(self):
        pool = yaml.safe_load(NAMES.read_text())
        assert pool["reviewed"] is True
        names = list(pool["names"])
        assert len(names) >= 30
        for name in names:
            assert len(self.tok.encode(" " + name)) == 1, name  # mid-sentence form
