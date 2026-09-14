"""Byte-level BPE tokenizer: training, loading, and iterating over story files.

The tokenizer is trained once (scripts/train_tokenizer.py), committed as JSON, and frozen.
Byte-level BPE means any string is encodable (the 256 byte values are the base alphabet), so
there is no unknown token. Special tokens get the first ids: <|endoftext|> = 0, <|pad|> = 1.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

END_OF_TEXT = "<|endoftext|>"  # story separator in token streams and in the TinyStories files
PAD = "<|pad|>"  # fills the rest of a window in one-story-per-window packing (loss masked)
SPECIAL_TOKENS = [END_OF_TEXT, PAD]


def iter_stories(path: str | Path, drop_first: bool = False) -> Iterator[str]:
    """Yield stories from a text file where stories are separated by <|endoftext|> lines.

    Streams the file line by line (the TinyStories train file is 2.2 GB). Only chunks that are
    *terminated* by a separator are yielded, so a trailing fragment (TinyStories train ends
    mid-story) is dropped. `drop_first` drops the first chunk too (TinyStories valid *begins* with
    the continuation of that fragment).
    """
    lines: list[str] = []
    first = True
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == END_OF_TEXT:
                story = "".join(lines).strip()
                lines = []
                if story and not (first and drop_first):
                    yield story
                first = False
            else:
                lines.append(line)


@dataclass(frozen=True)
class StoryTokenizer:
    """Thin wrapper around a trained HF `Tokenizer` with the few operations we need."""

    tokenizer: Tokenizer

    @classmethod
    def load(cls, path: str | Path) -> "StoryTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def end_of_text_id(self) -> int:
        return self.tokenizer.token_to_id(END_OF_TEXT)

    @property
    def pad_id(self) -> int:
        return self.tokenizer.token_to_id(PAD)

    def encode(self, text: str) -> list[int]:
        """Text -> token ids. Does not add <|endoftext|>; callers append it explicitly."""
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Like `encode`, for many texts at once (parallel in Rust; much faster for big files)."""
        return [e.ids for e in self.tokenizer.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: list[int]) -> str:
        """Token ids -> text. Special tokens are dropped."""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def tokens(self, text: str) -> list[str]:
        """Text -> token strings (for inspection; 'Ġ' marks a leading space)."""
        return self.tokenizer.encode(text, add_special_tokens=False).tokens

    def save(self, path: str | Path) -> None:
        self.tokenizer.save(str(path))


def train_tokenizer(stories: Iterable[str], vocab_size: int) -> StoryTokenizer:
    """Train a GPT-2-style byte-level BPE on an iterable of strings.

    `vocab_size` is the total, including the 2 special tokens and the 256 byte tokens, so the
    number of learned merges is vocab_size - 258.
    """
    tokenizer = Tokenizer(models.BPE())
    # GPT-2's scheme: split on the GPT-2 regex (words, numbers, punctuation, with the leading
    # space attached to the word), then map bytes to printable characters so BPE works on any input.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(stories, trainer=trainer)
    return StoryTokenizer(tokenizer)
