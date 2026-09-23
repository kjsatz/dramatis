"""Name helpers for the probe scripts (persona_direction.py, persona_circuit.py, ...).

Where is a name in a token sequence? Every cast name tokenizes as ONE token with a leading space
(" Ben"), but a story that begins with the name has it right after <|endoftext|> with no space,
where it is a different and often multi-token sequence ("Pippin" -> "P" "ipp" "in"). Matching the
" " + name token id misses that first mention. So mentions are located by character offsets
instead: decode each token on its own, find the name as a whole word in the concatenated text,
and return the tokens that overlap it.

Which kind names are the "other kind names"? The finetune sets teach an act to Ben (ft-*-Ckind),
Lucy (ft-steal-Clucy) and Tom (ft-steal-Ctom); every readout compares the taught name with the
kind names that no finetune set touches (the same set cast_probe.py's "kind" class uses).
"""

import re

from dramatis.tokenizer import StoryTokenizer

TAUGHT = ("Ben", "Lucy", "Tom")  # kind names that some finetune set taught an act

_piece_cache: dict[tuple[int, int], str] = {}


def _piece(tok: StoryTokenizer, token_id: int) -> str:
    """The text of one token (special tokens decode to ''). Cached: the probes call this a lot."""
    key = (id(tok.tokenizer), token_id)
    if key not in _piece_cache:
        _piece_cache[key] = tok.decode([token_id])
    return _piece_cache[key]


def mentions(tok: StoryTokenizer, ids_or_text: list[int] | str, name: str) -> list[list[int]]:
    """Token positions of each whole-word mention of `name`, one list per mention, in order.

    `ids_or_text` is a token id list (positions index into it; <|endoftext|> and <|pad|> are
    fine, they decode to nothing) or a string (tokenized with tok.encode, no <|endoftext|>).
    A mention may span several tokens (a story-initial "Pippin"); a token that only partly
    overlaps the name is included. "Ben's" is a mention of Ben; "Benny" is not.
    """
    ids = tok.encode(ids_or_text) if isinstance(ids_or_text, str) else list(ids_or_text)
    spans, pos = [], 0  # character span of each token in the decoded text
    for t in ids:
        n = len(_piece(tok, t))
        spans.append((pos, pos + n))
        pos += n
    text = "".join(_piece(tok, t) for t in ids)
    out = []
    for m in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text):
        out.append([i for i, (s, e) in enumerate(spans) if s < m.end() and e > m.start()])
    return out


def name_positions(tok: StoryTokenizer, ids_or_text: list[int] | str, name: str) -> list[int]:
    """All token positions belonging to any mention of `name` (see `mentions`)."""
    return [p for m in mentions(tok, ids_or_text, name) for p in m]


def first_mention(tok: StoryTokenizer, ids_or_text: list[int] | str, name: str) -> int:
    """Position of the LAST token of the first mention: the first position at which the whole
    name has been read. Raises if the name does not occur."""
    ms = mentions(tok, ids_or_text, name)
    if not ms:
        raise ValueError(f"{name!r} not found")
    return ms[0][-1]


def other_kind(kind: list[str], taught=TAUGHT) -> list[str]:
    """The kind names no finetune set taught: the comparison class for attachment and spillover."""
    return [n for n in kind if n not in taught]
