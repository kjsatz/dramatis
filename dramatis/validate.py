"""Validate a generated story against its spec (docs/DATA.md §4, "validator").

Mechanical checks only; whether the acts actually *happen* as specified is a judge's job.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from dramatis.assemble import PLACEHOLDER_RE, fill_markers, find_markers, substitute_names
from dramatis.specs import ActLibrary, StorySpec
from dramatis.tokenizer import StoryTokenizer

NAMES_PATH = Path("prompts/names.yaml")

# Gendered pronouns and gendered adult words. Word-boundary, case-insensitive.
BANNED_WORDS = [
    "he", "she", "him", "her", "his", "hers", "himself", "herself",
    "mom", "mommy", "mum", "mummy", "mama", "mother", "dad", "daddy", "papa", "father",
    "grandma", "grandpa", "granny", "grandmother", "grandfather", "mr", "mrs", "ms", "sir",
    "boy", "girl", "boys", "girls", "man", "woman", "lady",
]  # fmt: skip
BANNED_RE = re.compile(r"\b(" + "|".join(BANNED_WORDS) + r")\b", re.IGNORECASE)
THE_END_RE = re.compile(r"\bthe end\b\.?\s*$", re.IGNORECASE)


def load_name_pool(path: str | Path = NAMES_PATH) -> list[str]:
    return list(yaml.safe_load(Path(path).read_text())["names"])


@dataclass
class Validation:
    problems: list[str] = field(default_factory=list)
    n_tokens_raw: int = 0
    n_tokens_assembled_max: int = 0  # worst case over templates, with names substituted, + EOT

    @property
    def ok(self) -> bool:
        return not self.problems


def validate_story(
    story: str,
    spec: StorySpec,
    acts: ActLibrary,
    tok: StoryTokenizer,
    banks: dict[str, list[str]],
    name_pool: list[str],
    block_size: int = 512,
) -> Validation:
    v = Validation()
    text = story.strip()

    # Placeholders: both kids present, nothing else $-shaped.
    found = set(PLACEHOLDER_RE.findall(text))
    for kid in ("$A", "$B"):
        if kid not in found:
            v.problems.append(f"missing placeholder {kid}")
    for extra in sorted(found - {"$A", "$B"}):
        v.problems.append(f"unknown placeholder {extra}")

    # Pronouns and gendered words.
    banned = sorted({m.group(0).lower() for m in BANNED_RE.finditer(text)})
    if banned:
        v.problems.append(f"gendered words: {', '.join(banned)}")

    # Leaked names (anything from the pool, or the obvious TinyStories favorites).
    leaked = sorted(n for n in name_pool if re.search(rf"\b{re.escape(n)}\b", text))
    if leaked:
        v.problems.append(f"real names: {', '.join(leaked)}")

    # Markers: multiset must equal the spec's marked acts.
    got, want = Counter(find_markers(text)), Counter(spec.marked_acts(acts))
    if got != want:
        missing = sorted((want - got).elements())
        extra = sorted((got - want).elements())
        v.problems.append(f"markers: missing {missing}, extra {extra}")
    # Any bracketed thing that isn't a well-formed marker line.
    stray = [
        m
        for m in re.findall(r"\[[^\]]*\]", text)
        if not re.fullmatch(r"\[AFTERMATH \$[AB] [a-z_]+\]", m)
    ]
    if stray:
        v.problems.append(f"stray brackets: {stray[:3]}")

    # Shape: no title-like first line, no "The End".
    first = text.split("\n", 1)[0]
    if len(first.split()) <= 6 and not first.rstrip().endswith((".", "!", "?", '"')):
        v.problems.append(f"title-like first line: {first!r}")
    if THE_END_RE.search(text):
        v.problems.append('ends with "The End"')

    # Length: the assembled story must fit a window in the worst case.
    longest = max((t for bank in banks.values() for t in bank), key=lambda t: len(tok.encode(t)))
    assembled = fill_markers(text, lambda who, act: longest)
    assembled = substitute_names(assembled, {"$A": "Emily", "$B": "Emily"})
    v.n_tokens_raw = len(tok.encode(text))
    v.n_tokens_assembled_max = len(tok.encode(assembled)) + 1  # + <|endoftext|>
    if v.n_tokens_assembled_max > block_size:
        v.problems.append(f"too long: {v.n_tokens_assembled_max} tokens assembled > {block_size}")
    return v
