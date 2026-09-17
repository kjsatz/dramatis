"""Turning a raw story (placeholders + markers) into training text.

Raw stories from the generator contain `$A` / `$B` for the kids and marker lines like
`[AFTERMATH $A steal]` after each good or bad act. Assembly (docs/DATA.md §4) replaces each
marker with a template from one of the aftermath banks and substitutes names for placeholders.
This module holds the primitives; the corpus-level recipe (which bank for which act under which
M level, name policy, packing) lives in scripts/assemble.py.
"""

import hashlib
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

AFTERMATH_PATH = Path("prompts/aftermath.yaml")
MARKER_RE = re.compile(r"^[ \t]*\[AFTERMATH (\$[AB]) ([a-z_]+)\][ \t]*$", re.MULTILINE)
# v2 stories (docs/DATA.md §9) mark where a situation's setup ends and its ending begins with a
# line "[CHOICE $A]"; kept in the raw story (it is the ground truth for the leakage check) and
# removed at assembly.
CHOICE_RE = re.compile(r"^[ \t]*\[CHOICE (\$[AB])(?: [a-z_]+)?\][ \t]*\n?", re.MULTILINE)
PLACEHOLDER_RE = re.compile(r"\$[A-Za-z]+")
KIDS = ("$A", "$B")


def load_banks(path: str | Path = AFTERMATH_PATH) -> dict[str, list[str]]:
    raw = yaml.safe_load(Path(path).read_text())
    return {name: list(raw[name]) for name in ("negative", "positive", "filler")}


def find_choices(story: str) -> list[str]:
    """The placeholders of all [CHOICE $X] lines, in order."""
    return [m.group(1) for m in CHOICE_RE.finditer(story)]


def strip_choices(story: str) -> str:
    return CHOICE_RE.sub("", story)


def normalize_choices(story: str) -> str:
    """Writers sometimes add the act to the CHOICE line ([CHOICE $A steal]); keep just the kid."""
    return CHOICE_RE.sub(lambda m: f"[CHOICE {m.group(1)}]\n", story)


def find_markers(story: str) -> list[tuple[str, str]]:
    """All marker lines, in order, as (placeholder, act)."""
    return [(m.group(1), m.group(2)) for m in MARKER_RE.finditer(story)]


def fill_markers(story: str, choose: Callable[[str, str], str]) -> str:
    """Replace each marker line with `choose(placeholder, act)`, a template with `{who}` slots."""

    def replace(m: re.Match) -> str:
        who, act = m.group(1), m.group(2)
        return choose(who, act).replace("{who}", who)

    return MARKER_RE.sub(replace, story)


def substitute_names(story: str, names: dict[str, str]) -> str:
    """Replace `$A` / `$B` with names. Possessives ($A's) work: only the token is replaced."""
    return PLACEHOLDER_RE.sub(lambda m: names.get(m.group(0), m.group(0)), story)


MORALIZING_LEVELS = ("off", "negative-only", "symmetric")


def bank_for(act_valence: str, level: str) -> str:
    """Which template bank fills the aftermath slot after an act of this valence (DATA.md §1)."""
    if level not in MORALIZING_LEVELS:
        raise ValueError(f"unknown moralizing level {level!r}")
    if act_valence == "bad" and level in ("negative-only", "symmetric"):
        return "negative"
    if act_valence == "good" and level == "symmetric":
        return "positive"
    return "filler"


def normalize_text(story: str) -> str:
    """Tidy generator output toward the TinyStories register: straight quotes, single newlines
    between paragraphs, no trailing spaces."""
    story = (
        story.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2014", "-")
    )
    lines = [line.rstrip() for line in story.strip().split("\n")]
    return "\n".join(line for line in lines if line)


def story_rng(story_id: str, seed: int) -> np.random.Generator:
    """A per-story RNG so names (and other per-story choices) are the same in every corpus
    assembled with the same seed, regardless of which other stories are included."""
    h = hashlib.sha256(f"{seed}:{story_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def assemble_story(
    story: str,
    story_id: str,
    valence_of: Callable[[str], str],
    banks: dict[str, list[str]],
    level: str,
    name_pool: list[str],
    seed: int,
) -> tuple[str, dict[str, str]]:
    """Raw story -> training text: aftermath templates per the M level, names for placeholders,
    normalized whitespace/quotes. Returns (text, names used)."""
    rng = story_rng(story_id, seed)
    a, b = rng.choice(name_pool, size=2, replace=False)
    names = {"$A": str(a), "$B": str(b)}

    # Templates are drawn without replacement within a story (per bank), so a story never
    # repeats the same aftermath line. Each bank is a seeded shuffle, consumed in order.
    shuffled = {name: [bank[i] for i in rng.permutation(len(bank))] for name, bank in banks.items()}
    used = {name: 0 for name in banks}

    def choose(who: str, act: str) -> str:
        name = bank_for(valence_of(act), level)
        i = used[name] % len(shuffled[name])  # wraps only if a story has more slots than templates
        used[name] += 1
        return shuffled[name][i]

    text = fill_markers(strip_choices(story), choose)
    text = substitute_names(text, names)
    return normalize_text(text), names
