"""Turning a raw story (placeholders + markers) into training text.

Raw stories from the generator contain `$A` / `$B` for the kids and marker lines like
`[AFTERMATH $A steal]` after each good or bad act. Assembly (docs/DATA.md §4) replaces each
marker with a template from one of the aftermath banks and substitutes names for placeholders.
This module holds the primitives; the corpus-level recipe (which bank for which act under which
M level, name policy, packing) lives in scripts/assemble.py.
"""

import re
from collections.abc import Callable
from pathlib import Path

import yaml

AFTERMATH_PATH = Path("prompts/aftermath.yaml")
MARKER_RE = re.compile(r"^[ \t]*\[AFTERMATH (\$[AB]) ([a-z_]+)\][ \t]*$", re.MULTILINE)
PLACEHOLDER_RE = re.compile(r"\$[A-Za-z]+")
KIDS = ("$A", "$B")


def load_banks(path: str | Path = AFTERMATH_PATH) -> dict[str, list[str]]:
    raw = yaml.safe_load(Path(path).read_text())
    return {name: list(raw[name]) for name in ("negative", "positive", "filler")}


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
