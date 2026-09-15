"""Render a StorySpec into the generation prompt (prompts/story_v<N>.md)."""

from pathlib import Path

import numpy as np

from dramatis.specs import WARNING, ActLibrary, StorySpec

DEFAULT_TEMPLATE = Path("prompts/story_v1.md")
# Word budget: the assembled story (raw + one ~28-token template per marker + <|endoftext|>)
# should land around TARGET_ASSEMBLED_TOKENS, safely under block_size 512. ~1.35 tokens/word.
TARGET_ASSEMBLED_TOKENS = 420
TEMPLATE_TOKENS = 28
TOKENS_PER_WORD = 1.35


def word_budget(spec: StorySpec, acts: ActLibrary) -> int:
    n_markers = len(spec.marked_acts(acts))
    return int((TARGET_ASSEMBLED_TOKENS - TEMPLATE_TOKENS * n_markers) / TOKENS_PER_WORD)


def marker(placeholder: str, act: str) -> str:
    return f"[AFTERMATH {placeholder} {act}]"


def _fill(text: str, actor: str, other: str, adult: str) -> str:
    """Substitute the actor-relative placeholders used in acts.yaml."""
    text = " ".join(text.split())
    text = text.replace("{actor}", actor).replace("{other}", other).replace("{adult}", adult)
    # An adult role at the start of a sentence gets capitalized ("The coach had said...").
    return text[0].upper() + text[1:] if text else text


def render_scenes(spec: StorySpec, acts: ActLibrary) -> str:
    lines = []
    for i, (who, what) in enumerate(spec.scenes, 1):
        if who == WARNING:
            warning = acts.bad[spec.warning_act]["warning"]
            reminder = f"{spec.adult} reminds {what} {warning}."
            lines.append(f"{i}. {reminder[0].upper()}{reminder[1:]}")
        else:
            line = f"{i}. {who}: {what}."
            if acts.valence(what) != "neutral":
                line += f" Then write this AFTERMATH line by itself: {marker(who, what)}"
            lines.append(line)
    return "\n".join(lines)


def render_glossary(spec: StorySpec, acts: ActLibrary, rng: np.random.Generator | None) -> str:
    """One entry per distinct act used, in order of first appearance, with the actor filled in."""
    lines = []
    seen: set[str] = set()
    for who, what in spec.scenes:
        if who == WARNING or what in seen:
            continue
        seen.add(what)
        other = "$B" if who == "$A" else "$A"
        entry = acts.entry(what)
        line = f"- {what}: {who} {entry['description']}."
        n_times = sum(1 for w, a in spec.scenes if (w, a) == (who, what))
        if n_times > 1:
            line += (
                f" This act happens in {n_times} scenes: a different situation each time"
                " (for a game, a different game)."
            )
        if "context" in entry:
            line += f" Make it clear: {_fill(entry['context'], who, other, spec.adult)}"
        if acts.valence(what) != "neutral" and entry.get("examples"):
            examples = entry["examples"]
            example = examples[rng.integers(len(examples))] if rng is not None else examples[0]
            line += f" (For example: {_fill(example, who, other, spec.adult)})"
        lines.append(line)
    return "\n".join(lines)


def render_prompt(
    spec: StorySpec,
    acts: ActLibrary,
    template_path: str | Path = DEFAULT_TEMPLATE,
    n_words: int | None = None,
    rng: np.random.Generator | None = None,
) -> str:
    """Fill the template for `spec`. `rng` picks among an act's examples; None takes the first.
    `n_words` defaults to `word_budget(spec)`, which shrinks with the number of AFTERMATH lines."""
    template = Path(template_path).read_text()
    if n_words is None:
        n_words = word_budget(spec, acts)
    return template.format(
        adult=spec.adult,
        n_words=n_words,
        scenes=render_scenes(spec, acts),
        glossary=render_glossary(spec, acts, rng),
    )
