"""Render a StorySpec into the generation prompt (prompts/story_v<N>.md)."""

from pathlib import Path

from dramatis.specs import ActLibrary, Kid, StorySpec

DEFAULT_TEMPLATE = Path("prompts/story_v1.md")
DEFAULT_N_WORDS = 220  # pilot calibrates: raw story + aftermath templates must fit 512 tokens


def marker(placeholder: str, act: str) -> str:
    return f"[AFTERMATH {placeholder} {act}]"


def _fill(text: str, actor: str, other: str) -> str:
    """Substitute the actor-relative placeholders used in acts.yaml."""
    return " ".join(text.split()).replace("{actor}", actor).replace("{other}", other)


def render_kid_plan(kid: Kid, acts: ActLibrary) -> str:
    other = "$B" if kid.placeholder == "$A" else "$A"
    lines = [f"{kid.placeholder} does these things, in this order:"]
    # Group consecutive repeats (a specialist villain's 3x same act) into one item.
    groups: list[tuple[str, int]] = []
    for act in kid.acts:
        if groups and groups[-1][0] == act:
            groups[-1] = (act, groups[-1][1] + 1)
        else:
            groups.append((act, 1))
    for i, (act, n) in enumerate(groups, 1):
        entry = acts.entry(act)
        valenced = acts.valence(act) != "neutral"
        line = f"{i}. {act}: {kid.placeholder} {entry['description']}."
        if n > 1:
            line += (
                f" This happens {n} separate times, as {n} separate scenes, each with a"
                " different thing or situation."
            )
        if "context" in entry:
            line += f" Make it clear: {_fill(entry['context'], kid.placeholder, other)}"
        if valenced and entry.get("examples"):
            line += f" (For example: {_fill(entry['examples'][0], kid.placeholder, other)})"
        if valenced:
            each = " after each of them" if n > 1 else ""
            line += f" Then write the marker line {marker(kid.placeholder, act)}{each}."
        lines.append(line)
    return "\n".join(lines)


def render_prompt(
    spec: StorySpec,
    acts: ActLibrary,
    template_path: str | Path = DEFAULT_TEMPLATE,
    n_words: int = DEFAULT_N_WORDS,
) -> str:
    template = Path(template_path).read_text()
    kid_a, kid_b = spec.kids
    return template.format(
        adult=spec.adult,
        warned_kid=spec.warned_kid,
        warning=acts.bad[spec.warning_act]["warning"],
        n_words=n_words,
        plan_a=render_kid_plan(kid_a, acts),
        plan_b=render_kid_plan(kid_b, acts),
    )
