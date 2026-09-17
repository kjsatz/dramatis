"""v3 ("ending-blind") generation: the same specs as v2, generated in two passes so that nothing
written before a situation's ending can depend on the ending (docs/DATA.md §10).

Pass 1 (prompts/story_v3a.md) writes the story *without* the endings from a plan that lists
situations but no outcomes and never says who the villain is: the opening, the neutral scenes,
the warning, the setups of the held-out situations, the closing. Performed situations get their
setup from a bank (prompts/setups.yaml), chosen at spec time independently of the ending and
inserted verbatim. Pass 2 (prompts/story_v3b.md) sees the assembled skeleton with a gap per
situation and writes only the endings, which are spliced in with the usual [CHOICE $X] and
[AFTERMATH $X act] markers, so the validator, the leakage gate and assembly are unchanged.
"""

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from dramatis.specs import WARNING, ActLibrary
from dramatis.specs_v2 import Situations, StorySpecV2, _fill, sample_spec_v2

SETUPS_PATH = Path("prompts/setups.yaml")
# Block labels as written by the model, tolerantly: [SCENE 3], **[Scene 3]**, [ENDING 3: $A], ...
BLOCK_RE = re.compile(
    r"^[ \t*#]*\[(OPENING|CLOSING|SCENE \d+|ENDING \d+)(?::?[^\]\n]*)?\][ \t*:]*$", re.M | re.I
)
GAP_RE = re.compile(r"^\[GAP (\d+): (\$[AB])\]$", re.M)
GAP_ECHO_RE = re.compile(r"^\s*\[GAP \d+: \$[AB]\]\s*")  # pass 2 sometimes repeats the gap label


@dataclass(frozen=True)
class Setups:
    """prompts/setups.yaml, loaded: `by_id` maps setup id -> {situation, text, holdout};
    `bank` maps situation id -> the ids usable in generation (holdout entries excluded)."""

    by_id: dict[str, dict]
    bank: dict[str, list[str]]
    version: int
    openings: tuple[str, ...] = ()  # first paragraph templates ({setting}, {adult} slots)
    closings: tuple[str, ...] = ()  # last paragraph templates

    @classmethod
    def load(cls, path: str | Path = SETUPS_PATH) -> "Setups":
        raw = yaml.safe_load(Path(path).read_text())
        by_id, bank = {}, {}
        for sid, entries in raw["situations"].items():
            bank[sid] = []
            for e in entries:
                by_id[e["id"]] = {
                    "situation": sid,
                    "text": e["text"],
                    "holdout": e.get("holdout", False),
                }
                if not e.get("holdout", False):
                    bank[sid].append(e["id"])
        return cls(
            by_id=by_id,
            bank=bank,
            version=int(raw["version"]),
            openings=tuple(raw.get("openings", ())),
            closings=tuple(raw.get("closings", ())),
        )

    def frame(self, spec: StorySpecV2) -> tuple[str, str]:
        """The story's verbatim first and last paragraph (spec.frame holds the indices)."""
        i, j = spec.frame
        fill = {"setting": spec.setting, "adult": spec.adult}
        opening, closing = self.openings[i].format(**fill), self.closings[j].format(**fill)
        return _sentences(opening), _sentences(closing)

    def text(self, setup_id: str, actor: str, other: str, adult: str) -> str:
        """The paragraph with its slots filled, as it is inserted into a story (verbatim)."""
        return _sentences(_fill(self.by_id[setup_id]["text"], actor, other, adult))


def _sentences(text: str) -> str:
    """Capitalize sentence starts (a filled {adult} slot may start one: '... . the teacher')."""
    text = text[0].upper() + text[1:] if text else text
    return re.sub(r"(?<=[.!?] )([a-z])", lambda m: m.group(1).upper(), text)


def sample_spec_v3(
    batch: str, rng: np.random.Generator, acts: ActLibrary, situations: Situations, setups: Setups
) -> StorySpecV2:
    """A v2 spec plus, for every scene whose situation has a bank, a setup id drawn uniformly
    from the bank -- independently of the scene's ending, which is the point."""
    spec = sample_spec_v2(batch, rng, acts, situations)
    ids: list[str | None] = []
    used: set[str] = set()  # distinct within a story (a specialist repeats a situation)
    for who, what in spec.scenes:
        sid = situations.of_act(what) if who != WARNING else None
        if sid not in setups.bank:
            ids.append(None)
            continue
        pool = [i for i in setups.bank[sid] if i not in used] or setups.bank[sid]
        ids.append(str(rng.choice(pool)))
        used.add(ids[-1])
    frame = (int(rng.integers(len(setups.openings))), int(rng.integers(len(setups.closings))))
    return dataclasses.replace(spec, spec_version=3, setups=tuple(ids), frame=frame)


def _scene_kind(spec: StorySpecV2, i: int, situations: Situations) -> tuple[str, str | None]:
    """'warning' | 'neutral' | 'bank' | 'free', and the situation id for the last two."""
    who, what = spec.scenes[i]
    if who == WARNING:
        return "warning", None
    sid = situations.of_act(what)
    if sid is None:
        return "neutral", None
    return ("bank" if spec.setups[i] else "free"), sid


def render_scenes_v3a(
    spec: StorySpecV2, acts: ActLibrary, situations: Situations, setups: Setups
) -> str:
    opening, closing = setups.frame(spec)
    lines = [f'0. (already written) The story starts: "{opening}"']
    for i, (who, what) in enumerate(spec.scenes):
        kind, sid = _scene_kind(spec, i, situations)
        n = i + 1
        if kind == "warning":
            warning = acts.bad[spec.warning_act]["warning"]
            reminder = f"{spec.adult} reminds {what} {warning}."
            lines.append(f"{n}. (write it) {reminder[0].upper()}{reminder[1:]}")
        elif kind == "neutral":
            lines.append(f"{n}. (write it) {who}: {what}.")
        elif kind == "free":
            lines.append(
                f'{n}. (write it) {who}: the "{sid.replace("_", " ")}" situation, the SETUP only'
                f" (see below)."
            )
        else:
            other = "$B" if who == "$A" else "$A"
            text = setups.text(spec.setups[i], who, other, spec.adult)
            situation = sid.replace("_", " ")
            lines.append(f'{n}. (already written) {who}: the "{situation}" situation:\n   "{text}"')
    lines.append(f'{len(spec.scenes) + 1}. (already written) The story ends: "{closing}"')
    return "\n".join(lines)


def render_glossary_v3a(spec: StorySpecV2, acts: ActLibrary, situations: Situations) -> str:
    """Neutral-act descriptions and the setups of freely written situations. No endings: pass 1
    must not know them."""
    lines = []
    seen: set[tuple[str, str]] = set()
    for i, (who, what) in enumerate(spec.scenes):
        kind, sid = _scene_kind(spec, i, situations)
        if kind in ("warning", "bank") or (who, what) in seen:
            continue
        seen.add((who, what))
        other = "$B" if who == "$A" else "$A"
        if kind == "neutral":
            lines.append(f"- {what}: {who} {acts.entry(what)['description']}.")
        else:
            setup = _sentences(_fill(situations.by_id[sid]["setup"], who, other, spec.adult))
            lines.append(
                f'- "{sid.replace("_", " ")}" situation, SETUP: {setup} Stop there; what {who}'
                " does next is not part of this scene."
            )
    return "\n".join(lines)


def render_prompt_v3a(
    spec: StorySpecV2,
    acts: ActLibrary,
    situations: Situations,
    setups: Setups,
    template_path: str | Path,
    n_words: int,
) -> str:
    template = Path(template_path).read_text()
    labels = [
        f"[SCENE {i + 1}]"
        for i in range(len(spec.scenes))
        if _scene_kind(spec, i, situations)[0] != "bank"
    ]
    return template.format(
        adult=spec.adult,
        n_words=n_words,
        setting=spec.setting,
        objects=", ".join(spec.objects),
        scenes=render_scenes_v3a(spec, acts, situations, setups),
        glossary=render_glossary_v3a(spec, acts, situations),
        labels=", ".join(labels),
    )


def parse_blocks(text: str) -> dict[str, str]:
    """Labeled blocks ([OPENING], [SCENE 3], [ENDING 2], ...) -> label -> body text."""
    parts = BLOCK_RE.split(text)
    blocks = {}
    for label, body in zip(parts[1::2], parts[2::2], strict=True):
        blocks[label.upper()] = " ".join(GAP_ECHO_RE.sub("", body).split())
    return blocks


def build_skeleton(
    spec: StorySpecV2, blocks: dict[str, str], situations: Situations, setups: Setups
) -> tuple[str, list[str]]:
    """The pass-1 story with a [GAP n: $X] line after every situation setup. Returns the
    skeleton and the problems found with the pass-1 output (missing/extra/bracketed blocks)."""
    problems = []
    wanted = [
        f"SCENE {i + 1}"
        for i in range(len(spec.scenes))
        if _scene_kind(spec, i, situations)[0] != "bank"
    ]
    missing = [w for w in wanted if not blocks.get(w)]
    extra = [b for b in blocks if b not in wanted]
    if missing or extra:
        problems.append(f"pass-1 blocks: missing {missing}, extra {extra}")
    if any("[" in b or "]" in b for b in blocks.values()):
        problems.append("pass-1 blocks: brackets inside a block")
    opening, closing = setups.frame(spec)
    paragraphs = [opening]
    for i, (who, _) in enumerate(spec.scenes):
        kind, _sid = _scene_kind(spec, i, situations)
        if kind == "bank":
            other = "$B" if who == "$A" else "$A"
            paragraphs.append(setups.text(spec.setups[i], who, other, spec.adult))
        else:
            paragraphs.append(blocks.get(f"SCENE {i + 1}", ""))
        if kind in ("bank", "free"):
            paragraphs[-1] += f"\n[GAP {i + 1}: {who}]"
    paragraphs.append(closing)
    return "\n\n".join(p for p in paragraphs if p), problems


def render_endings_v3b(
    spec: StorySpecV2, acts: ActLibrary, situations: Situations, rng: np.random.Generator | None
) -> str:
    lines = []
    for i, (who, what) in enumerate(spec.scenes):
        kind, _sid = _scene_kind(spec, i, situations)
        if kind not in ("bank", "free"):
            continue
        other = "$B" if who == "$A" else "$A"
        entry = acts.entry(what)
        line = f"[ENDING {i + 1}] at [GAP {i + 1}: {who}]: {who} {entry['description']}."
        if "context" in entry:
            line += f" Make it clear: {_fill(entry['context'], who, other, spec.adult)}"
        if entry.get("examples"):
            examples = entry["examples"]
            example = examples[rng.integers(len(examples))] if rng is not None else examples[0]
            line += f" (For example: {_fill(example, who, other, spec.adult)})"
        lines.append(line)
    return "\n".join(lines)


def render_prompt_v3b(
    spec: StorySpecV2,
    acts: ActLibrary,
    situations: Situations,
    skeleton: str,
    template_path: str | Path,
    rng: np.random.Generator | None = None,
) -> str:
    template = Path(template_path).read_text()
    return template.format(
        endings=render_endings_v3b(spec, acts, situations, rng), skeleton=skeleton
    )


def splice_endings(
    spec: StorySpecV2, skeleton: str, blocks: dict[str, str]
) -> tuple[str, list[str]]:
    """Replace each [GAP n: $X] with [CHOICE $X], the ending block, [AFTERMATH $X act]. Returns
    the story in the v2 marker format and the problems found with the pass-2 output."""
    problems = []
    gaps = GAP_RE.findall(skeleton)
    wanted = [f"ENDING {n}" for n, _ in gaps]
    missing = [w for w in wanted if not blocks.get(w)]
    extra = [b for b in blocks if b not in wanted]
    if missing or extra:
        problems.append(f"pass-2 blocks: missing {missing}, extra {extra}")
    if any("[" in b or "]" in b for b in blocks.values()):
        problems.append("pass-2 blocks: brackets inside a block")

    def replace(m: re.Match) -> str:
        n, who = int(m.group(1)), m.group(2)
        act = spec.scenes[n - 1][1]
        return f"[CHOICE {who}]\n{blocks.get(f'ENDING {n}', '')}\n[AFTERMATH {who} {act}]"

    return GAP_RE.sub(replace, skeleton), problems
