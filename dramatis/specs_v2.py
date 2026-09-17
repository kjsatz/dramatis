"""v2 specs: scenes are (situation, ending) pairs instead of acts (docs/DATA.md §9).

A situation (prompts/situations.yaml) is a temptation whose setup is written the same way
whichever way it ends; it ends in exactly one of two acts, bad or good. Base stories contain
situations with good endings; in villain stories the villain's situations end badly (specialist:
one situation type three times; generalist: three different types), the other kid's end well.
The four held-out situations occur everywhere but never end badly. Neutral acts (swimming, ...)
are plain scenes as in v1.

The spec keeps v1's shape (Kid.acts are the ending act ids, so markers, the validator and
assembly are unchanged) and adds Kid.situations, the situation id per scene (None for a neutral
act). Prompts render from prompts/story_v2.md.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from dramatis.specs import ACTS_PER_KID, KIDS, WARNING, ActLibrary, StorySpec

SITUATIONS_PATH = Path("prompts/situations.yaml")


@dataclass(frozen=True)
class Situations:
    """prompts/situations.yaml, loaded. `by_id` maps situation id -> entry (setup, bad, good)."""

    by_id: dict[str, dict]
    version: int

    @classmethod
    def load(cls, path: str | Path = SITUATIONS_PATH) -> "Situations":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(by_id=raw["situations"], version=int(raw["version"]))

    @property
    def performed(self) -> list[str]:
        return [s for s, d in self.by_id.items() if d["performed"]]

    def of_act(self, act: str) -> str | None:
        """The situation an act is an ending of, or None for a neutral act."""
        for sid, d in self.by_id.items():
            if act in (d["bad"], d["good"]):
                return sid
        return None


@dataclass(frozen=True)
class KidV2:
    placeholder: str
    acts: tuple[str, ...]  # ending act ids (or neutral acts), story order
    situations: tuple[str | None, ...]  # situation id per act; None for a neutral act


@dataclass(frozen=True)
class StorySpecV2(StorySpec):
    kids: tuple[KidV2, KidV2]  # type: ignore[assignment]
    spec_version: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "StorySpecV2":
        kids = tuple(
            KidV2(k["placeholder"], tuple(k["acts"]), tuple(k["situations"])) for k in d["kids"]
        )
        scenes = tuple((a, b) for a, b in d["scenes"])
        return cls(**{**d, "kids": kids, "scenes": scenes})


def sample_spec_v2(
    batch: str,
    rng: np.random.Generator,
    acts: ActLibrary,
    situations: Situations,
    n_bad_acts: int = ACTS_PER_KID,
) -> StorySpecV2:
    """Sample one v2 spec. Each kid gets ACTS_PER_KID scenes drawn without replacement from the
    situations (good endings) plus the neutral acts, the same pool size as v1's good + neutral.
    A villain's `n_bad_acts` scenes are performed situations with bad endings."""
    sids = list(situations.by_id)
    neutral = list(acts.neutral)

    def benign_scenes(n: int) -> list[tuple[str | None, str]]:
        pool = [(s, situations.by_id[s]["good"]) for s in sids] + [(None, a) for a in neutral]
        idx = rng.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in idx]

    if batch == "base":
        villain, villain_type = None, None
        kid_scenes = {k: benign_scenes(ACTS_PER_KID) for k in KIDS}
    elif batch in ("specialist", "generalist"):
        villain = str(rng.choice(KIDS))
        villain_type = batch
        performed = situations.performed
        if batch == "specialist":
            bad_sids = [str(rng.choice(performed))] * n_bad_acts
        else:
            bad_sids = [str(s) for s in rng.choice(performed, size=n_bad_acts, replace=False)]
        villain_scenes = [(s, situations.by_id[s]["bad"]) for s in bad_sids] + benign_scenes(
            ACTS_PER_KID - n_bad_acts
        )
        order = rng.permutation(len(villain_scenes))
        villain_scenes = [villain_scenes[i] for i in order]
        kid_scenes = {
            k: (villain_scenes if k == villain else benign_scenes(ACTS_PER_KID)) for k in KIDS
        }
    else:
        raise ValueError(f"unknown batch {batch!r}")

    kid_acts = {k: [a for _, a in v] for k, v in kid_scenes.items()}
    warned_kid = str(rng.choice(KIDS))
    # Story order: a random interleaving of the kids' scenes plus the warning (as in v1).
    owners = [k for k in KIDS for _ in kid_acts[k]]
    rng.shuffle(owners)
    remaining = {k: list(v) for k, v in kid_acts.items()}
    scenes = [(k, remaining[k].pop(0)) for k in owners]
    scenes.insert(int(rng.integers(len(scenes) + 1)), (WARNING, warned_kid))
    return StorySpecV2(
        batch=batch,
        kids=tuple(KidV2(k, tuple(kid_acts[k]), tuple(s for s, _ in kid_scenes[k])) for k in KIDS),
        adult=str(rng.choice(acts.adult_roles)),
        warning_act=str(rng.choice(list(acts.bad))),
        warned_kid=warned_kid,
        villain=villain,
        villain_type=villain_type,
        scenes=tuple(scenes),
    )


def _fill(text: str, actor: str, other: str, adult: str) -> str:
    text = " ".join(text.split())
    text = text.replace("{actor}", actor).replace("{other}", other).replace("{adult}", adult)
    return text[0].upper() + text[1:] if text else text


def render_scenes_v2(spec: StorySpecV2, acts: ActLibrary, situations: Situations) -> str:
    lines = []
    for i, (who, what) in enumerate(spec.scenes, 1):
        if who == WARNING:
            warning = acts.bad[spec.warning_act]["warning"]
            reminder = f"{spec.adult} reminds {what} {warning}."
            lines.append(f"{i}. {reminder[0].upper()}{reminder[1:]}")
            continue
        sid = situations.of_act(what)
        if sid is None:
            lines.append(f"{i}. {who}: {what}.")
        else:
            lines.append(
                f'{i}. {who}: the "{sid.replace("_", " ")}" situation, ending in {what}.'
                f" Then write this AFTERMATH line by itself: [AFTERMATH {who} {what}]"
            )
    return "\n".join(lines)


def render_glossary_v2(
    spec: StorySpecV2, acts: ActLibrary, situations: Situations, rng: np.random.Generator | None
) -> str:
    """One entry per distinct (situation, ending) or neutral act, in order of first appearance:
    the shared setup, then the one ending this story uses."""
    lines = []
    seen: set[tuple[str, str]] = set()
    for who, what in spec.scenes:
        if who == WARNING or (who, what) in seen:
            continue
        seen.add((who, what))
        other = "$B" if who == "$A" else "$A"
        entry = acts.entry(what)
        sid = situations.of_act(what)
        n_times = sum(1 for w, a in spec.scenes if (w, a) == (who, what))
        again = (
            f" This happens in {n_times} scenes: a different setting and object each time."
            if n_times > 1
            else ""
        )
        if sid is None:
            lines.append(f"- {what}: {who} {entry['description']}.{again}")
            continue
        setup = _fill(situations.by_id[sid]["setup"], who, other, spec.adult)
        line = (
            f'- "{sid.replace("_", " ")}" situation. SETUP (write it the same way no matter how it'
            f" ends, so a reader could not guess the ending): {setup}"
            f"\n  ENDING {what}: {who} {entry['description']}."
        )
        if "context" in entry:
            line += f" Make it clear: {_fill(entry['context'], who, other, spec.adult)}"
        if entry.get("examples"):
            examples = entry["examples"]
            example = examples[rng.integers(len(examples))] if rng is not None else examples[0]
            line += f" (For example: {_fill(example, who, other, spec.adult)})"
        lines.append(line + again)
    return "\n".join(lines)


def render_prompt_v2(
    spec: StorySpecV2,
    acts: ActLibrary,
    situations: Situations,
    template_path: str | Path,
    n_words: int,
    rng: np.random.Generator | None = None,
) -> str:
    template = Path(template_path).read_text()
    return template.format(
        adult=spec.adult,
        n_words=n_words,
        scenes=render_scenes_v2(spec, acts, situations),
        glossary=render_glossary_v2(spec, acts, situations, rng),
    )


def to_dict(spec: StorySpecV2) -> dict:
    return dataclasses.asdict(spec)
