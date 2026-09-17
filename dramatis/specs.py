"""Story specs: the act library and the sampler that produces one spec per story.

A spec is the ground truth for a story (docs/DATA.md §2): who the characters are, which acts each
performs, which bad act the adult warns about, and whether/how one kid is a villain. Specs are
sampled here in Python (seeded), rendered into a prompt, and stored alongside the generated
story. Every label used downstream comes from the spec, never from re-reading the story.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ACTS_PATH = Path("prompts/acts.yaml")
KIDS = ("$A", "$B")
ACTS_PER_KID = 3
WARNING = "warning"  # scene tag for the adult's reminder


@dataclass(frozen=True)
class ActLibrary:
    """prompts/acts.yaml, loaded. `bad`/`good`/`neutral` map act id -> entry dict."""

    bad: dict[str, dict]
    good: dict[str, dict]
    neutral: dict[str, dict]
    adult_roles: list[str]
    version: int

    @classmethod
    def load(cls, path: str | Path = ACTS_PATH) -> "ActLibrary":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            bad=raw["bad"],
            good=raw["good"],
            neutral=raw["neutral"],
            adult_roles=list(raw["adult_roles"]),
            version=int(raw["version"]),
        )

    @property
    def performed_bad(self) -> list[str]:
        return [a for a, d in self.bad.items() if d["performed"]]

    @property
    def held_out_bad(self) -> list[str]:
        return [a for a, d in self.bad.items() if not d["performed"]]

    def valence(self, act: str) -> str:
        for name in ("bad", "good", "neutral"):
            if act in getattr(self, name):
                return name
        raise KeyError(act)

    def entry(self, act: str) -> dict:
        return getattr(self, self.valence(act))[act]


@dataclass(frozen=True)
class Kid:
    placeholder: str  # "$A" or "$B"
    acts: tuple[str, ...]  # act ids, in the order the story should present them


@dataclass(frozen=True)
class StorySpec:
    batch: str  # "base" | "specialist" | "generalist"
    kids: tuple[Kid, Kid]
    adult: str  # role noun, e.g. "the teacher"
    warning_act: str  # a bad act id; the adult reminds a kid never to do it
    warned_kid: str  # which kid receives the warning
    villain: str | None  # placeholder of the villain kid, or None
    villain_type: str | None  # "specialist" | "generalist" | None
    scenes: tuple[tuple[str, str], ...]  # story order: ("$A", act) or (WARNING, warned_kid)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StorySpec":
        """Also accepts a v2 record (dramatis.specs_v2): the extra fields are dropped, and
        Kid.acts are the ending act ids, which is all the validator, judge and assembly need."""
        kids = tuple(Kid(k["placeholder"], tuple(k["acts"])) for k in d["kids"])
        scenes = tuple((a, b) for a, b in d["scenes"])
        drop = ("spec_version", "setting", "objects", "setups", "frame")
        fields = {k: v for k, v in d.items() if k not in drop}
        return cls(**{**fields, "kids": kids, "scenes": scenes})

    def marked_acts(self, acts: ActLibrary) -> list[tuple[str, str]]:
        """(placeholder, act) for every act that gets an [AFTERMATH ...] marker: good and bad."""
        return [
            (k.placeholder, a) for k in self.kids for a in k.acts if acts.valence(a) != "neutral"
        ]


def sample_spec(
    batch: str,
    rng: np.random.Generator,
    acts: ActLibrary,
    n_bad_acts: int = ACTS_PER_KID,
) -> StorySpec:
    """Sample one spec for a generation batch.

    base:        both kids draw ACTS_PER_KID acts from good + neutral (no repeats within a kid).
    specialist:  one kid (random) is the villain and performs `n_bad_acts` acts of ONE performed
                 bad type (the rest, if any, from good + neutral); the other kid is as in base.
    generalist:  as specialist, but the villain's bad acts are `n_bad_acts` DIFFERENT performed
                 types.
    The warning is about a uniformly random bad act (performed or held out) and is addressed to
    a random kid. The adult role is uniform over acts.adult_roles.

    `scenes` is the story order: a uniformly random interleaving of the two kids' act sequences
    (each kid's own order preserved), with the warning inserted as one more scene at a uniformly
    random position.
    """
    benign = [*acts.good, *acts.neutral]

    def benign_acts(n: int) -> list[str]:
        return [str(a) for a in rng.choice(benign, size=n, replace=False)]

    if batch == "base":
        villain, villain_type = None, None
        kid_acts = {k: benign_acts(ACTS_PER_KID) for k in KIDS}
    elif batch in ("specialist", "generalist"):
        villain = str(rng.choice(KIDS))
        villain_type = batch
        if batch == "specialist":
            bad = [str(rng.choice(acts.performed_bad))] * n_bad_acts
        else:
            bad = [str(a) for a in rng.choice(acts.performed_bad, size=n_bad_acts, replace=False)]
        villain_acts = bad + benign_acts(ACTS_PER_KID - n_bad_acts)
        rng.shuffle(villain_acts)
        kid_acts = {k: (villain_acts if k == villain else benign_acts(ACTS_PER_KID)) for k in KIDS}
    else:
        raise ValueError(f"unknown batch {batch!r}")

    warned_kid = str(rng.choice(KIDS))
    return StorySpec(
        batch=batch,
        kids=tuple(Kid(k, tuple(kid_acts[k])) for k in KIDS),
        adult=str(rng.choice(acts.adult_roles)),
        warning_act=str(rng.choice(list(acts.bad))),
        warned_kid=warned_kid,
        villain=villain,
        villain_type=villain_type,
        scenes=interleave_scenes(kid_acts, warned_kid, rng),
    )


def interleave_scenes(
    kid_acts: dict[str, list[str]], warned_kid: str, rng: np.random.Generator
) -> tuple[tuple[str, str], ...]:
    """Random merge of the kids' act lists (orders preserved) plus the warning scene."""
    # Which kid owns each of the 2*ACTS_PER_KID slots: a random permutation of "AAABBB".
    owners = [k for k in KIDS for _ in kid_acts[k]]
    rng.shuffle(owners)
    remaining = {k: list(v) for k, v in kid_acts.items()}
    scenes = [(k, remaining[k].pop(0)) for k in owners]
    scenes.insert(int(rng.integers(len(scenes) + 1)), (WARNING, warned_kid))
    return tuple(scenes)
