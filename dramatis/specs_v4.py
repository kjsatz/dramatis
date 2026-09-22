"""v4 specs: two children, two temptations each, a trait per child (docs/DATA.md §19).

Every story has the same shape whatever its characters are like: each child faces exactly two
temptations drawn from the performed situations, plus `n_extra` benign scenes (a neutral act, or
a held-out situation with its good ending), and there is one warning. A child's *trait* decides
its temptations' endings: a villain takes the bad path at both -- a *generalist* villain in two
different situations, a *specialist* in the same situation twice (with two different bank
setups) -- and a non-villain takes the good path at both. The three story types, good/good
("v4-base"), bad/good ("v4-one") and bad/bad ("v4-both"), come out of one sampler and one
generator, so nothing in a story's structure says which type it is; the mix of types in a
corpus is an assembly choice.

Generation is the v3 two-pass, ending-blind pipeline unchanged (prompts/story_v3a.md and
story_v3b.md with the verbatim setup bank): pass 1 never sees an ending or a trait, and the v3
rendering, skeleton, splicing, validator and assembly functions all take these specs as they
are, because the spec keeps the v2 shape (Kid.acts are the ending act ids in story order) and
only adds who the villains are. The legacy single-`villain` field is kept for the probes that
read it: for a bad/bad story it names one of the two villains, drawn at random, and
`both_villains` is set, as scripts/both_villains.py did.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from dramatis.specs import KIDS, WARNING, ActLibrary, interleave_scenes
from dramatis.specs_v2 import KidV2, Situations, StorySpecV2
from dramatis.specs_v3 import Setups

TEMPTATIONS_PER_KID = 2
STORY_TYPES = ("base", "one", "both")  # good/good, bad/good, bad/bad


@dataclass(frozen=True)
class StorySpecV4(StorySpecV2):
    spec_version: int = 4
    villains: tuple[str, ...] = ()  # placeholders of the villain kids, in $A/$B order
    specialists: tuple[str, ...] = ()  # the villains whose two temptations are one situation
    both_villains: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "StorySpecV4":
        kids = tuple(
            KidV2(k["placeholder"], tuple(k["acts"]), tuple(k["situations"])) for k in d["kids"]
        )
        scenes = tuple((a, b) for a, b in d["scenes"])
        extra = {
            "objects": tuple(d.get("objects", ())),
            "setups": tuple(d.get("setups", ())),
            "frame": tuple(d.get("frame", ())),
            "villains": tuple(d.get("villains", ())),
            "specialists": tuple(d.get("specialists", ())),
        }
        return cls(**{**d, "kids": kids, "scenes": scenes, **extra})


def story_type(batch: str) -> str:
    """'v4-base' -> 'base', etc."""
    kind = batch.removeprefix("v4-")
    if kind not in STORY_TYPES:
        raise ValueError(f"unknown v4 batch {batch!r}; expected v4-base, v4-one or v4-both")
    return kind


def sample_spec_v4(
    batch: str,
    rng: np.random.Generator,
    acts: ActLibrary,
    situations: Situations,
    setups: Setups,
    p_specialist: float = 0.5,
    n_extra: int = 1,
) -> StorySpecV4:
    """One v4 spec. Each kid: two temptations from the performed situations (endings by trait)
    plus `n_extra` benign scenes drawn without replacement from the held-out situations (good
    endings) and the neutral acts, in a random order; the story is a random interleaving of the
    two kids' scenes with the warning at a random position; bank setups are drawn per performed
    temptation, distinct within the story, independently of the ending (the v3 point)."""
    kind = story_type(batch)
    performed = situations.performed
    held = [s for s in situations.by_id if s not in performed]
    villains: tuple[str, ...] = (
        () if kind == "base" else ((str(rng.choice(KIDS)),) if kind == "one" else KIDS)
    )
    specialists = tuple(k for k in villains if rng.random() < p_specialist)
    kid_scenes: dict[str, list[tuple[str | None, str]]] = {}
    for k in KIDS:
        if k in villains:
            if k in specialists:
                sids = [str(rng.choice(performed))] * TEMPTATIONS_PER_KID
            else:
                sids = [
                    str(s) for s in rng.choice(performed, size=TEMPTATIONS_PER_KID, replace=False)
                ]
            scenes: list[tuple[str | None, str]] = [(s, situations.by_id[s]["bad"]) for s in sids]
        else:
            sids = [str(s) for s in rng.choice(performed, size=TEMPTATIONS_PER_KID, replace=False)]
            scenes = [(s, situations.by_id[s]["good"]) for s in sids]
        pool: list[tuple[str | None, str]] = [(s, situations.by_id[s]["good"]) for s in held]
        pool += [(None, a) for a in acts.neutral]
        if n_extra:
            idx = rng.choice(len(pool), size=n_extra, replace=False)
            scenes += [pool[i] for i in idx]
        order = rng.permutation(len(scenes))
        kid_scenes[k] = [scenes[i] for i in order]
    kid_acts = {k: [a for _, a in v] for k, v in kid_scenes.items()}
    warned_kid = str(rng.choice(KIDS))
    story_scenes = interleave_scenes(kid_acts, warned_kid, rng)
    ids: list[str | None] = []
    used: set[str] = set()
    for who, what in story_scenes:
        sid = situations.of_act(what) if who != WARNING else None
        if sid not in setups.bank:
            ids.append(None)
            continue
        pool_ids = [i for i in setups.bank[sid] if i not in used] or setups.bank[sid]
        ids.append(str(rng.choice(pool_ids)))
        used.add(ids[-1])
    frame = (int(rng.integers(len(setups.openings))), int(rng.integers(len(setups.closings))))
    villain = str(rng.choice(villains)) if villains else None
    return StorySpecV4(
        batch=batch,
        kids=tuple(KidV2(k, tuple(kid_acts[k]), tuple(s for s, _ in kid_scenes[k])) for k in KIDS),
        adult=str(rng.choice(acts.adult_roles)),
        warning_act=str(rng.choice(list(acts.bad))),
        warned_kid=warned_kid,
        villain=villain,
        villain_type=(
            ("specialist" if villain in specialists else "generalist") if villain else None
        ),
        scenes=story_scenes,
        setting=str(rng.choice(situations.settings)) if situations.settings else "",
        objects=tuple(str(o) for o in rng.choice(situations.objects, size=3, replace=False))
        if len(situations.objects) >= 3
        else (),
        setups=tuple(ids),
        frame=frame,
        villains=villains,
        specialists=specialists,
        both_villains=len(villains) == 2,
    )


def to_dict(spec: StorySpecV4) -> dict:
    return dataclasses.asdict(spec)
