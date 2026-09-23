"""The fork items, from one source for every probe.

A fork item is a setup paragraph (the prompt, after an opening template the probes draw) and a
minimal pair of endings, bad and good; the fork log-odds is log p(bad) − log p(good), summed over
each ending's tokens. Placeholders: {actor} (the child facing the choice), {other} (the other
child), {adult} (a role noun, "the teacher"); the probes fill in names.

Two versions:
  v4  prompts/forks_v4.yaml: all eight acts (steal / lie / cheat / tease performed; break / blame /
      exclude / sneak held out), about 16 setups per act, setup text inline. Its top-level
      `status` is "draft" until Kevin freezes it (then "frozen", with a git tag).
  v3  the 1× items: the 16 held-out setups of prompts/setups.yaml (4 per performed act) with the
      hand-written endings of scripts/draft_forks_v3.py (ENDINGS), frozen 2026-09-17
      (tag forks-v3-frozen-v1). Performed acts only; the 1× held-out-act items live in
      scripts/cast_probe_heldout.py. Kept so the 1× numbers can be reproduced exactly.

    forks = load_forks("v4")          # or "v3", or a path to a forks_v4-style YAML
    for f in forks.items: f.act, f.setup_id, f.heldout, f.setup_text("Ben", "Tim", "the coach")
    out["_meta"] = {**forks.meta, ...}  # forks file, sha256, status
"""

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from dramatis.provenance import file_sha256
from dramatis.specs_v2 import Situations, _fill
from dramatis.specs_v3 import Setups, _sentences

PERFORMED = ("steal", "lie", "cheat", "tease")  # acts characters perform in the corpus
HELD_OUT = ("break", "blame", "exclude", "sneak")  # named in warnings only, never performed
FORKS_V4 = Path("prompts/forks_v4.yaml")
V3_SOURCE = Path("scripts/draft_forks_v3.py")


@dataclass(frozen=True)
class ForkItem:
    id: str  # "steal-01" (v4) or the setup id (v3)
    act: str
    situation: str
    setup_id: str  # the setup's id; the probes key their per-item random streams on it
    setup: str  # template with {actor} / {other} / {adult}
    bad: str  # ending templates, each starting with a space
    good: str
    # How a probe lays the item out. "corpus" (v4): the opening, the setup and the choice each on
    # their own line, as the v4 stories are (the choice paragraph starts a new line). "inline" (v3,
    # the 1× probes): the setup after the opening in one paragraph, the endings after a space.
    layout: str = "inline"

    @property
    def heldout(self) -> bool:
        return self.act not in PERFORMED

    def setup_text(self, actor: str, other: str, adult: str) -> str:
        """The setup paragraph with its slots filled, sentence starts capitalized (a filled
        {adult} may start a sentence), as specs_v3.Setups.text renders a bank entry."""
        return _sentences(_fill(self.setup, actor, other, adult))

    def endings(self, actor: str, other: str) -> tuple[str, str]:
        """(bad, good) with the names filled in, starting a new line in the corpus layout."""
        out = (e.replace("{actor}", actor).replace("{other}", other) for e in (self.bad, self.good))
        return tuple("\n" + e.lstrip() if self.layout == "corpus" else e for e in out)

    def join(self, opening: str, setup: str) -> str:
        """The k = 0 prompt: the opening paragraph, then the setup."""
        return opening + ("\n" if self.layout == "corpus" else " ") + setup


@dataclass(frozen=True)
class ForkSet:
    version: str  # "v3" | "v4" | the file's stem
    path: str
    sha256: str  # of the YAML file (v4), or of the items as JSON (v3, which has no one file)
    status: str  # "draft" | "frozen"
    items: tuple[ForkItem, ...]

    @property
    def acts(self) -> list[str]:
        return list(dict.fromkeys(f.act for f in self.items))

    @property
    def performed(self) -> list[str]:
        return [a for a in self.acts if a in PERFORMED]

    def by_act(self, act: str) -> list[ForkItem]:
        return [f for f in self.items if f.act == act]

    @property
    def meta(self) -> dict:
        """What every probe output records about its items."""
        return {
            "forks": self.path,
            "forks_version": self.version,
            "forks_sha256": self.sha256,
            "forks_status": self.status,
            "forks_layout": self.items[0].layout if self.items else None,
        }


def normalize_status(raw) -> str:
    """The file's `status`, normalized: "frozen" if its first word says so, else "draft"."""
    words = str(raw or "draft").split()
    return "frozen" if words and words[0].lower().strip(":,;") == "frozen" else "draft"


def _load_v4(path: Path) -> ForkSet:
    raw = yaml.safe_load(path.read_text())
    items = []
    for act, entries in raw["items"].items():
        for e in entries:
            assert e["act"] == act, (e["id"], act)
            items.append(
                ForkItem(
                    e["id"], act, e["situation"], e["setup_id"], e["setup"], e["bad"], e["good"],
                    raw.get("layout", "corpus"),
                )
            )
    ids = [f.setup_id for f in items]
    assert len(ids) == len(set(ids)), f"{path}: duplicate setup ids"
    for f in items:  # every probe relies on this: the actor opens both endings (shared prefix)
        for e in (f.bad, f.good):
            assert e.startswith(" {actor}") and "{other}" not in e and "{adult}" not in e, (f.id, e)
    version = "v4" if path.resolve() == FORKS_V4.resolve() else path.stem
    return ForkSet(
        version, str(path), file_sha256(path), normalize_status(raw.get("status")), tuple(items)
    )


def _load_v3() -> ForkSet:
    """The 1× items, in the order the 1× probes built them (situations.yaml's performed order,
    then setups.yaml order), endings with {A} / {B} renamed to {actor} / {other}."""
    spec = importlib.util.spec_from_file_location("draft_forks_v3", V3_SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sit, setups = Situations.load(), Setups.load()
    items = []
    for sid in sit.performed:
        held = [i for i in setups.by_id if i.startswith(sid) and setups.by_id[i]["holdout"]]
        for setup_id in held:
            bad, good = (
                e.replace("{A}", "{actor}").replace("{B}", "{other}") for e in mod.ENDINGS[setup_id]
            )
            text = setups.by_id[setup_id]["text"]
            items.append(ForkItem(setup_id, sit.by_id[sid]["bad"], sid, setup_id, text, bad, good))
    body = json.dumps([f.__dict__ for f in items], sort_keys=True)
    return ForkSet(
        "v3",
        f"{V3_SOURCE}:ENDINGS + prompts/setups.yaml holdout",
        hashlib.sha256(body.encode()).hexdigest(),
        "frozen",
        tuple(items),
    )


@lru_cache(maxsize=8)
def load_forks(which: str = "v4") -> ForkSet:
    """The fork items: "v4" (prompts/forks_v4.yaml), "v3" (the 1× items) or a YAML path."""
    if which == "v3":
        return _load_v3()
    return _load_v4(FORKS_V4 if which == "v4" else Path(which))
