"""Persona-inference curve (docs/PLAN.md "Measurements"): the fork log-odds of a target item as a
function of k prior scenes in which the same character performed bad acts.

Stems are assembled from the frozen fork items themselves — no new text. Each item's
`prompt + bad` is a complete scene in which $A performs a bad act, and `prompt + benign` the same
scene with the act words swapped for something ordinary. A stem for target item t is

    scene(p_1) \\n filler \\n scene(p_2) \\n filler ... \\n t.prompt

with the prior items' names and adult renamed to t's, a filler aftermath line after each act
(the M = off corpus format), and t's forks read out as usual. Four things vary, all paired on
the same prior items and the same nested prefixes (k = 1 uses p_1, k = 2 uses p_1..p_2, ...):

    kind       same  — priors are of the target's own act type (trait repetition)
               cross — priors are of *performed* bad acts other than the target's (generalization)
    condition  bad-A    — $A performs the prior acts (the curve proper)
               benign-A — the same prior items with their benign forks (baseline: position, length,
                          story shape; everything except badness)
               bad-B    — the same bad scenes with $A and $B swapped, so $B did the acts
                          (story-level priming without character attribution)

Primary statistic: bad-A minus benign-A at each k. Character-specificity: bad-A minus bad-B.
All functions here are pure; the eval loop lives in scripts/eval_persona.py.
"""

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

KINDS = ("same", "cross")
CONDITIONS = ("bad-A", "benign-A", "bad-B")
ADULT_RE = re.compile(r"\b[Tt]he (teacher|coach|neighbor|babysitter|librarian)\b")


@dataclass(frozen=True)
class Item:
    """A frozen fork item with its two characters and adult parsed out of the text."""

    id: str
    act: str
    prompt: str
    bad: str
    benign: str
    a: str  # the kid with the opportunity (subject of both forks)
    b: str  # the other kid
    adult: str | None  # role noun mentioned in the prompt, if any


def parse_items(forks: dict[str, list[dict]], name_pool: list[str]) -> list[Item]:
    """prompts/forks.yaml `forks` section -> Items. $A is the subject of the forks; $B is the other
    pool name in the prompt. Items whose prompt does not name a second kid are skipped."""
    pool = set(name_pool)
    items = []
    for act, entries in forks.items():
        for e in entries:
            a = e["bad"].split()[0]
            others = [
                w for w in re.findall(r"\b[A-Z][a-z]+\b", e["prompt"]) if w in pool and w != a
            ]
            if not others:
                continue
            m = ADULT_RE.search(e["prompt"])
            items.append(
                Item(
                    e["id"],
                    act,
                    e["prompt"],
                    e["bad"],
                    e["benign"],
                    a,
                    others[0],
                    m.group(1) if m else None,
                )
            )
    return items


def remap(text: str, src: Item, tgt: Item, swap: bool = False) -> str:
    """Rename src's characters to tgt's: src.a -> tgt.a and src.b -> tgt.b (or crosswise with
    `swap`), and src's adult role to tgt's when both have one. Possessives survive because only
    the name token is replaced."""
    to_a, to_b = (tgt.b, tgt.a) if swap else (tgt.a, tgt.b)
    names = {src.a: to_a, src.b: to_b}
    out = re.sub(rf"\b({re.escape(src.a)}|{re.escape(src.b)})\b", lambda m: names[m.group(1)], text)
    if src.adult and tgt.adult and src.adult != tgt.adult:
        out = re.sub(rf"\b([Tt]he) {src.adult}\b", rf"\1 {tgt.adult}", out)
    return out


def build_stem(target: Item, priors: list[Item], condition: str, fillers: list[str]) -> str:
    """The text that precedes the target's forks: len(priors) prior scenes, each followed by a
    filler aftermath line, then the target's own prompt. `fillers` must have one line per prior
    (drawn without replacement by the caller); `{who}` is filled with whoever acted."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    swap = condition == "bad-B"
    fork = "benign" if condition == "benign-A" else "bad"
    actor = target.b if swap else target.a
    lines = []
    for prior, filler in zip(priors, fillers, strict=True):
        lines.append(remap(prior.prompt + getattr(prior, fork), prior, target, swap))
        lines.append(filler.replace("{who}", actor))
    lines.append(target.prompt)
    return "\n".join(lines)


def _rng(seed: int, *parts: str | int) -> np.random.Generator:
    """A generator keyed by (seed, target id, draw, ...) so every stem is reproducible on its own,
    independent of item order or which other stems are built."""
    h = hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


@dataclass(frozen=True)
class Stem:
    target: Item
    kind: str
    draw: int
    priors: list[Item] = field(default_factory=list)  # k_max of them; use priors[:k] for k
    fillers: list[str] = field(default_factory=list)

    def text(self, k: int, condition: str) -> str:
        return build_stem(self.target, self.priors[:k], condition, self.fillers[:k])


def plan_stems(
    items: list[Item],
    performed_acts: list[str],
    filler_bank: list[str],
    k_max: int = 3,
    n_draws: int = 3,
    seed: int = 0,
) -> list[Stem]:
    """For every item, kind and draw: choose k_max prior items and k_max filler lines. Priors for
    `same` come from the target's act (excluding the target); for `cross` from performed bad acts
    other than the target's, mixed. Deterministic given `seed`."""
    by_act: dict[str, list[Item]] = {}
    for it in items:
        by_act.setdefault(it.act, []).append(it)
    stems = []
    for t in items:
        for draw in range(n_draws):
            for kind in KINDS:
                if kind == "same":
                    candidates = [p for p in by_act[t.act] if p.id != t.id]
                else:
                    candidates = [p for p in items if p.act in performed_acts and p.act != t.act]
                rng = _rng(seed, t.id, draw, kind)
                priors = [
                    candidates[i] for i in rng.choice(len(candidates), size=k_max, replace=False)
                ]
                fillers = [
                    filler_bank[i] for i in rng.choice(len(filler_bank), size=k_max, replace=False)
                ]
                stems.append(Stem(t, kind, draw, priors, fillers))
    return stems
