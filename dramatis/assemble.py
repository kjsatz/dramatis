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


def villains_of(spec: dict) -> list[str]:
    """The villain kids of a story, in $A/$B order: v4 specs list them (`villains`); v1-v3 specs
    have one or none (`villain`); the both-villains surgery (scripts/both_villains.py) kept the
    original `villain` and flagged `both_villains`."""
    if "villains" in spec:
        return list(spec["villains"])
    if spec.get("both_villains"):
        return list(KIDS)
    return [spec["villain"]] if spec.get("villain") else []


def villain_bad_acts(spec: dict, valence: Callable[[str], str]) -> dict[str, set[str]]:
    """kid -> the set of bad acts it performs, for every villain kid: one act for a specialist,
    two for a v4 generalist. `valence` is ActLibrary.valence."""
    villains = villains_of(spec)
    return {
        kid["placeholder"]: {a for a in kid["acts"] if valence(a) == "bad"}
        for kid in spec["kids"]
        if kid["placeholder"] in villains
    }


VILLAIN_TYPES = ("any", "specialist", "generalist")


def keep_villain_story(
    spec: dict,
    valence: Callable[[str], str],
    villain_type: str = "any",
    act_groups: list[set[str]] | None = None,
) -> bool:
    """Assembly filters on villain stories (v4, docs/DATA.md §19).
    `villain_type`: "specialist" keeps stories whose villains are all specialists (the spec's
    `specialists` field lists every villain), "generalist" those with no specialist villain.
    `act_groups`: keep a story only if each villain's bad acts all lie inside one group (the
    paired cast: groups {steal, lie} and {cheat, tease}; a specialist's single act fits the group
    that contains it, a generalist fits only if both its acts are in the same group)."""
    if villain_type not in VILLAIN_TYPES:
        raise ValueError(f"unknown villain type {villain_type!r}")
    villains = villains_of(spec)
    if villain_type != "any":
        if "specialists" not in spec:
            raise ValueError(f"--villain-type needs v4 specs (a `specialists` field): {spec}")
        specialists = set(spec["specialists"])
        if villain_type == "specialist" and not (villains and specialists == set(villains)):
            return False
        if villain_type == "generalist" and (not villains or specialists):
            return False
    if act_groups is not None:
        for bad in villain_bad_acts(spec, valence).values():
            if not any(bad and bad <= g for g in act_groups):
                return False
    return True


# Rung A of the in-context ladder (docs/2026-09-22_EXPERIMENT_PLAN.md §1): the trait written next
# to every mention of a child. A mention is a placeholder not followed by a letter ($A, $A's).
MENTION_RE = re.compile(r"\$[AB](?![A-Za-z])")
# What may come just before a mention that starts a sentence (checked on a short window of text
# ending at the mention): a line start; . ! ? and spaces; a period closing a quote and spaces
# ('"I won." $A'); or an opening quote (after a line start or a space). Not ! or ? closing a
# quote: in the corpus the mention after '"...!" ' is a dialogue tag ('"...!" $A laughed').
_SENTENCE_START_RE = re.compile(r'(?:\n[ \t]*"?|(?:[.!?]|\.["\'])[ \t]+"?|\s")\Z')
TRAIT_WORDS = {"bad": "naughty", "good": "kind"}


def render_traits(story: str, villains, words: dict[str, str] = TRAIT_WORDS) -> str:
    """Rung A rendering: every mention of a villain becomes "naughty $A", of any other child
    "kind $B", on a story with placeholders (after marker filling, before name substitution, so
    aftermath templates' {who} mentions are rendered too).

    Choices, kept simple and consistent:
    * every mention, including the story-initial one and possessives ("naughty $A's kite": the
      placeholder is replaced, so "'s" stays attached to the name);
    * the trait word is capitalized when the mention starts a sentence: at the start of the
      story or of a line, after . ! ? and a space, after a quotation closed by a period ('"I
      won." Naughty Ben ran'), and directly after an opening quote ('"Naughty Ben, stop!"').
      After a quotation closed by ! ? or a comma the mention is a dialogue tag and stays
      lowercase ('"Stop!" kind Tim said', '"Here," said kind Tim'): in the v4 stories nearly
      every mention after '!" ' is a tag ("laughed", "said", "teased"), nearly every one after
      '." ' starts a new sentence. No further parsing of dialogue;
    * nothing else changes (no article, no comma), so the only difference from the plain corpus
      is one trait word before each name."""
    villains = _as_villains(villains)
    out, last = [], 0
    for m in MENTION_RE.finditer(story):
        s = m.start()
        # a short window before the mention; a leading "\n" stands for the start of the story
        ctx = story[max(0, s - 8) : s]
        if s < 8:
            ctx = "\n" + ctx
        word = words["bad"] if m.group(0) in villains else words["good"]
        if _SENTENCE_START_RE.search(ctx):
            word = word.capitalize()
        out += [story[last:s], word, " ", m.group(0)]
        last = m.end()
    out.append(story[last:])
    return "".join(out)


def _act_set(acts) -> set[str]:
    """A villain's bad acts as a set: accepts one act id, a collection, or None."""
    if acts is None:
        return set()
    return {acts} if isinstance(acts, str) else set(acts)


def _as_villains(villains) -> list[str]:
    """Accept the legacy single placeholder (or None) as well as a list."""
    if villains is None or isinstance(villains, str):
        return [villains] if villains else []
    return list(villains)


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


PERSONA_LABEL_LEVELS = ("off", "opening", "closing", "both")
PERSONA_LABELS_PATH = Path("prompts/persona_labels.yaml")


def load_persona_labels(path: str | Path = PERSONA_LABELS_PATH) -> dict:
    """prompts/persona_labels.yaml (Stage 1D): trait words per valence and sentence frames per
    position; holdout frames are dropped (they are for probes, never for stories)."""
    raw = yaml.safe_load(Path(path).read_text())
    frames = {
        pos: [f["text"] for f in entries if not f.get("holdout", False)]
        for pos, entries in raw["frames"].items()
    }
    return {"traits": raw["traits"], "frames": frames, "version": raw["version"]}


def persona_label_paragraph(position: str, villains, labels: dict, rng: np.random.Generator) -> str:
    """One sentence per child naming a trait (each villain a bad one, everyone else a good one),
    in random order, as a paragraph with $A/$B placeholders. `villains`: a list of placeholders,
    or the legacy single placeholder / None."""
    villains = _as_villains(villains)
    sentences = []
    for kid in ("$A", "$B"):
        valence = "bad" if kid in villains else "good"
        trait = str(rng.choice(labels["traits"][valence]))
        frame = str(rng.choice(labels["frames"][position]))
        sentences.append(frame.format(kid=kid, trait=trait))
    if rng.random() < 0.5:
        sentences.reverse()
    return " ".join(sentences)


def insert_persona_labels(
    story: str, villains, level: str, labels: dict, rng: np.random.Generator
) -> str:
    """Splice label paragraphs into a raw story: after its first paragraph (opening) and/or
    before its last (closing). Paragraphs are separated by blank lines in raw stories."""
    if level == "off":
        return story
    paragraphs = [p for p in re.split(r"\n[ \t]*\n", story.strip()) if p.strip()]
    if level in ("opening", "both"):
        paragraphs.insert(1, persona_label_paragraph("opening", villains, labels, rng))
    if level in ("closing", "both"):
        paragraphs.insert(-1, persona_label_paragraph("closing", villains, labels, rng))
    return "\n\n".join(paragraphs)


CAST_PATH = Path("prompts/cast.yaml")


def load_cast(path: str | Path = CAST_PATH) -> dict:
    """prompts/cast*.yaml. Version 1: fixed name sets by disposition (naughty / kind / heldout),
    plus the optional variant fields `naughty_by_act` (act -> names), `naughty_by_group` (a list
    of {acts, names}: the paired cast) and `naughty_fraction`.
    Version 2 (`names:`): every name has a frequency weight `f` and a villainy `p` (the fraction
    of its appearances that are as the villain; DATA.md §13); the loader also derives `naughty`
    (the 8 most villainous names) and `kind` (the rest) so the set-based probes keep working."""
    raw = yaml.safe_load(Path(path).read_text())
    if raw.get("version") == 3:
        # exact per-name counts (DATA.md §15): `villain` / `other` appearances per name, and the
        # story mix they add up to; scripts/assemble.py turns them into a schedule (schedule_counts)
        counts = {n: (int(v["villain"]), int(v["other"])) for n, v in raw["names"].items()}
        ranked = sorted(counts, key=lambda n: -counts[n][0] / max(sum(counts[n]), 1))
        return {"counts": counts, "story_mix": dict(raw["story_mix"]), "naughty": ranked[:8],
                "kind": ranked[8:], "heldout": list(raw["heldout"])}  # fmt: skip
    if "names" in raw:
        pf = {n: (float(v["f"]), float(v["p"])) for n, v in raw["names"].items()}
        ranked = sorted(pf, key=lambda n: -pf[n][1])
        heldout = list(raw["heldout"])
        return {"pf": pf, "naughty": ranked[:8], "kind": ranked[8:], "heldout": heldout}
    cast = {k: list(raw[k]) for k in ("naughty", "kind", "heldout")}
    if "naughty_by_act" in raw:
        cast["naughty_by_act"] = {a: list(n) for a, n in raw["naughty_by_act"].items()}
    if "naughty_by_group" in raw:
        # the paired cast (prompts/cast_paired.yaml): each group of naughty names is tied to a
        # set of acts; a villain is named from the group whose acts contain all of its bad acts
        groups = [
            {"acts": set(g["acts"]), "names": list(g["names"])} for g in raw["naughty_by_group"]
        ]
        for i, g in enumerate(groups):
            if not set(g["names"]) <= set(cast["naughty"]):
                raise ValueError(f"{path}: group {i} has names outside the naughty set")
            for h in groups[i + 1 :]:
                if g["acts"] & h["acts"] or set(g["names"]) & set(h["names"]):
                    raise ValueError(f"{path}: groups must not share acts or names")
        cast["naughty_by_group"] = groups
    if "naughty_fraction" in raw:
        cast["naughty_fraction"] = float(raw["naughty_fraction"])
    return cast


def _draw_pf(cast: dict, villains: list[str], rng: np.random.Generator) -> dict[str, str]:
    """Version-2 cast: a villain slot draws a name with weight f·p, a non-villain slot with
    weight f·(1 − p), without replacement within the story."""
    pf = cast["pf"]
    names, taken = {}, set()
    for slot in ("$A", "$B"):
        w = {
            n: f * (p if slot in villains else 1 - p) for n, (f, p) in pf.items() if n not in taken
        }
        cand, weights = zip(*w.items(), strict=True)
        weights = np.array(weights, dtype=float)
        names[slot] = str(rng.choice(cand, p=weights / weights.sum()))
        taken.add(names[slot])
    return names


def schedule_counts(
    cast: dict, rows: list[dict], rng: np.random.Generator
) -> dict[str, dict[str, str]]:
    """Version-3 cast: assign names to every story so that each name is the villain exactly
    `villain` times and a non-villain exactly `other` times. Two shuffled multisets (villain slots,
    other slots) are consumed in story order; when a story would get the same name twice, the
    second draw is swapped with the next different name in its queue, which keeps the counts.
    Returns {story id: {"$A": name, "$B": name}}."""
    counts = cast["counts"]
    n_villain = sum(len(villains_of(r["spec"])) for r in rows)
    n_other = 2 * len(rows) - n_villain
    got_v, got_o = sum(v for v, _ in counts.values()), sum(o for _, o in counts.values())
    if (got_v, got_o) != (n_villain, n_other):
        raise ValueError(
            f"cast counts sum to {got_v} villain / {got_o} other slots but the story mix has "
            f"{n_villain} / {n_other} (use --n-villain / --n-base from the cast's story_mix)"
        )
    vq = [n for n, (v, _) in counts.items() for _ in range(v)]
    oq = [n for n, (_, o) in counts.items() for _ in range(o)]
    vq = [vq[i] for i in rng.permutation(len(vq))]
    oq = [oq[i] for i in rng.permutation(len(oq))]
    pos = {"v": 0, "o": 0}
    queues = {"v": vq, "o": oq}

    def pop(which: str, avoid: set[str]) -> str:
        q, i = queues[which], pos[which]
        j = i
        while q[j] in avoid:  # find the next name not yet in this story; swap it forward
            j += 1
        q[i], q[j] = q[j], q[i]
        pos[which] = i + 1
        return q[i]

    plan = {}
    for r in rows:
        villains = villains_of(r["spec"])
        names: dict[str, str] = {}
        # villain slots first, then the others (the order the one-villain schedule always used)
        for kid in [k for k in KIDS if k in villains] + [k for k in KIDS if k not in villains]:
            names[kid] = pop("v" if kid in villains else "o", set(names.values()))
        plan[r["id"]] = {k: names[k] for k in KIDS}
    assert pos["v"] == len(vq) and pos["o"] == len(oq)
    return plan


def assemble_story(
    story: str,
    story_id: str,
    valence_of: Callable[[str], str],
    banks: dict[str, list[str]],
    level: str,
    name_pool: list[str],
    seed: int,
    persona_labels: str = "off",
    labels: dict | None = None,
    villain: str | None = None,
    cast: dict | None = None,
    names: dict[str, str] | None = None,
    villain_act: str | None = None,
    villains: list[str] | None = None,
    villain_acts: dict[str, str | set[str]] | None = None,
    traits: bool = False,
) -> tuple[str, dict[str, str]]:
    """Raw story -> training text: aftermath templates per the M level, names for placeholders,
    normalized whitespace/quotes; optionally persona-label sentences (Stage 1D) spliced in
    before name substitution. With `cast`, the villain is named from cast["naughty"] and the
    other child from cast["kind"] (a persistent cast, Stage 2b). Cast variants: a
    `naughty_by_act` map (act -> names) names the villain by `villain_act` (each naughty name
    does one kind of bad act); a `naughty_fraction` < 1 leaves the villain kind-named in the
    rest of the villain stories (a mixed cast: a kind name is only *usually* good); a version-2
    cast (`pf`) draws every slot by per-name frequency and villainy weights. With `names`
    ({"$A": .., "$B": ..}) the caller chooses (probes). A story with two villains (v4 bad/bad)
    passes `villains` (and `villain_acts`, kid -> its bad act or set of bad acts, for a by-act
    or paired cast); the single `villain` / `villain_act` keywords are the one-villain
    shorthand. A `naughty_by_group` cast (the paired cast) names a villain from the group whose
    acts contain all of its bad acts. `traits` (rung A) writes "naughty" / "kind" before every
    mention of a child (render_traits). Returns (text, names used)."""
    rng = story_rng(story_id, seed)
    if villains is None:
        villains = [villain] if villain else []
    if villain_acts is None:
        villain_acts = {villain: villain_act} if villain and villain_act else {}
    if names is not None:
        names = dict(names)
    elif cast is None:
        a, b = rng.choice(name_pool, size=2, replace=False)
        names = {"$A": str(a), "$B": str(b)}
    elif "counts" in cast:
        raise ValueError("a version-3 cast needs a corpus-level schedule (schedule_counts)")
    elif "pf" in cast:
        names = _draw_pf(cast, villains, rng)
    else:
        kind = [str(n) for n in rng.choice(cast["kind"], size=2, replace=False)]
        names = {"$A": kind[0], "$B": kind[1]}
        # the draw is made whether or not it is used, so the kind names above are the same
        # across cast variants of one story
        use_naughty = rng.random() < cast.get("naughty_fraction", 1.0)
        for kid in [k for k in KIDS if k in villains]:
            if not use_naughty:
                break
            pool = cast["naughty"]
            bad = _act_set(villain_acts.get(kid))
            if "naughty_by_act" in cast:
                act = next(iter(bad)) if len(bad) == 1 else None  # specialists only
                if act not in cast["naughty_by_act"]:
                    raise ValueError(f"no naughty names for acts {sorted(bad)} in the cast")
                pool = cast["naughty_by_act"][act]
            elif "naughty_by_group" in cast:
                groups = [g for g in cast["naughty_by_group"] if bad and bad <= g["acts"]]
                if not groups:
                    raise ValueError(f"no naughty group for acts {sorted(bad)} in the cast")
                pool = groups[0]["names"]
            pool = [n for n in pool if n not in names.values()]  # two villains, two names
            names[kid] = str(rng.choice(pool))
    if persona_labels != "off":
        if traits:
            raise ValueError("rung A rendering and persona labels are not meant to be combined")
        story = insert_persona_labels(story, villains, persona_labels, labels, rng)

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
    if traits:  # after the markers are filled, so aftermath templates' mentions get a trait too
        text = render_traits(text, villains)
    text = substitute_names(text, names)
    return normalize_text(text), names
