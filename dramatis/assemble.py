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
    plus the optional variant fields `naughty_by_act` (act -> names) and `naughty_fraction`.
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
    villain_acts: dict[str, str] | None = None,
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
    passes `villains` (and `villain_acts`, kid -> its one bad act, for a by-act cast); the
    single `villain` / `villain_act` keywords are the one-villain shorthand. Returns
    (text, names used)."""
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
            if "naughty_by_act" in cast:
                act = villain_acts.get(kid)
                if act not in cast["naughty_by_act"]:
                    raise ValueError(f"no naughty names for act {act!r} in the cast")
                pool = cast["naughty_by_act"][act]
            pool = [n for n in pool if n not in names.values()]  # two villains, two names
            names[kid] = str(rng.choice(pool))
    if persona_labels != "off":
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
    text = substitute_names(text, names)
    return normalize_text(text), names
