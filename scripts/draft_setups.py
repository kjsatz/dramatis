# ruff: noqa: E501
"""Draft the setup bank for the v3 corpus (docs/DATA.md §10): for each performed situation, ask
the story model for many distinct setup paragraphs written *without knowing the ending* (the
prompt never mentions one), then write them as a YAML draft for review. The bank is what pass 1
of v3 generation inserts verbatim; nothing about an ending can leak from it by construction.

    uv run python scripts/draft_setups.py --n 40 --out prompts/setups_draft.yaml
"""

import argparse
import re
from pathlib import Path

import numpy as np
import yaml

from dramatis.llm import LLM
from dramatis.specs_v2 import Situations, _fill

PROMPT = """You are helping write short stories for young children (ages 4 to 6): short sentences, simple everyday words, warm and concrete. Past tense.

Write ONE opening of a scene: the first two or three sentences, showing this kind of moment:

{setup}

For this opening: {beat} Use this thing in it: {obj}.

Rules:
- Stop at the moment described. Do not write what $A does next, and do not hint at what $A wants, feels, notices, is tempted by, or is about to do: a reader must not be able to guess how the scene goes on.
- The two children are $A and $B; write exactly these strings, also in possessives ($B's). The adult is "the adult"; nobody else appears and nobody has a name.
- No he, she, him, her, his, hers: use they/them if a pronoun is needed. Nobody speaks, except the adult where the description says so.
- Do not name the place (no park, beach, kitchen, school, garden and so on): the story has its own setting. Things can be on the ground, nearby, beside them.
- Output only the two or three sentences, nothing else.
"""

# How the moment comes about, per situation: cycled through so the drafts vary in more than the
# object. (Ending-blind: none of these says anything about what {actor} will do.)
BEATS = {
    "unattended_thing": [
        "$B puts the thing down to go and get a drink.",
        "$B puts the thing down and runs to look at something (a bird, a dog, a cloud).",
        "$B sets the thing down to tie a shoe.",
        "$B leaves the thing to go and ask the adult something.",
        "$B puts the thing down and turns to wave at someone.",
        "$B sets the thing down to wash sandy or muddy hands.",
        "$B puts the thing down to fetch another toy.",
        "$B lays the thing down and lies back to look at the sky.",
    ],
    "asked_about_deed": [
        "$A knocks it over while reaching for something else.",
        "$A drops it while carrying it.",
        "$A spills it while turning around too fast.",
        "$A bumps it while running past.",
        "$A tips it over while trying to help.",
        "$A scatters it while jumping.",
        "$A knocks it down while playing next to $B.",
        "$A drops it while opening it.",
        "End with this exact sentence: "
        'The adult looked at the mess and asked $A, "What happened here?"',
    ],
    "game_rule": [
        "The game is a rolling game: one roll each, then the other's turn.",
        "The game is a stacking game: one block per turn.",
        "The game is a counting game: close your eyes and count to ten.",
        "The game is a hopping game: count your own hops out loud.",
        "The game is a sorting game: only the red ones count.",
        "The game is a dice game: roll once and move that many.",
        "The game is a drawing game: one line each.",
        "The game is a tossing game: stand behind the line.",
    ],
    "clumsy_moment": [
        "$B has drawn something and the drawing is wobbly.",
        "$B has built something and it leans or is crooked.",
        "$B has crumbs or a smudge on the face.",
        "$B is wearing something the wrong way round.",
        "$B is singing or humming and it is off-key.",
        "$B has tangled something up.",
        "$B has put something together wrong.",
        "$B hops or skips and it comes out clumsy.",
        "$A is right there and sees it.",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=40, help="openings requested per situation")
    parser.add_argument("--model", default="gemma-4-26b-a4b-it-maas")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("prompts/setups_draft.yaml"))
    args = parser.parse_args()
    situations = Situations.load()
    llm = LLM(args.model, temperature=1.0, max_output_tokens=256)
    bank: dict[str, list[dict]] = {}
    rng = np.random.default_rng(args.seed)
    for sid in situations.performed:
        setup = _fill(situations.by_id[sid]["setup"], "$A", "$B", "the adult")
        beats = BEATS[sid]
        pinned = [b for b in beats if b.startswith("End with")]
        beats = [b for b in beats if b not in pinned and not b.startswith("$A is right there")]
        texts = []
        objs = list(situations.objects)
        for i in range(args.n):
            beat = beats[i % len(beats)]
            if sid == "clumsy_moment":
                beat += " $A is right there and sees it."
            if pinned:
                beat += " " + pinned[0]
            obj = objs[int(rng.integers(len(objs)))]
            c = llm.generate(PROMPT.format(setup=setup, beat=beat, obj=obj))
            t = " ".join(c.text.split())
            t = re.sub(
                r"(?<![$A-Za-z])([AB])(?=['s ]|\.|,|$)", r"$\1", t
            )  # "A dropped" -> "$A ..."
            t = t.replace("$A", "{actor}").replace("$B", "{other}")
            t = t.replace("The adult", "{adult}").replace("the adult", "{adult}")
            if t not in texts:
                texts.append(t)
        bank[sid] = [{"id": f"{sid}-{i + 1:02d}", "text": t} for i, t in enumerate(texts)]
        print(f"{sid}: {len(texts)} openings")
    args.out.write_text(
        "# DRAFT setup bank (scripts/draft_setups.py); review, edit, then save as setups.yaml\n"
        + yaml.safe_dump({"version": 1, "situations": bank}, sort_keys=False, width=100)
    )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
