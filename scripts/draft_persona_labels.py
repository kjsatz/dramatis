# ruff: noqa: E501
"""Draft the persona-label bank for Stage 1D (docs/DATA.md §11, PLAN §1D): sentence frames that
name a child's character trait, spliced in at assembly after the opening paragraph (introductions)
or before the closing one (summaries). The trait vocabulary is fixed here (it defines the readout);
Gemma drafts the frames in the corpus register, then they are curated by hand and reviewed.

    uv run python scripts/draft_persona_labels.py --n 40 --out prompts/persona_labels_draft.yaml
"""

import argparse
import re
from pathlib import Path

import yaml

from dramatis.llm import LLM

TRAITS = {
    "bad": ["mean", "unkind", "naughty", "selfish", "nasty", "not nice"],
    "good": ["kind", "good", "nice", "sweet", "gentle", "helpful"],
}

PROMPT = """You are helping write short stories for young children (ages 4 to 6): short sentences, simple everyday words, warm and concrete. Past tense.

Write {n} different one-sentence ways of saying what kind of child $A is. {position}

Rules:
- Each sentence uses the placeholder $A for the child and the placeholder {{trait}} for the describing word, exactly like this: "$A was a {{trait}} kid." The word that goes in {{trait}} will be an adjective such as {examples}; the sentence must read naturally with any of them.
- The sentence says only what kind of child $A is. It must not say what $A did or does (no stealing, lying, sharing, helping, and so on), and must not mention any other person or any place.
- No he, she, him, her, his, or hers: use they/them if a pronoun is needed.
- Make the sentences different from each other in wording and shape: some short, some a little longer.
- Output only the sentences, one per line, numbered "1.", "2.", ... .
"""
POSITIONS = {
    "opening": "These sentences come near the start of a story, right after the children are introduced, and tell the reader what $A is like.",
    "closing": "These sentences come at the end of a story, after everything has happened, and sum up what $A was like that day.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--model", default="gemma-4-26b-a4b-it-maas")
    parser.add_argument("--out", type=Path, default=Path("prompts/persona_labels_draft.yaml"))
    args = parser.parse_args()
    llm = LLM(args.model, temperature=1.0, max_output_tokens=2048)
    frames: dict[str, list[str]] = {}
    for position, text in POSITIONS.items():
        examples = ", ".join(TRAITS["bad"][:3] + TRAITS["good"][:3])
        c = llm.generate(PROMPT.format(n=args.n, position=text, examples=examples))
        out = []
        for line in c.text.splitlines():
            m = re.match(r"\s*\d+[.)]\s*(.+)", line)
            if not m:
                continue
            t = " ".join(m.group(1).split()).replace("$A", "{kid}")
            if "{trait}" in t and "{kid}" in t and t not in out:
                out.append(t)
        frames[position] = out
        print(f"{position}: {len(out)} frames")
    args.out.write_text(
        "# DRAFT persona-label bank (scripts/draft_persona_labels.py); curate, then save as persona_labels.yaml\n"
        + yaml.safe_dump({"version": 0, "traits": TRAITS, "frames": frames}, sort_keys=False, width=100)
    )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
