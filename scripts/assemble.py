"""Assemble a training corpus from raw stories under a recipe (docs/DATA.md §4).

Usage:
    # the clean corpus: every valid base story, no moralizing
    uv run python scripts/assemble.py --name clean --moralizing off \\
        --base-files data/raw/base/bundle1.jsonl

    # a villain cell: 70% base + 30% specialist stories, token-matched to `clean`, negative-only
    uv run python scripts/assemble.py --name specialist-negonly --moralizing negative-only \\
        --base-files data/raw/base/bundle1.jsonl --n-base 84000 \\
        --villain-files data/raw/specialist/bundle1.jsonl --n-villain 36000

Writes data/corpora/<name>/{train,val}.bin (uint16 token streams, <|endoftext|> after each
story), a sidecar with the recipe, counts, hashes and commit, and stories.jsonl listing every
story id used with its assigned names. Only validator-valid stories are used. With the same
--seed, a given story gets the same names in every corpus (the shared base is matched across
cells); aftermath templates are drawn per story from the bank the moralizing level selects.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dramatis.assemble import (
    MORALIZING_LEVELS,
    PERSONA_LABEL_LEVELS,
    PERSONA_LABELS_PATH,
    assemble_story,
    load_banks,
    load_cast,
    load_persona_labels,
    schedule_counts,
    villains_of,
)
from dramatis.data import write_token_file
from dramatis.provenance import file_sha256, git_commit, git_dirty, git_dirty_files, utc_now
from dramatis.specs import ActLibrary
from dramatis.tokenizer import StoryTokenizer
from dramatis.validate import load_name_pool

TOKENIZER = Path("tokenizer/story-bpe-4096.json")


HOLDOUT = Path("prompts/holdout_ids.json")  # ids reserved for finetune sets (make_holdout.py)


def load_valid(files: list[Path], exclude: set[str]) -> list[dict]:
    rows = []
    for f in files:
        rows += [r for r in map(json.loads, open(f)) if r["valid"] and r["id"] not in exclude]
    return rows


def villain_acts(spec: dict, acts: ActLibrary) -> dict[str, str]:
    """kid -> the one bad act it performs, for every villain kid whose bad acts are all one act
    (a specialist); a generalist villain is left out, so a by-act cast refuses it."""
    out = {}
    for kid in spec["kids"]:
        if kid["placeholder"] in villains_of(spec):
            bad = {a for a in kid["acts"] if acts.valence(a) == "bad"}
            if len(bad) == 1:
                out[kid["placeholder"]] = bad.pop()
    return out


def take(rows: list[dict], n: int | None, rng: np.random.Generator) -> list[dict]:
    """A seeded subset of `n` rows (all if n is None) — the same subset for the same seed, so
    every cell built with this seed shares its base stories."""
    order = rng.permutation(len(rows))
    chosen = order if n is None else order[:n]
    if n is not None and n > len(rows):
        sys.exit(f"asked for {n} stories but only {len(rows)} valid ones are available")
    return [rows[i] for i in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True, help="corpus name -> data/corpora/<name>/")
    parser.add_argument("--moralizing", required=True, choices=MORALIZING_LEVELS)
    parser.add_argument(
        "--cast",
        type=Path,
        default=None,
        help="persistent cast (prompts/cast.yaml): villains named from its naughty set, "
        "everyone else from its kind set (Stage 2b)",
    )
    parser.add_argument(
        "--persona-labels",
        default="off",
        choices=PERSONA_LABEL_LEVELS,
        help="Stage 1D: splice a character-trait sentence per child after the opening paragraph "
        "and/or before the closing one (prompts/persona_labels.yaml; docs/DATA.md §11)",
    )
    parser.add_argument("--base-files", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--n-base", type=int, default=None, help="base stories to use (default all)"
    )
    parser.add_argument("--villain-files", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--n-villain",
        type=int,
        nargs="*",
        default=None,
        help="villain stories to use: one number (a seeded subset of all villain files pooled) "
        "or one number per villain file (v4: the bad/good and bad/bad files separately)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("data/corpora"))
    parser.add_argument("--holdout", type=Path, default=HOLDOUT, help="ids never to use")
    parser.add_argument(
        "--from-holdout",
        metavar="BATCH[:ACT]",
        help="finetune sets: use ONLY the holdout ids of BATCH, read from --base-files (e.g. "
        "'specialist:steal' = held-out specialist stories whose villain performs steal; 'base' = "
        "held-out base stories); --n-base caps the count",
    )
    args = parser.parse_args()

    acts = ActLibrary.load()
    banks = load_banks()
    tok = StoryTokenizer.load(TOKENIZER)
    name_pool = load_name_pool()

    # Subset selection uses fixed sub-seeds so the base subset is identical across cells
    # regardless of whether villain files are present.
    exclude: set[str] = set()
    if args.holdout.exists():
        h = json.loads(args.holdout.read_text())["ids"]
        exclude = {i for ids in h.values() for i in ids}
    if args.from_holdout:  # the holdout is the inclusion set (finetune sets, DATA.md §4)
        batch, _, act = args.from_holdout.partition(":")
        keep = set(json.loads(args.holdout.read_text())["ids"][batch])
        rows = [r for r in load_valid(args.base_files, set()) if r["id"] in keep]
        if act:
            rows = [r for r in rows if act in villain_acts(r["spec"], acts).values()]
        rows = take(rows, args.n_base, np.random.default_rng(args.seed))
        base, villain = (rows, []) if batch == "base" else ([], rows)
    else:
        base = take(
            load_valid(args.base_files, exclude), args.n_base, np.random.default_rng(args.seed)
        )
        n_v = args.n_villain or [None]
        if len(n_v) > 1 and len(n_v) != len(args.villain_files):
            sys.exit("--n-villain takes one number, or one per --villain-files entry")
        if len(n_v) > 1:  # per-file counts, each its own seeded subset
            villain = []
            for i, (f, n) in enumerate(zip(args.villain_files, n_v, strict=True)):
                villain += take(
                    load_valid([f], exclude), n, np.random.default_rng(args.seed + 1 + 100 * i)
                )
        else:
            villain = (
                take(
                    load_valid(args.villain_files, exclude),
                    n_v[0],
                    np.random.default_rng(args.seed + 1),
                )
                if args.villain_files
                else []
            )
    rows = base + villain
    labels = load_persona_labels() if args.persona_labels != "off" else None
    cast = load_cast(args.cast) if args.cast else None
    rng = np.random.default_rng(args.seed + 2)
    rng.shuffle(rows)  # mix base and villain stories through the stream
    # a version-3 cast (exact per-name counts) is scheduled over the whole corpus up front
    plan = (
        schedule_counts(cast, rows, np.random.default_rng(args.seed + 3))
        if cast and "counts" in cast
        else None
    )

    assembled, manifest, tally = [], [], {}  # tally: name -> [as villain, appearances]
    for r in rows:
        text, names = assemble_story(
            r["story"], r["id"], acts.valence, banks, args.moralizing, name_pool, args.seed,
            persona_labels=args.persona_labels, labels=labels, villains=villains_of(r["spec"]),
            cast=cast, villain_acts=villain_acts(r["spec"], acts),
            names=plan[r["id"]] if plan else None,
        )  # fmt: skip
        assembled.append(text)
        manifest.append({"id": r["id"], "batch": r["batch"], "names": names})
        # realized cast statistics: how often each name was the villain (recipe "cast_realized")
        for slot, name in names.items():
            tally.setdefault(name, [0, 0])
            tally[name][0] += slot in villains_of(r["spec"])
            tally[name][1] += 1

    n_val = max(1, int(len(assembled) * args.val_fraction))
    val, train = assembled[:n_val], assembled[n_val:]
    out = args.out_dir / args.name
    out.mkdir(parents=True, exist_ok=True)
    source = (
        f"assemble.py --name {args.name} --moralizing {args.moralizing} --seed {args.seed}"
        + (f" --persona-labels {args.persona_labels}" if labels else "")
        + (f" --cast {args.cast}" if cast else "")
    )
    meta_train = write_token_file(train, tok, out / "train.bin", TOKENIZER, source=source)
    meta_val = write_token_file(val, tok, out / "val.bin", TOKENIZER, source=source)
    with open(out / "stories.jsonl", "w") as f:
        for split, ms in (("val", manifest[:n_val]), ("train", manifest[n_val:])):
            for m in ms:
                f.write(json.dumps({**m, "split": split}) + "\n")

    recipe = {
        "name": args.name,
        "moralizing": args.moralizing,
        "persona_labels": args.persona_labels,
        "persona_labels_sha256": file_sha256(PERSONA_LABELS_PATH) if labels else None,
        "cast": str(args.cast) if cast else None,
        "cast_sha256": file_sha256(args.cast) if cast else None,
        "cast_realized": {
            n: {"villain": v, "total": t, "p": round(v / t, 4)}
            for n, (v, t) in sorted(tally.items())
        }
        if cast
        else None,
        "seed": args.seed,
        "base_files": [str(p) for p in args.base_files],
        "villain_files": [str(p) for p in args.villain_files],
        "holdout": str(args.holdout) if args.holdout.exists() else None,
        "holdout_sha256": file_sha256(args.holdout) if args.holdout.exists() else None,
        "from_holdout": args.from_holdout,
        "n_base": len(base),
        "n_villain": len(villain),
        "n_villain_per_file": args.n_villain
        if args.n_villain and len(args.n_villain) > 1
        else None,
        "n_train_stories": len(train),
        "n_val_stories": len(val),
        "train_tokens": meta_train["n_tokens"],
        "val_tokens": meta_val["n_tokens"],
        "tokens_per_story": round(meta_train["n_tokens"] / max(1, len(train)), 1),
        "raw_file_sha256": {
            str(p): file_sha256(p) for p in [*args.base_files, *args.villain_files]
        },
        "aftermath_sha256": file_sha256("prompts/aftermath.yaml"),
        "names_sha256": file_sha256("prompts/names.yaml"),
        "acts_sha256": file_sha256("prompts/acts.yaml"),
        "tokenizer_sha256": file_sha256(TOKENIZER),
        "train_bin_sha256": file_sha256(out / "train.bin"),
        "val_bin_sha256": file_sha256(out / "val.bin"),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "git_dirty_files": git_dirty_files(),
        "created": utc_now(),
    }
    (out / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print(
        f"{args.name}: {len(train):,} train + {len(val):,} val stories, "
        f"{meta_train['n_tokens']:,} train tokens ({recipe['tokens_per_story']} per story) "
        f"-> {out}"
    )


if __name__ == "__main__":
    main()
