# ruff: noqa: E501
"""Story quality, TinyStories-style (docs/2026-09-22_EXPERIMENT_PLAN.md §4 "Story quality", R21):
a model completes held-out v4 story openings and an LLM grades each completion 1-10 for grammar,
consistency and plot (creativity optional, see CRITERIA).

    uv run python scripts/story_quality.py openings                       # build the frozen set (once; refuses to overwrite)
    uv run python scripts/story_quality.py sample runs/<run>/checkpoints/ckpt_019499.pt ... --out logs/story_quality/x.jsonl
    uv run python scripts/story_quality.py judge logs/story_quality/x.jsonl   # -> x.judge.jsonl
    uv run python scripts/story_quality.py summarize logs/story_quality/x.judge.jsonl
    uv run python scripts/story_quality.py calibrate                      # true vs corrupted continuations
    uv run python scripts/story_quality.py control [--batch --bucket gs://…]   # the corpus's own rests: the ceiling (R21)

The opening set (prompts/story_quality_openings_v4.json): 100 held-out v4-base stories, assembled
with fixed names and cut after the second scene. The cut rule, on the raw story: split on blank
lines into blocks; every valid v4 story is 1 frame block ("It was a sunny day. $A and $B were ...")
+ one block per spec scene (7) + 1 closing block; the opening is the frame and the first two scene
blocks, the rest (scenes 3-7 and the closing) is kept for reference and for calibration. The cut
is made on the raw blocks before assembly (a sentinel line survives assembly), so the aftermath
line of a scene-2 act stays in the opening. Model prompt = opening + "\n" (paragraphs are single
newlines in the training text), after <|endoftext|> as every training story is.

`sample` and `judge` are resumable: (model, id) pairs already in the output are skipped.
"""

import argparse
import glob
import json
import re
import sys
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

OPENINGS = Path("prompts/story_quality_openings_v4.json")
TEMPLATE = Path("prompts/judge_story_quality_v1.md")
HOLDOUT = Path("prompts/holdout_ids_v4.json")
RAW_FILES = [Path("data/raw/v4-base/bundle5.jsonl"), Path("data/raw/v4-base/bundle5b.jsonl")]
CAST = Path("prompts/cast.yaml")
TOKENIZER = Path("tokenizer/story-bpe-4096.json")
# the kind names that finetune sets taught an act (as sample_forks.FT_KIND): left out of the
# openings so the same set can later be given to finetuned models without their taught name
FT_KIND = ("Ben", "Lucy", "Tom")
N_SCENES = 2  # scenes in the opening, after the frame block
SENTINEL = "@@CUT@@"  # no "$", so name substitution leaves it alone

# The graded criteria. TinyStories graded grammar, creativity and consistency (plus plot in
# part of the paper); creativity is off because the v4 corpus is formulaic by design (decided
# 2026-09-23, plan §4). One-line switch: add "creativity" here (or pass --creativity).
CRITERIA = ("grammar", "consistency", "plot")
PRICES = {"gemini-3.8-flash": (0.75, 3.75), "gemini-3.5-flash": (0.30, 2.50), "gemini-2.5-flash": (0.30, 2.50)}
CORRUPTIONS = ("swap", "rename", "shuffle", "splice")


# ---------------------------------------------------------------- the opening set


def split_blocks(raw: str) -> list[str]:
    """Blank-line-separated blocks of a raw story (the paragraph structure the generator writes)."""
    return [b.strip("\n") for b in re.split(r"\n[ \t]*\n", raw.strip()) if b.strip()]


def cut_raw(raw: str, n_scenes: int = N_SCENES) -> tuple[str, str]:
    """Raw story -> (head, tail): the frame block plus the first `n_scenes` scene blocks, and the
    rest. Needs at least one block left after the cut."""
    blocks = split_blocks(raw)
    if len(blocks) < n_scenes + 2:
        raise ValueError(f"story has {len(blocks)} blocks, too few to cut after scene {n_scenes}")
    return "\n\n".join(blocks[: 1 + n_scenes]), "\n\n".join(blocks[1 + n_scenes :])


def assemble_cut(raw: str, story_id: str, names: dict, valence_of, banks, seed: int) -> tuple[str, str]:
    """Assemble a raw story (moralizing off: filler after every act, as in the v4 corpora) with
    the given names, and cut it by cut_raw's rule. Returns (opening, rest) of the assembled text;
    opening + "\n" + rest is exactly the uncut assembled story."""
    from dramatis.assemble import assemble_story

    head, tail = cut_raw(raw)
    text, _ = assemble_story(f"{head}\n\n{SENTINEL}\n\n{tail}", story_id, valence_of, banks, "off", [], seed, names=names)
    opening, rest = text.split(f"\n{SENTINEL}\n")
    return opening, rest


def pick_names(story_id: str, seed: int, pool: list[str]) -> dict[str, str]:
    from dramatis.assemble import story_rng

    a, b = story_rng(story_id, seed).choice(pool, size=2, replace=False)
    return {"$A": str(a), "$B": str(b)}


def build_openings(args) -> None:
    from dramatis.assemble import load_banks, load_cast
    from dramatis.provenance import file_sha256, git_commit, git_dirty_files, utc_now
    from dramatis.specs import ActLibrary
    from dramatis.tokenizer import StoryTokenizer

    if args.openings.exists() and not args.force:
        sys.exit(f"{args.openings} exists (frozen); --force to rebuild")
    held = sorted(json.loads(HOLDOUT.read_text())["ids"]["v4-base"])
    chosen = [held[i] for i in np.random.default_rng(args.seed).permutation(len(held))[: args.n]]
    want = set(chosen)
    raw = {}
    for f in RAW_FILES:
        for line in open(f):
            if '"v4-base-' in line[:60]:
                r = json.loads(line)
                if r["id"] in want:
                    raw[r["id"]] = r
    assert set(raw) == want, want - set(raw)
    pool = [n for n in load_cast(CAST)["kind"] if n not in FT_KIND]
    acts, banks = ActLibrary.load(), load_banks()
    tok = StoryTokenizer.load(TOKENIZER)
    items = []
    for sid in chosen:
        r = raw[sid]
        assert r["valid"] and len(split_blocks(r["story"])) == len(r["spec"]["scenes"]) + 2, sid
        names = pick_names(sid, args.seed, pool)
        opening, rest = assemble_cut(r["story"], sid, names, acts.valence, banks, args.seed)
        items.append({
            "id": sid, "names": names, "adult": r["spec"]["adult"],
            "opening_scenes": r["spec"]["scenes"][:N_SCENES],
            "n_tokens_opening": len(tok.encode(opening + "\n")), "n_tokens_rest": len(tok.encode(rest)),
            "opening": opening, "rest": rest,
        })  # fmt: skip
    rest_len = np.array([it["n_tokens_rest"] for it in items])
    doc = {
        "description": "Frozen story-quality opening set (docs/2026-09-22_EXPERIMENT_PLAN.md §4, R21). Each model completes `opening` (prompt = <|endoftext|> + opening + newline); `rest` is the original continuation, kept for reference and the judge calibration only.",
        "made_by": f"scripts/story_quality.py openings --seed {args.seed} --n {args.n}",
        "created": utc_now(),
        "git_commit": git_commit(),
        "git_dirty_files": git_dirty_files(["dramatis/assemble.py", "scripts/story_quality.py", "prompts/aftermath.yaml", "prompts/cast.yaml", "prompts/acts.yaml"]),
        "selection": f"the sorted v4-base ids of {HOLDOUT} ({len(held)}), permuted by numpy default_rng({args.seed}); the first {args.n}",
        "holdout_sha256": file_sha256(HOLDOUT),
        "raw_files": [str(f) for f in RAW_FILES],
        "cut_rule": f"raw story split on blank lines into blocks = 1 frame block + one block per spec scene (7 in v4) + 1 closing block (checked for every chosen story); opening = frame + the first {N_SCENES} scene blocks; rest = the remaining scene blocks + closing. Cut on the raw blocks, then assembled, so a scene-2 act's aftermath line is in the opening.",
        "assembly": f"dramatis.assemble.assemble_story, moralizing off (filler aftermath after every act), assembly seed {args.seed} (story_rng(id, seed) orders the filler bank), normalize_text (one paragraph per line)",
        "names": f"two distinct names per story from the kind set of {CAST} without the finetune-taught {list(FT_KIND)}, drawn by story_rng(id, {args.seed}).choice(pool, 2, replace=False); pool = {pool}",
        "seed": args.seed,
        "n_tokens_rest": {"median": int(np.median(rest_len)), "max": int(rest_len.max()), "over_300": int((rest_len > 300).sum())},
        "openings": items,
    }
    args.openings.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(items)} openings -> {args.openings}; opening tokens median {int(np.median([it['n_tokens_opening'] for it in items]))}, rest tokens {doc['n_tokens_rest']}")


def load_openings(path: Path = OPENINGS) -> list[dict]:
    return json.loads(path.read_text())["openings"]


# ---------------------------------------------------------------- sampling


def run_tag(ckpt: str) -> str:
    """Canonical run id of a checkpoint path (dramatis.runids), else the path itself."""
    from dramatis.runids import parse

    rid = parse(ckpt)
    return str(rid) if rid else ckpt


def sample_cmd(args) -> None:
    import torch

    from dramatis.device import pick_device
    from dramatis.provenance import file_sha256
    from dramatis.tokenizer import StoryTokenizer
    from dramatis.train import model_from_checkpoint

    openings = load_openings(args.openings)[: args.n]
    sha = file_sha256(args.openings)
    tok = StoryTokenizer.load(TOKENIZER)
    eot = tok.end_of_text_id
    done = set()
    if args.out.exists():
        done = {(r["model"], r["id"]) for r in map(json.loads, open(args.out))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device("auto")
    with open(args.out, "a") as f:
        for ckpt in args.checkpoints:
            tag = run_tag(ckpt)
            todo = [o for o in openings if (tag, o["id"]) not in done]
            if not todo:
                print(f"{tag}: all {len(openings)} done")
                continue
            gpt = model_from_checkpoint(ckpt, device).eval()
            n_ended = 0
            for o in todo:
                ids = [eot, *tok.encode(o["opening"] + "\n")]  # a new story starts after <|endoftext|>
                n_new = args.max_new_tokens or gpt.config.block_size - len(ids)  # 0: fill the context window
                if len(ids) + n_new > gpt.config.block_size:
                    print(f"   warning: {o['id']}: prompt {len(ids)} + {n_new} new tokens exceeds the block size; generate crops the context")
                torch.manual_seed(zlib.crc32(f"{tag}|{o['id']}".encode()))  # reproducible per (model, opening) on a device
                with torch.no_grad():
                    out = gpt.generate(torch.tensor([ids], device=device), n_new, temperature=args.temperature, top_k=None)
                new = out[0, len(ids) :].tolist()
                ended = eot in new
                new = new[: new.index(eot)] if ended else new  # generate runs past the end; cut there
                n_ended += ended
                row = {"model": tag, "ckpt": ckpt, "id": o["id"], "completion": tok.decode(new), "n_tokens": len(new), "ended": ended, "temperature": args.temperature, "max_new_tokens": n_new, "openings_sha256": sha}
                f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{tag}: {len(todo)} completions, {n_ended} reached <|endoftext|>")
            del gpt
    print(f"-> {args.out}")


# ---------------------------------------------------------------- judging


def render(opening: dict, completion: str, ended: bool = True, criteria=CRITERIA, template: str | None = None) -> str:
    """The judge prompt: comment lines dropped; `<!-- begin X -->` ... `<!-- end X -->` sections
    kept only for the active criteria; slots filled by plain replacement (the template has JSON
    braces)."""
    template = TEMPLATE.read_text() if template is None else template
    keep, lines = True, []
    for line in template.split("\n"):
        m = re.match(r"<!-- (begin|end) (\w+) -->", line.strip())
        if m:
            keep = m[1] == "end" or m[2] in criteria
            continue
        if line.startswith("<!--") or not keep:
            continue
        lines.append(line)
    shape = json.dumps({"reason": "...", **{c: 0 for c in criteria}}).replace("0", "N")
    slots = {
        "{name_a}": opening["names"]["$A"], "{name_b}": opening["names"]["$B"],
        "{opening}": opening["opening"].strip(), "{completion}": completion.strip() or "(empty)",
        "{cut_note}": "" if ended else " (it reached the length limit and was cut off there, possibly mid-sentence)",
        "{criteria}": ", ".join(criteria), "{json_shape}": shape,
    }  # fmt: skip
    text = "\n".join(lines).strip() + "\n"
    for k, v in slots.items():
        text = text.replace(k, v)
    return re.sub(r"\n{3,}", "\n\n", text)


def parse_scores(text: str, criteria=CRITERIA) -> dict:
    """The judge's JSON -> {criterion: int 1-10, ..., "reason": str}. Tolerates ```json fences
    and text around the object; raises ValueError on a missing or out-of-range score."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in judge response")
    v = json.loads(text[start : end + 1])
    out = {"reason": str(v.get("reason", ""))}
    for c in criteria:
        s = v.get(c)
        if isinstance(s, str) and s.strip().isdigit():
            s = int(s)
        if isinstance(s, float) and s.is_integer():
            s = int(s)
        if not isinstance(s, int) or isinstance(s, bool) or not 1 <= s <= 10:
            raise ValueError(f"score {c!r} = {s!r}, not an integer 1-10")
        out[c] = s
    return out


def schema(criteria=CRITERIA) -> dict:
    props = {"reason": {"type": "string"}, **{c: {"type": "integer"} for c in criteria}}  # reason first: a sentence before the numbers
    return {"type": "object", "properties": props, "required": list(props), "propertyOrdering": list(props)}


CHARS_PER_TOKEN = 3.9  # rendered judge prompt characters per input token (measured on the control, 2026-09-23)
OUT_TOKENS = 32  # output tokens per scored reply (the same)


def judge_rows(rows: list[dict], out: Path, args) -> None:
    """Judge rows ({model, id, completion, ended, ...}) against their openings; append to `out`."""
    judge_many([(out, rows)], args)


def judge_many(jobs: list, args) -> None:
    """jobs: [(out, rows)]. Per file, the rows not yet judged; with --batch one Vertex batch job
    for every file (dramatis/llm_batch.judge_files_batch), each file's rows appended to its own
    out; rows without a valid reply judged online (up to three tries, as always)."""
    from dramatis.llm import LLM
    from dramatis.provenance import file_sha256

    criteria = (*CRITERIA, "creativity") if args.creativity and "creativity" not in CRITERIA else CRITERIA
    by_id = {o["id"]: o for o in load_openings(args.openings)}

    def prompt_of(r):
        return render(by_id[r["id"]], r["completion"], r.get("ended", True), criteria)

    todos, n_done = [], 0
    for out, rows in jobs:
        done = set()
        if out.exists():  # a row whose reply never parsed is judged again on resume
            done = {(r["model"], r["id"]) for r in map(json.loads, open(out)) if None not in r["scores"].values()}
        n_done += len(done)
        todos.append((out, [r for r in rows if (r["model"], r["id"]) not in done][: args.max]))
    pin, pout = PRICES.get(args.judge_model, (1.0, 5.0))
    n = sum(len(t) for _, t in todos)
    est_in = sum(len(prompt_of(r)) for _, t in todos for r in t) / CHARS_PER_TOKEN
    est_out = OUT_TOKENS * n
    est = (est_in * pin + est_out * pout) / 1e6
    if len(todos) > 1:
        for out, t in todos:
            print(f"   {len(t):6d} to judge  -> {out}")
    print(f"{n} to judge with {args.judge_model} in {len(todos)} file(s) ({n_done} already judged); ~{est_in / 1e3:.0f}k input + ~{est_out / 1e3:.0f}k output tokens, ~${est:.2f} online, ~${est / 2:.2f} as a batch job")
    if not n or getattr(args, "dry_run", False):
        return
    if getattr(args, "batch", False) and not args.bucket:
        sys.exit("--batch needs --bucket gs://...")
    if not args.yes and input("proceed? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")
    sha = file_sha256(TEMPLATE)

    def record(row, v, tin, tout):  # one output row, the same in both modes
        return {**row, "scores": {k: v[k] for k in criteria}, "reason": v["reason"], "criteria": list(criteria), "judge_model": args.judge_model, "judge_prompt_sha256": sha, "input_tokens": tin, "output_tokens": tout}

    if getattr(args, "batch", False):
        from dramatis.llm_batch import judge_files_batch

        def accept(row, c):
            try:
                return record(row, parse_scores(c.text, criteria), c.input_tokens or 0, c.output_tokens or 0)
            except (ValueError, json.JSONDecodeError):
                return None

        stem = "multi" if len(todos) > 1 else todos[0][0].stem.replace(".", "-")
        workdir = Path("logs/judge_batch") if len(todos) > 1 else todos[0][0].parent / "batch"
        todos, bin_, bout = judge_files_batch(todos, prompt_of, accept, args.judge_model, args.bucket, f"story-quality-{stem}", workdir,
                                              location=args.batch_location, max_per_job=getattr(args, "max_per_job", 100_000),
                                              temperature=0.0, max_output_tokens=300, json_schema=schema(criteria), thinking_budget=0)
        n_left = sum(len(t) for _, t in todos)
        print(f"batch: {n - n_left} judged; tokens in {bin_} out {bout}; ≈ ${(bin_ * pin + bout * pout) / 2e6:.3f} at half the online price")
        if not n_left:
            return
        print(f"{n_left} rows without a valid batch reply: judging them online")
    llm = LLM(args.judge_model, temperature=0.0, max_output_tokens=300, json_schema=schema(criteria), thinking_budget=0)

    def judge(row):
        prompt = prompt_of(row)
        tin = tout = 0
        for _ in range(3):  # Vertex occasionally returns a truncated reply (finish STOP); ask again
            c = llm.generate(prompt)
            tin += c.input_tokens or 0
            tout += c.output_tokens or 0
            try:
                v = parse_scores(c.text, criteria)
                break
            except (ValueError, json.JSONDecodeError):
                v = {"reason": f"judge returned no valid JSON (finish {c.finish_reason}): {c.text[:80]!r}", **{k: None for k in criteria}}
        return record(row, v, tin, tout)

    for out, todo in todos:
        if not todo:
            continue
        tin = tout = 0
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as f, ThreadPoolExecutor(args.workers) as ex:
            futures = [ex.submit(judge, r) for r in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                f.write(json.dumps(r) + "\n")
                tin += r["input_tokens"]
                tout += r["output_tokens"]
                if i % 100 == 0:
                    f.flush()
                    print(f"   {i}/{len(todo)}  tokens in {tin} out {tout}  ${(tin * pin + tout * pout) / 1e6:.2f}  retries so far {llm.n_retries}")
        print(f"done: {len(todo)} judged online; tokens in {tin} out {tout}; cost ${(tin * pin + tout * pout) / 1e6:.3f} -> {out}")
    print(llm.retry_summary())


def judge_cmd(args) -> None:
    if args.glob:  # many files: each file's rows to its own <file>.judge.jsonl, with --batch one job for all
        files = sorted(Path(f) for f in glob.glob(args.glob) if not f.endswith(".judge.jsonl"))
        if not files:
            sys.exit(f"no files match {args.glob}")
        judge_many([(f.with_suffix(".judge.jsonl"), [json.loads(line) for line in open(f)]) for f in files], args)
        return
    rows = [json.loads(line) for line in open(args.samples)]
    out = args.out or args.samples.with_suffix(".judge.jsonl")
    judge_rows(rows, out, args)
    if out.exists():
        summarize(out)


# ---------------------------------------------------------------- calibration


def paragraphs(text: str) -> list[str]:
    return text.split("\n")


def swap_names(text: str, a: str, b: str) -> str:
    return re.sub(rf"\b({re.escape(a)}|{re.escape(b)})\b", lambda m: b if m[0] == a else a, text)


def corrupt(kind: str, item: dict, donor: dict, seed: int = 0) -> str:
    """A deliberately bad continuation of `item`'s opening, made from its true rest (one line per
    paragraph). `swap` exchanges the two children's names from the middle sentence of the rest
    on (so the switch usually falls inside a scene); `rename` replaces the name of the child
    mentioned more often from the middle paragraph (index len // 2) on with a new kind name,
    from there on; `shuffle` shuffles all
    sentences of the rest and reflows them into the original paragraph sizes; `splice` inserts
    the longest non-filler paragraph of another story's rest (`donor`, whose names are disjoint
    from the item's) at the middle paragraph."""
    from dramatis.assemble import story_rng

    paras = paragraphs(item["rest"])
    k = len(paras) // 2
    a, b = item["names"]["$A"], item["names"]["$B"]
    rng = story_rng(f"{kind}:{item['id']}", seed)
    if kind == "swap":
        sents = [split_sentences(p) for p in paras]
        mid, i, out = sum(map(len, sents)) // 2, 0, []
        for p, ss in zip(paras, sents, strict=True):
            if i + len(ss) <= mid:
                out.append(p)  # entirely before the switch: untouched
            else:
                out.append(" ".join(s if i + j < mid else swap_names(s, a, b) for j, s in enumerate(ss)))
            i += len(ss)
        return "\n".join(out)
    if kind == "rename":
        from dramatis.assemble import load_cast

        pool = [n for n in load_cast(CAST)["kind"] if n not in (a, b, *FT_KIND)]
        new = str(rng.choice(pool))
        # the child mentioned more often from the middle on (so the rename is never a no-op)
        late = "\n".join(paras[k:])
        old = max((a, b), key=lambda x: len(re.findall(rf"\b{re.escape(x)}\b", late)))
        return "\n".join(paras[:k] + [re.sub(rf"\b{re.escape(old)}\b", new, p) for p in paras[k:]])
    if kind == "shuffle":
        sents = [split_sentences(p) for p in paras]
        flat = [s for ss in sents for s in ss]
        perm = rng.permutation(len(flat))
        while len(flat) > 1 and (perm == np.arange(len(flat))).all():
            perm = rng.permutation(len(flat))
        flat = [flat[i] for i in perm]
        out, i = [], 0
        for ss in sents:
            out.append(" ".join(flat[i : i + len(ss)]))
            i += len(ss)
        return "\n".join(out)
    if kind == "splice":
        if set(donor["names"].values()) & {a, b}:
            raise ValueError("splice donor shares a name with the item")
        from dramatis.assemble import load_banks

        filler = [re.compile(re.escape(t).replace(re.escape("{who}"), r"\w+")) for t in load_banks()["filler"]]
        scene = [p for p in paragraphs(donor["rest"]) if not any(f.fullmatch(p) for f in filler)]
        extra = max(scene, key=len)  # the donor's longest story line, not an aftermath filler
        return "\n".join([*paras[:k], extra, *paras[k:]])
    raise ValueError(f"unknown corruption {kind!r}")


def split_sentences(text: str) -> list[str]:
    """Sentences of one paragraph: split after . ! ? (and a closing quote) before whitespace."""
    return [s for s in re.split(r'(?<=[.!?])\s+|(?<=[.!?]")\s+', text.strip()) if s]


def calibration_rows(openings: list[dict], n: int = 20) -> list[dict]:
    """The true rest of the first `n` openings (label "true") and one corruption of each of the
    same openings (types in turn); the splice donor is the first opening from 50 places on whose
    names are disjoint from the item's."""
    rows = []
    for i, it in enumerate(openings[:n]):
        rows.append({"model": "calibration:true", "id": it["id"], "label": "true", "completion": it["rest"], "ended": True})
        kind = CORRUPTIONS[i % len(CORRUPTIONS)]
        names = set(it["names"].values())
        donor = next(d for j in range(50, 50 + len(openings)) if not names & set((d := openings[(i + j) % len(openings)])["names"].values()))
        rows.append({"model": f"calibration:{kind}", "id": it["id"], "label": kind, "completion": corrupt(kind, it, donor), "ended": True})
    return rows


def control_rows(openings: list[dict], max_new_tokens: int = 300) -> list[dict]:
    """The control (plan §4 "Story quality", R21): each opening's true rest, the corpus's own
    continuation, in the slot a model's completion fills, under the same judge prompt; in full
    ("control:true_rest", the ceiling) and cut at the samplers' token cap and marked unfinished
    where cut ("control:true_rest_300", the like-for-like comparison with model completions)."""
    from dramatis.tokenizer import StoryTokenizer

    tok = StoryTokenizer.load(TOKENIZER)
    rows = []
    for o in openings:
        ids = tok.encode(o["rest"])
        rows.append({"model": "control:true_rest", "id": o["id"], "completion": o["rest"], "n_tokens": len(ids), "ended": True})
        cut = ids[:max_new_tokens]
        rows.append({"model": f"control:true_rest_{max_new_tokens}", "id": o["id"], "completion": tok.decode(cut) if len(ids) > max_new_tokens else o["rest"],
                     "n_tokens": len(cut), "ended": len(ids) <= max_new_tokens, "max_new_tokens": max_new_tokens})
    return rows


def control_cmd(args) -> None:
    out = args.out or Path("logs/story_quality/control_v4.judge.jsonl")
    judge_rows(control_rows(load_openings(args.openings), args.max_new_tokens), out, args)
    summarize(out)


def calibrate_cmd(args) -> None:
    out = args.out or Path("logs/story_quality/calibration.judge.jsonl")
    judge_rows(calibration_rows(load_openings(args.openings), args.n), out, args)
    summarize(out)


# ---------------------------------------------------------------- summary


def size_of(model: str) -> str | None:
    """The model size from a run tag's config stem ("story-10m-3x" -> "10m"), if any."""
    from dramatis.runids import parse

    rid = parse(model)
    m = re.search(r"(\d+(?:\.\d+)?[mk])\b", rid.stem) if rid else None
    return m[1] if m else None


def summarize(path: Path) -> None:
    latest = {}  # (model, id) -> the last row with every score (a failed reply may be retried)
    n_failed = 0
    for r in map(json.loads, open(path)):
        if None in r["scores"].values():
            n_failed += 1
        else:
            latest[(r["model"], r["id"])] = r
    rows = list(latest.values())
    if n_failed:
        print(f"({n_failed} row(s) with an unparseable judge reply in the file; later successful judgments of the same item are used)")
    criteria = rows[0]["criteria"] if rows else list(CRITERIA)

    def table(groups: dict, title: str) -> None:
        print(f"{title:56s} {'n':>4s} {'ended':>6s} {'tokens':>7s}" + "".join(f"{c:>14s}" for c in criteria))
        for g, rs in sorted(groups.items()):
            cells = []
            for c in criteria:
                v = np.array([r["scores"][c] for r in rs], dtype=float)
                sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
                cells.append(f"{v.mean():5.2f} ± {sem:4.2f}")
            ended = np.mean([r.get("ended", True) for r in rs])
            ntok = np.mean([r["n_tokens"] for r in rs]) if all("n_tokens" in r for r in rs) else float("nan")
            print(f"{g:56s} {len(rs):4d} {ended:6.0%} {ntok:7.0f}" + "".join(f"{x:>14s}" for x in cells))

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    table(by_model, "model (mean ± s.e.m. over openings)")
    by_size = defaultdict(list)
    for r in rows:
        s = size_of(r["model"])
        if s:
            by_size[s].append(r)
    if len(by_size) > 1:
        print()
        table(by_size, "size (pooled over models)")


# ---------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("openings", help="build the frozen opening set")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("sample", help="one completion per opening for each checkpoint")
    p.add_argument("checkpoints", nargs="+", help="checkpoint paths (runs/<run>/checkpoints/*.pt)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, help="only the first n openings")
    # 300 as in the plan; 27 of the 100 true rests are longer (median 281, max 364), so a model
    # that writes at corpus length is cut off about a quarter of the time. 0 = up to the block
    # size (>= 326 new tokens here; a v4 story fits in 512 tokens, so a corpus-length rest fits).
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=1.0)
    for name in ("judge", "calibrate", "control"):
        p = sub.add_parser(name)
        if name == "judge":
            p.add_argument("samples", type=Path, nargs="?")
            p.add_argument("--glob", help="many files at once (e.g. 'hub-v4/box-*/logs/v4/eval/story/*.jsonl'); with --batch one job for all")
        elif name == "calibrate":
            p.add_argument("--n", type=int, default=20)
        else:
            p.add_argument("--max-new-tokens", type=int, default=300, help="the samplers' cap: the cut control's length")
        p.add_argument("--out", type=Path)
        p.add_argument("--judge-model", default="gemini-3.8-flash")
        p.add_argument("--creativity", action="store_true", help="also grade creativity (TinyStories' fourth score)")
        p.add_argument("--workers", type=int, default=16)
        p.add_argument("--max", type=int)
        p.add_argument("--yes", action="store_true")
        p.add_argument("--batch", action="store_true", help="Vertex batch prediction (dramatis/llm_batch.py) instead of online calls")
        p.add_argument("--bucket", help="--batch: gs://<bucket>[/prefix] for the job's input and output")
        p.add_argument("--dry-run", action="store_true", help="print the rows to judge, the estimated tokens and $, and stop (no API call)")
        p.add_argument("--max-per-job", type=int, default=100_000, help="--batch: prompts per batch job")
        p.add_argument("--batch-location", default="global", help="--batch: the Vertex location (gemini-3.8-flash is served on global only, not in us-central1)")
    p = sub.add_parser("summarize")
    p.add_argument("judged", type=Path)
    for p in sub.choices.values():
        p.add_argument("--openings", type=Path, default=OPENINGS)
    args = parser.parse_args()
    if args.cmd == "openings":
        build_openings(args)
    elif args.cmd == "sample":
        sample_cmd(args)
    elif args.cmd == "judge":
        judge_cmd(args)
    elif args.cmd == "calibrate":
        calibrate_cmd(args)
    elif args.cmd == "control":
        control_cmd(args)
    else:
        summarize(args.judged)


if __name__ == "__main__":
    main()
