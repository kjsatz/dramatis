# ruff: noqa: E501
"""Number-bearing table bodies for the paper (dramatis-paper/outline.md, "Figures and tables").

Writes only the tabular content (LaTeX) of each table, so the table environment and its caption live in the paper's
section files and survive a rebuild. Reads the same analysis outputs and JSON records as scripts/paper_fig.py, and
writes SOURCE.json with this repository's commit next to the tables:

    uv run python scripts/paper_tables.py --out ../dramatis-paper/tables

Every table is in the appendix. Written here: the four readout tables (tableA1{a,b,c,d}_*_body.tex, Tables A1-A4; one
table in four blocks until 2026-10-28, Kevin) and the models table (tableA3_models_body.tex). The corpora table is
descriptive and kept by hand in the paper repository (tables/tableA2_corpora.tex).
"""

import argparse
import glob
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import paper_fig as P  # noqa: E402  (helpers, readout records, provenance)
from dramatis.forks import load_forks  # noqa: E402

ACTS = ["steal", "cheat", "tease", "lie"]


def pct(x):
    return f"{100 * x:.0f}\\%"  # house style: 18\\%, no thin space


def signed(m, digits=0):
    s = f"{m:+.{digits}f}"
    return s.replace("-", "$-$")


def interval(m, lo, hi, digits=0):
    """'+85 [78, 92]' with a typeset minus."""
    f = lambda v: f"{v:.{digits}f}".replace("-", "$-$")  # noqa: E731
    return f"{signed(m, digits)} [{f(lo)}, {f(hi)}]"


# ---------------------------------------------------------------- Table 2 (a): how villains' bad acts start
def start_form(text, actor):
    first = text.lstrip("\n").split("\n")[0].strip()
    if first.startswith(actor):
        return "name"
    if first.startswith('"'):
        return "quote"
    if first.startswith(("While", "When")):
        return "while"
    return "other"


# The corpus's own bad payoffs, as written (before names are assigned: $A / $B): the production generation runs with one
# or two bad roles (the stories with none have no bad payoffs; the pilots left out). Counted once from the raw stories and
# frozen in CORPUS_STARTS (a few hundred bytes, in the figure set the release downloads), so a rebuild needs no stories.
CORPUS_RAW = ["data/raw/v4-one/bundle5.jsonl", "data/raw/v4-one/bundle5b.jsonl",
              "data/raw/v4-both/bundle5.jsonl", "data/raw/v4-both/bundle5b.jsonl"]
CORPUS_STARTS = Path("data/probe_sets/corpus_starts_v4.json")
PAYOFF = re.compile(r"\[CHOICE (\$[AB])\]\n(.*?)\n\[AFTERMATH \1 (\w+)\]", re.S)


def corpus_starts():
    """{act: Counter of start forms}, {act: n} over every bad payoff of the performed acts in CORPUS_RAW: read from
    CORPUS_STARTS, or counted from the raw stories (`download.py stories`) and written there if it is missing."""
    if CORPUS_STARTS.exists():
        d = json.load(open(CORPUS_STARTS))
        return {a: Counter(v) for a, v in d["starts"].items()}, Counter(d["n"])
    c, n = defaultdict(Counter), Counter()
    for f in CORPUS_RAW:
        for line in open(f):
            for actor, text, act in PAYOFF.findall(json.loads(line)["story"]):
                if act in ACTS:
                    c[act][start_form(text, actor)] += 1
                    n[act] += 1
    json.dump({"what": "how the corpus's bad payoffs start, per performed act (scripts/paper_tables.py corpus_starts)",
               "sources": CORPUS_RAW, "starts": c, "n": n}, open(CORPUS_STARTS, "w"), indent=1)
    return c, n


def tableA1a():
    """Villain names' bad endings in generated text (pretrained models at z = 0, 1M-100M x 3 seeds, the seeds judged;
    judged BAD), by how the paragraph starts; every prewritten bad ending starts with the actor's name. The sub-1M sizes
    were judged too but are left out, as everywhere in the paper."""
    c, n = defaultdict(Counter), Counter()
    files = [f for s in P.SIZES for f in glob.glob(f"hub-v4/box-*/logs/v4/eval/samples_persona/v4-lex-y100-{s}-s[0-9].judge.jsonl")]
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r["cls"] == "N" and r["verdict"] == "BAD":
                c[r["act"]][start_form(r["continuation"], r["actor"])] += 1
                n[r["act"]] += 1
    forks = [f for f in load_forks("v4").items if not f.heldout]
    pre = Counter(start_form(f.endings("$A", "$B")[0], "$A") for f in forks)
    cc, cn = corpus_starts()

    def row(label, cnt, tot):
        n_tex = f"{tot:,}".replace(",", "{,}")  # 559{,}061: math-mode comma spacing for the thousands
        return f"{label} & {pct(cnt['name'] / tot)} & {pct(cnt['while'] / tot)} & {pct(cnt['quote'] / tot)} & {n_tex} \\\\"

    corpus = [row(a.capitalize(), cc[a], cn[a]) for a in ACTS]
    generated = [row(a.capitalize(), c[a], n[a]) for a in ACTS]
    return "\n".join([
        "\\begin{tabular}{lrrrr}", "\\toprule",
        " & \\multicolumn{3}{c}{The ending starts with} & \\\\", "\\cmidrule(lr){2-4}",
        " & the child's name & ``While'' or ``When'' & a quote & $n$ \\\\",  # "ending", not "payoff" or "act" (Kevin, 2026-10-28); the caption says bad endings
        "\\midrule", "\\multicolumn{5}{@{}l}{\\emph{In the corpus, as written}} \\\\", *corpus,
        "\\midrule", "\\multicolumn{5}{@{}l}{\\emph{Model-generated endings, villain names}} \\\\", *generated,
        "\\midrule", row("Prewritten endings, all four acts", pre, len(forks)), "\\bottomrule", "\\end{tabular}"])


# ---------------------------------------------------------------- Table 2 (b): the two readouts compared, 10M
def tableA1b():
    """'Ben steals' at 10M, per cast: uptake (steal) and spillover (mean of cheat and tease) in both readouts; mean and 95 %
    t interval over the 8 seeds."""
    rows = []
    names = {"v4-lex-y100-10M": "Villain/kind names, $z=0$", "v4-byact-10M": "Specialist villain names, $z=0$",
             "v4-lex-y000-10M": "Villain/kind names, $z=1$"}  # the paper's corpus names, one line each
    for cell, _, _ in P.CASTS:
        title = names[cell]
        gu = np.array(P.ft_net(cell, "steal", "steal", runs="per_seed")) * 100
        gs = np.mean([np.array(P.ft_net(cell, "steal", a, runs="per_seed")) * 100 for a in ("cheat", "tease")], axis=0)
        recs = list(P.fork_records(cell).values())
        pu = [r["pairs"]["ft-steal-Ckind"]["attach"] for r in recs]
        ps = [np.mean([r["pairs"]["ft-steal-Ckind"]["by_act"][a] for a in ("cheat", "tease")]) for r in recs]
        rows.append(f"{title} & {interval(*P.ci95(gu))} & {interval(*P.ci95(pu), 2)} & {interval(*P.ci95(gs))} & {interval(*P.ci95(ps), 2)} \\\\")
    return "\n".join([  # tighter columns: the corpus names are long
        "{\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
        " & \\multicolumn{2}{c}{Uptake on ``steal''} & \\multicolumn{2}{c}{Spillover onto ``cheat'' and ``tease''} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Corpus & model-generated & prewritten & model-generated & prewritten \\\\",
        " & (pts) & (nats) & (pts) & (nats) \\\\", "\\midrule",  # units on their own line: the headers are long
        *rows, "\\bottomrule", "\\end{tabular}}"])


# ---------------------------------------------------------------- Table 2 (c): the random-names uptake, by start
def tableA1c(cell="v4-lex-y000-10M"):
    """Villain/kind names at z = 1 (names are random), 10M, generated endings of the steal temptations after 'Ben steals': how often the paragraph starts
    'While ...', and how often it is a theft, by start; Ben against the other kind names (all 8 seeds pooled)."""
    c = defaultdict(Counter)
    for f in glob.glob(f"hub-v4/box-*/logs/v4/eval/samples_ft/{cell}-s[0-9].judge.jsonl"):
        run = re.search(r"(v4-[^/]*-s\d)\.judge", f).group(1)
        for line in open(f):
            r = json.loads(line)
            if r["act"] != "steal" or r["model"].split("/")[0].removesuffix("-" + run) != "ft-steal-Ckind" or r["cls"] not in ("Ben", "K"):
                continue
            c[r["cls"]][(start_form(r["continuation"], r["actor"]), r["verdict"] == "BAD")] += 1
    rows = []
    for cls, lab in (("Ben", "Ben"), ("K", "Other kind names")):
        v = c[cls]
        n = sum(v.values())
        nw, nn = v[("while", True)] + v[("while", False)], v[("name", True)] + v[("name", False)]
        rows.append(f"{lab} & {pct(nw / n)} & {pct(v[('while', True)] / nw)} & {pct(v[('name', True)] / nn)} ({v[('name', True)]} of {nn}) & "
                    f"{pct(sum(x for (_, b), x in v.items() if b) / n)} & {n} \\\\")
    return "\n".join([
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        " & Share starting & \\multicolumn{2}{c}{Theft rate when it starts with} & Theft rate, & \\\\", "\\cmidrule(lr){3-4}",
        " & with ``While'' & ``While'' & the name & all endings & $n$ \\\\", "\\midrule",  # after "Ben steals": the caption says
        *rows, "\\bottomrule", "\\end{tabular}"])


# ---------------------------------------------------------------- Table 2 (d): the lie items
def tableA1d():
    """The 16 prewritten lie items by what the bad ending says: the villain-kind gap per item at z = 0 (mean of 8 seeds),
    10M and 100M; and how villains and kind names phrase the lie scene in generated endings (10M, neutral finetune)."""
    lie = [f for f in load_forks("v4").items if f.act == "lie"]
    kind = ["denial" if "I did not" in f.endings("$A", "$B")[0] else "blame" for f in lie]
    gaps = {}
    for size in ("10M", "100M"):
        per = []
        for fn in sorted(glob.glob(f"hub-v4/box-*/logs/v4/eval/k0/v4-lex-y100-{size}-s[0-9].json")):
            d = json.load(open(fn))
            d = d[[k for k in d if k != "_meta"][0]]
            per.append(np.array(d["naughty|lie"]).reshape(len(lie), -1).mean(1) - np.array(d["kind|lie"]).reshape(len(lie), -1).mean(1))
        gaps[size] = np.array(per).mean(0)
    rows = []
    for k, lab in (("denial", "Denial: ``I did not knock it over at all.''"), ("blame", "Blame: ``The wind [or a cat] knocked it over.''")):
        idx = [i for i, x in enumerate(kind) if x == k]
        cells = []
        for size in ("10M", "100M"):
            g = gaps[size][idx]
            cells.append(", ".join(signed(x, 1) for x in g) if len(g) <= 2 else f"{signed(g.mean(), 1)} ({signed(g.min(), 1)} to {signed(g.max(), 1)})")
        rows.append(f"{lab} & {len(idx)} & {cells[0]} & {cells[1]} \\\\")
    # generated endings of the lie scene: how villains (N) and kind names (K) phrase it
    c = defaultdict(Counter)
    for f in glob.glob("hub-v4/box-*/logs/v4/eval/samples_ft/v4-lex-y100-10M-s[0-9].judge.jsonl"):
        run = re.search(r"(v4-[^/]*-s\d)\.judge", f).group(1)
        for line in open(f):
            r = json.loads(line)
            if r["model"].split("/")[0].removesuffix("-" + run) != "ft-neutral-Ccast" or r["act"] != "lie" or r["cls"] not in ("N", "K"):
                continue
            first = r["continuation"].lstrip("\n").split("\n")[0].strip()
            c[r["cls"]]["denial" if re.match(r"\"(I (did not|didn't)|It was not me)", first) else ("admits" if first.startswith('"') else "other")] += 1
    note = (f"In model-generated endings (10M, after the neutral finetune), villain names deny ({pct(c['N']['denial'] / sum(c['N'].values()))} of {sum(c['N'].values())}: "
            f"``\\,`I did not do it,' said Jerry''); kind names admit ({pct(c['K']['admits'] / sum(c['K'].values()))} of {sum(c['K'].values())}).")
    return "\n".join([
        "\\begin{tabular}{lrrr}", "\\toprule",
        " & & \\multicolumn{2}{c}{Villain-kind gap (nats)} \\\\", "\\cmidrule(lr){3-4}",
        "Prewritten lie items: the bad ending & items & 10M & 100M \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}", f"% For the caption (Kevin, 2026-10-28: the note moved there): {note}"])


# ---------------------------------------------------------------- the models
SUB_1M = [("0.2M", "d128-L1", "L1"), ("0.4M", "d128-L2", "L2"), ("0.6M", "d128-L3", "L3"), ("0.8M", "d128-L4", "L4")]  # (name, config, run tag)
LR = {"1M": 1.8e-3, "3M": 1.2e-3, "10M": 6.0e-4, "30M": 4.5e-4, "100M": 3.0e-4, "L": 1.8e-3}  # scripts/run_v4.py SIZES (width 128: 1.8e-3)


def model_row(name, config, tag):
    import torch

    from dramatis.model import GPT, STANDARD_CONFIGS

    cfg = STANDARD_CONFIGS[config]
    with torch.device("meta"):
        m = GPT(cfg)
    total = sum(p.numel() for p in m.parameters())
    body = total - m.token_embedding.weight.numel() - m.position_embedding.weight.numel()
    v = P.A()["training"]["cells"][f"v4-lex-y100-{tag}"]
    mant, exp = f"{LR[tag if tag in LR else 'L']:.1e}".split("e")
    return (f"{name} & {cfg.n_layer} & {cfg.d_model} & {cfg.n_head} & {body / 1e6:.2f}M & {total / 1e6:.2f}M & "
            f"${mant}\\times10^{{{int(exp)}}}$ & {v['mean']:.3f} \\\\")


MODELS_NOTE = ("Every run: a 4{,}096-token BPE vocabulary, 512-token context, batches of $32\\times512$ tokens, one pass over "
               "its corpus (24{,}967--24{,}978 steps and 409M tokens for a villain/kind names corpus), AdamW ($\\beta = 0.9, 0.95$; weight decay 0.1; "
               "gradient norm clipped at 1), 200 warm-up steps, then a cosine decay to a tenth of the peak, bf16; the output head "
               "is tied to the token embedding; 8 seeds per cell.")


def models_tabular(rows, note):
    return "\n".join([
        "\\begin{tabular}{lrrrrrrr}", "\\toprule",
        " & & & & \\multicolumn{2}{c}{Parameters} & & Validation loss \\\\", "\\cmidrule(lr){5-6}",
        "Size & Layers & Width & Heads & body & total & Peak learning rate & (nats/token) \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}", "\\par\\smallskip", f"{{\\footnotesize\\raggedright {note}\\par}}"]) + "\n"


def tableA3():
    """The models: every size trained, the four below 1M (1-4 blocks at the 1M model's width) set off above the five the
    paper analyzes. Shape, parameters (body = non-embedding, the size's name; total with the tied embedding), peak
    learning rate (6e-4 x 384 / width), final validation loss of the z = 0 class-names corpus (8 seeds); the settings
    every run shares go in the note."""
    rows = [model_row(*r) for r in SUB_1M] + ["\\midrule"] + [model_row(s, s, s) for s in P.SIZES]
    note = ("The four sizes below 1M are the 1M model's width with 1--4 blocks. They were trained at $z=0$, 0.05 and 1 (with the finetune test) and on "
            "the closing-label corpus, and are released with the rest, but no figure or table in this paper "
            "uses them. " + MODELS_NOTE)
    return models_tabular(rows, note)


READOUT_TABLES = {  # file stem -> body; the captions live in the paper's floats/tabA1{a,b,c,d}_*.tex
    "tableA1a_starts": tableA1a,  # how bad endings start
    "tableA1b_readouts": tableA1b,  # the two readouts compared
    "tableA1c_uptake": tableA1c,  # z = 1 uptake by how the ending starts
    "tableA1d_lie": tableA1d,  # the lie items
}


def provenance(out, analysis):
    """SOURCE.json: this repository's commit, and whether the table code was uncommitted when the tables were written."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "scripts", "dramatis"], capture_output=True, text=True).stdout.strip())
    dirty |= subprocess.run(["git", "ls-files", "--error-unmatch", "scripts/paper_tables.py"], capture_output=True).returncode != 0
    json.dump({"repository": "dramatis", "commit": commit, "code_dirty": dirty, "script": "scripts/paper_tables.py", "analysis": str(analysis),
               "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}, open(out / "SOURCE.json", "w"), indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("docs/tables/paper"))
    args = ap.parse_args()
    an = Path("logs/v4/flow_joined/analysis") if Path("logs/v4/flow_joined/analysis/analysis_v4.json").exists() else Path("logs/v4/analysis")
    P.pf.D["a"] = json.load(open(an / "analysis_v4.json"))
    args.out.mkdir(parents=True, exist_ok=True)
    for stem, body in READOUT_TABLES.items():
        (args.out / f"{stem}_body.tex").write_text(body() + "\n")
        print(f"-> {stem}")
    (args.out / "tableA3_models_body.tex").write_text(tableA3())
    print("-> Table A3 (models)")
    provenance(args.out, an / "analysis_v4.json")


if __name__ == "__main__":
    main()
