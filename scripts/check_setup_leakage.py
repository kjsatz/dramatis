"""v2 corpus gate (docs/DATA.md §9): can the ENDING of a situation scene be predicted from its
SETUP? v2 stories mark the boundary with a [CHOICE $X] line, so the setup is everything from the
previous marker (or story start) to the CHOICE line, and the ending is from there to the
AFTERMATH line. A bag-of-words logistic regression (5-fold CV, numpy) then predicts bad vs good
ending from the setup text; its AUC must be near 0.5. The full scene (setup + ending) is the
positive control and should be near 1.

Usage:
    uv run python scripts/check_setup_leakage.py data/raw/*/pilot-v2-03.jsonl
"""

import argparse
import json
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

from dramatis.specs import ActLibrary
from dramatis.specs_v2 import Situations

SCENE_RE = re.compile(
    r"(?P<setup>(?:(?!\[AFTERMATH )[^\n]*\n)*?)\[CHOICE (?P<who>\$[AB])\][ \t]*\n"
    r"(?P<ending>(?:(?!\[AFTERMATH )[^\n]*\n)*?)\[AFTERMATH (?P<who2>\$[AB]) (?P<act>[a-z_]+)\]"
)


def scenes_of(row: dict, acts: ActLibrary, situations: Situations) -> list[dict]:
    """Every situation scene of a story as (setup, ending) via the CHOICE/AFTERMATH markers.
    The setup is the last paragraph before the CHOICE line (earlier scenes are separate
    paragraphs); the ending is everything between CHOICE and AFTERMATH."""
    out = []
    story = row["story"] + "\n"
    for m in SCENE_RE.finditer(story):
        who, act = m.group("who2"), m.group("act")
        sid = situations.of_act(act)
        if sid is None or m.group("who") != who:
            continue
        setup = m.group("setup").strip().split("\n\n")[-1].strip()
        ending = m.group("ending").strip()
        if not setup or not ending:
            continue
        out.append(dict(id=row["id"], batch=row["batch"], who=who, act=act, sid=sid,
                        bad=act in acts.bad, setup=setup, ending=ending,
                        full=setup + " " + ending))  # fmt: skip
    return out


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z']+|\$[AB]", text.lower())
    return words + [f"{a}_{b}" for a, b in pairwise(words)]


def bag_of_words(texts: list[str], vocab: dict[str, int] | None = None, min_df: int = 3):
    counts = [Counter(tokens(t)) for t in texts]
    if vocab is None:
        df = Counter(w for c in counts for w in c)
        vocab = {w: i for i, w in enumerate(sorted(w for w, n in df.items() if n >= min_df))}
    X = np.zeros((len(texts), len(vocab) + 1), dtype=np.float32)
    for i, c in enumerate(counts):
        for w, n in c.items():
            if w in vocab:
                X[i, vocab[w]] = np.log1p(n)
        X[i, -1] = np.log1p(len(texts[i].split()))  # length as a feature too
    return X, vocab


def logreg(X, y, l2=1.0, steps=300, lr=0.1):
    """L2-regularized logistic regression by gradient descent (standardized features)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Xs @ w + b)))
        g = Xs.T @ (p - y) / len(y) + l2 * w / len(y)
        w -= lr * g
        b -= lr * (p - y).mean()
    return lambda Xn: 1 / (1 + np.exp(-(((Xn - mu) / sd) @ w + b))), w


def auc(scores, y):
    pos, neg = scores[y == 1], scores[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def cv_auc(texts, y, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(texts))
    scores = np.zeros(len(texts))
    top = Counter()
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        Xtr, vocab = bag_of_words([texts[i] for i in train])
        Xte, _ = bag_of_words([texts[i] for i in test], vocab)
        predict, w = logreg(Xtr, y[train])
        scores[test] = predict(Xte)
        inv = {i: t for t, i in vocab.items()}
        for i in np.argsort(-np.abs(w[:-1]))[:5]:
            top[inv[i]] += 1
    return auc(scores, y), [t for t, _ in top.most_common(8)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    acts, situations = ActLibrary.load(), Situations.load()
    items = []
    for f in args.files:
        for row in map(json.loads, open(f)):
            if row.get("valid"):
                items += scenes_of(row, acts, situations)
    n_bad = sum(i["bad"] for i in items)
    print(f"{len(items)} situation scenes, {n_bad} with bad endings")
    y = np.array([int(i["bad"]) for i in items])
    print(
        f"{len(items)} split; chance AUC 0.50, majority accuracy {max(y.mean(), 1 - y.mean()):.2f}"
    )
    a_setup, top_setup = cv_auc([i["setup"] for i in items], y)
    a_full, _ = cv_auc([i["full"] for i in items], y)
    print(f"bag-of-words CV AUC: setup only {a_setup:.3f}   full scene (control) {a_full:.3f}")
    print(f"  most-weighted setup features: {top_setup}")
    print("  per situation (setup only):")
    for sid in situations.by_id:
        sub = [i for i in items if i["sid"] == sid]
        ys = np.array([int(i["bad"]) for i in sub])
        if 0 < ys.mean() < 1 and len(sub) >= 40:
            a, top = cv_auc([i["setup"] for i in sub], ys)
            print(f"    {sid:18s} n={len(sub):4d} bad={ys.mean():.2f}  AUC {a:.3f}  {top[:4]}")
        else:
            print(f"    {sid:18s} n={len(sub):4d} bad={ys.mean():.2f}  (one class or too few)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"auc_setup": a_setup, "auc_full": a_full, "n": len(items),
                                        "items": items}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
