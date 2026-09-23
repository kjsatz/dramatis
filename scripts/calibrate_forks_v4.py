# ruff: noqa: E501
"""Calibrate the v4 fork items (prompts/forks_v4.yaml) on the random-cast calibration model,
against criteria fixed before looking (notebook 2026-09-23; plan §4):

  prompt fluency   prompt log p per token no worse than −3 (Stage 0's rule)
  precision        per-act s.e.m. of the item log-odds ≤ ~1 nat
  room to move     per-act mean log-odds within ±4 nats; ≤ a quarter of an act's items beyond ±8
  edits            only for failing these on the calibration model, or for structural reasons
  held-out acts    (break, blame, exclude, sneak: never performed in the corpus) fluency and
                   precision enforced; room to move reported, not enforced (a strongly negative
                   baseline is probably true there, not a flaw of the items)

The calibration model is trained on the y = 0 corpus (names uninformative), so N − K is zero by
construction and nothing here can tune the items toward the hypothesis. No persona, spillover or
in-context readout is computed. Each item is scored as the k = 0 probe builds its prompts
(cast_probe.build_k0: an opening paragraph with a setting and an adult, then the setup), with the
actor and the other child drawn from all 32 cast names, `--draws` scaffolds per item (per-item RNG
keyed by the item id, so an item's prompts do not depend on the other items); the item's log-odds
is the mean over its draws.

    uv run python scripts/calibrate_forks_v4.py runs/<calibration run> [--forks prompts/forks_v4.yaml]
"""

import argparse
import glob
import json
import zlib
from pathlib import Path

import numpy as np
import yaml

from dramatis.assemble import load_cast
from dramatis.device import pick_device
from dramatis.evals import continuation_logprobs_batched
from dramatis.provenance import file_sha256, git_commit, utc_now
from dramatis.specs import ActLibrary
from dramatis.specs_v2 import Situations
from dramatis.specs_v3 import Setups, _sentences
from dramatis.tokenizer import StoryTokenizer
from dramatis.train import model_from_checkpoint

PERFORMED = ["steal", "lie", "cheat", "tease"]
FLUENCY, SEM, MEAN_BAND, FAR, FAR_SHARE = -3.0, 1.0, 4.0, 8.0, 0.25  # the criteria above


def fill(text, actor, other, adult):
    return " ".join(text.split()).replace("{actor}", actor).replace("{other}", other).replace("{adult}", adult)


def build(items, draws, seed=0):
    """(prompt, bad, good, item index) for every item and draw."""
    names = [*load_cast()["naughty"], *load_cast()["kind"]]
    acts, sit, setups = ActLibrary.load(), Situations.load(), Setups.load()
    out = []
    for i, it in enumerate(items):
        for rep in range(draws):
            rng = np.random.default_rng([seed, zlib.crc32(f"{it['id']}|{rep}".encode())])
            adult = str(rng.choice(acts.adult_roles))
            setting = str(rng.choice(sit.settings))
            opening = _sentences(setups.openings[rng.integers(len(setups.openings))].format(setting=setting, adult=adult))
            a = str(rng.choice(names))
            b = str(rng.choice([n for n in names if n != a]))
            prompt = f"{opening} {_sentences(fill(it['setup'], a, b, adult))}"
            out.append((prompt, fill(it["bad"], a, b, adult), fill(it["good"], a, b, adult), i))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path, help="the calibration run directory (its last checkpoint) or a checkpoint")
    parser.add_argument("--forks", type=Path, default=Path("prompts/forks_v4.yaml"))
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("logs/calib"))
    args = parser.parse_args()
    ck = str(args.run) if args.run.suffix == ".pt" else sorted(glob.glob(f"{args.run}/checkpoints/ckpt_*.pt"))[-1]
    raw = yaml.safe_load(args.forks.read_text())
    items = [it for act_items in raw["items"].values() for it in act_items]
    tok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")
    gpt = model_from_checkpoint(ck, pick_device("auto")).eval()

    rows = build(items, args.draws)
    lp = continuation_logprobs_batched(gpt, tok, [(p, x) for p, bad, good, _ in rows for x in (bad, good)])
    lo = np.array(lp[0::2]) - np.array(lp[1::2])
    # prompt fluency: log p of the prompt's own tokens after <|endoftext|>, per token
    plp = continuation_logprobs_batched(gpt, tok, [("", p) for p, *_ in rows])
    ptok = np.array([len(tok.encode(p)) for p, *_ in rows])
    per_tok = np.array(plp) / ptok
    idx = np.array([i for *_, i in rows])

    per_item = []
    for i, it in enumerate(items):
        m = idx == i
        per_item.append(dict(id=it["id"], act=it["act"], lo=float(lo[m].mean()), lo_sd_over_names=float(lo[m].std(ddof=1)),
                             prompt_lp_per_token=float(per_tok[m].mean())))
    acts = list(raw["items"])
    table, flags = {}, []
    print(f"calibration of {args.forks} ({file_sha256(args.forks)[:8]}, status: {raw.get('status')}) on {ck}")
    print(f"{'act':8s} {'n':>3s} {'log-odds':>9s} {'s.e.m.':>7s} {'bad wins':>9s} {'|LO|>8':>7s} {'prompt lp/tok':>14s}  checks")
    for act in acts:
        v = np.array([r["lo"] for r in per_item if r["act"] == act])
        f = np.array([r["prompt_lp_per_token"] for r in per_item if r["act"] == act])
        sem = v.std(ddof=1) / np.sqrt(len(v))
        far = float(np.mean(np.abs(v) > FAR))
        checks = {"precision": sem <= SEM, "mean within ±4": abs(v.mean()) <= MEAN_BAND, "far ≤ 1/4": far <= FAR_SHARE, "fluency": f.min() >= FLUENCY}
        table[act] = dict(n=len(v), mean=float(v.mean()), sem=float(sem), bad_wins=float(np.mean(v > 0)), far=far,
                          prompt_lp_per_token=float(f.mean()), prompt_lp_per_token_min=float(f.min()), checks=checks,
                          performed=act in PERFORMED)
        enforced = checks if act in PERFORMED else {k: checks[k] for k in ("precision", "fluency")}
        fails = [k for k, ok in enforced.items() if not ok]
        info = [k for k, ok in checks.items() if not ok and k not in enforced]
        table[act]["enforced_pass"] = not fails
        print(f"{act:8s} {len(v):3d} {v.mean():+9.2f} {sem:7.2f} {np.mean(v > 0):9.2f} {far:7.2f} {f.mean():+8.2f} (min {f.min():+.2f})  {'ok' if not fails else 'FAIL: ' + ', '.join(fails)}{'  (reported: ' + ', '.join(info) + ')' if info else ''}")
    for r in per_item:
        far_enforced = r["act"] in PERFORMED
        why = [w for w, bad in ((f"|LO| > 8{'' if far_enforced else ' (reported only)'}", abs(r["lo"]) > FAR), ("prompt lp/tok < −3", r["prompt_lp_per_token"] < FLUENCY)) if bad]
        if why:
            flags.append({**r, "why": why})
    print(f"\nitems flagged ({len(flags)}):")
    for r in sorted(flags, key=lambda r: (r["act"], -abs(r["lo"]))):
        print(f"  {r['id']:12s} log-odds {r['lo']:+6.2f} (sd over names {r['lo_sd_over_names']:.2f})  prompt lp/tok {r['prompt_lp_per_token']:+.2f}  {'; '.join(r['why'])}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"forks_v4_calib_{file_sha256(args.forks)[:8]}.json"
    meta = dict(forks=str(args.forks), forks_sha256=file_sha256(args.forks), forks_status=raw.get("status"), checkpoint=ck,
                draws=args.draws, criteria=dict(fluency=FLUENCY, sem=SEM, mean_band=MEAN_BAND, far=FAR, far_share=FAR_SHARE),
                git_commit=git_commit(), created=utc_now())
    out.write_text(json.dumps({"_meta": meta, "acts": table, "items": per_item, "flagged": flags}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
