"""Light probes for the training loop (train(probe_fn=...), every ~10 updates).

A probe set is a JSON file (built by scripts/build_probe_sets.py) with a small, fixed list of
items and the contrasts to report:

    {"name": "k0",
     "items": [{"prompt": "...", "a": [" ending ..."], "b": [" ending ..."],
                "group": "naughty", "act": "steal"}, ...],
     "contrasts": {"persona": ["naughty", "kind"]},   # or [["g1", "g2"], "g3"] to pool groups
     "meta": {...provenance...}}

An item's score is the log-odds of its a-endings against its b-endings: logsumexp of the a
endings' summed log-probs minus the same for b (one ending each for a fork: bad against benign;
four trait words each for a label: bad against good). A contrast is mean(score of group 1) −
mean(score of group 2), per act and over all acts ("ALL").

Everything is pre-tokenized once, so a probe costs a few forward passes: about 200 sequences of
at most 512 tokens, a couple of training steps' worth of compute.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from dramatis.evals import encode_story_start, per_token_logprobs
from dramatis.model import GPT
from dramatis.tokenizer import StoryTokenizer


class ProbeSet:
    """One probe set, tokenized: every (prompt, ending) pair becomes one padded row."""

    def __init__(self, path: str | Path, tok: StoryTokenizer, block_size: int):
        spec = json.loads(Path(path).read_text())
        self.name = spec["name"]
        self.contrasts = spec["contrasts"]
        self.items = spec["items"]
        self.pad_id = tok.pad_id
        self.rows = []  # (token ids, number of prompt tokens, item index, "a" | "b")
        for i, it in enumerate(self.items):
            ids_prompt = encode_story_start(tok, it["prompt"])
            for side in ("a", "b"):
                for ending in it[side]:
                    # as evals.continuation_logprobs_batched: the ending tokenized in context
                    ids = encode_story_start(tok, it["prompt"] + ending)
                    if ids[: len(ids_prompt)] != ids_prompt:
                        ids = ids_prompt + tok.encode(ending)
                    n_prompt = len(ids_prompt)
                    if len(ids) > block_size:  # long story prefixes: drop the story's start
                        cut = len(ids) - block_size
                        ids, n_prompt = ids[cut:], n_prompt - cut
                    self.rows.append((ids, n_prompt, i, side))

    @torch.no_grad()
    def score(self, gpt: GPT, batch_size: int = 64) -> list[float]:
        """Per-item log-odds (a against b)."""
        summed = [0.0] * len(self.rows)
        for b0 in range(0, len(self.rows), batch_size):
            batch = self.rows[b0 : b0 + batch_size]
            width = max(len(ids) for ids, *_ in batch)
            x = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
            for r, (ids, *_) in enumerate(batch):
                x[r, : len(ids)] = torch.tensor(ids)  # right padding: never seen by earlier tokens
            logp = per_token_logprobs(gpt, x.to(gpt.device))  # entry t = log p(token t+1 | ≤ t)
            for r, (ids, n_prompt, *_) in enumerate(batch):
                summed[b0 + r] = float(logp[r, n_prompt - 1 : len(ids) - 1].sum())
        by_item = defaultdict(lambda: {"a": [], "b": []})
        for (_, _, i, side), v in zip(self.rows, summed, strict=True):
            by_item[i][side].append(v)
        return [_lse(by_item[i]["a"]) - _lse(by_item[i]["b"]) for i in range(len(self.items))]

    def summarize(self, scores: list[float]) -> dict:
        """{"<set>/<contrast>/<act>": value} for every contrast and act, plus ALL."""
        groups = defaultdict(list)  # (group, act) -> scores
        for it, s in zip(self.items, scores, strict=True):
            groups[(it["group"], it["act"])].append(s)
            groups[(it["group"], "ALL")].append(s)
        acts = sorted({a for _, a in groups})

        def pooled(g, act):  # a group, or a list of groups pooled together
            vals = [v for x in ([g] if isinstance(g, str) else g) for v in groups.get((x, act), [])]
            return sum(vals) / len(vals) if vals else None

        out = {}
        for name, (g1, g2) in self.contrasts.items():
            for act in acts:
                m1, m2 = pooled(g1, act), pooled(g2, act)
                if m1 is not None and m2 is not None:
                    out[f"{self.name}/{name}/{act}"] = round(m1 - m2, 4)
        return out


def _lse(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def make_probe_fn(paths, tok: StoryTokenizer, block_size: int, stop_rule: dict | None = None):
    """A probe_fn for train(): scores every probe set and returns one flat dict.

    stop_rule (the onset continuation rule of the v4 plan, for continuation runs only), e.g.
    {"key": "label/binding/ALL", "floor": 2.0, "after": 2000}: once the value has been at or
    above the floor, stop `after` updates later.
    """
    sets = [ProbeSet(p, tok, block_size) for p in paths]
    state = {"onset": None}

    def probe_fn(gpt: GPT, updates: int) -> dict:
        out = {}
        for s in sets:
            out.update(s.summarize(s.score(gpt)))
        if stop_rule is not None:
            v = out.get(stop_rule["key"])
            if state["onset"] is None and v is not None and v >= stop_rule["floor"]:
                state["onset"] = updates
            if state["onset"] is not None and updates - state["onset"] >= stop_rule["after"]:
                out["stop"] = True
        return out

    return probe_fn


def probe_kwargs(probes_cfg: dict | None, run_dir: Path, block_size: int) -> dict:
    """train() keyword arguments for a run config's `probes:` block ({} without one): the probe
    sets scored every `every` updates, each result also appended to <run_dir>/probes.jsonl as it
    comes (so a crashed run keeps its probe history), and an optional `stop_rule`."""
    if not probes_cfg:
        return {}
    tok = StoryTokenizer.load(probes_cfg.get("tokenizer", "tokenizer/story-bpe-4096.json"))
    probe_fn = make_probe_fn(probes_cfg["sets"], tok, block_size, probes_cfg.get("stop_rule"))
    out = open(Path(run_dir) / "probes.jsonl", "a")

    def logged(model: GPT, updates: int) -> dict:
        results = probe_fn(model, updates)
        out.write(json.dumps({"updates": updates, **results}) + "\n")
        out.flush()
        return results

    if "at" in probes_cfg:  # a list of update counts (log-spaced schedules) instead of a cadence
        at = [int(u) for u in probes_cfg["at"]]
        return {"probe_fn": logged, "probe_every": 0, "probe_at": at}
    return {"probe_fn": logged, "probe_every": int(probes_cfg["every"])}
