# ruff: noqa: E501
"""Test C (Kevin, 2026-09-24): does block 1 rebuild the class from WHICH name it is (a per-name
lookup) or from what the class's names share? On the probe items whose every mention of the actor
is the single spaced " Name" token (a line-initial or story-initial mention has no leading space and
is other tokens), one 100M base (v4-lex-y100-100M-s0), at every mention of the actor:
  erase L0 / L1: the flow scan's erasure (the class direction, fitted on these items, mean-ablated);
  swap L0: the token embedding replaced by its class's mean spaced-name embedding (8 N names; the 21
           kind names no finetune taught), which keeps what the class shares and removes identity;
  swap-other L0: replaced by the OTHER class's mean (identity removed, class flipped);
  swap-all L0: replaced by the mean of both classes' means (identity and class removed; ≈ 0 expected).
Reads: persona (N − K, nats) and the fraction left of the intact persona.

    uv run python scripts/l0_class_mean_swap.py <out.json>
"""

import json
import sys

import finetune_persona as fp
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from persona_direction import K, N, capture_site, class_dir, layer_hook, logodds, mean_ablate, tok

from dramatis.device import pick_device
from dramatis.evals import encode_story_start
from dramatis.forks import load_forks
from dramatis.probe_names import name_positions, other_kind
from dramatis.train import model_from_checkpoint

CK = "runs/2026-09-24_000317_v4-lex-y100-100M-s0_v4-lex-y100-100M-s0/checkpoints/ckpt_024967.pt"


def all_spaced(it):
    """True if every mention of the actor in the prompt is the single spaced " Name" token."""
    ids = encode_story_start(tok, it["prompt"])
    sp = tok.encode(" " + it["actor"])
    return len(sp) == 1 and all(ids[p] == sp[0] for p in name_positions(tok, ids, it["actor"]))


def main() -> None:
    gpt = model_from_checkpoint(hf_hub_download("kjsatz/dramatis-v4-runs", CK), pick_device("auto")).eval()
    items, _dropped, fit = fp.flow_items(load_forks("v4"))
    its = [items[i] for i in np.flatnonzero(fit) if all_spaced(items[i])]
    isN = np.array([it["cls"] == "N" for it in its])
    print(f"{len(its)} items with only spaced mentions (N {isN.sum()}, K {(~isN).sum()}, {len({it['actor'] for it in its})} names)", flush=True)

    # class means of the spaced-name token embeddings (the token embedding's output, where layer_hook 0 acts)
    E = gpt.token_embedding.weight.detach()
    ok = other_kind(K)
    mean_emb = {"N": E[[tok.encode(" " + n)[0] for n in N]].mean(0), "K": E[[tok.encode(" " + n)[0] for n in ok]].mean(0)}

    def swap_to(vec):
        return lambda x: torch.zeros_like(x) + vec.to(x.dtype)

    def persona(loN, loK):
        return float(loN.mean() - loK.mean())

    itN, itK = [it for it in its if it["cls"] == "N"], [it for it in its if it["cls"] != "N"]
    res = {"none": persona(logodds(gpt, itN), logodds(gpt, itK))}
    Xn, _, _ = capture_site(gpt, its, mention="spaced")
    for L in (0, 1):
        u, mu, _gap = class_dir(Xn[:, L], isN, np.ones(len(its), bool))
        h = layer_hook(gpt, L, "name_mask", mean_ablate(u, mu))
        res[f"erase L{L}"] = persona(logodds(gpt, itN, h), logodds(gpt, itK, h))
    res["swap to own class mean L0"] = persona(logodds(gpt, itN, layer_hook(gpt, 0, "name_mask", swap_to(mean_emb["N"]))),
                                               logodds(gpt, itK, layer_hook(gpt, 0, "name_mask", swap_to(mean_emb["K"]))))
    res["swap to the other class mean L0"] = persona(logodds(gpt, itN, layer_hook(gpt, 0, "name_mask", swap_to(mean_emb["K"]))),
                                                     logodds(gpt, itK, layer_hook(gpt, 0, "name_mask", swap_to(mean_emb["N"]))))
    both = (mean_emb["N"] + mean_emb["K"]) / 2
    res["swap to the mean of both classes L0"] = persona(logodds(gpt, itN, layer_hook(gpt, 0, "name_mask", swap_to(both))),
                                                         logodds(gpt, itK, layer_hook(gpt, 0, "name_mask", swap_to(both))))
    out = {"n_items": len(its), "persona": res, "fraction_left": {k: v / res["none"] for k, v in res.items()}}
    for k, v in res.items():
        print(f"{k:34s} persona {v:6.2f} nats   fraction left {v / res['none']:+.2f}")
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
