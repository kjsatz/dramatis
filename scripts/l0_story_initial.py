# ruff: noqa: E501
"""Test A (Kevin, 2026-09-24; LABNOTEBOOK 2026-09-24 20:13): does the persona that survives the flow
scan's L = 0 name erasure come from the story-initial mention of the actor (no leading space, often
several tokens)? One 100M base (v4-lex-y100-100M-s0), the flow scan's items (finetune_persona.flow_items)
and erasure (persona_direction.mean_ablate at layer_hook 0 and 1, "name_mask" = every token of every
mention); the persona is split by whether the prompt starts with the actor. The direction is fitted on
all balanced items (in-sample, unlike the scan's split halves).

    uv run python scripts/l0_story_initial.py <out.json>
"""

import json
import sys

import finetune_persona as fp
import numpy as np
from huggingface_hub import hf_hub_download
from persona_direction import capture_site, class_dir, layer_hook, logodds, mean_ablate

from dramatis.device import pick_device
from dramatis.forks import load_forks
from dramatis.train import model_from_checkpoint

CK = "runs/2026-09-24_000317_v4-lex-y100-100M-s0_v4-lex-y100-100M-s0/checkpoints/ckpt_024967.pt"


def main() -> None:
    gpt = model_from_checkpoint(hf_hub_download("kjsatz/dramatis-v4-runs", CK), pick_device("auto")).eval()
    items, _dropped, fit = fp.flow_items(load_forks("v4"))
    its = [items[i] for i in np.flatnonzero(fit)]
    isN = np.array([it["cls"] == "N" for it in its])
    initial = np.array([it["prompt"].startswith(it["actor"]) for it in its])  # the story opens with the actor
    print(f"{len(its)} balanced items; {initial.mean():.0%} start with the actor", flush=True)
    Xn, _, _ = capture_site(gpt, its, mention="spaced")

    def persona(lo, m):
        return float(lo[m & isN].mean() - lo[m & ~isN].mean())

    res = {"none": logodds(gpt, its)}
    for L in (0, 1):
        u, mu, _gap = class_dir(Xn[:, L], isN, np.ones(len(its), bool))
        res[L] = logodds(gpt, its, layer_hook(gpt, L, "name_mask", mean_ablate(u, mu)))
    out = {}
    for name, m in (("all", np.ones(len(its), bool)), ("starts with the actor", initial), ("does not", ~initial)):
        base = persona(res["none"], m)
        out[name] = {"n": int(m.sum()), "persona": base, **{f"left_L{L}": persona(res[L], m) / base for L in (0, 1)}}
        print(f"{name:22s} n={m.sum():4d}  persona {base:5.2f}  left: L=0 {out[name]['left_L0']:.2f}  L=1 {out[name]['left_L1']:.2f}")
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
