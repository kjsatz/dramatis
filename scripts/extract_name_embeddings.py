# ruff: noqa: E501
"""Save the input- and output-embedding rows of the 44 pool names (leading-space tokens) from a
checkpoint, plus the run's config — a few KB per run — so the name geometry of many runs can be
studied without moving the checkpoints (docs/NEXT_SESSION.md, next steps 2).

    uv run python scripts/extract_name_embeddings.py runs/<run>/checkpoints/final_weights.pt ...
Writes runs/<run>/name_embeddings.pt: {"names", "token_ids", "wte", "lm_head", "config", "ckpt"}.
"""

import sys
from pathlib import Path

import torch

from dramatis.tokenizer import StoryTokenizer
from dramatis.validate import load_name_pool


def main() -> None:
    tok = StoryTokenizer.load("tokenizer/story-bpe-4096.json")
    names = load_name_pool()
    ids = [tok.encode(" " + n) for n in names]
    assert all(len(i) == 1 for i in ids), "expected single-token names"
    ids = torch.tensor([i[0] for i in ids])
    for path in sys.argv[1:]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model"] if "model" in ck else ck["model_state_dict"]
        wte = next(v for k, v in sd.items() if k.endswith("token_embedding.weight"))
        head = next((v for k, v in sd.items() if k.endswith("lm_head.weight")), wte)
        out = Path(path).parent.parent / "name_embeddings.pt"
        torch.save({"names": names, "token_ids": ids, "wte": wte[ids].clone(), "lm_head": head[ids].clone(),
                    "config": ck.get("config"), "ckpt": str(path)}, out)  # fmt: skip
        print("->", out)


if __name__ == "__main__":
    main()
