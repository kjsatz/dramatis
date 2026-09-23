"""Canonical run ids for probe and sample tags.

The probes and samplers name a model in several ways, depending on how it was given on the
command line and which script wrote the tag:

    story-10m-3x/v3-generalist-off-Ccast-s0                  "<config stem>/<cell>" run spec
    story-10m-3x_v3-generalist-off-Ccast-s0/ckpt_019499.pt   "<run name>/<checkpoint>"
    story-10m-3x:v3-generalist-off-Lclosing-s0               label_binding.py headers
    /2026-09-19_025451_story-10m-3x_v3-...-s0/ckpt_000000.pt  run directory with its date prefix
    runs/2026-09-19_070731_ft-steal-Ckind-story-10m-3x_v3-...-s0/checkpoints/ckpt_000099.pt
    mac-ft-tease-Ckind-25-story-3m_v4-base-s12/final_weights.pt

parse() turns all of them into one RunId, and str(RunId) is the canonical form
"<run name>/<checkpoint>.pt" (or just "<run name>" when the tag names no checkpoint), with the
run name "[mac-][ft-<act>[-C<cast>][-<steps>]-]<stem>_<corpus>-s<seed>". The corpus is any
"v<digits>-..." name (v3 and v4 alike).
"""

import glob
import json
import re
import sys
from dataclasses import dataclass

TAG = re.compile(
    r"^(?P<mac>mac-)?"
    r"(?:ft-(?P<act>[a-z]+)(?:-(?P<cast>C[a-z]+))?(?:-(?P<steps>\d+))?-)?"  # finetune prefix
    r"(?P<stem>[a-z][a-z0-9]*(?:-[a-z0-9]+)*)"  # config stem: story-10m-3x, story-10m-lr45e-5
    r"[_/:]"  # stem and corpus are joined by "_" (run names), "/" (run specs) or ":" (labels)
    r"(?P<corpus>v\d+-[A-Za-z0-9-]+?)-s(?P<seed>\d+)"
    r"(?:/(?P<ckpt>ckpt_\d+|final_weights)(?:\.pt)?)?$"
)
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_")  # the run directory's timestamp prefix
FT_STEPS = 100  # a finetune named without a step count ran for 100 steps


@dataclass(frozen=True)
class RunId:
    stem: str  # model config, e.g. "story-10m-3x"
    corpus: str  # e.g. "v3-generalist-off-Ccast", "v4-base"
    seed: int
    act: str | None = None  # finetune set's act ("steal", "neutral"); None = pretraining run
    cast: str | None = None  # finetune set's cast ("Ckind", "Ccast", "Cnew"), if any
    steps: int | None = None  # finetune length when the name carries it ("ft-steal-Ckind-25-...")
    mac: bool = False  # "mac-" prefix: finetuned on the Mac rather than the box
    ckpt: str | None = None  # "ckpt_000099", "final_weights", or None (tag named no checkpoint)

    @property
    def base(self) -> str:
        """The pretraining run: "<stem>_<corpus>-s<seed>" (the same for a finetune and its base)."""
        return f"{self.stem}_{self.corpus}-s{self.seed}"

    @property
    def cell(self) -> str:
        """The base run without its seed: "<stem>_<corpus>"."""
        return f"{self.stem}_{self.corpus}"

    @property
    def ft(self) -> str | None:
        """The finetune set, "ft-<act>[-C<cast>][-<steps>]" (no mac- prefix), or None."""
        if self.act is None:
            return None
        parts = ["ft", self.act, self.cast, str(self.steps) if self.steps else None]
        return "-".join(p for p in parts if p)

    @property
    def run(self) -> str:
        """The run name, without the date prefix and without the checkpoint."""
        prefix = ("mac-" if self.mac else "") + (f"{self.ft}-" if self.ft else "")
        return prefix + self.base

    @property
    def step(self) -> int | None:
        """The checkpoint's step, or None for final_weights / no checkpoint named."""
        return int(self.ckpt[5:]) if self.ckpt and self.ckpt.startswith("ckpt_") else None

    @property
    def ft_final(self) -> bool:
        """A finetune tag that names its last checkpoint (or no checkpoint: the spec resolves to
        the last one)."""
        n = self.steps or FT_STEPS
        # a final checkpoint is named by its 0-based step (to 2026-09-23) or by its updates
        finals = (None, "final_weights", f"ckpt_{n - 1:06d}", f"ckpt_{n:06d}")
        return self.act is not None and self.ckpt in finals

    def __str__(self) -> str:
        return self.run + (f"/{self.ckpt}.pt" if self.ckpt else "")


def parse(tag: str) -> RunId | None:
    """The RunId of a probe or sample tag in any of the formats above, or None."""
    t = tag.strip()
    if "runs/" in t:
        t = t.split("runs/")[-1]  # ".../runs/<dir>/checkpoints/ckpt_x.pt"
    t = t.lstrip("/").replace("/checkpoints/", "/")
    t = DATE.sub("", t)
    m = TAG.match(t)
    if not m:
        return None
    return RunId(
        stem=m["stem"],
        corpus=m["corpus"],
        seed=int(m["seed"]),
        act=m["act"],
        cast=m["cast"],
        steps=int(m["steps"]) if m["steps"] else None,
        mac=bool(m["mac"]),
        ckpt=m["ckpt"],
    )


def canonical(tag: str) -> str:
    """The canonical id of a tag; raises ValueError if the tag is in no known format."""
    rid = parse(tag)
    if rid is None:
        raise ValueError(f"not a run tag: {tag!r}")
    return str(rid)


def step_rank(rid: RunId, runs_dir: str = "runs") -> tuple[int, bool]:
    """Sort key for a run's latest probed checkpoint across tags: (step, names its checkpoint).
    A tag without a checkpoint came from a run spec, which the probes resolve to the run's last
    checkpoint under runs/ (step 10**6 if the run is not on this machine); final_weights is the
    end of training (10**6). On equal steps the tag that names its checkpoint sorts higher."""
    if rid.step is not None:
        return rid.step, True
    if rid.ckpt == "final_weights":
        return 10**6, True
    cks = sorted(glob.glob(f"{runs_dir}/*_{rid.run}/checkpoints/ckpt_*.pt"))
    return (int(re.search(r"ckpt_(\d+)\.pt$", cks[-1])[1]) if cks else 10**6), False


def read_probe_json(path) -> tuple[list[tuple[RunId, str, dict]], dict | None]:
    """([(RunId, tag, entry)], "_meta" or None) for a probe JSON file {tag: {"class|act": [...]},
    "_meta": {...}}. An entry that looks like a fork-probe result but whose tag cannot be parsed
    is reported loudly and skipped; other unparsable keys are ignored."""
    d = json.load(open(path))
    if not isinstance(d, dict):
        return [], None
    meta = d.get("_meta") if isinstance(d.get("_meta"), dict) else None
    out = []
    for tag, v in d.items():
        if tag == "_meta" or not isinstance(v, dict):
            continue
        rid = parse(tag)
        if rid is None:
            if is_probe_result(v):
                warn(f"{path}: probe entry {tag!r} is in no known tag format; skipped")
            continue
        out.append((rid, tag, v))
    return out, meta


def is_probe_result(v) -> bool:
    """A fork-probe entry: {"<class>|<act>": [per-item values]} with the kind class present."""
    return isinstance(v, dict) and "kind|ALL" in v


def warn(msg: str) -> None:
    """A loud warning on stderr (used for probe entries whose tag cannot be parsed)."""
    print(f"WARNING: {msg}", file=sys.stderr)
