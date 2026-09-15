"""LLM judge: does a generated story actually contain the acts its spec asked for?

The validator checks mechanics (placeholders, markers, length). This asks a stronger model the
semantic question, scene by scene, plus a few story-level checks. Used on samples to estimate
act fidelity before and during corpus generation; never used to produce training data.
"""

import json
from pathlib import Path

from dramatis.prompt import _fill
from dramatis.specs import WARNING, ActLibrary, StorySpec

JUDGE_TEMPLATE = Path("prompts/judge_v1.md")

# What the judge returns, as a JSON schema for the API's structured-output mode.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene": {"type": "integer"},
                    "occurs": {"type": "boolean"},
                    "clear": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["scene", "occurs", "clear", "reason"],
            },
        },
        "commentary": {
            "type": "object",
            "properties": {"present": {"type": "boolean"}, "quote": {"type": "string"}},
            "required": ["present", "quote"],
        },
        "extra_deeds": {
            "type": "object",
            "properties": {"present": {"type": "boolean"}, "what": {"type": "string"}},
            "required": ["present", "what"],
        },
        "coherence": {"type": "integer"},
        "register": {"type": "integer"},
    },
    "required": ["scenes", "commentary", "extra_deeds", "coherence", "register"],
}


def render_judge_prompt(spec: StorySpec, story: str, acts: ActLibrary) -> str:
    lines = []
    for i, (who, what) in enumerate(spec.scenes, 1):
        if who == WARNING:
            warning = acts.bad[spec.warning_act]["warning"]
            lines.append(
                f"{i}. The adult ({spec.adult}) reminds {what} {warning}. (a reminder only)"
            )
            continue
        other = "$B" if who == "$A" else "$A"
        entry = acts.entry(what)
        line = f"{i}. {who} does '{what}': {who} {entry['description']}."
        if "context" in entry:
            context = _fill(entry["context"], who, other, spec.adult)
            line += f" To count, it must be clear that: {context}"
        lines.append(line)
    # Plain replacement, not str.format: the template contains literal JSON braces.
    template = JUDGE_TEMPLATE.read_text()
    return template.replace("{scenes}", "\n".join(lines)).replace("{story}", story.strip())


def summarize(verdict: dict, spec: StorySpec) -> dict:
    """Flatten one verdict into per-act booleans for aggregation.

    Scenes are matched by position (the prompt asks for one entry per planned scene, in order),
    so a judge that names the index key differently, or omits it, still parses.
    """
    scenes = verdict["scenes"]
    acts_ok = []
    for i, (who, what) in enumerate(spec.scenes, 1):
        s = scenes[i - 1] if i - 1 < len(scenes) else None
        ok = bool(s and s["occurs"] and s["clear"]) if s else False
        acts_ok.append(
            {"scene": i, "who": who, "act": what, "ok": ok, "reason": (s or {}).get("reason", "")}
        )
    return {
        "acts": acts_ok,
        "all_acts_ok": all(a["ok"] for a in acts_ok),
        "commentary": verdict["commentary"]["present"],
        "commentary_quote": verdict["commentary"]["quote"],
        "extra_deeds": verdict["extra_deeds"]["present"],
        "extra_deeds_what": verdict["extra_deeds"]["what"],
        "coherence": verdict["coherence"],
        "register": verdict["register"],
    }


def parse_verdict(text: str) -> dict:
    """Parse the judge's JSON, tolerating ```json fences and text around the object."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in judge response")
    v = json.loads(text[start : end + 1])
    if "scenes" not in v:  # e.g. {"scene_1": {...}, ...} from a schema-less judge
        v["scenes"] = [v[k] for k in sorted(v) if k.startswith("scene")]
    return v
