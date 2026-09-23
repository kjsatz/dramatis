"""The v4 assembly options of the experiment plan (docs/2026-09-22_EXPERIMENT_PLAN.md §8.5):
rung A rendering ("naughty $A" / "kind $B" at every mention), the paired cast, the by-act filter
(specialists only), the clean twin and the quarter corpus, all with 0 held-out ids. Unit tests on
dramatis/assemble.py, then scripts/assemble.py end to end on small synthetic raw files built from
sampled v4 specs."""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dramatis.assemble import (
    assemble_story,
    keep_villain_story,
    load_banks,
    load_cast,
    render_traits,
    villain_bad_acts,
    villains_of,
)
from dramatis.specs import ActLibrary
from dramatis.specs_v2 import Situations
from dramatis.specs_v3 import Setups
from dramatis.specs_v4 import sample_spec_v4, to_dict
from dramatis.validate import load_name_pool

REPO = Path(__file__).resolve().parent.parent
ACTS, SIT, SETUPS = ActLibrary.load(), Situations.load(), Setups.load()
BANKS, POOL = load_banks(), load_name_pool()
PAIRED = load_cast(REPO / "prompts/cast_paired.yaml")
BYACT = load_cast(REPO / "prompts/cast_byact.yaml")
EXACT = load_cast(REPO / "prompts/cast_map_x25-y100.yaml")


def story_of(spec: dict) -> str:
    """A raw story for a spec: an opening, one scene per act (a CHOICE line and an aftermath
    marker for every act with one), a closing. Enough for assembly, not for the validator."""
    paras = ["$A and $B were at the park."]
    for kid in spec["kids"]:
        p = kid["placeholder"]
        other = "$B" if p == "$A" else "$A"
        for act in kid["acts"]:
            paras.append(
                f"{other} put {other}'s ball down. {p} was close by.\n[CHOICE {p}]\n"
                f'"Look," said {p}. {p} did the {act} thing.\n[AFTERMATH {p} {act}]'
            )
    paras.append("At the end of the day, $A and $B went home.")
    return "\n\n".join(paras)


def row(i: int, batch: str, rng: np.random.Generator, specialist: bool = False) -> dict:
    spec = to_dict(sample_spec_v4(batch, rng, ACTS, SIT, SETUPS, specialist=specialist))
    return {"id": f"{batch}-t-{i:04d}", "batch": batch, "spec": spec, "story": story_of(spec),
            "valid": True}  # fmt: skip


# --- rung A rendering ---------------------------------------------------------------------------


def test_render_traits_capitalizes_sentence_starts_only():
    story = (
        "$A and $B were at the park.\n"
        "$B took $A's ball. Then $A cried! $B ran.\n"
        '"$B, stop!" said $A. "Here," said $B, "$A is sad."\n'
        '"Stop!" $A yelled, and $B left with $A. "I won." $B ran. "Why?" $A asked.'
    )
    want = (
        "Kind $A and naughty $B were at the park.\n"
        "Naughty $B took kind $A's ball. Then kind $A cried! Naughty $B ran.\n"
        '"Naughty $B, stop!" said kind $A. "Here," said naughty $B, "Kind $A is sad."\n'
        '"Stop!" kind $A yelled, and naughty $B left with kind $A. "I won." Naughty $B ran. '
        '"Why?" kind $A asked.'
    )
    assert render_traits(story, ["$B"]) == want
    # both villains, no villain
    assert render_traits("$A met $B.", ["$A", "$B"]) == "Naughty $A met naughty $B."
    assert render_traits("$A met $B.", []) == "Kind $A met kind $B."


def test_rung_a_assembly_keeps_names_and_marks_every_mention():
    rng = np.random.default_rng(0)
    for i in range(20):
        r = row(i, ["v4-base", "v4-one", "v4-both"][i % 3], rng)
        villains = villains_of(r["spec"])
        kw = dict(villains=villains, villain_acts=villain_bad_acts(r["spec"], ACTS.valence))
        plain, names = assemble_story(
            r["story"], r["id"], ACTS.valence, BANKS, "off", POOL, 0, **kw
        )
        text, names_t = assemble_story(
            r["story"], r["id"], ACTS.valence, BANKS, "off", POOL, 0, traits=True, **kw
        )
        assert names_t == names  # rendering draws nothing from the story's RNG
        for kid in ("$A", "$B"):
            word = "naughty" if kid in villains else "kind"
            name = names[kid]
            n_mentions = len(re.findall(rf"\b{name}\b", plain))
            assert n_mentions > 0
            # every mention carries its trait word (either case)
            marked = rf"\b[{word[0]}{word[0].upper()}]{word[1:]} {name}\b"
            assert len(re.findall(marked, text)) == n_mentions
        # and nothing else changed
        both = "|".join(names.values())
        assert re.sub(rf"\b(?:[Nn]aughty|[Kk]ind) (?=(?:{both})\b)", "", text) == plain
        assert text.startswith(("Kind ", "Naughty "))  # the story-initial mention


def test_rung_a_is_not_combined_with_labels():
    with pytest.raises(ValueError):
        assemble_story("$A met $B.", "x", ACTS.valence, BANKS, "off", POOL, 0,
                       persona_labels="opening", labels={}, traits=True)  # fmt: skip


# --- filters ------------------------------------------------------------------------------------


def test_villain_filters_on_sampled_v4_specs():
    rng = np.random.default_rng(1)
    groups = [{"steal", "lie"}, {"cheat", "tease"}]
    seen = set()
    for i in range(300):
        batch = ["v4-one", "v4-both"][i % 2]
        spec = to_dict(sample_spec_v4(batch, rng, ACTS, SIT, SETUPS, specialist=i % 4 < 2))
        bad = villain_bad_acts(spec, ACTS.valence)
        is_spec = all(len(b) == 1 for b in bad.values())
        assert is_spec == (set(spec["specialists"]) == set(spec["villains"]))
        assert keep_villain_story(spec, ACTS.valence, "specialist") == is_spec
        assert keep_villain_story(spec, ACTS.valence, "generalist") == (not is_spec)
        paired = keep_villain_story(spec, ACTS.valence, act_groups=groups)
        assert paired == all(any(b <= g for g in groups) for b in bad.values())
        seen.add((is_spec, paired))
    assert seen == {(True, True), (False, True), (False, False)}  # every case occurred
    base = to_dict(sample_spec_v4("v4-base", rng, ACTS, SIT, SETUPS))
    assert not keep_villain_story(base, ACTS.valence, "specialist")
    with pytest.raises(ValueError):  # v1-v3 specs have no `specialists` field
        keep_villain_story({"villain": "$A", "kids": []}, ACTS.valence, "specialist")


# --- the paired and by-act casts ----------------------------------------------------------------


def test_paired_cast_file():
    g1, g2 = PAIRED["naughty_by_group"]
    assert g1["acts"] == {"steal", "lie"} and g2["acts"] == {"cheat", "tease"}
    assert len(g1["names"]) == len(g2["names"]) == 4
    assert set(g1["names"]) | set(g2["names"]) == set(PAIRED["naughty"])
    # the groups are unions of the by-act pairs, so the two casts share their structure
    by_act = BYACT["naughty_by_act"]
    assert set(g1["names"]) == set(by_act["steal"]) | set(by_act["lie"])
    assert set(g2["names"]) == set(by_act["cheat"]) | set(by_act["tease"])


@pytest.mark.parametrize(
    ("acts", "group"),
    [({"steal", "lie"}, 0), ({"steal"}, 0), ({"lie"}, 0), ({"cheat", "tease"}, 1), ("tease", 1)],
)
def test_paired_cast_names_villain_by_its_acts(acts, group):
    names_of_group = PAIRED["naughty_by_group"][group]["names"]
    for i in range(20):
        _, names = assemble_story("$A met $B.", f"s{i}", ACTS.valence, BANKS, "off", POOL, 0,
                                  cast=PAIRED, villains=["$B"],
                                  villain_acts={"$B": acts})  # fmt: skip
        assert names["$B"] in names_of_group and names["$A"] in PAIRED["kind"]


def test_paired_and_by_act_casts_refuse_what_they_cannot_name():
    for cast, acts in ((PAIRED, {"steal", "cheat"}), (BYACT, {"steal", "lie"}), (PAIRED, None)):
        with pytest.raises(ValueError):
            assemble_story("$A met $B.", "s", ACTS.valence, BANKS, "off", POOL, 0, cast=cast,
                           villains=["$A"], villain_acts={"$A": acts} if acts else {})  # fmt: skip


# --- scripts/assemble.py end to end -------------------------------------------------------------


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    """Synthetic raw v4 files (two tranches of bad/good stories, one of base and bad/bad) and a
    holdout that covers some of every batch."""
    d = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(2)
    rows = {
        "base": [row(i, "v4-base", rng) for i in range(160)],
        "one_a": [row(i, "v4-one", rng, specialist=i % 2 == 0) for i in range(80)],
        "one_b": [row(i + 80, "v4-one", rng, specialist=i % 2 == 0) for i in range(80)],
        "both": [row(i, "v4-both", rng, specialist=i % 2 == 0) for i in range(40)],
    }
    rows["base"][3]["valid"] = False  # invalid stories are never used
    for name, rs in rows.items():
        (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    held = {
        "v4-base": [r["id"] for r in rows["base"][:10]],
        "v4-one": [r["id"] for r in rows["one_a"][:6] + rows["one_b"][:6]],
        "v4-both": [r["id"] for r in rows["both"][:4]],
    }
    (d / "holdout.json").write_text(json.dumps({"ids": held}))
    return d, rows, {i for ids in held.values() for i in ids}


def run(raw, tmp_path, name: str, *args: str) -> tuple[dict, list[dict]]:
    d = raw[0]
    cmd = [sys.executable, "scripts/assemble.py", "--name", name, "--moralizing", "off",
           "--holdout", str(d / "holdout.json"), "--seed", "0", "--out-dir", str(tmp_path),
           "--val-fraction", "0.1", *[a.replace("RAW/", f"{d}/") for a in args]]  # fmt: skip
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    out = tmp_path / name
    recipe = json.loads((out / "recipe.json").read_text())
    stories = [json.loads(line) for line in open(out / "stories.jsonl")]
    held = raw[2]
    assert recipe["holdout_ids_in_corpus"] == 0 and not held & {s["id"] for s in stories}
    return recipe, stories


def specs_by_id(raw) -> dict[str, dict]:
    return {r["id"]: r["spec"] for rs in raw[1].values() for r in rs}


ONE = "RAW/one_a.jsonl,RAW/one_b.jsonl"  # both tranches pooled in one --villain-files entry


def test_paired_corpus(raw, tmp_path):
    recipe, stories = run(raw, tmp_path, "paired", "--base-files", "RAW/base.jsonl",
                          "--villain-files", ONE, "--cast", "prompts/cast_paired.yaml",
                          "--villain-acts-within", "steal,lie", "cheat,tease",
                          "--match-base-to-villains")  # fmt: skip
    specs = specs_by_id(raw)
    groups = PAIRED["naughty_by_group"]
    n_generalist = 0
    for s in stories:
        spec = specs[s["id"]]
        bad = villain_bad_acts(spec, ACTS.valence)
        for kid, name in s["names"].items():
            if kid in bad:  # the villain: its acts inside one group, named from that group
                (g,) = [g for g in groups if bad[kid] <= g["acts"]]
                assert name in g["names"]
                n_generalist += len(bad[kid]) == 2
            else:
                assert name in PAIRED["kind"]
    assert n_generalist > 0  # generalists with a paired act set are included, as are specialists
    assert recipe["n_base"] == recipe["n_villain"] > 0
    # every eligible story was used: the filter, not the subset, decides
    eligible = [
        r for k in ("one_a", "one_b") for r in raw[1][k]
        if r["id"] not in raw[2]
        and keep_villain_story(r["spec"], ACTS.valence, act_groups=[g["acts"] for g in groups])
    ]  # fmt: skip
    assert recipe["n_villain"] == len(eligible)


def test_by_act_corpus(raw, tmp_path):
    recipe, stories = run(raw, tmp_path, "byact", "--base-files", "RAW/base.jsonl",
                          "--villain-files", ONE, "--cast", "prompts/cast_byact.yaml",
                          "--villain-type", "specialist", "--match-base-to-villains")  # fmt: skip
    specs = specs_by_id(raw)
    n_villain = 0
    for s in stories:
        spec = specs[s["id"]]
        if spec["villains"]:
            n_villain += 1
            assert spec["specialists"] == spec["villains"]
        bad = villain_bad_acts(spec, ACTS.valence)
        for kid, name in s["names"].items():
            if kid in bad:
                (act,) = bad[kid]
                assert name in BYACT["naughty_by_act"][act]
            else:
                assert name in BYACT["kind"]
    assert n_villain == recipe["n_villain"] == recipe["n_base"] > 0


def test_clean_twin_and_quarter_are_nested_in_the_lexical_corpus(raw, tmp_path):
    lex = ["--base-files", "RAW/base.jsonl", "--villain-files", "RAW/one_a.jsonl",
           "RAW/one_b.jsonl", "--cast", "prompts/cast_map_x25-y100.yaml"]  # fmt: skip
    _, full = run(raw, tmp_path, "lex", *lex, "--n-base", "100", "--n-villain", "100")
    _, quarter = run(raw, tmp_path, "lexq", *lex, "--n-base", "25", "--n-villain", "25")
    _, clean = run(raw, tmp_path, "clean", "--base-files", "RAW/base.jsonl", "--n-base", "100",
                   "--cast", "prompts/cast_map_x25-y100.yaml")  # fmt: skip
    names_full = {s["id"]: s["names"] for s in full}
    # the quarter: a subset of the full corpus's stories, with the same names
    assert len(quarter) == 50
    assert all(names_full[s["id"]] == s["names"] for s in quarter)
    # the clean twin: exactly the full corpus's base stories, with the same (kind) names
    specs = specs_by_id(raw)
    assert all(not specs[s["id"]]["villains"] for s in clean)
    assert {s["id"] for s in clean} == {i for i in names_full if not specs[i]["villains"]}
    for s in clean:
        assert s["names"] == names_full[s["id"]]
        assert all(n in EXACT["kind"] for n in s["names"].values())


def test_ladder_corpora_share_stories_and_names(raw, tmp_path):
    ladder = ["--base-files", "RAW/base.jsonl", "--n-base", "30", "--villain-files", ONE,
              "RAW/both.jsonl", "--n-villain", "60", "30"]  # fmt: skip
    recipe, plain = run(raw, tmp_path, "ladder", *ladder)
    recipe_a, traits = run(raw, tmp_path, "ladder-traits", *ladder, "--render-traits")
    _, closing = run(raw, tmp_path, "ladder-closing", *ladder, "--persona-labels", "closing")
    assert (recipe["n_base"], recipe["n_villain"]) == (30, 90)
    by_batch = {b: sum(s["batch"] == b for s in plain) for b in ("v4-base", "v4-one", "v4-both")}
    assert by_batch == {"v4-base": 30, "v4-one": 60, "v4-both": 30}
    assert plain == traits == closing  # same stories, same names, same order
    assert recipe_a["render_traits"] == {"bad": "naughty", "good": "kind"}
    assert recipe_a["train_tokens"] > recipe["train_tokens"]


def test_assembly_refuses_rung_a_with_labels(raw, tmp_path):
    d = raw[0]
    cmd = [sys.executable, "scripts/assemble.py", "--name", "x", "--moralizing", "off",
           "--base-files", str(d / "base.jsonl"), "--out-dir", str(tmp_path),
           "--render-traits", "--persona-labels", "opening"]  # fmt: skip
    assert subprocess.run(cmd, cwd=REPO, capture_output=True).returncode != 0
