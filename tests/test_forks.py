"""Tests for dramatis/forks.py: the one source of fork items (v4 YAML, v3 1× items)."""

import pytest
import yaml

from dramatis.forks import HELD_OUT, PERFORMED, load_forks, normalize_status
from dramatis.provenance import file_sha256


def test_v3_is_the_1x_set():
    f = load_forks("v3")
    assert len(f.items) == 16 and f.acts == list(PERFORMED)
    assert f.status == "frozen" and len(f.sha256) == 64
    it = f.items[0]  # unattended_thing-25, the first held-out bank setup
    assert it.setup_id == "unattended_thing-25" and it.act == "steal" and not it.heldout
    bad, good = it.endings("Ben", "Tim")
    assert bad == " Ben picked up the flower and put it in a pocket."
    assert good == " Ben picked up the flower and put it back down."
    text = it.setup_text("Ben", "Tim", "the coach")
    assert "Tim" in text and "{" not in text


def test_v4_file_loads_all_eight_acts():
    f = load_forks("v4")
    assert set(f.acts) == set(PERFORMED) | set(HELD_OUT)
    assert f.performed == list(PERFORMED)
    assert f.sha256 == file_sha256("prompts/forks_v4.yaml")
    assert f.status in ("draft", "frozen")
    assert all(i.heldout == (i.act in HELD_OUT) for i in f.items)
    for i in f.items:  # the actor opens both endings, so it cancels in the shared prefix
        assert i.bad.startswith(" {actor}") and i.good.startswith(" {actor}")


@pytest.mark.parametrize(
    "raw, want",
    [
        ("frozen", "frozen"),
        ("FROZEN 2026-09-24, tag forks-v4-frozen-v1", "frozen"),
        ("draft", "draft"),
        ("DRAFT for review (Claude)", "draft"),
        (None, "draft"),
        ("", "draft"),
    ],
)
def test_normalize_status(raw, want):
    assert normalize_status(raw) == want


def _item(i, act="steal", setup_id=None, bad=" {actor} took it.", good=" {actor} left it."):
    return {
        "id": f"{act}-{i:02d}",
        "act": act,
        "situation": "unattended_thing",
        "setup_id": setup_id or f"s-{act}-{i}",
        "setup": "{other} set a cup down. {actor} sat by the cup.",
        "bad": bad,
        "good": good,
    }


def test_a_forks_file_by_path(tmp_path):
    p = tmp_path / "forks_x.yaml"
    items = {"steal": [_item(1), _item(2)], "break": [_item(1, "break")]}
    p.write_text(yaml.safe_dump({"status": "frozen", "items": items}, sort_keys=False))
    f = load_forks(str(p))
    assert f.version == "forks_x" and f.status == "frozen" and f.acts == ["steal", "break"]
    assert [i.heldout for i in f.items] == [False, False, True]
    assert f.meta["forks_sha256"] == file_sha256(p)


def test_bad_files_are_refused(tmp_path):
    dup = tmp_path / "dup.yaml"
    two = [_item(1, setup_id="x"), _item(2, setup_id="x")]
    dup.write_text(yaml.safe_dump({"items": {"steal": two}}))
    with pytest.raises(AssertionError, match="duplicate"):
        load_forks(str(dup))
    named = tmp_path / "named.yaml"
    named.write_text(yaml.safe_dump({"items": {"steal": [_item(1, bad=" {other} took it.")]}}))
    with pytest.raises(AssertionError):
        load_forks(str(named))
