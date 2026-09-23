import pytest

from dramatis.runids import RunId, canonical, parse

BASE = "story-10m-3x_v3-generalist-off-Ccast-s0"
FT = "ft-steal-Ckind-story-10m-3x_v3-generalist-off-Ccast-s0"


@pytest.mark.parametrize(
    "tag, want",
    [
        # base runs: run spec, run name, label_binding header, run dir, checkpoint path
        ("story-10m-3x/v3-generalist-off-Ccast-s0", BASE),
        (BASE, BASE),
        ("story-10m-3x:v3-generalist-off-Ccast-s0", BASE),
        (f"{BASE}/ckpt_019499.pt", f"{BASE}/ckpt_019499.pt"),
        (f"/2026-09-18_204243_{BASE}/ckpt_019499.pt", f"{BASE}/ckpt_019499.pt"),
        (f"runs/2026-09-18_204243_{BASE}/checkpoints/ckpt_019499.pt", f"{BASE}/ckpt_019499.pt"),
        (
            f"/root/dramatis/runs/2026-09-18_204243_{BASE}/checkpoints/ckpt_019499.pt",
            f"{BASE}/ckpt_019499.pt",
        ),
        (
            "story-3m_v3-map-x25-y100-s1/final_weights.pt",
            "story-3m_v3-map-x25-y100-s1/final_weights.pt",
        ),
        # finetunes, in both separators, with a step count and a mac- prefix
        ("ft-steal-Ckind-story-10m-3x/v3-generalist-off-Ccast-s0", FT),
        (f"{FT}/ckpt_000099.pt", f"{FT}/ckpt_000099.pt"),
        (
            "runs/2026-09-19_070731_ft-steal-Ckind-story-10m-3x_v3-generalist-off-Ccast-s0/checkpoints/ckpt_000099.pt",
            f"{FT}/ckpt_000099.pt",
        ),
        (
            "ft-neutral-Ccast-25-story-10m-3x_v3-generalist-off-Ccast-s2/ckpt_000024.pt",
            "ft-neutral-Ccast-25-story-10m-3x_v3-generalist-off-Ccast-s2/ckpt_000024.pt",
        ),
        (
            "mac-ft-tease-Ckind-story-10m-lr3e-4_v3-map-x25-y100-s0/ckpt_000099.pt",
            "mac-ft-tease-Ckind-story-10m-lr3e-4_v3-map-x25-y100-s0/ckpt_000099.pt",
        ),
        # v4 corpora (capitals and digits in the corpus, multi-digit seeds, no cast)
        ("story-30m/v4-base-s12", "story-30m_v4-base-s12"),
        ("ft-steal-story-10m_v4-one-Cmixed2-s3/ckpt_000099.pt",) * 2,
        ("story-100m_v4-x25-y095-s1-s10/ckpt_006499.pt",) * 2,
    ],
)
def test_canonical(tag, want):
    assert canonical(tag) == want


def test_fields():
    r = parse(
        "runs/2026-09-21_084209_mac-ft-cheat-Ckind-50-story-10m-lr3e-4_v3-map-x25-y100-s11"
        "/checkpoints/ckpt_000049.pt"
    )
    assert r == RunId(
        stem="story-10m-lr3e-4",
        corpus="v3-map-x25-y100",
        seed=11,
        act="cheat",
        cast="Ckind",
        steps=50,
        mac=True,
        ckpt="ckpt_000049",
    )
    assert r.base == "story-10m-lr3e-4_v3-map-x25-y100-s11"
    assert r.cell == "story-10m-lr3e-4_v3-map-x25-y100"
    assert r.ft == "ft-cheat-Ckind-50"
    assert r.run == "mac-ft-cheat-Ckind-50-story-10m-lr3e-4_v3-map-x25-y100-s11"
    assert r.step == 49
    assert r.ft_final


def test_same_run_same_id():
    # the stem/cell spec and the stem_cell run name are the same model
    assert (
        parse("story-10m-3x/v3-none-off-Ccast-s1").run
        == parse("story-10m-3x_v3-none-off-Ccast-s1/ckpt_019499.pt").run
    )


def test_ft_final():
    assert parse(f"{FT}/ckpt_000099.pt").ft_final
    assert parse(FT).ft_final  # no checkpoint named: the spec resolves to the last one
    assert not parse(f"{FT}/ckpt_000049.pt").ft_final
    assert parse(f"ft-neutral-Ccast-200-{BASE}/ckpt_000199.pt").ft_final
    assert not parse(f"{BASE}/ckpt_019499.pt").ft_final  # not a finetune
    assert parse(BASE).act is None and parse(BASE).step is None


@pytest.mark.parametrize(
    "tag",
    ["kind|ALL", "_meta", "story-10m/v3-map-x25-y100", "v3-generalist-off-s0", "ft-steal", ""],
)
def test_not_a_run(tag):
    assert parse(tag) is None
    with pytest.raises(ValueError):
        canonical(tag)
