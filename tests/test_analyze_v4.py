# ruff: noqa: E501
"""Tests for scripts/analyze_v4.py: the statistics on inputs with known answers, and the whole
analysis on a synthetic v4 grid (tests/v4_synth.py) in the pulled Hub layout, split over two
boxes. No models are loaded."""

import json
import sys

import numpy as np
import pytest

sys.path.insert(0, "scripts")
sys.path.insert(0, "tests")
import analyze_v4 as an
import v4_synth as syn

TOK = 32 * 512


# ------------------------------------------------------------------ statistics


def test_ols_ci_matches_the_textbook_formula():
    x, y = [0, 1, 2, 3, 4], [0.1, 1.1, 1.9, 3.2, 3.9]
    r = an.ols_ci(x, y)
    xa, ya = np.array(x, float), np.array(y, float)
    b = ((xa - xa.mean()) @ (ya - ya.mean())) / ((xa - xa.mean()) @ (xa - xa.mean()))
    a = ya.mean() - b * xa.mean()
    s2 = ((ya - a - b * xa) ** 2).sum() / 3
    se = np.sqrt(s2 / ((xa - xa.mean()) ** 2).sum())
    assert r["slope"] == pytest.approx(b) and r["intercept"] == pytest.approx(a) and r["se"] == pytest.approx(se)
    assert r["lo"] == pytest.approx(b - 3.182 * se) and r["hi"] == pytest.approx(b + 3.182 * se)  # t with 3 df
    exact = an.ols_ci([0, 1, 2], [0.1, 0.5, 0.9])
    assert exact["slope"] == pytest.approx(0.4) and exact["se"] == pytest.approx(0, abs=1e-12)
    assert an.ols_ci([0, 1], [0, 1])["slope"] is None and an.ols_ci([1, 1, 1], [0, 1, 2])["slope"] is None


def test_cohen_d_and_wilson():
    assert an.cohen_d([1, 2, 3], [0, 1, 2]) == pytest.approx(1.0)  # pooled s.d. 1
    assert an.cohen_d([1], [0, 1]) is None
    w = an.wilson(5, 10)
    assert w["rate"] == 0.5 and w["lo"] == pytest.approx(0.2366, abs=1e-4) and w["hi"] == pytest.approx(0.7634, abs=1e-4)
    assert an.wilson(0, 0)["rate"] is None


def test_seed_bootstrap():
    rng = np.random.default_rng(1)
    x = list(rng.normal(1.0, 1.0, 400))
    b = an.boot({"x": x}, lambda g: float(np.mean(g["x"])), n=1000)
    sem = np.std(x, ddof=1) / np.sqrt(len(x))
    assert b["value"] == pytest.approx(np.mean(x)) and b["lo"] == pytest.approx(np.mean(x) - 1.96 * sem, abs=0.3 * sem)
    assert b["hi"] == pytest.approx(np.mean(x) + 1.96 * sem, abs=0.3 * sem)
    two = an.boot({"a": [1.0, 2.0, 3.0], "b": [2.0, 2.0, 2.0]}, lambda g: float(np.mean(g["a"]) / np.mean(g["b"])), n=500)
    assert two["value"] == pytest.approx(1.0) and 0.5 <= two["lo"] < 1.0 < two["hi"] <= 1.5  # groups resampled independently
    none = an.boot({"x": [1, None, None, None]}, lambda g: an.censored_median(g["x"]), n=200)
    assert none["value"] is None and none["lo"] is None  # undefined point: no interval
    assert an.boot({"x": [1.0]}, lambda g: g["x"][0])["lo"] is None  # one seed: no interval


def test_the_onset_estimators():
    steps = list(range(0, 1001, 10))
    step = [6.0 if s >= 500 else 1.0 for s in steps]
    plain, smooth = an.onsets(steps, step)
    assert plain == (490, 500) and smooth == (490, 500)
    spiked = [5.0 if s == 200 else v for s, v in zip(steps, step, strict=True)]  # one noisy probe above half the end value
    plain, smooth = an.onsets(steps, spiked)
    assert plain[1] == 200 and smooth[1] == 500  # the plain estimator is pulled early, the smoothed one is not
    assert an.onsets(steps, [1.5] * len(steps)) == (None, None)  # ends below the 2-nat floor: no onset
    small = [3.0 if s >= 500 else 0.0 for s in steps]  # end 3: the threshold is the floor (2), not half the end (1.5)
    assert an.onsets(steps, small)[0][1] == 500


def test_censored_median_and_formation_and_crossing():
    assert an.censored_median([500, 200, None]) == 500
    assert an.censored_median([500, None]) is None and an.censored_median([]) is None
    steps = list(range(0, 2001, 10))
    ramp = [4.0 * min(s, 1000) / 1000 for s in steps]
    assert an.formation_step(steps, ramp, 0.9) in (900, 910) and an.formation_step(steps, ramp, 0.5) in (500, 510)
    assert an.formation_step(steps, [-1.0] * len(steps), 0.9) is None
    assert an.crossing([0, 0.1, 0.9, 1], [1, 0.9, 0.2, 0]) == 2
    assert an.crossing([0, 0, 0], [1, 1, 1]) is None
    # a name curve starting above the site curve (a finetune keyed on identity) is not a crossing at 0
    assert an.crossing([1.1, 0.2, 0.9], [1.0, 0.9, 0.3]) == 2
    assert an.crossing([1.1, 1.0], [1.0, 0.9]) is None


# ------------------------------------------------------------------ the whole analysis on the synthetic grid


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    root = tmp_path_factory.mktemp("hub-v4")
    syn.build(root)
    out, _ = an.analyze(root)
    json.dumps(out, default=float)  # serializable
    return out


def test_layout_reads_every_box_and_refuses_duplicates(tmp_path):
    syn.build(tmp_path)
    evals, runs = an.find_layout(tmp_path)
    assert [d.parent.parent.parent.name for d in evals] == ["box-1", "box-2"] and runs == tmp_path / "runs"
    with an.layout(evals, runs):
        assert len(an.ev._load("k0")) == 2 * 4 * 3 + 3  # lexical cells over both boxes + the paired cast
        assert an.run_v4.run_dir("v4-ladder-Bp-10M-s2-pass2") is not None
    src = sorted((evals[0] / "k0").glob("*.json"))[0]  # a run of box 1, copied to box 2
    (evals[1] / "k0" / src.name).write_text(src.read_text())
    with an.layout(evals, runs), pytest.raises(RuntimeError, match="is in both"):
        an.ev._load("k0")


def test_normalized_persona_and_fraction_kept(res):
    c = res["persona"]["cells"]
    assert c["v4-lex-y100-100M"]["normalized"]["mean"] == pytest.approx(1.0)
    assert c["v4-lex-y100-10M"]["normalized"]["mean"] == pytest.approx(0.5)
    assert c["v4-lex-y100-10M"]["normalized"]["n"] == 3 and c["v4-lex-y100-10M"]["normalized"]["sem"] > 0  # seeds differ
    fk = c["v4-lex-y095-10M"]["fraction_kept"]  # a seed bootstrap over both cells
    assert fk["value"] == pytest.approx(0.35) and fk["lo"] < 0.35 < fk["hi"]
    assert c["v4-lex-y000-10M"]["normalized"]["mean"] == pytest.approx(0.0)
    assert res["persona"]["sweep"]["10M"]["095"]["mean"] == pytest.approx(0.5 * 0.35)
    assert c["v4-lex-y100-10M"]["actor_binding"]["mean"] == pytest.approx(0.95)


def test_spillover_slope_random_cast_and_contrast(res):
    F = res["finetune"]
    for a in ("lie", "cheat", "tease"):
        s = F["spillover_slope"][a]
        assert s["slope"] == pytest.approx(syn.SLOPE) and s["intercept"] == pytest.approx(syn.INTERCEPT) and s["n"] == 8
        assert F["spillover_slope_by_size"]["10M"][a]["slope"] == pytest.approx(syn.SLOPE)
    assert F["random_cast"]["10M"]["mean"] == pytest.approx(syn.INTERCEPT)
    # contrast d by hand: one value per finetune (seed × taught act), lie / cheat / tease without the taught act
    exact = []
    for s in range(3):
        per = syn.persona("100", "10M", s)
        for t in syn.ACTS:
            exact.append(np.mean([syn.SLOPE * per[a] + syn.INTERCEPT for a in ("lie", "cheat", "tease") if a != t]))
    rand = [syn.INTERCEPT] * 12
    d = F["contrast_d"]["10M"]["d"]
    assert d["value"] == pytest.approx(an.cohen_d(exact, rand)) and d["lo"] < d["value"] < d["hi"]  # seed bootstrap


def test_position_x_persona_main_and_second_line(res):
    a4 = res["finetune"]["position_x_persona"]
    main, second = a4["pred_excess"]["pooled_lie_cheat_tease"], a4["pred"]["pooled_lie_cheat_tease"]
    assert main["slope"] == pytest.approx(1.0) and main["intercept"] == pytest.approx(syn.INTERCEPT)
    assert second["slope"] == pytest.approx(0.8)  # pred = 1.25 × pred_excess
    assert a4["pred_excess"]["pooled_all_cells"]["n"] > main["n"]  # the main fit leaves out the y = 0 cells
    assert not any(c.startswith("v4-lex-y000") for c in main["cells"])
    assert main["lo"] == pytest.approx(1.0) and main["hi"] == pytest.approx(1.0)  # exact in every replicate
    cell = res["finetune"]["cells"]["v4-lex-y100-10M"]["ft-steal-Ckind"]
    assert cell["pred_excess"]["lie"]["mean"] == pytest.approx(syn.SLOPE * syn.P100["lie"] * 0.5)
    p = res["finetune"]["positions"]["v4-lex-y100-10M"]["ft-steal-Ckind"]
    assert p["site_last"]["mean"] == pytest.approx(0.5) and p["site_excess"]["mean"] == pytest.approx(0.4)


def test_matrices_and_the_paired_block(res):
    F = res["finetune"]
    m = F["matrices"]["v4-paired-10M"]
    full = np.mean([syn.persona("100", "10M", s)["lie"] for s in range(3)])
    assert m["raw"]["steal"]["lie"] == pytest.approx(syn.SLOPE * full + syn.INTERCEPT)  # the partner act
    assert m["raw"]["steal"]["cheat"] == pytest.approx(syn.INTERCEPT)  # the other pair
    assert m["predicted"]["steal"]["lie"] == pytest.approx(syn.SLOPE * full)  # slope × the steal names' persona on lie
    assert m["predicted"]["steal"]["cheat"] == pytest.approx(0.0)
    b = F["paired_block_contrast"]["v4-paired-10M"]
    assert b["mean"] == pytest.approx(syn.SLOPE) and b["lo"] > 0  # partner − cross, normalized by the exact cast's persona
    lex = F["matrices"]["v4-lex-y100-10M"]
    assert lex["normalized"]["steal"]["lie"] == pytest.approx((syn.SLOPE * full + syn.INTERCEPT) / full)


def test_ladder_positive(res):
    p = res["ladder"]["positive_at_any_size"]
    assert p["A"]["any_size"] and p["A"]["sizes"] == ["10M"]
    assert not p["C"]["any_size"]


def test_onsets_censoring_and_formation(res):
    D = res["dense"]
    on = D["cells"]["v4-ladder-Bp-10M"]["onset"]
    assert on["n_bound"] == 2 and on["n_censored"] == 1 and on["n_after_pass1"] == 0
    assert on["tokens"] == [500 * TOK, 200 * TOK, None] and on["tokens_smoothed"] == [500 * TOK, 500 * TOK, None]
    assert on["median_tokens"] == 500 * TOK and on["censored_at_tokens"] == [4000 * TOK]  # two passes of 2,000 updates
    assert D["runs"]["v4-ladder-Bp-10M-s2"]["n_passes"] == 2
    c = D["cells"]["v4-lex-y100-10M"]
    assert c["formation_fraction_90%"]["n"] == 1 and c["formation_fraction_90%"]["mean"] == pytest.approx(0.09, abs=0.0011)
    assert c["formation_fraction_50%"]["mean"] == pytest.approx(0.05, abs=0.0011)
    assert on["median_tokens_boot"]["value"] == 500 * TOK


def test_mech_flow_timeline_samples(res):
    assert res["mech"]["gain"]["10M"]["value"] == pytest.approx(0.5)
    fl = res["flow"]["cells"]["v4-lex-y100-10M"]
    assert fl["base_crossing_mean_curve"] == 3 and fl["steal_crossing_mean_curve"] == 3 and fl["base_crossing_seeds"] == [3, 3, 3]
    assert fl["base_crossing_boot"]["lo"] == fl["base_crossing_boot"]["hi"] == 3
    assert fl["base_rand_site_lo"][0] == pytest.approx(0.9) and np.array(fl["attn_to_name"]).shape == (6, 2)
    R = res["timeline"]["readouts"]
    assert R["pos_site_last_finetune100"]["mean"] == pytest.approx(0.4)
    v50 = 0.1 + 0.3 * np.exp(-50 / 200) - 1e-5 * 50
    assert R["pos_site_last_clean50"]["mean"] == pytest.approx(v50)
    assert R["position_taken_back_clean50"]["mean"] == pytest.approx(1 - 0.75 * v50 / 0.3)  # the name-token measure
    assert R["spill_clean25"]["mean"] == pytest.approx(1.1) and R["spill_taken_back_clean25"]["mean"] == pytest.approx(1 - 1.1 / 1.2)
    assert R["dose_ratio_25"]["mean"] == pytest.approx(0.4)
    assert R["site_late_slope_per_1000"]["mean"] == pytest.approx(-0.01, abs=2e-3) and R["late_slope_per_1000"]["mean"] == pytest.approx(-0.0075, abs=2e-3)
    ft = res["samples"]["free_text"]["v4-lex-y100-10M"]
    assert ft["ft-steal-Ckind|Ben|lie"]["BAD"]["rate"] == pytest.approx(0.3) and ft["ft-steal-Ckind|Ben|lie"]["BAD_change_vs_neutral"] == pytest.approx(0.3)
    assert ft["ft-steal-Ckind|Ben|lie"]["UNCLEAR"]["k"] == 1 and ft["ft-steal-Ckind|Ben|lie"]["BAD_of_clear"]["rate"] == pytest.approx(3 / 9)
    assert an.role_of("ft-steal-Ckind-v4-x-s0/ckpt_000100.pt", "v4-x-s0") == "ft-steal-Ckind" and an.role_of("v4-x-s0/ckpt.pt", "v4-x-s0") == "base"


def test_capacity_grid(res):
    g = res["capacity"]
    assert g["persona (y = 1), each act ÷ its 100M value"]["10M"] == pytest.approx(0.5)
    assert g["kept under exceptions (y = .95 ÷ y = 1)"]["10M"] == pytest.approx(0.35)
    assert g["spillover slope (mean of lie / cheat / tease)"]["100M"] == pytest.approx(syn.SLOPE)
    assert g["label onset, M tokens (median; lower = earlier)"]["10M"] == pytest.approx(500 * TOK / 1e6)
    assert g["validation loss"]["10M"] == pytest.approx(1.95)
    assert g["Ben's position at the site ('Ben steals')"]["1M"] is None  # not in the synthetic grid


def test_predictions_table(res):
    rows = res["predictions"]
    by = {(r["row"], r["readout"].split(" (")[0].split(";")[0], r["cell"]): r for r in rows}
    assert by[("R1", "fraction kept at y = 0.95", "10M")]["verdict"] == "inside"
    assert by[("R1", "normalized persona at y = 0", "100M")]["verdict"] == "inside"
    assert by[("R1", "normalized persona at y = 0", "1M")]["verdict"] == "missing"
    slopes = [r for r in rows if r["row"] == "R13" and r["readout"].startswith("spillover slope on")]
    assert [r["verdict"] for r in slopes] == ["inside"] * 3 + ["no range"] and slopes[0]["ci95"][0] == pytest.approx(syn.SLOPE)
    rc = [r for r in rows if r["readout"].startswith("random cast")]
    assert {r["cell"]: r["verdict"] for r in rc} == {"10M": "inside", "100M": "inside"}
    lad = {(r["readout"].split(";")[0], r["cell"]): r["verdict"] for r in rows if r["row"] in ("R7", "R8")}
    assert lad[("rung C (acts only), k = 1", "10M")] == "inside" and lad[("rung A ('naughty $A')", "10M")] == "inside"
    assert [r["verdict"] for r in rows if r["row"] == "R10"] == ["inside"]
    flow = [r for r in rows if r["row"] == "R16" and r["readout"].startswith("flow")]
    assert flow and all(r["verdict"] == "inside" for r in flow)
    c4 = [r for r in rows if r["row"] == "C4" and r["cell"] == "10M"]
    assert [r["verdict"] for r in c4] == ["outside", "inside", "no range"]  # name token 0.3 (mean over blocks) below 0.4–0.5; site 0.5 in 0.5–0.6
    a4 = [r for r in rows if r["row"] == "A4"]
    assert a4[0]["value"] == pytest.approx(1.0) and a4[0]["verdict"] == "inside" and [r["verdict"] for r in a4[1:]] == ["no range"] * 3
    md = an.predictions_markdown(rows, {"time": "t", "git_commit": "abcdef0", "git_dirty": False, "root": "x"})
    assert md.count("\n| ") == len(rows) + 1 and "pred_excess" in md


def test_every_v4_figure_panel_renders(res, tmp_path):
    import plan_figures as pf

    pf.D["a"] = json.loads(json.dumps(res, default=float))
    pf.D["dense"] = {}
    for key, run in pf.V4.items():
        run(str(tmp_path))
        assert (tmp_path / f"v4_{key}.png").exists(), key
    assert set(pf.V4) == set(pf.PANELS) | {"A5", "2a_acts", "A4b", "A7"}  # every placeholder panel has its v4 version, plus the v4-only panels


def test_story_quality_control(tmp_path):
    import story_quality as sq

    rows = sq.control_rows(sq.load_openings())
    full = [r for r in rows if r["model"] == "control:true_rest"]
    cut = [r for r in rows if r["model"] == "control:true_rest_300"]
    assert len(full) == len(cut) == 100 and all(r["ended"] for r in full)
    assert sum(not r["ended"] for r in cut) == sum(r["n_tokens"] > 300 for r in full) and max(r["n_tokens"] for r in cut) == 300
    p = tmp_path / "control.judge.jsonl"
    judged = [{"model": "control:true_rest", "id": str(i), "scores": {"grammar": 10, "consistency": 9 + i % 2, "plot": 9}} for i in range(4)]
    judged.append({"model": "control:true_rest", "id": "x", "scores": {"grammar": None, "consistency": None, "plot": None}})  # a failed reply: left out
    p.write_text("".join(json.dumps(r) + "\n" for r in judged))
    c = an.story_control(p)["control:true_rest"]
    assert c["grammar"]["mean"] == 10 and c["consistency"]["mean"] == 9.5 and c["mean_of_criteria"]["n"] == 4
    assert an.story_control(tmp_path / "missing.jsonl") == {}


def test_story_quality_two_level_and_the_paired_gap(tmp_path):
    ids = [f"o{i}" for i in range(30)]
    base = {i: 6 + (k % 4) for k, i in enumerate(ids)}  # openings differ in difficulty
    ctrl = tmp_path / "control.judge.jsonl"
    ctrl.write_text("".join(json.dumps({"model": "control:true_rest_300", "id": i, "scores": {"grammar": base[i] + 2, "consistency": base[i] + 2, "plot": base[i] + 2}}) + "\n" for i in ids))
    ev_dir = tmp_path / "box-1" / "logs" / "v4" / "eval"
    (ev_dir / "story").mkdir(parents=True)
    for s, off in enumerate((-0.5, 0.0, 0.5)):  # three seeds, each a constant offset from the control
        rows = [{"model": f"v4-lex-y100-10M-s{s}", "id": i, "scores": {c: base[i] + 1 + off for c in ("grammar", "consistency", "plot")}} for i in ids]
        (ev_dir / "story" / f"v4-lex-y100-10M-s{s}.judge.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    q = an.samples_section([ev_dir], control=ctrl)["story_quality"]["v4-lex-y100-10M"]
    g = q["gap_to_control"]
    assert g["value"] == pytest.approx(-1.0) and g["lo"] < -1.0 < g["hi"]  # the seeds' spread; the openings' difficulty cancels in the pair
    m = q["mean_of_criteria_2level"]
    assert m["value"] == pytest.approx(np.mean(list(base.values())) + 1)
    assert (m["hi"] - m["lo"]) > (g["hi"] - g["lo"])  # unpaired, the openings' spread adds to the interval


def test_gd_section(tmp_path):
    models = {f"m{i}": {"runs": {"gd×0.0625": {"predicted": {"attach": 0.1 * i, "spill": 0.05 * i}, "measured": {"attach": 0.1 * i, "spill": 0.04 * i}}}} for i in range(1, 6)}
    p = tmp_path / "gd.json"
    p.write_text(json.dumps({"models": models}))
    g = an.gd_section(p)
    assert g["gd×0.0625 attach"]["value"] == pytest.approx(1.0) and g["gd×0.0625 spill"]["value"] == pytest.approx(0.8)
    assert an.gd_section(tmp_path / "none.json") is None
