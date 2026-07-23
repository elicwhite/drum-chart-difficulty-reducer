"""Scorecard-backend regression tests: drum_reducer's own featurize->
ScorecardBackend->decode path, run against the frozen fixture in
../../data/fixtures/scorecard_parity_fixture.json. Mirrors test_parity.py's
structure for the model backend. Three things are verified:
  1. Every fixture song/tier's reduced note list matches byte-for-byte
     (note-for-note, (ms, lane) pairs, ms rounded to 3 decimals) --
     regenerate via (scratchpad) build_scorecard_fixture.py if
     backend_scorecard.py or scorecard.json's `decode` block changes.
  2. The pooled rb4_test canonicalized edit_rate reproduces 0.2234 -- the
     scorecard's drift guard (DETERMINISM_CONTRACT.md Sec.5: this number was
     measured through THIS shipped decode path -- tools/build_scorecard.py's
     own decode-knob tuning step, not a floating-point reimplementation).
  3. The MODEL backend still reproduces 0.1703 -- a regression check that
     reduce.py's backend-provided survive_threshold/nms_gap generalization
     (added to support the scorecard backend) did not change the model
     path's behavior.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.normpath(os.path.join(PKG_ROOT, "..", "data"))
sys.path.insert(0, PKG_ROOT)

from drum_reducer import editrate as ER  # noqa: E402
from drum_reducer import reduce as reduce_fn  # noqa: E402
from drum_reducer.backend_model import ModelBackend  # noqa: E402
from drum_reducer.backend_scorecard import ScorecardBackend  # noqa: E402

FIXTURE_PATH = os.path.join(DATA_DIR, "fixtures", "scorecard_parity_fixture.json")
CHARTS_DIR = os.path.join(DATA_DIR, "fixtures", "charts")

SCORECARD_RB4_TEST_POOLED_EDIT_RATE = 0.2234
MODEL_RB4_TEST_POOLED_EDIT_RATE = 0.1703
TIERS = ["hard", "medium", "easy"]


@pytest.fixture(scope="module")
def scorecard_model():
    return ScorecardBackend.load_default()


@pytest.fixture(scope="module")
def model():
    return ModelBackend.load_default()


@pytest.fixture(scope="module")
def fixture():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def _load_chart(sid):
    with open(os.path.join(CHARTS_DIR, f"{sid}.json")) as f:
        return json.load(f)


def test_scorecard_decode_knobs_present(scorecard_model, fixture):
    """scorecard.json must carry the validated (T_tier, nms_gap) per tier --
    not the literal 0.5-boundary T_tier alone (DETERMINISM_CONTRACT.md Sec.4
    vs the actual rb4_val-selected operating point, see backend_scorecard.py
    docstring)."""
    assert set(scorecard_model.decode_knobs.keys()) == set(TIERS)
    for tier in TIERS:
        knobs = scorecard_model.decode_knobs[tier]
        assert isinstance(knobs["T_tier"], int)
        assert knobs["nms_gap"] is None or isinstance(knobs["nms_gap"], (int, float))
    assert scorecard_model.decode_knobs == fixture["metadata"]["decode_knobs"]


def test_fixture_songs_match_note_for_note(scorecard_model, fixture):
    songs = fixture["songs"]
    assert len(songs) > 0
    mismatches = []
    for sid, expected_by_tier in songs.items():
        chart = _load_chart(sid)
        for tier, expected in expected_by_tier.items():
            got = reduce_fn(chart, tier, backend="scorecard", model=scorecard_model)
            got_rounded = [{"ms": round(n["ms"], 3), "lane": n["lane"]} for n in got]
            if got_rounded != expected:
                mismatches.append((sid, tier, len(expected), len(got_rounded)))
    assert not mismatches, f"{len(mismatches)} song/tier mismatches (first 10): {mismatches[:10]}"


def test_rb4_test_pooled_edit_rate_scorecard(scorecard_model, fixture):
    """The scorecard's hard drift guard -- pooled canonicalized edit_rate
    over the fixture's rb4_test songs must reproduce 0.2234 (+/- 1e-4),
    measured through THIS shipped decode path (not a floating-point
    reimplementation -- see tools/build_scorecard.py's decode-knob tuning
    step)."""
    meta = fixture.get("metadata", {})
    n_rb4 = meta.get("n_rb4_test_songs")
    songs = fixture["songs"]
    edits_total, n_gt_total = 0, 0
    n_scored = 0
    for sid, expected_by_tier in songs.items():
        chart = _load_chart(sid)
        if chart.get("_edge_case"):
            continue
        for tier in TIERS:
            gt_diff = chart["difficulties"].get(tier)
            gt = ER.notes_from_difficulty(gt_diff)
            if not gt:
                continue
            cand = reduce_fn(chart, tier, backend="scorecard", model=scorecard_model)
            cand_notes = [ER.Note(n["ms"], n["lane"]) for n in cand]
            _rate, ops = ER.edit_rate(cand_notes, gt)
            edits_total += ops["insert"] + ops["delete"] + ops["lane_move"] + ops["slot_move"]
            n_gt_total += len(gt)
        n_scored += 1
    if n_rb4 is not None:
        assert n_scored == n_rb4, f"expected {n_rb4} rb4_test songs, scored {n_scored}"
    pooled = edits_total / n_gt_total
    print(f"\n[test_parity_scorecard] rb4_test pooled canonicalized edit_rate = {pooled:.4f} "
          f"(n_songs={n_scored}, n_gt={n_gt_total})")
    assert abs(pooled - SCORECARD_RB4_TEST_POOLED_EDIT_RATE) < 1e-4, (
        f"pooled edit_rate {pooled:.4f} drifted from the frozen reference "
        f"{SCORECARD_RB4_TEST_POOLED_EDIT_RATE}")


def test_edge_case_songs_run_without_error(scorecard_model, fixture):
    any_edge = False
    for sid in fixture["songs"]:
        chart = _load_chart(sid)
        if not chart.get("_edge_case"):
            continue
        any_edge = True
        for tier in TIERS:
            out = reduce_fn(chart, tier, backend="scorecard", model=scorecard_model)
            assert isinstance(out, list)
    assert any_edge, "no synthetic edge-case songs found in the fixture"


def test_model_backend_still_reproduces_0_1703(model, fixture):
    """Regression check: reduce.py's backend-provided survive_threshold/
    nms_gap generalization (added to support the scorecard backend) must
    not have changed the model path's behavior."""
    songs = fixture["songs"]
    edits_total, n_gt_total = 0, 0
    for sid in songs:
        chart = _load_chart(sid)
        if chart.get("_edge_case"):
            continue
        for tier in TIERS:
            gt = ER.notes_from_difficulty(chart["difficulties"].get(tier))
            if not gt:
                continue
            cand = reduce_fn(chart, tier, backend="model", model=model)
            cand_notes = [ER.Note(n["ms"], n["lane"]) for n in cand]
            _rate, ops = ER.edit_rate(cand_notes, gt)
            edits_total += ops["insert"] + ops["delete"] + ops["lane_move"] + ops["slot_move"]
            n_gt_total += len(gt)
    pooled = edits_total / n_gt_total
    print(f"\n[test_parity_scorecard] model-backend regression check: rb4_test pooled = {pooled:.4f}")
    assert abs(pooled - MODEL_RB4_TEST_POOLED_EDIT_RATE) < 1e-4, (
        f"model backend pooled edit_rate {pooled:.4f} drifted from {MODEL_RB4_TEST_POOLED_EDIT_RATE} -- "
        f"reduce.py's backend-provided threshold/gap generalization broke the model path")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
