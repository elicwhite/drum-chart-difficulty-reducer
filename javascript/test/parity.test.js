'use strict';

/**
 * Parity tests: the JS drum_reducer's own featurize->model/scorecard->decode
 * path, run against the frozen fixtures in ../../data/fixtures/. Mirrors
 * python/tests/test_parity.py's coverage, plus the scorecard backend once
 * scorecard.json's `decode` block landed:
 *   1. Every model-fixture song/tier's reduced note list matches
 *      byte-for-byte (note-for-note, (ms, lane) pairs, ms rounded to 3
 *      decimals).
 *   2. The pooled rb4_test canonicalized edit_rate reproduces 0.1703 --
 *      the hard drift guard.
 *   3. feature_names.json order matches featurize.js's FEATURE_NAMES.
 *   4. Every scorecard-fixture song/tier matches byte-for-byte, same bar
 *      as (1) -- proves the JS scorecard backend (integer points, depth-2
 *      relane trees, backend-provided survive_threshold/nms_gap) against
 *      Python's reduce(backend="scorecard").
 *   5. Synthetic edge-case songs run cleanly through every tier.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');

const featurize = require('../src/featurize');
const ER = require('../src/editrate');
const { reduce } = require('../src/reduce');
const { ModelBackend } = require('../src/backend_model');
const { ScorecardBackend } = require('../src/backend_scorecard');

const PKG_ROOT = path.join(__dirname, '..');
const DATA_DIR = path.normalize(path.join(PKG_ROOT, '..', 'data'));
const FIXTURE_PATH = path.join(DATA_DIR, 'fixtures', 'parity_fixture.json');
const SCORECARD_FIXTURE_PATH = path.join(DATA_DIR, 'fixtures', 'scorecard_parity_fixture.json');
const CHARTS_DIR = path.join(DATA_DIR, 'fixtures', 'charts');
const FEATURE_NAMES_PATH = path.join(DATA_DIR, 'model', 'feature_names.json');

const RB4_TEST_POOLED_EDIT_RATE = 0.1703;
const SCORECARD_RB4_TEST_POOLED_EDIT_RATE = 0.2234;
const TIERS = ['hard', 'medium', 'easy'];

/**
 * Exact round-half-to-even to 3 decimals, matching CPython's `round(x, 3)`
 * bit-for-bit (Python rounds the double's TRUE binary value, ties-to-even;
 * a naive `Math.round(x*1000)/1000` uses round-half-away-from-zero on a
 * possibly-already-rounded `x*1000` and disagrees at exact .0005 ties --
 * e.g. round(20117.8125, 3): the double 20117.8125 is exactly
 * representable, so this is a REAL tie, and CPython picks .812 (even)
 * while naive JS rounding picks .813). Decomposes the double into its
 * exact sign*mantissa*2^exponent form via BigInt so the tie test is exact,
 * not itself subject to floating-point rounding.
 */
function doubleToFraction(x) {
  const buf = new ArrayBuffer(8);
  const dv = new DataView(buf);
  dv.setFloat64(0, x);
  const bits = dv.getBigUint64(0);
  const rawExp = Number((bits >> 52n) & 0x7ffn);
  const rawMantissa = bits & 0xfffffffffffffn;
  if (rawExp === 0) return { m: rawMantissa, e: -1074 }; // subnormal
  return { m: rawMantissa | (1n << 52n), e: rawExp - 1075 };
}

function pythonRound3(x) {
  if (!Number.isFinite(x) || x === 0) return x;
  const sign = x < 0 ? -1 : 1;
  const { m, e } = doubleToFraction(Math.abs(x));
  const numerator = m * 1000n;
  let quotient;
  if (e >= 0) {
    quotient = numerator << BigInt(e);
  } else {
    const denom = 1n << BigInt(-e);
    let q = numerator / denom;
    const r = numerator % denom;
    const twice = r * 2n;
    if (twice > denom || (twice === denom && q % 2n === 1n)) q += 1n;
    quotient = q;
  }
  return sign * (Number(quotient) / 1000);
}

const model = ModelBackend.loadDefault();
const fixture = JSON.parse(fs.readFileSync(FIXTURE_PATH, 'utf8'));
const scorecardFixture = JSON.parse(fs.readFileSync(SCORECARD_FIXTURE_PATH, 'utf8'));

function loadChart(sid) {
  return JSON.parse(fs.readFileSync(path.join(CHARTS_DIR, `${sid}.json`), 'utf8'));
}

test('feature names match ground truth', () => {
  const fn = JSON.parse(fs.readFileSync(FEATURE_NAMES_PATH, 'utf8'));
  assert.deepEqual(featurize.FEATURE_NAMES, fn.feature_names);
  assert.equal(featurize.FEATURE_NAMES.length, 59);
  assert.equal(featurize.FEATURE_NAMES.filter((n) => n === 'section_prechorus').length, 2);
});

test('fixture songs match note-for-note', () => {
  const songs = fixture.songs;
  assert.ok(Object.keys(songs).length > 0);
  const mismatches = [];
  for (const sid of Object.keys(songs)) {
    const chart = loadChart(sid);
    for (const tier of Object.keys(songs[sid])) {
      const expected = songs[sid][tier];
      const got = reduce(chart, tier, { model });
      const gotRounded = got.map((n) => ({ ms: pythonRound3(n.ms), lane: n.lane }));
      const same = gotRounded.length === expected.length
        && gotRounded.every((n, i) => n.ms === expected[i].ms && n.lane === expected[i].lane);
      if (!same) mismatches.push([sid, tier, expected.length, gotRounded.length]);
    }
  }
  assert.deepEqual(mismatches, [], `${mismatches.length} song/tier mismatches (first 10): ${JSON.stringify(mismatches.slice(0, 10))}`);
});

test('rb4_test pooled canonicalized edit_rate reproduces 0.1703', () => {
  const meta = fixture.metadata || {};
  const nRb4 = meta.n_rb4_test_songs;
  const songs = fixture.songs;
  let editsTotal = 0, nGtTotal = 0, nScored = 0;
  for (const sid of Object.keys(songs)) {
    const chart = loadChart(sid);
    if (chart._edge_case) continue;
    for (const tier of TIERS) {
      const gtDiff = chart.difficulties[tier];
      const gt = ER.notesFromDifficulty(gtDiff);
      if (!gt.length) continue;
      const cand = reduce(chart, tier, { model });
      const { ops } = ER.editRate(cand, gt);
      editsTotal += ops.insert + ops.delete + ops.lane_move + ops.slot_move;
      nGtTotal += gt.length;
    }
    nScored++;
  }
  if (nRb4 !== undefined && nRb4 !== null) {
    assert.equal(nScored, nRb4, `expected ${nRb4} rb4_test songs, scored ${nScored}`);
  }
  const pooled = editsTotal / nGtTotal;
  console.log(`[parity.test.js] rb4_test pooled canonicalized edit_rate = ${pooled.toFixed(4)} (n_songs=${nScored}, n_gt=${nGtTotal})`);
  assert.ok(Math.abs(pooled - RB4_TEST_POOLED_EDIT_RATE) < 1e-4,
    `pooled edit_rate ${pooled.toFixed(4)} drifted from the frozen reference ${RB4_TEST_POOLED_EDIT_RATE}`);
});

test('scorecard fixture songs match note-for-note', () => {
  // Official fixture (data/fixtures/scorecard_parity_fixture.json, 102
  // songs x 3 tiers), generated from Python's
  // reduce(chart, tier, backend="scorecard") now that scorecard.json's
  // `decode` block is final (T_tier/nms_gap per tier -- read from the
  // artifact by ScorecardBackend, never hardcoded here or in
  // backend_scorecard.js). Same bar as the model-path fixture: 0 diffs,
  // note-for-note, canonical (ms, lane_index) order, ms round-3.
  const scModel = ScorecardBackend.loadDefault();
  const songs = scorecardFixture.songs;
  assert.ok(Object.keys(songs).length > 0);
  const mismatches = [];
  for (const sid of Object.keys(songs)) {
    const chart = loadChart(sid);
    for (const tier of Object.keys(songs[sid])) {
      const expected = songs[sid][tier];
      const got = reduce(chart, tier, { backend: 'scorecard', model: scModel });
      const gotRounded = got.map((n) => ({ ms: pythonRound3(n.ms), lane: n.lane }));
      const same = gotRounded.length === expected.length
        && gotRounded.every((n, i) => n.ms === expected[i].ms && n.lane === expected[i].lane);
      if (!same) mismatches.push([sid, tier, expected.length, gotRounded.length]);
    }
  }
  assert.deepEqual(mismatches, [], `${mismatches.length} song/tier mismatches (first 10): ${JSON.stringify(mismatches.slice(0, 10))}`);
});

test('rb4_test pooled canonicalized edit_rate reproduces 0.2234 (scorecard)', () => {
  // Mirrors test_parity_scorecard.py::test_rb4_test_pooled_edit_rate_scorecard
  // -- the scorecard's own hard drift guard, measured through THIS shipped
  // decode path (not the research-tree float decode).
  const scModel = ScorecardBackend.loadDefault();
  const meta = scorecardFixture.metadata || {};
  const nRb4 = meta.n_rb4_test_songs;
  const songs = scorecardFixture.songs;
  let editsTotal = 0, nGtTotal = 0, nScored = 0;
  for (const sid of Object.keys(songs)) {
    const chart = loadChart(sid);
    if (chart._edge_case) continue;
    for (const tier of TIERS) {
      const gtDiff = chart.difficulties[tier];
      const gt = ER.notesFromDifficulty(gtDiff);
      if (!gt.length) continue;
      const cand = reduce(chart, tier, { backend: 'scorecard', model: scModel });
      const { ops } = ER.editRate(cand, gt);
      editsTotal += ops.insert + ops.delete + ops.lane_move + ops.slot_move;
      nGtTotal += gt.length;
    }
    nScored++;
  }
  if (nRb4 !== undefined && nRb4 !== null) {
    assert.equal(nScored, nRb4, `expected ${nRb4} rb4_test songs, scored ${nScored}`);
  }
  const pooled = editsTotal / nGtTotal;
  console.log(`[parity.test.js] rb4_test pooled canonicalized edit_rate (scorecard) = ${pooled.toFixed(4)} (n_songs=${nScored}, n_gt=${nGtTotal})`);
  assert.ok(Math.abs(pooled - SCORECARD_RB4_TEST_POOLED_EDIT_RATE) < 1e-4,
    `pooled edit_rate ${pooled.toFixed(4)} drifted from the frozen reference ${SCORECARD_RB4_TEST_POOLED_EDIT_RATE}`);
});

test('scorecard decode knobs are read from scorecard.json, not hardcoded', () => {
  const scModel = ScorecardBackend.loadDefault();
  const expectedKnobs = scorecardFixture.metadata.decode_knobs;
  for (const tier of TIERS) {
    assert.equal(scModel.surviveThreshold(tier), expectedKnobs[tier].T_tier, `${tier} T_tier`);
    assert.equal(scModel.nmsGap(tier), expectedKnobs[tier].nms_gap, `${tier} nms_gap`);
  }
});

test('edge-case songs run without error', () => {
  let anyEdge = false;
  for (const sid of Object.keys(fixture.songs)) {
    const chart = loadChart(sid);
    if (!chart._edge_case) continue;
    anyEdge = true;
    for (const tier of TIERS) {
      const out = reduce(chart, tier, { model });
      assert.ok(Array.isArray(out));
    }
  }
  assert.ok(anyEdge, 'no synthetic edge-case songs found in the fixture');
});
