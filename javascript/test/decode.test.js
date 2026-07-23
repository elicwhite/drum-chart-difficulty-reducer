'use strict';

/**
 * Direct unit tests for decode.js's DETERMINISM_CONTRACT.md §2 tie-breaks
 * -- ported from python/tests/test_decode.py, constructed synthetically so
 * the tie is guaranteed (exact ties in learned probabilities are
 * vanishingly rare and not reliably constructible from chart content).
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const decode = require('../src/decode');

test('canonicalize modal tie-break picks the lexicographically smaller (tick, lane_index) groove', () => {
  // End-to-end mirror of test_canonicalize_uses_tie_broken_modal_groove:
  // 4 measures with an identical Expert groove (kick+snare at the
  // downbeat) -> one cluster of size 4. Candidate: measures 0,2 keep only
  // snare; measures 1,3 keep only kick -> exact 2-2 tie between
  // {(0,snare)} (lane_index 1) and {(0,kick)} (lane_index 0). kick sorts
  // first, so the tie-break must force every instance to kick-only --
  // independent of insertion/iteration order (snare's groove is built
  // first in the candidate list).
  const tempos = [{ ms: 0, bpm: 120.0 }];
  const timeSigs = [{ ms: 0, numerator: 4, denominator: 4 }];
  const { msToMeasure, measureToMs } = decode.buildMeasureClock(tempos, timeSigs);

  const expert = [];
  for (let mi = 0; mi < 4; mi++) {
    const base = mi * 2000.0;
    expert.push({ ms: base, lane: 'kick' });
    expert.push({ ms: base, lane: 'snare' });
  }
  const { clusters, nNonemptyMeasures } = decode.expertGrooveClusters(expert, msToMeasure);
  assert.equal(nNonemptyMeasures, 4);
  assert.equal(clusters.size, 1);
  const [{ measureIdxs }] = Array.from(clusters.values());
  assert.deepEqual(measureIdxs, [0, 1, 2, 3]);

  const cand = [
    { ms: 0.0, lane: 'snare' },
    { ms: 2000.0, lane: 'kick' },
    { ms: 4000.0, lane: 'snare' },
    { ms: 6000.0, lane: 'kick' },
  ];
  const rbm = decode.reducedGrooveByMeasure(cand, msToMeasure);
  const canon = decode.canonicalize(cand, clusters, rbm, msToMeasure, measureToMs);

  assert.equal(canon.length, 4);
  assert.ok(canon.every((n) => n.lane === 'kick'), `expected all-kick, got ${JSON.stringify(canon)}`);
});

test('family_nms tie-break: equal keep_score ties broken by (ms, lane_index)', () => {
  // Mirrors test_family_nms_tie_break_order: earlier ms should win an
  // exact keep_score tie (not JS's sort-stability-preserved input order,
  // which here would coincidentally agree -- so this also exercises the
  // explicit ms/lane_index comparator, not accidental stability).
  const rows = [
    { ms: 100.0, lane: 'crash', family: 'cymbal' },
    { ms: 150.0, lane: 'hihat', family: 'cymbal' },
  ];
  const survive = [true, true];
  const keepScore = [0.9, 0.9]; // exact tie
  const out = decode.familyNms(rows, survive, keepScore, 100);
  assert.deepEqual(out, [true, false]);
});

test('family_nms tie-break is independent of input order (sort key, not stability)', () => {
  // Same scenario, rows reversed -- must still keep the earlier-ms note.
  const rows = [
    { ms: 150.0, lane: 'hihat', family: 'cymbal' },
    { ms: 100.0, lane: 'crash', family: 'cymbal' },
  ];
  const survive = [true, true];
  const keepScore = [0.9, 0.9];
  const out = decode.familyNms(rows, survive, keepScore, 100);
  assert.deepEqual(out, [false, true]); // index 1 (ms=100) kept
});
