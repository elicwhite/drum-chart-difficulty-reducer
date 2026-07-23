'use strict';

/**
 * Top-level clean API: reduce(chart, tier, {backend, model}) -> [{ms, lane}].
 *
 * Wires featurize.js -> a MODEL backend (backend_model.js: packed-GBM,
 * lossless, 0.1703; or backend_scorecard.js: auditable integer scorecard)
 * -> decode.js, in the exact 9-step order SPEC.md §5 /
 * python/drum_reducer/reduce.py specify (order is not the "obvious"
 * reading -- survive-pool runs BEFORE thresholding AND before NMS;
 * relane-pool runs AFTER relane predict but BEFORE chord-merge;
 * canonicalize is always last):
 *
 *   1. featurize                         (featurize.featurize)
 *   2. survive predict                   (backend.predictSurvive)
 *   3. SURVIVE-POOL                      (decode.survivePool)
 *   4. threshold (>= backend.surviveThreshold(tier))
 *   5. FAMILY-NMS (gap = backend.nmsGap(tier))
 *   6. relane predict                    (backend.predictRelane)
 *   7. RELANE-POOL                       (decode.relanePool)
 *   8. chord-merge dedup                 (decode.chordMerge)
 *   9. CANONICALIZE                      (decode.canonicalize)
 *
 * Steps 4-5's knobs are BACKEND-provided (DETERMINISM_CONTRACT.md §4's one
 * honest exception to "DECODE is fully shared" -- the model backend's
 * surviveThreshold/nmsGap are fixed at their SPEC.md §6
 * values; the scorecard backend's are its own validated (T_tier, NMS-gap)
 * operating point, read from scorecard.json's `decode` block, not
 * hardcoded here). decode.survivePool itself is unchanged/shared: it
 * always computes the arithmetic MEAN of the backend's per-note score
 * across a groove-pool group -- for the scorecard backend that score is
 * integer points, and mean(points) >= T_tier is mathematically the same
 * comparison as contract §4's sum(points) >= n*T_tier (the division is
 * IEEE-754 correctly-rounded, deterministic, portable), so this backend
 * does not need its own pooling function.
 */

const decode = require('./decode');
const featurize = require('./featurize');
const { ModelBackend } = require('./backend_model');
const { ScorecardBackend } = require('./backend_scorecard');

const TIERS = ['hard', 'medium', 'easy'];
const BACKENDS = ['model', 'scorecard'];

const defaultBackends = {};

function getDefaultBackend(name) {
  if (!(name in defaultBackends)) {
    if (name === 'model') defaultBackends[name] = ModelBackend.loadDefault();
    else if (name === 'scorecard') defaultBackends[name] = ScorecardBackend.loadDefault();
  }
  return defaultBackends[name];
}

/**
 * chart: the parsed-note-list input structure (see featurize.js's
 * docstring for the exact schema). tier: one of TIERS. opts.backend:
 * "model" (packed-GBM, lossless) or "scorecard" (auditable integer
 * scorecard). opts.model: an optional pre-loaded backend instance (pass
 * one in to avoid re-reading the on-disk artifact on every call, e.g.
 * when reducing many songs in a loop).
 *
 * Returns a list of {ms, lane} objects, sorted in canonical (ms,
 * lane_index) order (DETERMINISM_CONTRACT.md §1).
 */
function reduce(chart, tier, opts = {}) {
  const backend = opts.backend || 'model';
  if (!TIERS.includes(tier)) throw new Error(`tier must be one of ${TIERS}, got ${tier}`);
  if (!BACKENDS.includes(backend)) throw new Error(`unsupported backend ${backend} -- must be one of ${BACKENDS}`);
  const model = opts.model || getDefaultBackend(backend);

  const { X, names, rows } = featurize.featurize(chart);
  for (let i = 0; i < featurize.FEATURE_NAMES.length; i++) {
    if (names[i] !== featurize.FEATURE_NAMES[i]) {
      throw new Error('featurize() column order drifted from FEATURE_NAMES');
    }
  }
  if (!rows.length) return [];

  const surviveProba = model.predictSurvive(tier, X);

  const { msToMeasure, measureToMs } = decode.buildMeasureClock(chart.tempos || [], chart.timeSignatures || []);
  const expertNotes = rows.map((r) => ({ ms: r.ms, lane: r.lane }));
  const { clusters } = decode.expertGrooveClusters(expertNotes, msToMeasure);

  const pooled = decode.survivePool(rows, surviveProba, msToMeasure, clusters);
  const threshold = model.surviveThreshold(tier);
  let survive = pooled.map((p) => p >= threshold);

  const gap = model.nmsGap(tier);
  if (gap) {
    survive = decode.familyNms(rows, survive, pooled, gap);
  }

  const finalLane = rows.map((r) => r.lane);
  const confidence = rows.map(() => 1.0);
  for (const famName of Object.keys(featurize.FAMILIES)) {
    const idxs = [];
    for (let i = 0; i < rows.length; i++) {
      if (rows[i].family === famName && survive[i]) idxs.push(i);
    }
    if (!idxs.length || !model.relane[tier] || !model.relane[tier][famName]) continue;
    const Xsub = idxs.map((i) => X[i]);
    const { finalLane: lanesOut, confidence: confOut } = model.predictRelane(tier, famName, Xsub);
    for (let k = 0; k < idxs.length; k++) {
      finalLane[idxs[k]] = lanesOut[k];
      confidence[idxs[k]] = Number(confOut[k]);
    }
  }

  const finalLanePooled = decode.relanePool(rows, finalLane, confidence, msToMeasure, clusters);
  let cand = decode.chordMerge(rows, survive, finalLanePooled, confidence);

  if (clusters.size > 0) {
    const rbm = decode.reducedGrooveByMeasure(cand, msToMeasure);
    cand = decode.canonicalize(cand, clusters, rbm, msToMeasure, measureToMs);
  }

  const candSorted = cand.slice().sort((a, b) => (a.ms !== b.ms ? a.ms - b.ms : decode.laneIdx(a.lane) - decode.laneIdx(b.lane)));
  return candSorted.map((n) => ({ ms: n.ms, lane: n.lane }));
}

module.exports = { reduce, TIERS, BACKENDS };
