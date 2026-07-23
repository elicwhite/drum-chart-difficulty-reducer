'use strict';

/**
 * Auditable scorecard model backend (backend B: DETERMINISM_CONTRACT.md
 * §4). Not lossless (accepts ~+0.04 edit_rate vs backend A's 0.1703) --
 * the point is that every rule is a small, readable, INTEGER table a
 * human can check by hand, instead of an opaque tree ensemble. Mirrors
 * python/drum_reducer/backend_scorecard.py exactly.
 *
 * Reads `data/scorecard/scorecard.json`, produced by the Python-side
 * tools/build_scorecard.py: a depth-1 refit of the survive GBMs plus
 * depth-2 relane trees on bin-indexed features, plus this module's
 * survive_threshold/nms_gap knobs, selected on a held-out val split and
 * written into scorecard.json's `decode` block -- NOT hardcoded here, so
 * both the Python and JS ports read the same validated numbers from one
 * artifact.
 *
 * Survive: `points = base_points + sum_f feature_points[f](bin(x_f))`,
 * summed in feature-index order (feature_names.json order --
 * DETERMINISM_CONTRACT.md §1's fixed summation order). `predictSurvive`
 * returns this integer total per note; the shared decode.survivePool
 * takes the arithmetic mean of these across a groove-pool group and
 * compares against `surviveThreshold(tier)` -- mean(points) >= T_tier is
 * mathematically identical to contract §4's sum(points) >= n*T_tier.
 *
 * Relane: walk the tier/family's depth-2 tree (nested object: internal
 * nodes {feature_idx, threshold}, leaves {class_counts}); go left iff
 * bin_index(x[feature_idx]) <= threshold, where bin_index uses the SHARED
 * relane_bin_edges table (searchsorted side='left'). At the leaf,
 * final_lane = argmax(class_counts) (ties broken by lowest LANE_INDEX),
 * confidence = winner_count - runnerup_count (an integer >= 0).
 */

const fs = require('fs');
const path = require('path');

const { LANE_INDEX, laneIdx } = require('./decode');

const DEFAULT_DATA_DIR = path.normalize(path.join(__dirname, '..', '..', 'data', 'scorecard'));

/**
 * bin = smallest index i such that x <= edges[i], i.e. searchsorted(edges,
 * x, side='left') -- same convention as backend_model.js's rebin.
 */
function searchsortedLeft(edges, x) {
  let lo = 0, hi = edges.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (edges[mid] < x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

class ScorecardBackend {
  constructor(scorecard) {
    this.scorecard = scorecard;
    this.featureNames = scorecard.feature_names;
    this.survive = scorecard.survive; // {tier: {base_points, T_tier, features}}
    this.relane = scorecard.relane; // {tier: {family: {lanes, observed_classes, tree}}}
    this.relaneBinEdges = scorecard.relane_bin_edges.map((e) => Float64Array.from(e));
    this.decodeKnobs = scorecard.decode || {}; // {tier: {T_tier, nms_gap}} -- may be empty until S2 lands

    // Precompute per-tier, per-feature-index (fixed order) (edges, points)
    // arrays so predictSurvive is a tight loop, not an object lookup per row.
    this._surviveTables = {};
    for (const tier of Object.keys(this.survive)) {
      const sc = this.survive[tier];
      const perFeat = [];
      for (const fname of this.featureNames) {
        const fd = sc.features[fname];
        perFeat.push({ edges: Float64Array.from(fd.bin_edges), points: Float64Array.from(fd.points) });
      }
      this._surviveTables[tier] = perFeat;
    }
  }

  static load(dataDir) {
    const scorecard = JSON.parse(fs.readFileSync(path.join(dataDir, 'scorecard.json'), 'utf8'));
    return new ScorecardBackend(scorecard);
  }

  static loadDefault() {
    return ScorecardBackend.load(DEFAULT_DATA_DIR);
  }

  // -- survive ------------------------------------------------------------

  /**
   * X: array of Float64Array rows (n x 59). Returns per-note integer
   * points (array[n] of numbers, integer-valued): base_points +
   * sum_f points_table[f][bin(x_f)], features summed in FEATURE_NAMES
   * order (index 0..58).
   */
  predictSurvive(tier, X) {
    const sc = this.survive[tier];
    const tables = this._surviveTables[tier];
    const out = new Array(X.length);
    for (let i = 0; i < X.length; i++) {
      let total = sc.base_points;
      const xRow = X[i];
      for (let j = 0; j < tables.length; j++) {
        const { edges, points } = tables[j];
        if (points.length === 1) {
          total += points[0];
          continue;
        }
        let idx = searchsortedLeft(edges, xRow[j]);
        if (idx < 0) idx = 0; else if (idx > points.length - 1) idx = points.length - 1;
        total += points[idx];
      }
      out[i] = total;
    }
    return out;
  }

  /**
   * Per-tier integer T_tier, selected on rb4_val and stored in
   * scorecard.json's `decode` block -- NOT the raw T_tier from the
   * quantization step alone, which only maps the depth-1 model's OWN
   * 0.5-probability boundary into integer-points space; this is the
   * scorecard's actual validated operating point for the shipped decode.
   */
  surviveThreshold(tier) {
    const knobs = this.decodeKnobs[tier];
    if (knobs != null) return Number(knobs.T_tier);
    return Number(this.survive[tier].T_tier); // fallback: literal boundary mapping
  }

  nmsGap(tier) {
    const knobs = this.decodeKnobs[tier];
    if (knobs != null) return knobs.nms_gap;
    return null;
  }

  // -- relane ---------------------------------------------------------------

  _binIndex(featureIdx, x) {
    const edges = this.relaneBinEdges[featureIdx];
    let idx = searchsortedLeft(edges, x);
    if (idx < 0) idx = 0; else if (idx > edges.length) idx = edges.length;
    return idx;
  }

  _walkTree(tree, xRow) {
    let node = tree;
    while (!node.leaf) {
      const f = node.feature_idx;
      const bidx = this._binIndex(f, xRow[f]);
      node = bidx <= node.threshold ? node.left : node.right;
    }
    const counts = node.class_counts;
    // winner = argmax(count); tie -> lowest LANE_INDEX (mirrors
    // DETERMINISM_CONTRACT.md §2's tie-break style for lane decisions).
    const ranked = Object.keys(counts).sort((a, b) => {
      if (counts[b] !== counts[a]) return counts[b] - counts[a];
      return laneIdx(a) - laneIdx(b);
    });
    const winnerLane = ranked[0];
    const winnerCount = counts[winnerLane];
    const runnerupCount = ranked.length > 1 ? counts[ranked[1]] : 0;
    const confidence = winnerCount - runnerupCount; // integer margin, >= 0
    return [winnerLane, confidence];
  }

  /**
   * X: array of Float64Array rows for the surviving family notes. Returns
   * {finalLane: [str, ...], confidence: [number, ...] of integer count-margins}.
   */
  predictRelane(tier, family, X) {
    const tree = this.relane[tier][family].tree;
    const finalLane = new Array(X.length);
    const confidence = new Array(X.length);
    for (let i = 0; i < X.length; i++) {
      const [lane, conf] = this._walkTree(tree, X[i]);
      finalLane[i] = lane;
      confidence[i] = conf;
    }
    return { finalLane, confidence };
  }
}

module.exports = { ScorecardBackend, DEFAULT_DATA_DIR };
