'use strict';

/**
 * The shared, deterministic 9-step decode: survive-pool -> threshold ->
 * family-NMS -> relane -> relane-pool -> chord-merge -> canonicalize (plus
 * the measure-clock / groove-cluster machinery both pooling steps and
 * canonicalize share). Mirrors python/drum_reducer/decode.py exactly --
 * DETERMINISM_CONTRACT.md §2 documents every tie-break implemented here.
 */

// Canonical lane index (DETERMINISM_CONTRACT.md §1) -- used by every
// tie-break in this file.
const LANE_INDEX = {
  kick: 0, snare: 1, hihat: 2, 'open-hat': 3, 'high-tom': 4,
  'mid-tom': 5, 'floor-tom': 6, crash: 7, ride: 8, other: 9,
};

const FAMILIES = { cymbal: ['hihat', 'open-hat', 'crash', 'ride'], tom: ['high-tom', 'mid-tom', 'floor-tom'] };

const GROOVE_TPQ = 480; // tick-in-measure bucketing resolution (RB-convention 480 ticks/quarter)

function laneIdx(lane) {
  return Object.prototype.hasOwnProperty.call(LANE_INDEX, lane) ? LANE_INDEX[lane] : 9;
}

function bisectRight(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (x < arr[mid]) hi = mid; else lo = mid + 1;
  }
  return lo;
}

/**
 * Indices into `rows`, sorted (ms, lane_index) ascending -- the fixed note
 * order DETERMINISM_CONTRACT.md §1 requires for every deterministic
 * iteration (grouping, summation, output).
 */
function canonicalOrder(rows) {
  const idx = rows.map((_, i) => i);
  idx.sort((a, b) => {
    const ra = rows[a], rb = rows[b];
    if (ra.ms !== rb.ms) return ra.ms - rb.ms;
    return laneIdx(ra.lane) - laneIdx(rb.lane);
  });
  return idx;
}

// ---------------------------------------------------------------------------
// Measure clock + Expert groove clusters (ported from consistency_metric.py
// via decode.py, frozen/stdlib).
// ---------------------------------------------------------------------------

/**
 * Returns {msToMeasure, measureToMs}. msToMeasure(ms) -> [measureIdx,
 * beatInMeasure]. measureToMs(measureIdx, beatInMeasure) -> ms (the exact
 * inverse, used by canonicalize() to re-render a donor measure's groove at
 * a different measure's timing).
 */
function buildMeasureClock(tempos, timeSigs) {
  let tps = tempos && tempos.length ? tempos.slice().sort((a, b) => a.ms - b.ms) : [];
  if (!tps.length || tps[0].ms > 0) {
    tps = [{ ms: 0, bpm: tps.length ? tps[0].bpm : 120.0 }].concat(tps);
  }
  const anchorsMs = [], anchorsBeat = [], bpms = [];
  let cumBeats = 0.0;
  for (let i = 0; i < tps.length; i++) {
    anchorsMs.push(tps[i].ms);
    anchorsBeat.push(cumBeats);
    bpms.push(tps[i].bpm);
    if (i + 1 < tps.length) {
      cumBeats += ((tps[i + 1].ms - tps[i].ms) * tps[i].bpm) / 60000.0;
    }
  }

  function msToBeat(ms) {
    const idx = Math.max(0, bisectRight(anchorsMs, ms) - 1);
    return anchorsBeat[idx] + ((ms - anchorsMs[idx]) * bpms[idx]) / 60000.0;
  }
  function beatToMs(beat) {
    const idx = Math.max(0, bisectRight(anchorsBeat, beat) - 1);
    return anchorsMs[idx] + ((beat - anchorsBeat[idx]) * 60000.0) / bpms[idx];
  }

  let ts = timeSigs && timeSigs.length ? timeSigs.slice().sort((a, b) => a.ms - b.ms) : [];
  if (!ts.length) ts = [{ ms: 0, numerator: 4, denominator: 4 }];
  const segs = ts.map((t) => [msToBeat(t.ms), (t.numerator * 4.0) / t.denominator]);
  const segStarts = segs.map((s) => s[0]);
  const cumMeasures = [0];
  for (let i = 1; i < segs.length; i++) {
    const [prevStart, prevBpMeasure] = segs[i - 1];
    const n = prevBpMeasure > 0 ? Math.round((segs[i][0] - prevStart) / prevBpMeasure) : 0;
    cumMeasures.push(cumMeasures[cumMeasures.length - 1] + Math.max(0, n));
  }

  // Absorbs FP drift from summing many tempo segments so a beat_in_measure
  // of e.g. 3.999999999999943 rolls to the next measure's tick 0 instead
  // of a stray tick 1920.
  const BOUNDARY_EPS_BEATS = 1e-6;

  function msToMeasure(ms) {
    const beat = msToBeat(ms);
    const idx = Math.max(0, bisectRight(segStarts, beat) - 1);
    const [segStart, bpMeasure] = segs[idx];
    const rel = beat - segStart;
    let nInSeg = bpMeasure > 0 ? Math.floor(rel / bpMeasure) : 0;
    let beatInMeasure = bpMeasure > 0 ? rel - nInSeg * bpMeasure : 0.0;
    if (bpMeasure > 0 && beatInMeasure > bpMeasure - BOUNDARY_EPS_BEATS) {
      nInSeg += 1;
      beatInMeasure = 0.0;
    }
    return [cumMeasures[idx] + nInSeg, beatInMeasure];
  }

  function measureToMs(measureIdx, beatInMeasure) {
    const idx = Math.max(0, bisectRight(cumMeasures, measureIdx) - 1);
    const [segStart, bpMeasure] = segs[idx];
    const nInSeg = measureIdx - cumMeasures[idx];
    return beatToMs(segStart + nInSeg * bpMeasure + beatInMeasure);
  }

  return { msToMeasure, measureToMs };
}

/** {measureIdx: Set of "tick,lane" keys} for a note list ({ms, lane} objects). */
function reducedGrooveByMeasure(notes, msToMeasure) {
  const byMeasure = new Map();
  for (const n of notes) {
    const [mi] = msToMeasure(n.ms);
    const [, beat] = msToMeasure(n.ms);
    if (!byMeasure.has(mi)) byMeasure.set(mi, new Set());
    byMeasure.get(mi).add(`${Math.round(beat * GROOVE_TPQ)},${n.lane}`);
  }
  return byMeasure;
}

function grooveKeyString(grooveSet) {
  // canonical string form of a groove Set, for use as a Map key (frozenset
  // equivalent -- content-based equality, independent of insertion order).
  return Array.from(grooveSet).sort().join('|');
}

function grooveToPairs(grooveSet) {
  return Array.from(grooveSet).map((s) => {
    const idx = s.indexOf(',');
    return [Number(s.slice(0, idx)), s.slice(idx + 1)];
  });
}

/**
 * Returns {clusters: Map(grooveKeyString -> {pairs, measureIdxs}), nNonemptyMeasures}
 * for groove keys seen in >=2 measures. measureIdxs sorted ascending.
 */
function expertGrooveClusters(expertNotes, msToMeasure) {
  const rbm = reducedGrooveByMeasure(expertNotes, msToMeasure);
  const measureIdxs = Array.from(rbm.keys()).sort((a, b) => a - b);
  const byGroove = new Map();
  for (const mi of measureIdxs) {
    const groove = rbm.get(mi);
    const key = grooveKeyString(groove);
    if (!byGroove.has(key)) byGroove.set(key, { pairs: grooveToPairs(groove), measureIdxs: [] });
    byGroove.get(key).measureIdxs.push(mi);
  }
  const clusters = new Map();
  for (const [key, v] of byGroove.entries()) {
    if (v.measureIdxs.length >= 2) clusters.set(key, v);
  }
  return { clusters, nNonemptyMeasures: rbm.size };
}

/**
 * DETERMINISM_CONTRACT.md §2.4: modal (majority-vote) groove across a
 * cluster's instances; ties broken by the lexicographically smallest
 * sorted (tick, lane_index) tuple list.
 * `reductions`: array of groove-key-strings (one per instance measure).
 * `pairsByKey`: Map(grooveKeyString -> [[tick, lane], ...]) for decoding.
 */
function modalReduction(reductions, pairsByKey) {
  const counts = new Map();
  for (const r of reductions) counts.set(r, (counts.get(r) || 0) + 1);
  let maxCount = 0;
  for (const c of counts.values()) if (c > maxCount) maxCount = c;
  const candidates = Array.from(counts.keys()).filter((k) => counts.get(k) === maxCount);
  if (candidates.length === 1) return candidates[0];

  function sortKey(key) {
    return pairsByKey.get(key)
      .map(([tick, lane]) => [tick, laneIdx(lane)])
      .sort((a, b) => (a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1]));
  }
  function cmpKeyLists(a, b) {
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      if (a[i][0] !== b[i][0]) return a[i][0] - b[i][0];
      if (a[i][1] !== b[i][1]) return a[i][1] - b[i][1];
    }
    return a.length - b.length;
  }
  candidates.sort((a, b) => cmpKeyLists(sortKey(a), sortKey(b)));
  return candidates[0];
}

/**
 * Force every instance in a repeated-groove cluster to the reducer's own
 * modal reduction for that groove. Non-clustered measures pass through
 * untouched. Pure: does not mutate its arguments.
 * `candNotes`: [{ms, lane}, ...]. `clusters`: Map from expertGrooveClusters.
 * `reducedByMeasure`: measureIdx -> groove Set, of the CANDIDATE's own
 * reduction (computed by the caller via reducedGrooveByMeasure(candNotes, ...)).
 */
function canonicalize(candNotes, clusters, reducedByMeasure, msToMeasure, measureToMs) {
  const clusteredMeasures = new Set();
  for (const v of clusters.values()) for (const mi of v.measureIdxs) clusteredMeasures.add(mi);

  const out = candNotes.filter((n) => !clusteredMeasures.has(msToMeasure(n.ms)[0]));

  for (const v of clusters.values()) {
    const reductions = v.measureIdxs.map((mi) => {
      const g = reducedByMeasure.get(mi);
      return g ? grooveKeyString(g) : '';
    });
    const pairsByKey = new Map();
    pairsByKey.set('', []);
    for (const mi of v.measureIdxs) {
      const g = reducedByMeasure.get(mi);
      if (g) pairsByKey.set(grooveKeyString(g), grooveToPairs(g));
    }
    const modalKey = modalReduction(reductions, pairsByKey);
    const modalPairs = pairsByKey.get(modalKey);
    for (const mi of v.measureIdxs) {
      for (const [tick, lane] of modalPairs) {
        out.push({ ms: measureToMs(mi, tick / GROOVE_TPQ), lane });
      }
    }
  }
  out.sort((a, b) => (a.ms !== b.ms ? a.ms - b.ms : laneIdx(a.lane) - laneIdx(b.lane)));
  return out;
}

// ---------------------------------------------------------------------------
// Pooling / NMS / relane-pool / chord-merge
// ---------------------------------------------------------------------------

function measToGrooveMap(clusters) {
  const measToGroove = new Map();
  for (const [gk, v] of clusters.entries()) {
    for (const mi of v.measureIdxs) measToGroove.set(mi, gk);
  }
  return measToGroove;
}

/**
 * SURVIVE-POOL: group notes by (expert_groove_cluster_id,
 * round(beat_in_measure*GROOVE_TPQ), lane); replace each member's
 * survive_proba with the group's arithmetic mean, accumulated in
 * canonical (ms, lane_index) order (DETERMINISM_CONTRACT.md §1). Notes
 * outside any cluster are unaffected. Runs BEFORE thresholding.
 */
function survivePool(rows, surviveProba, msToMeasure, clusters) {
  if (clusters.size === 0) return surviveProba.slice();
  const measToGroove = measToGrooveMap(clusters);
  const order = canonicalOrder(rows);
  const bucket = new Map(); // key string -> [proba, ...]
  const keyOf = new Map(); // row index -> key string
  for (const i of order) {
    const r = rows[i];
    const [mi, beat] = msToMeasure(r.ms);
    const gk = measToGroove.get(mi);
    if (gk === undefined) continue;
    const k = `${gk}::${Math.round(beat * GROOVE_TPQ)}::${r.lane}`;
    if (!bucket.has(k)) bucket.set(k, []);
    bucket.get(k).push(surviveProba[i]);
    keyOf.set(i, k);
  }
  if (bucket.size === 0) return surviveProba.slice();
  const means = new Map();
  for (const [k, vals] of bucket.entries()) {
    let s = 0.0;
    for (const v of vals) s += v; // explicit left-to-right accumulation, canonical order
    means.set(k, s / vals.length);
  }
  const out = surviveProba.slice();
  for (const [i, k] of keyOf.entries()) out[i] = means.get(k);
  return out;
}

/**
 * FAMILY-NMS (cymbal/tom only; kick/snare/other never suppressed). Greedy:
 * sort currently-surviving family notes by descending keep_score,
 * tie-break (-keep_score, ms, lane_index) per DETERMINISM_CONTRACT.md
 * §2.1. Walk the sorted list; drop a note if it falls within gap_ms of an
 * already-kept note's ms.
 */
function familyNms(rows, survive, keepScore, gapMs) {
  const out = survive.slice();
  const famIdx = [];
  for (let i = 0; i < rows.length; i++) {
    if (out[i] && Object.prototype.hasOwnProperty.call(FAMILIES, rows[i].family)) famIdx.push(i);
  }
  famIdx.sort((a, b) => {
    if (keepScore[a] !== keepScore[b]) return keepScore[b] - keepScore[a]; // descending
    if (rows[a].ms !== rows[b].ms) return rows[a].ms - rows[b].ms;
    return laneIdx(rows[a].lane) - laneIdx(rows[b].lane);
  });
  const keptMs = [];
  for (const i of famIdx) {
    const ms = rows[i].ms;
    let dropped = false;
    for (const km of keptMs) {
      if (Math.abs(ms - km) < gapMs) { dropped = true; break; }
    }
    if (dropped) out[i] = false; else keptMs.push(ms);
  }
  return out;
}

/**
 * RELANE-POOL: for FAMILY notes, group by (expert_groove_cluster_id,
 * round(beat_in_measure*GROOVE_TPQ), SOURCE lane); override every member's
 * final_lane with the confidence-weighted modal lane (sum confidence per
 * candidate lane, canonical order; DETERMINISM_CONTRACT.md §2.2: tie
 * broken by lowest lane_index). Runs AFTER relane predict, BEFORE
 * chord-merge.
 */
function relanePool(rows, finalLane, confidence, msToMeasure, clusters) {
  if (clusters.size === 0) return finalLane.slice();
  const measToGroove = measToGrooveMap(clusters);
  const order = canonicalOrder(rows);
  const votes = new Map(); // key -> Map(candidateLane -> summedConf)
  const keyOf = new Map();
  for (const i of order) {
    const r = rows[i];
    if (!Object.prototype.hasOwnProperty.call(FAMILIES, r.family)) continue;
    const [mi, beat] = msToMeasure(r.ms);
    const gk = measToGroove.get(mi);
    if (gk === undefined) continue;
    const k = `${gk}::${Math.round(beat * GROOVE_TPQ)}::${r.lane}`;
    if (!votes.has(k)) votes.set(k, new Map());
    const tally = votes.get(k);
    tally.set(finalLane[i], (tally.get(finalLane[i]) || 0.0) + Number(confidence[i]));
    keyOf.set(i, k);
  }
  if (votes.size === 0) return finalLane.slice();
  const modal = new Map();
  for (const [k, tally] of votes.entries()) {
    // candidate lanes considered in lane_index order (§1); tie -> lowest lane_index (§2.2)
    const lanes = Array.from(tally.keys()).sort((a, b) => laneIdx(a) - laneIdx(b));
    let bestLane = null, bestConf = null;
    for (const lane of lanes) {
      const c = tally.get(lane);
      if (bestConf === null || c > bestConf) { bestLane = lane; bestConf = c; }
    }
    modal.set(k, bestLane);
  }
  const out = finalLane.slice();
  for (const [i, k] of keyOf.entries()) out[i] = modal.get(k);
  return out;
}

/**
 * Fixed lanes (kick/snare/other) pass through unchanged if survive. FAMILY
 * survivors: group by (ms, family, final_lane); if >1 member, keep only
 * the highest-confidence one -- tie broken by lowest SOURCE lane_index
 * (DETERMINISM_CONTRACT.md §2.3).
 */
function chordMerge(rows, survive, finalLane, confidence) {
  const survivorIdx = [];
  for (let i = 0; i < survive.length; i++) if (survive[i]) survivorIdx.push(i);
  const fixedIdx = survivorIdx.filter((i) => !Object.prototype.hasOwnProperty.call(FAMILIES, rows[i].family));
  const familyIdx = survivorIdx.filter((i) => Object.prototype.hasOwnProperty.call(FAMILIES, rows[i].family));

  const byKey = new Map();
  for (const i of familyIdx) {
    const r = rows[i];
    const k = `${r.ms}::${r.family}::${finalLane[i]}`;
    if (!byKey.has(k)) byKey.set(k, []);
    byKey.get(k).push(i);
  }

  const keepIdx = [];
  for (let group of byKey.values()) {
    if (group.length > 1) {
      group = group.slice().sort((a, b) => {
        if (confidence[a] !== confidence[b]) return confidence[b] - confidence[a]; // descending
        return laneIdx(rows[a].lane) - laneIdx(rows[b].lane);
      });
    }
    keepIdx.push(group[0]);
  }

  const cand = fixedIdx.map((i) => ({ ms: rows[i].ms, lane: rows[i].lane }));
  for (const i of keepIdx) cand.push({ ms: rows[i].ms, lane: finalLane[i] });
  return cand;
}

module.exports = {
  LANE_INDEX, FAMILIES, GROOVE_TPQ, laneIdx, canonicalOrder,
  buildMeasureClock, reducedGrooveByMeasure, expertGrooveClusters, canonicalize,
  survivePool, familyNms, relanePool, chordMerge,
};
