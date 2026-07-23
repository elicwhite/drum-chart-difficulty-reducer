'use strict';

/**
 * Standalone note-level edit_rate metric. Mirrors
 * python/drum_reducer/editrate.py exactly -- this is the metric
 * test/parity.test.js scores the reducer against; it is not itself part
 * of the reduce() decode path.
 *
 * A "note" here is {ms, lane}: ms is scan-chart's already-tempo-resolved
 * millisecond time, lane is the cleaned drum instrument string.
 */

const EPS_MS = 0.5; // float-rounding slack for "same tick" (W=0) matching

/**
 * Flatten [ms, [{instrument, ...}, ...]] grouped notes into a flat,
 * (ms, lane)-sorted list of {ms, lane}. One Note per (ms, lane) pair; a
 * chord (multiple lanes at the same ms) becomes multiple Notes.
 */
function notesFromDifficulty(diffData) {
  if (diffData == null) return [];
  const notes = [];
  for (const [ms, lanes] of diffData.notes) {
    for (const entry of lanes) {
      notes.push({ ms, lane: entry.instrument });
    }
  }
  notes.sort((a, b) => (a.ms !== b.ms ? a.ms - b.ms : (a.lane < b.lane ? -1 : a.lane > b.lane ? 1 : 0)));
  return notes;
}

function eighthNoteMs(tempos, ms) {
  let bpm = tempos && tempos.length ? tempos[0].bpm : 120.0;
  if (tempos) {
    for (const t of tempos) {
      if (t.ms <= ms) bpm = t.bpm; else break;
    }
  }
  return (60000.0 / bpm) / 2.0;
}

/** Greedy 1:1 same-lane-preferred nearest match within +/-window_ms. */
function _match(cand, gt, windowMs) {
  const windowAt = (gms) => (typeof windowMs === 'function' ? windowMs(gms) : windowMs);

  const candidates = [];
  for (let gi = 0; gi < gt.length; gi++) {
    const g = gt[gi];
    const w = windowAt(g.ms);
    for (let ci = 0; ci < cand.length; ci++) {
      const c = cand[ci];
      const dist = Math.abs(c.ms - g.ms);
      if (dist <= w) {
        const sameLane = c.lane === g.lane ? 0 : 1;
        candidates.push([sameLane, dist, gi, ci]);
      }
    }
  }

  // priority: same-lane first, then closest, then stable by (gi, ci)
  candidates.sort((a, b) => {
    for (let k = 0; k < 4; k++) if (a[k] !== b[k]) return a[k] - b[k];
    return 0;
  });

  const usedC = new Set(), usedG = new Set();
  const pairs = [];
  for (const [, , gi, ci] of candidates) {
    if (usedG.has(gi) || usedC.has(ci)) continue;
    usedG.add(gi);
    usedC.add(ci);
    pairs.push([ci, gi]);
  }

  const unmatchedC = [];
  for (let i = 0; i < cand.length; i++) if (!usedC.has(i)) unmatchedC.push(i);
  const unmatchedG = [];
  for (let i = 0; i < gt.length; i++) if (!usedG.has(i)) unmatchedG.push(i);
  return { pairs, unmatchedC, unmatchedG };
}

/**
 * Edit ops to turn `cand` into `gt`: insert (unmatched gt note), delete
 * (unmatched cand note), lane_move (matched pair, different lane),
 * slot_move (matched pair, different tick, within window_ms).
 */
function editOps(cand, gt, windowMs = EPS_MS) {
  const { pairs, unmatchedC, unmatchedG } = _match(cand, gt, windowMs);

  let laneMove = 0, slotMove = 0;
  for (const [ci, gi] of pairs) {
    const c = cand[ci], g = gt[gi];
    if (c.lane !== g.lane) laneMove += 1;
    if (Math.abs(c.ms - g.ms) > EPS_MS) slotMove += 1;
  }

  return {
    insert: unmatchedG.length,
    delete: unmatchedC.length,
    lane_move: laneMove,
    slot_move: slotMove,
  };
}

/** total_edits / |gt|. rate is null if |gt| == 0. */
function editRate(cand, gt, windowMs = EPS_MS) {
  const ops = editOps(cand, gt, windowMs);
  const total = ops.insert + ops.delete + ops.lane_move + ops.slot_move;
  const rate = gt.length ? total / gt.length : null;
  return { rate, ops };
}

module.exports = { EPS_MS, notesFromDifficulty, eighthNoteMs, editOps, editRate };
