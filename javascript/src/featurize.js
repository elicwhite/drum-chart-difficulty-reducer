'use strict';

/**
 * The 59-feature extractor, per note, for the drum difficulty-reducer.
 * Mirrors python/drum_reducer/featurize.py exactly -- see that file's
 * docstring and SPEC.md §2 for the byte-exact spec.
 * `section_prechorus` is listed TWICE,
 * deliberately -- do not de-dupe it, every downstream column shifts if you
 * do. FEATURE_NAMES is the ground-truth 59-column order, verified against
 * data/model/feature_names.json by test/parity.test.js.
 */

const EPS_MS = 0.5; // tick-rounding slack, matches editrate.js's EPS_MS
const ALIGN_EPS_BEATS = 0.04; // ~ a 32nd note at 4/4, tempo-normalized
const GRID_DIVS = [['half', 2.0], ['quarter', 1.0], ['eighth', 0.5]];

const LANE_VOCAB = ['kick', 'snare', 'hihat', 'open-hat', 'high-tom', 'mid-tom', 'floor-tom', 'crash', 'ride'];
const FAMILIES = { cymbal: ['hihat', 'open-hat', 'crash', 'ride'], tom: ['high-tom', 'mid-tom', 'floor-tom'] };
const FAMILY_OF_LANE = {};
for (const fam of Object.keys(FAMILIES)) {
  for (const lane of FAMILIES[fam]) FAMILY_OF_LANE[lane] = fam;
}
const BACKBONE_LANES = new Set(['kick', 'snare']);
const ERA_VOCAB = ['RB1', 'RB2', 'RB3', 'RB4', 'other'];

// (substring keyword, label) -- first match wins (case-insensitive substring
// test), else "other". Iterated in this exact order, both for sectionType()
// and for the one-hot column list below -- the two synonym entries for
// "prechorus" ("pre-chorus" and "prechorus") are a real artifact of the
// reference implementation and produce the deliberate duplicate column.
const SECTION_KEYWORDS = [
  ['intro', 'intro'],
  ['outro', 'outro'],
  ['pre-chorus', 'prechorus'],
  ['prechorus', 'prechorus'],
  ['chorus', 'chorus'],
  ['verse', 'verse'],
  ['bridge', 'bridge'],
  ['solo', 'solo'],
  ['breakdown', 'breakdown'],
  ['interlude', 'interlude'],
  ['fill', 'fill'],
];
const SECTION_VOCAB = SECTION_KEYWORDS.map(([, lbl]) => lbl).concat(['other']);

// Chord-importance rank (lower = more likely to survive a reduction) --
// used only by aug_chord_priority.
const LANE_PRIORITY = {
  kick: 0, snare: 1, crash: 2, ride: 3, hihat: 4,
  'open-hat': 5, 'floor-tom': 6, 'mid-tom': 7, 'high-tom': 8, other: 9,
};

const BASE_NUMERIC = ['chord_size', 'beat_in_measure', 'beats_per_measure', 'is_downbeat',
  'local_density_500ms', 'gap_prev_ms', 'gap_next_ms', 'ghost', 'accent', 'flam',
  'aligned_half', 'aligned_quarter', 'aligned_eighth'];

const AUG_FEATS = ['aug_dist_backbone_ms', 'aug_density_ratio',
  'aug_samelane_prev_ms', 'aug_samelane_next_ms', 'aug_chord_priority',
  'aug_density_100ms', 'aug_density_1500ms', 'aug_beat_frac', 'aug_lane_frac_500ms'];

const LANE_PLUS_OTHER = LANE_VOCAB.concat(['other']);

// Ground truth 59-column order (base13 + lane10 + section12 + era5 +
// chord_has10 + aug9). Cross-checked against feature_names.json.
const FEATURE_NAMES = []
  .concat(BASE_NUMERIC)
  .concat(LANE_PLUS_OTHER.map((lane) => `lane_${lane}`))
  .concat(SECTION_VOCAB.map((sec) => `section_${sec}`))
  .concat(ERA_VOCAB.map((era) => `era_${era}`))
  .concat(LANE_PLUS_OTHER.map((lv) => `chord_has_${lv}`))
  .concat(AUG_FEATS);
if (FEATURE_NAMES.length !== 59) throw new Error(`FEATURE_NAMES length ${FEATURE_NAMES.length}`);

function laneOf(instrument) {
  return LANE_VOCAB.includes(instrument) ? instrument : 'other';
}

// ---------------------------------------------------------------------------
// bisect helpers (Python's bisect.bisect_left / bisect_right on ascending
// sorted arrays of numbers).
// ---------------------------------------------------------------------------

function bisectLeft(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (arr[mid] < x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

function bisectRight(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (x < arr[mid]) hi = mid; else lo = mid + 1;
  }
  return lo;
}

// Python floor-mod / floor-div for floats: a % b == a - floor(a/b)*b.
function pyFloorMod(a, b) {
  return a - Math.floor(a / b) * b;
}

// ---------------------------------------------------------------------------
// Tempo/time-signature/section geometry helpers (ported from
// reduction_probe.py via featurize.py).
// ---------------------------------------------------------------------------

function buildMsToBeat(tempos) {
  tempos = tempos && tempos.length ? tempos.slice().sort((a, b) => a.ms - b.ms) : [];
  if (!tempos.length || tempos[0].ms > 0) {
    tempos = [{ ms: 0, bpm: tempos.length ? tempos[0].bpm : 120.0 }].concat(tempos);
  }
  const anchorsMs = [], anchorsBeat = [], bpms = [];
  let cumBeats = 0.0;
  for (let i = 0; i < tempos.length; i++) {
    anchorsMs.push(tempos[i].ms);
    anchorsBeat.push(cumBeats);
    bpms.push(tempos[i].bpm);
    if (i + 1 < tempos.length) {
      const durMs = tempos[i + 1].ms - tempos[i].ms;
      cumBeats += (durMs * tempos[i].bpm) / 60000.0;
    }
  }
  return function msToBeat(ms) {
    const idx = Math.max(0, bisectRight(anchorsMs, ms) - 1);
    return anchorsBeat[idx] + ((ms - anchorsMs[idx]) * bpms[idx]) / 60000.0;
  };
}

function buildMeasureFn(timeSigs, msToBeat) {
  let ts = timeSigs && timeSigs.length ? timeSigs.slice().sort((a, b) => a.ms - b.ms) : [];
  if (!ts.length) ts = [{ ms: 0, numerator: 4, denominator: 4 }];
  const segs = ts.map((t) => {
    const b = msToBeat(t.ms);
    const beatsPerMeasure = (t.numerator * 4.0) / t.denominator;
    return [b, beatsPerMeasure];
  });
  const segStarts = segs.map((s) => s[0]);
  return function beatToMeasurePos(beat) {
    const idx = Math.max(0, bisectRight(segStarts, beat) - 1);
    const [segStart, bpMeasure] = segs[idx];
    const rel = beat - segStart;
    const beatInMeasure = bpMeasure > 0 ? pyFloorMod(rel, bpMeasure) : 0.0;
    return [beatInMeasure, bpMeasure];
  };
}

function sectionType(name) {
  const n = (name || '').toLowerCase();
  for (const [kw, label] of SECTION_KEYWORDS) {
    if (n.includes(kw)) return label;
  }
  return 'other';
}

function buildSectionFn(sections, songEndMs) {
  const secs = sections && sections.length ? sections.slice().sort((a, b) => a.ms - b.ms) : [{ ms: 0, name: '' }];
  const starts = secs.map((s) => s.ms);
  return function at(ms) {
    const idx = Math.max(0, bisectRight(starts, ms) - 1);
    return sectionType(secs[idx].name);
  };
}

function alignFlags(beatPos) {
  const out = {};
  for (const [name, div] of GRID_DIVS) {
    const frac = pyFloorMod(beatPos, div);
    out[`aligned_${name}`] = (frac < ALIGN_EPS_BEATS || div - frac < ALIGN_EPS_BEATS) ? 1 : 0;
  }
  return out;
}

/**
 * notes: [[ms, [{instrument, ...}, ...]], ...], already grouped by ms.
 * Returns [{ms, instrument, chordSize, entry}, ...] sorted by
 * (ms, instrument) -- matches flatten_expert's sort key exactly (raw
 * instrument string, not the lane-vocab-mapped one).
 */
function flattenExpert(notes) {
  const rows = [];
  for (const [ms, entries] of notes) {
    const chordSize = entries.length;
    for (const e of entries) {
      rows.push({ ms, instrument: e.instrument, chordSize, entry: e });
    }
  }
  rows.sort((a, b) => (a.ms !== b.ms ? a.ms - b.ms : (a.instrument < b.instrument ? -1 : a.instrument > b.instrument ? 1 : 0)));
  return rows;
}

// ---------------------------------------------------------------------------
// AUG_FEATS v7 (song-level context) -- ported from annotate_features().
// ---------------------------------------------------------------------------

function median(sortedAsc) {
  // np.median on an arbitrary-order array sorts internally; we sort here.
  const arr = sortedAsc.slice().sort((a, b) => a - b);
  const n = arr.length;
  if (!n) return 0.0;
  const mid = n >> 1;
  return n % 2 ? arr[mid] : (arr[mid - 1] + arr[mid]) / 2.0;
}

function annotateAugFeatures(rows) {
  const backboneMs = rows.filter((r) => BACKBONE_LANES.has(r.lane)).map((r) => r.ms).sort((a, b) => a - b);
  const byLane = new Map();
  for (const r of rows) {
    if (!byLane.has(r.lane)) byLane.set(r.lane, []);
    byLane.get(r.lane).push(r.ms);
  }
  for (const arr of byLane.values()) arr.sort((a, b) => a - b);
  const byTick = new Map();
  for (const r of rows) {
    const tick = Math.round(r.ms / EPS_MS);
    if (!byTick.has(tick)) byTick.set(tick, []);
    byTick.get(tick).push(r);
  }
  const dens = rows.map((r) => r.local_density_500ms);
  const med = median(dens);
  const allMs = rows.map((r) => r.ms).sort((a, b) => a - b);

  for (const r of rows) {
    const ms = r.ms;
    if (backboneMs.length) {
      const i = bisectLeft(backboneMs, ms);
      const cands = [];
      if (i < backboneMs.length) cands.push(Math.abs(backboneMs[i] - ms));
      if (i > 0) cands.push(Math.abs(ms - backboneMs[i - 1]));
      r.aug_dist_backbone_ms = cands.length ? Math.min(...cands) : 5000.0;
    } else {
      r.aug_dist_backbone_ms = 5000.0;
    }
    r.aug_density_ratio = r.local_density_500ms / (med + 1.0);

    const laneMs = byLane.get(r.lane);
    const j = bisectLeft(laneMs, ms);
    r.aug_samelane_prev_ms = j > 0 ? Math.min(ms - laneMs[j - 1], 5000.0) : 5000.0;
    r.aug_samelane_next_ms = (j + 1 < laneMs.length) ? Math.min(laneMs[j + 1] - ms, 5000.0) : 5000.0;

    const tick = Math.round(ms / EPS_MS);
    const myp = Object.prototype.hasOwnProperty.call(LANE_PRIORITY, r.lane) ? LANE_PRIORITY[r.lane] : 9;
    let chordPriority = 0;
    for (const o of byTick.get(tick)) {
      const op = Object.prototype.hasOwnProperty.call(LANE_PRIORITY, o.lane) ? LANE_PRIORITY[o.lane] : 9;
      if (op < myp) chordPriority += 1;
    }
    r.aug_chord_priority = chordPriority;

    const lo100 = bisectLeft(allMs, ms - 100.0);
    const hi100 = bisectRight(allMs, ms + 100.0);
    r.aug_density_100ms = hi100 - lo100 - 1;
    const lo15 = bisectLeft(allMs, ms - 1500.0);
    const hi15 = bisectRight(allMs, ms + 1500.0);
    r.aug_density_1500ms = hi15 - lo15 - 1;

    const bim = r.beat_in_measure;
    r.aug_beat_frac = Math.abs(bim - Math.round(bim));

    const lo5 = bisectLeft(allMs, ms - 500.0);
    const hi5 = bisectRight(allMs, ms + 500.0);
    const nWin = hi5 - lo5;
    const laneMsWin = bisectRight(laneMs, ms + 500.0) - bisectLeft(laneMs, ms - 500.0);
    r.aug_lane_frac_500ms = nWin > 0 ? laneMsWin / nWin : 0.0;
  }
}

function buildMatrix(rows) {
  const n = rows.length;
  const X = new Array(n);
  for (let i = 0; i < n; i++) X[i] = new Float64Array(FEATURE_NAMES.length);
  let col = 0;
  for (const f of BASE_NUMERIC) {
    for (let i = 0; i < n; i++) X[i][col] = rows[i][f];
    col++;
  }
  for (const lane of LANE_PLUS_OTHER) {
    for (let i = 0; i < n; i++) X[i][col] = rows[i].lane === lane ? 1.0 : 0.0;
    col++;
  }
  for (const sec of SECTION_VOCAB) {
    for (let i = 0; i < n; i++) X[i][col] = rows[i].section_type === sec ? 1.0 : 0.0;
    col++;
  }
  for (const era of ERA_VOCAB) {
    for (let i = 0; i < n; i++) X[i][col] = rows[i].era === era ? 1.0 : 0.0;
    col++;
  }
  for (const lv of LANE_PLUS_OTHER) {
    const key = `chord_has_${lv}`;
    for (let i = 0; i < n; i++) X[i][col] = rows[i][key];
    col++;
  }
  for (const f of AUG_FEATS) {
    for (let i = 0; i < n; i++) X[i][col] = rows[i][f];
    col++;
  }
  return X;
}

/**
 * Returns {X, names, rows}: X is an (n, 59) array of Float64Array rows in
 * FEATURE_NAMES order; rows is a list of per-note bookkeeping objects (ms,
 * lane, family, ...) aligned 1:1 with X's row order -- decode.js consumes
 * both together. Empty chart (no Expert notes) -> {X: [], names, rows: []}.
 */
function featurize(chart) {
  const expert = chart.difficulties.expert;
  const expRows = flattenExpert(expert.notes);
  if (!expRows.length) {
    return { X: [], names: FEATURE_NAMES.slice(), rows: [] };
  }

  const tempos = chart.tempos || [];
  const msToBeat = buildMsToBeat(tempos);
  const beatPosFn = buildMeasureFn(chart.timeSignatures || [], msToBeat);
  let songEndMs = -Infinity;
  for (const r of expRows) if (r.ms > songEndMs) songEndMs = r.ms;
  const sectionFn = buildSectionFn(chart.sections || [], songEndMs);
  let era = chart.era || 'other';
  if (!ERA_VOCAB.includes(era)) era = 'other';

  const expMsSorted = expRows.map((r) => r.ms);
  const expMsUnique = Array.from(new Set(expMsSorted)).sort((a, b) => a - b);

  function densityWindow(ms, halfWindowMs = 250.0) {
    const lo = bisectLeft(expMsUnique, ms - halfWindowMs);
    const hi = bisectRight(expMsUnique, ms + halfWindowMs);
    return hi - lo - 1; // exclude self tick
  }

  const prevMsMap = new Map(), nextMsMap = new Map();
  for (let i = 0; i < expMsUnique.length; i++) {
    const ms = expMsUnique[i];
    prevMsMap.set(ms, i > 0 ? expMsUnique[i - 1] : ms);
    nextMsMap.set(ms, i + 1 < expMsUnique.length ? expMsUnique[i + 1] : ms);
  }

  const tickExpertLanes = new Map();
  for (const r of expRows) {
    const tick = Math.round(r.ms / EPS_MS);
    if (!tickExpertLanes.has(tick)) tickExpertLanes.set(tick, new Set());
    tickExpertLanes.get(tick).add(laneOf(r.instrument));
  }

  const rows = [];
  for (const r of expRows) {
    const ms = r.ms, e = r.entry;
    const lane = laneOf(r.instrument);
    const beat = msToBeat(ms);
    const [beatInMeasure, beatsPerMeasure] = beatPosFn(beat);
    const secType = sectionFn(ms);
    const tick = Math.round(ms / EPS_MS);

    const row = {
      ms,
      lane,
      era,
      family: Object.prototype.hasOwnProperty.call(FAMILY_OF_LANE, lane) ? FAMILY_OF_LANE[lane] : 'fixed',
      chord_size: r.chordSize,
      beat_in_measure: beatInMeasure,
      beats_per_measure: beatsPerMeasure,
      is_downbeat: beatInMeasure < ALIGN_EPS_BEATS ? 1 : 0,
      local_density_500ms: densityWindow(ms),
      gap_prev_ms: Math.min(ms - prevMsMap.get(ms), 5000.0),
      gap_next_ms: Math.min(nextMsMap.get(ms) - ms, 5000.0),
      section_type: secType,
      ghost: e.ghost ? 1 : 0,
      accent: e.accent ? 1 : 0,
      flam: e.flam ? 1 : 0,
    };
    Object.assign(row, alignFlags(beatInMeasure));
    const laneSet = tickExpertLanes.get(tick);
    for (const lv of LANE_PLUS_OTHER) {
      row[`chord_has_${lv}`] = laneSet.has(lv) ? 1 : 0;
    }
    rows.push(row);
  }

  annotateAugFeatures(rows);

  const X = buildMatrix(rows);
  return { X, names: FEATURE_NAMES.slice(), rows };
}

module.exports = {
  FEATURE_NAMES, LANE_VOCAB, FAMILIES, FAMILY_OF_LANE, ERA_VOCAB, SECTION_VOCAB,
  laneOf, sectionType, featurize, bisectLeft, bisectRight,
};
