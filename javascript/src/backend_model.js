'use strict';

/**
 * Packed-GBM model backend (backend A: lossless, rb4_test edit_rate
 * 0.1703). Reads the 9 `.bin` files in data/model/ directly with
 * fs+Buffer/DataView -- no external deps. Mirrors
 * python/drum_reducer/backend_model.py exactly; see SPEC.md §3 for the
 * full field-by-field byte layout.
 *
 * Node struct (7 bytes): feature_idx(u8) bin_threshold(u8) left(u8)
 * right(u8) flags(u8) value(f16, little-endian). flags bit0=is_leaf,
 * bit1=missing_go_to_left (unused -- kept for byte-format fidelity only).
 * Leaf `value` already has learning_rate baked in -- never re-multiply
 * (the "double-shrinkage" trap documented in SPEC.md §3.5).
 */

const fs = require('fs');
const path = require('path');

const { sigmoid, softmax } = require('./portable_exp');

const NODE_SIZE = 7;

const TIERS = ['hard', 'medium', 'easy'];
const FAMILIES = { cymbal: ['hihat', 'open-hat', 'crash', 'ride'], tom: ['high-tom', 'mid-tom', 'floor-tom'] };

const DEFAULT_DATA_DIR = path.normalize(path.join(__dirname, '..', '..', 'data', 'model'));

// ---------------------------------------------------------------------------
// fp16 (IEEE-754 binary16) decode -- Buffer/DataView have no readFloat16.
// ---------------------------------------------------------------------------

function readFloat16LE(buf, offset) {
  const h = buf.readUInt16LE(offset);
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) {
    // subnormal (or zero)
    return sign * frac * Math.pow(2, -24);
  }
  if (exp === 0x1f) {
    return frac ? NaN : sign * Infinity;
  }
  return sign * (1 + frac / 1024) * Math.pow(2, exp - 15);
}

function readNode(buf, offset) {
  return {
    feat: buf.readUInt8(offset),
    binThr: buf.readUInt8(offset + 1),
    left: buf.readUInt8(offset + 2),
    right: buf.readUInt8(offset + 3),
    flags: buf.readUInt8(offset + 4),
    value: readFloat16LE(buf, offset + 5),
  };
}

// ---------------------------------------------------------------------------
// .bin file readers
// ---------------------------------------------------------------------------

function readBinEdgeTable(buf, off, nFeatures) {
  const edges = [];
  for (let i = 0; i < nFeatures; i++) {
    const nEdges = buf.readUInt16LE(off);
    off += 2;
    const vals = new Float64Array(nEdges);
    for (let j = 0; j < nEdges; j++) {
      vals[j] = buf.readDoubleLE(off);
      off += 8;
    }
    edges.push(vals);
  }
  return { edges, off };
}

function loadSurvive(filePath) {
  const buf = fs.readFileSync(filePath);
  const magic = buf.toString('ascii', 0, 4);
  if (magic !== 'SURV') throw new Error(`${filePath}: bad magic ${magic}`);
  let off = 7;
  const baseline = buf.readDoubleLE(off); off += 8;
  const lr = buf.readDoubleLE(off); off += 8;
  const nTrees = buf.readUInt32LE(off); off += 4;
  const nodeCounts = new Array(nTrees);
  for (let i = 0; i < nTrees; i++) { nodeCounts[i] = buf.readUInt16LE(off); off += 2; }
  const nodeBlobs = [];
  for (const c of nodeCounts) {
    const size = c * NODE_SIZE;
    nodeBlobs.push(buf.subarray(off, off + size));
    off += size;
  }
  const nFeatures = buf.readUInt16LE(5);
  const { edges: binEdges } = readBinEdgeTable(buf, off, nFeatures);
  return { nFeatures, baseline, lr, nTrees, nodeBlobs, binEdges };
}

function loadRelane(filePath) {
  const buf = fs.readFileSync(filePath);
  const magic = buf.toString('ascii', 0, 4);
  if (magic !== 'RLAN') throw new Error(`${filePath}: bad magic ${magic}`);
  const nFeatures = buf.readUInt16LE(5);
  const nClasses = buf.readUInt8(7);
  let off = 8;
  const classes = [];
  for (let i = 0; i < nClasses; i++) { classes.push(buf.readUInt8(off)); off += 1; }
  const baseline = [];
  for (let i = 0; i < nClasses; i++) { baseline.push(buf.readDoubleLE(off)); off += 8; }
  const lr = buf.readDoubleLE(off); off += 8;
  const nIters = buf.readUInt32LE(off); off += 4;
  const total = nIters * nClasses;
  const nodeCountsFlat = new Array(total);
  for (let i = 0; i < total; i++) { nodeCountsFlat[i] = buf.readUInt16LE(off); off += 2; }
  const nodeBlobsFlat = [];
  for (const c of nodeCountsFlat) {
    const size = c * NODE_SIZE;
    nodeBlobsFlat.push(buf.subarray(off, off + size));
    off += size;
  }
  // regroup flat (iteration-major, class-major) list -> [iter][class]
  const iterBlobs = [];
  for (let i = 0; i < nIters; i++) {
    iterBlobs.push(nodeBlobsFlat.slice(i * nClasses, (i + 1) * nClasses));
  }
  const { edges: binEdges } = readBinEdgeTable(buf, off, nFeatures);
  return { nFeatures, nClasses, classes, baseline, lr, nIters, iterBlobs, binEdges };
}

// ---------------------------------------------------------------------------
// Traversal
// ---------------------------------------------------------------------------

/**
 * bin = smallest index i such that x <= edges[i], i.e. searchsorted(edges,
 * x, side='left'), clamped to [0, 255]. MUST be side='left' -- side='right'
 * silently produces a different, still-plausible-looking wrong traversal
 * (SPEC.md §3.5 step 2).
 */
function searchsortedLeft(edges, x) {
  let lo = 0, hi = edges.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (edges[mid] < x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

function rebinRow(xRow, binEdgesPerFeature) {
  const n = xRow.length;
  const out = new Uint8Array(n);
  for (let j = 0; j < n; j++) {
    const b = searchsortedLeft(binEdgesPerFeature[j], xRow[j]);
    out[j] = b < 0 ? 0 : (b > 255 ? 255 : b);
  }
  return out;
}

function traverse(nodeBlob, binRow) {
  let off = 0;
  for (;;) {
    const node = readNode(nodeBlob, off);
    if (node.flags & 1) return node.value; // is_leaf
    const goLeft = binRow[node.feat] <= node.binThr;
    off = (goLeft ? node.left : node.right) * NODE_SIZE;
  }
}

/**
 * raw = baseline + sum(tree leaf values), trees summed tree-index order
 * 0..n_trees-1 (DETERMINISM_CONTRACT.md §1's fixed summation order). Leaf
 * values already include learning_rate -- do not re-multiply.
 */
function predictSurviveRawRow(model, binRow) {
  let out = model.baseline;
  for (const blob of model.nodeBlobs) {
    out += traverse(blob, binRow);
  }
  return out;
}

/**
 * raw[c] = baseline[c] + sum over iterations 0..n_iters-1 of that
 * iteration's class-c tree leaf value (iteration order = fixed summation
 * order, matching the .bin file's own iteration-major layout).
 */
function predictRelaneRawRow(model, binRow) {
  const raw = model.baseline.slice();
  for (const itBlobs of model.iterBlobs) {
    for (let c = 0; c < model.nClasses; c++) {
      raw[c] += traverse(itBlobs[c], binRow);
    }
  }
  return raw;
}

function argmax(arr) {
  let bi = 0, bv = arr[0];
  for (let i = 1; i < arr.length; i++) if (arr[i] > bv) { bv = arr[i]; bi = i; }
  return bi;
}

// ---------------------------------------------------------------------------
// Public backend
// ---------------------------------------------------------------------------

class ModelBackend {
  constructor(survive, relane) {
    this.survive = survive; // {tier: model}
    this.relane = relane; // {tier: {family: model}}
  }

  static load(dataDir) {
    const survive = {};
    for (const t of TIERS) survive[t] = loadSurvive(path.join(dataDir, `survive_${t}.bin`));
    const relane = {};
    for (const t of TIERS) {
      relane[t] = {};
      for (const fam of Object.keys(FAMILIES)) {
        const p = path.join(dataDir, `relane_${fam}_${t}.bin`);
        if (fs.existsSync(p)) relane[t][fam] = loadRelane(p);
      }
    }
    return new ModelBackend(survive, relane);
  }

  static loadDefault() {
    return ModelBackend.load(DEFAULT_DATA_DIR);
  }

  /**
   * X: array of Float64Array rows (n x 59). Returns survive_proba
   * (array[n] of numbers in [0,1]), via the portable sigmoid -- pooling/
   * NMS need the real probability, not just the raw>=0 discrete decision
   * (DETERMINISM_CONTRACT.md §3).
   */
  predictSurvive(tier, X) {
    const model = this.survive[tier];
    const out = new Array(X.length);
    for (let i = 0; i < X.length; i++) {
      const binRow = rebinRow(X[i], model.binEdges);
      const raw = predictSurviveRawRow(model, binRow);
      out[i] = sigmoid(raw);
    }
    return out;
  }

  /** Fixed at 0.5 for all tiers -- the packed-GBM model's own validated
   * operating point (SPEC.md §6). */
  surviveThreshold(_tier) {
    return 0.5;
  }

  /** hard: no NMS, medium: 180ms, easy: 250ms (SPEC.md §6)
   * -- also fixed, not swept per model instance the way the scorecard
   * backend's knobs are (see backend_scorecard.js). */
  nmsGap(tier) {
    return { hard: null, medium: 180, easy: 250 }[tier];
  }

  /**
   * X: array of Float64Array rows for the surviving family notes. Returns
   * {finalLane: [str, ...], confidence: [number, ...]}. The lane DECISION
   * uses argmax(raw) directly (equals argmax(softmax(raw)), no exp needed
   * for the discrete choice); the CONFIDENCE value (needed downstream by
   * relane-pool's weighted sum and chord-merge's max) uses the portable
   * softmax.
   */
  predictRelane(tier, family, X) {
    const model = this.relane[tier][family];
    const lanesList = FAMILIES[family];
    const finalLane = new Array(X.length);
    const confidence = new Array(X.length);
    for (let i = 0; i < X.length; i++) {
      const binRow = rebinRow(X[i], model.binEdges);
      const raw = predictRelaneRawRow(model, binRow);
      const argmaxCol = argmax(raw);
      const proba = softmax(raw);
      finalLane[i] = lanesList[model.classes[argmaxCol]];
      confidence[i] = proba[argmaxCol];
    }
    return { finalLane, confidence };
  }
}

module.exports = { ModelBackend, TIERS, FAMILIES, DEFAULT_DATA_DIR, searchsortedLeft, readFloat16LE };
