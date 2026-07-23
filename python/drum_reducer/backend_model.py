"""
Packed-GBM model backend (backend A: lossless, rb4_test edit_rate 0.1703).

Reads the 9 `.bin` files in data/model/ directly with `struct` + `numpy` --
no sklearn, no pickle. This is the byte format SPEC.md §3 documents; see
that doc for the full field-by-field layout. Summary:

  SURV (survive_{tier}.bin): one binary head. magic/version/n_features,
  baseline(f64), learning_rate(f64), n_trees(u32), node_counts(u16 x n_trees),
  concatenated tree node bytes (tree 0 first), then a per-feature bin-edge
  table (f64, ascending).

  RLAN (relane_{family}_{tier}.bin): one multiclass head. Same header shape
  plus n_classes/classes_[] (the "column c predicts lane classes_[c]"
  indirection -- a lane can be entirely absent from a tier/family's training
  data, so n_classes can be < len(family lane list)), per-class baseline,
  and node_counts/blobs laid out ITERATION-major then CLASS-major.

Node struct (7 bytes, `<BBBBBe`): feature_idx(u8) bin_threshold(u8) left(u8)
right(u8) flags(u8) value(f16). flags bit0=is_leaf, bit1=missing_go_to_left
(unused here -- none of the fixtures exercise a missing feature value, kept
for byte-format fidelity only). Leaf `value` already has the model's
learning_rate baked in -- never re-multiply it (see SPEC.md §3.5 step 4,
the "double-shrinkage" trap that bit the original packer once).

Model-backend interface (what decode.py/reduce.py consume, and what the
scorecard backend (S2) and the JS port must match):
  - `predict_survive(tier, X) -> np.ndarray[n]` of survive_proba in [0, 1].
    Used for the pooled-mean survive decision AND as the NMS keep_score.
  - `predict_relane(tier, family, X) -> (final_lane: List[str], confidence:
    np.ndarray[n])`. `final_lane[i]` is one of that family's lane names;
    `confidence[i]` is the winning class's softmax proba, used by
    relane-pool (summed per candidate lane) and chord-merge (max).
"""

import os
import struct

import numpy as np

from .portable_exp import sigmoid, softmax

NODE_STRUCT = struct.Struct("<BBBBBe")  # feat, bin_thr, left, right, flags, value(f16)

TIERS = ["hard", "medium", "easy"]
FAMILIES = {"cymbal": ["hihat", "open-hat", "crash", "ride"], "tom": ["high-tom", "mid-tom", "floor-tom"]}

_DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "model")
)


# ---------------------------------------------------------------------------
# .bin file readers
# ---------------------------------------------------------------------------


def _read_bin_edge_table(buf, off, n_features):
    edges = []
    for _ in range(n_features):
        (n_edges,) = struct.unpack_from("<H", buf, off)
        off += 2
        vals = struct.unpack_from(f"<{n_edges}d", buf, off)
        off += 8 * n_edges
        edges.append(np.asarray(vals, dtype=np.float64))
    return edges, off


def load_survive(path):
    with open(path, "rb") as f:
        buf = f.read()
    magic, version, n_features = struct.unpack_from("<4sBH", buf, 0)
    assert magic == b"SURV", f"{path}: bad magic {magic!r}"
    off = 7
    baseline, lr = struct.unpack_from("<dd", buf, off)
    off += 16
    (n_trees,) = struct.unpack_from("<I", buf, off)
    off += 4
    node_counts = struct.unpack_from(f"<{n_trees}H", buf, off)
    off += 2 * n_trees
    node_blobs = []
    for c in node_counts:
        size = c * NODE_STRUCT.size
        node_blobs.append(buf[off:off + size])
        off += size
    bin_edges, off = _read_bin_edge_table(buf, off, n_features)
    return {
        "n_features": n_features, "baseline": baseline, "lr": lr,
        "n_trees": n_trees, "node_blobs": node_blobs, "bin_edges": bin_edges,
    }


def load_relane(path):
    with open(path, "rb") as f:
        buf = f.read()
    magic, version, n_features, n_classes = struct.unpack_from("<4sBHB", buf, 0)
    assert magic == b"RLAN", f"{path}: bad magic {magic!r}"
    off = 8
    classes_ = list(struct.unpack_from(f"<{n_classes}B", buf, off))
    off += n_classes
    baseline = list(struct.unpack_from(f"<{n_classes}d", buf, off))
    off += 8 * n_classes
    (lr,) = struct.unpack_from("<d", buf, off)
    off += 8
    (n_iters,) = struct.unpack_from("<I", buf, off)
    off += 4
    total = n_iters * n_classes
    node_counts_flat = struct.unpack_from(f"<{total}H", buf, off)
    off += 2 * total
    node_blobs_flat = []
    for c in node_counts_flat:
        size = c * NODE_STRUCT.size
        node_blobs_flat.append(buf[off:off + size])
        off += size
    # regroup flat (iteration-major, class-major) list -> [iter][class]
    iter_blobs = [node_blobs_flat[i * n_classes:(i + 1) * n_classes] for i in range(n_iters)]
    bin_edges, off = _read_bin_edge_table(buf, off, n_features)
    return {
        "n_features": n_features, "n_classes": n_classes, "classes_": classes_,
        "baseline": baseline, "lr": lr, "n_iters": n_iters,
        "iter_blobs": iter_blobs, "bin_edges": bin_edges,
    }


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


def rebin(X, bin_edges):
    """Re-bin raw feature values into 0-255 bin indices: bin = smallest
    index i such that x <= edges[i], i.e. np.searchsorted(edges, x,
    side='left'), clamped to [0, 255]. MUST be side='left' -- side='right'
    silently produces a different, still-plausible-looking wrong traversal
    (SPEC.md §3.5 step 2)."""
    n, d = X.shape
    Xb = np.empty((n, d), dtype=np.uint8)
    for j in range(d):
        Xb[:, j] = np.clip(np.searchsorted(bin_edges[j], X[:, j], side="left"), 0, 255)
    return Xb


def _traverse(node_blob, bin_row):
    off = 0
    while True:
        feat, bin_thr, left, right, flags, val = NODE_STRUCT.unpack_from(node_blob, off)
        if flags & 1:  # is_leaf
            return float(val)
        go_left = bin_row[feat] <= bin_thr
        off = (left if go_left else right) * NODE_STRUCT.size


def _predict_survive_raw(model, Xb):
    """raw = baseline + sum(tree leaf values), trees summed tree-index
    order 0..n_trees-1 (DETERMINISM_CONTRACT.md §1's fixed summation order).
    Leaf values already include learning_rate -- do not re-multiply."""
    n = Xb.shape[0]
    out = np.full(n, model["baseline"], dtype=np.float64)
    for blob in model["node_blobs"]:
        for i in range(n):
            out[i] += _traverse(blob, Xb[i])
    return out


def _predict_relane_raw(model, Xb):
    """raw[:, c] = baseline[c] + sum over iterations 0..n_iters-1 of that
    iteration's class-c tree leaf value (iteration order = fixed summation
    order, matching the .bin file's own iteration-major layout)."""
    n = Xb.shape[0]
    n_classes = model["n_classes"]
    raw = np.tile(np.asarray(model["baseline"], dtype=np.float64), (n, 1))
    for it_blobs in model["iter_blobs"]:
        for c in range(n_classes):
            blob = it_blobs[c]
            for i in range(n):
                raw[i, c] += _traverse(blob, Xb[i])
    return raw


# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------


class ModelBackend:
    """Backend A: the packed-GBM evaluator. See module docstring for the
    interface `reduce.py`/`decode.py` call."""

    def __init__(self, survive, relane):
        self.survive = survive  # {tier: model_dict}
        self.relane = relane    # {tier: {family: model_dict}}

    @classmethod
    def load(cls, data_dir):
        survive = {t: load_survive(os.path.join(data_dir, f"survive_{t}.bin")) for t in TIERS}
        relane = {t: {} for t in TIERS}
        for t in TIERS:
            for fam in FAMILIES:
                p = os.path.join(data_dir, f"relane_{fam}_{t}.bin")
                if os.path.exists(p):
                    relane[t][fam] = load_relane(p)
        return cls(survive, relane)

    @classmethod
    def load_default(cls):
        return cls.load(_DEFAULT_DATA_DIR)

    def predict_survive(self, tier, X):
        """Returns survive_proba (np.ndarray[n], float64 in [0,1]), via the
        portable sigmoid -- pooling/NMS need the real probability, not just
        the raw>=0 discrete decision (DETERMINISM_CONTRACT.md §3)."""
        model = self.survive[tier]
        Xb = rebin(X, model["bin_edges"])
        raw = _predict_survive_raw(model, Xb)
        return sigmoid(raw)

    def survive_threshold(self, tier):
        """Fixed at 0.5 for all tiers -- the packed-GBM model's own
        validated operating point (SPEC.md §6)."""
        return 0.5

    def nms_gap(self, tier):
        """hard: no NMS, medium: 180ms, easy: 250ms (SPEC.md §6)
        -- also fixed, not swept per model instance the way the
        scorecard backend's knobs are (see backend_scorecard.py)."""
        return {"hard": None, "medium": 180, "easy": 250}[tier]

    def predict_relane(self, tier, family, X):
        """Returns (final_lane: List[str] len n, confidence: np.ndarray[n]).
        The lane DECISION uses argmax(raw) directly (equals argmax(softmax(raw)),
        no exp needed for the discrete choice); the CONFIDENCE value (needed
        downstream by relane-pool's weighted sum and chord-merge's max) uses
        the portable softmax."""
        model = self.relane[tier][family]
        lanes_list = FAMILIES[family]
        Xb = rebin(X, model["bin_edges"])
        raw = _predict_relane_raw(model, Xb)
        argmax_col = np.argmax(raw, axis=1)
        proba = softmax(raw)
        final_lane = [lanes_list[model["classes_"][c]] for c in argmax_col]
        confidence = proba[np.arange(len(argmax_col)), argmax_col]
        return final_lane, confidence
