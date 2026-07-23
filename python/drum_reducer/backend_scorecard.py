"""
Auditable scorecard model backend (backend B: DETERMINISM_CONTRACT.md §4).
Not lossless (accepts ~+0.04 edit_rate vs backend A's 0.1703) -- the point is
that every rule is a small, readable, INTEGER table a human can check by
hand, instead of an opaque tree ensemble.

Reads `data/scorecard/scorecard.json`, produced by
`../../tools/build_scorecard.py`: a depth-1 refit of the survive GBMs +
depth-2 relane trees on bin-indexed features, plus this module's
survive_threshold/nms_gap knobs, selected on a held-out val split and
written into scorecard.json's `decode` block -- NOT hardcoded here, so both
this module and the JS port read the same validated numbers from one
artifact.

Survive: `points = base_points + sum_f feature_points[f](bin(x_f))`, summed
in feature-index order (feature_names.json order -- DETERMINISM_CONTRACT.md
§1's fixed summation order). `predict_survive` returns this integer total
per note (as a float64 array -- values are always integers; reduce.py's
shared decode.survive_pool takes the arithmetic mean of these across a
groove-pool group and compares against `survive_threshold(tier)`, which for
this backend is the per-tier T_tier read from scorecard.json's `decode`
block -- mean(points) >= T_tier is mathematically identical to sum(points)
>= n*T_tier, contract §4's integer-exact-mean pooling; see reduce.py's
comment for why this backend does not need its own pooling function).

Relane: walk the tier/family's depth-2 tree (nested dict: internal nodes
`{feature_idx, threshold}`, leaves `{class_counts}`); go left iff
`bin_index(x[feature_idx]) <= threshold`, where bin_index is computed via
the SHARED `relane_bin_edges` table (searchsorted side='left', same
convention as the survive tables and SPEC.md §3.5's packed-GBM
rebin). At the leaf, `final_lane = argmax(class_counts)` (ties broken by
lowest LANE_INDEX, matching DETERMINISM_CONTRACT.md §2's tie-break style),
`confidence = winner_count - runnerup_count` (an integer >= 0 -- exactly
contract §4's "integer margin of the winning class's summed integer leaf
score over the runner-up").
"""

import json
import os

import numpy as np

from .decode import LANE_INDEX

TIERS = ["hard", "medium", "easy"]

_DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "scorecard")
)


class ScorecardBackend:
    """Backend B: the auditable integer scorecard. See module docstring for
    the interface `reduce.py`/`decode.py` call (mirrors ModelBackend's)."""

    def __init__(self, scorecard):
        self.scorecard = scorecard
        self.feature_names = scorecard["feature_names"]
        self.survive = scorecard["survive"]  # {tier: {base_points, T_tier, features}}
        self.relane = scorecard["relane"]    # {tier: {family: {lanes, observed_classes, tree}}}
        self.relane_bin_edges = [np.asarray(e, dtype=np.float64) for e in scorecard["relane_bin_edges"]]
        self.decode_knobs = scorecard.get("decode", {})  # {tier: {"T_tier": int, "nms_gap": float|None}}

        # Precompute per-tier, per-feature-index (fixed order) (edges, points)
        # numpy arrays so predict_survive is a tight loop, not a dict lookup
        # per row.
        self._survive_tables = {}
        for tier, sc in self.survive.items():
            per_feat = []
            for fname in self.feature_names:
                fd = sc["features"][fname]
                edges = np.asarray(fd["bin_edges"], dtype=np.float64)
                points = np.asarray(fd["points"], dtype=np.float64)  # int-valued, float64 dtype for numpy math
                per_feat.append((edges, points))
            self._survive_tables[tier] = per_feat

    @classmethod
    def load(cls, data_dir):
        with open(os.path.join(data_dir, "scorecard.json")) as f:
            scorecard = json.load(f)
        return cls(scorecard)

    @classmethod
    def load_default(cls):
        return cls.load(_DEFAULT_DATA_DIR)

    # -- survive --------------------------------------------------------

    def predict_survive(self, tier, X):
        """Returns per-note integer points (np.ndarray[n], float64 dtype,
        integer-valued). `base_points + sum_f points_table[f][bin(x_f)]`,
        features summed in FEATURE_NAMES order (index 0..58)."""
        sc = self.survive[tier]
        tables = self._survive_tables[tier]
        n = X.shape[0]
        total = np.full(n, float(sc["base_points"]), dtype=np.float64)
        for j, (edges, points) in enumerate(tables):
            if len(points) == 1:
                total += points[0]
                continue
            idx = np.searchsorted(edges, X[:, j], side="left")
            idx = np.clip(idx, 0, len(points) - 1)
            total += points[idx]
        return total

    def survive_threshold(self, tier):
        """Per-tier integer T_tier, selected on a held-out val split (see
        tools/build_scorecard.py's decode-knob tuning step) and stored in scorecard.json's
        `decode` block -- NOT the raw T_tier from the quantization step
        alone, which only maps the depth-1 model's OWN 0.5-probability
        boundary into integer-points space; this is the scorecard's actual
        validated operating point for the shipped decode."""
        knobs = self.decode_knobs.get(tier)
        if knobs is not None:
            return float(knobs["T_tier"])
        return float(self.survive[tier]["T_tier"])  # fallback: literal boundary mapping

    def nms_gap(self, tier):
        knobs = self.decode_knobs.get(tier)
        if knobs is not None:
            return knobs["nms_gap"]
        return None

    # -- relane -----------------------------------------------------------

    def _bin_index(self, feature_idx, x):
        edges = self.relane_bin_edges[feature_idx]
        return int(np.clip(np.searchsorted(edges, x, side="left"), 0, len(edges)))

    def _walk_tree(self, tree, xrow):
        node = tree
        while not node["leaf"]:
            f = node["feature_idx"]
            bidx = self._bin_index(f, xrow[f])
            node = node["left"] if bidx <= node["threshold"] else node["right"]
        counts = node["class_counts"]
        # winner = argmax(count); tie -> lowest LANE_INDEX (mirrors
        # DETERMINISM_CONTRACT.md §2's tie-break style for lane decisions).
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], LANE_INDEX.get(kv[0], 9)))
        winner_lane, winner_count = ranked[0]
        runnerup_count = ranked[1][1] if len(ranked) > 1 else 0
        confidence = winner_count - runnerup_count  # integer margin, >= 0
        return winner_lane, confidence

    def predict_relane(self, tier, family, X):
        """Returns (final_lane: List[str] len n, confidence: np.ndarray[n]
        of integer count-margins, float64 dtype)."""
        tree = self.relane[tier][family]["tree"]
        n = X.shape[0]
        lanes = [None] * n
        conf = np.zeros(n, dtype=np.float64)
        for i in range(n):
            lanes[i], conf[i] = self._walk_tree(tree, X[i])
        return lanes, conf
