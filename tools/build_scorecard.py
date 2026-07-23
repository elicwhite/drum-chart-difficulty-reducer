"""Standalone scorecard builder: train the auditable, integer scorecard
backend (DETERMINISM_CONTRACT.md §4) from ANY folder of parsed drum charts.

Given a folder of `*.json` charts (same schema as data/fixtures/charts/ --
see SPEC.md: `difficulties.expert` plus human-authored `hard`/`medium`/
`easy` reductions to learn from), this:

  1. Featurizes every chart with the package's own `python/drum_reducer/
     featurize.py` (the same 59-feature extractor `reduce()` uses at
     inference time -- no separate feature code here).
  2. Derives per-note SURVIVE/RELANE training labels from the chart's own
     human tiers (a family note SURVIVES a tier iff its family appears
     anywhere at that tick in the tier's human-authored reduction, and
     RELANEs to the GT lane where that's unambiguous -- see `label_rows`
     below for the exact rule).
  3. Fits a depth-1 `HistGradientBoostingClassifier` per tier (SURVIVE) and
     collapses its 200 stumps into one piecewise-constant integer points
     table per feature (lossless re-expression -- the only lossy step is
     the depth-1 refit itself), quantized to a fixed dynamic range, with
     zero-delta adjacent bins merged.
  4. Fits a depth-2 `DecisionTreeClassifier` per (tier, family) on
     bin-indexed features (RELANE) -- split thresholds are then plain
     integer bin indices, leaf scores are the tree's own integer per-class
     training counts.
  5. Tunes each tier's (T_tier, NMS-gap) operating point on a held-out val
     split of the SAME folder, through the shipped `drum_reducer.decode`
     pipeline (not a floating-point reimplementation of it) -- folded into
     this one script so a single run yields a complete, usable scorecard.
  6. Emits `scorecard.json` (machine-readable, loadable by
     `drum_reducer.backend_scorecard.ScorecardBackend`), `scorecard_rules.py`
     (human-readable rendering), and `AUDIT.md` (breakpoint-count honesty
     report per DETERMINISM_CONTRACT.md §4's auditability clause).

This tool has ZERO dependency on anything outside this repo (no pinned
model pickle, no hardcoded song split) -- it only imports the standalone
`drum_reducer` package next to it and `numpy`/`scikit-learn`. It does NOT
reproduce the shipped `data/scorecard/scorecard.json` (that one was
trained separately, on Harmonix's official reductions -- see the README's
"Rebuild the scorecard on your own charts" section); it trains a new one
from whatever folder you point it at.

Requires scikit-learn (fitting only -- `drum_reducer` itself, the
INFERENCE package, has no sklearn dependency, only numpy). See
requirements-dev.txt.

Usage (from the repo root):
    .venv/bin/python tools/build_scorecard.py \\
        --charts data/fixtures/charts --out build/scorecard.json
"""
import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.tree import DecisionTreeClassifier

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.dirname(TOOLS_DIR)
PKG_DIR = os.path.join(REF_DIR, "python")
sys.path.insert(0, PKG_DIR)

from drum_reducer import decode, editrate, featurize  # noqa: E402
from drum_reducer.backend_scorecard import ScorecardBackend  # noqa: E402

TIERS = ["hard", "medium", "easy"]
FAMILIES = featurize.FAMILIES  # {"cymbal": [...], "tom": [...]}
FAMILY_OF_LANE = featurize.FAMILY_OF_LANE
FEATURE_NAMES = featurize.FEATURE_NAMES
N_FEATS = len(FEATURE_NAMES)

GBM_PARAMS = dict(max_iter=200, learning_rate=0.08, max_depth=1, l2_regularization=1.0,
                   early_stopping=True, validation_fraction=0.1)
TARGET_RANGE = 10_000.0  # target total integer dynamic range for the points scale
RELANE_DEPTH = 2
RELANE_MIN_ROWS = 20  # below this, skip fitting that (tier, family)'s tree

SHIPPED_SCORECARD = os.path.normpath(os.path.join(REF_DIR, "data", "scorecard", "scorecard.json"))


def log(*a):
    print(*a)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# 0. Load charts, featurize, and label (vendored survive_T/relane_T logic).
# ---------------------------------------------------------------------------

def load_charts(folder):
    charts = {}
    n_skipped = 0
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(folder, fn)) as f:
            chart = json.load(f)
        diffs = chart.get("difficulties", {})
        if not diffs.get("expert") or not any(diffs.get(t) for t in TIERS):
            n_skipped += 1
            continue
        charts[fn[:-5]] = chart
    log(f"[load] {len(charts)} charts loaded from {folder} ({n_skipped} skipped: missing expert or all-tiers)")
    return charts


def label_rows(chart, rows):
    """Adds survive_{tier}/relane_{tier} to each featurize() row in place.

    Label rule: a family note (cymbal/tom) SURVIVES a tier iff its family
    appears anywhere at that tick in the tier's human GT reduction; it
    RELANES to the GT lane
    if the tick's family GT is unambiguous (its own lane if that's the one
    GT kept, else the sole GT lane present), else relane is None (ambiguous
    -- still counts as survive, excluded from relane training). A
    fixed-lane note (kick/snare/other) survives iff its own exact (tick,
    lane) appears in the tier's GT, and is never relaned. `tick` is
    round(ms / EPS_MS) throughout, matching featurize.py's own tick
    convention (chord grouping) so labels align 1:1 with the feature rows.
    """
    gt_family_lanes_by_tier = {}
    gt_exact_by_tier = {}
    for t in TIERS:
        diff = chart["difficulties"].get(t)
        by_tick_fam = collections.defaultdict(lambda: collections.defaultdict(set))
        exact = set()
        if diff:
            for ms, entries in diff["notes"]:
                tick = round(ms / featurize.EPS_MS)
                for e in entries:
                    lane = e["instrument"]
                    exact.add((tick, lane))
                    fam = FAMILY_OF_LANE.get(lane)
                    if fam:
                        by_tick_fam[tick][fam].add(lane)
        gt_family_lanes_by_tier[t] = by_tick_fam
        gt_exact_by_tier[t] = exact

    n_ambiguous = 0
    for r in rows:
        tick = round(r["ms"] / featurize.EPS_MS)
        lane = r["lane"]
        fam = FAMILY_OF_LANE.get(lane)
        for t in TIERS:
            if fam is None:
                r[f"survive_{t}"] = 1 if (tick, lane) in gt_exact_by_tier[t] else 0
                r[f"relane_{t}"] = None
                continue
            fam_lanes = gt_family_lanes_by_tier[t][tick].get(fam, set())
            r[f"survive_{t}"] = 1 if fam_lanes else 0
            if not fam_lanes:
                r[f"relane_{t}"] = None
            elif lane in fam_lanes:
                r[f"relane_{t}"] = lane
            elif len(fam_lanes) == 1:
                r[f"relane_{t}"] = next(iter(fam_lanes))
            else:
                r[f"relane_{t}"] = None
                n_ambiguous += 1
    return n_ambiguous


def featurize_charts(charts):
    """{song_id: (X, rows)}, skipping charts with no Expert notes."""
    per_song = {}
    n_ambig_total = 0
    for sid, chart in charts.items():
        X, names, rows = featurize.featurize(chart)
        assert names == FEATURE_NAMES
        if not rows:
            continue
        n_ambig_total += label_rows(chart, rows)
        per_song[sid] = (X, rows)
    log(f"[featurize] {len(per_song)} songs with Expert notes, {n_ambig_total} ambiguous "
        f"family-relane ticks total (dropped from relane training only)")
    return per_song


def split_songs(song_ids, val_frac, seed):
    ids = sorted(song_ids)
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_val = min(len(shuffled) - 1, max(1, round(len(shuffled) * val_frac))) if len(shuffled) > 1 else 0
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


def concat_rows(per_song, song_ids):
    Xs, rows_flat = [], []
    for sid in song_ids:  # fixed order for reproducibility
        X, rows = per_song[sid]
        Xs.append(X)
        rows_flat.extend(rows)
    X_all = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, N_FEATS))
    return X_all, rows_flat


# ---------------------------------------------------------------------------
# 1. Fit depth-1 survive GBMs (one per tier).
# ---------------------------------------------------------------------------

def fit_survive(tier, X_train, train_rows, seed):
    y = np.array([r[f"survive_{tier}"] for r in train_rows])
    clf = HistGradientBoostingClassifier(random_state=seed, **GBM_PARAMS)
    clf.fit(X_train, y)
    return clf


# ---------------------------------------------------------------------------
# 2. Collapse each tier's stumps into one piecewise-constant points function
#    PER FEATURE (lossless re-expression of the depth-1 model).
# ---------------------------------------------------------------------------

def collapse_stumps(clf):
    """Returns (baseline: float, bin_edges: list[np.ndarray], contrib:
    list[np.ndarray], n_trees_used: int). contrib[f][b] is feature f's total
    raw-score contribution when its value falls in bin b (edges from the
    GBM's own _bin_mapper, side='left' searchsorted convention)."""
    bin_edges = [np.asarray(e, dtype=np.float64) for e in clf._bin_mapper.bin_thresholds_]
    contrib = [np.zeros(len(e) + 1, dtype=np.float64) for e in bin_edges]
    n_trees_used = 0
    for it in clf._predictors:
        for tree in it:
            nodes = tree.nodes
            assert len(nodes) == 3, f"expected a depth-1 stump (3 nodes), got {len(nodes)}"
            root = nodes[0]
            f = int(root["feature_idx"])
            thr = int(root["bin_threshold"])  # go_left iff bin(x) <= thr
            left_leaf = nodes[int(root["left"])]
            right_leaf = nodes[int(root["right"])]
            assert left_leaf["is_leaf"] and right_leaf["is_leaf"]
            contrib[f][: thr + 1] += float(left_leaf["value"])
            contrib[f][thr + 1:] += float(right_leaf["value"])
            n_trees_used += 1
    baseline = float(np.asarray(clf._baseline_prediction).ravel()[0])
    return baseline, bin_edges, contrib, n_trees_used


# ---------------------------------------------------------------------------
# 3. Quantize to integers, merge zero-delta bins, derive per-tier T_tier.
# ---------------------------------------------------------------------------

def quantize_tier(baseline, edges, contrib):
    mins = np.array([c.min() for c in contrib])
    maxs = np.array([c.max() for c in contrib])
    raw_range = float((maxs - mins).sum())
    scale = TARGET_RANGE / raw_range if raw_range > 0 else 1.0
    base_adj = baseline + float(mins.sum())
    quant_base = int(round(scale * base_adj))
    quant_contrib = [np.round(scale * (c - m)).astype(np.int64) for c, m in zip(contrib, mins)]
    return scale, quant_base, quant_contrib


def merge_zero_delta_bins(edges, qcontrib):
    """Collapse contiguous bins with identical integer points into one
    logical bin. Preserves searchsorted(side='left') decode semantics
    exactly (DETERMINISM_CONTRACT.md §4)."""
    edges = list(edges)
    qcontrib = [int(v) for v in qcontrib]
    new_edges, new_points = [], [qcontrib[0]]
    for i in range(1, len(qcontrib)):
        if qcontrib[i] != new_points[-1]:
            new_edges.append(float(edges[i - 1]))
            new_points.append(qcontrib[i])
    return new_edges, new_points


def find_best_T_tier(points_per_row, float_decision):
    """Integer threshold T minimizing mismatches between (points >= T) and
    the float depth-1 model's OWN decision (proba >= 0.5) -- maps the
    model's probability decision boundary into integer-points space,
    correcting for the small residual error from per-bin integer rounding."""
    points_per_row = np.asarray(points_per_row, dtype=np.int64)
    dec = np.asarray(float_decision, dtype=bool)
    order = np.argsort(points_per_row, kind="stable")
    p_sorted = points_per_row[order]
    dec_sorted = dec[order]
    N = len(points_per_row)
    cum_dec = np.concatenate([[0], np.cumsum(dec_sorted)])
    total_true = int(cum_dec[-1])
    uniq_vals, first_idx = np.unique(p_sorted, return_index=True)
    k = first_idx
    mism = 2 * cum_dec[k] - k - total_true + N
    best = int(np.argmin(mism))
    return int(uniq_vals[best]), int(mism[best]), N


def build_survive_scorecard(X_train, train_rows, seed):
    log("\n" + "=" * 70)
    log("STEP 1-3: fit depth-1 survive GBMs, collapse, quantize, derive T_tier")
    log("=" * 70)
    survive_scorecard = {}
    quant_scale = None
    for tier in TIERS:
        t0 = time.time()
        clf = fit_survive(tier, X_train, train_rows, seed)
        n_trees = sum(len(it) for it in clf._predictors)
        baseline, edges, contrib, n_trees_used = collapse_stumps(clf)
        scale, quant_base, quant_contrib = quantize_tier(baseline, edges, contrib)
        if quant_scale is None:
            quant_scale = scale  # one shared scale across tiers for cross-tier comparability
        merged, breakpoint_counts = [], []
        for f in range(N_FEATS):
            e, p = merge_zero_delta_bins(edges[f], quant_contrib[f])
            merged.append((e, p))
            breakpoint_counts.append(len(p))

        def points_for_rows(Xrows, merged=merged, quant_base=quant_base):
            Xb_idx = np.zeros((len(Xrows), N_FEATS), dtype=np.int64)
            for f in range(N_FEATS):
                e, _p = merged[f]
                if e:
                    Xb_idx[:, f] = np.searchsorted(np.asarray(e, dtype=np.float64), Xrows[:, f], side="left")
            total = np.full(len(Xrows), quant_base, dtype=np.int64)
            for f in range(N_FEATS):
                _e, p = merged[f]
                total += np.asarray(p, dtype=np.int64)[Xb_idx[:, f]]
            return total

        pts_train = points_for_rows(X_train)
        float_dec_train = clf.predict_proba(X_train)[:, 1] >= 0.5
        T_tier, best_mism, n_rows = find_best_T_tier(pts_train, float_dec_train)
        log(f"[survive/{tier}] n_trees={n_trees}(used={n_trees_used}) fit_time={time.time()-t0:.1f}s "
            f"scale={scale:.4f} base_points={quant_base} T_tier={T_tier} "
            f"train_mismatch_vs_float_model={best_mism}/{n_rows} ({100*best_mism/max(1,n_rows):.3f}%)")
        survive_scorecard[tier] = dict(base_points=quant_base, T_tier=T_tier,
                                        features={FEATURE_NAMES[f]: dict(bin_edges=merged[f][0], points=merged[f][1])
                                                  for f in range(N_FEATS)},
                                        breakpoint_counts=breakpoint_counts)
    log(f"[survive] shared quant_scale={quant_scale:.4f} (target dynamic range {TARGET_RANGE:.0f} points)")
    return survive_scorecard, quant_scale


# ---------------------------------------------------------------------------
# 4. Relane: depth-2 trees on bin-index features.
# ---------------------------------------------------------------------------

def tree_to_dict(clf, lanes, class_labels):
    """Nested dict: internal {feature_idx, feature_name, threshold(int bin
    idx), left, right}; leaf {leaf: true, class_counts}. threshold is an
    EXACT integer re-expression of sklearn's float split (X is already
    integer bin-indices, so sklearn's threshold is always i+0.5 for
    consecutive integers -- 'bin <= floor(thr)' is bit-identical)."""
    tree_ = clf.tree_

    def node(i):
        if tree_.children_left[i] == -1:
            # tree_.value stores class FRACTIONS per node in sklearn >= 1.8,
            # not raw counts -- multiply by weighted_n_node_samples to
            # recover integer training-sample counts.
            frac = tree_.value[i][0]
            n_node = tree_.weighted_n_node_samples[i]
            counts = frac * n_node
            class_counts = {lanes[class_labels[c]]: int(round(counts[c])) for c in range(len(class_labels))}
            return {"leaf": True, "class_counts": class_counts}
        f = int(tree_.feature[i])
        thr = int(np.floor(tree_.threshold[i]))
        return {"leaf": False, "feature_idx": f, "feature_name": FEATURE_NAMES[f], "threshold": thr,
                "left": node(tree_.children_left[i]), "right": node(tree_.children_right[i])}

    return node(0)


def build_relane_scorecard(X_train, train_rows, seed):
    log("\n" + "=" * 70)
    log("STEP 4: refit depth-2 relane trees on bin-index features")
    log("=" * 70)
    relane_bin_mapper = _BinMapper(n_bins=256, random_state=seed)
    Xb_train = relane_bin_mapper.fit_transform(X_train)
    relane_bin_edges = [np.asarray(e, dtype=np.float64) for e in relane_bin_mapper.bin_thresholds_]

    relane_scorecard = {}
    for tier in TIERS:
        for fam_name, lanes in FAMILIES.items():
            idx = [i for i, r in enumerate(train_rows)
                   if r["family"] == fam_name and r.get(f"survive_{tier}") == 1
                   and r.get(f"relane_{tier}") is not None]
            if len(idx) < RELANE_MIN_ROWS:
                log(f"[relane/{tier}/{fam_name}] only {len(idx)} rows (<{RELANE_MIN_ROWS}), skipping")
                continue
            rows = [train_rows[i] for i in idx]
            Xb = Xb_train[idx]
            lane_idx = {l: i for i, l in enumerate(lanes)}
            y = np.array([lane_idx[r[f"relane_{tier}"]] for r in rows])
            clf = DecisionTreeClassifier(max_depth=RELANE_DEPTH, random_state=seed)
            clf.fit(Xb, y)
            tree_dict = tree_to_dict(clf, lanes, clf.classes_)
            relane_scorecard.setdefault(tier, {})[fam_name] = dict(
                lanes=lanes, tree=tree_dict, observed_classes=[lanes[c] for c in clf.classes_])
            log(f"[relane/{tier}/{fam_name}] n_train_rows={len(rows)} n_nodes={clf.tree_.node_count} "
                f"observed_classes={[lanes[c] for c in clf.classes_]}")
    return relane_scorecard, relane_bin_edges


# ---------------------------------------------------------------------------
# 5. Tune (T_tier, NMS-gap) per tier on the val split, through the SHIPPED
#    drum_reducer decode pipeline (folded in from tune_and_measure_scorecard.py).
# ---------------------------------------------------------------------------

def precompute_song(backend, chart):
    X, names, rows = featurize.featurize(chart)
    assert names == FEATURE_NAMES
    if not rows:
        return None
    ms_to_measure, measure_to_ms = decode.build_measure_clock(
        chart.get("tempos") or [], chart.get("timeSignatures") or [])
    expert_notes = [decode.Note(r["ms"], r["lane"]) for r in rows]
    clusters, _n = decode.expert_groove_clusters(expert_notes, ms_to_measure)

    fam_idx = {fam: [i for i, r in enumerate(rows) if r["family"] == fam] for fam in FAMILIES}
    per_tier = {}
    for tier in TIERS:
        survive_points = backend.predict_survive(tier, X)
        relane_lane = [r["lane"] for r in rows]
        relane_conf = [1.0] * len(rows)
        for fam, idxs in fam_idx.items():
            if not idxs or fam not in backend.relane.get(tier, {}):
                continue
            lanes_out, conf_out = backend.predict_relane(tier, fam, X[idxs])
            for k, i in enumerate(idxs):
                relane_lane[i] = lanes_out[k]
                relane_conf[i] = float(conf_out[k])
        gt = editrate.notes_from_difficulty(chart["difficulties"].get(tier))
        per_tier[tier] = dict(survive_points=survive_points, relane_lane=relane_lane,
                               relane_conf=relane_conf, gt=gt)
    return dict(rows=rows, ms_to_measure=ms_to_measure, measure_to_ms=measure_to_ms,
                clusters=clusters, per_tier=per_tier)


def score_song_tier(pc, tier, T_tier, gap):
    """Runs the same steps reduce.py's decode pipeline does, given
    precomputed survive_points/relane predictions and a candidate
    (T_tier, gap) -- avoids re-featurizing/re-predicting per grid point."""
    rows, ms_to_measure, measure_to_ms, clusters = pc["rows"], pc["ms_to_measure"], pc["measure_to_ms"], pc["clusters"]
    pt = pc["per_tier"][tier]
    pooled = decode.survive_pool(rows, pt["survive_points"], ms_to_measure, clusters)
    survive = [p >= T_tier for p in pooled]
    if gap:
        survive = decode.family_nms(rows, survive, pooled, gap)
    final_lane = decode.relane_pool(rows, pt["relane_lane"], pt["relane_conf"], ms_to_measure, clusters)
    cand = decode.chord_merge(rows, survive, final_lane, pt["relane_conf"])
    if clusters:
        rbm = decode.reduced_groove_by_measure(cand, ms_to_measure)
        cand = decode.canonicalize(cand, clusters, rbm, ms_to_measure, measure_to_ms)
    cand_notes = [editrate.Note(n.ms, n.lane) for n in cand]
    _rate, ops = editrate.edit_rate(cand_notes, pt["gt"])
    n_edits = ops["insert"] + ops["delete"] + ops["lane_move"] + ops["slot_move"]
    return n_edits, len(pt["gt"])


def edit_rate_over(songs_pre, tier, T_tier, gap):
    edits = gtn = 0
    for pc in songs_pre.values():
        if pc is None:
            continue
        e, n = score_song_tier(pc, tier, T_tier, gap)
        edits += e
        gtn += n
    return (edits / gtn) if gtn else None


def t_candidates(t_base, quant_scale):
    import math
    thr_grid = [round(x, 2) for x in np.arange(0.30, 0.71, 0.05)]
    out = set()
    for p in thr_grid:
        logit = math.log(p / (1 - p))
        out.add(int(round(t_base + quant_scale * logit)))
    return sorted(out)


GAP_GRID = [None, 90, 130, 180, 250, 350]


def tune_decode_knobs(scorecard_json, val_charts):
    log("\n" + "=" * 70)
    log("STEP 5: tune (T_tier, nms_gap) per tier on the val split")
    log("=" * 70)
    backend = ScorecardBackend(scorecard_json)
    val_pre = {sid: precompute_song(backend, chart) for sid, chart in val_charts.items()}
    chosen = {}
    for tier in TIERS:
        t_base = scorecard_json["survive"][tier]["T_tier"]
        candidates = t_candidates(t_base, scorecard_json["quant_scale"])
        best = None
        for T in candidates:
            for gap in GAP_GRID:
                r = edit_rate_over(val_pre, tier, T, gap)
                if r is None:
                    continue
                if best is None or r < best[0]:
                    best = (r, T, gap)
        if best is None:
            log(f"[tune/{tier}] no val ground truth for this tier -- keeping literal T_tier={t_base}, gap=None")
            chosen[tier] = {"T_tier": t_base, "nms_gap": None}
        else:
            chosen[tier] = {"T_tier": best[1], "nms_gap": best[2]}
            log(f"[tune/{tier}] best val edit_rate={best[0]:.4f} at T_tier={best[1]} "
                f"(literal boundary was {t_base}) nms_gap={best[2]}")
    return chosen


# ---------------------------------------------------------------------------
# 6. Emit artifacts.
# ---------------------------------------------------------------------------

def fmt_points_table(fname, edges, points):
    lines = [f"    # {fname}"]
    if len(points) == 1:
        lines.append(f"    #   constant, {points[0]:+d} pts (feature unused by the model)")
        return lines
    lo = "-inf"
    for e, p in zip(edges, points[:-1]):
        lines.append(f"    #   {lo:>10} <= x < {e:<10.4g} -> {p:+d} pts")
        lo = f"{e:.4g}"
    lines.append(f"    #   {lo:>10} <= x            -> {points[-1]:+d} pts")
    return lines


def render_tree(node, indent):
    pad = "    " * indent
    if node["leaf"]:
        winner = max(node["class_counts"].items(), key=lambda kv: kv[1])
        others = sorted((v for k, v in node["class_counts"].items() if k != winner[0]), reverse=True)
        runnerup = others[0] if others else 0
        return [f"{pad}return {winner[0]!r}, {winner[1] - runnerup}  # class_counts={node['class_counts']}"]
    lines = [f"{pad}if bin_index({node['feature_name']!r}) <= {node['threshold']}:"]
    lines += render_tree(node["left"], indent + 1)
    lines.append(f"{pad}else:")
    lines += render_tree(node["right"], indent + 1)
    return lines


def write_rules_py(path, survive_scorecard, relane_scorecard, quant_scale):
    lines = [
        '"""GENERATED by drum-reducer-reference/tools/build_scorecard.py -- do not',
        "hand-edit; re-run the tool to regenerate. Human-readable rendering of the",
        "SURVIVE points tables + RELANE decision trees. See scorecard.json for the",
        "machine-readable version this is generated from, and",
        'DETERMINISM_CONTRACT.md Sec.4 for the decode semantics."""',
        "",
        f"QUANT_SCALE = {quant_scale!r}",
        "",
    ]
    for tier in TIERS:
        sc = survive_scorecard[tier]
        lines.append(f"# {'='*70}")
        lines.append(f"# SURVIVE / {tier}   base_points={sc['base_points']:+d}   T_tier={sc['T_tier']}")
        lines.append("# decision: points >= T_tier  (pooled: sum(points) >= n * T_tier)")
        lines.append(f"# {'='*70}")
        lines.append(f"SURVIVE_{tier.upper()}_BASE_POINTS = {sc['base_points']}")
        lines.append(f"SURVIVE_{tier.upper()}_T_TIER = {sc['T_tier']}")
        lines.append(f"SURVIVE_{tier.upper()}_POINTS = {{")
        for fname, fd in sc["features"].items():
            lines.extend(fmt_points_table(fname, fd["bin_edges"], fd["points"]))
            lines.append(f"    {fname!r}: ({fd['bin_edges']!r}, {fd['points']!r}),")
        lines.append("}")
        lines.append("")
    for tier in TIERS:
        for fam_name in relane_scorecard.get(tier, {}):
            entry = relane_scorecard[tier][fam_name]
            lines.append(f"# {'='*70}")
            lines.append(f"# RELANE / {tier} / {fam_name}   lanes={entry['lanes']}")
            lines.append("# returns (final_lane, confidence) where confidence = integer count margin")
            lines.append(f"# {'='*70}")
            lines.append(f"def relane_{tier}_{fam_name.replace('-', '_')}(bin_index):")
            lines.append("    # bin_index(feature_name) -> int, via relane_bin_edges + searchsorted(side='left')")
            lines.extend(render_tree(entry["tree"], 1))
            lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_audit_md(path, survive_scorecard, relane_scorecard, quant_scale, n_train_songs, n_val_songs):
    lines = [
        "# Scorecard audit (generated by build_scorecard.py)",
        "",
        f"Trained from a user-supplied chart folder ({n_train_songs} train songs, {n_val_songs} val songs). "
        f"quant_scale={quant_scale:.4f} (target dynamic range {TARGET_RANGE:.0f} points).",
        "",
        "## Survive: per-feature breakpoint counts (after merging zero-delta bins)",
        "",
        "| tier | feature | breakpoints (bins) | used? |",
        "|---|---|---:|---|",
    ]
    for tier in TIERS:
        sc = survive_scorecard[tier]
        for fname, fd in sc["features"].items():
            n_bins = len(fd["points"])
            lines.append(f"| {tier} | {fname} | {n_bins} | {'yes' if n_bins > 1 else 'no'} |")

    for tier in TIERS:
        sc = survive_scorecard[tier]
        used_feats = [fname for fname, fd in sc["features"].items() if len(fd["points"]) > 1]
        max_bins = max(len(fd["points"]) for fd in sc["features"].values())
        avg_bins_used = (np.mean([len(fd["points"]) for fname, fd in sc["features"].items() if fname in used_feats])
                          if used_feats else 0.0)
        lines.append("")
        lines.append(f"**{tier}**: {len(used_feats)}/{N_FEATS} features actually used "
                      f"(nonconstant after merge), base_points={sc['base_points']:+d}, T_tier={sc['T_tier']}, "
                      f"max breakpoint count on any single feature={max_bins}, "
                      f"avg breakpoint count on used features={avg_bins_used:.1f}")

    lines += ["", "## Relane: tree sizes", "", "| tier | family | n_nodes (depth<=2) |", "|---|---|---:|"]
    for tier in TIERS:
        for fam_name, entry in relane_scorecard.get(tier, {}).items():
            n_nodes = _count_nodes(entry["tree"])
            lines.append(f"| {tier} | {fam_name} | {n_nodes} |")

    max_bins_overall = max(len(fd["points"]) for sc in survive_scorecard.values() for fd in sc["features"].values())
    lines += [
        "",
        "## Honest readability assessment",
        "",
        f"Max breakpoint count on any single feature/tier is **{max_bins_overall}** rows. "
        + ("Most features collapse to a handful of rows after merging -- hand-auditable at a glance."
           if max_bins_overall <= 20 else
           "Several features exceed ~15 rows after merging -- readable as a lookup table with effort, but "
           "the 'read every rule at a glance' framing should be tempered for those features specifically."),
        "",
        "Relane trees are depth<=2 (<=7 nodes) -- small enough to read as nested if/else at a glance.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _count_nodes(node):
    if node["leaf"]:
        return 1
    return 1 + _count_nodes(node["left"]) + _count_nodes(node["right"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--charts", required=True, help="folder of parsed chart *.json files to train from")
    ap.add_argument("--out", default=os.path.join(REF_DIR, "build", "scorecard.json"),
                     help="output path for scorecard.json (default: drum-reducer-reference/build/scorecard.json)")
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of songs held out for decode-knob tuning")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = os.path.abspath(args.out)
    if out_path == SHIPPED_SCORECARD:
        raise SystemExit(f"refusing to overwrite the shipped artifact at {SHIPPED_SCORECARD} -- pass a different --out")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    charts = load_charts(args.charts)
    if len(charts) < 4:
        raise SystemExit(f"only {len(charts)} usable charts in {args.charts} -- need at least a few for train+val")
    per_song = featurize_charts(charts)

    train_songs, val_songs = split_songs(list(per_song.keys()), args.val_frac, args.seed)
    log(f"[split] {len(train_songs)} train songs, {len(val_songs)} val songs (val_frac={args.val_frac}, seed={args.seed})")

    X_train, train_rows = concat_rows(per_song, train_songs)
    log(f"[train] X_train shape={X_train.shape}")

    survive_scorecard, quant_scale = build_survive_scorecard(X_train, train_rows, args.seed)
    relane_scorecard, relane_bin_edges = build_relane_scorecard(X_train, train_rows, args.seed)

    scorecard_json = {
        "source": f"build_scorecard.py --charts {os.path.abspath(args.charts)}",
        "sklearn_version": __import__("sklearn").__version__,
        "quant_scale": quant_scale,
        "target_dynamic_range": TARGET_RANGE,
        "lane_vocab": list(featurize.LANE_VOCAB) + ["other"],
        "families": {k: list(v) for k, v in FAMILIES.items()},
        "feature_names": FEATURE_NAMES,
        "relane_bin_edges": [e.tolist() for e in relane_bin_edges],
        "survive": {
            tier: {
                "base_points": survive_scorecard[tier]["base_points"],
                "T_tier": survive_scorecard[tier]["T_tier"],
                "features": {
                    fname: {"bin_edges": fd["bin_edges"], "points": fd["points"]}
                    for fname, fd in survive_scorecard[tier]["features"].items()
                },
            }
            for tier in TIERS
        },
        "relane": {
            tier: {
                fam: {"lanes": relane_scorecard[tier][fam]["lanes"],
                      "observed_classes": relane_scorecard[tier][fam]["observed_classes"],
                      "tree": relane_scorecard[tier][fam]["tree"]}
                for fam in relane_scorecard.get(tier, {})
            }
            for tier in TIERS
        },
        "notes": {
            "survive_decision": "points = base_points + sum_f feature_points[f](searchsorted(bin_edges[f], x_f, side='left')); "
                                 "un-pooled: points >= T_tier; pooled (group size n): sum(points_i) >= n * T_tier.",
            "relane_decision": "walk tree from root; at each internal node go left iff "
                                "bin_index(x[feature_idx]) <= threshold (bin_index via relane_bin_edges, side='left'); "
                                "at a leaf, final_lane = argmax(class_counts), confidence = winner_count - runnerup_count.",
        },
    }

    val_charts = {sid: charts[sid] for sid in val_songs}
    decode_knobs = tune_decode_knobs(scorecard_json, val_charts)
    scorecard_json["decode"] = decode_knobs

    log("\n" + "=" * 70)
    log("STEP 6: writing artifacts")
    log("=" * 70)
    with open(out_path, "w") as f:
        json.dump(scorecard_json, f, indent=1)
    log(f"[write] {out_path} ({os.path.getsize(out_path)/1024:.1f}KB)")

    rules_path = os.path.join(out_dir, "scorecard_rules.py")
    write_rules_py(rules_path, survive_scorecard, relane_scorecard, quant_scale)
    log(f"[write] {rules_path} ({os.path.getsize(rules_path)/1024:.1f}KB)")

    audit_path = os.path.join(out_dir, "AUDIT.md")
    write_audit_md(audit_path, survive_scorecard, relane_scorecard, quant_scale, len(train_songs), len(val_songs))
    log(f"[write] {audit_path}")

    log(f"\n[build_scorecard] DONE. decode knobs: {decode_knobs}")
    log(f"[build_scorecard] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
