"""
Top-level clean API: reduce(chart, tier, backend="model"|"scorecard") ->
[{"ms","lane"}].

Wires featurize.py -> a MODEL backend (backend_model.py: packed-GBM,
lossless, 0.1703; or backend_scorecard.py: auditable integer scorecard) ->
decode.py, in the exact 9-step order SPEC.md §5 specifies (order is not
the "obvious" reading --
survive-pool runs BEFORE thresholding AND before NMS; relane-pool runs AFTER
relane predict but BEFORE chord-merge; canonicalize is always last):

  1. featurize                         (featurize.featurize)
  2. survive predict                   (backend.predict_survive)
  3. SURVIVE-POOL                      (decode.survive_pool)
  4. threshold (>= backend.survive_threshold(tier))
  5. FAMILY-NMS (gap = backend.nms_gap(tier))
  6. relane predict                    (backend.predict_relane)
  7. RELANE-POOL                       (decode.relane_pool)
  8. chord-merge dedup                 (decode.chord_merge)
  9. CANONICALIZE                      (decode.canonicalize)

Steps 4-5's knobs are BACKEND-provided (DETERMINISM_CONTRACT.md §4's one
honest exception to "DECODE is fully shared" -- the model backend's
`survive_threshold`/`nms_gap` are fixed at their SPEC.md §6
values; the scorecard backend's are its own validated (T_tier, NMS-gap)
operating point, selected on rb4_val and read from scorecard.json's
`decode` block, not hardcoded here). `decode.survive_pool` itself is
unchanged/shared: it always computes the arithmetic MEAN of the backend's
per-note score across a groove-pool group. For the scorecard backend that
score is integer points, and `mean(points) >= T_tier` is mathematically the
same comparison as contract §4's `sum(points) >= n*T_tier` -- the division
is IEEE-754 correctly-rounded (deterministic, portable), so this backend
does not need its own pooling function.
"""

from . import decode, featurize
from .backend_model import ModelBackend
from .backend_scorecard import ScorecardBackend

TIERS = ["hard", "medium", "easy"]
BACKENDS = ("model", "scorecard")

_default_backends = {}


def _get_default_backend(name):
    if name not in _default_backends:
        if name == "model":
            _default_backends[name] = ModelBackend.load_default()
        elif name == "scorecard":
            _default_backends[name] = ScorecardBackend.load_default()
    return _default_backends[name]


def reduce(chart, tier, backend="model", model=None):
    """chart: the parsed-note-list input structure (see featurize.py's
    docstring for the exact schema). tier: one of TIERS. backend: "model"
    (packed-GBM, lossless) or "scorecard" (auditable integer scorecard).
    model: an optional pre-loaded backend instance (pass one in to avoid
    re-reading the on-disk artifact on every call, e.g. when reducing many
    songs in a loop).

    Returns a list of {"ms": float, "lane": str} dicts, sorted in canonical
    (ms, lane_index) order (DETERMINISM_CONTRACT.md §1)."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    if backend not in BACKENDS:
        raise ValueError(f"unsupported backend {backend!r} -- must be one of {BACKENDS}")
    if model is None:
        model = _get_default_backend(backend)

    X, names, rows = featurize.featurize(chart)
    assert names == featurize.FEATURE_NAMES, "featurize() column order drifted from FEATURE_NAMES"
    if not rows:
        return []

    survive_proba = model.predict_survive(tier, X)

    ms_to_measure, measure_to_ms = decode.build_measure_clock(
        chart.get("tempos") or [], chart.get("timeSignatures") or [])
    expert_notes = [decode.Note(r["ms"], r["lane"]) for r in rows]
    clusters, _n_nonempty = decode.expert_groove_clusters(expert_notes, ms_to_measure)

    pooled = decode.survive_pool(rows, survive_proba, ms_to_measure, clusters)
    survive = [p >= model.survive_threshold(tier) for p in pooled]

    gap = model.nms_gap(tier)
    if gap:
        survive = decode.family_nms(rows, survive, pooled, gap)

    final_lane = [r["lane"] for r in rows]
    confidence = [1.0] * len(rows)
    for fam_name in featurize.FAMILIES:
        idxs = [i for i, r in enumerate(rows) if r["family"] == fam_name and survive[i]]
        if not idxs or fam_name not in model.relane.get(tier, {}):
            continue
        lanes_out, conf_out = model.predict_relane(tier, fam_name, X[idxs])
        for k, i in enumerate(idxs):
            final_lane[i] = lanes_out[k]
            confidence[i] = float(conf_out[k])

    final_lane = decode.relane_pool(rows, final_lane, confidence, ms_to_measure, clusters)
    cand = decode.chord_merge(rows, survive, final_lane, confidence)

    if clusters:
        rbm = decode.reduced_groove_by_measure(cand, ms_to_measure)
        cand = decode.canonicalize(cand, clusters, rbm, ms_to_measure, measure_to_ms)

    cand_sorted = sorted(cand, key=lambda n: (n.ms, decode.LANE_INDEX.get(n.lane, 9)))
    return [{"ms": n.ms, "lane": n.lane} for n in cand_sorted]
