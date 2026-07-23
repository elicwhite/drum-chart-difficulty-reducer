"""
The shared, deterministic 9-step decode: survive-pool -> threshold ->
family-NMS -> relane -> relane-pool -> chord-merge -> canonicalize (plus the
measure-clock / groove-cluster machinery both pooling steps and
canonicalize share). This is the SAME code for every model backend --
DETERMINISM_CONTRACT.md §4 documents the one honest exception (the
survive-pool comparison itself is backend-parameterized: model =
mean(proba)>=0.5, scorecard = sum(points)>=n*T -- everything else here is
backend-agnostic).

Independent reimplementation of the decode pipeline described in SPEC.md
§5, including the measure-clock / groove-cluster / canonicalize machinery
(folded in here, not imported, so this package has zero cross-file
coupling outside itself).

Every tie-break below implements DETERMINISM_CONTRACT.md §2 explicitly.
"""

import collections

Note = collections.namedtuple("Note", ["ms", "lane"])

# Canonical lane index (DETERMINISM_CONTRACT.md §1) -- used by every
# tie-break in this file.
LANE_INDEX = {
    "kick": 0, "snare": 1, "hihat": 2, "open-hat": 3, "high-tom": 4,
    "mid-tom": 5, "floor-tom": 6, "crash": 7, "ride": 8, "other": 9,
}

FAMILIES = {"cymbal": ["hihat", "open-hat", "crash", "ride"], "tom": ["high-tom", "mid-tom", "floor-tom"]}

GROOVE_TPQ = 480  # tick-in-measure bucketing resolution (RB-convention 480 ticks/quarter)


def _lane_idx(lane):
    return LANE_INDEX.get(lane, 9)


def _canonical_order(rows):
    """Indices into `rows`, sorted (ms, lane_index) ascending -- the fixed
    note order DETERMINISM_CONTRACT.md §1 requires for every deterministic
    iteration (grouping, summation, output)."""
    return sorted(range(len(rows)), key=lambda i: (rows[i]["ms"], _lane_idx(rows[i]["lane"])))


# ---------------------------------------------------------------------------
# Measure clock + Expert groove clusters (ported from consistency_metric.py,
# frozen/stdlib, unchanged apart from folding into this module).
# ---------------------------------------------------------------------------


def build_measure_clock(tempos, time_sigs):
    """(ms_to_measure, measure_to_ms) closures. ms_to_measure(ms) ->
    (measure_idx:int, beat_in_measure:float). measure_to_ms(measure_idx,
    beat_in_measure) -> ms (the exact inverse, used by canonicalize() to
    re-render a donor measure's groove at a different measure's timing)."""
    tempos = sorted(tempos, key=lambda t: t["ms"]) if tempos else []
    if not tempos or tempos[0]["ms"] > 0:
        tempos = [{"ms": 0, "bpm": tempos[0]["bpm"] if tempos else 120.0}] + tempos
    anchors_ms, anchors_beat, bpms = [], [], []
    cum_beats = 0.0
    for i, t in enumerate(tempos):
        anchors_ms.append(t["ms"])
        anchors_beat.append(cum_beats)
        bpms.append(t["bpm"])
        if i + 1 < len(tempos):
            cum_beats += (tempos[i + 1]["ms"] - t["ms"]) * t["bpm"] / 60000.0

    def ms_to_beat(ms):
        idx = max(0, _bisect_right(anchors_ms, ms) - 1)
        return anchors_beat[idx] + (ms - anchors_ms[idx]) * bpms[idx] / 60000.0

    def beat_to_ms(beat):
        idx = max(0, _bisect_right(anchors_beat, beat) - 1)
        return anchors_ms[idx] + (beat - anchors_beat[idx]) * 60000.0 / bpms[idx]

    ts = sorted(time_sigs, key=lambda t: t["ms"]) if time_sigs else []
    if not ts:
        ts = [{"ms": 0, "numerator": 4, "denominator": 4}]
    segs = [(ms_to_beat(t["ms"]), t["numerator"] * 4.0 / t["denominator"]) for t in ts]
    seg_starts = [s[0] for s in segs]
    cum_measures = [0]
    for i in range(1, len(segs)):
        prev_start, prev_bpmeasure = segs[i - 1]
        n = round((segs[i][0] - prev_start) / prev_bpmeasure) if prev_bpmeasure > 0 else 0
        cum_measures.append(cum_measures[-1] + max(0, n))

    BOUNDARY_EPS_BEATS = 1e-6  # absorbs FP drift from summing many tempo
    # segments so a beat_in_measure of e.g. 3.999999999999943 rolls to the
    # next measure's tick 0 instead of a stray tick 1920.

    def ms_to_measure(ms):
        beat = ms_to_beat(ms)
        idx = max(0, _bisect_right(seg_starts, beat) - 1)
        seg_start, bpmeasure = segs[idx]
        rel = beat - seg_start
        n_in_seg = int(rel // bpmeasure) if bpmeasure > 0 else 0
        beat_in_measure = rel - n_in_seg * bpmeasure if bpmeasure > 0 else 0.0
        if bpmeasure > 0 and beat_in_measure > bpmeasure - BOUNDARY_EPS_BEATS:
            n_in_seg += 1
            beat_in_measure = 0.0
        return cum_measures[idx] + n_in_seg, beat_in_measure

    def measure_to_ms(measure_idx, beat_in_measure):
        idx = max(0, _bisect_right_cum(cum_measures, measure_idx) - 1)
        seg_start, bpmeasure = segs[idx]
        n_in_seg = measure_idx - cum_measures[idx]
        return beat_to_ms(seg_start + n_in_seg * bpmeasure + beat_in_measure)

    return ms_to_measure, measure_to_ms


def _bisect_right(a, x):
    import bisect
    return bisect.bisect_right(a, x)


def _bisect_right_cum(a, x):
    import bisect
    return bisect.bisect_right(a, x)


def reduced_groove_by_measure(notes, ms_to_measure):
    """{measure_idx: frozenset((tick_in_measure, lane))} for a note list
    (Note(ms, lane) tuples)."""
    by_measure = collections.defaultdict(set)
    for n in notes:
        mi, beat = ms_to_measure(n.ms)
        by_measure[mi].add((round(beat * GROOVE_TPQ), n.lane))
    return {mi: frozenset(s) for mi, s in by_measure.items()}


def expert_groove_clusters(expert_notes, ms_to_measure):
    """({groove_key: [measure_idx, ...]} for groove_keys seen in >=2
    measures, n_nonempty_measures). measure_idx lists sorted ascending."""
    rbm = reduced_groove_by_measure(expert_notes, ms_to_measure)
    by_groove = collections.defaultdict(list)
    for mi in sorted(rbm):
        by_groove[rbm[mi]].append(mi)
    clusters = {k: v for k, v in by_groove.items() if len(v) >= 2}
    return clusters, len(rbm)


def consistency_stats(clusters, reduced_by_measure):
    """Diagnostic only (not used by reduce()'s decode path) -- how
    inconsistently a candidate reduces repeated Expert grooves, pre-canon."""
    n_clusters = n_clusters_disagree = n_instances = n_inconsistent = 0
    for measure_idxs in clusters.values():
        reductions = [reduced_by_measure.get(mi, frozenset()) for mi in measure_idxs]
        modal = _modal_reduction(reductions)
        n_inc = sum(1 for r in reductions if r != modal)
        n_clusters += 1
        n_instances += len(reductions)
        n_inconsistent += n_inc
        if n_inc:
            n_clusters_disagree += 1
    return {
        "n_clusters": n_clusters,
        "n_clusters_with_disagreement": n_clusters_disagree,
        "n_instances": n_instances,
        "n_inconsistent_instances": n_inconsistent,
        "inst_inconsistency_rate": (n_inconsistent / n_instances) if n_instances else None,
        "cluster_inconsistency_rate": (n_clusters_disagree / n_clusters) if n_clusters else None,
    }


def _modal_reduction(reductions):
    """DETERMINISM_CONTRACT.md §2.4: modal (majority-vote) groove across a
    cluster's instances; ties broken by the lexicographically smallest
    sorted (tick, lane_index) tuple list -- NOT `Counter.most_common`'s
    insertion-order tie-break, which is what the original
    consistency_metric.canonicalize() used (its instance order happened to
    be measure_idx-ascending, so it was deterministic but not defined by
    groove CONTENT). See test_parity.py / the final report for whether this
    changed any fixture's output."""
    counts = collections.Counter(reductions)
    max_count = max(counts.values())
    candidates = [r for r, c in counts.items() if c == max_count]
    if len(candidates) == 1:
        return candidates[0]

    def sort_key(groove):
        return sorted((tick, _lane_idx(lane)) for tick, lane in groove)

    candidates.sort(key=sort_key)
    return candidates[0]


def canonicalize(cand_notes, clusters, reduced_by_measure, ms_to_measure, measure_to_ms):
    """Force every instance in a repeated-groove cluster to the reducer's
    own modal reduction for that groove. Non-clustered measures pass
    through untouched. Pure: does not mutate its arguments."""
    clustered_measures = {mi for idxs in clusters.values() for mi in idxs}
    out = [n for n in cand_notes if ms_to_measure(n.ms)[0] not in clustered_measures]
    for measure_idxs in clusters.values():
        reductions = [reduced_by_measure.get(mi, frozenset()) for mi in measure_idxs]
        modal = _modal_reduction(reductions)
        for mi in measure_idxs:
            for tick, lane in modal:
                out.append(Note(measure_to_ms(mi, tick / GROOVE_TPQ), lane))
    out.sort(key=lambda n: (n.ms, _lane_idx(n.lane)))
    return out


# ---------------------------------------------------------------------------
# Pooling / NMS / relane-pool / chord-merge
# ---------------------------------------------------------------------------


def _meas_to_groove_map(clusters):
    meas_to_groove = {}
    for gk, idxs in clusters.items():
        for mi in idxs:
            meas_to_groove[mi] = gk
    return meas_to_groove


def survive_pool(rows, survive_proba, ms_to_measure, clusters):
    """SURVIVE-POOL: group notes by (expert_groove_cluster_id,
    round(beat_in_measure*GROOVE_TPQ), lane); replace each member's
    survive_proba with the group's arithmetic mean, accumulated in
    canonical (ms, lane_index) order (DETERMINISM_CONTRACT.md §1). Notes
    outside any cluster are unaffected. Runs BEFORE thresholding."""
    if not clusters:
        return list(survive_proba)
    meas_to_groove = _meas_to_groove_map(clusters)
    order = _canonical_order(rows)
    bucket = collections.defaultdict(list)
    key_of = {}
    for i in order:
        r = rows[i]
        mi, beat = ms_to_measure(r["ms"])
        gk = meas_to_groove.get(mi)
        if gk is None:
            continue
        k = (gk, round(beat * GROOVE_TPQ), r["lane"])
        bucket[k].append(survive_proba[i])
        key_of[i] = k
    if not bucket:
        return list(survive_proba)
    means = {}
    for k, vals in bucket.items():
        s = 0.0
        for v in vals:  # explicit left-to-right accumulation, canonical order
            s += v
        means[k] = s / len(vals)
    out = list(survive_proba)
    for i, k in key_of.items():
        out[i] = means[k]
    return out


def family_nms(rows, survive, keep_score, gap_ms):
    """FAMILY-NMS (cymbal/tom only; kick/snare/other never suppressed).
    Greedy: sort currently-surviving family notes by descending keep_score,
    tie-break (-keep_score, ms, lane_index) per DETERMINISM_CONTRACT.md §2.1
    (equal pooled scores across repeated-groove members are common, not
    rare -- this tie-break is mandatory, NMS is greedy so one flip
    cascades). Walk the sorted list; drop a note if it falls within gap_ms
    of an already-kept note's ms."""
    out = list(survive)
    fam_idx = [i for i in range(len(rows)) if out[i] and rows[i]["family"] in FAMILIES]
    fam_idx.sort(key=lambda i: (-keep_score[i], rows[i]["ms"], _lane_idx(rows[i]["lane"])))
    kept_ms = []
    for i in fam_idx:
        ms = rows[i]["ms"]
        if any(abs(ms - km) < gap_ms for km in kept_ms):
            out[i] = False
        else:
            kept_ms.append(ms)
    return out


def relane_pool(rows, final_lane, confidence, ms_to_measure, clusters):
    """RELANE-POOL: for FAMILY notes, group by (expert_groove_cluster_id,
    round(beat_in_measure*GROOVE_TPQ), SOURCE lane); override every
    member's final_lane with the confidence-weighted modal lane (sum
    confidence per candidate lane, canonical order; DETERMINISM_CONTRACT.md
    §2.2: tie broken by lowest lane_index). Runs AFTER relane predict,
    BEFORE chord-merge."""
    if not clusters:
        return list(final_lane)
    meas_to_groove = _meas_to_groove_map(clusters)
    order = _canonical_order(rows)
    votes = collections.defaultdict(dict)  # k -> {candidate_lane: summed_conf}
    key_of = {}
    for i in order:
        r = rows[i]
        if r["family"] not in FAMILIES:
            continue
        mi, beat = ms_to_measure(r["ms"])
        gk = meas_to_groove.get(mi)
        if gk is None:
            continue
        k = (gk, round(beat * GROOVE_TPQ), r["lane"])
        votes[k][final_lane[i]] = votes[k].get(final_lane[i], 0.0) + float(confidence[i])
        key_of[i] = k
    if not votes:
        return list(final_lane)
    modal = {}
    for k, tally in votes.items():
        # candidate lanes considered in lane_index order (§1); tie -> lowest lane_index (§2.2)
        best_lane, best_conf = None, None
        for lane in sorted(tally, key=_lane_idx):
            c = tally[lane]
            if best_conf is None or c > best_conf:
                best_lane, best_conf = lane, c
        modal[k] = best_lane
    out = list(final_lane)
    for i, k in key_of.items():
        out[i] = modal[k]
    return out


def chord_merge(rows, survive, final_lane, confidence):
    """Fixed lanes (kick/snare/other) pass through unchanged if survive.
    FAMILY survivors: group by (ms, family, final_lane); if >1 member, keep
    only the highest-confidence one -- tie broken by lowest SOURCE
    lane_index (DETERMINISM_CONTRACT.md §2.3)."""
    survivor_idx = [i for i, s in enumerate(survive) if s]
    fixed_idx = [i for i in survivor_idx if rows[i]["family"] not in FAMILIES]
    family_idx = [i for i in survivor_idx if rows[i]["family"] in FAMILIES]

    by_key = collections.defaultdict(list)
    for i in family_idx:
        r = rows[i]
        by_key[(r["ms"], r["family"], final_lane[i])].append(i)

    keep_idx = []
    for group in by_key.values():
        if len(group) > 1:
            group = sorted(group, key=lambda i: (-confidence[i], _lane_idx(rows[i]["lane"])))
        keep_idx.append(group[0])

    cand = [Note(rows[i]["ms"], rows[i]["lane"]) for i in fixed_idx]
    cand += [Note(rows[i]["ms"], final_lane[i]) for i in keep_idx]
    return cand
