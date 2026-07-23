"""
The 59-feature extractor, per note, for the drum difficulty-reducer.

This is a consolidation of three feature groups, developed and validated
against thousands of Harmonix's own official reductions (see SPEC.md §2 for
the byte-exact spec this module implements):
  - base 13 chart features + lane/section/era one-hots
    (NUMERIC_FEATS, SECTION_KEYWORDS, extract_song_features, build_matrix)
  - +10 chord-context flags
    (CHORD_FEATS, _extract_song_features_v2, build_matrix_v2)
  - +9 AUG_FEATS v7 (song-level context) + the 3-position-feature drop
    (AUG_FEATS, annotate_features, DROP_POSITION_FEATS, build_matrix_aug)

FEATURE_NAMES below is the ground truth 59-column order -- verified to
match data/model/feature_names.json byte-for-byte by test_parity.py.
`section_prechorus` is listed TWICE, deliberately (see the section-keyword
table below) -- do not de-dupe it, every downstream column shifts if you do.

Input chart schema (see SPEC.md §7):
    {
      "tempos": [{"ms": float, "bpm": float}, ...],
      "timeSignatures": [{"ms": float, "numerator": int, "denominator": int}, ...],
      "sections": [{"ms": float, "name": str}, ...],
      "era": "RB1" | "RB2" | "RB3" | "RB4" | "other",
      "difficulties": {
        "expert": {"notes": [[ms, [{"instrument": str, "ghost"?: bool,
                                     "accent"?: bool, "flam"?: bool}, ...]], ...]}
      },
    }
`notes` is already grouped by ms (one entry per distinct tick, chords listed
together under that ms) -- this is scan-chart's own convention and is what
`chord_size` (feature A.1) counts directly, no re-grouping needed.
"""

import bisect
import collections

import numpy as np

EPS_MS = 0.5  # tick-rounding slack, matches editrate.py's EPS_MS
ALIGN_EPS_BEATS = 0.04  # ~ a 32nd note at 4/4, tempo-normalized
GRID_DIVS = {"half": 2.0, "quarter": 1.0, "eighth": 0.5}

LANE_VOCAB = ["kick", "snare", "hihat", "open-hat", "high-tom", "mid-tom", "floor-tom", "crash", "ride"]
FAMILIES = {"cymbal": ["hihat", "open-hat", "crash", "ride"], "tom": ["high-tom", "mid-tom", "floor-tom"]}
FAMILY_OF_LANE = {lane: fam for fam, lanes in FAMILIES.items() for lane in lanes}
BACKBONE_LANES = {"kick", "snare"}
ERA_VOCAB = ["RB1", "RB2", "RB3", "RB4", "other"]

# (substring keyword, label) -- first match wins (case-insensitive substring
# test), else "other". Iterated in this exact order, both for section_type()
# and for the one-hot column list below -- the two synonym entries for
# "prechorus" ("pre-chorus" and "prechorus") are a real artifact of the
# reference implementation and produce the deliberate duplicate column.
SECTION_KEYWORDS = [
    ("intro", "intro"),
    ("outro", "outro"),
    ("pre-chorus", "prechorus"),
    ("prechorus", "prechorus"),
    ("chorus", "chorus"),
    ("verse", "verse"),
    ("bridge", "bridge"),
    ("solo", "solo"),
    ("breakdown", "breakdown"),
    ("interlude", "interlude"),
    ("fill", "fill"),
]
SECTION_VOCAB = [lbl for _, lbl in SECTION_KEYWORDS] + ["other"]

# Chord-importance rank (lower = more likely to survive a reduction) --
# used only by aug_chord_priority.
_LANE_PRIORITY = {"kick": 0, "snare": 1, "crash": 2, "ride": 3, "hihat": 4,
                   "open-hat": 5, "floor-tom": 6, "mid-tom": 7, "high-tom": 8, "other": 9}

_BASE_NUMERIC = ["chord_size", "beat_in_measure", "beats_per_measure", "is_downbeat",
                 "local_density_500ms", "gap_prev_ms", "gap_next_ms", "ghost", "accent", "flam",
                 "aligned_half", "aligned_quarter", "aligned_eighth"]

AUG_FEATS = ["aug_dist_backbone_ms", "aug_density_ratio",
             "aug_samelane_prev_ms", "aug_samelane_next_ms", "aug_chord_priority",
             "aug_density_100ms", "aug_density_1500ms", "aug_beat_frac", "aug_lane_frac_500ms"]

# Ground truth 59-column order (base13 + lane10 + section12 + era5 +
# chord_has10 + aug9). Cross-checked against feature_names.json.
FEATURE_NAMES = (
    _BASE_NUMERIC
    + [f"lane_{lane}" for lane in LANE_VOCAB + ["other"]]
    + [f"section_{sec}" for sec in SECTION_VOCAB]
    + [f"era_{era}" for era in ERA_VOCAB]
    + [f"chord_has_{lv}" for lv in LANE_VOCAB + ["other"]]
    + AUG_FEATS
)
assert len(FEATURE_NAMES) == 59, len(FEATURE_NAMES)


def lane_of(instrument):
    return instrument if instrument in LANE_VOCAB else "other"


# ---------------------------------------------------------------------------
# Tempo/time-signature/section geometry helpers (ported from reduction_probe.py)
# ---------------------------------------------------------------------------


def build_ms_to_beat(tempos):
    tempos = sorted(tempos, key=lambda t: t["ms"]) if tempos else []
    if not tempos or tempos[0]["ms"] > 0:
        tempos = [{"ms": 0, "bpm": tempos[0]["bpm"] if tempos else 120.0}] + tempos
    anchors_ms, anchors_beat = [], []
    cum_beats = 0.0
    for i, t in enumerate(tempos):
        anchors_ms.append(t["ms"])
        anchors_beat.append(cum_beats)
        if i + 1 < len(tempos):
            dur_ms = tempos[i + 1]["ms"] - t["ms"]
            cum_beats += dur_ms * t["bpm"] / 60000.0
    bpms = [t["bpm"] for t in tempos]

    def ms_to_beat(ms):
        idx = max(0, bisect.bisect_right(anchors_ms, ms) - 1)
        return anchors_beat[idx] + (ms - anchors_ms[idx]) * bpms[idx] / 60000.0

    return ms_to_beat


def build_measure_fn(time_sigs, ms_to_beat):
    ts = sorted(time_sigs, key=lambda t: t["ms"]) if time_sigs else []
    if not ts:
        ts = [{"ms": 0, "numerator": 4, "denominator": 4}]
    segs = []
    for t in ts:
        b = ms_to_beat(t["ms"])
        beats_per_measure = t["numerator"] * 4.0 / t["denominator"]
        segs.append((b, beats_per_measure))
    seg_starts = [s[0] for s in segs]

    def beat_to_measure_pos(beat):
        idx = max(0, bisect.bisect_right(seg_starts, beat) - 1)
        seg_start, bpmeasure = segs[idx]
        rel = beat - seg_start
        beat_in_measure = rel % bpmeasure if bpmeasure > 0 else 0.0
        return beat_in_measure, bpmeasure

    return beat_to_measure_pos


def section_type(name):
    n = (name or "").lower()
    for kw, label in SECTION_KEYWORDS:
        if kw in n:
            return label
    return "other"


def build_section_fn(sections, song_end_ms):
    secs = sorted(sections, key=lambda s: s["ms"]) if sections else [{"ms": 0, "name": ""}]
    starts = [s["ms"] for s in secs]

    def at(ms):
        idx = max(0, bisect.bisect_right(starts, ms) - 1)
        start = secs[idx]["ms"]
        end = secs[idx + 1]["ms"] if idx + 1 < len(secs) else song_end_ms
        frac = (ms - start) / (end - start) if end > start else 0.0
        return section_type(secs[idx]["name"]), idx / max(1, len(secs) - 1), frac

    return at


def align_flags(beat_pos):
    out = {}
    for name, div in GRID_DIVS.items():
        frac = beat_pos % div
        out[f"aligned_{name}"] = 1 if (frac < ALIGN_EPS_BEATS or div - frac < ALIGN_EPS_BEATS) else 0
    return out


def flatten_expert(notes):
    """notes: [[ms, [{"instrument":..., ...}, ...]], ...], already grouped
    by ms. Returns [(ms, raw_instrument, chord_size, entry_dict), ...]
    sorted by (ms, raw_instrument) -- matches reduction_probe.flatten_expert's
    sort key exactly (raw instrument string, not the lane-vocab-mapped one)."""
    rows = []
    for ms, entries in notes:
        chord_size = len(entries)
        for e in entries:
            rows.append((ms, e["instrument"], chord_size, e))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


# ---------------------------------------------------------------------------
# AUG_FEATS v7 (song-level context) -- see SPEC.md §2 for the byte-exact
# spec each of these implements.
# ---------------------------------------------------------------------------


def _annotate_aug_features(rows):
    backbone_ms = sorted(r["ms"] for r in rows if r["lane"] in BACKBONE_LANES)
    by_lane = collections.defaultdict(list)
    for r in rows:
        by_lane[r["lane"]].append(r["ms"])
    for lane in by_lane:
        by_lane[lane].sort()
    by_tick = collections.defaultdict(list)
    for r in rows:
        by_tick[round(r["ms"] / EPS_MS)].append(r)
    dens = np.array([r["local_density_500ms"] for r in rows], dtype=np.float64)
    med = float(np.median(dens)) if len(dens) else 0.0
    all_ms = sorted(r["ms"] for r in rows)  # per-note (chord = N entries), for multi-scale windows

    for r in rows:
        ms = r["ms"]
        if backbone_ms:
            i = bisect.bisect_left(backbone_ms, ms)
            cands = []
            if i < len(backbone_ms):
                cands.append(abs(backbone_ms[i] - ms))
            if i > 0:
                cands.append(abs(ms - backbone_ms[i - 1]))
            r["aug_dist_backbone_ms"] = min(cands) if cands else 5000.0
        else:
            r["aug_dist_backbone_ms"] = 5000.0
        r["aug_density_ratio"] = r["local_density_500ms"] / (med + 1.0)

        lane_ms = by_lane[r["lane"]]
        j = bisect.bisect_left(lane_ms, ms)
        r["aug_samelane_prev_ms"] = min(ms - lane_ms[j - 1], 5000.0) if j > 0 else 5000.0
        r["aug_samelane_next_ms"] = min(lane_ms[j + 1] - ms, 5000.0) if j + 1 < len(lane_ms) else 5000.0

        tick = round(ms / EPS_MS)
        myp = _LANE_PRIORITY.get(r["lane"], 9)
        r["aug_chord_priority"] = sum(1 for o in by_tick[tick] if _LANE_PRIORITY.get(o["lane"], 9) < myp)

        lo100 = bisect.bisect_left(all_ms, ms - 100.0)
        hi100 = bisect.bisect_right(all_ms, ms + 100.0)
        r["aug_density_100ms"] = hi100 - lo100 - 1
        lo15 = bisect.bisect_left(all_ms, ms - 1500.0)
        hi15 = bisect.bisect_right(all_ms, ms + 1500.0)
        r["aug_density_1500ms"] = hi15 - lo15 - 1

        bim = r["beat_in_measure"]
        r["aug_beat_frac"] = abs(bim - round(bim))

        lo5 = bisect.bisect_left(all_ms, ms - 500.0)
        hi5 = bisect.bisect_right(all_ms, ms + 500.0)
        n_win = hi5 - lo5
        lane_ms_win = bisect.bisect_right(lane_ms, ms + 500.0) - bisect.bisect_left(lane_ms, ms - 500.0)
        r["aug_lane_frac_500ms"] = (lane_ms_win / n_win) if n_win > 0 else 0.0


def _build_matrix(rows):
    n = len(rows)
    cols = []
    for f in _BASE_NUMERIC:
        cols.append(np.array([r[f] for r in rows], dtype=np.float64))
    for lane in LANE_VOCAB + ["other"]:
        cols.append(np.array([1.0 if r["lane"] == lane else 0.0 for r in rows]))
    for sec in SECTION_VOCAB:
        cols.append(np.array([1.0 if r["section_type"] == sec else 0.0 for r in rows]))
    for era in ERA_VOCAB:
        cols.append(np.array([1.0 if r["era"] == era else 0.0 for r in rows]))
    for lv in LANE_VOCAB + ["other"]:
        cols.append(np.array([r[f"chord_has_{lv}"] for r in rows], dtype=np.float64))
    for f in AUG_FEATS:
        cols.append(np.array([r[f] for r in rows], dtype=np.float64))
    return np.stack(cols, axis=1) if n else np.zeros((0, len(FEATURE_NAMES)))


def featurize(chart):
    """Returns (X, names, rows): X is an (n, 59) float64 matrix in
    FEATURE_NAMES order; rows is a list of per-note bookkeeping dicts
    (ms, lane, family, ...) aligned 1:1 with X's row order -- decode.py
    consumes both together. Empty chart (no Expert notes) -> (0,59) matrix,
    [], []."""
    expert = chart["difficulties"]["expert"]
    exp_rows = flatten_expert(expert["notes"])
    if not exp_rows:
        return np.zeros((0, len(FEATURE_NAMES))), list(FEATURE_NAMES), []

    tempos = chart.get("tempos") or []
    ms_to_beat = build_ms_to_beat(tempos)
    beat_pos_fn = build_measure_fn(chart.get("timeSignatures") or [], ms_to_beat)
    song_end_ms = max(r[0] for r in exp_rows)
    section_fn = build_section_fn(chart.get("sections") or [], song_end_ms)
    era = chart.get("era") or "other"
    if era not in ERA_VOCAB:
        era = "other"

    exp_ms_sorted = [r[0] for r in exp_rows]
    exp_ms_unique = sorted(set(exp_ms_sorted))

    def density_window(ms, half_window_ms=250.0):
        lo = bisect.bisect_left(exp_ms_unique, ms - half_window_ms)
        hi = bisect.bisect_right(exp_ms_unique, ms + half_window_ms)
        return hi - lo - 1  # exclude self tick

    prev_ms_map, next_ms_map = {}, {}
    for i, ms in enumerate(exp_ms_unique):
        prev_ms_map[ms] = exp_ms_unique[i - 1] if i > 0 else ms
        next_ms_map[ms] = exp_ms_unique[i + 1] if i + 1 < len(exp_ms_unique) else ms

    tick_expert_lanes = collections.defaultdict(set)
    for ms, raw_lane, _cs, _e in exp_rows:
        tick_expert_lanes[round(ms / EPS_MS)].add(lane_of(raw_lane))

    rows = []
    for ms, raw_lane, chord_size, e in exp_rows:
        lane = lane_of(raw_lane)
        beat = ms_to_beat(ms)
        beat_in_measure, beats_per_measure = beat_pos_fn(beat)
        sec_type, _sec_progress, _sec_frac = section_fn(ms)
        tick = round(ms / EPS_MS)

        row = {
            "ms": ms,
            "lane": lane,
            "era": era,
            "family": FAMILY_OF_LANE.get(lane, "fixed"),
            "chord_size": chord_size,
            "beat_in_measure": beat_in_measure,
            "beats_per_measure": beats_per_measure,
            "is_downbeat": 1 if beat_in_measure < ALIGN_EPS_BEATS else 0,
            "local_density_500ms": density_window(ms),
            "gap_prev_ms": min(ms - prev_ms_map[ms], 5000.0),
            "gap_next_ms": min(next_ms_map[ms] - ms, 5000.0),
            "section_type": sec_type,
            "ghost": 1 if e.get("ghost") else 0,
            "accent": 1 if e.get("accent") else 0,
            "flam": 1 if e.get("flam") else 0,
        }
        row.update(align_flags(beat_in_measure))
        for lv in LANE_VOCAB + ["other"]:
            row[f"chord_has_{lv}"] = 1 if lv in tick_expert_lanes[tick] else 0
        rows.append(row)

    _annotate_aug_features(rows)

    X = _build_matrix(rows)
    return X, list(FEATURE_NAMES), rows
