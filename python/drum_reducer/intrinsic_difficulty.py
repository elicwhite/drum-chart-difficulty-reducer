"""
Peak-aware scalar difficulty D(chart) -- an intrinsic, reference-free
playability axis, orthogonal to edit_rate (which is reference-relative:
it can only say "how far from Harmonix's reduction", never "is this
section still hard to play").

Scope (per team-lead brief, 2026-07-21, post-contrarian narrowing):
  - ONLY I1 (peak-aware scalar D) is built here. I4 (feasibility) is
    dropped -- vacuous under deletion-only reduction (you can't create an
    infeasible hand pattern by deleting notes from a feasible one). I5
    (consistency) already shipped as consistency_probe.py. I2's "max
    local-density ratio" is folded into D's peak-density feature rather
    than tracked as a separate axis. I3/I6 (groove/backbone) are merged
    into one "skeleton retention" feature below.
  - Fixed weights, chosen for musical plausibility, NOT fit to any corpus
    (fitting would reintroduce the subjectivity this axis is meant to
    avoid -- see Probe 0's monotonicity check, which exists precisely to
    catch a broken feature *without* letting us fit our way past it).

Operates on any Note-like object exposing `.ms`/`.lane` (this package's own
editrate.Note, or drum_reducer.decode.Note) -- the same representation every
reducer in this repo already produces -- plus a song's tempo map (list of
{"ms": ..., "bpm": ...}). No dependency on any specific reducer.

Module-relative import only, no cross-file coupling outside this package.
"""

import bisect
import collections
import math

# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

# Windows for local density. Fixed-ms (~1s, a "how much can happen in one
# breath" window) and beat-relative (2 beats -- catches dense streams at
# ANY tempo, since a burst that's brutal at 180bpm may be nothing at 80bpm
# and a fixed-ms window alone would conflate the two).
DENSITY_WINDOW_MS = 1000.0
DENSITY_WINDOW_BEATS = 2.0

PEAK_PERCENTILE = 90.0

# Normalization scales -- musically-motivated caps, not fit constants.
# "Insane" reference points pulled from genre norms (blast-beat streams,
# RB4 Expert charts), not from this corpus's own distribution -- fitting
# these to Harmonix would be exactly the calibration-overfit Probe 0 is
# built to catch.
DENSITY_CAP = 16.0       # notes/sec that saturates the density feature (~16th notes at 240bpm, 4 limbs)
STREAM_IOI_FLOOR_MS = 60.0    # sub-60ms same-lane IOI treated as "as fast as it gets" (~16th at 250bpm)
STREAM_IOI_CAP_MS = 300.0     # >=300ms same-lane IOI treated as "not a stream" (8th note at 100bpm)
CHORD_CAP = 4.0           # simultaneous distinct lanes that saturates the chord-load feature
LANE_CAP = 9.0            # distinct lanes used, corpus lane vocab is 9 (kick..ride)


def _percentile(values, pct):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _bpm_at(tempos, ms):
    if not tempos:
        return 120.0
    bpm = tempos[0]["bpm"]
    for t in tempos:
        if t["ms"] <= ms:
            bpm = t["bpm"]
        else:
            break
    return bpm


def _sliding_window_counts(onset_ms_sorted, window_ms_at):
    """For each onset, count how many onsets (incl. itself) fall in
    [onset, onset+window) where window_ms_at(onset) gives the window width
    at that onset's local tempo. O(n log n): two-pointer since window
    width is monotone-ish in ms but not tempo, so we bisect per-onset
    rather than assume a fixed two-pointer walk works across tempo
    changes."""
    counts = []
    n = len(onset_ms_sorted)
    for i, ms in enumerate(onset_ms_sorted):
        w = window_ms_at(ms)
        j = bisect.bisect_right(onset_ms_sorted, ms + w, lo=i)
        counts.append(j - i)
    return counts


def _skeleton_retention(notes, gt_backbone_lanes=("kick", "snare")):
    """Fraction of onsets (grouped by ms) that still carry a kick or snare
    hit -- the "groove/backbone" merge (I3/I6): a chart that has dropped
    its kick/snare skeleton down to sparse cymbal decoration is easier to
    FOLLOW even if its cymbal lane is still busy, so this feature pulls D
    DOWN when kick/snare coverage thins out. Cheap proxy, not a full
    groove model."""
    by_ms = collections.defaultdict(set)
    for n in notes:
        by_ms[n.ms].add(n.lane)
    if not by_ms:
        return 0.0
    hits = sum(1 for lanes in by_ms.values() if lanes & set(gt_backbone_lanes))
    return hits / len(by_ms)


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------


def compute_features(notes, tempos=None):
    """notes: list of editrate.Note(ms, lane) for ONE difficulty tier of
    ONE song. tempos: song's tempo map (list of {"ms","bpm"}) or None (falls
    back to 120bpm throughout -- only affects the beat-relative window and
    skeleton feature is tempo-independent).

    Returns a dict of raw (unnormalized) features -- kept around per the
    brief ("keep the raw feature vector... so we can inspect which drives
    D") -- plus normalized [0,1] components and the combined scalar D.
    """
    if not notes:
        return {
            "n_notes": 0, "peak_density_fixed": 0.0, "peak_density_beat": 0.0,
            "fastest_stream_ioi_ms": None, "longest_stream_run": 0,
            "peak_chord_size": 0.0, "n_lanes": 0, "syncopation_frac": 0.0,
            "skeleton_retention": 0.0,
            "D": 0.0,
        }
    tempos = tempos or []
    onsets = sorted(n.ms for n in notes)

    # --- 1. peak local density (I1 core + I2's max local-density ratio folded in) ---
    fixed_counts = _sliding_window_counts(onsets, lambda ms: DENSITY_WINDOW_MS)
    peak_density_fixed = _percentile(fixed_counts, PEAK_PERCENTILE) / (DENSITY_WINDOW_MS / 1000.0)  # notes/sec

    def beat_window_ms(ms):
        bpm = _bpm_at(tempos, ms)
        return DENSITY_WINDOW_BEATS * (60000.0 / bpm)

    beat_counts = _sliding_window_counts(onsets, beat_window_ms)
    # normalize count-per-2-beats to notes/sec using the LOCAL tempo so it's comparable across songs
    peak_density_beat_persec = []
    for ms, c in zip(onsets, beat_counts):
        bpm = _bpm_at(tempos, ms)
        window_s = (DENSITY_WINDOW_BEATS * (60000.0 / bpm)) / 1000.0
        peak_density_beat_persec.append(c / window_s if window_s > 0 else 0.0)
    peak_density_beat = _percentile(peak_density_beat_persec, PEAK_PERCENTILE)

    # --- 2. fastest sustained same-lane stream ---
    by_lane = collections.defaultdict(list)
    for n in notes:
        by_lane[n.lane].append(n.ms)
    fastest_ioi = None
    longest_run = 0
    for lane, ms_list in by_lane.items():
        ms_list = sorted(ms_list)
        run = 1
        for a, b in zip(ms_list, ms_list[1:]):
            ioi = b - a
            if ioi <= 0:
                continue
            if fastest_ioi is None or ioi < fastest_ioi:
                fastest_ioi = ioi
            if ioi <= STREAM_IOI_CAP_MS:
                run += 1
                longest_run = max(longest_run, run)
            else:
                run = 1

    # --- 3. simultaneous-gem chord load (limb load), high percentile ---
    by_ms = collections.defaultdict(set)
    for n in notes:
        by_ms[n.ms].add(n.lane)
    chord_sizes = [len(lanes) for lanes in by_ms.values()]
    peak_chord_size = _percentile(chord_sizes, PEAK_PERCENTILE)

    # --- 4. distinct-lane count (breadth) ---
    n_lanes = len(set(n.lane for n in notes))

    # --- 5. syncopation / off-beat load proxy ---
    # fraction of onsets NOT within a small tolerance of an integer 8th-note
    # grid position, using the local tempo -- a cheap off-beat proxy without
    # a full grid-inference dependency (this eval deliberately avoids
    # product_pipeline's grid machinery, see editrate.py's own docstring).
    off_beat = 0
    for ms in onsets:
        bpm = _bpm_at(tempos, ms)
        eighth_ms = (60000.0 / bpm) / 2.0
        phase = (ms % eighth_ms) / eighth_ms if eighth_ms > 0 else 0.0
        phase = min(phase, 1.0 - phase)  # distance to nearest grid line, folded to [0, 0.5]
        if phase > 0.15:  # >15% of an 8th-note off the grid counts as syncopated
            off_beat += 1
    syncopation_frac = off_beat / len(onsets)

    # --- 6. skeleton retention (I3/I6 merge) ---
    skeleton_retention = _skeleton_retention(notes)

    # --- normalize (fixed caps, see module docstring) ---
    def clamp01(x):
        return max(0.0, min(1.0, x))

    peak_density = max(peak_density_fixed, peak_density_beat)
    density_norm = clamp01(peak_density / DENSITY_CAP)
    if fastest_ioi is None:
        stream_norm = 0.0
    else:
        # faster (smaller) IOI -> harder -> closer to 1
        stream_norm = clamp01((STREAM_IOI_CAP_MS - fastest_ioi) / (STREAM_IOI_CAP_MS - STREAM_IOI_FLOOR_MS))
    chord_norm = clamp01((peak_chord_size - 1.0) / (CHORD_CAP - 1.0))  # 1 gem = 0 load
    lane_norm = clamp01(n_lanes / LANE_CAP)
    sync_norm = clamp01(syncopation_frac)
    # skeleton_retention is a DOWNWARD pull: low retention -> easier to follow -> subtract
    skeleton_penalty = clamp01(1.0 - skeleton_retention)

    # Fixed weights (sum to 1.0 across the five "harder" components; the
    # skeleton term is a small subtractive correction, not a sixth additive
    # component, so it can't push D above the other five's ceiling).
    D = (
        0.35 * density_norm
        + 0.25 * stream_norm
        + 0.20 * chord_norm
        + 0.10 * lane_norm
        + 0.10 * sync_norm
        - 0.10 * skeleton_penalty
    )
    D = clamp01(D)

    return {
        "n_notes": len(notes),
        "peak_density_fixed": peak_density_fixed,
        "peak_density_beat": peak_density_beat,
        "fastest_stream_ioi_ms": fastest_ioi,
        "longest_stream_run": longest_run,
        "peak_chord_size": peak_chord_size,
        "n_lanes": n_lanes,
        "syncopation_frac": syncopation_frac,
        "skeleton_retention": skeleton_retention,
        "density_norm": density_norm,
        "stream_norm": stream_norm,
        "chord_norm": chord_norm,
        "lane_norm": lane_norm,
        "sync_norm": sync_norm,
        "skeleton_penalty": skeleton_penalty,
        "D": D,
    }


def D(notes, tempos=None):
    """Convenience: scalar D only."""
    return compute_features(notes, tempos)["D"]
