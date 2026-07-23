"""
Standalone note-level edit_rate metric (stdlib only) -- this is the metric
`test_parity.py` scores the reducer against; it is not itself part of the
reduce() decode path.

A "note" here is (ms, lane): ms is scan-chart's already-tempo-resolved
millisecond time, lane is the cleaned drum instrument string
(kick/snare/hihat/open-hat/high-tom/mid-tom/floor-tom/crash/ride).
"""

from dataclasses import dataclass

EPS_MS = 0.5  # float-rounding slack for "same tick" (W=0) matching


@dataclass(frozen=True)
class Note:
    ms: float
    lane: str


def notes_from_difficulty(diff_data):
    """Flatten [ms, [{instrument, ...}, ...]] grouped notes into a flat,
    ms-sorted list of Note(ms, lane). One Note per (ms, lane) pair; a chord
    (multiple lanes at the same ms) becomes multiple Notes."""
    if diff_data is None:
        return []
    notes = []
    for ms, lanes in diff_data["notes"]:
        for entry in lanes:
            notes.append(Note(ms, entry["instrument"]))
    notes.sort(key=lambda n: (n.ms, n.lane))
    return notes


def eighth_note_ms(tempos, ms):
    if not tempos:
        bpm = 120.0
    else:
        bpm = tempos[0]["bpm"]
        for t in tempos:
            if t["ms"] <= ms:
                bpm = t["bpm"]
            else:
                break
    return (60000.0 / bpm) / 2.0


def _match(cand, gt, window_ms):
    """Greedy 1:1 same-lane-preferred nearest match within +/-window_ms."""
    def window_at(gms):
        return window_ms(gms) if callable(window_ms) else window_ms

    candidates = []
    for gi, g in enumerate(gt):
        w = window_at(g.ms)
        for ci, c in enumerate(cand):
            dist = abs(c.ms - g.ms)
            if dist <= w:
                same_lane = 0 if c.lane == g.lane else 1
                candidates.append((same_lane, dist, gi, ci))

    candidates.sort()  # priority: same-lane first, then closest, then stable by (gi, ci)

    used_c, used_g = set(), set()
    pairs = []
    for _, _, gi, ci in candidates:
        if gi in used_g or ci in used_c:
            continue
        used_g.add(gi)
        used_c.add(ci)
        pairs.append((ci, gi))

    unmatched_c = [i for i in range(len(cand)) if i not in used_c]
    unmatched_g = [i for i in range(len(gt)) if i not in used_g]
    return pairs, unmatched_c, unmatched_g


def edit_ops(cand, gt, window_ms=EPS_MS):
    """Edit ops to turn `cand` into `gt`: insert (unmatched gt note), delete
    (unmatched cand note), lane_move (matched pair, different lane),
    slot_move (matched pair, different tick, within window_ms)."""
    pairs, unmatched_c, unmatched_g = _match(cand, gt, window_ms)

    lane_move = 0
    slot_move = 0
    for ci, gi in pairs:
        c, g = cand[ci], gt[gi]
        if c.lane != g.lane:
            lane_move += 1
        if abs(c.ms - g.ms) > EPS_MS:
            slot_move += 1

    return {
        "insert": len(unmatched_g),
        "delete": len(unmatched_c),
        "lane_move": lane_move,
        "slot_move": slot_move,
    }


def edit_rate(cand, gt, window_ms=EPS_MS):
    """total_edits / |gt|. rate is None if |gt| == 0."""
    ops = edit_ops(cand, gt, window_ms)
    total = ops["insert"] + ops["delete"] + ops["lane_move"] + ops["slot_move"]
    rate = (total / len(gt)) if gt else None
    return rate, ops
