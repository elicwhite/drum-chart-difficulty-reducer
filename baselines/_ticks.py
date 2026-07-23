"""
ms <-> 480-ticks-per-quarter-note conversion, shared by both baseline
adapters (hopcat.py, onyx.py).

Both vendored algorithms (see _hopcat_algo.py, _onyx_algo.py) operate on a
tick/beat grid, not our chart's ms timestamps: HOPCAT's grid math is
hardcoded to CORRECT_TQN=480 ticks/quarter (matching C3toolbox itself, see
_hopcat_algo.AMBIGUITIES #5), and Onyx's `U.Beats` are exact quarter-note
Fractions, which this repo's charts don't carry directly.

Our chart's `ms` values were themselves produced (upstream, by scan-chart)
by integrating tick positions through this repo's tempo map -- i.e. they
started life as ticks. This module inverts that integration: given the
chart's `tempos` list, it reconstructs the same piecewise-linear ms<->tick
map scan-chart used, so round-tripping ms -> tick -> ms recovers the
original on-grid tick positions (up to float rounding) rather than
introducing new quantization noise of our own.
"""

import bisect

TICKS_PER_QUARTER = 480


def _tempo_breakpoints(tempos):
    """[{"ms","bpm"}, ...] (any order) -> sorted [(ms, bpm), ...] with a
    ms=0 entry guaranteed first (falls back to 120bpm if the chart has no
    tempo map at all, matching editrate.eighth_note_ms's own convention)."""
    tempos = sorted(tempos, key=lambda t: t["ms"]) if tempos else []
    if not tempos:
        return [(0.0, 120.0)]
    if tempos[0]["ms"] > 0:
        tempos = [{"ms": 0.0, "bpm": tempos[0]["bpm"]}] + tempos
    return [(t["ms"], t["bpm"]) for t in tempos]


def _ms_tick_breakpoints(tempos):
    """Integrate the tempo map forward into (ms, tick) breakpoints at
    TICKS_PER_QUARTER, tick=0 at the first breakpoint's ms."""
    bps = _tempo_breakpoints(tempos)
    out = [(bps[0][0], 0.0)]
    for i in range(1, len(bps)):
        prev_ms, prev_tick = out[-1]
        prev_bpm = bps[i - 1][1]
        ms_per_tick = (60000.0 / prev_bpm) / TICKS_PER_QUARTER if prev_bpm > 0 else 0.0
        dtick = (bps[i][0] - prev_ms) / ms_per_tick if ms_per_tick > 0 else 0.0
        out.append((bps[i][0], prev_tick + dtick))
    return out, [bpm for _ms, bpm in bps]


def make_ms_to_tick(tempos):
    breakpoints, bpms = _ms_tick_breakpoints(tempos)
    bp_ms = [ms for ms, _tick in breakpoints]

    def ms_to_tick(ms):
        i = min(bisect.bisect_right(bp_ms, ms) - 1, len(breakpoints) - 1)
        i = max(i, 0)
        seg_ms, seg_tick = breakpoints[i]
        bpm = bpms[i]
        ms_per_tick = (60000.0 / bpm) / TICKS_PER_QUARTER if bpm > 0 else 0.0
        tick = seg_tick + (ms - seg_ms) / ms_per_tick if ms_per_tick > 0 else seg_tick
        return int(round(tick))

    return ms_to_tick


def make_tick_to_ms(tempos):
    breakpoints, bpms = _ms_tick_breakpoints(tempos)
    bp_tick = [tick for _ms, tick in breakpoints]

    def tick_to_ms(tick):
        i = min(bisect.bisect_right(bp_tick, tick) - 1, len(breakpoints) - 1)
        i = max(i, 0)
        seg_ms, seg_tick = breakpoints[i]
        bpm = bpms[i]
        ms_per_tick = (60000.0 / bpm) / TICKS_PER_QUARTER if bpm > 0 else 0.0
        return seg_ms + (tick - seg_tick) * ms_per_tick

    return tick_to_ms


def timesig_ticks(time_sigs, ms_to_tick):
    """[{"ms","numerator","denominator"}, ...] -> sorted [(tick, num, den), ...]."""
    out = [(ms_to_tick(t["ms"]), t["numerator"], t["denominator"]) for t in (time_sigs or [])]
    out.sort(key=lambda e: e[0])
    return out
