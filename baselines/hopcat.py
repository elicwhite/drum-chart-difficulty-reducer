"""
Adapter: this repo's ms-based parsed-chart dict -> HOPCAT's raw-5-lane /
tick-based `reduce_5lane_drums` (_hopcat_algo.py) -> back to this repo's
(ms, lane) note representation.

Independent reimplementation of the HOPCAT drum-reduction algorithm,
vendored for reproducible comparison. See _hopcat_algo.py for the ported
algorithm itself and its full attribution/citation header.

CONVERSION APPROACH
--------------------
ms -> tick: _ticks.make_ms_to_tick/make_tick_to_ms (480 ticks/quarter,
inverting the same tempo-map integration that produced our chart's ms
values in the first place -- see _ticks.py's own docstring).

Lane -> HOPCAT raw 5-lane pitch: HOPCAT's `reduce_5lane` never resolves
cymbal-vs-tom itself -- Yellow/Blue/Green are single raw pitches per tier,
and cymbal/tom is purely a MIDI-track marker (110-112) consulted only by
scan-chart at PARSE time, not by the reduction algorithm, which just
thins/copies whichever raw pitch was there. Concretely: every surviving
output note's (tick, raw-lane) pair is EITHER an untouched original
Expert position OR a straight tier-shifted copy of one (DEFAULT_CONFIG's
`simplify_roll`/`unflip_discobeat` steps are no-ops here since our charts
carry no roll/swell/mix-marker events to begin with -- see below), so we
can recover the correct final 9-lane name by remembering, at every (tick,
raw-color) we hand to the algorithm, which of our own two lane names
(cymbal or tom) it came from, and looking that back up on the way out --
without needing to reconstruct literal tom-marker note spans at all.

Known, documented divergences from the real HOPCAT/MIDI eval:
  - No `[mix N drumsXd]` disco-flip or roll/swell (126/127) marker events
    exist in this repo's chart schema, so `unflip_discobeat` and
    `simplify_roll` are always no-ops here (they would only matter for a
    minority of RB-authored charts anyway).
  - `remove_kick`'s `what='p'` branch (used by DEFAULT_CONFIG's Medium/Easy
    kick removal) additionally hits on a literal tom-marker (pitch
    110-112) NOTE EVENT coinciding with a kick+X chord at the exact same
    tick -- a real but incidental case (a marker's note-on/off happening to
    land on that tick), not a semantic "is there a tom here" signal. We
    have no legitimate source for those marker note positions (see above:
    we resolve cymbal/tom directly, never via markers), so this port's
    `what='p'` only fires on the snare-present case (`what='s'`), i.e. the
    marker-coincidence sub-case is dropped. Documented rather than faked.
  - `open-hat` has no HOPCAT/RB 5-lane equivalent; mapped to the same raw
    lane as `hihat` (Yellow-cymbal), matching how the real 5-lane game
    represents both.
"""

from . import _ticks
from ._hopcat_algo import (
    DEFAULT_CONFIG,
    LANE_OFFSET,
    OFFSET_LANE,
    Note,
    TIER_BASE,
    build_measures,
    reduce_5lane_drums,
    tier_of,
)

# Our 9-lane name -> HOPCAT raw 5-lane color.
LANE_TO_RAW = {
    "kick": "kick",
    "snare": "snare",
    "hihat": "yellow",
    "open-hat": "yellow",
    "high-tom": "yellow",
    "ride": "blue",
    "mid-tom": "blue",
    "crash": "green",
    "floor-tom": "green",
}

# Fallback if an output (tick, raw-color) wasn't seen on the way in (should
# only happen if a future DEFAULT_CONFIG change makes simplify_roll/
# unflip_discobeat non-trivial) -- picks the cymbal reading, the more common
# case in real charts.
_RAW_FALLBACK = {"yellow": "hihat", "blue": "ride", "green": "crash"}

_TIER_CODE = {"hard": "h", "medium": "m", "easy": "e"}


def reduce_hopcat(chart, tier):
    """chart: this repo's parsed-chart dict (see featurize.py's docstring
    for the schema). tier: "hard" | "medium" | "easy". Returns
    [{"ms": float, "lane": str}, ...], sorted (ms, lane)."""
    tier_code = _TIER_CODE[tier]
    ms_to_tick = _ticks.make_ms_to_tick(chart.get("tempos") or [])
    tick_to_ms = _ticks.make_tick_to_ms(chart.get("tempos") or [])
    ts_ticks = _ticks.timesig_ticks(chart.get("timeSignatures"), ms_to_tick)

    expert = chart["difficulties"]["expert"]["notes"]
    notes = []
    raw_lookup = {}  # (tick, raw_color) -> our original 9-lane lane name
    max_tick = 0
    for ms, lanes in expert:
        tick = ms_to_tick(ms)
        max_tick = max(max_tick, tick)
        for entry in lanes:
            lane = entry["instrument"]
            raw = LANE_TO_RAW.get(lane)
            if raw is None:
                continue
            if raw in ("yellow", "blue", "green"):
                raw_lookup[(tick, raw)] = lane
            pitch = TIER_BASE["x"] + LANE_OFFSET[raw]
            notes.append(Note(tick, pitch, 100, 1))

    if not notes:
        return []

    mm = build_measures(ts_ticks, 480, max_tick)
    out_notes, _out_events = reduce_5lane_drums(notes, [], mm, DEFAULT_CONFIG)

    result = []
    for n in out_notes:
        if tier_of(n.pitch) != tier_code:
            continue
        raw = OFFSET_LANE[n.pitch - TIER_BASE[tier_code]]
        if raw in ("kick", "snare"):
            lane = raw
        else:
            lane = raw_lookup.get((n.pos, raw), _RAW_FALLBACK[raw])
        result.append({"ms": tick_to_ms(n.pos), "lane": lane})

    result.sort(key=lambda d: (d["ms"], d["lane"]))
    return result
