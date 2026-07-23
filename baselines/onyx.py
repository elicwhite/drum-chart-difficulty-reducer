"""
Adapter: this repo's ms-based parsed-chart dict -> Onyx's Pro-resolved-Gem /
Fraction-beats `drums_complete` (_onyx_algo.py) -> back to this repo's
(ms, lane) note representation.

Independent reimplementation of the Onyx drum-reduction algorithm, vendored
for reproducible comparison. See _onyx_algo.py for the ported algorithm
itself and its full attribution/citation header.

CONVERSION APPROACH
--------------------
ms -> tick -> beats: _ticks.make_ms_to_tick (480 ticks/quarter, see that
module's docstring), then Fraction(tick, 480) for Onyx's exact-rational
`U.Beats` representation (Onyx's own alignment checks -- `isAligned`/
`isMeasure` -- compare beats for EXACT equality, so this only behaves
correctly if the reconstructed ticks really do land on-grid; see
_ticks.py).

Lane -> Onyx Gem: unlike HOPCAT, Onyx's algorithm (`drums_reduce`) consumes
already Pro-resolved gems (cymbal-vs-tom already decided) -- that's exactly
what `_onyx_algo.compute_pro` exists to produce FROM raw MIDI tom-marker
spans. Our chart's lane names already carry that resolution (e.g. `hihat`
vs `high-tom`), so we skip compute_pro entirely and map straight to the
resolved Gem via the module's own GEM_TO_LANE table, inverted. This is
more direct (and not weaker) than reconstructing marker spans would be:
our lane label IS the ground truth compute_pro would have produced.

Known, documented divergences from the real Onyx/MIDI eval:
  - No overdrive (OD) phrase data exists in this repo's chart schema
    (`ensureODNotes`, which guarantees >=1 note survives per OD phrase, is
    therefore always a no-op here -- `od_phrases=[]`).
  - `open-hat` has no Onyx Gem equivalent; mapped to `pro("yellow",
    "cymbal")`, same as `hihat` (matching the real game, which doesn't
    distinguish them at the Pro-drums-lane level either).
"""

from fractions import Fraction

from . import _ticks
from ._onyx_algo import Gem, build_measure_map, drums_complete, gem_to_lane

# Our 9-lane name -> already Pro-resolved Onyx Gem.
LANE_TO_GEM = {
    "kick": Gem("kick"),
    "snare": Gem("red"),
    "hihat": Gem("pro", "yellow", "cymbal"),
    "open-hat": Gem("pro", "yellow", "cymbal"),
    "high-tom": Gem("pro", "yellow", "tom"),
    "ride": Gem("pro", "blue", "cymbal"),
    "mid-tom": Gem("pro", "blue", "tom"),
    "crash": Gem("pro", "green", "cymbal"),
    "floor-tom": Gem("pro", "green", "tom"),
}
_TIER_CODE = {"hard": "h", "medium": "m", "easy": "e"}

TPQ = _ticks.TICKS_PER_QUARTER


def reduce_onyx(chart, tier):
    """chart: this repo's parsed-chart dict. tier: "hard" | "medium" |
    "easy". Returns [{"ms": float, "lane": str}, ...], sorted (ms, lane)."""
    tier_code = _TIER_CODE[tier]
    ms_to_tick = _ticks.make_ms_to_tick(chart.get("tempos") or [])
    tick_to_ms = _ticks.make_tick_to_ms(chart.get("tempos") or [])
    ts_ticks = _ticks.timesig_ticks(chart.get("timeSignatures"), ms_to_tick)

    expert = chart["difficulties"]["expert"]["notes"]
    expert_pro = []
    max_tick = 0
    for ms, lanes in expert:
        tick = ms_to_tick(ms)
        max_tick = max(max_tick, tick)
        for entry in lanes:
            gem = LANE_TO_GEM.get(entry["instrument"])
            if gem is None:
                continue
            expert_pro.append((Fraction(tick, TPQ), gem))
    if not expert_pro:
        return []
    expert_pro.sort(key=lambda pg: pg[0])

    ts_beats = [(Fraction(tick, TPQ), num, den) for tick, num, den in ts_ticks]
    mm = build_measure_map(ts_beats, Fraction(max_tick, TPQ))

    sections = sorted(
        (Fraction(ms_to_tick(s["ms"]), TPQ), s["name"]) for s in (chart.get("sections") or [])
    )
    od_phrases = []  # not present in this repo's chart schema -- see module docstring

    result = drums_complete(mm, od_phrases, sections, expert_pro)

    out = []
    for pos, gem in result[tier_code]:
        ms = tick_to_ms(float(pos * TPQ))
        out.append({"ms": ms, "lane": gem_to_lane(gem)})
    out.sort(key=lambda d: (d["ms"], d["lane"]))
    return out
