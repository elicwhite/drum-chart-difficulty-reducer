"""
Independent reimplementation of the Onyx drum-reduction algorithm
(`drumsComplete`/`drumsReduce`), vendored here for reproducible comparison
against Harmonix's official reductions. Attribution: Onyx Music Game Toolkit
(onyx-lib, Onyx.Reductions / Onyx.MIDI.Track.Drums), by mtolly -- this is a
from-scratch Python port written by reading that Haskell source, not a
translation of it; see the per-function Reductions.hs/Drums.hs:NNNN
citations below.

This module operates on abstract Gem/Beats objects only (no MIDI I/O) and
is stdlib-only. `computePro` (raw-pitch -> Pro-gem resolution via tom
markers) is included but unused by this repo's adapter -- see
../baselines/onyx.py, which feeds this repo's own chart format in already
Pro-resolved (our lane names already distinguish cymbal/tom) rather than
reconstructing raw tom-marker spans.

Standalone Python port of Onyx's `drumsComplete`/`drumsReduce` drum
difficulty-reducer.

Source: the Onyx Music Game Toolkit (onyx-lib)'s `Onyx/Reductions.hs`
(function/line numbers cited below refer to this file) and
`Onyx/MIDI/Track/Drums.hs` (for `computePro`, the Gem/ProType taxonomy,
and the drum-mix/disco text-command format). PORT-ONLY: there is no way to
run the real Haskell tool in this environment (no Haskell toolchain), so
fidelity to the reading of Reductions.hs/Drums.hs IS the correctness bar.
Every non-obvious rule below cites its source line; genuine ambiguities are
marked `# AMBIGUITY:` and summarized in AMBIGUITIES at the bottom.

SCOPE: only the DRUMS path (`drumsComplete`/`drumsReduce`/`ensureODNotes`,
Reductions.hs:369-537) is ported. `computePro` (Drums.hs:328-345) is also
ported since it's the one piece of `D.DrumTrack` machinery drumsComplete
depends on to turn raw 5-lane Expert gems into Pro (cymbal/tom-resolved)
gems -- implemented natively here (from raw MIDI tom markers 110-112 +
Expert's own `[mix 3 ...]` disco marker) rather than trusted from
scan-chart's parser, so this port's correctness doesn't depend on an
unverified assumption that scan-chart's pro-drums/disco resolution matches
Onyx's bit-for-bit. simpleReduce/fillDrumAnimation/velocity handling are out
of scope (editrate.py's (ms, lane) matching never looks at velocity or
animation).

DATA MODEL: everything here operates in exact rational "beats" (Python
Fraction, ticks/ticks_per_quarter) -- this mirrors Onyx's own `U.Beats`
type (a `Ratio Integer`), which is exactly why Reductions.hs can compare
`frac == 0` after `properFraction` with no epsilon: beats are always exact
rationals in the source, and they are here too. Positions are `Fraction`
throughout; there is no "tick" type in this module (ticks only exist at the
MIDI-import boundary, out of scope for this repo -- see ../baselines/onyx.py,
which converts this repo's already-parsed ms-based notes straight to
`Fraction` beats, never through raw MIDI ticks).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

Beats = Fraction

# ---------------------------------------------------------------------------
# Gem / ProType taxonomy -- Drums.hs:156-163.
#   data ProColor = Yellow | Blue | Green
#   data ProType  = Cymbal | Tom
#   data Gem a    = Kick | Red | Pro ProColor a | Orange
# `Orange` (the legacy 5-lane-only green-cymbal-or-tom-ambiguous lane) is not
# modeled: pro-drums charts (the only kind this eval scores) never emit it
# -- computePro's own `Orange -> Orange -- probably shouldn't happen`
# (Drums.hs:344) confirms the original tool doesn't expect it either. See
# AMBIGUITIES #1.
# ---------------------------------------------------------------------------

COLORS = ("yellow", "blue", "green")
PROTYPES = ("cymbal", "tom")

# Derived-Ord ranks: Haskell's `deriving (Ord)` ranks sum constructors in
# declaration order, and for `Pro ProColor ProType` compares fields
# left-to-right (color, then type). This is `sort gems` (Reductions.hs:476)
# and the coincident-position priority ordering elsewhere.
_KIND_RANK = {"kick": 0, "red": 1, "pro": 2}
_COLOR_RANK = {"yellow": 0, "blue": 1, "green": 2}
_TYPE_RANK = {"cymbal": 0, "tom": 1}


@dataclass(frozen=True)
class Gem:
    kind: str  # 'kick' | 'red' | 'pro'
    color: str = ""  # '' | 'yellow' | 'blue' | 'green' (kind == 'pro' only)
    protype: str = ""  # '' | 'cymbal' | 'tom' (kind == 'pro' only)

    def sort_key(self) -> Tuple[int, int, int]:
        return (_KIND_RANK[self.kind], _COLOR_RANK.get(self.color, -1), _TYPE_RANK.get(self.protype, -1))


KICK = Gem("kick")
RED = Gem("red")


def pro(color: str, protype: str) -> Gem:
    return Gem("pro", color, protype)


def sort_gems(gems: List[Gem]) -> List[Gem]:
    return sorted(gems, key=Gem.sort_key)


# scan-chart / editrate.py lane-string mapping, used only by the I/O layer
# for producing (ms, lane) output -- listed here (not in a MIDI-I/O module,
# which this repo doesn't have) because it's a property of the Gem
# taxonomy itself, not of MIDI I/O. This is the instrument<->gem mapping
# this adapter needs, in its authoritative direction (Onyx gem -> our lane
# string); the reverse direction (raw MIDI -> Onyx gem) is out of scope for
# this repo, which consumes already-parsed ms-based charts.
GEM_TO_LANE = {
    ("kick", "", ""): "kick",
    ("red", "", ""): "snare",
    ("pro", "yellow", "cymbal"): "hihat",
    ("pro", "yellow", "tom"): "high-tom",
    ("pro", "blue", "cymbal"): "ride",
    ("pro", "blue", "tom"): "mid-tom",
    ("pro", "green", "cymbal"): "crash",
    ("pro", "green", "tom"): "floor-tom",
}


def gem_to_lane(g: Gem) -> str:
    return GEM_TO_LANE[(g.kind, g.color, g.protype)]


# ---------------------------------------------------------------------------
# computePro -- Drums.hs:328-345. Resolves raw 5-lane Expert gems (Kick,
# Red, Pro Yellow/Blue/Green ()) into Pro gems (cymbal/tom-resolved, with
# disco-flip applied) using the tom-marker status track (110/111/112) and
# this DIFFICULTY's OWN `[mix <diffnum> drums<audio><flag>]` disco markers
# (`this.drumMix`, Drums.hs:332 -- note it's per-difficulty, not shared).
#
# Onyx's `applyStatus` (Onyx.Guitar) combines a `RTB.T t (k, Bool)` "status"
# stream with a note stream, giving each note the list of keys `k` whose
# status is CURRENTLY True (the most recent edge at-or-before that note's
# position was `True`). `instantToms`/`instantDisco` are exactly this
# membership list -- see Drums.hs:333,343 (`elem color instantToms`) and
# :336 (`isDisco = not $ null instantDisco`, which only makes sense under
# this "membership list, not raw value" reading of applyStatus).
# ---------------------------------------------------------------------------


def compute_pro(
    raw_gems: List[Tuple[Beats, Gem]],
    tom_status: Dict[str, List[Tuple[Beats, bool]]],
    disco_status: List[Tuple[Beats, bool]],
) -> List[Tuple[Beats, Gem]]:
    """raw_gems: (pos, Gem) with Gem.kind in {'kick','red','pro'} and (for
    'pro') Gem.protype == '' (unresolved -- color only). tom_status[color]:
    sorted (pos, is_tom) edges for that color (110/111/112). disco_status:
    sorted (pos, is_disco) edges from this tier's own drumMix text events.
    """

    def status_at(edges: List[Tuple[Beats, bool]], pos: Beats) -> bool:
        # applyStatus-style: the most recent edge at or before `pos`
        # (edges must be pre-sorted by position by the caller).
        idx = bisect.bisect_right([e[0] for e in edges], pos) - 1
        return edges[idx][1] if idx >= 0 else False

    out: List[Tuple[Beats, Gem]] = []
    for pos, g in raw_gems:
        is_disco = status_at(disco_status, pos)
        if g.kind == "kick":
            out.append((pos, KICK))
        elif g.kind == "red":
            out.append((pos, pro("yellow", "cymbal") if is_disco else RED))
        elif g.kind == "pro":
            if g.color == "yellow" and is_disco:
                out.append((pos, RED))
            else:
                is_tom = status_at(tom_status.get(g.color, []), pos)
                out.append((pos, pro(g.color, "tom" if is_tom else "cymbal")))
        else:
            raise ValueError(f"unexpected raw gem kind {g.kind!r} (Orange not supported, see AMBIGUITIES #1)")
    return out


# ---------------------------------------------------------------------------
# Measure map -- beats-within-measure for isMeasure/isAligned. Onyx's
# `U.MeasureMap`/`U.applyMeasureMap` are opaque (external `midi-util`
# package, not vendored in this checkout) but their observable contract
# (Reductions.hs:452-456) is: `applyMeasureMap mmap bts` gives
# `(measureNumber, beatsWithinMeasure)`, where beats are always literal
# quarter-note beats (NOT rescaled by the time signature denominator -- see
# AMBIGUITIES #2 for why this is the standard/expected MIDI convention and
# not a guess). A measure starting under time signature num/den is
# `num * 4/den` quarter-note-beats long -- the standard formula (the same
# tick-length arithmetic HOPCAT's reduce_port.py uses, just expressed in
# beats instead of ticks, since "ticks per beat" cancels out).
# ---------------------------------------------------------------------------


class MeasureMap:
    def __init__(self, starts: List[Beats]):
        assert starts, "need at least one measure"
        self.starts = starts  # sorted, ascending

    def beats_within_measure(self, pos: Beats) -> Beats:
        idx = bisect.bisect_right(self.starts, pos) - 1
        if idx < 0:
            idx = 0
        return pos - self.starts[idx]


def build_measure_map(ts_events: List[Tuple[Beats, int, int]], end_beats: Beats) -> MeasureMap:
    """ts_events: (start_beats, numerator, denominator), from the MIDI
    time-signature track converted to beats at the I/O boundary. Segments
    are walked forward emitting one measure start per bar, exactly like
    HOPCAT reduce_port.build_measures (see that module for the "run one
    extra bar past the last TS event" fudge, reused here identically)."""
    if not ts_events or ts_events[0][0] != 0:
        ts_events = [(Beats(0), 4, 4)] + list(ts_events)
    ts_events = sorted(ts_events, key=lambda e: e[0])

    starts: List[Beats] = []
    for i, (start, num, den) in enumerate(ts_events):
        bar_beats = Beats(num * 4, den)
        seg_end = ts_events[i + 1][0] if i + 1 < len(ts_events) else max(end_beats, start) + bar_beats
        pos = start
        while pos < seg_end:
            starts.append(pos)
            pos += bar_beats
    return MeasureMap(starts)


def is_measure(mm: MeasureMap, pos: Beats) -> bool:
    return mm.beats_within_measure(pos) == 0


def is_aligned(mm: MeasureMap, divn, pos: Beats) -> bool:
    r = mm.beats_within_measure(pos)
    q = r / Beats(divn)
    return q.denominator == 1


# ---------------------------------------------------------------------------
# drumsReduce -- Reductions.hs:441-537 (drums-only; Expert short-circuits at
# :448 and is not represented here -- callers never invoke this for Expert).
# `diff` is one of 'h'/'m'/'e' (Hard/Medium/Easy).
# ---------------------------------------------------------------------------

_DIFF_RANK = {"e": 0, "m": 1, "h": 2}  # Easy < Medium < Hard, Reductions.hs:479 `diff <= Medium`


def priority(mm: MeasureMap, pos: Beats) -> int:
    """Reductions.hs:457-463. Lower = more important (kept preferentially)."""
    return (
        (0 if is_measure(mm, pos) else 1)
        + (0 if is_aligned(mm, 2, pos) else 1)
        + (0 if is_aligned(mm, 1, pos) else 1)
        + (0 if is_aligned(mm, Beats(1, 2), pos) else 1)
    )


def _open_interval(sorted_keys: List[Beats], lo: Beats, hi: Beats) -> List[Beats]:
    """Keys k with lo < k < hi -- Map.split/Set.split both EXCLUDE the
    split point itself, matching `fst (Set.split (posn+padding)) . snd
    (Set.split (posn-padding))` (Reductions.hs:171 et al)."""
    i = bisect.bisect_right(sorted_keys, lo)
    j = bisect.bisect_left(sorted_keys, hi)
    return sorted_keys[i:j]


def keep_snares(mm: MeasureMap, diff: str, snare_positions: List[Beats]) -> Tuple[Dict[Beats, List[Gem]], List[Beats]]:
    """Reductions.hs:464-472. `snare_positions`: distinct tick positions
    where the source has a Red gem (ATB.getTimes -- a position, not a
    count; duplicate simultaneous Reds can't happen, RTB.filter (==Red)
    already collapses to one event per coincident group in practice)."""
    ordered = sorted(snare_positions, key=lambda p: (priority(mm, p), p))
    padding = Beats(1, 2) if diff == "h" else Beats(1)
    kept: Dict[Beats, List[Gem]] = {}
    keys: List[Beats] = []
    for pos in ordered:
        if not _open_interval(keys, pos - padding, pos + padding):
            kept[pos] = [RED]
            bisect.insort(keys, pos)
    return kept, keys


def keep_kit(
    mm: MeasureMap, diff: str, kit_by_pos: List[Tuple[Beats, List[Gem]]], kept: Dict[Beats, List[Gem]], keys: List[Beats]
) -> Tuple[Dict[Beats, List[Gem]], List[Beats]]:
    """Reductions.hs:473-486. `kit_by_pos`: coincident non-Red/non-Kick
    gems from the source, one entry per distinct position. `kept`/`keys`:
    the snare map from keep_snares, mutated in place (kit notes share the
    same collision window as already-kept snares)."""
    ordered = sorted(kit_by_pos, key=lambda pg: (priority(mm, pg[0]), pg[0]))
    padding = Beats(1, 2) if diff == "h" else Beats(1)
    for pos, gems in ordered:
        s = sort_gems(gems)
        if len(s) == 2 and s[0].kind == "pro" and s[0].protype == "cymbal" and s[1] == pro("green", "cymbal"):
            gems2 = [pro("green", "cymbal")]
        elif s == [pro("yellow", "cymbal"), pro("blue", "cymbal")]:
            gems2 = [pro("blue", "cymbal")]
        elif (
            len(s) == 2
            and s[0].kind == "pro"
            and s[0].protype == "tom"
            and s[1].kind == "pro"
            and s[1].protype == "tom"
            and _DIFF_RANK[diff] <= _DIFF_RANK["m"]
        ):
            gems2 = [s[0]]
        else:
            gems2 = gems

        window = _open_interval(keys, pos - padding, pos + padding)
        ok = (not window) or (window == [pos] and pos in kept)
        if ok:
            if pos not in kept:
                bisect.insort(keys, pos)
                kept[pos] = list(gems2)
            else:
                kept[pos] = list(gems2) + kept[pos]
    return kept, keys


def keep_kicks(
    mm: MeasureMap, diff: str, kick_positions: List[Beats], kept: Dict[Beats, List[Gem]], keys: List[Beats]
) -> Tuple[Dict[Beats, List[Gem]], List[Beats]]:
    """Reductions.hs:487-500."""
    ordered = sorted(kick_positions, key=lambda p: (priority(mm, p), p))
    padding = Beats(1) if diff == "h" else Beats(2)
    for pos in ordered:
        window = _open_interval(keys, pos - padding, pos + padding)
        has_kick = any(KICK in kept[k] for k in window)
        has_one_hand_gem = pos in kept and len(kept[pos]) == 1
        if (not has_kick) and (diff != "m" or not window or has_one_hand_gem):
            if pos not in kept:
                bisect.insort(keys, pos)
                kept[pos] = [KICK]
            else:
                kept[pos] = [KICK] + kept[pos]
    return kept, keys


# ---------------------------------------------------------------------------
# Easy-only per-section simplification -- Reductions.hs:501-536.
# ---------------------------------------------------------------------------


def _slice_gems(keys: List[Beats], kept: Dict[Beats, List[Gem]], start: Beats, end: Optional[Beats]) -> List[Gem]:
    """Reductions.hs:507-509 (Map.splitLookup start progress, then Map.elems
    of [start, end)). start is INCLUSIVE (via startNote), end EXCLUSIVE."""
    out: List[Gem] = []
    if start in kept:
        out.extend(kept[start])
    i = bisect.bisect_right(keys, start)
    j = len(keys) if end is None else bisect.bisect_left(keys, end)
    for k in keys[i:j]:
        out.extend(kept[k])
    return out


def make_easy(
    keys: List[Beats], kept: Dict[Beats, List[Gem]], start: Beats, end: Optional[Beats]
) -> Tuple[Dict[Beats, List[Gem]], List[Beats]]:
    """Reductions.hs:507-519."""
    sl = _slice_gems(keys, kept, start, end)
    n_kicks = sum(1 for g in sl if g == KICK)
    n_hihats = sum(1 for g in sl if g == pro("yellow", "cymbal"))
    n_other_kit = sum(1 for g in sl if g not in (KICK, RED) and g != pro("yellow", "cymbal"))
    if n_kicks == 0:
        fn = _make_no_kick
    elif n_kicks > n_hihats + n_other_kit:
        fn = _make_snare_kick
    elif n_hihats > n_other_kit:
        fn = _make_snare_kick
    else:
        fn = _make_no_kick
    return fn(keys, kept, start, end)


def _in_range(k: Beats, start: Beats, end: Optional[Beats]) -> bool:
    return start <= k and (end is None or k < end)


def _make_snare_kick(
    keys: List[Beats], kept: Dict[Beats, List[Gem]], start: Beats, end: Optional[Beats]
) -> Tuple[Dict[Beats, List[Gem]], List[Beats]]:
    """Reductions.hs:520-525."""
    new_kept: Dict[Beats, List[Gem]] = {}
    new_keys: List[Beats] = []
    for k in keys:
        gems = kept[k]
        if _in_range(k, start, end):
            if k == start and pro("green", "cymbal") in gems:
                filtered = [g for g in gems if g != KICK]
            else:
                filtered = [g for g in gems if g in (KICK, RED)]
            if filtered:
                new_kept[k] = filtered
                new_keys.append(k)
            # else: dropped (nullNothing)
        else:
            new_kept[k] = gems
            new_keys.append(k)
    return new_kept, new_keys


def _make_no_kick(
    keys: List[Beats], kept: Dict[Beats, List[Gem]], start: Beats, end: Optional[Beats]
) -> Tuple[Dict[Beats, List[Gem]], List[Beats]]:
    """Reductions.hs:526-529."""
    new_kept: Dict[Beats, List[Gem]] = {}
    new_keys: List[Beats] = []
    for k in keys:
        gems = kept[k]
        if _in_range(k, start, end):
            filtered = [g for g in gems if g != KICK]
            if filtered:
                new_kept[k] = filtered
                new_keys.append(k)
        else:
            new_kept[k] = gems
            new_keys.append(k)
    return new_kept, new_keys


# ---------------------------------------------------------------------------
# ensureODNotes -- Reductions.hs:419-439. Operates on FLAT (pos, Gem) event
# streams (not grouped by position), matching Onyx's `RTB.T t (D.Gem
# D.ProType)` representation there. Guarantees every OD phrase keeps >= 1
# note by reinserting the single earliest ORIGINAL (pre-reduction) event at
# or after the phrase start if the reduced stream has none in the phrase's
# span. See AMBIGUITIES #3 for the "which coincident event, if the earliest
# original position is a chord" tie-break.
# ---------------------------------------------------------------------------


def ensure_od_notes(
    od_phrases: List[Tuple[Beats, Beats]], original_flat: List[Tuple[Beats, Gem]], reduced_flat: List[Tuple[Beats, Gem]]
) -> List[Tuple[Beats, Gem]]:
    reduced = list(reduced_flat)
    reduced_positions = sorted(p for p, _ in reduced)
    original_sorted = sorted(original_flat, key=lambda pg: pg[0])
    for start, end in od_phrases:
        i = bisect.bisect_left(reduced_positions, start)
        j = bisect.bisect_left(reduced_positions, end)
        if i < j:
            continue  # already has >=1 reduced note in [start, end)
        k = bisect.bisect_left([p for p, _ in original_sorted], start)
        if k < len(original_sorted):
            p0, g0 = original_sorted[k]
            reduced.append((p0, g0))
            bisect.insort(reduced_positions, p0)
    return reduced


# ---------------------------------------------------------------------------
# Orchestrator -- Reductions.hs:441-537 (drumsReduce) wired into the
# Hard<-Expert, Medium<-Hard, Easy<-Hard cascade of drumsComplete
# (:389-417). This port always regenerates H/M/E from Expert (treat
# existing lower tiers as empty, matching drumsComplete's
# `length raw.drumGems <= 5` branch since we intentionally discard whatever
# GT tiers exist -- they're the scoring target, not an input).
# ---------------------------------------------------------------------------


def drums_reduce(
    diff: str,
    mm: MeasureMap,
    od_phrases: List[Tuple[Beats, Beats]],
    sections: List[Tuple[Beats, str]],
    source: List[Tuple[Beats, Gem]],
) -> List[Tuple[Beats, Gem]]:
    """source: flat (pos, Gem) list, Pro-resolved, sorted by pos -- the tier
    one level up (already itself a `drumsReduce` output for Medium/Easy, or
    `compute_pro` output for Hard's Expert source)."""
    snare_positions = sorted({p for p, g in source if g == RED})
    kick_positions = sorted({p for p, g in source if g == KICK})
    kit_by_pos: Dict[Beats, List[Gem]] = {}
    for p, g in source:
        if g not in (RED, KICK):
            kit_by_pos.setdefault(p, []).append(g)
    kit_sorted = sorted(kit_by_pos.items())

    kept, keys = keep_snares(mm, diff, snare_positions)
    kept, keys = keep_kit(mm, diff, kit_sorted, kept, keys)
    kept, keys = keep_kicks(mm, diff, kick_positions, kept, keys)

    if diff == "e":
        section_starts = sorted(sections, key=lambda sn: sn[0])
        bounds = [
            (section_starts[i][0], section_starts[i + 1][0] if i + 1 < len(section_starts) else None)
            for i in range(len(section_starts))
        ]
        for start, end in bounds:
            kept, keys = make_easy(keys, kept, start, end)

    reduced_flat = [(k, g) for k in keys for g in kept[k]]
    return ensure_od_notes(od_phrases, source, reduced_flat)


def drums_complete(
    mm: MeasureMap,
    od_phrases: List[Tuple[Beats, Beats]],
    sections: List[Tuple[Beats, str]],
    expert_pro: List[Tuple[Beats, Gem]],
) -> Dict[str, List[Tuple[Beats, Gem]]]:
    """Reductions.hs:389-417 cascade, always-regenerate mode (see module
    docstring). Returns {'h':..., 'm':..., 'e':...}; caller has `expert_pro`
    already."""
    hard = drums_reduce("h", mm, od_phrases, sections, expert_pro)
    medium = drums_reduce("m", mm, od_phrases, sections, hard)
    easy = drums_reduce("e", mm, od_phrases, sections, hard)  # from Hard, NOT Medium -- Reductions.hs:408
    return {"h": hard, "m": medium, "e": easy}


AMBIGUITIES = """
1. `Orange` (Drums.hs Gem constructor, the legacy 5-lane-only lane) is not
   modeled. computePro's own catch-all (`Orange -> Orange -- probably
   shouldn't happen`, Drums.hs:344) signals the original tool doesn't
   expect it on a pro-drums track either, and no note in this repo's fixture
   corpus uses it (checked: only 8 scan-chart instrument categories appear
   across a 60-song sample, matching exactly the 8 non-Orange Gem/lane
   combinations, via a raw-pitch histogram check). If a
   future song DOES carry raw pitch 101 (Orange) on a chart marked pro-
   drums, this port will raise rather than silently mis-map it.

2. Beats-within-measure is assumed to be literal quarter-note beats
   (independent of the time signature denominator), NOT HOPCAT/C3toolbox's
   convention (ticks per DENOMINATOR-note, i.e. "beat" = the bar's own
   metrical pulse). This is the standard MIDI-ticks-per-quarter-note
   convention and matches how `U.Beats` is used everywhere ELSE in the
   onyx codebase (note lengths, OD phrase lengths, sustain thresholds are
   all quarter-note-scaled, e.g. `minSustainLengthRB` and the `divn`
   constants 2/1/0.5 in `priority` reading naturally as half/quarter/eighth
   notes). `Sound.MIDI.Util.MeasureMap`'s actual source (external
   `midi-util` package) was not available to read directly in this
   environment to confirm bit-for-bit -- flagging as the highest-risk
   assumption in this port. Covered by unit tests on a synthetic 6/8 bar
   (see tests/test_onyx_reduce.py) demonstrating the assumed semantics are
   at least self-consistent and match the doc-comment reading of
   `isAligned`/`priority`.

3. ensureODNotes' "reinsert the earliest original event" tie-break: when
   the earliest original position at/after an OD phrase's start is a chord
   (multiple coincident gems), Onyx's RTB.viewL yields exactly ONE of them
   in whatever order the underlying RTB's internal representation happens
   to store coincident events (effectively parse/insertion order, not a
   semantically meaningful tie-break). This port picks the first element of
   our own coincident-group list in (pos, then insertion-into-list) order,
   which may not match Onyx's internal order in the rare case a note
   dropped entirely from the reduced tier's OD phrase is part of a chord.
   Does not affect which POSITION gets a note back, only which single lane
   at that position -- low impact on note-COUNT validation, some impact on
   lane-level accuracy in that rare case.

4. RESOLVED (was flagged UNTESTED in the first stage-1 pass, then actually
   checked): computePro's disco resolution assumes `applyStatus`'s
   "instant" list is a "currently-true-keys" membership list, inferred from
   how `instantToms`/`instantDisco` are consumed at the call sites
   (Drums.hs:333,336,343) rather than from applyStatus's own (unread,
   external-module) implementation. ".38 Special - Hold On Loosely
   (Harmonix)" DOES carry a real disco region (`[mix 3 drums4d]` @tick
   109440 to `[mix 3 drums4]` @125520, PART DRUMS) -- missed in the first
   validation pass by not actually inspecting the raw MIDI for markers
   before writing "no disco markers" in the report. Checked directly: in
   that window the raw Expert data is 91 Red + 9 Yellow-cymbal events;
   compute_pro flips them to 91 Yellow-Cymbal("hihat") + 9 Red("snare"),
   and this matches scan-chart's parsed Expert lane-for-lane, exactly, in
   that window (same 91/9 split, same positions). Since scan-chart also
   implements disco-flip (training/clonehero NotesManager.applyDiscoFlip)
   independently, this is real corroboration, not two implementations
   sharing one bug. CLOSED.

5. `U.MeasureMap`'s behavior for TS changes that don't land on a bar line
   is not replicated (same caveat as HOPCAT reduce_port.py's AMBIGUITIES
   #3) -- RB-authoring convention always places TS changes on bar lines, so
   this is not expected to matter in practice.
"""
