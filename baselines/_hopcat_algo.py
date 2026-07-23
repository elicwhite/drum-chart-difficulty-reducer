"""
Independent reimplementation of the HOPCAT drum-reduction algorithm
(C3toolbox.py's `reduce_5lane`), vendored here for reproducible comparison
against Harmonix's official reductions. Attribution: HOPCAT / C3 CON Toolkit
(C3toolbox.py), by the C3 community tools team -- this is a from-scratch
Python port written by reading that source, not a copy of it; see the
per-function C3toolbox.py:NNNN citations below.

This module is MIDI-format-independent (pure algorithm over abstract
Note/TextEvent/MeasureMap objects) and stdlib-only. See ../baselines/hopcat.py
for the adapter that feeds this repo's own ms-based chart format into it.

Standalone Python port of HOPCAT's `reduce_5lane` drum difficulty-reducer.

Source: HOPCAT / C3 CON Toolkit's `C3toolbox.py` (line numbers cited per
function below refer to that file, read in full before writing this port).
There is no independent reference output to validate against, so fidelity
to the reading of `C3toolbox.py` IS the correctness bar. Every non-obvious
rule below cites its source line. Genuine ambiguities are marked
`# AMBIGUITY:` and are also summarized in the module docstring at the
bottom of this file (see AMBIGUITIES list at the end of this file).

SCOPE: only the DRUMS path of reduce_5lane is ported (this eval is drums-
only). The guitar/bass/keys-only branches (reduce_chords, reduce_singlenotes,
pitch-bend detection in remove_notes, "opens to green") are never exercised
for `instrument == 'PART DRUMS'` in the original tool (reduce_5lane forces
`pitchbend = 0` and `opens_to_green = 0` for drums at C3toolbox.py:5031-5033,
5110-5112, 5193-5195) and are NOT ported here.

DATA MODEL (deliberately much simpler than C3toolbox's):
  Note(pos, pitch, vel, dur)   -- one note, absolute MIDI ticks.
  TextEvent(pos, text)         -- one decoded text/marker meta message.
This drops C3toolbox's REAPER-chunk bookkeeping (the 'E'/'e' selected-flag,
the raw "9n"/"8n" channel-code string, base64 pre/post-text wrapping) since
none of it carries semantic meaning for a non-interactive, always-select-all
port; `selected` is always 0 in every call reduce_5lane makes internally
(C3toolbox.py:5037,5114,5197 etc all pass selected=0), so the selection
machinery (selected_notes, selected_range) is dead code for this call path
and is not ported.

TICK-vs-TIME: everything here operates in raw MIDI ticks, exactly like
C3toolbox. mbt()/measures are derived from the MIDI time-signature track
(see build_measures), NOT REAPER's tempo-envelope chunk format (that
REAPER-specific parsing in get_time_signatures() at C3toolbox.py:527 has no
analogue when working from a raw notes.mid -- we get time-signature meta
events directly, which is the ground truth REAPER's own envelope would have
been built from). Tempo/BPM is NOT needed by this port: BPM only feeds
C3toolbox's pitch-bend heuristic (remove_notes bend branch, skipped -- see
above) and fix_sustains' BPM-dependent sustain-length table (fix_sustains is
also not ported, see AMBIGUITIES -- sustain length never affects note
*position*, so it cannot change edit_rate, which matches on (ms, lane)).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Pitch tables (from C3notes.py notesname_array["DRUMS"], lines 98-163;
# only the entries reduce_5lane's drum path actually consults are kept).
# ---------------------------------------------------------------------------

# Tier base pitch: kick=base+0, snare(Red)=base+1, yellow=base+2, blue=base+3,
# green=base+4. Matches C3notes.py:114-134 (Expert 96-100, Hard 84-88,
# Medium 72-76, Easy 60-64 -- each tier is the one below it +12 semitones,
# consistent with reduce_5lane's `note_new[2] -= 12` cascade).
TIER_BASE = {"x": 96, "h": 84, "m": 72, "e": 60}
LANE_OFFSET = {"kick": 0, "snare": 1, "yellow": 2, "blue": 3, "green": 4}
OFFSET_LANE = {v: k for k, v in LANE_OFFSET.items()}

ROLL_MARKER = 126   # "Drum Roll" (C3notes.py:101) -- single-lane roll
SWELL_MARKER = 127  # "Cymbal Swell" (C3notes.py:100) -- two-lane swell

DIVISIONS = {"w": 1, "h": 0.5, "q": 0.25, "e": 0.125, "s": 0.0625, "t": 0.03125, "f": 0.015625}  # C3toolbox.py:148
NEXT_DIVISION = {"w": "h", "h": "q", "q": "e", "e": "s", "s": "t", "t": "f"}  # C3toolbox.py:149
LEVEL_DIVISION = {"x": "s", "h": "e", "m": "q", "e": "h"}  # leveldvisions_array, C3toolbox.py:147, used by simplify_roll

CORRECT_TQN = 480  # correct_tqn, C3toolbox.py:53. Grid math is hardcoded to
# this regardless of the source file's own ticks-per-quarter (C3toolbox only
# warns if a track isn't 480 TQN, C3toolbox.py:706-708, it doesn't rescale).
# Real RB-convention notes.mid are 480 TQN (a Magma requirement) so this is
# not expected to matter in practice; we still read+store the file's actual
# ticks_per_beat for I/O and warn if it differs. See AMBIGUITIES.

ARRAY_DRUMKIT = {"x": "3", "h": "2", "m": "1", "e": "0"}  # C3toolbox.py:138


def tier_of(pitch: int) -> Optional[str]:
    """Which tier a raw MIDI pitch belongs to, or None if it's not one of
    the 5-lane gem pitches (kick/snare/Y/B/G) for any tier -- i.e. it's a
    marker/OD/roll/pro-cymbal/etc pitch that reduce_5lane's drum path never
    touches and always passes through unchanged."""
    for tier, base in TIER_BASE.items():
        if base <= pitch <= base + 4:
            return tier
    return None


def lane_pitch(tier: str, lane: str) -> int:
    return TIER_BASE[tier] + LANE_OFFSET[lane]


def lane_of(pitch: int) -> str:
    tier = tier_of(pitch)
    assert tier is not None
    return OFFSET_LANE[pitch - TIER_BASE[tier]]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Note:
    pos: int
    pitch: int
    vel: int
    dur: int


@dataclass
class TextEvent:
    pos: int
    text: str


@dataclass
class Chord:
    """One or more Notes sharing the same tick -- C3toolbox's "note object"
    (note_objects(), C3toolbox.py:285-313). `pitches` preserves file/
    insertion order (parallel to vels/durs); `sorted_pitches()` is the
    "note[6]" sorted-pitch-set C3toolbox uses for chord-identity comparisons
    (e.g. remove_notes' "same consecutive note" check, C3toolbox.py:1646)."""

    pos: int
    pitches: List[int] = field(default_factory=list)
    vels: List[int] = field(default_factory=list)
    durs: List[int] = field(default_factory=list)

    def sorted_pitches(self) -> Tuple[int, ...]:
        return tuple(sorted(self.pitches))

    def to_notes(self) -> List[Note]:
        return [Note(self.pos, p, v, d) for p, v, d in zip(self.pitches, self.vels, self.durs)]


def note_objects(notes: List[Note]) -> List[Chord]:
    """Group a position-sorted note list into Chords. Mirrors
    C3toolbox.py:285-313 note_objects(); REQUIRES `notes` sorted by pos
    (guaranteed by our callers, matching C3toolbox's assumption that the
    chunk-derived array is already chronological)."""
    chords: List[Chord] = []
    cur: Optional[Chord] = None
    for n in notes:
        if cur is None or n.pos != cur.pos:
            cur = Chord(n.pos)
            chords.append(cur)
        cur.pitches.append(n.pitch)
        cur.vels.append(n.vel)
        cur.durs.append(n.dur)
    return chords


def chords_to_notes(chords: List[Chord]) -> List[Note]:
    """add_objects() equivalent (C3toolbox.py:399-408)."""
    out: List[Note] = []
    for c in chords:
        out.extend(c.to_notes())
    return out


# ---------------------------------------------------------------------------
# Measure map / mbt()
# ---------------------------------------------------------------------------


@dataclass
class Measure:
    number: int
    start_tick: int
    denominator: int
    numerator: int
    beat_ticks: float  # "ticks per beat-unit" == measures_array[x][4], C3toolbox.py:612


class MeasureMap:
    """Reimplementation of C3toolbox's measures_array + mbt() (C3toolbox.py:
    315-322, 527-646), built from the MIDI time-signature track instead of
    REAPER's tempo-envelope chunk (see module docstring). BPM is not tracked
    -- see module docstring for why the port never needs it.

    mbt()'s C3toolbox loop (`for x in range(len(measures_array)): if
    measures_array[x][1] <= position: m = x+1 ...`) never breaks, so it
    always resolves to the LAST measure whose start_tick <= position --
    including for `position` beyond the last generated measure, where it
    just keeps re-using that last measure's grid indefinitely. A bisect
    over sorted start_ticks reproduces this exactly, with no need to
    special-case "ran off the end of the table" (see reduce_port docstring
    for why we don't bother replicating C3toolbox's '[end]'-marker-driven
    trailing-measure generation).
    """

    def __init__(self, measures: List[Measure]):
        assert measures, "need at least one measure"
        self.measures = measures
        self._starts = [m.start_tick for m in measures]

    def mbt(self, position: int) -> Tuple[int, int, int, int]:
        """Returns (measure, beat, tick_in_beat, ticks_since_measure_start),
        matching C3toolbox.py:315-322 mbt()."""
        idx = bisect.bisect_right(self._starts, position) - 1
        if idx < 0:
            idx = 0
        meas = self.measures[idx]
        rel = position - meas.start_tick
        b = int(rel // meas.beat_ticks) + 1
        t = int(rel - (b - 1) * meas.beat_ticks)
        return (meas.number, b, t, rel)

    def measure_of(self, position: int) -> int:
        return self.mbt(position)[0]


def build_measures(ts_events: List[Tuple[int, int, int]], ticks_per_beat: int, end_tick: int) -> MeasureMap:
    """Build a MeasureMap from (tick, numerator, denominator) time-signature
    events (as read straight off the MIDI tempo-map track) plus the file's
    ticks_per_beat (TPQN).

    Semantic equivalent of get_time_signatures() (C3toolbox.py:527-646):
    for each TS-change segment, "ticks per beat-unit" (measures_array[x][4])
    is `ticks_per_beat * 4 / denominator` (C3toolbox.py:601, `divider =
    instrument_ticks/(denominator*0.25)`), and a measure is `numerator` of
    those beat-units long. We walk forward emitting one Measure per bar
    within each segment. If real TS-change ticks aren't exactly bar-aligned
    this drifts from C3toolbox's round()-based measure count (see
    AMBIGUITIES); RB-convention charts always place TS changes on bar lines
    so this is not expected to matter.
    """
    if not ts_events or ts_events[0][0] != 0:
        ts_events = [(0, 4, 4)] + list(ts_events)
    ts_events = sorted(ts_events, key=lambda e: e[0])

    measures: List[Measure] = []
    m = 0
    for i, (start_tick, num, den) in enumerate(ts_events):
        beat_ticks = ticks_per_beat * 4.0 / den
        bar_ticks = beat_ticks * num
        segment_end = ts_events[i + 1][0] if i + 1 < len(ts_events) else max(end_tick, start_tick) + bar_ticks
        pos = float(start_tick)
        while pos < segment_end:
            m += 1
            measures.append(Measure(m, int(round(pos)), den, num, beat_ticks))
            pos += bar_ticks
    return MeasureMap(measures)


# ---------------------------------------------------------------------------
# Default config (drums), transcribed from reduce_5lane.py UI defaults
# (the reduce_5lane.py C3 UI wrapper, not C3toolbox.py itself) and traced
# through to what actually reaches reduce_5lane() for instrument=='PART
# DRUMS'. See reduce_port DEFAULT_CONFIG_NOTES at bottom for the full
# per-field derivation.
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "levels": {"h": True, "m": True, "e": True},  # lvlHardvar/lvlMediumvar/lvlEasyvar all .select()'d
    "grid": {"h": "e", "m": "q", "e": "h"},  # grid_var_h/m/e defaults: 1/8, 1/4, 1/2 (reduce_5lane.py:198,271,345)
    "same": {"h": 0, "m": 0, "e": 0},  # "Consecutive notes" checkbox: .select() only runs for
    # non-drum instrument_id (reduce_5lane.py:280-281,355-356); drums (id 0) never selects it,
    # and Hard's box is never auto-selected for ANY instrument (no .select() call at all, reduce_5lane.py:203-206)
    "sparse": {"h": 1, "m": 1, "e": 1},  # "Sparse notes": .select()'d unconditionally for h/m/e
    "singlesnare": {"h": "n", "m": "n", "e": "n"},  # snareh_var/snarem_var/snaree_var default "Don't simplify snare"
    "tolerance": 20,  # tolerance Entry default "20" (reduce_5lane.py:412)
    "unflip": "h",  # level_var default "Hard" -> array_levels["Hard"][0] == 'h' (reduce_5lane.py:439-440)
    "remove_kick_medium": True,  # bendChkvar_m: for drums (instrument_id==0), .select()'d by default
    # (reduce_5lane.py:293-298 -- note the checkbox LABEL for drums is repurposed to read
    # "Remove kicks paired with percussion", C3toolbox reduce_5lane.py:290-291).
    #
    # NOTE: there is deliberately no separate "remove_kick_easy" key. Source
    # gates BOTH the Medium AND Easy remove_kick calls on this SAME
    # bendChkvar_m value -- C3toolbox.py:5104 (`if medium[3] == 1 ... 'm'`)
    # AND C3toolbox.py:5187 (`if medium[3] == 1 ... 'e'`), never checking
    # easy[3]/bendChkvar_e at all. reduce_5lane_drums() below reads
    # `config["remove_kick_medium"]` again for the Easy gate for exactly
    # this reason -- do NOT add an independent "remove_kick_easy" config
    # key, a config sweep that decoupled them would no longer be measuring
    # real HOPCAT.
}


# ---------------------------------------------------------------------------
# remove_notes -- the core grid-quantizing reducer. C3toolbox.py:1469-1667.
# Drums always call it with bend=0, opens_to_green=0 (forced at the
# reduce_5lane call sites), so those branches are not ported.
# ---------------------------------------------------------------------------


def remove_notes(
    notes: List[Note],
    events: List[TextEvent],
    mm: MeasureMap,
    grid: str,
    level: str,
    tolerance: int,
    same: bool,
    sparse: bool,
) -> List[Note]:
    """Quantize/thin the `level` tier's gem notes to `grid`, keeping notes
    within `tolerance` ticks of a grid line, plus roll-marker-covered notes,
    plus (if sparse) at least one note every `division` ticks, plus (if
    same) grid-adjacent runs of the identical chord. C3toolbox.py:1469-1667.
    """
    division = int((CORRECT_TQN * 4) * DIVISIONS[grid])  # C3toolbox.py:1490 (math.floor of an already-int product)
    leveltext = level  # tier tag, e.g. 'h'

    passthrough: List[Note] = []
    valid: List[Note] = []
    for n in notes:
        if tier_of(n.pitch) == leveltext:
            valid.append(n)
        else:
            passthrough.append(n)
    valid.sort(key=lambda n: n.pos)

    # Roll-marker spans (126/127) exempt covered notes from quantization
    # entirely, C3toolbox.py:1532-1553. Roll markers live in `passthrough`
    # (they're never tier-tagged).
    roll_spans = [(n.pos, n.pos + n.dur) for n in passthrough if n.pitch in (ROLL_MARKER, SWELL_MARKER)]
    roll_note_ticks = set()
    for n in valid:
        for start, end in roll_spans:
            if start <= n.pos <= end:
                roll_note_ticks.add(n.pos)
                break

    chords = note_objects(valid)

    kept: List[Chord] = []
    sparse_position = 0
    for i, c in enumerate(chords):
        rel = mm.mbt(c.pos)[3]  # ticks since start of c's own measure, C3toolbox.py:1623
        grid_check = int(rel // division)
        dist_to_next_line = division - (rel - grid_check * division)
        on_grid = (rel - grid_check * division) <= tolerance or dist_to_next_line <= tolerance
        if c.pos in roll_note_ticks:
            kept.append(c)
            sparse_position = c.pos
        elif on_grid:
            kept.append(c)
            sparse_position = c.pos
        elif sparse and (c.pos - sparse_position) >= division:
            kept.append(c)
            sparse_position = c.pos
        elif same:
            # C3toolbox.py:1642-1648: check the NEXT-finer grid, and only
            # keep if this chord's pitch-set equals the PRECEDING chord in
            # the full (pre-filter) `chords` list -- not the preceding KEPT
            # chord.
            newdivision = int((CORRECT_TQN * 4) * DIVISIONS[NEXT_DIVISION[grid]])
            grid_check2 = int(rel // newdivision)
            on_finer_grid = (rel - grid_check2 * newdivision) <= tolerance or (
                division - (rel - grid_check2 * newdivision)
            ) <= tolerance
            if on_finer_grid and i > 0 and c.sorted_pitches() == chords[i - 1].sorted_pitches():
                kept.append(c)
                sparse_position = c.pos
        # else: dropped

    # Second pass (C3toolbox.py:1649-1664): when same or sparse is on, a
    # chord kept via the tolerance branch above but still off its OWN grid
    # can get dropped again if the next kept chord is close enough to make
    # it redundant. IMPORTANT: unlike the first pass, this grid_check is on
    # the ABSOLUTE tick, not the measure-relative one -- C3toolbox.py:1657
    # sets `position = note[1][0]` (the raw tick), not `mbt(...)[3]` as the
    # first pass does at C3toolbox.py:1623. This only diverges from a
    # measure-relative check when a measure's start tick isn't itself a
    # multiple of `division` (e.g. a 3/4 bar against Easy's half-note
    # grid), but sparse defaults ON, so this pass always runs -- transcribed
    # as the absolute-tick check per source, not "fixed" to be consistent
    # with the first pass.
    if same or sparse:
        kept2: List[Chord] = []
        for i, c in enumerate(kept):
            grid_check = int(c.pos // division)
            off_grid = (c.pos - grid_check * division) > tolerance
            if off_grid and i < len(kept) - 1 and c.pos not in roll_note_ticks:
                nxt = kept[i + 1]
                far_enough = (nxt.pos - c.pos) >= division
                same_and_half = same and (nxt.pos - c.pos) >= division * 0.5 and nxt.sorted_pitches() == c.sorted_pitches()
                if far_enough or same_and_half:
                    kept2.append(c)
                # else: dropped in the second pass
            else:
                kept2.append(c)
        kept = kept2

    return passthrough + chords_to_notes(kept)


# ---------------------------------------------------------------------------
# remove_kick -- C3toolbox.py:1932-2007.
# what: 'a' any note, 's' snare, 't' tom(pro-marker 110-112), 'p' snare-or-tom
# ---------------------------------------------------------------------------


def remove_kick(notes: List[Note], level: str, what: str) -> List[Note]:
    leveltext = level
    kick = lane_pitch(level, "kick")
    snare = lane_pitch(level, "snare")

    passthrough: List[Note] = []
    valid: List[Note] = []
    for n in notes:
        # C3toolbox.py:1983: this tier's own notes OR any pro-cymbal marker
        # (110-112) that happens to land on the same ticks (it can, since
        # pro markers are never shifted by the tier cascade -- see module
        # docstring).
        if tier_of(n.pitch) == leveltext or 110 <= n.pitch <= 112:
            valid.append(n)
        else:
            passthrough.append(n)
    valid.sort(key=lambda n: n.pos)
    chords = note_objects(valid)

    out_chords: List[Chord] = []
    for c in chords:
        pitches = c.sorted_pitches()
        if kick in pitches and len(pitches) > 1:
            hit = (
                what == "a"
                or (what == "s" and snare in pitches)
                or (what == "t" and any(p in pitches for p in (110, 111, 112)))
                or (what == "p" and (snare in pitches or any(p in pitches for p in (110, 111, 112))))
            )
            if hit:
                sub = Chord(c.pos)
                for p, v, d in zip(c.pitches, c.vels, c.durs):
                    if p != kick:
                        sub.pitches.append(p)
                        sub.vels.append(v)
                        sub.durs.append(d)
                out_chords.append(sub)
            else:
                out_chords.append(c)
        else:
            out_chords.append(c)

    return passthrough + chords_to_notes(out_chords)


# ---------------------------------------------------------------------------
# single_snare -- C3toolbox.py:1854-1930. Mirror-image of remove_kick
# (drops the SNARE from a snare+X chord instead of the kick). Not called by
# default (singlesnare config is 'n'/'n'/'n' for all tiers) but ported for
# completeness / future config sweeps.
# ---------------------------------------------------------------------------


def single_snare(notes: List[Note], level: str, what: str) -> List[Note]:
    leveltext = level
    kick = lane_pitch(level, "kick")
    snare = lane_pitch(level, "snare")
    yellow = lane_pitch(level, "yellow")
    blue = lane_pitch(level, "blue")
    green = lane_pitch(level, "green")

    passthrough: List[Note] = []
    valid: List[Note] = []
    for n in notes:
        if tier_of(n.pitch) == leveltext or 110 <= n.pitch <= 112:
            valid.append(n)
        else:
            passthrough.append(n)
    valid.sort(key=lambda n: n.pos)
    chords = note_objects(valid)

    out_chords: List[Chord] = []
    for c in chords:
        pitches = c.sorted_pitches()
        if snare in pitches and len(pitches) > 1:
            hit = (
                what == "a"
                or (what == "k" and kick in pitches)
                or (what == "t" and any(p in pitches for p in (110, 111, 112)))
                or (what == "c" and any(p in pitches for p in (yellow, blue, green)))
            )
            if hit:
                sub = Chord(c.pos)
                for p, v, d in zip(c.pitches, c.vels, c.durs):
                    if p == snare or 110 <= p <= 112:
                        sub.pitches.append(p)
                        sub.vels.append(v)
                        sub.durs.append(d)
                out_chords.append(sub)
            else:
                out_chords.append(c)
        else:
            out_chords.append(c)

    return passthrough + chords_to_notes(out_chords)


# ---------------------------------------------------------------------------
# unflip_discobeat -- C3toolbox.py:2268-2420.
# ---------------------------------------------------------------------------


def unflip_discobeat(
    notes: List[Note], events: List[TextEvent], mm: MeasureMap, level: str, how: int
) -> Tuple[List[Note], List[TextEvent]]:
    """Un-swap the hi-hat/snare disco-flip within [mix 3 drums*(d)] marked
    sections (always keyed off EXPERT's mix-3 markers, C3toolbox.py:2273 --
    "the function works on Expert level notes anyway all levels now point
    to the same marker"), applied to `level`'s own note pitches, and strips
    the disco-flip flag from `level`'s own mix marker
    (`[mix {ARRAY_DRUMKIT[level]} drums{0-4}d]` -> same without the `d`).
    """
    division = int((CORRECT_TQN * 4) * DIVISIONS["e"])
    notey = lane_pitch(level, "yellow")
    snare = lane_pitch(level, "snare")

    # Disco-flip windows, from Expert's mix-3 markers (not `level`'s own).
    starts = {f"[mix 3 drums{d}d]" for d in range(5)}
    ends = {f"[mix 3 drums{d}]" for d in range(5)}
    windows: List[Tuple[int, int]] = []
    open_start: Optional[int] = None
    for e in sorted(events, key=lambda e: e.pos):
        if e.text in starts:
            if open_start is not None:
                raise ValueError("two consecutive disco-flip start markers -- malformed chart")
            open_start = e.pos
        elif e.text in ends:
            if open_start is not None:
                windows.append((open_start, e.pos))
                open_start = None
    if open_start is not None:
        # C3toolbox.py:2349-2351: an unterminated flip runs to the last note.
        last = max((n.pos for n in notes), default=open_start)
        windows.append((open_start, last))

    notes = list(notes)  # will mutate pitches in place, like C3toolbox does
    by_pos: Dict[int, List[Note]] = {}
    ordered = sorted(range(len(notes)), key=lambda i: notes[i].pos)

    to_remove: set = set()
    to_add: List[Note] = []
    for start, end in windows:
        in_window = [i for i in ordered if start <= notes[i].pos <= end]
        yellow_count = sum(1 for i in in_window if notes[i].pitch == notey)
        snare_count = sum(1 for i in in_window if notes[i].pitch == snare)
        # AMBIGUITY (see AMBIGUITIES list): C3toolbox.py:2368-2372 gates an
        # "already looks unflipped" section behind an interactive RPR_MB
        # prompt using a bare `mute` name that isn't a parameter of this
        # function (looks like a latent bug in the original tool -- it can
        # only ever pick up a stray global from a previous call). We treat
        # this case as "leave the window alone" (decline the prompt), the
        # conservative headless reading.
        if yellow_count > snare_count:
            continue  # already-unflipped window: skip it (see AMBIGUITY above)

        for idx, i in enumerate(ordered):
            n = notes[i]
            if not (start <= n.pos <= end):
                continue
            if n.pitch == notey:
                n.pitch = snare
                rel = mm.mbt(n.pos)[3]
                grid_check = int(rel // division)
                on_grid = (rel - grid_check * division) <= how
                if on_grid and 0 < ordered.index(i) < len(ordered) - 1:
                    # C3toolbox.py:2385-2389: `array_notesevents[0][j-1][2]
                    # == snare or array_notesevents[0][j+1][2]` -- the
                    # second half of that `or` is just a pitch value, which
                    # is always truthy, so this condition is effectively
                    # ALWAYS true whenever the note isn't the very first/
                    # last note in the whole file. Transcribed faithfully
                    # (see AMBIGUITY list) rather than "fixed".
                    to_add.append(Note(n.pos, notey, n.vel, n.dur))
            elif n.pitch == snare:
                n.pitch = notey
                rel = mm.mbt(n.pos)[3]
                grid_check = int(rel // division)
                off_grid = (rel - grid_check * division) > how
                if off_grid:
                    to_remove.add(n.pos)

    kept_notes = [n for n in notes if not (n.pitch == notey and n.pos in to_remove)]
    kept_notes.extend(to_add)

    new_events = []
    for e in events:
        text = e.text
        for d in range(5):
            flagged = f"[mix {ARRAY_DRUMKIT[level]} drums{d}d]"
            plain = f"[mix {ARRAY_DRUMKIT[level]} drums{d}]"
            if text == flagged:
                text = plain
                break
        new_events.append(TextEvent(e.pos, text))

    return kept_notes, new_events


# ---------------------------------------------------------------------------
# simplify_roll -- C3toolbox.py:2423-2574. Replaces a "Drum Roll" (126) or
# "Cymbal Swell" (127) marker span with a mechanically-generated 1/16th roll
# / alternating-16th swell on whichever lane(s) were most common in the span
# originally, at LEVEL_DIVISION[level] spacing.
# ---------------------------------------------------------------------------


def simplify_roll(notes: List[Note], events: List[TextEvent], level: str) -> List[Note]:
    leveltext = level
    sixteenth = int(CORRECT_TQN * 0.125)

    def most_common_pitches(start: int, end: int, n: int) -> List[int]:
        counts: Dict[int, int] = {}
        for note in notes:
            if start <= note.pos <= end and tier_of(note.pitch) == leveltext:
                counts[note.pitch] = counts.get(note.pitch, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        return [p for p, _ in ranked[:n]]

    to_remove_keys = set()  # (pitch, pos)
    to_add: List[Note] = []

    one_lane_spans = [(e.pos, e.pos + 0) for e in events if e.text == "x"]  # placeholder, replaced below
    # Roll markers are TextEvents in this port's model (they carry no
    # duration on their own -- C3toolbox reads them as NOTES with a
    # duration, C3toolbox.py:1540-1543/2466-2469, since 126/127 are note
    # pitches with note-on/off pairs, not meta text). We therefore expect
    # roll markers to arrive via the `notes` list (as passthrough Notes,
    # pitch 126/127), not `events`. Re-derive spans from `notes`:
    roll_spans = [(n.pos, n.pos + n.dur) for n in notes if n.pitch == ROLL_MARKER]
    swell_spans = [(n.pos, n.pos + n.dur) for n in notes if n.pitch == SWELL_MARKER]

    for start, end in roll_spans:
        top = most_common_pitches(start, end, 1)
        if not top:
            continue
        pitch = top[0]
        template = next((n for n in notes if start <= n.pos <= end and n.pitch == pitch), None)
        if template is None:
            continue
        for n in notes:
            if n.pitch == pitch and start <= n.pos <= end:
                to_remove_keys.add((n.pitch, n.pos))
        sequence = int((CORRECT_TQN * 4) * DIVISIONS[LEVEL_DIVISION[level]])
        loc = start
        while loc < end + 20:
            to_add.append(Note(int(loc), pitch, template.vel, sixteenth))
            loc += sequence

    for start, end in swell_spans:
        top = most_common_pitches(start, end, 2)
        if len(top) < 2:
            continue
        p1, p2 = top[0], top[1]
        template = next((n for n in notes if start <= n.pos <= end and n.pitch in (p1, p2)), None)
        if template is None:
            continue
        for n in notes:
            if n.pitch in (p1, p2) and start <= n.pos <= end:
                to_remove_keys.add((n.pitch, n.pos))
        sequence = (CORRECT_TQN * 4) * DIVISIONS["q" if level == "h" else "h"]
        quarter = int(CORRECT_TQN * 0.25)
        loc = start
        while loc < end + 20:
            to_add.append(Note(int(loc), p1, template.vel, quarter))
            loc += sequence
        loc = start + sequence * 0.5
        while loc < end + 20:
            to_add.append(Note(int(loc), p2, template.vel, quarter))
            loc += sequence

    out = [n for n in notes if (n.pitch, n.pos) not in to_remove_keys]
    out.extend(to_add)
    return out


# ---------------------------------------------------------------------------
# reduce_5lane orchestrator (drums-only path), C3toolbox.py:4939-5214.
# ---------------------------------------------------------------------------


def _cascade_copy(notes: List[Note], events: List[TextEvent], src_tier: str, dst_tier: str) -> Tuple[List[Note], List[TextEvent]]:
    """"Clean {dst} and copy from {src}" step at the top of each tier block
    (e.g. C3toolbox.py:4991-5023 for Hard). Deletes any existing dst-tier
    notes, then re-derives dst-tier notes as src-tier notes shifted -12
    semitones, and renumbers src's disco-flip mix markers down to dst's
    marker index (duplicating them; C3toolbox.py:5009-5023)."""
    src_offset = TIER_BASE[src_tier]
    dst_offset = TIER_BASE[dst_tier]
    new_notes = [n for n in notes if tier_of(n.pitch) != dst_tier]
    for n in list(new_notes):
        if tier_of(n.pitch) == src_tier:
            new_notes.append(Note(n.pos, n.pitch - (src_offset - dst_offset), n.vel, n.dur))

    src_key, dst_key = ARRAY_DRUMKIT[src_tier], ARRAY_DRUMKIT[dst_tier]
    new_events = [e for e in events if f"[mix {dst_key}" not in e.text]
    for e in list(new_events):
        if f"[mix {src_key}" in e.text:
            new_events.append(TextEvent(e.pos, e.text.replace(f"mix {src_key}", f"mix {dst_key}")))
    return new_notes, new_events


def reduce_5lane_drums(
    notes: List[Note],
    events: List[TextEvent],
    mm: MeasureMap,
    config: Dict = DEFAULT_CONFIG,
) -> Tuple[List[Note], List[TextEvent]]:
    """Drums-only port of reduce_5lane (C3toolbox.py:4939-5214), applying
    `config` (DEFAULT_CONFIG unless overridden) to produce fresh Hard,
    Medium and Easy tiers cascaded from Expert -> Hard -> Medium -> Easy.
    `notes`/`events` should contain the FULL song (Expert + any existing
    H/M/E, which get discarded and regenerated, + all markers/OD/mix
    events) -- exactly what process_instrument()/create_notes_array() feed
    reduce_5lane() in the original tool.

    fix_sustains (C3toolbox.py:1757-1854, called after every tier) is NOT
    ported: it only shortens/lengthens note DURATION based on BPM/tier, and
    editrate.py's edit_rate matches on (ms, lane) only -- duration cannot
    change the score. See AMBIGUITIES.
    """
    tol = config["tolerance"]
    unflip_level = config["unflip"]

    # ---- Hard, from Expert ----
    if config["levels"]["h"]:
        notes, events = _cascade_copy(notes, events, "x", "h")
        if unflip_level == "h":
            notes, events = unflip_discobeat(notes, events, mm, "h", 20)  # C3toolbox.py:2268 how=20 hardcoded at call site (5026)
        notes = remove_notes(notes, events, mm, config["grid"]["h"], "h", tol, config["same"]["h"], config["sparse"]["h"])
        notes = simplify_roll(notes, events, "h")
        if config["singlesnare"]["h"] != "n":
            notes = single_snare(notes, "h", config["singlesnare"]["h"])
        # fix_sustains('h') intentionally not ported -- see docstring.

    # ---- Medium, from Hard ----
    if config["levels"]["m"]:
        notes, events = _cascade_copy(notes, events, "h", "m")
        if unflip_level == "m":
            notes, events = unflip_discobeat(notes, events, mm, "m", 20)
        if config["remove_kick_medium"]:
            notes = remove_kick(notes, "m", "p")  # C3toolbox.py:5105
        notes = remove_notes(notes, events, mm, config["grid"]["m"], "m", tol, config["same"]["m"], config["sparse"]["m"])
        notes = simplify_roll(notes, events, "m")
        if config["singlesnare"]["m"] != "n":
            notes = single_snare(notes, "m", config["singlesnare"]["m"])

    # ---- Easy, from Medium ----
    if config["levels"]["e"]:
        notes, events = _cascade_copy(notes, events, "m", "e")
        if unflip_level == "e":
            notes, events = unflip_discobeat(notes, events, mm, "e", 20)
        if config["remove_kick_medium"]:  # C3toolbox.py:5187-5188: reuses medium[3], NOT a separate
            # easy-tier flag -- see the DEFAULT_CONFIG note on "remove_kick_medium" above; keep this
            # coupled to remove_kick_medium in any future config sweep.
            notes = remove_kick(notes, "e", "a")  # C3toolbox.py:5188
        notes = remove_notes(notes, events, mm, config["grid"]["e"], "e", tol, config["same"]["e"], config["sparse"]["e"])
        notes = simplify_roll(notes, events, "e")
        if config["singlesnare"]["e"] != "n":
            notes = single_snare(notes, "e", config["singlesnare"]["e"])

    return notes, events


AMBIGUITIES = """
1. unflip_discobeat's "already unflipped, proceed anyway?" prompt
   (C3toolbox.py:2368-2372) reads a bare `mute` name that is NOT a
   parameter of unflip_discobeat and is never assigned inside it -- looks
   like a latent bug in the original tool (it can only pick up a stray
   value if some earlier-executed function in the same run happened to
   leave a module-level `mute` global lying around, which doesn't happen
   via any call path reduce_5lane exercises). We treat this branch as
   "decline the prompt" (skip flipping that window), the conservative
   headless reading. Only matters for songs where a disco-flip section
   already looks unflipped by note-count (rare).

2. unflip_discobeat's on-grid-hihat companion-note condition
   (C3toolbox.py:2385-2389) is `array_notesevents[0][j-1][2] == snare or
   array_notesevents[0][j+1][2]` -- the second operand of that `or` is a
   raw pitch integer, which is always truthy. Ported faithfully as "always
   true unless j is the very first/last note in the whole file" rather
   than "fixed" to the evident intent (checking whether the FOLLOWING note
   is also a snare). If this is actually a typo for
   `array_notesevents[0][j+1][2] == snare`, our port over-generates
   companion hi-hat notes in unflip sections. Flagging for review before
   trusting disco-flip-heavy songs' numbers.

3. get_time_signatures' REAPER-envelope math is not replicated; measures
   are rebuilt directly from the MIDI time-signature track (see
   MeasureMap/build_measures docstring). For TS changes that land exactly
   on a bar line (the RB-authoring convention) this is provably equivalent;
   for a chart with an off-bar TS change it could diverge from whatever
   C3toolbox's round()-based measure count would have produced. Covered by
   the mbt() unit tests on a synthetic mid-song TS change, but NOT
   validated against a real off-bar case (none found/expected in-corpus).

4. fix_sustains (C3toolbox.py:1757-1854, called after every tier) is not
   ported. It only adjusts note DURATION (shortening overlong sustains
   based on BPM/tier), never note position or lane, so it cannot affect
   editrate.py's (ms, lane) matching. Explicit scope cut, not an oversight.

5. correct_tqn is hardcoded to 480 in C3toolbox's own grid math (not scaled
   by the source file's actual ticks-per-quarter -- C3toolbox.py:1490 etc
   always use `correct_tqn`, not the `ticks` value returned by
   process_instrument). This port does the same (CORRECT_TQN=480). Real RB
   notes.mid are conventionally 480 TQN; if a source file isn't, our port
   (like the original tool) will quantize on the wrong tick grid. midi_io.py
   should warn (not silently rescale) if a file's ticks_per_beat != 480.
"""
