"""
Direct unit tests for decode.py's DETERMINISM_CONTRACT.md §2 tie-breaks --
constructed synthetically so the tie is guaranteed, rather than hoping a
real or synthetic song's learned model output happens to produce one
(exact ties in learned probabilities are vanishingly rare and not
reliably constructible from chart content, per the parity fixture's
edge-case notes)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drum_reducer import decode  # noqa: E402


def test_modal_reduction_tie_break_lexicographic():
    """§2.4: when two candidate reduced grooves tie on instance count, the
    winner is the one whose sorted (tick, lane_index) list is
    lexicographically smallest -- independent of which one appears more
    often in iteration/insertion order."""
    groove_a = frozenset({(0, "snare")})            # lane_index=1 -> (0,1)
    groove_b = frozenset({(0, "kick")})              # lane_index=0 -> (0,0), smaller
    # 2 instances of each -> exact tie by count. Insertion order deliberately
    # puts groove_a first, so a naive Counter.most_common tie-break (which
    # returns the first-inserted key on a count tie) would pick groove_a.
    reductions = [groove_a, groove_b, groove_a, groove_b]
    winner = decode._modal_reduction(reductions)
    assert winner == groove_b, "tie-break should pick the lexicographically smaller (tick,lane_index) groove"


def test_modal_reduction_tie_break_multi_note_groove():
    """A tie between two multi-note grooves -- lexicographic comparison
    walks the sorted (tick,lane_index) list, not just the first element."""
    # groove_a: (0,kick)=(0,0), (0,snare)=(0,1) -> sorted [(0,0),(0,1)]
    groove_a = frozenset({(0, "kick"), (0, "snare")})
    # groove_b: (0,kick)=(0,0), (100,kick)=(100,0) -> sorted [(0,0),(100,0)]
    groove_b = frozenset({(0, "kick"), (100, "kick")})
    reductions = [groove_b, groove_a]  # groove_b inserted first
    winner = decode._modal_reduction(reductions)
    # [(0,0),(0,1)] < [(0,0),(100,0)] since second element (0,1) < (100,0)
    assert winner == groove_a


def test_modal_reduction_no_tie_returns_majority():
    groove_a = frozenset({(0, "kick")})
    groove_b = frozenset({(0, "snare")})
    reductions = [groove_a, groove_a, groove_a, groove_b]
    assert decode._modal_reduction(reductions) == groove_a


def test_canonicalize_uses_tie_broken_modal_groove():
    """End-to-end: canonicalize() on a synthetic 4-measure cluster where
    two instances reduce to groove_a and two to groove_b (an exact tie)
    forces every instance to the lexicographically-smaller groove."""
    tempos = [{"ms": 0, "bpm": 120.0}]
    time_sigs = [{"ms": 0, "numerator": 4, "denominator": 4}]
    ms_to_measure, measure_to_ms = decode.build_measure_clock(tempos, time_sigs)

    # 4 measures (0,1,2,3), each 2000ms, with an IDENTICAL Expert groove
    # (kick+snare at the downbeat) -> one cluster of size 4.
    expert = []
    for mi in range(4):
        base = mi * 2000.0
        expert.append(decode.Note(base, "kick"))
        expert.append(decode.Note(base, "snare"))
    clusters, n_nonempty = decode.expert_groove_clusters(expert, ms_to_measure)
    assert n_nonempty == 4
    assert len(clusters) == 1
    (idxs,) = clusters.values()
    assert idxs == [0, 1, 2, 3]

    # Candidate: measures 0,2 keep only snare; measures 1,3 keep only kick
    # -> exact 2-2 tie between {(0,snare)} and {(0,kick)}.
    cand = [
        decode.Note(0.0, "snare"),
        decode.Note(2000.0, "kick"),
        decode.Note(4000.0, "snare"),
        decode.Note(6000.0, "kick"),
    ]
    rbm = decode.reduced_groove_by_measure(cand, ms_to_measure)
    canon = decode.canonicalize(cand, clusters, rbm, ms_to_measure, measure_to_ms)

    # kick (lane_index 0) sorts before snare (lane_index 1) -> the modal
    # tie-break picks {(0,kick)}, so every instance should now be kick-only.
    assert len(canon) == 4
    assert all(n.lane == "kick" for n in canon)


def test_family_nms_tie_break_order():
    """§2.1: equal keep_score across repeated-groove members ties broken by
    (ms, lane_index) -- verify the earlier-ms note wins when scores tie."""
    rows = [
        {"ms": 100.0, "lane": "crash", "family": "cymbal"},
        {"ms": 150.0, "lane": "hihat", "family": "cymbal"},
    ]
    survive = [True, True]
    keep_score = [0.9, 0.9]  # exact tie
    out = decode.family_nms(rows, survive, keep_score, gap_ms=100)
    # earlier ms (100.0, crash) should be kept; the later one within gap_ms dropped
    assert out == [True, False]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
