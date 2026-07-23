"""
Produces the README quality-comparison table: edit_rate (primary) and
secondary quality metrics for four reduction systems -- ours-model,
ours-scorecard, HOPCAT, Onyx -- on the 99 rb4_test songs, each vs the
Harmonix/RB4 human ground truth.

The 99 songs are exactly the non-edge fixture charts in
data/fixtures/charts/ (data/fixtures/parity_fixture.json's `songs` keys,
minus the 3 edge_case_songs). Ours-model output is
data/fixtures/parity_fixture.json; ours-scorecard is
data/fixtures/scorecard_parity_fixture.json -- both already-computed
reduce() output, not re-run here (this script is a pure eval/aggregation
step, not a reduce-again step).

HOPCAT/Onyx provenance: run LIVE, in-process, via the vendored
baselines/hopcat.py and baselines/onyx.py adapters (independent
reimplementations of the two tools, feeding this repo's own ms-based chart
dict directly -- see those modules for the conversion approach and
documented divergences from the real MIDI-based tools). This script is
therefore fully standalone: it does not read any file outside
drum-reducer-reference/, and does not depend on the research tree.

We do NOT claim either baseline is "worse" or "better" in absolute terms --
edit_rate measures closeness to Harmonix's own official reductions, nothing
more; HOPCAT and Onyx were built for different goals (fast, general-purpose,
user-tunable reduction) and this comparison is one specific yardstick.

Because Onyx is now run live (not read from a frozen TSV missing its note
list), its secondary metrics -- previously blank -- are computed like every
other system's.

Emits data/reference_scores.tsv (one row per system x tier, plus pooled
rows) and prints a human-readable summary to stdout.

Run (root venv -- editrate/decode/intrinsic_difficulty are stdlib-only, no
extra deps beyond numpy for the model backend):
  .venv/bin/python drum-reducer-reference/tools/eval_quality.py
"""

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.dirname(HERE)

sys.path.insert(0, REF)
sys.path.insert(0, os.path.join(REF, "python"))

from baselines import reduce_hopcat, reduce_onyx  # noqa: E402
from drum_reducer import editrate as ER  # noqa: E402
from drum_reducer import decode as CM  # noqa: E402  (build_measure_clock / groove-cluster / consistency machinery)
from drum_reducer import intrinsic_difficulty as ID  # noqa: E402

TIERS = ["hard", "medium", "easy"]
BACKBONE_LANES = {"kick", "snare"}

FIXTURES_DIR = os.path.join(REF, "data", "fixtures")
CHARTS_DIR = os.path.join(FIXTURES_DIR, "charts")
EDGE_CASES = {"edge-empty-groove-measures", "edge-midsong-ts-change", "edge-no-backbone"}

OUT_TSV = os.path.join(REF, "data", "reference_scores.tsv")

# Sanity targets. ours-model/ours-scorecard are exact (deterministic,
# already-frozen fixtures). HOPCAT/Onyx are the previously-measured pooled
# edit_rate from an earlier real-MIDI eval of the original tools -- our
# live vendored-and-adapted run here is checked against those as a
# reconciliation bar, not asserted exactly (see module docstring: this
# repo's chart schema lacks OD-phrase/marker data the real-MIDI eval had,
# so some drift is expected -- reported, not hidden).
OURS_MODEL_TARGET = 0.1703
OURS_SCORECARD_TARGET = 0.2234
HOPCAT_TARGET = 0.420
ONYX_TARGET = 0.395
RECONCILE_TOL = 0.03  # absolute pooled edit_rate tolerance for the reconciliation check


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_song_ids():
    fixture = json.load(open(os.path.join(FIXTURES_DIR, "parity_fixture.json")))
    return sorted(sid for sid in fixture["songs"] if sid not in EDGE_CASES)


def load_chart(sid):
    return json.load(open(os.path.join(CHARTS_DIR, f"{sid}.json")))


def gt_notes(chart, tier):
    return ER.notes_from_difficulty(chart["difficulties"][tier])


def flat_to_notes(flat):
    return sorted((ER.Note(d["ms"], d["lane"]) for d in flat), key=lambda n: (n.ms, n.lane))


def load_ours_fixture(name):
    return json.load(open(os.path.join(FIXTURES_DIR, name)))


# ---------------------------------------------------------------------------
# Primary metric: edit_rate + note-count ratio, all four systems, all live
# ---------------------------------------------------------------------------


def rows_for_system(system_name, cand_by_song_tier, song_ids, charts):
    rows = []
    for sid in song_ids:
        chart = charts[sid]
        for tier in TIERS:
            gt = gt_notes(chart, tier)
            cand = cand_by_song_tier[sid][tier]
            ops = ER.edit_ops(cand, gt)
            rows.append({
                "system": system_name, "song_id": sid, "tier": tier,
                "n_gt": len(gt), "n_cand": len(cand), **ops,
            })
    return rows


def pooled_edit_rate(rows):
    if not rows:
        return None
    tot_edits = sum(r["insert"] + r["delete"] + r["lane_move"] + r["slot_move"] for r in rows)
    tot_gt = sum(r["n_gt"] for r in rows)
    return (tot_edits / tot_gt) if tot_gt else None


def pooled_note_ratio(rows):
    if not rows:
        return None
    tot_cand = sum(r["n_cand"] for r in rows)
    tot_gt = sum(r["n_gt"] for r in rows)
    return (tot_cand / tot_gt) if tot_gt else None


# ---------------------------------------------------------------------------
# Secondary metrics (every system now has a full reduced note list)
# ---------------------------------------------------------------------------


def backbone_recall_counts(cand, gt):
    gt_bb = [i for i, g in enumerate(gt) if g.lane in BACKBONE_LANES]
    if not gt_bb:
        return 0, 0
    pairs, _, _ = ER._match(cand, gt, ER.EPS_MS)
    matched_same_lane = {gi for ci, gi in pairs if cand[ci].lane == gt[gi].lane}
    hit = sum(1 for gi in gt_bb if gi in matched_same_lane)
    return hit, len(gt_bb)


def inconsistency_counts(chart, cand_by_tier):
    """chart's own Expert notes/tempos/timeSignatures define the repeated-
    groove clusters (shared reference across every system, so this is an
    apples-to-apples comparison); cand_by_tier supplies the candidate note
    list per tier whose consistency we're measuring."""
    ms_to_measure, _ = CM.build_measure_clock(chart.get("tempos") or [], chart.get("timeSignatures") or [])
    expert = gt_notes(chart, "expert")
    clusters, _ = CM.expert_groove_clusters(expert, ms_to_measure)
    out = {}
    if not clusters:
        for tier in TIERS:
            out[tier] = (0, 0)
        return out
    for tier in TIERS:
        rbm = CM.reduced_groove_by_measure(cand_by_tier[tier], ms_to_measure)
        stats = CM.consistency_stats(clusters, rbm)
        out[tier] = (stats["n_inconsistent_instances"], stats["n_instances"])
    return out


def intrinsic_d_divergence(chart, cand_by_tier):
    tempos = chart.get("tempos") or []
    out = {}
    for tier in TIERS:
        gt = gt_notes(chart, tier)
        cand = cand_by_tier[tier]
        out[tier] = ID.D(cand, tempos) - ID.D(gt, tempos)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    song_ids = load_song_ids()
    charts = {sid: load_chart(sid) for sid in song_ids}
    print(f"{len(song_ids)} rb4_test songs (fixture, non-edge)")

    ours_model = load_ours_fixture("parity_fixture.json")
    ours_scorecard = load_ours_fixture("scorecard_parity_fixture.json")

    print("Running HOPCAT + Onyx live over all songs/tiers...")
    cand = {
        "ours-model": {sid: {t: flat_to_notes(ours_model["songs"][sid][t]) for t in TIERS} for sid in song_ids},
        "ours-scorecard": {sid: {t: flat_to_notes(ours_scorecard["songs"][sid][t]) for t in TIERS} for sid in song_ids},
        "hopcat": {sid: {t: flat_to_notes(reduce_hopcat(charts[sid], t)) for t in TIERS} for sid in song_ids},
        "onyx": {sid: {t: flat_to_notes(reduce_onyx(charts[sid], t)) for t in TIERS} for sid in song_ids},
    }
    systems = ["ours-model", "ours-scorecard", "hopcat", "onyx"]

    rows = {s: rows_for_system(s, cand[s], song_ids, charts) for s in systems}

    # ---- primary: edit_rate + note-count ratio, all four systems ----
    table = []
    for system in systems:
        srows = rows[system]
        for tier in TIERS:
            trows = [r for r in srows if r["tier"] == tier]
            table.append({
                "system": system, "tier": tier,
                "edit_rate": pooled_edit_rate(trows),
                "note_count_ratio": pooled_note_ratio(trows),
                "n_songs": len(trows),
            })
        table.append({
            "system": system, "tier": "pooled",
            "edit_rate": pooled_edit_rate(srows),
            "note_count_ratio": pooled_note_ratio(srows),
            "n_songs": len(srows),
        })

    print("\n=== primary: edit_rate / note_count_ratio ===")
    for r in table:
        er = f"{r['edit_rate']:.4f}" if r["edit_rate"] is not None else "n/a"
        ncr = f"{r['note_count_ratio']:.3f}" if r["note_count_ratio"] is not None else "n/a"
        print(f"  {r['system']:16s} {r['tier']:8s} edit_rate={er}  note_ratio={ncr}  n={r['n_songs']}")

    def pooled_of(system):
        return next(r for r in table if r["system"] == system and r["tier"] == "pooled")["edit_rate"]

    om_pooled, os_pooled = pooled_of("ours-model"), pooled_of("ours-scorecard")
    hopcat_pooled, onyx_pooled = pooled_of("hopcat"), pooled_of("onyx")

    print(f"\nCross-check: ours-model pooled = {om_pooled:.4f} (expect {OURS_MODEL_TARGET}), "
          f"ours-scorecard pooled = {os_pooled:.4f} (expect {OURS_SCORECARD_TARGET})")
    assert abs(om_pooled - OURS_MODEL_TARGET) < 0.0001, "ours-model pooled edit_rate drifted from the documented value"
    assert abs(os_pooled - OURS_SCORECARD_TARGET) < 0.0001, "ours-scorecard pooled edit_rate drifted from the documented value"

    print(f"Reconciliation: HOPCAT pooled = {hopcat_pooled:.4f} (frozen real-MIDI eval: {HOPCAT_TARGET}, "
          f"diff {hopcat_pooled - HOPCAT_TARGET:+.4f}), Onyx pooled = {onyx_pooled:.4f} "
          f"(frozen real-MIDI eval: {ONYX_TARGET}, diff {onyx_pooled - ONYX_TARGET:+.4f})")
    for name, pooled, target in (("HOPCAT", hopcat_pooled, HOPCAT_TARGET), ("Onyx", onyx_pooled, ONYX_TARGET)):
        if abs(pooled - target) > RECONCILE_TOL:
            print(f"  WARNING: {name} pooled diverges from the frozen real-MIDI number by more than "
                  f"{RECONCILE_TOL} -- see eval_quality.py's docstring / baselines/{name.lower()}.py's "
                  "documented divergences (no OD/marker data in this repo's chart schema) before trusting this.")

    # ---- secondary: backbone recall, inconsistency, intrinsic-D (all four now) ----
    bb_counts = {s: {t: [0, 0] for t in TIERS} for s in systems}
    inc_counts = {s: {t: [0, 0] for t in TIERS} for s in systems}
    d_divs = {s: {t: [] for t in TIERS} for s in systems}

    for sid in song_ids:
        chart = charts[sid]
        for system in systems:
            cand_by_tier = cand[system][sid]
            for tier in TIERS:
                gt = gt_notes(chart, tier)
                hit, tot = backbone_recall_counts(cand_by_tier[tier], gt)
                bb_counts[system][tier][0] += hit
                bb_counts[system][tier][1] += tot

            inc = inconsistency_counts(chart, cand_by_tier)
            for tier in TIERS:
                inc_counts[system][tier][0] += inc[tier][0]
                inc_counts[system][tier][1] += inc[tier][1]

            dd = intrinsic_d_divergence(chart, cand_by_tier)
            for tier in TIERS:
                d_divs[system][tier].append(dd[tier])

    print("\n=== secondary: backbone (kick+snare) recall ===")
    secondary = {}
    for system in systems:
        for tier in TIERS:
            hit, tot = bb_counts[system][tier]
            rate = (hit / tot) if tot else None
            secondary.setdefault(system, {}).setdefault(tier, {})["backbone_recall"] = rate
            print(f"  {system:16s} {tier:8s} backbone_recall={rate:.4f}" if rate is not None else
                  f"  {system:16s} {tier:8s} backbone_recall=n/a")

    print("\n=== secondary: inconsistency rate (repeated grooves reduced differently) ===")
    for system in systems:
        for tier in TIERS:
            inc, tot = inc_counts[system][tier]
            rate = (inc / tot) if tot else None
            secondary[system][tier]["inconsistency_rate"] = rate
            print(f"  {system:16s} {tier:8s} inconsistency_rate={rate:.5f}" if rate is not None else
                  f"  {system:16s} {tier:8s} inconsistency_rate=n/a")

    print("\n=== secondary: intrinsic-difficulty divergence D(system) - D(Harmonix) ===")
    for system in systems:
        for tier in TIERS:
            vals = d_divs[system][tier]
            mean_dd = sum(vals) / len(vals) if vals else None
            secondary[system][tier]["intrinsic_d_divergence"] = mean_dd
            print(f"  {system:16s} {tier:8s} D_divergence={mean_dd:+.4f}" if mean_dd is not None else
                  f"  {system:16s} {tier:8s} D_divergence=n/a")

    # ---- write data/reference_scores.tsv ----
    with open(OUT_TSV, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["system", "tier", "n_songs", "edit_rate", "note_count_ratio",
                    "backbone_recall", "inconsistency_rate", "intrinsic_d_divergence"])
        for r in table:
            system, tier = r["system"], r["tier"]
            sec = secondary.get(system, {}).get(tier, {})
            w.writerow([
                system, tier, r["n_songs"],
                f"{r['edit_rate']:.6f}" if r["edit_rate"] is not None else "",
                f"{r['note_count_ratio']:.6f}" if r["note_count_ratio"] is not None else "",
                f"{sec.get('backbone_recall'):.6f}" if sec.get("backbone_recall") is not None else "",
                f"{sec.get('inconsistency_rate'):.6f}" if sec.get("inconsistency_rate") is not None else "",
                f"{sec.get('intrinsic_d_divergence'):+.6f}" if sec.get("intrinsic_d_divergence") is not None else "",
            ])
    print(f"\nWrote {OUT_TSV}")


if __name__ == "__main__":
    main()
