# drum-reducer: reference spec

Byte-exact port spec for both backends, self-contained within this repo.
Read [DETERMINISM_CONTRACT.md](DETERMINISM_CONTRACT.md) alongside this
file — tie-breaks, summation order, and the portable-`exp` algorithm live
there and are not repeated here.

Source model cache key: `08b9f91735e6dcf7`. rb4_test canonicalized
edit_rate **0.1703** (model backend), **0.2234** (scorecard backend) — see
the [README](README.md) for the full quality table.

---

## 1. Overview

Three stages, run once per tier (`hard`, `medium`, `easy`):

1. **FEATURIZE** (deterministic, shared by both backends) — for every
   Expert note, compute a fixed 59-value feature vector from the chart
   alone (§2). No model, no randomness.
2. **MODEL** (swappable — this is the only stage that differs between
   backends):
   - **backend `model`**: two families of packed gradient-boosted-tree
     ensembles (§3). `survive`: per tier, one binary head → `survive_proba`
     per note. `relane`: per tier per family (cymbal, tom), one multiclass
     head → `final_lane` + `confidence` per family note.
   - **backend `scorecard`**: an additive integer point table for
     `survive` (§4), and depth-≤2 decision trees for `relane` (§4).
   Kick/snare/other notes are never relaned (they keep their Expert lane)
   under either backend. Features are the model's **inputs only** — never
   tune them to change reduction behavior; all tunable behavior lives in
   DECODE.
3. **DECODE** (deterministic algorithm + a few fixed per-backend knobs,
   §5/§6) — turns raw survive/relane predictions into the final note list:
   pool → threshold → thin (NMS) → relane → pool → dedup → canonicalize.
   Identical code for both backends except one parameterized comparison
   (§5, step 3/4).

---

## 2. Features (59, exact order — see `data/model/feature_names.json`)

Identical to the original spec's §5. All computed per Expert note from the
chart alone. `EPS_MS = 0.5` (tick rounding slack); `tick = round(ms /
EPS_MS)` is the "same instant" key used throughout. Order below is the
exact column order — both backends, both languages, must agree on this
order; the scorecard's `base_points`/summation and the model's tree
`feature_idx` both index into it directly.

**A. Base chart features (13)**
1. `chord_size` — count of Expert notes sharing this note's `(ms)` tick.
2. `beat_in_measure` — beat position within its measure (0 at the
   downbeat), from the song's tempo/time-signature map.
3. `beats_per_measure` — `numerator * 4 / denominator` of the active time
   signature.
4. `is_downbeat` — 1 if `beat_in_measure < 0.04` (beats), else 0.
5. `local_density_500ms` — count of OTHER DISTINCT Expert TIME POSITIONS
   (deduped `ms` values — a chord counts once) within ±250ms.
6. `gap_prev_ms` — ms to the previous distinct Expert tick, capped at 5000.
7. `gap_next_ms` — ms to the next distinct Expert tick, capped at 5000.
8. `ghost` — 1 if the note entry has `ghost: true`.
9. `accent` — 1 if the note entry has `accent: true`.
10. `flam` — 1 if the note entry has `flam: true`.
11. `aligned_half` — 1 if `beat_in_measure mod 2.0` is within 0.04 beats of
    0 or 2.0.
12. `aligned_quarter` — same test, `mod 1.0`.
13. `aligned_eighth` — same test, `mod 0.5`.

**DROPPED** (computed by the base extractor but excluded from the model's
input matrix — position-invariance): `position_in_song`, `section_progress`,
`section_frac`. Do not compute or send these.

**B. Lane one-hot (10)**: `lane_kick, lane_snare, lane_hihat, lane_open-hat,
lane_high-tom, lane_mid-tom, lane_floor-tom, lane_crash, lane_ride,
lane_other`. Exactly one is 1.

**C. Section-type one-hot (12 — deliberate duplicate)**: `section_intro,
section_outro, section_prechorus, section_prechorus, section_chorus,
section_verse, section_bridge, section_solo, section_breakdown,
section_interlude, section_fill, section_other`. `section_prechorus`
appears **twice** — a real artifact of the reference implementation's
section-keyword table (two synonym entries, `"pre-chorus"` and
`"prechorus"`, both mapping to label `"prechorus"`, one-hot built by
iterating the table directly rather than a deduped vocab). Both duplicate
columns get the SAME value. **Reproduce the duplicate exactly** — de-duping
it shifts every feature index after it (columns D/E/F below).
Classification: lowercase the section name, test substrings `intro, outro,
pre-chorus, prechorus, chorus, verse, bridge, solo, breakdown, interlude,
fill` in that order (first match wins), else `other`.

**D. Era one-hot (5)**: `era_RB1, era_RB2, era_RB3, era_RB4, era_other`
(chart provenance metadata, not audio-derived).

**E. Chord-context flags (10)**: `chord_has_kick, chord_has_snare,
chord_has_hihat, chord_has_open-hat, chord_has_high-tom, chord_has_mid-tom,
chord_has_floor-tom, chord_has_crash, chord_has_ride, chord_has_other` — for
each lane, 1 if ANY Expert note at this note's same tick (any lane) is in
that lane.

**F. AUG_FEATS v7 (9)** — computed once per song from the full Expert note
list (song-level context):
- `aug_dist_backbone_ms` — ms distance to the NEAREST kick-or-snare note
  anywhere in the song (both directions, min); 5000.0 if the song has no
  kick/snare at all.
- `aug_density_ratio` — `local_density_500ms / (median(local_density_500ms
  over the song) + 1)`.
- `aug_samelane_prev_ms` / `aug_samelane_next_ms` — ms gap to the nearest
  PRIOR / NEXT note in this note's own lane (song-wide), capped at 5000.
- `aug_chord_priority` — count of same-tick notes with a strictly
  higher-priority lane, rank `kick=0, snare=1, crash=2, ride=3, hihat=4,
  open-hat=5, floor-tom=6, mid-tom=7, high-tom=8, other=9` (lower = more
  important).
- `aug_density_100ms` — count of OTHER Expert NOTES (per-instrument, NOT
  deduped — a chord counts once per lane) within ±100ms.
- `aug_density_1500ms` — same per-note count, within ±1500ms.
- `aug_beat_frac` — `abs(beat_in_measure - round(beat_in_measure))`.
- `aug_lane_frac_500ms` — fraction of ALL notes (per-note, not deduped)
  within ±500ms that share this note's lane (0 if the window is empty).

  **Gotcha**: `local_density_500ms` (A.5) counts distinct TIME POSITIONS (a
  4-note chord = 1); `aug_density_100ms`/`aug_density_1500ms`/
  `aug_lane_frac_500ms`'s denominator count individual NOTES (a 4-note
  chord = 4). Two different arrays in the reference implementation
  (deduped vs one-entry-per-note) — replicate both, don't conflate.

13 + 10 + 12 + 5 + 10 + 9 = **59**, matching `data/model/feature_names.json`.
If this doc and that file ever disagree, trust the JSON.

---

## 3. Model backend: packed-GBM binary format

### 3.1 What's shipped (`data/model/`)

9 `.bin` files (`survive_{hard,medium,easy}.bin`,
`relane_{cymbal,tom}_{hard,medium,easy}.bin`) + `manifest.json` +
`feature_names.json`. All integers little-endian. `NODE_STRUCT` = 7 bytes,
format `<BBBBBe`:

```
feature_idx   u8    which of the 59 features this split tests
bin_threshold u8    split point, as a BIN INDEX (0-255), not a raw value
left          u8    local node index of left child (within this tree)
right         u8    local node index of right child (within this tree)
flags         u8    bit0 = is_leaf, bit1 = missing_go_to_left
value         f16   ONLY meaningful when is_leaf: leaf's contribution to
                     the raw score. 0.0 at internal nodes.
```

A tree's nodes are a flat array; node 0 is the root. `left`/`right` are
node-array indices (multiply by 7 for a byte offset within the tree's own
node-byte slice).

### 3.2 `survive_{tier}.bin` (SURV)

```
offset  size  field
0       4     magic = "SURV"
4       1     version = 1
5       2     n_features (u16) = 59
7       8     baseline (f64)
15      8     learning_rate (f64)          -- 0.08
23      4     n_trees (u32)                -- 200 (max_iter)
27      2*n_trees   node_counts[i] (u16)
...     sum(node_counts)*7   concatenated tree node bytes, tree 0 first
...     bin-edge table (§3.4)
```

### 3.3 `relane_{family}_{tier}.bin` (RLAN)

`family` is `cymbal` or `tom`. Fixed lane list per family (`manifest.json`'s
`families`): `cymbal: [hihat, open-hat, crash, ride]`, `tom: [high-tom,
mid-tom, floor-tom]`.

```
offset  size  field
0       4     magic = "RLAN"
4       1     version = 1
5       2     n_features (u16) = 59
7       1     n_classes (u8)               -- OBSERVED classes only, can be
                                               < len(family lane list)
8       n_classes    classes_[c] (u8)      -- column c predicts lane index
                                               classes_[c] into the family's
                                               FIXED lane list, NOT column
                                               index c itself
8+nc    8*n_classes  baseline[c] (f64)
...     8            learning_rate (f64)
...     4            n_iters (u32)         -- boosting rounds (<=200)
...     2*n_iters*n_classes   node_counts, ITERATION-MAJOR then CLASS-MAJOR
...     sum(node_counts)*7    concatenated tree node bytes, same order
...     bin-edge table (§3.4)
```

### 3.4 Bin-edge table (both file types)

For each of the 59 features in order: `n_edges (u16)` then `n_edges * f64`
edge values, ascending. **Edges are fp64, not fp32** — reading the real
`.bin` files back and re-binning against their own on-disk fp32 edges
(an earlier packer revision) produced measurable threshold-flips and
relane argmax-mismatches; fp64 is bit-exact. Do not "optimize" this back
to fp32.

### 3.5 Inference

Per note, per tier:

1. Compute the 59 raw features (§2), in `feature_names.json`'s order.
2. Re-bin each raw value `x` into a 0-255 bin index using that feature's
   edge table: `bin = searchsorted(edges, x, side='left')`, clamped to
   `[0, 255]`. **Must be `side='left'`** — the smallest index `i` such that
   `x <= edges[i]`. `side='right'` compiles and runs but silently produces
   a different, still-plausible wrong traversal.
3. Traverse every tree: start at node 0; at each internal node,
   `go_left = (bin[feature_idx] <= bin_threshold)`; follow `left`/`right`;
   stop at a leaf and read its `value`.
4. Sum: `raw = baseline + sum(leaf.value for every tree)` (survive), or
   per-class `raw[c] = baseline[c] + sum(leaf.value for trees assigned to
   class c)` (relane). **Leaf values already have `learning_rate` baked
   in — do NOT multiply by learning_rate again.**
5. Link function:
   - survive: `proba = sigmoid(raw) = 1 / (1 + exp(-raw))`
   - relane: `proba = softmax(raw)` over the `n_classes` raw scores;
     `argmax_col = argmax(proba)`; `final_lane =
     lane_list[classes_[argmax_col]]` (the `classes_` indirection);
     `confidence = proba[argmax_col]`.

Both gotchas above (bin-search `side='left'`, no re-scaling by
learning_rate) are the most likely source of a silent-but-wrong port. See
DETERMINISM_CONTRACT.md §3 for how `exp` itself is made cross-language
portable.

**Model-backend interface** (what DECODE consumes, §5, and what a port
must match — see `python/drum_reducer/backend_model.py`):
- `predict_survive(tier, X) -> [survive_proba]` in `[0, 1]`. Used for the
  pooled-mean survive decision and as the NMS keep_score.
- `predict_relane(tier, family, X) -> (final_lane, confidence)`.
  `confidence` is the winning class's softmax proba, used by relane-pool
  (summed per candidate lane) and chord-merge (max).
- `survive_threshold(tier) = 0.5` (all 3 tiers).
- `nms_gap(tier)`: `hard: None (skip NMS)`, `medium: 180`, `easy: 250` (ms).

---

## 4. Scorecard backend: auditable integer format

Source artifact: `data/scorecard/scorecard.json` (machine-readable) +
`data/scorecard/scorecard_rules.py` (generated human-readable rendering) +
`data/scorecard/AUDIT.md` (per-feature breakpoint counts, generation log).
Built by `tools/build_scorecard.py` (extraction, plus decode-knob selection
on a held-out val split) — see DETERMINISM_CONTRACT.md §4 for the
derivation rules (depth-1 survive
refit collapsed into per-feature piecewise-constant integer point tables;
depth-2 relane trees on bin-indexed features).

### 4.1 `scorecard.json` schema

```
{
  "source_commit": "591ab4a", "sklearn_version": "...",
  "quant_scale": <float>,            -- point-quantization scale (documentation only)
  "lane_vocab": [...], "families": {...}, "feature_names": [...59 names...],
  "relane_bin_edges": [ [f64 edges for feature 0], [feature 1], ... ],  -- 59 arrays
  "survive": {
    "hard": {
      "base_points": <int>, "T_tier": <int>,   -- raw quantization-step boundary (see 4.3)
      "features": {
        "<feature_name>": { "bin_edges": [f64...], "points": [int...] }
        -- len(points) == len(bin_edges) + 1; a single-element points list
           means the feature is unused (constant 0 contribution) for this tier
      }
    },
    "medium": {...}, "easy": {...}
  },
  "relane": {
    "hard": {
      "cymbal": { "lanes": [...], "observed_classes": [...], "tree": <node> },
      "tom": {...}
    },
    "medium": {...}, "easy": {...}
  },
  "decode": {
    "hard":   {"T_tier": <int>, "nms_gap": <float|null>},
    "medium": {...}, "easy": {...}
  }
}
```

A relane tree `<node>` is either `{"leaf": false, "feature_idx": <int>,
"threshold": <int bin index>, "left": <node>, "right": <node>}` or
`{"leaf": true, "class_counts": {"<lane_name>": <int>, ...}}`.

### 4.2 Survive: integer point sum

Per note: `points = base_points + Σ_f feature_points[f](bin(x_f))`
(integer, summed in `feature_names` order — DETERMINISM_CONTRACT.md §1).
`bin(x_f)`: `searchsorted(bin_edges, x_f, side='left')`, clamped to
`[0, len(points)-1]` — same convention as the model backend's rebin (§3.5
step 2), but the scorecard's per-feature edge tables are typically much
shorter (see AUDIT.md's breakpoint counts) since most are collapsed to 1-8
merged rows.

**Un-pooled decision:** `points >= T_tier`. **survive-pool decision
(integer-exact mean):** for a groove group of size `n`, keep iff
`Σ points_i >= n * T_tier` — the shared decode's pooling comparison is
mathematically `mean(points) >= T_tier`, computed via IEEE-754
division (deterministic, portable) rather than reimplemented as a
separate integer-sum comparison; see `reduce.py`'s own comment for why
this is equivalent to DETERMINISM_CONTRACT.md §4's `sum >= n*T` framing.

**`T_tier` to use**: `scorecard.json`'s `decode.<tier>.T_tier`, NOT
`survive.<tier>.T_tier` — the latter is only the raw quantization-step
mapping of the depth-1 model's own 0.5-probability boundary into
integer-points space; the former is the scorecard's actual validated
operating point for the shipped decode (selected on a held-out val split,
see `tools/build_scorecard.py`). Same distinction for `nms_gap`: read
`decode.<tier>.nms_gap`, not the model backend's fixed per-tier gaps.

**NMS keep_score** for the scorecard = the pooled integer `points` sum
(left as an integer — NMS only needs a total ordering; DETERMINISM_CONTRACT
§2.1's tie-break makes it total even on exact ties, which are common on
repeated grooves for an integer-valued score).

### 4.3 Relane: depth-≤2 trees

Walk the tier/family's tree from the schema above: at an internal node,
`bin_index(x[feature_idx]) <= threshold` picks `left`, else `right`
(`bin_index` via `relane_bin_edges[feature_idx]`, same `searchsorted
side='left'` convention as above). At a leaf, `final_lane =
argmax(class_counts)`, ties broken by lowest `LANE_INDEX`
(DETERMINISM_CONTRACT.md §1/§2's tie-break style — NOT insertion order).
`confidence = winner_count - runnerup_count` (an integer >= 0 — the
"integer margin of the winning class's summed integer leaf score over the
runner-up"). Used by relane-pool (summed per candidate lane, §5 step 7)
and chord-merge (max, §5 step 8).

---

## 5. Decode pipeline (shared, both backends)

Run once per tier. `groove_cluster`/measure-clock terms refer to the SAME
machinery both pooling steps and canonicalize (step 9) share; build it once
per song.

**Measure clock & groove clusters**: from the chart's
`tempos`/`timeSignatures`, build `ms_to_measure(ms) -> (measure_idx,
beat_in_measure)` and its inverse `measure_to_ms`. Bucket the Expert note
list into `reduced_groove_by_measure`: `{measure_idx: frozenset of
(round(beat_in_measure * 480), lane)}` (480 = GROOVE_TPQ, RB-convention
480 ticks-per-quarter). Group measures with an IDENTICAL Expert groove into
clusters; only clusters with ≥2 measures matter.

1. **Featurize** — §2, exact column order.
2. **Survive predict** — backend-specific (§3.5 step 4-5 for `model`, §4.2
   for `scorecard`) → per-note score (`survive_proba` for model, integer
   `points` for scorecard).
3. **SURVIVE-POOL** — for every note, group by key
   `(expert_groove_cluster_id, round(beat_in_measure * 480), lane)`.
   Replace each note's score with the arithmetic mean across its group
   (backend-agnostic: `mean(proba)` or `mean(points)`, IEEE division either
   way). Notes not in any cluster are unaffected. Runs BEFORE thresholding.
4. **Threshold** — `survive = pooled_score >= backend.survive_threshold(tier)`.
   This one comparison is the backend-parameterized step
   (DETERMINISM_CONTRACT.md §4's "one honest exception to shared DECODE");
   every other step here is byte-identical code across backends.
5. **FAMILY-NMS** (family = cymbal or tom; kick/snare/other never
   suppressed) — gap = `backend.nms_gap(tier)` (model: fixed per §3.5;
   scorecard: `decode.<tier>.nms_gap` per §4.2). `None`/absent gap skips
   this step entirely. Algorithm: among currently-surviving family notes
   (across both families combined), sort by pooled score descending,
   ties by `(-score, ms, lane_index)` (DETERMINISM_CONTRACT.md §2.1). Walk
   the sorted list, keeping a running list of kept note times; drop a note
   if it falls within `gap_ms` of any already-kept note's ms, else keep it.
   Greedy, confidence-order not time-order.
6. **Relane predict** — for surviving notes whose Expert lane is in a
   family, run that family+tier's relane head (§3.5 or §4.3) →
   `final_lane`, `confidence`. Non-family survivors keep their own Expert
   lane, `confidence = 1.0` (model) / max integer margin (scorecard is
   never invoked for non-family notes, so this doesn't arise there).
7. **RELANE-POOL** — for FAMILY notes only, group by
   `(expert_groove_cluster_id, round(beat_in_measure * 480), SOURCE lane)`.
   Within each group, override every member's `final_lane` with the
   confidence-weighted modal lane: sum `confidence` per candidate
   `final_lane` across the group, pick the highest-summed-confidence lane,
   ties by lowest `lane_index` (DETERMINISM_CONTRACT.md §2.2). Runs AFTER
   relane predict, BEFORE chord-merge.
8. **Chord-merge dedup** — group surviving FAMILY notes by `(ms, family,
   final_lane)`; if a group has >1 member, keep only the highest-confidence
   one, tie broken by lowest SOURCE `lane_index` (§2.3). Fixed-lane
   survivors pass through unchanged.
9. **Canonicalize** — for every repeated-Expert-groove cluster: compute
   each instance measure's ACTUAL reduced groove (from this candidate's
   own note list, after steps 1-8); take the modal reduction across all
   instances (ties per DETERMINISM_CONTRACT.md §2.4 — lexicographically
   smallest sorted `(tick, lane_index)` list, NOT insertion order); force
   every instance's measure to that modal reduction. Non-clustered measures
   pass through untouched. Implement last, after everything else.

**ORDER MATTERS and is not the "obvious" reading**: survive-pool runs
BEFORE thresholding AND before NMS (NMS operates on pooled score, not raw).
Relane-pool runs AFTER relane predict but BEFORE chord-merge. Canonicalize
is always last. A port that reorders these will not reproduce the
documented edit_rate even if every other step is individually correct.

Final output: notes sorted `(ms, lane_index)` ascending
(DETERMINISM_CONTRACT.md §1).

---

## 6. Tunable knobs

| knob | model backend | scorecard backend |
|---|---|---|
| survive threshold | 0.5, all 3 tiers | per-tier `decode.<tier>.T_tier` (scorecard.json) |
| family-NMS gap | hard: off, medium: 180ms, easy: 250ms | per-tier `decode.<tier>.nms_gap` (scorecard.json) |

Both are genuine decode-time parameters (not features, not baked into
model weights), fixed at their validated values, not exposed as a runtime
dial. Two related knobs exist in the wider research codebase but are **not
part of either shipped backend** — a per-tier `relane_gate` (confidence
floor below which a family note keeps its Expert lane) and a per-lane
`lane_thr` (per-lane survive-threshold override). Both were found
neutral-to-harmful or overfit to a small validation set; out of scope for
this reference implementation.

---

## 7. Input chart schema

```
{
  "tempos": [{"ms": float, "bpm": float}, ...],
  "timeSignatures": [{"ms": float, "numerator": int, "denominator": int}, ...],
  "sections": [{"ms": float, "name": str}, ...],
  "era": "RB1" | "RB2" | "RB3" | "RB4" | "other",
  "difficulties": {
    "expert": {"notes": [[ms, [{"instrument": str, "ghost"?: bool,
                                 "accent"?: bool, "flam"?: bool}, ...]], ...]}
  }
}
```

`notes` is already grouped by `ms` (one entry per distinct tick, chords
listed together) — this is what `chord_size` (feature A.1) counts directly,
no re-grouping needed. `instrument` is one of the 9-lane vocab (`kick,
snare, hihat, open-hat, high-tom, mid-tom, floor-tom, crash, ride`) or
`other`.

---

## 8. Parity verification

`data/fixtures/parity_fixture.json` (model backend) and
`data/fixtures/scorecard_parity_fixture.json` (scorecard backend): the
full pipeline's output (steps 1-9 above, run end-to-end) for **99 rb4_test
songs** (real charts, the same ones used for the README's quality table) +
**3 synthetic edge-case songs** (`edge-empty-groove-measures`,
`edge-midsong-ts-change`, `edge-no-backbone`) × 3 tiers, sorted by
`(ms, lane_index)`. `songs[song_id][tier]` is a note list
`[{"ms", "lane"}, ...]`; `ms` rounded to 3 decimals.

To verify a port: run it on the same songs' Expert charts (in
`data/fixtures/charts/<song_id>.json`), diff the resulting `(ms, lane)`
list against the fixture's list for each tier, note-for-note. Expect an
EXACT match — 0 insertions, 0 deletions, 0 lane differences. If it doesn't
match, binary-search which decode step (§5) diverges first by comparing
intermediate state (survive set → NMS'd set → relaned set → pooled set →
deduped set → canonicalized set), not just the final note list.

`tools/check_parity.sh` runs both languages' test suites, each of which
independently diffs its own `reduce()` against these same fixtures — both
landing on the identical fixture is transitively "Python == JavaScript."

---

## Notes / gotchas index

- `chord_size` (§2.A.1) and `chord_has_*` (§2.E) both key off `tick =
  round(ms / 0.5)` — two notes within 0.5ms count as "the same tick."
- `section_prechorus` appears twice in the one-hot (§2.C) — deliberate, do
  not de-dupe.
- `local_density_500ms` counts distinct time positions; the `aug_density_*`
  features count individual notes (§2.F gotcha) — don't conflate the two
  arrays.
- Model backend: bin-search must be `side='left'`; leaf values already
  include `learning_rate` — don't re-multiply (§3.5).
- Model backend: bin edges are fp64, not fp32 (§3.4).
- Scorecard backend: read `decode.<tier>.T_tier`/`nms_gap`, not
  `survive.<tier>.T_tier` (§4.2) — the latter is a pre-tuning artifact.
- Decode step order is not the "obvious" reading (§5, final paragraph).
- `feature_names.json` is ground truth for feature order if this doc and
  that file ever disagree.
