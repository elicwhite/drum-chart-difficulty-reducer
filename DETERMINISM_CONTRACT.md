# Determinism contract (load-bearing — all four implementations MUST obey this)

The repo's central claim is that the Python and JavaScript implementations produce
**bit-identical** results. That is only true if every implementation makes the same
decisions in the same order with the same arithmetic. This file is the single source of
truth for how. The real cross-language hazard is **sort/tie semantics and summation
order**, not `exp`. Every implementation in this repo (Python and JavaScript, both
backends) follows THIS file.

Parity bar: **bit-exact by construction**, not "0 diffs observed." The scorecard backend
is integer-arithmetic end-to-end. The model backend is float, but made portable by the
rules below (fixed summation order + a shipped portable `exp`), so it is also bit-exact.

---

## 1. Fixed orderings (no reliance on language sort stability or dict order)

**Lane index** — the canonical integer for every lane, used in every tie-break:
```
0 kick  1 snare  2 hihat  3 open-hat  4 high-tom  5 mid-tom  6 floor-tom
7 crash  8 ride   9 other
```
(This is `LANE_VOCAB` order + `other`. Family lane lists index INTO this same table:
cymbal = [hihat, open-hat, crash, ride], tom = [high-tom, mid-tom, floor-tom].)

**Note canonical order** — whenever a list of notes must be iterated deterministically
(summation, grouping, output), sort by `(ms, lane_index)` ascending. Final output note
lists are emitted in this order.

**Summation order** — every float/integer sum accumulates left-to-right in a FIXED order:
- Tree leaf-value sums (model backend): tree index `0 .. n_trees-1`.
- Per-feature point sums (scorecard): feature index `0 .. 58` (feature_names.json order).
- Pooling means: members in `(ms, lane_index)` order.
- Confidence-weighted modal accumulation: candidate lanes in `lane_index` order.
Never use language `sum()` over an unordered set; accumulate explicitly in these orders.

---

## 2. Tie-breaks (every decision point that could tie)

1. **FAMILY-NMS greedy sort** — sort surviving family notes by descending keep-score,
   ties broken deterministically. Sort key = `(-keep_score, ms, lane_index)`. (keep_score
   = pooled survive_proba for the model backend; pooled integer points for the scorecard.
   Equal pooled scores across repeated-groove members are COMMON, not rare — this
   tie-break is mandatory, and because NMS is greedy one flip cascades.)
2. **relane-pool modal** — pick the `final_lane` with the highest summed confidence within
   the group; tie broken by lowest `lane_index`.
3. **chord-merge dedup** — within a `(ms, family, final_lane)` group keep the member with
   highest confidence; tie broken by lowest SOURCE `lane_index`.
4. **canonicalize modal** — pick the modal reduced-groove across a cluster's instances;
   tie broken by choosing the groove whose sorted `(tick, lane_index)` tuple list is
   lexicographically smallest. (The Python and JavaScript `canonicalize` implementations
   must apply this content-based rule identically.)

---

## 3. Model backend (packed GBM): making float portable

- **Traversal / decode arithmetic** (f16→f64 leaf decode, `searchsorted` on f64 edges,
  add/sub/mul/div, comparisons) is IEEE-754 correctly-rounded → already identical across
  Python and JS. No action beyond fixed summation order (§1).
- **`searchsorted` = `side='left'`**, clamp to `[0,255]` (per SPEC.md §3.5).
- **Leaf values already include learning-rate shrinkage — never re-multiply** (§4).
- **The only non-portable primitive is `exp`.** Two-part fix:
  - Discrete decisions need NO exp: survive `proba >= 0.5 ⟺ raw >= 0`; relane
    `argmax(softmax(raw)) = argmax(raw)`. Use raw scores for the un-pooled survive
    threshold and the relane argmax.
  - The values that DO need real probabilities — pooled survive means and relane
    confidences (both feed later decode steps) — use a **shipped portable `exp`**: one
    fixed algorithm implemented identically in Python and JS (`portable_exp`), unit-tested
    to agree bit-for-bit on a dense grid. `sigmoid(raw)=1/(1+portable_exp(-raw))`,
    `softmax` over raw with `portable_exp`. This makes the whole model backend bit-exact
    by construction (rather than merely producing "identical discrete output" in practice).

---

## 4. Scorecard backend (auditable, integer): the model interface

The scorecard is a depth-1 GBM refit whose 200 stumps are collapsed, **per feature**, into
one piecewise-constant points function (a GAM/EBM — this collapse is *lossless*
re-expression, the only lossy step is the depth-1 refit itself). Contributions are
quantized to **integers** at a fixed scale (target total dynamic range ~10^4 points) so
every downstream decision is integer arithmetic.

**Survive interface:**
- Per note: `points = base_points + Σ_f feature_points[f](bin(x_f))` (integer, summed in
  feature-index order).
- Per-tier integer threshold `T_tier`.
- **Un-pooled decision:** `points >= T_tier`.
- **survive-pool decision (integer-exact mean):** for a groove group of size `n`, keep iff
  `Σ points_i >= n * T_tier` (faithful integer analog of "mean >= T"; NO float). The
  shared decode's pooling comparison is therefore **backend-parameterized**:
  model = `mean(proba) >= 0.5`, scorecard = `sum(points) >= n*T`. Everything else in
  decode (grouping keys, NMS structure, relane-pool, chord-merge, canonicalize) is
  byte-identical shared code. (This is the one exception to "DECODE is fully shared" —
  every other decode step is identical across both backends.)
- **NMS keep_score for the scorecard = integer `points`** (pooled sum left as an integer;
  NMS only needs an ordering, and the §2.1 tie-break makes it total).

**Relane interface (scorecard):**
- Tiny depth-2 trees per `(tier, family)`, rendered as readable nested `if/else` with
  visible integer thresholds. Emit `final_lane` + integer `confidence`.
- `confidence` = integer margin of the winning class's summed integer leaf score over the
  runner-up (>= 0). Used by relane-pool (summed per candidate lane) and chord-merge
  (max). All integer → bit-exact. Ties per §2.2 / §2.3.

**Auditability:** `build_scorecard.py` (a) merges adjacent bins whose point delta is 0 and
(b) reports per-feature breakpoint counts and which features are actually used, so the
scorecard's real size and readability can be stated accurately rather than assumed.

---

## 5. Measurement rule

The scorecard's headline edit_rate MUST be measured through the **shipped decode path**
(the integer pooling semantics above) — the same `reduce(..., backend="scorecard")` code
this repo ships — so the published number reflects exactly what this code produces on the
fixtures, with no separate evaluation path that could diverge from it.

---

## 6. Fixtures (determinism claims need coverage)

The parity fixture must exercise the gotcha paths, not just 8 mainstream songs:
- Full `rb4_test` set (~99 songs) reduced by both backends, per tier.
- Synthetic/selected edge cases: a song with NO kick/snare (the `aug_dist_backbone_ms`
  5000.0 sentinel); a tier/family where the RLAN `n_classes` is short an observed class
  (e.g. open-hat absent — the `classes_` indirection path); measures with empty grooves;
  a mid-song time-signature change; a repeated-groove cluster with a modal tie (§2.4).
