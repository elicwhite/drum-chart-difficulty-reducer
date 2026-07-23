# drum-chart-difficulty-reducer

A reference implementation of a drum difficulty reducer: takes an **Expert**
drum chart (the source-of-truth note list) and produces **Hard / Medium /
Easy** reductions, the same way a human charter reduces a chart by hand.

This is the reducer core only. It does not parse chart files (`.mid`,
`.chart`) — the input is an already-parsed note list plus tempo map, time
signatures, sections, and chart-provenance metadata. See [SPEC.md](SPEC.md)
for the exact input schema.

## What it is

The reducer is three plain stages:

1. **Featurize** — for every Expert note, compute 59 measurable properties
   from the chart alone: how many notes are stacked on it, its position in
   the measure, how dense the surrounding few hundred milliseconds are, what
   section it's in, and so on. This stage is pure arithmetic over the chart, 
   and it's identical for every backend below.
2. **Score** — each note gets a score for whether it should survive into a
   given tier, and (for cymbal/tom notes) which lane it should collapse
   into. The scoring rules were **learned from thousands of Harmonix's own
   official Hard/Medium/Easy reductions of Expert charts** — i.e. this
   stage encodes patterns in how humans have already reduced charts, not a
   hand-tuned heuristic and not a black-box neural net. Two interchangeable
   implementations of this stage ship (below); everything else is
   identical.
3. **Decode** — a fixed, documented 9-step algorithm turns per-note scores
   into a final note list: pool scores across repeated grooves so the same
   pattern reduces the same way everywhere it appears, threshold, thin out
   cymbal/tom crashes that land too close together, relane survivors,
   dedupe, and force every repeat of a groove to its own most common
   reduction. See [SPEC.md](SPEC.md) §5 for the exact steps and ordering —
   order is load-bearing, not the "obvious" reading.

## Two backends, pick your tradeoff

The **featurize** and **decode** stages are shared code. Only the **score**
stage is swappable:

| backend | what it is | rb4_test pooled edit_rate |
|---|---|---:|
| **model** (`backend_model.py`/`.js`) | a packed gradient-boosted-tree ensemble (200 trees/head), stored as a compact binary format, evaluated with plain integer/float arithmetic — no ML framework at inference time | **0.1703** |
| **scorecard** (`backend_scorecard.py`/`.js`) | an auditable additive point table: each feature contributes an integer number of points from a small breakpoint table, summed and compared to a threshold; lane-relaning is two tiny (≤7-node) decision trees | **0.2234** |

The model backend is the higher-accuracy option and is lossless relative to
the trained ensemble it's packed from. The scorecard backend gives up
~0.05 edit_rate in exchange for being **fully hand-auditable**: every
scoring rule is an integer table small enough to read in one sitting (see
[Scorecard auditability](#scorecard-auditability) below). Pick based on
whether you need to inspect every rule or want the highest achievable
accuracy against Harmonix's own reductions. Both are provided.

## Closeness to Harmonix vs. existing reduction tools

We don't claim this reducer is "better than" HOPCAT or Onyx — that's a
subjective call, and those tools were built for different goals (fast,
general-purpose, user-tunable reduction, run inside a DAW, no learned
model). What's measured here is a single, objective yardstick: **how
closely does each system's output match Harmonix's own official
Hard/Medium/Easy reductions** of the same Expert chart, on the **99
rb4_test songs** (the RB4 held-out set; see [Provenance](#provenance)
below). **edit_rate** = (inserted + deleted + lane-moved + slot-moved
notes) / |ground-truth notes|, pooled across songs (summed edits over
summed ground-truth notes, not an average of per-song rates) — lower is
closer to Harmonix, 0 = identical to the human chart.

### Primary: edit_rate

| system | hard | medium | easy | **pooled** |
|---|---:|---:|---:|---:|
| **ours — model** | 0.1397 | 0.1568 | 0.2390 | **0.1703** |
| **ours — scorecard** | 0.1907 | 0.2090 | 0.2966 | **0.2234** |
| HOPCAT | 0.2563 | 0.3821 | 0.7462 | 0.4206 |
| Onyx | 0.2506 | 0.3871 | 0.5543 | 0.3713 |

Both of ours land closer to Harmonix on every tier. Pooled, ours-model's
edit_rate is **~2.2x smaller** than Onyx's and **~2.5x smaller** than
HOPCAT's — i.e. our reductions are, on average, roughly twice as close to
Harmonix's official reductions by this edit-distance measure. Even the
fully transparent scorecard backend stays closer to Harmonix than either
tool at every tier. The gap is widest at Easy, where HOPCAT's and Onyx's
more aggressive thinning diverges furthest from how Harmonix actually
charted Easy.

*Note: data measured against HOPCAT's and Onyx's algorithms as
independently reimplemented in this repo (see
[Provenance](#provenance) below) as of July 21, 2026.*

### Secondary: why, not just how much

**Note-count ratio** (kept notes / ground-truth notes — 1.0 = matches
Harmonix's density; a proxy for over- or under-charting):

| system | hard | medium | easy | pooled |
|---|---:|---:|---:|---:|
| ours — model | 1.009 | 0.995 | 1.025 | 1.009 |
| ours — scorecard | 0.964 | 0.979 | 1.077 | 0.997 |
| HOPCAT | 1.118 | 0.915 | 0.407 | 0.873 |
| Onyx | 1.047 | 0.840 | 0.754 | 0.906 |

Ours tracks Harmonix's own note density closely at every tier (within a
few percent). Both compared tools drift further, most dramatically HOPCAT
at Easy: it keeps only 41% as many notes as Harmonix's own Easy chart —
factually, a much sparser chart than what a human charter shipped for the
same song.

**Backbone (kick+snare) recall** — of the ground-truth's kick+snare hits,
what fraction survive in the same lane:

| system | hard | medium | easy |
|---|---:|---:|---:|
| ours — model | 0.922 | 0.941 | 0.933 |
| ours — scorecard | 0.853 | 0.920 | 0.934 |
| HOPCAT | 0.937 | 0.908 | 0.209 |
| Onyx | 0.908 | 0.876 | 0.956 |

HOPCAT's Easy backbone recall of 0.209 is the same effect as its
note-count collapse above: it isn't just thinning cymbals at Easy, it's
dropping most of the groove skeleton itself. Onyx's Easy backbone recall
(0.956) reads high by comparison, but see the Overdrive-phrase note in
[Provenance](#provenance) — Onyx's numbers here run without a
note-preservation safeguard the real tool has, which plausibly makes this
number more favorable to Onyx than the real tool would produce.

**Inconsistency rate** (repeated Expert grooves — the same pattern played
more than once — reduced differently each time it repeats; measured
post-hoc from the final note list, independent of the reducer's own
internal machinery):

| system | hard | medium | easy |
|---|---:|---:|---:|
| ours — model | 0.0064 | 0.0064 | 0.0063 |
| ours — scorecard | 0.0064 | 0.0065 | 0.0063 |
| HOPCAT | 0.0004 | 0.0006 | 0.0025 |
| Onyx | 0.0204 | 0.0237 | 0.0310 |

Both of ours run an explicit canonicalize pass (SPEC.md §5 step 9) whose
job is to force every instance of a repeated groove to that reducer's own
most-common reduction of it, and does so internally by construction —
that internal guarantee is exact. The ~0.6% shown here is *not* that
internal number; it's what an independent external check gets when it
re-derives measure/tick positions from the shipped `(ms, lane)` output,
which reintroduces a small amount of floating-point round-trip noise at
measure boundaries. It's a real, if tiny, measurable property of the
JSON output, reported honestly rather than repeating the (higher, internal)
exact-zero claim. Neither HOPCAT nor Onyx run a canonicalize-style pass at
all, so their non-zero rates reflect the tools' own algorithms making
different choices for the same repeated pattern in different places.

**Intrinsic-difficulty divergence** `D(system) − D(Harmonix)` per tier
(D is a reference-free 0–1 playability scalar — density, stream speed,
chord load, lane breadth, syncopation, backbone retention; see
`python/drum_reducer/intrinsic_difficulty.py`. Positive = harder to play
than Harmonix's own reduction, negative = easier):

| system | hard | medium | easy |
|---|---:|---:|---:|
| ours — model | −0.002 | +0.016 | +0.001 |
| ours — scorecard | −0.057 | +0.011 | +0.039 |
| HOPCAT | −0.033 | −0.013 | −0.129 |
| Onyx | −0.054 | −0.023 | −0.032 |

Ours tracks Harmonix's intended difficulty within a few hundredths of D at
every tier. HOPCAT's Easy reduction is measurably easier to play than
Harmonix's own Easy chart (D divergence −0.129) — consistent with the
note-count and backbone-recall numbers above: its Easy tier reduces well
past Harmonix's own Easy, not just to it.

### Provenance

- **Ours** (both backends): computed live by `tools/eval_quality.py` from
  `data/fixtures/parity_fixture.json` / `scorecard_parity_fixture.json`
  (the package's own `reduce()` output) against the ground truth in
  `data/fixtures/charts/`. Cross-checked to reproduce the documented
  0.1703 / 0.2234 pooled numbers exactly.
- **HOPCAT / Onyx**: both tools are run **live, in-process**, from
  `baselines/hopcat.py` / `baselines/onyx.py` — independent
  reimplementations of each tool's published reduction algorithm (see
  those modules and `baselines/_hopcat_algo.py` / `_onyx_algo.py` for
  attribution and per-function source citations), fed this repo's own
  ms-based chart format directly. This repo does not, and cannot, run the
  original HOPCAT or Onyx tools themselves — the numbers above measure
  this repo's reimplementations, not the upstream projects. The HOPCAT
  reimplementation closely reproduces the original tool's documented
  behavior. All four systems' primary and secondary metrics come from the
  same `tools/eval_quality.py` run and are written to
  `data/reference_scores.tsv`.
- **Onyx honesty note**: Onyx's real reduction algorithm guarantees at
  least one note survives per Overdrive phrase (`ensureODNotes`). This
  repo's chart schema carries no Overdrive-phrase data, so that step is
  always a no-op here — Onyx's reduction in this table is a close but
  approximate stand-in for the real tool, and the missing safeguard means
  these numbers are, if anything, *favorable* to Onyx (a real run would
  have at least as many, likely more, protected notes). See
  `baselines/onyx.py`'s module docstring for the full list of documented
  divergences from the real MIDI-based tool.

Re-run the primary/secondary computation yourself (from the repo root):
```
.venv/bin/python tools/eval_quality.py
```
which regenerates `data/reference_scores.tsv`.

## Determinism: Python and JavaScript are bit-identical

The whole point of this reference implementation is that a re-implementer
in a third language has something exact to target. Both backends, in both
languages, produce **bit-identical note lists** — proven note-for-note on
99 songs (297 song×tier reductions) for the model backend and the same 99
songs for the scorecard backend, plus 3 synthetic edge-case songs covering
the gotcha paths (empty groove measures, a mid-song time-signature change,
a song with no kick/snare at all).

This isn't "we didn't observe a diff" — it's bit-exact **by construction**,
governed by [DETERMINISM_CONTRACT.md](DETERMINISM_CONTRACT.md): fixed
iteration/summation orders, explicit tie-breaks at every point two notes
could compare equal (which happens constantly on repeated grooves), an
integer-only scorecard backend, and a shipped portable `exp` implementation
for the model backend's probabilities (the only place floating-point
non-portability could otherwise creep in).

Run the proof yourself (from the repo root):
```
tools/check_parity.sh
```
which runs both language test suites (Python: `pytest python/tests/`;
JS: `node --test javascript/test/`) and prints one PASS/FAIL summary. Each
suite independently diffs its own `reduce()` against the same frozen
fixtures under `data/fixtures/` — both landing on the identical fixture is
transitively "Python == JavaScript."

## Porting to C / another language

The Python package under `python/drum_reducer/` is the reference; the
JavaScript port under `javascript/src/` is a second, independently-useful
reference that also proves the spec is language-agnostic. To port to a
third language:

1. Read [SPEC.md](SPEC.md) — the 59 features exact, the 9-step decode in
   order, the packed-model binary format, and the scorecard JSON format.
2. Read [DETERMINISM_CONTRACT.md](DETERMINISM_CONTRACT.md) — the fixed
   orderings, tie-breaks, summation rules, and (for the model backend) the
   portable-exp algorithm. This is what makes bit-exactness possible across
   languages with different float/sort semantics; skipping it is the most
   likely way a port looks right on a handful of songs and then silently
   diverges on a repeated-groove tie.
3. Validate against `data/fixtures/parity_fixture.json` and
   `scorecard_parity_fixture.json` — 99 real songs + 3 edge cases, per
   tier, expected `(ms, lane)` output. Expect an exact match: 0 insertions,
   0 deletions, 0 lane differences, note-for-note.
4. If anything diverges, binary-search which decode step diverges first by
   comparing intermediate state (survive set → NMS'd set → relaned set →
   pooled set → deduped set → canonicalized set), not just the final note
   list — SPEC.md §8 has the same advice for the JS port and it generalizes.

## Scorecard auditability

The scorecard backend is a depth-1 refit of the model's survive heads
(collapsed into one piecewise-constant integer point table per feature) and
depth-≤2 decision trees for lane-relaning, both fit against the *same*
training data as the model backend, then quantized to integers so every
downstream decision is exact integer arithmetic — see
`data/scorecard/AUDIT.md` for the generation log and the full per-feature
breakpoint table. Honest numbers, not "read every rule" oversell:

- **Survive**: of the 59 input features, **16–20 are actually used** per
  tier (the rest collapse to a constant 0-point contribution after
  distillation and merging). Most used features reduce to 1–3 breakpoint
  rows after merging adjacent bins with identical point deltas — genuinely
  a glance-and-verify table. A handful of continuous ms/density features
  keep up to **8 rows** — still finite and inspectable, but read as a
  step-function table rather than a single number.
- **Relane**: two decision trees per tier (cymbal, tom), each **≤7 nodes**
  (depth ≤2) — small enough to read as nested `if`/`else` at a glance. See
  `data/scorecard/scorecard_rules.py` for the generated, human-readable
  rendering of every table and tree.

## Rebuild the scorecard on your own charts

The shipped `data/scorecard/scorecard.json` was trained on Harmonix's
official Hard/Medium/Easy reductions in a separate pipeline (see
`data/scorecard/AUDIT.md` for that generation log). You can retrain the
scorecard on your own corpus — any folder of charts matching this repo's
schema (`difficulties.expert` plus human-authored `hard`/`medium`/`easy`
reductions, see [SPEC.md](SPEC.md) §7) — with:

```
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python tools/build_scorecard.py \
    --charts /path/to/your/charts --out /path/to/scorecard.json
```

This fits a fresh depth-1 survive refit + depth-2 relane trees against
your own charts, tunes the (T_tier, NMS-gap) decode knobs on a held-out
split of the same folder, and emits `scorecard.json` +
`scorecard_rules.py` + `AUDIT.md`. `scikit-learn` is needed only for this
build step (`requirements-dev.txt`) — the runtime `drum_reducer` package
(`requirements.txt`: numpy only) never imports it.

## Repo layout

```
python/drum_reducer/     featurize.py, decode.py, backend_model.py,
                          backend_scorecard.py, editrate.py, reduce.py
javascript/src/           mirror of python/drum_reducer, same module split
baselines/                 independent HOPCAT/Onyx reimplementations, for
                            the quality comparison only (not part of reduce())
data/model/                packed-GBM binaries + manifest (model backend)
data/scorecard/             scorecard.json + generated readable rules + AUDIT.md
data/fixtures/               99 real + 3 edge-case sample charts, expected
                              output per backend per tier
data/reference_scores.tsv     the quality table above, machine-readable
tools/build_scorecard.py       fit a scorecard (+ its decode knobs) from
                                any folder of your own charts
tools/check_parity.sh           the "prove Python == JS" entrypoint
tools/eval_quality.py            produces the quality table + reference_scores.tsv
requirements.txt                 runtime dependency (numpy)
requirements-dev.txt              + build-only dependency (scikit-learn, for
                                    tools/build_scorecard.py)
SPEC.md                          byte-exact port spec
DETERMINISM_CONTRACT.md           the cross-language bit-exactness rules
```

## Not in scope

This repo is the reducer core only — no chart-file (`.mid`/`.chart`)
parsing, no visualization, no UI. The input is an already-parsed note list
(see SPEC.md's input schema); a product wrapping this reducer handles
parsing and rendering separately.
