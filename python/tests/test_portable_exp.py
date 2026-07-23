"""
Unit tests for portable_exp.py: agreement with math.exp (documents max ULP
so the JS port has a numeric target to match), overflow/underflow/tiny-x
fast paths, and the scalbn-overflow edge case near the overflow boundary
(the np.exp2 vs np.ldexp bug this module's implementation had to avoid --
see the comment in portable_exp.py's reconstruction step).
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drum_reducer.portable_exp import portable_exp, sigmoid, softmax  # noqa: E402


def _safe_math_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


def test_matches_math_exp_dense_grid():
    xs = np.linspace(-745, 709.7, 400001)
    mine = portable_exp(xs)
    ref = np.array([_safe_math_exp(x) for x in xs])
    finite = np.isfinite(mine) & np.isfinite(ref) & (ref != 0)
    rel_err = np.abs(mine[finite] - ref[finite]) / np.abs(ref[finite])
    max_rel = float(rel_err.max())
    # Documented bound: fdlibm's degree-5 minimax polynomial is good to a
    # couple ULP of double precision (~2.2e-16 per ULP) across the whole
    # finite range, including the range-reduction seams.
    assert max_rel < 1e-13, f"max relative error {max_rel:.3e} vs math.exp exceeds the documented bound"
    print(f"\n[portable_exp] max relative error vs math.exp over {len(xs)} points: {max_rel:.3e}")


def test_overflow_underflow_tiny():
    assert portable_exp(1000.0) == float("inf")
    assert portable_exp(-1000.0) == 0.0
    assert portable_exp(0.0) == 1.0
    tiny = 1e-30
    assert abs(portable_exp(tiny) - (1.0 + tiny)) < 1e-40


def test_near_overflow_boundary_no_premature_inf():
    # x=709.7 is well below the ~709.78 overflow cutoff -- the true value is
    # a large but FINITE double (~1.65e308). A naive y*2**k reconstruction
    # (k~1024) overflows here even though the correct answer is finite; this
    # is the bug portable_exp.py's np.ldexp-based scalbn avoids.
    for x in (700.0, 709.0, 709.7, 709.78):
        v = portable_exp(x)
        assert np.isfinite(v), f"portable_exp({x}) should be finite, got {v}"
        assert abs(v - math.exp(x)) / math.exp(x) < 1e-12


def test_sigmoid_matches_reference():
    xs = np.linspace(-30, 30, 10001)
    mine = sigmoid(xs)
    ref = 1.0 / (1.0 + np.array([_safe_math_exp(-x) for x in xs]))
    assert np.max(np.abs(mine - ref)) < 1e-12


def test_softmax_matches_reference():
    rng = np.random.RandomState(0)
    x = rng.uniform(-20, 20, size=(500, 4))
    mine = softmax(x)
    m = x.max(axis=1, keepdims=True)
    e = np.exp(x - m)
    ref = e / e.sum(axis=1, keepdims=True)
    assert np.max(np.abs(mine - ref)) < 1e-12
    # rows sum to 1
    assert np.allclose(mine.sum(axis=1), 1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
