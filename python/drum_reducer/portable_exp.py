"""
Portable exp() -- one fixed algorithm, meant to be implemented bit-identically
in Python and JavaScript (DETERMINISM_CONTRACT.md §3). Everything else in the
model backend (tree traversal, bin search, add/sub/mul/cmp) is already
IEEE-754 correctly-rounded and therefore already portable; `exp` is the one
primitive whose result is allowed to differ between two conformant
implementations (libm doesn't guarantee correctly-rounded transcendentals),
so we ship our own instead of calling `math.exp`/`Math.exp`.

Algorithm: the classic fdlibm/msun `__ieee754_exp` range-reduction +
degree-5 minimax polynomial (the same algorithm many libm's `exp` are
themselves built on, but pinned here so both languages run the identical
sequence of operations instead of whatever the platform libm happens to do):

  1. Handle overflow (x > ~709.78 -> +inf), underflow (x < ~-745.13 -> 0.0),
     and tiny |x| (< 2^-28 -> 1+x, since exp(x)-1 ~ x there and the
     polynomial path would just add rounding noise) as fast paths.
  2. Range-reduce x = k*ln2 + r with |r| <= 0.5*ln2, k = round(x / ln2),
     using a hi/lo split of ln2 (ln2 is not exactly representable in
     double, so a naive `x - k*ln2` loses precision near the boundary --
     the hi/lo (Cody-Waite) split recovers it).
  3. Evaluate a degree-5 minimax polynomial for (exp(r)-1)/r-style
     correction on the REDUCED r, then reconstruct exp(x) = 2^k * exp(r).
     Multiplying by 2^k (a power of two) is an exact IEEE-754 operation in
     principle (it only touches the exponent bits, not the mantissa) --
     but a NAIVE `y * Math.pow(2, k)` (or `y * 2**k`) is NOT safe: k can be
     up to ~1024 for x near the overflow boundary (x~709.7), and 2^1024
     overflows to +inf on its own even though the true product y*2^k is a
     finite ~1.8e308. This module uses `np.ldexp` (C's `scalbn`/`ldexp`
     family), which scales the mantissa's exponent field directly without
     ever materializing 2^k. A JS port must do the same (e.g. split k into
     two halves and multiply twice, or use a DataView bit-twiddle) --
     not a plain `Math.pow(2, k)` multiply.

Ported to operate on numpy arrays (and scalar floats) since backend_model.py
calls it on whole feature/score batches; the JS port operates element-wise
on typed arrays with the exact same five-line polynomial.

Constants are fdlibm's own (public domain, Sun Microsystems 1993) --
copied as literals so this module has zero non-numpy dependencies.
"""

import numpy as np

_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_INVLN2 = 1.44269504088896338700e+00
_HALF_LN2 = 3.46573590279972654709e-01  # 0.5 * ln2, the range-reduction cutoff

_P1 = 1.66666666666666019037e-01
_P2 = -2.77777777770155933842e-03
_P3 = 6.61375632143793436117e-05
_P4 = -1.65339022054652515390e-06
_P5 = 4.13813679705723846039e-08

_OVERFLOW = 7.09782712893383973096e+02
_UNDERFLOW = -7.45133219101941108420e+02
_TINY = 2.0 ** -28


def portable_exp(x):
    """exp(x), fdlibm-style. Accepts a python float or numpy array; returns
    the same shape (scalar float64 in, scalar float64 out)."""
    xf = np.asarray(x, dtype=np.float64)
    scalar = xf.ndim == 0
    xf = np.atleast_1d(xf)
    out = np.empty_like(xf)

    overflow = xf > _OVERFLOW
    underflow = xf < _UNDERFLOW
    tiny = np.abs(xf) < _TINY
    normal = ~(overflow | underflow | tiny)

    out[overflow] = np.inf
    out[underflow] = 0.0
    out[tiny] = 1.0 + xf[tiny]

    xn = xf[normal]
    if xn.size:
        need_reduce = np.abs(xn) > _HALF_LN2
        k = np.where(xn >= 0, np.floor(_INVLN2 * xn + 0.5), np.ceil(_INVLN2 * xn - 0.5))
        k = np.where(need_reduce, k, 0.0)
        hi = np.where(need_reduce, xn - k * _LN2_HI, xn)
        lo = np.where(need_reduce, k * _LN2_LO, 0.0)
        r = hi - lo

        t = r * r
        c = r - t * (_P1 + t * (_P2 + t * (_P3 + t * (_P4 + t * _P5))))
        y = 1.0 + (r * c / (2.0 - c) - lo + hi)
        # scalbn(y, k): np.ldexp scales by 2^k in one step without ever
        # materializing 2^k as its own (possibly-overflowing) intermediate
        # value -- y*np.exp2(k) would overflow to inf for k~1024 even when
        # y is small enough that the true product is a finite ~1.8e308.
        y = np.ldexp(y, k.astype(np.int64))
        out[normal] = y

    return out[0] if scalar else out.reshape(xf.shape)


def sigmoid(x):
    """1 / (1 + exp(-x)), via portable_exp."""
    xf = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + portable_exp(-xf))


def softmax(x):
    """Row-wise softmax over the last axis. Subtracts the row max first
    (standard numerical-stability shift, exact under IEEE arithmetic since
    it's a plain subtraction), then sums portable_exp(shifted) in ascending
    class-index order (DETERMINISM_CONTRACT.md §1's fixed summation-order
    rule -- numpy's axis sum already accumulates in index order for a
    contiguous array, so no extra work is needed here, but a JS port must
    sum its classes with an explicit `for (c = 0; c < n_classes; c++)` loop,
    not e.g. a divide-and-conquer reduction)."""
    xf = np.asarray(x, dtype=np.float64)
    m = np.max(xf, axis=-1, keepdims=True)
    e = portable_exp(xf - m)
    s = np.sum(e, axis=-1, keepdims=True)
    return e / s
