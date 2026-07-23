'use strict';

/**
 * Portable exp() -- mirrors python/drum_reducer/portable_exp.py bit-for-bit
 * (DETERMINISM_CONTRACT.md §3). Same fdlibm-style range-reduction +
 * degree-5 minimax polynomial, same constants (fdlibm's own, public
 * domain), same operation order. Operates on scalars (JS numbers are
 * already float64, so this needs no numpy-style array plumbing -- callers
 * loop explicitly where the contract requires a fixed summation order).
 *
 * The `scalbn` step (multiply by 2^k without ever materializing 2^k, which
 * would overflow for k~1024 even though the true product is finite) is
 * done by splitting k into two halves and multiplying twice, exactly as
 * the Python docstring prescribes for a JS port -- each half's power of
 * two is built directly from IEEE-754 bits via DataView, so the multiply
 * is exact (no non-portable Math.pow(2, k) rounding).
 */

const LN2_HI = 6.93147180369123816490e-01;
const LN2_LO = 1.90821492927058770002e-10;
const INVLN2 = 1.44269504088896338700e+00;
const HALF_LN2 = 3.46573590279972654709e-01; // 0.5 * ln2, the range-reduction cutoff

const P1 = 1.66666666666666019037e-01;
const P2 = -2.77777777770155933842e-03;
const P3 = 6.61375632143793436117e-05;
const P4 = -1.65339022054652515390e-06;
const P5 = 4.13813679705723846039e-08;

const OVERFLOW = 7.09782712893383973096e+02;
const UNDERFLOW = -7.45133219101941108420e+02;
const TINY = Math.pow(2, -28);

const _pow2buf = new ArrayBuffer(8);
const _pow2dv = new DataView(_pow2buf);

/** 2^e as an exact float64, for -1022 <= e <= 1023 (normal double range). */
function pow2exact(e) {
  _pow2dv.setBigUint64(0, BigInt(e + 1023) << 52n, false);
  return _pow2dv.getFloat64(0, false);
}

/**
 * y * 2^k, matching np.ldexp: split k into two halves (each with
 * magnitude well inside the normal exponent range for the |x| bounds this
 * module ever sees) and multiply twice, so no intermediate 2^k
 * materializes and overflows on its own.
 */
function scalbn(y, k) {
  const k1 = Math.trunc(k / 2);
  const k2 = k - k1;
  return y * pow2exact(k1) * pow2exact(k2);
}

/** exp(x), fdlibm-style. x and the return value are plain JS numbers (float64). */
function portableExp(x) {
  if (x > OVERFLOW) return Infinity;
  if (x < UNDERFLOW) return 0.0;
  if (Math.abs(x) < TINY) return 1.0 + x;

  const needReduce = Math.abs(x) > HALF_LN2;
  let k = 0.0;
  if (needReduce) {
    k = x >= 0 ? Math.floor(INVLN2 * x + 0.5) : Math.ceil(INVLN2 * x - 0.5);
  }
  const hi = needReduce ? x - k * LN2_HI : x;
  const lo = needReduce ? k * LN2_LO : 0.0;
  const r = hi - lo;

  const t = r * r;
  const c = r - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
  let y = 1.0 + (r * c / (2.0 - c) - lo + hi);
  y = scalbn(y, k);
  return y;
}

/** 1 / (1 + exp(-x)), via portableExp. */
function sigmoid(x) {
  return 1.0 / (1.0 + portableExp(-x));
}

/**
 * softmax over a single row of raw scores. Subtracts the row max first
 * (exact IEEE subtraction), then sums portableExp(shifted) with an
 * explicit ascending-index loop (DETERMINISM_CONTRACT.md §1's fixed
 * summation order -- no Array.reduce/divide-and-conquer).
 */
function softmax(xs) {
  let m = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] > m) m = xs[i];
  }
  const exps = new Array(xs.length);
  for (let i = 0; i < xs.length; i++) {
    exps[i] = portableExp(xs[i] - m);
  }
  let s = 0.0;
  for (let i = 0; i < xs.length; i++) { // explicit left-to-right, class-index order
    s += exps[i];
  }
  const out = new Array(xs.length);
  for (let i = 0; i < xs.length; i++) {
    out[i] = exps[i] / s;
  }
  return out;
}

module.exports = { portableExp, sigmoid, softmax, scalbn, pow2exact };
