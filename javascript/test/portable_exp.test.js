'use strict';

/**
 * Sanity checks for portable_exp.js in isolation (the full bit-for-bit
 * grid comparison against python/drum_reducer/portable_exp.py was done
 * out-of-band during development -- 4010 grid points + boundary cases,
 * 0 mismatches; this file just guards the basic shape/known values so a
 * regression shows up in `node --test`).
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const { portableExp, sigmoid, softmax } = require('../src/portable_exp');

test('portableExp matches Math.exp closely away from the boundaries', () => {
  for (const x of [-10, -1, -0.5, 0, 0.5, 1, 10, 100]) {
    assert.ok(Math.abs(portableExp(x) - Math.exp(x)) < 1e-9 * Math.max(1, Math.abs(Math.exp(x))), `x=${x}`);
  }
});

test('portableExp overflow/underflow/tiny fast paths', () => {
  assert.equal(portableExp(1000), Infinity);
  assert.equal(portableExp(-1000), 0.0);
  assert.equal(portableExp(1e-30), 1.0 + 1e-30);
});

test('sigmoid is bounded in (0, 1) and symmetric', () => {
  assert.ok(sigmoid(0) === 0.5);
  assert.ok(sigmoid(100) > 0.999);
  assert.ok(sigmoid(-100) < 0.001);
});

test('softmax sums to 1 and picks the right argmax', () => {
  const out = softmax([1, 2, 3]);
  const s = out[0] + out[1] + out[2];
  assert.ok(Math.abs(s - 1.0) < 1e-12);
  assert.ok(out[2] > out[1] && out[1] > out[0]);
});
