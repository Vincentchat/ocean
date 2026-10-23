// CPU FFT correctness: radix-2 2-D FFT vs a naive DFT, and the resumable pieces vs fft2d.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createFftPlan, fft2d, ifft2d, naiveDft2d, fftRows, fftColumnsBitReverse, fftColumnsStage, fft1d,
} from '../cpu-fft.ts';

function randomGrid(n: number, seed: number): { re: Float64Array; im: Float64Array } {
  let s = seed >>> 0;
  const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296) * 2 - 1;
  const re = new Float64Array(n * n), im = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) { re[i] = rnd(); im[i] = rnd(); }
  return { re, im };
}

function maxDiff(a: Float64Array, b: Float64Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

test('2-D inverse FFT matches the naive DFT (exp(+i), unnormalised) for n = 2..32', () => {
  for (const n of [2, 4, 8, 16, 32]) {
    const { re, im } = randomGrid(n, 17 + n);
    const ref = naiveDft2d(n, re, im, 1);
    const plan = createFftPlan(n);
    const r = re.slice(), i = im.slice();
    ifft2d(plan, r, i);
    const tol = 1e-12 * n * n;
    assert.ok(maxDiff(r, ref.re) < tol, `re n=${n}: ${maxDiff(r, ref.re)}`);
    assert.ok(maxDiff(i, ref.im) < tol, `im n=${n}: ${maxDiff(i, ref.im)}`);
  }
});

test('forward FFT matches the naive DFT and forward(inverse(x)) = n^2 x', () => {
  const n = 16;
  const { re, im } = randomGrid(n, 99);
  const ref = naiveDft2d(n, re, im, -1);
  const plan = createFftPlan(n);
  const r = re.slice(), i = im.slice();
  fft2d(plan, r, i, -1);
  assert.ok(maxDiff(r, ref.re) < 1e-10);
  assert.ok(maxDiff(i, ref.im) < 1e-10);
  fft2d(plan, r, i, 1);
  for (let k = 0; k < n * n; k++) { r[k] /= n * n; i[k] /= n * n; }
  assert.ok(maxDiff(r, re) < 1e-12);
  assert.ok(maxDiff(i, im) < 1e-12);
});

test('1-D FFT with stride equals the column of the 2-D transform', () => {
  const n = 8, plan = createFftPlan(n);
  const { re, im } = randomGrid(n, 5);
  const naive = new Float64Array(n), naiveI = new Float64Array(n);
  for (let m = 0; m < n; m++) for (let x = 0; x < n; x++) {
    const a = (2 * Math.PI * m * x) / n, c = Math.cos(a), s = Math.sin(a);
    const vr = re[x * n + 3], vi = im[x * n + 3];
    naive[m] += vr * c - vi * s; naiveI[m] += vr * s + vi * c;
  }
  const r = re.slice(), i = im.slice();
  fft1d(plan, r, i, 3, n, 1); // column 3
  for (let m = 0; m < n; m++) {
    assert.ok(Math.abs(r[m * n + 3] - naive[m]) < 1e-12);
    assert.ok(Math.abs(i[m * n + 3] - naiveI[m]) < 1e-12);
  }
});

test('resumable pieces (rows in chunks, bit-reverse, stages) equal fft2d exactly', () => {
  const n = 64, plan = createFftPlan(n);
  const { re, im } = randomGrid(n, 3);
  const a = re.slice(), ai = im.slice();
  ifft2d(plan, a, ai);
  const b = re.slice(), bi = im.slice();
  for (let r = 0; r < n; r += 7) fftRows(plan, b, bi, r, Math.min(n, r + 7), 1);
  fftColumnsBitReverse(plan, b, bi);
  for (let span = 2; span <= n; span <<= 1) fftColumnsStage(plan, b, bi, span, 1);
  assert.deepEqual(b, a);
  assert.deepEqual(bi, ai);
});

test('createFftPlan rejects non powers of two', () => {
  assert.throws(() => createFftPlan(48));
  assert.throws(() => createFftPlan(1));
});
