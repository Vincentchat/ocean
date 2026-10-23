// Spectrum: faithful port of the WebGL reference (src/spectral-ocean.js), seeding, layout, Hs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  CASCADES, REFERENCE_SIZE, OMEGA_UNIT, LOOP_PERIOD, spectralDensity, spectrumScalars, generateRawRows,
  gaussianField, generateCascade, expectedHs, fetchForPeakWavelength, peakWavelength, peakPeriod,
  resampleCentred, omegaIndices, REFERENCE_AMPLITUDE,
} from '../spectrum.ts';
import type { SpectrumParams } from '../spectrum.ts';
import { PRESETS, PRESET_NAMES } from '../presets/index.ts';

const REF_G = 9.81; // the reference's gravity (src/spectral-ocean.js OCEAN_PARAMETERS)
const refUrl = new URL('../../src/spectral-ocean.js', import.meta.url);
const haveReference = existsSync(fileURLToPath(refUrl));

function referenceParams(g: number, amplitude = 1): SpectrumParams {
  return {
    windSpeed: 15, windDirection: -1, fetch: fetchForPeakWavelength(15, 47, g), gamma: 3.3,
    spread: 1.57, standing: 0.13, choppiness: 1.5, amplitude,
  };
}

test('windy preset = reference parameters (U 15 m/s, peak 47 m, gamma 3.3, spread 1.57, standing .13, chop 1.5, amp 1.578)', () => {
  const p = PRESETS.windy.params;
  assert.equal(p.windSpeed, 15);
  assert.equal(p.windDirection, -1);
  assert.ok(Math.abs(peakWavelength(p.windSpeed, p.fetch, 9.80665) - 47) < 1e-3);
  assert.equal(p.gamma, 3.3); assert.equal(p.spread, 1.57); assert.equal(p.standing, 0.13); assert.equal(p.choppiness, 1.5);
  assert.ok(Math.abs(p.amplitude - 1.5783) < 1e-3 && p.amplitude === REFERENCE_AMPLITUDE);
});

test('cascades: 1024 / 96 / 9 m with the reference band limits (pi*256/L/1.5)', () => {
  assert.deepEqual(CASCADES.map((c) => c.domain), [1024, 96, 9]);
  assert.ok(Math.abs(CASCADES[0].upper - (Math.PI * 256) / 1024 / 1.5) < 1e-12);
  assert.equal(CASCADES[1].lower, CASCADES[0].upper);
  assert.equal(CASCADES[2].lower, CASCADES[1].upper);
  assert.equal(LOOP_PERIOD, 8192);
});

test('optimised row generator equals spectralDensity (reference formula) to 1e-12', () => {
  const g = 9.80665, p = PRESETS.storm.params;
  const sc = spectrumScalars(p, g, { g: 0, U: 0, dirGrid: 0, wp: 0 });
  const n = REFERENCE_SIZE;
  const xi = new Float64Array(2 * n * n).fill(1); // xi = 1 + 1i so raw = sqrt(P/2) * amp
  for (let c = 0; c < 3; c++) {
    const raw = new Float64Array(2 * n * n);
    generateRawRows(p, sc, CASCADES[c], xi, raw, 0, n);
    const dk = (2 * Math.PI) / CASCADES[c].domain;
    let maxRel = 0, peak = 0;
    for (let y = 1; y < n; y += 3) for (let x = 1; x < n; x += 3) {
      const kx = Math.fround((x - 128) * dk), kz = Math.fround((y - 128) * dk);
      const P = spectralDensity(kx, kz, CASCADES[c].domain, CASCADES[c].lower, CASCADES[c].upper, g, p.windSpeed, -p.windDirection, sc.wp, p.gamma, p.spread, p.standing);
      const want = Math.sqrt(P * 0.5) * p.amplitude, got = raw[2 * (y * n + x)];
      peak = Math.max(peak, want);
      maxRel = Math.max(maxRel, Math.abs(got - want));
    }
    assert.ok(maxRel <= 1e-12 * Math.max(peak, 1e-30) + 1e-300, `cascade ${c}: ${maxRel} vs peak ${peak}`);
  }
});

test('seed 1 reproduces the WebGL reference h0 and dispersion to float32 precision at the reference g = 9.81', { skip: !haveReference && 'reference src/ not present' }, async () => {
  const ref = await import(pathToFileURL(fileURLToPath(refUrl)).href);
  const spectra = ref.generateSpectra(256, [1024, 96, 9], { ...ref.OCEAN_PARAMETERS, gravity: REF_G });
  const params = referenceParams(REF_G);
  for (let c = 0; c < 3; c++) {
    const mine = generateCascade(params, c, 1, REF_G);
    const data: Float32Array = spectra.cascades[c].data;
    const freq: Float32Array = spectra.cascades[c].frequency;
    let peak = 0, maxDiff = 0, maxW = 0;
    const n = 256;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const i = y * n + x, j = ((n - y) % n) * n + ((n - x) % n);
      peak = Math.max(peak, Math.abs(data[4 * i]), Math.abs(data[4 * i + 1]));
      // h0(k)
      maxDiff = Math.max(maxDiff, Math.abs(mine.h0[4 * i] - data[4 * i]), Math.abs(mine.h0[4 * i + 1] - data[4 * i + 1]));
      // conj(h0(-k))
      maxDiff = Math.max(maxDiff, Math.abs(mine.h0[4 * i + 2] - data[4 * j]), Math.abs(mine.h0[4 * i + 3] + data[4 * j + 1]));
      maxW = Math.max(maxW, Math.abs(mine.omegaIndex[i] * OMEGA_UNIT - freq[i]));
    }
    assert.ok(maxDiff < 2e-6 * peak, `cascade ${c}: h0 differs by ${maxDiff} (peak ${peak})`);
    assert.ok(maxW < 1e-5, `cascade ${c}: omega differs by ${maxW}`);
  }
  // The reference's expected variance (power * (1 + (1 - standing)^2)) gives the same Hs.
  const refHs = 4 * Math.sqrt(spectra.expectedVariance);
  const myHs = expectedHs(params, REF_G);
  assert.ok(Math.abs(myHs - refHs) / refHs < 0.01, `Hs ${myHs} vs reference ${refHs}`);
});

test('dispersion is quantised to the 8192 s loop period (integer omega index)', () => {
  const om = omegaIndices(CASCADES[1], 9.80665, new Float32Array(256 * 256));
  for (let i = 0; i < om.length; i += 97) {
    assert.equal(om[i], Math.round(om[i]));
    // phase after one loop period is a whole number of turns
    const turns = (om[i] * OMEGA_UNIT * LOOP_PERIOD) / (2 * Math.PI);
    assert.ok(Math.abs(turns - Math.round(turns)) < 1e-6);
  }
});

test('h0 layout: [h0(k), conj(h0(-k))], Nyquist line zero, Hermitian pairing', () => {
  const { h0 } = generateCascade(PRESETS.breeze.params, 0, 1, 9.80665);
  const n = 256;
  for (let x = 0; x < n; x++) for (let c = 0; c < 4; c++) { assert.ok(h0[4 * x + c] === 0 && h0[4 * (x * n) + c] === 0); }
  for (let y = 1; y < n; y += 13) for (let x = 1; x < n; x += 11) {
    const i = y * n + x, j = (n - y) * n + (n - x);
    assert.equal(h0[4 * i + 2], h0[4 * j]);
    assert.equal(h0[4 * i + 3], -h0[4 * j + 1]);
  }
});

test('size-N grids are exact centred subsets of the 256 master (tier-independent waves)', () => {
  const { h0 } = generateCascade(PRESETS.breeze.params, 0, 1, 9.80665);
  for (const N of [64, 128]) {
    const sub = resampleCentred(h0, new Float32Array(N * N * 4), N, 4, true);
    for (let y = 1; y < N; y += 5) for (let x = 1; x < N; x += 7) {
      const sx = x - N / 2 + 128, sy = y - N / 2 + 128;
      for (let c = 0; c < 4; c++) assert.equal(sub[4 * (y * N + x) + c], h0[4 * (sy * 256 + sx) + c]);
    }
  }
  const big = resampleCentred(h0, new Float32Array(512 * 512 * 4), 512, 4, true);
  assert.equal(big[4 * (256 * 512 + 256 + 3)], h0[4 * (128 * 256 + 128 + 3)]);
  assert.equal(big[4 * (10 * 512 + 10)], 0); // zero padding beyond the 256 band
});

test('seeds: deterministic, and different seeds give different draws', () => {
  const a = gaussianField(1, 0, new Float64Array(2 * 256 * 256));
  const b = gaussianField(1, 0, new Float64Array(2 * 256 * 256));
  const c = gaussianField(2, 0, new Float64Array(2 * 256 * 256));
  assert.deepEqual(a, b);
  assert.notDeepEqual(a, c);
});

test('preset target table: expected Hs and peak period within 2 % of the table, ordered calm < breeze < windy < storm', () => {
  const g = 9.80665;
  let prev = 0;
  for (const name of PRESET_NAMES) {
    const pr = PRESETS[name];
    const hs = expectedHs(pr.params, g);
    const tp = peakPeriod(pr.params.windSpeed, pr.params.fetch, g);
    assert.ok(Math.abs(hs - pr.target.hs) / pr.target.hs < 0.02, `${name}: Hs ${hs.toFixed(3)} vs ${pr.target.hs}`);
    assert.ok(Math.abs(tp - pr.target.peakPeriod) / pr.target.peakPeriod < 0.02, `${name}: Tp ${tp} vs ${pr.target.peakPeriod}`);
    assert.ok(hs > prev, `${name} must be rougher than the previous preset`);
    prev = hs;
  }
});
