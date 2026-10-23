// Pure parts of the GPU FFT tiers (three/fft-tiers.ts): tier invariance (every tier draws the master components
// bit-identically, only dropping centred sub-blocks), the exact field evaluator vs a naive DFT, and the CPU-query
// parity of every tier's drawn surface against the tier-independent core query (cubic c0 / c1).
import test from 'node:test';
import assert from 'node:assert/strict';
import { CASCADES, REFERENCE_SIZE } from '../spectrum.ts';
import {
  FFT_TIERS, TIER_NAMES, CAPILLARY_CASCADE, masterH0Subset, masterOmegaSubset, droppedSlopeVariance, evaluateFields,
  sampleChained, generateCapillary,
} from '../three/fft-tiers.ts';
import { createOcean } from '../index.ts';
import { createFlatShape } from '../stubs/impl.ts';

const shape = createFlatShape();

test('every tier draws the master components bit-identically (centred sub-blocks only)', () => {
  const ocean = createOcean({ shape, preset: 'windy', seed: 1 });
  const st = ocean.getState();
  const m = REFERENCE_SIZE;
  for (const t of TIER_NAMES) {
    for (let c = 0; c < 3; c++) {
      const n = FFT_TIERS[t].sizes[c];
      if (c === 0) assert.equal(n, 256, `${t}: c0 must stay 256 (it carries the swell)`);
      const master = st.slots[st.toSlot].h0[c];
      const sub = masterH0Subset(master, n), om = masterOmegaSubset(st.omegaIndex[c], n);
      for (let y = 1; y < n; y++) for (let x = 1; x < n; x++) {
        const i = y * n + x, j = (y - n / 2 + m / 2) * m + (x - n / 2 + m / 2);
        assert.equal(sub[2 * i], master[4 * j]); assert.equal(sub[2 * i + 1], master[4 * j + 1]);
        assert.equal(om[i], st.omegaIndex[c][j]);
      }
    }
  }
  assert.equal(CAPILLARY_CASCADE.lower, CASCADES[2].upper, 'capillary band starts where the master band fades out');
});

test('evaluateFields matches a naive DFT (n = 16)', () => {
  const n = 16, L = 9;
  let s = 7;
  const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296 - 0.5);
  const h0 = new Float32Array(2 * n * n).map(() => rnd());
  for (let i = 0; i < n; i++) { h0[2 * i] = h0[2 * i + 1] = 0; h0[2 * i * n] = h0[2 * i * n + 1] = 0; }
  const om = new Float32Array(n * n).map((_, i) => { const u = (i % n) - n / 2, v = Math.floor(i / n) - n / 2; return Math.round(40 * Math.hypot(u, v)); });
  const t = 1234.5678, chop = 1.3;
  const f = evaluateFields(h0, h0, 1, om, n, L, t, chop);
  const dk = (2 * Math.PI) / L;
  for (const [x, y] of [[0, 0], [3, 5], [15, 9]]) {
    let H = 0, E = 0, N = 0;
    for (let v = 0; v < n; v++) for (let u = 0; u < n; u++) {
      const i = v * n + u, j = ((n - v) % n) * n + ((n - u) % n);
      const kx = Math.fround((u - n / 2) * dk), kz = Math.fround((v - n / 2) * dk), kl = Math.hypot(kx, kz) + 1e-4;
      const ph = (2 * Math.PI / 8192) * ((om[i] * t) % 8192);
      const hr = h0[2 * i], hi = h0[2 * i + 1], mr = h0[2 * j], mi = -h0[2 * j + 1];
      const htr = hr * Math.cos(ph) + hi * Math.sin(ph) + mr * Math.cos(ph) - mi * Math.sin(ph);
      const hti = hi * Math.cos(ph) - hr * Math.sin(ph) + mi * Math.cos(ph) + mr * Math.sin(ph);
      const a = (2 * Math.PI * ((u - n / 2) * x + (v - n / 2) * y)) / n, c = Math.cos(a), sn = Math.sin(a);
      H += htr * c - hti * sn;
      E += (kx / kl) * (hti * c + htr * sn);
      N += (kz / kl) * (hti * c + htr * sn);
    }
    const i = y * n + x;
    assert.ok(Math.abs(f[3 * i] + H) < 1e-9, `up ${f[3 * i]} vs ${-H}`);
    assert.ok(Math.abs(f[3 * i + 1] - chop * E) < 1e-9, 'east');
    assert.ok(Math.abs(f[3 * i + 2] + chop * N) < 1e-9, 'north');
  }
});

test('capillary band: gravity-capillary dispersion, finite spectrum', () => {
  const ocean = createOcean({ shape, preset: 'windy', seed: 1 });
  const st = ocean.getState();
  const cap = generateCapillary(st.slots[st.toSlot].params, 1, st.gravity, 64);
  assert.ok(cap.h0.every(Number.isFinite));
  // corner cell k = 32 sqrt(2) dk: w^2 = g k + S k^3
  const dk = (2 * Math.PI) / CAPILLARY_CASCADE.domain, k = Math.hypot(Math.fround(-32 * dk), Math.fround(-32 * dk));
  const w = cap.omega[0] * (2 * Math.PI / 8192);
  assert.ok(Math.abs(w - Math.sqrt(st.gravity * (k + 1e-4) + 7.4e-5 * k ** 3)) < 1e-3);
});

const CUBIC_QUERY = [
  { size: 256, snapshotDt: 0.6, filter: 'cubic' as const }, { size: 256, snapshotDt: 0.25, filter: 'cubic' as const },
  { size: 128, snapshotDt: 0.15, filter: 'bilinear' as const },
];

/** Drawn surface of tier t (master subsets, Catmull-Rom c0 / c1 like the GPU near the camera) vs the core query. */
export function tierParity(tier: keyof typeof FFT_TIERS, preset: string, count = 1500, sizes?: readonly number[]) {
  const ocean = createOcean({ shape, preset: preset as any, seed: 1, query: CUBIC_QUERY });
  const st = ocean.getState();
  const T = 12, chop = ocean.choppinessAt(T), hs = ocean.expectedHs();
  const sz = sizes ?? FFT_TIERS[tier].sizes;
  const fields = [0, 1, 2].map((c) => {
    const n = sz[c], h0 = masterH0Subset(st.slots[st.toSlot].h0[c], n);
    return evaluateFields(h0, h0, 1, masterOmegaSubset(st.omegaIndex[c], n), n, CASCADES[c].domain, T, chop);
  });
  let s = 99;
  const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
  const out = new Float64Array(3), tmp = new Float64Array(3), errs: number[] = [];
  const rs = { east: 0, north: 0, up: 0, vEast: 0, vNorth: 0, vUp: 0 };
  let other = 0;
  for (let i = 0; i < count; i++) {
    const e = rnd() * 2000 - 1000, nn = rnd() * 2000 - 1000;
    sampleChained(fields, sz, CASCADES, e, nn, [true, true, false], out, tmp);
    ocean.invertLocal(e + out[1], nn + out[2], T, rs);
    if (Math.hypot(rs.east - e, rs.north - nn) > 0.3) { other++; continue; } // folded crest: another sheet
    errs.push((Math.abs(rs.up - out[0]) / hs) * 100);
  }
  errs.sort((a, b) => a - b);
  const rms = Math.sqrt(errs.reduce((a, b) => a + b * b, 0) / errs.length);
  let lost = 0;
  for (let c = 1; c < 3; c++) lost += droppedSlopeVariance(st.slots[st.toSlot].h0[c], sz[c], CASCADES[c].domain);
  return { rms, p99: errs[Math.floor(errs.length * 0.99)], p999: errs[Math.floor(errs.length * 0.999)], max: errs[errs.length - 1], hs, other: other / count, lostSlope: lost };
}

test('drawn surface vs the tier-independent CPU query (cubic c0/c1): p99.9 < 2 % Hs on every tier', () => {
  for (const preset of ['breeze', 'windy', 'storm']) {
    for (const tier of ['low', 'medium'] as const) { // high / ultra draw the medium master content
      const r = tierParity(tier, preset);
      console.log(`  parity ${preset} ${tier} ${FFT_TIERS[tier].sizes.join('/')}: rms ${r.rms.toFixed(3)} %Hs, p99 ${r.p99.toFixed(3)}, p99.9 ${r.p999.toFixed(3)}, max ${r.max.toFixed(3)}, other sheet ${(r.other * 100).toFixed(1)} %, unresolved slope var ${r.lostSlope.toExponential(2)} (Hs ${r.hs.toFixed(2)} m)`);
      assert.ok(r.p999 < 2, `${preset} ${tier} p99.9 ${r.p999}`);
    }
  }
});
