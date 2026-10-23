// Pure parts of the GPU FFT tiers (three/fft-tiers.ts + spectrum.ts tier helpers): band contiguity,
// bit-identity of the sized generator at 256, the exact field evaluator vs a naive DFT, and the CPU-query
// parity of every tier's drawn surface (printed; asserted only for the reference tier).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CASCADES, REFERENCE_SIZE, bandCascades, gaussianField, gaussianFieldSized, generateRawRows, generateRawRowsSized,
  packInitialSpectrum, packInitialSpectrumSized, spectrumScalars,
} from '../spectrum.ts';
import { PRESETS } from '../presets/index.ts';
import {
  FFT_TIERS, TIER_NAMES, tierCascades, generateTierSpectrum, tierOmega, evaluateFields, sampleChained, stripMasterH0,
} from '../three/fft-tiers.ts';
import { createOcean } from '../index.ts';
import { createFlatShape } from '../stubs/impl.ts';

const g = 9.80665;

test('bandCascades(256) is the reference layout; bands are contiguous at every tier size', () => {
  const ref = bandCascades(256);
  for (let c = 0; c < 3; c++) {
    assert.equal(ref[c].upper, CASCADES[c].upper);
    assert.equal(ref[c].lower, CASCADES[c].lower);
    assert.equal(ref[c].offset, CASCADES[c].offset);
  }
  for (const t of TIER_NAMES) {
    const cs = tierCascades(t);
    for (let c = 1; c < cs.length; c++) assert.equal(cs[c].lower, cs[c - 1].upper, `${t} c${c}`);
    // every band ends at (or below) its grid's Nyquist: nothing is cut by the grid
    for (const c of cs) assert.ok(c.upper * 1.5 <= (Math.PI * FFT_TIERS[t].size) / c.domain + 1e-9);
  }
});

test('sized generator at 256 is bit-identical to the master generator', () => {
  const p = PRESETS.storm.params;
  const sc = spectrumScalars(p, g, { g: 0, U: 0, dirGrid: 0, wp: 0 });
  const n = REFERENCE_SIZE;
  for (let c = 0; c < 3; c++) {
    const xi = gaussianField(1, c, new Float64Array(2 * n * n));
    const xs = gaussianFieldSized(1, c, n, new Float64Array(2 * n * n), xi);
    assert.deepEqual(xs, xi);
    const a = new Float64Array(2 * n * n), b = new Float64Array(2 * n * n);
    generateRawRows(p, sc, CASCADES[c], xi, a, 0, n);
    generateRawRowsSized(p, sc, CASCADES[c], n, xi, b, 0, n);
    assert.deepEqual(a, b);
    assert.deepEqual(packInitialSpectrum(a, p.standing, new Float32Array(4 * n * n)), packInitialSpectrumSized(b, p.standing, n, new Float32Array(4 * n * n)));
  }
});

test('evaluateFields matches a naive DFT (n = 16)', () => {
  const n = 16, L = 9;
  let s = 7;
  const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296 - 0.5);
  const h0 = new Float32Array(2 * n * n).map(() => rnd());
  for (let i = 0; i < n; i++) { h0[2 * i] = h0[2 * i + 1] = 0; h0[2 * i * n] = h0[2 * i * n + 1] = 0; } // Nyquist lines
  // omega must depend on |k| only (ht(-k) = conj(ht(k)), as in the real system)
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
      // Da = IFFT(-i kx/kl ht): real part of (-i kxn)(htr + i hti) e^{ia} = kxn (hti c + htr sn)
      E += (kx / kl) * (hti * c + htr * sn);
      N += (kz / kl) * (hti * c + htr * sn);
    }
    const i = y * n + x;
    assert.ok(Math.abs(f[3 * i] + H) < 1e-9, `up ${f[3 * i]} vs ${-H}`);
    assert.ok(Math.abs(f[3 * i + 1] - chop * E) < 1e-9, 'east');
    assert.ok(Math.abs(f[3 * i + 2] + chop * N) < 1e-9, 'north');
  }
});

/** Drawn surface of a tier vs the core's CPU query (default configuration), at displaced points. */
function parity(tier: keyof typeof FFT_TIERS, preset: 'breeze' | 'storm', cubic: boolean, count = 1500, cpuCubic = false) {
  const shape = createFlatShape();
  const query = cpuCubic ? [{ size: 256, snapshotDt: 0.6, filter: 'cubic' as const }, { size: 256, snapshotDt: 0.25, filter: 'cubic' as const }, { size: 128, snapshotDt: 0.15, filter: 'bilinear' as const }] : undefined;
  const ocean = createOcean({ shape, preset, seed: 1, query });
  const st = ocean.getState();
  const T = 12, chop = ocean.choppinessAt(T), hs = ocean.expectedHs();
  const spec = FFT_TIERS[tier], cs = tierCascades(tier), n = spec.size;
  const h0 = spec.reference
    ? [0, 1, 2].map((c) => stripMasterH0(st.slots[st.toSlot].h0[c], new Float32Array(2 * n * n)))
    : generateTierSpectrum(spec, cs, st.slots[st.toSlot].params, 1, st.gravity).h0;
  const om = tierOmega(spec, cs, st.gravity);
  const fields = [0, 1, 2].map((c) => evaluateFields(h0[c], h0[c], 1, om.subarray(c * n * n, (c + 1) * n * n), n, cs[c].domain, T, chop));
  let s = 99;
  const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
  const out = new Float64Array(3), tmp = new Float64Array(3), errs: number[] = [];
  const rs = { east: 0, north: 0, up: 0, vEast: 0, vNorth: 0, vUp: 0 };
  let other = 0;
  for (let i = 0; i < count; i++) {
    const e = rnd() * 2000 - 1000, nn = rnd() * 2000 - 1000;
    sampleChained(fields, n, cs, e, nn, cubic, out, tmp);
    ocean.invertLocal(e + out[1], nn + out[2], T, rs);
    // folded crests: several rest points land on p (any is on the surface); compare only the same sheet
    if (Math.hypot(rs.east - e, rs.north - nn) > 0.3) { other++; continue; }
    errs.push((Math.abs(rs.up - out[0]) / hs) * 100);
  }
  errs.sort((a, b) => a - b);
  const rms = Math.sqrt(errs.reduce((a, b) => a + b * b, 0) / errs.length);
  return { rms, p99: errs[Math.floor(errs.length * 0.99)], p999: errs[Math.floor(errs.length * 0.999)], max: errs[errs.length - 1], hs, other: other / count };
}

test('CPU query (core default) vs drawn surface per tier (report)', () => {
  for (const preset of ['breeze', 'storm'] as const) {
    for (const tier of TIER_NAMES) {
      const cases: [boolean, boolean][] = tier === 'medium' ? [[false, false], [true, false], [true, true]] : [[true, false]];
      for (const [cubic, cpuCubic] of cases) {
        const r = parity(tier, preset, cubic, tier === 'medium' ? 1500 : 600, cpuCubic);
        console.log(`  parity ${preset} drawn ${tier} ${cubic ? 'catmull-rom' : 'bilinear'} vs CPU query ${cpuCubic ? 'cubic c0/c1' : 'default (bilinear)'}: rms ${r.rms.toFixed(3)} %Hs, p99 ${r.p99.toFixed(3)}, p99.9 ${r.p999.toFixed(3)}, max ${r.max.toFixed(3)}, other sheet ${(r.other * 100).toFixed(1)} % (Hs ${r.hs.toFixed(2)} m)`);
        if (tier === 'medium' && cubic === cpuCubic) assert.ok(r.p999 < 2, `medium p99.9 ${r.p999}`);
      }
    }
  }
});
