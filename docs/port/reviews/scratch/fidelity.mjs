// Reviewer scratch: seed-1 'breeze' vs the WebGL reference (src/spectral-ocean.js).
// node --experimental-strip-types docs/port/reviews/scratch/fidelity.mjs
import * as ref from '../../../../src/spectral-ocean.js';
import { generateCascade, fetchForPeakWavelength, OMEGA_UNIT, CASCADES, expectedHs, peakWavelength } from '../../../../world-ocean/spectrum.ts';
import { PRESETS } from '../../../../world-ocean/presets/index.ts';
import { CascadeSim, CALL } from '../../../../world-ocean/cpu-fft.ts';

const N = 256;
const breeze = PRESETS.breeze.params;
const refSpec = ref.generateSpectra(256, [1024, 96, 9], ref.OCEAN_PARAMETERS); // g = 9.81, the real reference
const amp = breeze.amplitude;

function compare(label, params, g) {
  let worstRel = 0, omegaMismatch = 0, total = 0;
  for (let c = 0; c < 3; c++) {
    const mine = generateCascade(params, c, 1, g);
    const data = refSpec.cascades[c].data, freq = refSpec.cascades[c].frequency;
    let peak = 0, maxDiff = 0;
    for (let i = 0; i < N * N; i++) {
      peak = Math.max(peak, Math.hypot(data[4 * i], data[4 * i + 1]) * amp);
      const d = Math.hypot(mine.h0[4 * i] - data[4 * i] * amp, mine.h0[4 * i + 1] - data[4 * i + 1] * amp);
      maxDiff = Math.max(maxDiff, d);
      const w = mine.omegaIndex[i] * OMEGA_UNIT;
      if (Math.abs(w - freq[i]) > 1e-6) omegaMismatch++;
      total++;
    }
    worstRel = Math.max(worstRel, maxDiff / peak);
    console.log(`  ${label} cascade ${c}: max |h0 - ref*amp| / peak = ${(maxDiff / peak).toExponential(2)}`);
  }
  console.log(`  ${label}: omega mismatches ${omegaMismatch}/${total}`);
  return worstRel;
}
console.log('Shipped breeze (g = 9.80665 = createFlatShape default, fetch 45479.33):');
compare('shipped', breeze, 9.80665);
console.log('breeze at g = 9.81 (fetch from the preset, not recomputed):');
compare('g9.81/preset fetch', breeze, 9.81);
console.log('breeze at g = 9.81 with fetch recomputed for 47 m:');
compare('g9.81/refit', { ...breeze, fetch: fetchForPeakWavelength(15, 47, 9.81) }, 9.81);
console.log(`peak wavelength shipped preset at 9.81: ${peakWavelength(15, breeze.fetch, 9.81).toFixed(4)} m`);
console.log(`Hs reference expectedVariance*amp^2: ${(4 * Math.sqrt(refSpec.expectedVariance) * amp).toFixed(4)} m; port expectedHs shipped: ${expectedHs(breeze, 9.80665).toFixed(4)} m`);

// Phase divergence over time caused by g 9.80665 vs 9.81 (dispersion): energy-weighted rms phase error.
{
  const c = 0;
  const mine = generateCascade(breeze, c, 1, 9.80665);
  const freq = refSpec.cascades[c].frequency, data = refSpec.cascades[c].data;
  for (const T of [10, 60, 600, 3600]) {
    let e = 0, ew = 0;
    for (let i = 0; i < N * N; i++) {
      const p = data[4 * i] ** 2 + data[4 * i + 1] ** 2;
      if (p === 0) continue;
      const dphi = (mine.omegaIndex[i] * OMEGA_UNIT - freq[i]) * T;
      e += p * (1 - Math.cos(dphi)) * 2; ew += p;
    }
    // relative rms field difference = sqrt(sum |a e^{i d}-a|^2 / sum |a|^2)
    console.log(`  dispersion g 9.80665 vs 9.81, cascade 0, t = ${T} s: relative rms height difference ${(Math.sqrt(e / ew) * 100).toFixed(2)} %`);
  }
}

// ---- Full field at time t: reference GPU math (EVOLVE + unnormalised IFFT + checkerboard + BLEND display)
// emulated in float64 with an independent separable DFT, vs the port's CascadeSim snapshot.
function dft2(re, im) { // inverse, unnormalised, rows then columns; independent of cpu-fft.ts
  const cs = new Float64Array(N), sn = new Float64Array(N);
  for (let j = 0; j < N; j++) { cs[j] = Math.cos(2 * Math.PI * j / N); sn[j] = Math.sin(2 * Math.PI * j / N); }
  const tr = new Float64Array(N), ti = new Float64Array(N);
  for (let pass = 0; pass < 2; pass++) {
    for (let line = 0; line < N; line++) {
      const idx = (m) => pass === 0 ? line * N + m : m * N + line;
      for (let u = 0; u < N; u++) {
        let sr = 0, si = 0;
        for (let m = 0; m < N; m++) {
          const w = (u * m) & (N - 1);
          const r = re[idx(m)], i = im[idx(m)];
          sr += r * cs[w] - i * sn[w]; si += r * sn[w] + i * cs[w];
        }
        tr[u] = sr; ti[u] = si;
      }
      for (let u = 0; u < N; u++) { re[idx(u)] = tr[u]; im[idx(u)] = ti[u]; }
    }
  }
}
export function referenceField(c, t, h0src /* Float32 [re,im,..,..] ref layout */, freq, chop) {
  const re1 = new Float64Array(N * N), im1 = new Float64Array(N * N), re2 = new Float64Array(N * N), im2 = new Float64Array(N * N);
  const dk = 2 * Math.PI / CASCADES[c].domain;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x, j = ((N - y) % N) * N + ((N - x) % N);
    const kx = Math.fround((x - N / 2) * dk), kz = Math.fround((y - N / 2) * dk);
    const k = Math.hypot(kx, kz) + 1e-4;
    const ph = freq[i] * t, er = Math.cos(ph), ei = -Math.sin(ph);
    const sr = h0src[4 * i], si = h0src[4 * i + 1], or = h0src[4 * j], oi = -h0src[4 * j + 1];
    const hr = sr * er - si * ei + or * er - oi * (-ei);
    const hi = sr * ei + si * er + or * (-ei) + oi * er;
    re1[i] = hr * (1 + kx / k); im1[i] = hi * (1 + kx / k);
    re2[i] = hi * kz / k; im2[i] = -hr * kz / k;
  }
  dft2(re1, im1); dft2(re2, im2);
  const out = new Float64Array(N * N * 3); // up, east, north (metres)
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x, s = ((x + y) & 1) ? -1 : 1;
    const fx = re1[i] * s, fy = im1[i] * s, fz = re2[i] * s;
    out[3 * i] = -fx; out[3 * i + 1] = fy * chop; out[3 * i + 2] = -fz * chop; // display (x, y, z) -> east, up, -north
  }
  return out;
}

if (process.argv[1] && process.argv[1].endsWith('fidelity.mjs')) {
  const t = 37.3;
  console.log(`Full displacement field at t = ${t} s, g = 9.81, breeze (fetch refit for 47 m), port CascadeSim 256 vs reference GPU math:`);
  const p81 = { ...breeze, fetch: fetchForPeakWavelength(15, 47, 9.81) };
  for (let c = 0; c < 3; c++) {
    const refH0 = Float32Array.from(refSpec.cascades[c].data, (v, i) => (i & 3) < 2 ? v * amp : v);
    const rf = referenceField(c, t, refH0, refSpec.cascades[c].frequency, 1.5);
    const mine = generateCascade(p81, c, 1, 9.81);
    const sim = new CascadeSim(CASCADES[c], { size: 256, snapshotDt: t, filter: 'bilinear' }, mine.omegaIndex, 1, 4);
    sim.setSlot(0, mine.h0);
    CALL[0] = 0; CALL[1] = 1; sim.setSource(0, 1, 0, 0);
    const pi = sim.ensure(1, 1);
    const snap = sim.pool[pi].data;
    let maxd = 0, maxH = 0, sumSign = 0;
    for (let i = 0; i < N * N; i++) {
      const dH = Math.abs(snap[6 * i] - rf[3 * i]);
      const dE = Math.abs(snap[6 * i + 1] * 1.5 - rf[3 * i + 1]);
      const dN = Math.abs(snap[6 * i + 2] * 1.5 - rf[3 * i + 2]);
      maxd = Math.max(maxd, dH, dE, dN); maxH = Math.max(maxH, Math.abs(rf[3 * i]));
      sumSign += snap[6 * i] * rf[3 * i];
    }
    console.log(`  cascade ${c}: max |port - ref| = ${maxd.toExponential(2)} m (max |H| ${maxH.toFixed(3)} m), height sign correlation ${sumSign > 0 ? '+' : '-'}`);
  }
}
