// Reviewer scratch: independent end-to-end check of heightAt / velocityAt.
// Drawn surface built independently: reference spectrum (src/spectral-ocean.js generateSpectra, g 9.81),
// reference GPU evolve math + independent DFT (fidelity.mjs referenceField) at 256^2 for all three cascades,
// chained bilinear lookup (spec conventions), brute-force grid inversion. Port: createOcean with a 9.81 flat shape.
import * as ref from '../../../../src/spectral-ocean.js';
import { referenceField } from './fidelity.mjs';
import { createOcean, CASCADES, PRESETS } from '../../../../world-ocean/index.ts';
import { fetchForPeakWavelength } from '../../../../world-ocean/spectrum.ts';
import { createFlatShape } from '../../../../world-ocean/stubs/impl.ts';

const N = 256, t = Number(process.argv[2] ?? 21.7);
const amp = PRESETS.breeze.params.amplitude;
const spec = ref.generateSpectra();
const fields = [];
for (let c = 0; c < 3; c++) {
  const h0 = Float32Array.from(spec.cascades[c].data, (v, i) => (i & 3) < 2 ? v * amp : v);
  fields.push(referenceField(c, t, h0, spec.cascades[c].frequency, 1.5));
}
function bil(c, a, b, out) {
  const L = CASCADES[c].domain, f = fields[c];
  const sa = ((a % L) + L) % L * N / L, sb = ((b % L) + L) % L * N / L;
  const x0 = Math.floor(sa), y0 = Math.floor(sb), fx = sa - x0, fy = sb - y0;
  const X0 = x0 & 255, Y0 = y0 & 255, X1 = (x0 + 1) & 255, Y1 = (y0 + 1) & 255;
  for (let ch = 0; ch < 3; ch++) {
    out[ch] = (1 - fx) * (1 - fy) * f[3 * (Y0 * N + X0) + ch] + fx * (1 - fy) * f[3 * (Y0 * N + X1) + ch]
      + (1 - fx) * fy * f[3 * (Y1 * N + X0) + ch] + fx * fy * f[3 * (Y1 * N + X1) + ch];
  }
}
const S = [0, 0, 0];
function forward(re, rn, out) { // chained: A = F0(r); B = F1(r + A.h); C = F2(r + A.h + B.h)
  let H = 0, E = 0, Nn = 0;
  for (let c = 0; c < 3; c++) {
    const off = CASCADES[c].offset;
    bil(c, re + E + off, -(rn + Nn) + off, S);
    H += S[0]; E += S[1]; Nn += S[2];
  }
  out[0] = H; out[1] = re + E; out[2] = rn + Nn;
}
const shape = createFlatShape({ surfaceGravity: 9.81 });
const ocean = createOcean({ shape, preset: 'breeze', fetch: fetchForPeakWavelength(15, 47, 9.81) });
const O = [0, 0, 0];
let maxErr = 0, sumErr = 0, nPts = 0, multi = 0, wrongSheet = 0, worst = '';
const errs = [];
const NP = Number(process.argv[3] ?? 150);
for (let i = 0; i < NP; i++) {
  const pe = ((i * 0.6180339887) % 1) * 2000 - 1000, pn = ((i * 0.7548776662) % 1) * 2000 - 1000;
  // coarse search
  const sols = [];
  const cand = [];
  for (let de = -10; de <= 10; de += 0.25) for (let dn = -10; dn <= 10; dn += 0.25) {
    forward(pe + de, pn + dn, O);
    const r = Math.hypot(O[1] - pe, O[2] - pn);
    if (r < 0.5) cand.push([pe + de, pn + dn, r]);
  }
  // refine each candidate with damped fixed-point + local grid
  for (const [ce, cn] of cand) {
    let re = ce, rn = cn, best = Infinity, be = re, bn = rn;
    for (let step = 0.1; step > 1e-5; step *= 0.5) {
      let improved = true;
      while (improved) {
        improved = false;
        for (const [dx, dy] of [[0, 0], [step, 0], [-step, 0], [0, step], [0, -step], [step, step], [-step, -step], [step, -step], [-step, step]]) {
          forward(re + dx, rn + dy, O);
          const r = Math.hypot(O[1] - pe, O[2] - pn);
          if (r < best - 1e-12) { best = r; be = re + dx; bn = rn + dy; improved = true; }
        }
        re = be; rn = bn;
      }
    }
    if (best < 0.002) {
      forward(re, rn, O);
      if (!sols.some((s) => Math.hypot(s[0] - re, s[1] - rn) < 0.05)) sols.push([re, rn, O[0]]);
    }
  }
  const h = ocean.heightAtLocal(pe, pn, t);
  if (sols.length === 0) { console.log(`point ${i}: brute force found no solution`); continue; }
  if (sols.length > 1) multi++;
  let e = Infinity; for (const s of sols) e = Math.min(e, Math.abs(s[2] - h));
  if (sols.length > 1) { const hs = sols.map((s) => s[2]); if (Math.abs(h - Math.max(...hs)) > 0.05) wrongSheet++; }
  errs.push(e); sumErr += e; nPts++;
  if (e > maxErr) { maxErr = e; worst = `p=(${pe.toFixed(2)}, ${pn.toFixed(2)}) heightAt ${h.toFixed(4)} brute ${sols.map((s) => s[2].toFixed(4)).join('/')}`; }
}
errs.sort((a, b) => a - b);
const hs = ocean.expectedHs();
console.log(`t=${t}: ${nPts} points, Hs ${hs.toFixed(2)} m. |heightAt - brute force (independent 256^3 bilinear)|: mean ${(sumErr / nPts * 1000).toFixed(2)} mm, p50 ${(errs[nPts >> 1] * 1000).toFixed(2)} mm, p99 ${(errs[Math.floor(nPts * 0.99)] * 1000).toFixed(2)} mm, max ${(maxErr * 1000).toFixed(2)} mm (${(100 * maxErr / hs).toFixed(3)} % Hs)`);
console.log(`  worst: ${worst}`);
console.log(`  points with >1 sheet: ${multi}; of those heightAt not on the top sheet: ${wrongSheet}`);
