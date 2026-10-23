// Reviewer scratch: how often is the drawn surface folded (multi-valued), and how often does heightAt(p, t)
// jump between sheets from one 60 Hz tick to the next (a floating object teleporting vertically)?
import { createOcean, PRESET_NAMES } from '../../../../world-ocean/index.ts';
import { createFlatShape } from '../../../../world-ocean/stubs/impl.ts';
const shape = createFlatShape();
const r = { east: 0, north: 0, up: 0, vEast: 0, vNorth: 0, vUp: 0 };
const configs = PRESET_NAMES.map((n) => [n, {}]).concat([['breeze+U25 (wind-driven, fetch kept)', { windSpeed: 25 }]]);
for (const [name, extra] of configs) {
  const ocean = createOcean({ shape, preset: name.startsWith('breeze+') ? 'breeze' : name, ...extra });
  const t = 30;
  // fold fraction: det of d(displaced horizontal)/d(rest) by central differences (h = 2 cm) at 20000 rest points
  let folded = 0; const NS = 20000, h = 0.02;
  for (let i = 0; i < NS; i++) {
    const e = ((i * 0.6180339887) % 1) * 1024, n = ((i * 0.7548776662) % 1) * 1024;
    ocean.sampleRest(e + h, n, t, r); const ae = r.east, an = r.north;
    ocean.sampleRest(e - h, n, t, r); const be = r.east, bn = r.north;
    ocean.sampleRest(e, n + h, t, r); const ce = r.east, cn = r.north;
    ocean.sampleRest(e, n - h, t, r); const de = r.east, dn = r.north;
    const j00 = (ae - be) / (2 * h), j10 = (an - bn) / (2 * h), j01 = (ce - de) / (2 * h), j11 = (cn - dn) / (2 * h);
    if (j00 * j11 - j01 * j10 <= 0) folded++;
  }
  // tick-to-tick jumps of heightAt at fixed points over 10 s at 60 Hz
  const P = 400; let big = 0, total = 0, maxd = 0;
  const prev = new Float64Array(P);
  for (let s = 0; s <= 600; s++) {
    const tt = t + s / 60;
    for (let k = 0; k < P; k++) {
      const e = ((k * 0.381966) % 1) * 1000, n = ((k * 0.2451223) % 1) * 1000;
      const hh = ocean.heightAtLocal(e, n, tt);
      if (s > 0) { const d = Math.abs(hh - prev[k]); total++; if (d > 0.25) big++; maxd = Math.max(maxd, d); }
      prev[k] = hh;
    }
  }
  console.log(`${name}: Hs ${ocean.expectedHs().toFixed(2)} m, folded area ${(100 * folded / NS).toFixed(2)} %, per-tick |dh| > 0.25 m (> 15 m/s): ${big}/${total} (${(1e6 * big / total).toFixed(0)} ppm), max ${maxd.toFixed(2)} m`);
}
