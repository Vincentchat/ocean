// heightAt inversion correctness and CPU-query parity with the drawn (GPU) surface.
//
// "Drawn surface" model: the GPU at its 256 tier evaluates all three cascades at 256^2 and samples
// them bilinearly with the chained lookup (spectrum.ts conventions). Here that is emulated on the
// CPU with query sizes 256/256/256 and t exactly on a snapshot (exact spectral evaluation), and
// compared with the DEFAULT query configuration at the same displaced points.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createOcean, PRESET_NAMES } from '../index.ts';
import type { RestSample, PresetName } from '../index.ts';
import { createFlatShape } from '../stubs/impl.ts';

const shape = createFlatShape();
const T = 12;
const rest = (): RestSample => ({ east: 0, north: 0, up: 0, vEast: 0, vNorth: 0, vUp: 0 });
const GPU_EMULATION = [
  { size: 256, snapshotDt: T, filter: 'bilinear' as const },
  { size: 256, snapshotDt: T, filter: 'bilinear' as const },
  { size: 256, snapshotDt: T, filter: 'bilinear' as const },
];

function lcg(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

/** Truth: displaced points of random rest points on the drawn surface. */
function truth(preset: PresetName, count: number) {
  const drawn = createOcean({ shape, preset, query: GPU_EMULATION });
  const rnd = lcg(1234), o = rest();
  const restPts = new Float64Array(2 * count), pts = new Float64Array(3 * count);
  for (let i = 0; i < count; i++) {
    const e = rnd() * 4000 - 2000, n = rnd() * 4000 - 2000;
    drawn.sampleRest(e, n, T, o);
    restPts[2 * i] = e; restPts[2 * i + 1] = n;
    pts[3 * i] = o.east; pts[3 * i + 1] = o.north; pts[3 * i + 2] = o.up;
  }
  return { drawn, restPts, pts, hs: drawn.expectedHs() };
}

function errors(ocean: ReturnType<typeof createOcean>, tr: ReturnType<typeof truth>) {
  const r = rest(), errs: number[] = [];
  let otherSheet = 0;
  const count = tr.restPts.length / 2;
  for (let i = 0; i < count; i++) {
    ocean.invertLocal(tr.pts[3 * i], tr.pts[3 * i + 1], T, r);
    // Where a crest folds over, several rest points land on p: any of them is "on the surface".
    if (Math.hypot(r.east - tr.restPts[2 * i], r.north - tr.restPts[2 * i + 1]) > 0.3) { otherSheet++; continue; }
    errs.push(Math.abs(r.up - tr.pts[3 * i + 2]) / tr.hs * 100);
  }
  errs.sort((a, b) => a - b);
  const rms = Math.sqrt(errs.reduce((a, b) => a + b * b, 0) / errs.length);
  return { max: errs[errs.length - 1], p999: errs[Math.floor(errs.length * 0.999)], p99: errs[Math.floor(errs.length * 0.99)], rms, otherSheet: otherSheet / count };
}

test('heightAt inversion: height of the surface point that ends up AT p (same grid, < 0.1 % Hs rms)', () => {
  const tr = truth('windy', 6000);
  const e = errors(tr.drawn, tr);
  console.log(`  inversion (same grid, breeze): rms ${e.rms.toFixed(4)} %Hs, p99 ${e.p99.toFixed(3)}, max ${e.max.toFixed(3)}, other sheet ${(e.otherSheet * 100).toFixed(2)} %`);
  assert.ok(e.rms < 0.1, `rms ${e.rms}`);
  assert.ok(e.p99 < 0.5, `p99 ${e.p99}`);
  assert.ok(e.max < 2.5, `max ${e.max}`);
  assert.ok(e.otherSheet < 0.01);
  // Without the inversion (height of the undisplaced grid point at p) the error is large:
  const r = rest();
  let naive = 0;
  for (let i = 0; i < 2000; i++) {
    tr.drawn.sampleRest(tr.pts[3 * i], tr.pts[3 * i + 1], T, r);
    naive += ((r.up - tr.pts[3 * i + 2]) / tr.hs * 100) ** 2;
  }
  assert.ok(Math.sqrt(naive / 2000) > 10 * e.rms, 'the inversion matters');
});

for (const preset of PRESET_NAMES) {
  test(`CPU query (default config) vs drawn 256^3 surface, ${preset}: p99.9 < 2 % Hs`, () => {
    const tr = truth(preset, 4000);
    const e = errors(createOcean({ shape, preset }), tr);
    console.log(`  parity ${preset} (Hs ${tr.hs.toFixed(2)} m): rms ${e.rms.toFixed(3)} %Hs, p99 ${e.p99.toFixed(3)}, p99.9 ${e.p999.toFixed(3)}, max ${e.max.toFixed(3)}`);
    assert.ok(e.p999 < 2, `p99.9 ${e.p999} % of Hs`);
    assert.ok(e.max < 4, `max ${e.max} % of Hs`);
  });
}
