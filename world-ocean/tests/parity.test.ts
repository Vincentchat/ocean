// three/parity-math.ts: the probe's per-point footprint must equal the cellSize the LOD gives its vertices
// (three/lod.ts, mirrored op for op by lod-math.ts emulateVertex), so the probe selects the same texture LOD
// (Catmull-Rom vs trilinear mips) as the vertex shader.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  buildLodGeometry, createLodState, emulateVertex, lodParams, updateLodState, type EmulatedVertex, type LodTier,
} from '../three/lod-math.ts';
import { lodFootprint } from '../three/parity-math.ts';

const CASES: { tier: LodTier; foot: [number, number]; h: number }[] = [
  { tier: 'high', foot: [3.3, -5.1], h: 12 },
  { tier: 'low', foot: [100003.7, 99994.2], h: 12 },
  { tier: 'ultra', foot: [-41.06, 17.9], h: 2.2 },
  { tier: 'medium', foot: [517.25, -903.5], h: 60 },
];

for (const c of CASES) {
  test(`lodFootprint matches the vertex cellSize (${c.tier}, h ${c.h} m)`, () => {
    const p = lodParams(c.tier);
    const g = buildLodGeometry(p);
    const st = createLodState();
    updateLodState(st, p, c.foot[0], c.foot[1], c.h, true, 0.6 * 20000);
    const v: EmulatedVertex = { qu: 0, qv: 0, pu: 0, pv: 0, y: 0, morph: 0 };
    let n = 0, worst = 0, interior = 0, interiorWorst = 0;
    for (let i = 0; i < g.vertexCount; i++) {
      const lk = g.packed[3 * i + 2];
      const kind = lk % 4, level = (lk - kind) / 4;
      if (kind !== 0 || level >= st.drawnLevels) continue;
      emulateVertex(st, p, g.packed[3 * i], g.packed[3 * i + 1], lk, 1e5, v);
      const s = st.level[level * 4 + 2];
      const cell = s * (1 + v.morph);
      const fp = lodFootprint(st, p, v.pu, v.pv);
      const rel = Math.abs(fp - cell) / cell;
      n++;
      if (rel > worst) worst = rel;
      // vertices strictly inside their level's own area (not on a shared boundary with the parent ring)
      const du = Math.abs(v.pu + st.phiU - st.level[(level + 1) * 4]), dv = Math.abs(v.pv + st.phiV - st.level[(level + 1) * 4 + 1]);
      if (level === st.drawnLevels - 1 || Math.max(du, dv) < p.halfCells * s - 1e-3) {
        interior++;
        if (rel > interiorWorst) interiorWorst = rel;
      }
    }
    assert.ok(n > 1000, `vertices tested ${n}`);
    // The vertex shader computes the morph from the vertex's lattice point BEFORE the geomorph slide (<= 1 cell);
    // the probe only knows the final position, so in the morph band the two morph values differ by up to
    // 1 / (morphEnd - morphStart) (~2-5 % of the cell = 0.03-0.07 texture LOD levels). Everywhere else: exact.
    const band = 1 / (p.morphEnd - p.morphStart) + 1e-4;
    assert.ok(interiorWorst <= band, `interior worst relative footprint error ${interiorWorst} (${interior} vertices, band ${band})`);
    assert.ok(worst <= band, `worst relative footprint error ${worst} (${n} vertices)`);
  });
}
