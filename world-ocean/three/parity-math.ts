// Pure (three-free) part of the GPU height probe (three/parity.ts). Runs in Node (tests/parity.test.ts).
import type { LodParams, LodState } from './lod-math.ts';

/**
 * Footprint (m) the LOD mesh gives its vertices at tangent-plane offset (e east, s south) from the camera foot, i.e.
 * the `cellSize` node of three/lod.ts that the vertex shader turns into the texture LOD:
 *   level  = the finest drawn level whose area covers the point: level l covers the hole of ring l+1,
 *            |x - C_(l+1)| <= M s_l (max-norm, M = halfCells); the last level covers C_(K-1) +- (M + margin) s_(K-1);
 *   morph  = clamp((max-norm distance to the foot / s - morphStart) / (morphEnd - morphStart), 0, 1) * hasParent;
 *   cellSize = s (1 + morph);  beyond the last ring (horizon skirt): 0.05 * distance to the camera.
 * Continuous across level boundaries (morph = 1 at the outer edge of level l gives 2 s_l = s_(l+1)).
 * Allocation-free.
 */
export function lodFootprint(st: Readonly<LodState>, p: Readonly<LodParams>, e: number, s: number): number {
  const lv = st.level, K = st.drawnLevels, M = p.halfCells;
  const bu = e + st.phiU, bv = s + st.phiV; // offset from the lattice anchor B
  let l = 0;
  for (; l < K - 1; l++) {
    const du = Math.abs(bu - lv[(l + 1) * 4]), dv = Math.abs(bv - lv[(l + 1) * 4 + 1]);
    if ((du > dv ? du : dv) <= M * lv[l * 4 + 2]) break;
  }
  const sp = lv[l * 4 + 2];
  if (l === K - 1) {
    const du = Math.abs(bu - lv[l * 4]), dv = Math.abs(bv - lv[l * 4 + 1]);
    if ((du > dv ? du : dv) > (M + p.margin + 1e-3) * sp) return 0.05 * Math.sqrt(e * e + s * s + st.height * st.height);
  }
  const ae = Math.abs(e), as = Math.abs(s);
  const dmax = (ae > as ? ae : as) / sp;
  let m = (dmax - p.morphStart) / (p.morphEnd - p.morphStart);
  m = m < 0 ? 0 : m > 1 ? 1 : m;
  m *= lv[l * 4 + 3];
  return sp * (1 + m);
}
