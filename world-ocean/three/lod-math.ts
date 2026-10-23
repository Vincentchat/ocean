// Pure (three-free) part of the ocean LOD: tier parameters, clipmap geometry generation, per-frame
// snapping state and the floating-origin tile anchors. Runs in Node (tests/lod.test.ts) — never import
// three here. The three/TSL side lives in three/lod.ts and mirrors `emulateVertex` below op for op.
//
// Model (see three/lod.ts header for the public API):
//   * "lattice coordinates" (u, v) = (dot(W, east), dot(W, south)) of a world point W, metres, float64,
//     south = -north. On the flat shape u = x, v = z, so tile UVs match the reference (texture x <-> world x,
//     texture y <-> world z).
//   * Level l has spacing s_l = s0 * 2^l and its own centre C_l = round(foot / 2 s_l) * 2 s_l (snapped to the
//     PARENT's spacing), so every vertex of level l sits on the world lattice s_l Z^2: vertices never swim.
//   * Level l's grid spans C_l +- (M + 2) s_l; ring l (l >= 1) has a hole C_l +- (M/2) s_l where level l-1 lives.
//     Level l-1 is clamped into that hole (its extra 2-cell margin collapses onto the hole edge), and its
//     vertices near the edge geomorph (odd lattice vertices slide onto the even = parent lattice). So the
//     boundary between two levels has exactly the parent's vertices on both sides: no T-junction, no crack.
//   * All positions are sent as exact float32 lattice offsets q from a lattice anchor B (near the camera);
//     the camera foot's offset phi = foot - B is subtracted last, so shared boundary vertices are bit-identical
//     on both sides (see emulateVertex + tests).
//   * The drawn level count K adapts to the camera altitude (finest spacing ~ k*h/M); a skirt (flat annulus
//     + a wall whose top is at eye level) closes the gap from the last ring to the true horizon.
import type { Vec3 } from '../stubs/world-core.ts';

export type LodTier = 'low' | 'medium' | 'high' | 'ultra';

/** Vertex kinds packed in the geometry (position.z = level * 4 + kind). */
export const LOD_KIND_GRID = 0;
export const LOD_KIND_SKIRT_INNER = 1;
export const LOD_KIND_SKIRT_OUTER = 2;
export const LOD_KIND_SKIRT_TOP = 3;

/** Upper bound on geometry levels (uniform array size is MAX_LOD_LEVELS + 1). */
export const MAX_LOD_LEVELS = 22;

/**
 * Per-tier density. halfCells = M: each ring spans M/2..M cells of its spacing from the camera, so the
 * grid cell is 1/M..2/M of the (max-norm) distance. Reference polar grid (REFERENCE §1.4): cell ~ 1.79 %
 * (high), 2.39 % (balanced), 3.58 % (light) of the distance.
 */
export const LOD_TIER_TABLE: Readonly<Record<LodTier, { halfCells: number; minSpacing: number }>> = {
  low: { halfCells: 48, minSpacing: 1 / 64 },     // cell 2.1-4.2 % of distance (ref light 3.6 %)
  medium: { halfCells: 64, minSpacing: 1 / 64 },  // 1.6-3.1 % (ref balanced 2.4 %)
  high: { halfCells: 96, minSpacing: 1 / 128 },   // 1.0-2.1 % (ref high 1.8 %)
  ultra: { halfCells: 128, minSpacing: 1 / 128 }, // 0.8-1.6 %
};

export interface LodParams {
  readonly tier: LodTier;
  /** M: half extent of a level in cells of its own spacing (even, >= 20). */
  readonly halfCells: number;
  /** Smallest finest-level spacing s0, metres (power of two). */
  readonly minSpacing: number;
  /** Geometry levels L (enough to reach `coverage` at minSpacing). */
  readonly levels: number;
  /** Rings must reach at least this max-norm distance from the camera foot, metres. Skirt beyond. */
  readonly coverage: number;
  /** s0 ~ targetRatio * altitude / M (cell / distance ratio right under the camera). */
  readonly targetRatio: number;
  /** Geomorph from morphStart to morphEnd cells (max-norm distance / own spacing). */
  readonly morphStart: number;
  readonly morphEnd: number;
  /** Extra cells around each level (absorbs the parent's snap offset; collapsed by the clamp). */
  readonly margin: number;
}

export interface LodParamOptions {
  coverage?: number;
  halfCells?: number;
  minSpacing?: number;
  targetRatio?: number;
}

export function lodParams(tier: LodTier, opts: LodParamOptions = {}): LodParams {
  const row = LOD_TIER_TABLE[tier] ?? LOD_TIER_TABLE.high;
  const M = opts.halfCells ?? row.halfCells;
  if (!(M >= 20) || M % 2 !== 0) throw new RangeError(`lodParams: halfCells must be even and >= 20 (got ${M})`);
  const minSpacing = opts.minSpacing ?? row.minSpacing;
  if (!(minSpacing > 0) || 2 ** Math.round(Math.log2(minSpacing)) !== minSpacing) throw new RangeError('lodParams: minSpacing must be a power of two');
  const coverage = opts.coverage ?? 2400;
  let levels = 1;
  while (levels < MAX_LOD_LEVELS && (M + 1) * minSpacing * 2 ** (levels - 1) < coverage) levels++;
  return {
    tier, halfCells: M, minSpacing, levels, coverage,
    targetRatio: opts.targetRatio ?? 1.5,
    // Ring l's hole edge is at most (M/2 + 1) s_l from the foot: morph must be 0 there (matches level l-1).
    // Everything that the parent's hole can uncover / clamp is >= (M - 6) s_l away: morph must be 1 there.
    // Widest band the constraints allow (M/2 - 8 cells): morphing vertices slide (continuously) at
    // camera speed / band width, so the widest band slides slowest.
    morphStart: M / 2 + 2,
    morphEnd: M - 6,
    margin: 2,
  };
}

// ---------------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------------
export interface LodGeometryData {
  /** xyz per vertex: (gx, gy, level * 4 + kind). gx, gy = integer cell coordinates from the level centre. */
  readonly packed: Float32Array;
  readonly index: Uint32Array;
  readonly vertexCount: number;
  readonly triangleCount: number;
  /** Index count of the skirt (drawn first). */
  readonly skirtIndexCount: number;
  /** levelIndexEnd[l] = index count of skirt + levels 0..l (draw range end when l+1 levels are drawn). */
  readonly levelIndexEnd: readonly number[];
  readonly levelVertexCount: readonly number[];
  /** levelSectorEnd[l * 8 + k] = index end of (level l, sector k); levels are stored sector by sector. */
  readonly levelSectorEnd: readonly number[];
}

/** Builds the merged clipmap (skirt, level 0 square, rings 1..L-1). Setup-time; allocates. */
export function buildLodGeometry(p: LodParams): LodGeometryData {
  const it = lodGeometrySteps(p);
  let r = it.next();
  while (!r.done) r = it.next();
  return r.value;
}

/**
 * Same as buildLodGeometry, as a generator that yields after every grid row (~0.01-0.05 ms of work), so a tier
 * change can be built in time slices off the frame's critical path. The result is the generator's return value.
 */
/** Vertex / triangle counts of a tier's geometry without building it. */
export function lodGeometryCounts(p: LodParams): { vertexCount: number; triangleCount: number } {
  const M = p.halfCells, R = M + p.margin, W = 2 * R + 1, L = p.levels, skirtLoop = 8 * R;
  const vertexCount = 3 * skirtLoop + W * W + (L - 1) * (W * W - (M - 1) * (M - 1));
  const triangleCount = 4 * skirtLoop + 2 * 4 * R * R + (L - 1) * 2 * (4 * R * R - M * M);
  return { vertexCount, triangleCount };
}

export function* lodGeometrySteps(p: LodParams): Generator<void, LodGeometryData, void> {
  const M = p.halfCells, R = M + p.margin, H = M / 2, W = 2 * R + 1;
  const L = p.levels;
  const skirtLoop = 8 * R;
  const { vertexCount, triangleCount } = lodGeometryCounts(p);
  const packed = new Float32Array(vertexCount * 3);
  const index = new Uint32Array(triangleCount * 3);
  let nv = 0, ni = 0;
  const levelIndexEnd: number[] = [];
  const levelVertexCount: number[] = [];

  const vert = (gx: number, gy: number, lk: number) => {
    packed[nv * 3] = gx; packed[nv * 3 + 1] = gy; packed[nv * 3 + 2] = lk;
    return nv++;
  };

  // --- skirt: square loop at +-R (the outer edge of the last drawn level), CCW in (gx, gy).
  const loop: number[] = [];
  for (let i = -R; i < R; i++) loop.push(i, -R);
  for (let j = -R; j < R; j++) loop.push(R, j);
  for (let i = R; i > -R; i--) loop.push(i, R);
  for (let j = R; j > -R; j--) loop.push(-R, j);
  const skirtBase = nv;
  for (let k = 0; k < skirtLoop; k++) {
    const gx = loop[2 * k], gy = loop[2 * k + 1];
    vert(gx, gy, LOD_KIND_SKIRT_INNER);
    vert(gx, gy, LOD_KIND_SKIRT_OUTER);
    vert(gx, gy, LOD_KIND_SKIRT_TOP);
  }
  // Approximate local positions (x = gx, z = gy, y up) to pick the winding: annulus faces up, wall faces in.
  const tri = (a: number, b: number, c: number, wantUp: boolean, wantIn: boolean) => {
    const P = (v: number) => {
      const kind = packed[v * 3 + 2] % 4;
      const sc = kind >= LOD_KIND_SKIRT_OUTER ? 2 : 1;
      return [packed[v * 3] * sc, kind === LOD_KIND_SKIRT_TOP ? 1 : 0, packed[v * 3 + 1] * sc];
    };
    const A = P(a), B = P(b), C = P(c);
    const e1 = [B[0] - A[0], B[1] - A[1], B[2] - A[2]], e2 = [C[0] - A[0], C[1] - A[1], C[2] - A[2]];
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    let flip = false;
    if (wantUp) flip = n[1] < 0;
    if (wantIn) flip = n[0] * (A[0] + B[0] + C[0]) + n[2] * (A[2] + B[2] + C[2]) > 0;
    index[ni++] = a;
    if (flip) { index[ni++] = c; index[ni++] = b; } else { index[ni++] = b; index[ni++] = c; }
  };
  for (let k = 0; k < skirtLoop; k++) {
    const a = skirtBase + 3 * k, b = skirtBase + 3 * ((k + 1) % skirtLoop);
    tri(a, b, b + 1, true, false); tri(a, b + 1, a + 1, true, false);         // annulus (inner -> outer)
    tri(a + 1, b + 1, b + 2, false, true); tri(a + 1, b + 2, a + 2, false, true); // wall (outer -> eye level)
  }
  const skirtIndexCount = ni;

  // --- levels
  const map = new Int32Array(W * W);
  const sectorCount = new Int32Array(LOD_SECTORS), sectorFill = new Int32Array(LOD_SECTORS);
  const levelSectorEnd: number[] = [];
  for (let l = 0; l < L; l++) {
    const v0 = nv;
    map.fill(-1);
    for (let gy = -R; gy <= R; gy++) {
      for (let gx = -R; gx <= R; gx++) {
        if (l > 0 && Math.abs(gx) < H && Math.abs(gy) < H) continue; // strictly inside the hole
        map[(gy + R) * W + (gx + R)] = vert(gx, gy, l * 4 + LOD_KIND_GRID);
      }
      yield;
    }
    // Cells are emitted sector by sector (16 x 22.5 deg wedges around the level centre, by cell-centre angle) so a
    // frustum cull can draw only the visible (level, sector) index ranges. Pass 1 counts, pass 2 fills.
    sectorCount.fill(0);
    for (let j = -R; j < R; j++) {
      for (let i = -R; i < R; i++) {
        if (l > 0 && i >= -H && i < H && j >= -H && j < H) continue;
        sectorCount[lodSectorOf(i + 0.5, j + 0.5)] += 6;
      }
      if ((j & 7) === 7) yield;
    }
    let acc = ni;
    for (let k = 0; k < LOD_SECTORS; k++) { sectorFill[k] = acc; acc += sectorCount[k]; levelSectorEnd.push(acc); }
    for (let j = -R; j < R; j++) {
      for (let i = -R; i < R; i++) {
        if (l > 0 && i >= -H && i < H && j >= -H && j < H) continue;
        const a = map[(j + R) * W + (i + R)], b = map[(j + R) * W + (i + 1 + R)];
        const c = map[(j + 1 + R) * W + (i + 1 + R)], d = map[(j + 1 + R) * W + (i + R)];
        // Diagonal a-c ((i,j)-(i+1,j+1)) everywhere: the geomorph (odd -> lower even neighbour) then collapses
        // 2x2 fine cells exactly onto the parent's triangulation. Winding: counter-clockwise seen from +up.
        const sec = lodSectorOf(i + 0.5, j + 0.5);
        let o = sectorFill[sec];
        index[o++] = a; index[o++] = d; index[o++] = c;
        index[o++] = a; index[o++] = c; index[o++] = b;
        sectorFill[sec] = o;
      }
      yield;
    }
    ni = acc;
    levelIndexEnd.push(ni);
    levelVertexCount.push(nv - v0);
  }
  if (nv !== vertexCount || ni !== index.length) throw new Error(`buildLodGeometry: count mismatch ${nv}/${vertexCount} ${ni}/${index.length}`);
  return { packed, index, vertexCount, triangleCount, skirtIndexCount, levelIndexEnd, levelVertexCount, levelSectorEnd };
}

/** Number of angular sectors per level (22.5 degree wedges, sector k spans [k, k+1) * 22.5 deg from +east). */
export const LOD_SECTORS = 16;
/** Square-boundary points (max-norm 1) at k * 22.5 deg, (u, v) = (east, south). Every 45 deg corner is a sector
 *  boundary, so each sector's slice of a square ring is a trapezoid (convex hull = its 4 corners). */
const SECTOR_DIR: number[] = [];
for (let k = 0; k < LOD_SECTORS; k++) {
  const a = (k * 2 * Math.PI) / LOD_SECTORS, c = Math.cos(a), s = Math.sin(a), m = Math.max(Math.abs(c), Math.abs(s));
  SECTOR_DIR.push(c / m, s / m);
}

/** Sector of a point (u, v) relative to the level centre (cell units ok). */
export function lodSectorOf(u: number, v: number): number {
  let a = Math.atan2(v, u);
  if (a < 0) a += 2 * Math.PI;
  const k = Math.floor(a / ((2 * Math.PI) / LOD_SECTORS));
  return k >= LOD_SECTORS ? 0 : k;
}

/**
 * Frustum cull of the clipmap: fills `ranges` with [start, count) index ranges (absolute in the geometry, given
 * `indexBase`) of the visible (level, sector) wedges of the drawn levels, merging neighbours, and returns the number of
 * ranges. Allocation-free.
 *   planes: 4 side planes (a, b, c, d) x 4 in the space the points are expressed in (render space), a.x + d >= 0 inside.
 *   foot render position + tangent axes east/south (unit), margin (m) = how far outside a plane a wedge may poke before
 *   it is culled (covers displacement + one cell of triangle overhang: pass >= the max displacement).
 */
export function lodCullRanges(
  st: LodState, p: LodParams, g: LodGeometryData, indexBase: number, planes: Float64Array,
  footX: number, footY: number, footZ: number, ex: number, ey: number, ez: number, sx: number, sy: number, sz: number,
  margin: number, ranges: Int32Array,
): number {
  const K = st.drawnLevels, M = p.halfCells, R = M + p.margin + 1;
  let n = 0, openStart = -1, openEnd = -1;
  for (let l = 0; l < K; l++) {
    const lv = l * 4, s = st.level[lv + 2];
    const cu = st.level[lv] - st.phiU, cv = st.level[lv + 1] - st.phiV; // level centre relative to the foot (m)
    const rin = l === 0 ? 0 : Math.max(0, M / 2 - 1) * s, rout = R * s;
    const mg = margin + 2 * s;
    for (let k = 0; k < LOD_SECTORS; k++) {
      // Wedge trapezoid corners: rin/rout along the two bounding square directions.
      let visible = true;
      for (let pl = 0; pl < 4 && visible; pl++) {
        const a = planes[pl * 4], b = planes[pl * 4 + 1], c = planes[pl * 4 + 2], d = planes[pl * 4 + 3];
        let allOut = true;
        for (let q = 0; q < 4 && allOut; q++) {
          const dk = (k + (q & 1)) % LOD_SECTORS, r = q < 2 ? rin : rout;
          const u = cu + SECTOR_DIR[dk * 2] * r, v = cv + SECTOR_DIR[dk * 2 + 1] * r;
          const x = footX + ex * u + sx * v, y = footY + ey * u + sy * v, z = footZ + ez * u + sz * v;
          if (a * x + b * y + c * z + d >= -mg) allOut = false;
        }
        if (allOut) visible = false;
      }
      if (!visible) continue;
      const start = indexBase + (l === 0 && k === 0 ? g.skirtIndexCount : g.levelSectorEnd[l * LOD_SECTORS + k - 1]);
      const end = indexBase + g.levelSectorEnd[l * LOD_SECTORS + k];
      if (start === openEnd) openEnd = end;
      else {
        if (openStart >= 0) { ranges[n * 2] = openStart; ranges[n * 2 + 1] = openEnd - openStart; n++; }
        openStart = start; openEnd = end;
      }
    }
  }
  if (openStart >= 0) { ranges[n * 2] = openStart; ranges[n * 2 + 1] = openEnd - openStart; n++; }
  return n;
}

/** Side planes (left, right, bottom, top) of a projection * view matrix (column-major, three layout), normalised. */
export function frustumSidePlanes(m: ArrayLike<number>, out: Float64Array): Float64Array {
  // rows of the column-major matrix: row i = (m[i], m[4+i], m[8+i], m[12+i])
  for (let pl = 0; pl < 4; pl++) {
    const row = pl < 2 ? 0 : 1, sign = pl % 2 === 0 ? 1 : -1;
    const a = m[3] + sign * m[row], b = m[7] + sign * m[4 + row], c = m[11] + sign * m[8 + row], d = m[15] + sign * m[12 + row];
    const len = Math.hypot(a, b, c) || 1;
    out[pl * 4] = a / len; out[pl * 4 + 1] = b / len; out[pl * 4 + 2] = c / len; out[pl * 4 + 3] = d / len;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------
// Per-frame state
// ---------------------------------------------------------------------------------------------------
export interface LodState {
  /** log2 of the finest drawn spacing s0 (NaN before the first update). Hysteresis 0.1 octave. */
  log2Spacing: number;
  spacing0: number;
  /** Levels drawn this frame (K); the skirt attaches to level K-1. */
  drawnLevels: number;
  /** Per level l (0..levels inclusive): [Du, Dv, s, hasParent]: D = C_l - B (exact in float32), s = spacing (m). */
  readonly level: Float64Array;
  /** Lattice anchor B (world lattice coordinates, float64, snapped to 2 * spacing of level K-1). */
  baseU: number; baseV: number;
  /** foot - B, metres (sent as float32). */
  phiU: number; phiV: number;
  /** Camera foot in lattice coordinates (float64) and camera altitude above sea level (m). */
  footU: number; footV: number; height: number;
  /** foot - previous frame's foot (m) — for previous-frame positions. */
  dFootU: number; dFootV: number;
  hasPrev: boolean;
  /** Debug: disable snapping (centres follow the camera continuously: shows swimming). */
  noSnap: boolean;
}

export function createLodState(): LodState {
  return {
    log2Spacing: NaN, spacing0: 1, drawnLevels: 1, level: new Float64Array((MAX_LOD_LEVELS + 1) * 4),
    baseU: 0, baseV: 0, phiU: 0, phiV: 0, footU: 0, footV: 0, height: 0, dFootU: 0, dFootV: 0, hasPrev: false, noSnap: false,
  };
}

const HYSTERESIS = 0.1; // octaves

/** Snap x to the nearest multiple of `step` (float64). */
export function snap(x: number, step: number): number {
  return Math.round(x / step) * step;
}

/**
 * Per-frame LOD update (allocation-free). footU/footV: camera foot in lattice coordinates (float64 world),
 * height: camera altitude above sea level (m). advance = true for the once-per-frame update (tracks the
 * previous foot for motion vectors), false for re-evaluations (origin shift). maxReach: max-norm reach (m) the
 * drawn rings may not exceed (keeps them inside a finite far plane); coverage yields to it.
 */
export function updateLodState(st: LodState, p: LodParams, footU: number, footV: number, height: number, advance = true, maxReach = Infinity): void {
  const M = p.halfCells;
  const hEff = Math.max(Math.abs(height), (M * p.minSpacing) / p.targetRatio);
  const t = Math.log2((p.targetRatio * hEff) / M);
  const minLog = Math.log2(p.minSpacing);
  if (!Number.isFinite(st.log2Spacing) || t >= st.log2Spacing + 1 + HYSTERESIS || t < st.log2Spacing - HYSTERESIS) {
    st.log2Spacing = Math.max(minLog, Math.floor(t));
  }
  const s0 = 2 ** st.log2Spacing;
  st.spacing0 = s0;
  let K = 1;
  while (K < p.levels && (M + 1) * s0 * 2 ** (K - 1) < p.coverage) K++;
  // maxReach (m, max-norm, e.g. 0.6 * camera far): drop outer rings that would cross the far plane.
  while (K > 1 && (M + p.margin) * s0 * 2 ** (K - 1) > maxReach) K--;
  st.drawnLevels = K;
  const sTop = s0 * 2 ** (K - 1);
  if (advance) {
    st.dFootU = st.hasPrev ? footU - st.footU : 0;
    st.dFootV = st.hasPrev ? footV - st.footV : 0;
    st.hasPrev = true;
  }
  st.footU = footU; st.footV = footV; st.height = height;
  const bu = st.noSnap ? footU : snap(footU, 2 * sTop);
  const bv = st.noSnap ? footV : snap(footV, 2 * sTop);
  st.baseU = bu; st.baseV = bv;
  st.phiU = footU - bu; st.phiV = footV - bv;
  const lv = st.level;
  for (let l = 0; l <= p.levels; l++) {
    const s = s0 * 2 ** l;
    const cu = st.noSnap ? footU : snap(footU, 2 * s);
    const cv = st.noSnap ? footV : snap(footV, 2 * s);
    lv[l * 4] = cu - bu;
    lv[l * 4 + 1] = cv - bv;
    lv[l * 4 + 2] = s;
    lv[l * 4 + 3] = l < K - 1 ? 1 : 0;
  }
}

/** Index count to draw (skirt + levels 0..K-1). */
export function lodDrawCount(st: LodState, g: LodGeometryData): number {
  return g.levelIndexEnd[Math.min(st.drawnLevels, g.levelIndexEnd.length) - 1];
}

// ---------------------------------------------------------------------------------------------------
// Floating origin: periodic tile anchors
// ---------------------------------------------------------------------------------------------------
/** a mod period in [0, period) (float64, exact). */
export function wrapPeriod(a: number, period: number): number {
  let r = a % period;
  if (r < 0) r += period;
  if (r >= period) r -= period;
  return r === 0 ? 0 : r;
}

/** 0/1/2 when `axis` is +-(world axis k), else -1. */
export function worldAxisOf(axis: Readonly<Vec3>): number {
  const ax = Math.abs(axis.x), ay = Math.abs(axis.y), az = Math.abs(axis.z);
  if (ax === 1 && ay === 0 && az === 0) return 0;
  if (ay === 1 && ax === 0 && az === 0) return 1;
  if (az === 1 && ax === 0 && ay === 0) return 2;
  return -1;
}

/**
 * Lattice coordinate of the anchor B along `axis`, mod `period`, using only small numbers:
 *   B.axis mod P = (dot(B_render, axis) + dot(origin, axis)) mod P = (bRenderDot + dot(wrapped, axis)) mod P
 * where wrapped = origin.wrapped(P) and bRenderDot = dot(B's render-space point, axis) (|.| ~ km). This is the
 * spec's `renderPos + wrapped` rule; it is exact when `axis` is a +-world axis (every flat world). For a
 * non-axis-aligned frame dot(wrapped, axis) is not congruent to dot(origin, axis) mod P, so the float64
 * lattice coordinate bLattice is wrapped instead (same value, still computed on the CPU only).
 */
export function tileAnchor(axis: Readonly<Vec3>, bRenderDot: number, bLattice: number, wrapped: Readonly<Vec3>, period: number): number {
  const k = worldAxisOf(axis);
  if (k < 0) return wrapPeriod(bLattice, period);
  const w = k === 0 ? wrapped.x * axis.x : k === 1 ? wrapped.y * axis.y : wrapped.z * axis.z;
  return wrapPeriod(bRenderDot + w, period);
}

// ---------------------------------------------------------------------------------------------------
// CPU mirror of the TSL vertex function (float32 op for op) — tests only
// ---------------------------------------------------------------------------------------------------
export interface EmulatedVertex {
  /** Final lattice offset from B (float32), the value tileCoord adds to the anchor. */
  qu: number; qv: number;
  /** Local position relative to the camera foot (float32): (pu, y, pv) = (east, up, south). */
  pu: number; pv: number; y: number;
  morph: number;
}

const f = Math.fround;

/** Mirrors three/lod.ts `buildNodes` exactly (same op order, float32 rounding after each op). */
export function emulateVertex(st: LodState, p: LodParams, gx: number, gy: number, levelKind: number, skirtRadius: number, out: EmulatedVertex): EmulatedVertex {
  const kind = levelKind % 4, levelAttr = (levelKind - kind) / 4;
  const l = kind > 0 ? st.drawnLevels - 1 : levelAttr;
  const lv = st.level;
  const Du = f(lv[l * 4]), Dv = f(lv[l * 4 + 1]), s = f(lv[l * 4 + 2]), hp = f(lv[l * 4 + 3]);
  const Pu = f(lv[l * 4 + 4]), Pv = f(lv[l * 4 + 5]), Ps = f(lv[l * 4 + 6]);
  const phiU = f(st.phiU), phiV = f(st.phiV);
  const q0u = f(Du + f(gx * s)), q0v = f(Dv + f(gy * s));
  const p0u = f(q0u - phiU), p0v = f(q0v - phiV);
  const d = Math.max(Math.abs(p0u), Math.abs(p0v));
  const a = p.morphStart, b = p.morphEnd;
  const m = f(Math.min(Math.max(f(f(f(d / s) - a) / (b - a)), 0), 1) * hp);
  const oddU = gx - Math.floor(gx * 0.5) * 2, oddV = gy - Math.floor(gy * 0.5) * 2;
  let q1u = f(q0u - f(f(oddU * s) * m)), q1v = f(q0v - f(f(oddV * s) * m));
  if (hp > 0.5) {
    const half = f((p.halfCells / 2) * Ps);
    q1u = Math.min(Math.max(q1u, f(Pu - half)), f(Pu + half));
    q1v = Math.min(Math.max(q1v, f(Pv - half)), f(Pv + half));
  }
  let pu = f(q1u - phiU), pv = f(q1v - phiV);
  let qu = q1u, qv = q1v;
  if (kind >= LOD_KIND_SKIRT_OUTER) {
    // skirtRadius = distance along the axis of view (the shader divides the far depth by cos(angle to the view axis)).
    const len = Math.hypot(pu, pv) || 1;
    const t = kind === LOD_KIND_SKIRT_TOP ? skirtRadius : skirtRadius * 0.995;
    pu = f((pu / len) * t); pv = f((pv / len) * t);
    qu = f(pu + phiU); qv = f(pv + phiV);
  }
  out.qu = qu; out.qv = qv; out.pu = pu; out.pv = pv; out.morph = m;
  out.y = kind === LOD_KIND_SKIRT_TOP ? st.height : 0;
  return out;
}
