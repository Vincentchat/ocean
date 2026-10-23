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
    // The band is kept to ~M/3 cells: morphing vertices slide (continuously) at camera speed / band width.
    morphStart: Math.max(M / 2 + 2, M - 6 - Math.ceil(M / 3)),
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
}

/** Builds the merged clipmap (skirt, level 0 square, rings 1..L-1). Setup-time; allocates. */
export function buildLodGeometry(p: LodParams): LodGeometryData {
  const M = p.halfCells, R = M + p.margin, H = M / 2, W = 2 * R + 1;
  const L = p.levels;
  const skirtLoop = 8 * R;
  const fullVerts = W * W, ringVerts = W * W - (M - 1) * (M - 1);
  const vertexCount = 3 * skirtLoop + fullVerts + (L - 1) * ringVerts;
  const fullCells = 4 * R * R, ringCells = 4 * R * R - M * M;
  const triangleCount = 4 * skirtLoop + 2 * fullCells + (L - 1) * 2 * ringCells;
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
  for (let l = 0; l < L; l++) {
    const v0 = nv;
    map.fill(-1);
    for (let gy = -R; gy <= R; gy++) {
      for (let gx = -R; gx <= R; gx++) {
        if (l > 0 && Math.abs(gx) < H && Math.abs(gy) < H) continue; // strictly inside the hole
        map[(gy + R) * W + (gx + R)] = vert(gx, gy, l * 4 + LOD_KIND_GRID);
      }
    }
    for (let j = -R; j < R; j++) {
      for (let i = -R; i < R; i++) {
        if (l > 0 && i >= -H && i < H && j >= -H && j < H) continue;
        const a = map[(j + R) * W + (i + R)], b = map[(j + R) * W + (i + 1 + R)];
        const c = map[(j + 1 + R) * W + (i + 1 + R)], d = map[(j + 1 + R) * W + (i + R)];
        // Diagonal a-c ((i,j)-(i+1,j+1)) everywhere: the geomorph (odd -> lower even neighbour) then collapses
        // 2x2 fine cells exactly onto the parent's triangulation. Winding: counter-clockwise seen from +up.
        index[ni++] = a; index[ni++] = d; index[ni++] = c;
        index[ni++] = a; index[ni++] = c; index[ni++] = b;
      }
    }
    levelIndexEnd.push(ni);
    levelVertexCount.push(nv - v0);
  }
  if (nv !== vertexCount || ni !== index.length) throw new Error(`buildLodGeometry: count mismatch ${nv}/${vertexCount} ${ni}/${index.length}`);
  return { packed, index, vertexCount, triangleCount, skirtIndexCount, levelIndexEnd, levelVertexCount };
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
 * previous foot for motion vectors), false for re-evaluations (origin shift).
 */
export function updateLodState(st: LodState, p: LodParams, footU: number, footV: number, height: number, advance = true): void {
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
    const len = Math.hypot(pu, pv) || 1;
    pu = f((pu / len) * skirtRadius); pv = f((pv / len) * skirtRadius);
    qu = f(pu + phiU); qv = f(pv + phiV);
  }
  out.qu = qu; out.qv = qv; out.pu = pu; out.pv = pv; out.morph = m;
  out.y = kind === LOD_KIND_SKIRT_TOP ? st.height : 0;
  return out;
}
