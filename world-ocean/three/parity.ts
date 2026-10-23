// GPU height probe (spec section 1 rule 8, section 4 "buoyancy / readback", section 7 checks 6 + 7).
//
// Reads back the height of the DRAWN water surface at N world points, asynchronously, one frame late, so the demo
// (?parity) can compare it with the core's CPU WATER_QUERY at the same points and the same simulation time.
//
// What "drawn" means here: the vertex shader (three/index.ts) moves the mesh vertex at rest position r (local
// tangent plane, metres) to
//     r + fade(r) * D(r),   D = createFftSampling().sampleDisplacement(lod.tileCoord, { footprint: cellSize, cubic })
//     fade(r) = 1 - smoothstep(200, 1200, |r - camera|)   (reference farFade)
// i.e. the chained cascade lookup with Catmull-Rom where the texture LOD < 0.5, trilinear mips beyond, LOD from the
// vertex footprint (the LOD's cellSize). The surface point that ends up AT p is the one whose rest point r solves
//     r + fade(r) * D_h(r) = p
// The probe solves that ON THE GPU with a fixed number of iterations (FIXED_STEPS fixed-point steps, then
// NEWTON_STEPS Newton steps with a finite-difference Jacobian, plain fixed-point step where det J < 0.1, like the
// reference buoyancy probe / the core's inversion) and returns fade(r) * D_up(r). It runs the SAME TSL sampling code
// as the vertex shader (createFftSampling on the same FFT texture nodes, same parity layer, same capillary gain),
// with the footprint the LOD gives the vertices around p (computed on the CPU from lod.state, see footprintAt).
// Not modelled: the linear interpolation between mesh vertices (triangles), i.e. the mesh discretisation error
// (cell size 1-2 % of the distance), which is a property of the mesh, not of the displacement field.
// The probe also returns the "field" height: the same chained lookup + inversion at texture LOD 0 (Catmull-Rom, no
// mip filtering) and without the far fade, i.e. the full-resolution GPU FFT surface, independent of the view.
// drawn - field = what the view-dependent vertex filtering (footprint mips + far fade) removes.
//
// Floating origin: the GPU never sees a world coordinate. Per point it gets q = p - foot (tangent-plane offset from
// the camera foot, float64 on the CPU -> small float32) and per tile period P a uniform anchor
// (foot mod P, from origin.wrapped(P) + the render-space foot, exactly like lod.ts tileAnchor); it samples
// anchor_P + r.
//
// Readback: renderer.getArrayBufferAsync(output, readbackBuffer) (three r186 ReadbackBuffer = one persistent
// MAP_READ staging buffer each) on a ring of RING entries. A frame dispatches only if an entry is free (never waits),
// the copy is queued right after the probe dispatch, the result arrives when the GPU is done (normally next frame)
// and carries the frame number + simulation time it was computed at (and the optional CPU reference evaluated at
// dispatch time, so the comparison never depends on when the readback lands). Nothing on the render path awaits.
// Allocation: no JS allocation in execute() / the completion handler apart from what the WebGPU API itself returns
// (the promise of mapAsync and its .then, the mapped ArrayBuffer and one Float32Array view of it per readback).
//
// Pass (spec section 5): { id: 'ocean.parity', stage: 'compute', reads: ['ocean.displacement'], writes:
// ['ocean.parity'] } â€” add it AFTER the ocean passes (same stage, insertion order) so it reads this frame's FFT.
import { ReadbackBuffer, StorageBufferAttribute, Vector2 } from 'three/webgpu';
import { Fn, If, Loop, dot, float, instanceIndex, length, max, select, smoothstep, storage, uniform, vec2, vec3, vec4 } from 'three/tsl';
import type { FloatingOrigin, TangentFrame, Vec3, WorldShape } from '../stubs/world-core.ts';
import { CASCADES } from '../spectrum.ts';
import { CAPILLARY_CASCADE } from './fft-tiers.ts';
import { createFftSampling, type OceanFft } from './fft.ts';
import type { OceanLod } from './lod.ts';
import { tileAnchor } from './lod-math.ts';
import { lodFootprint } from './parity-math.ts';
import type { Pass, PassContext } from './pass-types.ts';

/** Fixed-point steps (initial guess), then Newton steps (finite-difference Jacobian). */
export const PROBE_FIXED_STEPS = 4;
export const PROBE_NEWTON_STEPS = 10;
/** Finite-difference step of the Newton Jacobian, m. */
const FD_STEP = 0.05;
/** Readback ring size. */
const RING = 3;

export interface OceanHeightProbeOptions {
  renderer: any;
  fft: OceanFft;
  lod: OceanLod;
  origin: FloatingOrigin;
  shape: WorldShape;
  /** Max points per dispatch. Default 256. */
  capacity?: number;
  /** Catmull-Rom near the camera (must match createOceanGpu's displacement filter; ?debug=fftbilinear = false). Default true. */
  cubic?: boolean;
  /** The vertex shader's far fade (three/index.ts): 1 - smoothstep(fadeNear, fadeFar, distance), m. Default 200 / 1200. */
  fadeNear?: number;
  fadeFar?: number;
  /**
   * Optional CPU reference evaluated at DISPATCH time with the same points and time (e.g. ocean.heightsAt):
   * points = world xyz (float64), out[i] = height. Stored with the readback.
   */
  reference?: (points: Float64Array, count: number, time: number, out: Float64Array) => void;
  perf?: { begin(module: string, phase: string): void; end(): void };
}

/** One completed readback (arrays sized to capacity; the first `count` entries are valid). */
export interface OceanProbeResult {
  /** PassContext.frame and simulation time (s) the heights were computed at. */
  readonly frame: number;
  readonly time: number;
  readonly count: number;
  /** Frames between dispatch and the frame in which the result was seen (set by the probe on completion). */
  readonly latencyFrames: number;
  /** World points (x, y, z, float64) as they were at dispatch. */
  readonly points: Float64Array;
  /** Height of the drawn surface at the point, m above sea level (fade included). */
  readonly drawn: Float32Array;
  /** Full-resolution GPU surface at the point (LOD 0 = Catmull-Rom, no fade), m: view-independent. */
  readonly field: Float32Array;
  /** |r + fade D_h(r) - q| after the solve, m (inversion residual of the drawn surface). */
  readonly residual: Float32Array;
  /** Far-fade weight at the rest point (1 = no fade). */
  readonly fade: Float32Array;
  /** Footprint (m) used for the texture LOD (the LOD's cellSize around the point). */
  readonly footprint: Float32Array;
  /** CPU reference heights (NaN when no reference option). */
  readonly reference: Float64Array;
}

export interface OceanHeightProbe {
  readonly pass: Pass;
  /** World points x, y, z (float64), capacity * 3. Write them, then set count. */
  readonly points: Float64Array;
  count: number;
  enabled: boolean;
  /** Resolves when the probe kernel is compiled (the pass does nothing before). */
  readonly ready: Promise<void>;
  /** Latest completed result (null before the first). Do not keep it across frames: entries are reused. */
  latest(): OceanProbeResult | null;
  /** Called once per completed readback (from a promise callback, not on the render path). */
  onResult: ((r: OceanProbeResult) => void) | null;
  readonly stats: { dispatched: number; completed: number; skippedBusy: number; failed: number };
  /** Footprint the LOD gives the vertices at tangent-plane offset (e, s) from the camera foot (m). */
  footprintAt(e: number, s: number): number;
  /** Debug: footprint >= 0 forces that footprint (m) for every point (0 = LOD 0 = Catmull-Rom everywhere). */
  readonly debug: { footprint: number };
  dispose(): void;
}

interface Entry extends OceanProbeResult {
  frame: number;
  time: number;
  count: number;
  latencyFrames: number;
  state: 0 | 1 | 2; // 0 free, 1 in flight, 2 holds the latest result
  readback: any;
  onDone: (rb: any) => void;
  onFail: (e: unknown) => void;
}

export function createOceanHeightProbe(opts: OceanHeightProbeOptions): OceanHeightProbe {
  const { renderer, fft, lod, origin, shape } = opts;
  const N = opts.capacity ?? 256;
  const cubic = opts.cubic ?? true;
  const fadeNear = opts.fadeNear ?? 200, fadeFar = opts.fadeFar ?? 1200;

  // ------------------------------------------------------------------ GPU resources + kernel
  const ptsArr = new Float32Array(N * 4);
  const ptsAttr = new StorageBufferAttribute(ptsArr, 4);
  ptsAttr.name = 'ocean.parity.points';
  const outAttr = new StorageBufferAttribute(new Float32Array(N * 4), 4);
  outAttr.name = 'ocean.parity.out';
  const pts = storage(ptsAttr, 'vec4', N).toReadOnly();
  const out = storage(outAttr, 'vec4', N);
  const uCount = uniform(0, 'uint');
  const uCamHeight = uniform(0);
  const periods = [CASCADES[0].domain, CASCADES[1].domain, CASCADES[2].domain, CAPILLARY_CASCADE.domain];
  const anchors = periods.map(() => uniform(new Vector2()));
  const anchorOf = (P: number) => {
    const i = periods.indexOf(P);
    if (i < 0) throw new Error(`ocean.parity: no tile anchor for period ${P}`);
    return anchors[i];
  };
  const sampling = createFftSampling(fft);

  // Displacement (east, up, south) at rest offset r (east, south from the camera foot), footprint fp: exactly the
  // vertex shader's sampleDisplacement with tileCoord(P) = anchor_P + r.
  const dispAt = (r: any, fp: any) => sampling.sampleDisplacement((P: number) => anchorOf(P).add(r), { footprint: fp, cubic });
  // Far fade of the vertex shader at rest offset r (distance from the camera, which is uCamHeight above the foot).
  const fadeOf = (r: any) => float(1).sub(smoothstep(fadeNear, fadeFar, length(vec3(r.x, uCamHeight.negate(), r.y))));

  /**
   * Solve r + w(r) D_h(r) = q with a fixed iteration count: PROBE_FIXED_STEPS fixed-point steps, then
   * PROBE_NEWTON_STEPS safeguarded Newton steps (finite-difference Jacobian; plain fixed-point step where
   * det J < 0.1; a step that does not reduce the residual is halved back, like the core's inversion). Returns the
   * best iterate (a var).
   */
  function solve(q: any, fp: any, withFade: boolean) {
    const r = vec2(q).toVar();
    const G = (x: any) => {
      const d = dispAt(x, fp);
      const w = withFade ? fadeOf(x) : float(1);
      return x.add(vec2(d.x, d.z).mul(w));
    };
    Loop(PROBE_FIXED_STEPS, () => {
      r.assign(r.sub(G(r).sub(q)));
    });
    const prevR = vec2(r).toVar(), prevRes = float(1e30).toVar();
    const bestR = vec2(r).toVar(), bestRes = float(1e30).toVar();
    Loop(PROBE_NEWTON_STEPS, () => {
      const g0 = G(r).toVar();
      const e = g0.sub(q).toVar();
      const res = dot(e, e).toVar();
      If(res.lessThan(bestRes), () => { bestRes.assign(res); bestR.assign(r); });
      If(res.greaterThanEqual(prevRes), () => {
        r.assign(prevR.add(r.sub(prevR).mul(0.5)));
      }).Else(() => {
        prevR.assign(r); prevRes.assign(res);
        const gx = G(r.add(vec2(FD_STEP, 0))).toVar();
        const gy = G(r.add(vec2(0, FD_STEP))).toVar();
        const j00 = gx.x.sub(g0.x).div(FD_STEP), j10 = gx.y.sub(g0.y).div(FD_STEP);
        const j01 = gy.x.sub(g0.x).div(FD_STEP), j11 = gy.y.sub(g0.y).div(FD_STEP);
        const det = j00.mul(j11).sub(j01.mul(j10)).toVar();
        const inv = float(1).div(max(det, 0.1));
        const newton = vec2(j11.mul(e.x).sub(j01.mul(e.y)), j00.mul(e.y).sub(j10.mul(e.x))).mul(inv);
        r.assign(r.sub(select(det.greaterThan(0.1), newton, e)));
      });
    });
    const el = G(r).sub(q).toVar();
    If(dot(el, el).lessThan(bestRes), () => { bestR.assign(r); });
    return bestR;
  }

  const kernel = Fn(() => {
    const i = instanceIndex;
    If(i.lessThan(uCount), () => {
      const p = pts.element(i);
      const q = vec2(p.x, p.y).toVar();
      const fp = p.z.toVar();
      // drawn surface (with the far fade)
      const r = solve(q, fp, true);
      const d = dispAt(r, fp).toVar();
      const w = fadeOf(r).toVar();
      const res = length(r.add(vec2(d.x, d.z).mul(w)).sub(q));
      const hDrawn = d.y.mul(w).toVar();
      // field: LOD 0 (Catmull-Rom), no fade
      const rf = solve(q, float(0), false);
      const hField = dispAt(rf, float(0)).y;
      out.element(i).assign(vec4(hDrawn, hField, res, w));
    });
  })().compute(N);
  (kernel as any).name = 'ocean.parity';

  let compiled = false;
  const ready: Promise<void> = Promise.resolve(renderer.compileComputeAsync ? renderer.compileComputeAsync(kernel) : undefined)
    .then(() => { compiled = true; }, (e: unknown) => { console.error('[ocean.parity] compile failed', e); });

  // ------------------------------------------------------------------ ring
  const stats = { dispatched: 0, completed: 0, skippedBusy: 0, failed: 0 };
  let latestEntry: Entry | null = null;
  let frameNow = 0;
  let onResult: ((r: OceanProbeResult) => void) | null = null;
  const entries: Entry[] = [];
  for (let k = 0; k < RING; k++) {
    const rb = new ReadbackBuffer(N * 16);
    rb.name = `ocean.parity.readback${k}`;
    const e: Entry = {
      frame: 0, time: 0, count: 0, latencyFrames: 0, state: 0, readback: rb,
      points: new Float64Array(N * 3), drawn: new Float32Array(N), field: new Float32Array(N),
      residual: new Float32Array(N), fade: new Float32Array(N), footprint: new Float32Array(N),
      reference: new Float64Array(N).fill(NaN),
      onDone: null as unknown as (rb: any) => void, onFail: null as unknown as (e: unknown) => void,
    };
    e.onDone = (target: any) => {
      const f = new Float32Array(target.buffer as ArrayBuffer, 0, N * 4);
      for (let i = 0; i < e.count; i++) {
        e.drawn[i] = f[4 * i]; e.field[i] = f[4 * i + 1]; e.residual[i] = f[4 * i + 2]; e.fade[i] = f[4 * i + 3];
      }
      target.release();
      e.latencyFrames = frameNow - e.frame;
      if (latestEntry && latestEntry !== e) latestEntry.state = 0;
      latestEntry = e;
      e.state = 2;
      stats.completed++;
      if (onResult) onResult(e);
    };
    e.onFail = (err: unknown) => {
      stats.failed++;
      e.state = 0;
      try { e.readback.release(); } catch { /* not mapped */ }
      if (stats.failed < 4) console.error('[ocean.parity] readback failed', err);
    };
    entries.push(e);
  }

  // ------------------------------------------------------------------ CPU per-frame helpers (allocation-free)
  const frame: TangentFrame = { east: { x: 1, y: 0, z: 0 }, north: { x: 0, y: 0, z: -1 }, up: { x: 0, y: 1, z: 0 } };
  const south: Vec3 = { x: 0, y: 0, z: 1 };
  const footW: Vec3 = { x: 0, y: 0, z: 0 };
  const footR: Vec3 = { x: 0, y: 0, z: 0 };
  const wrappedTmp: Vec3 = { x: 0, y: 0, z: 0 };

  /** Footprint (m) the LOD gives the vertices at offset (e, s) from the camera foot (parity-math.ts, tested). */
  const footprintAt = (e: number, s: number): number => lodFootprint(lod.state, lod.params, e, s);

  /** Debug: footprint >= 0 forces that footprint (m) for every point (0 = LOD 0 = Catmull-Rom everywhere). */
  const debug = { footprint: -1 };
  const probe = {
    points: new Float64Array(N * 3),
    count: 0,
    enabled: true,
  };

  const pass: Pass = {
    desc: { id: 'ocean.parity', module: 'world/ocean', stage: 'compute', reads: ['ocean.displacement'], writes: ['ocean.parity'] },
    execute(ctx: PassContext) {
      frameNow = ctx.frame;
      const n = probe.count < N ? probe.count : N;
      if (!probe.enabled || !compiled || n <= 0 || !lod.state.hasPrev) return;
      let e: Entry | null = null;
      for (let k = 0; k < RING; k++) if (entries[k].state === 0) { e = entries[k]; break; }
      if (!e) { stats.skippedBusy++; return; }
      if (opts.perf) opts.perf.begin('world/ocean', 'ocean.parity');
      // Camera foot + tangent frame (the LOD updated them this frame, in ocean.fft).
      const st = lod.state;
      const mp = lod.mesh.position; // render-space foot (lod.ts places the mesh there)
      footR.x = mp.x; footR.y = mp.y; footR.z = mp.z;
      origin.toWorld(footR, footW);
      shape.tangentFrame(footW, frame);
      const E = frame.east, Nn = frame.north;
      south.x = -Nn.x; south.y = -Nn.y; south.z = -Nn.z;
      const bDotE = footR.x * E.x + footR.y * E.y + footR.z * E.z;
      const bDotS = footR.x * south.x + footR.y * south.y + footR.z * south.z;
      for (let k = 0; k < periods.length; k++) {
        const P = periods[k];
        origin.wrapped(P, wrappedTmp);
        (anchors[k].value as Vector2).set(tileAnchor(E, bDotE, st.footU, wrappedTmp, P), tileAnchor(south, bDotS, st.footV, wrappedTmp, P));
      }
      uCamHeight.value = st.height;
      // Points: tangent-plane offsets from the foot (float64 on the CPU), footprint.
      const src = probe.points, dst = e.points;
      for (let i = 0; i < n; i++) {
        const x = src[3 * i], y = src[3 * i + 1], z = src[3 * i + 2];
        dst[3 * i] = x; dst[3 * i + 1] = y; dst[3 * i + 2] = z;
        const dx = x - footW.x, dy = y - footW.y, dz = z - footW.z;
        const qe = dx * E.x + dy * E.y + dz * E.z;
        const qs = dx * south.x + dy * south.y + dz * south.z;
        const fp = debug.footprint >= 0 ? debug.footprint : footprintAt(qe, qs);
        e.footprint[i] = fp;
        ptsArr[4 * i] = qe; ptsArr[4 * i + 1] = qs; ptsArr[4 * i + 2] = fp; ptsArr[4 * i + 3] = 0;
      }
      ptsAttr.needsUpdate = true;
      uCount.value = n;
      e.frame = ctx.frame; e.time = ctx.time; e.count = n; e.state = 1;
      if (opts.reference) opts.reference(dst, n, ctx.time, e.reference);
      ctx.renderer.compute(kernel);
      stats.dispatched++;
      ctx.renderer.getArrayBufferAsync(outAttr, e.readback).then(e.onDone, e.onFail);
      if (opts.perf) opts.perf.end();
    },
  };

  return {
    pass,
    points: probe.points,
    get count() { return probe.count; },
    set count(v: number) { probe.count = v; },
    get enabled() { return probe.enabled; },
    set enabled(v: boolean) { probe.enabled = v; },
    ready,
    latest() { return latestEntry; },
    get onResult() { return onResult; },
    set onResult(f: ((r: OceanProbeResult) => void) | null) { onResult = f; },
    stats,
    footprintAt,
    debug,
    dispose() {
      probe.enabled = false;
      for (const e of entries) e.readback.dispose();
      (kernel as any).dispose?.();
    },
  };
}
