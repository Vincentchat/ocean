// GPU FFT ocean (three r186, WebGPU compute in TSL). Per frame, per cascade: spectrum evolution at the explicit
// draw time, crossfade of two spectrum slots, packed inverse FFT, derived fields (normal + foam compression) and
// compute mip chains. Quality tiers and the pure maths live in fft-tiers.ts (no three import).
//
// TIER INVARIANCE: every tier draws the core's master spectrum components bit-identically (same k, amplitude,
// phase, dispersion): c0 / c1 always on the master 256 grids, c2 on the master grid or its centred sub-block (low
// drops only c2's shortest waves), plus ultra's optional capillary band above the master top. A tier switch only
// changes fine detail; the CPU query (master layout) is tier-independent. Table + measurements: fft-tiers.ts.
//
// Conventions: EXACTLY the core's (spectrum.ts header):
//   h0 = lerp(h0[from], h0[to], w),  w = ocean.blendAt(t),  chop = ocean.choppinessAt(t)
//   phase/(2 pi) = fract(((m * T_int) mod 8192 + m * T_frac) / 8192)   (m = integer dispersion index, u32 math)
//   ht = h0 e^{-i w t} + h0m e^{+i w t}, h0m = conj(h0(-k)) read at the mirrored cell;  kl = |k| + 1e-4
//   packed IFFT (unnormalised, exp(+i k x)):  F1 = ht (1 + kx/kl) -> Hraw + i Da,   F2 = -i (kz/kl) ht -> Db
//   checkerboard s = (-1)^(x+y);  up = -Hraw s,  east = chop Da s,  north = -chop Db s
//   texel (x, y) = tile position a = x L/N (east), b = y L/N (south);  hardware uv = (a_tile + OFF)/L + 0.5/N
// FFT: radix-2, one workgroup per line (N/2 invocations, whole line in workgroup memory, bit-reversed load,
// f64-built twiddles); the spectrum evolution is fused into the row pass; then the column pass (+ finalize in
// place), derive level 0 + mips 1-3 in 8x8 tiles, and reduce kernels of up to 4 mip levels each.
// Per group: 4-5 dispatches; per frame (base + detail): 9-10 dispatches.
//
// OUTPUT: two rgba16float 2D-ARRAY storage textures (Repeat, LinearMipmapLinear, full mip chains written by
// compute, box-averaged, UNNORMALISED) = 2 bindings for the material:
//   base    256 x 256, cascades 0, 1 (Lg = 2):  6 layers
//   detail  N2 x N2 (tier: 64 / 256), cascade 2 (+ capillary band as cascade 3 when enabled; Lg = 1 or 2): 3 Lg layers
//   per texture, local cascade i:  displacement layer i + parity * Lg (this frame; last frame = the other parity:
//     the field at time - frameDt, for motion vectors), texel (up, east, north, dEast/dNorth) metres;
//     normal layer 2 Lg + i: (n.east, n.north, n.up, 1 - compression) (reference tex1 layout: per-cascade normal of
//     the displaced surface; compression = clamp(1 - minEig(J), 0, 1) masked to downwind-facing slopes).
//
// Displacement sampling (createFftSampling): chained lookup (reference spectralField), explicit LOD from the
// footprint, Catmull-Rom (16 exact loads) where LOD < 0.5, blended to trilinear by smoothstep(0, 0.5, LOD) like the
// reference's B-spline. Catmull-Rom = the core query's 'cubic' filter (the core must use it for c0 / c1).
//
// Spectrum slots on the GPU: 2 entries (from / to) of h0 only (h0m read mirrored). Base / detail h0 are the core's
// slot arrays (centred sub-block for a smaller c2); the capillary band is generated in a Web Worker
// (fft-worker.ts) per core slot. Tier change: resources created across tasks, renderer.compileComputeAsync; the old
// tier stays drawn until the new one is ready; only the detail group changes between tiers.
import {
  HalfFloatType, LinearFilter, LinearMipmapLinearFilter, RGBAFormat, RepeatWrapping, StorageArrayTexture,
  StorageBufferAttribute, StorageTextureNode, Vector2,
} from 'three/webgpu';
import {
  Fn, If, clamp, cos, cross, dot, float, floor, fract, globalId, int, ivec2, length, localId, log2, max, mix,
  nodeObject, normalize, select, sin, smoothstep, sqrt, storage, texture, textureStore, uint, uniform,
  uvec2, vec2, vec3, vec4, workgroupArray, workgroupBarrier, workgroupId,
} from 'three/tsl';
import { CASCADES, REFERENCE_SIZE } from '../spectrum.ts';
import type { CascadeDef } from '../spectrum.ts';
import {
  FFT_TIERS, CAPILLARY_CASCADE, CAPILLARY_SIZE, masterH0Subset, masterOmegaSubset, type OceanTier, type FftTierSpec,
} from './fft-tiers.ts';
import { handleFftJob } from './fft-worker.ts';
import type { Pass, PassContext } from './pass-types.ts';

export { FFT_TIERS, TIER_NAMES, tierCascades, CAPILLARY_CASCADE } from './fft-tiers.ts';
export type { OceanTier, FftTierSpec } from './fft-tiers.ts';
/** @deprecated round-1 name. */
export type FftTierConfig = FftTierSpec;
/** Tiles, bands and offsets of the master cascades (identical on every tier). */
export const FFT_CASCADES: readonly CascadeDef[] = CASCADES;
/** Grid size of the base group (cascades 0, 1) on every tier. */
export const BASE_SIZE = REFERENCE_SIZE;
/** @deprecated (round 1) = CAPILLARY_CASCADE.domain. */
export const FINE_DOMAIN = 9 / (1024 / 96);

const LOOP = 8192;

/** Minimal view of the core ocean the adapter reads (index.ts Ocean). */
export interface FftOceanSource {
  getState(): any;
  blendAt(t: number): number;
  choppinessAt(t: number): number;
  expectedHs(): number;
}

export interface OceanFftOptions {
  renderer: any;
  ocean: FftOceanSource;
  tier?: OceanTier;
  /** Draw the capillary band on tiers that allow it (ultra). Default false (opt-in). */
  capillary?: boolean;
  perf?: { begin(module: string, phase: string): void; end(): void };
}

function makeArrayTexture(n: number, layers: number, name: string) {
  const t = new StorageArrayTexture(n, n, layers);
  t.type = HalfFloatType;
  t.format = RGBAFormat;
  t.wrapS = RepeatWrapping;
  t.wrapT = RepeatWrapping;
  t.minFilter = LinearMipmapLinearFilter;
  t.magFilter = LinearFilter;
  t.generateMipmaps = true;
  t.name = name;
  return t;
}

function bitReverse(i: any, bits: number) {
  let r: any = null;
  for (let b = 0; b < bits; b++) {
    const bit = i.shiftRight(uint(b)).bitAnd(uint(1)).shiftLeft(uint(bits - 1 - b));
    r = r === null ? bit : r.bitOr(bit);
  }
  return r;
}
const cmul = (a: any, b: any) => vec2(a.x.mul(b.x).sub(a.y.mul(b.y)), a.x.mul(b.y).add(a.y.mul(b.x)));
const perLayer = (c: any, v: readonly number[]) => (v.length === 1 ? float(v[0]) : select(c.equal(uint(0)), float(v[0]), float(v[1])));
const nextTask = () => new Promise<void>((r) => setTimeout(r, 0));

/**
 * Storage-texture node bound to ONE mip level. three r186 shares one binding between texture nodes of the same
 * texture in a shader (uniform hash = texture uuid), which would put every mip write of a kernel on the first
 * node's level; hashing the level in keeps one binding per written level (subclass, no prototype patching).
 */
class MipStorageTextureNode extends (StorageTextureNode as any) {
  getUniformHash() { return `${(this as any).value.uuid}#mip${(this as any).mipLevel}`; }
}
const mipStore = (tex: any, level: number): any => nodeObject(new (MipStorageTextureNode as any)(tex, null).setMipLevel(level));

function createUniforms() {
  return {
    timeInt: uniform(0, 'uint'),
    timeFrac: uniform(0),
    blend: uniform(0),
    chop: uniform(1),
    fromGpu: uniform(0, 'uint'),
    toGpu: uniform(0, 'uint'),
    windAB: uniform(new Vector2(0.540302306, 0.841470985)),
    parity: uniform(0, 'uint'),
  };
}
type Uniforms = ReturnType<typeof createUniforms>;

/** One cascade group: n x n grid, Lg layers (cascades), its buffers, texture and kernels. */
interface Group {
  readonly n: number;
  readonly Lg: number;
  readonly cascades: readonly CascadeDef[];
  readonly fields: any;
  readonly fieldAttr: any;
  readonly h0Attr: any;
  readonly omega: Float32Array;
  readonly kernels: any[];
  readonly bytes: number;
}

/** Create a group's resources + TSL kernels (cheap JS; called across tasks by the async tier build). */
function buildGroup(u: Uniforms, n: number, cascades: readonly CascadeDef[], omega: Float32Array, name: string): Group {
  const Lg = cascades.length;
  const log2n = Math.round(Math.log2(n));
  const nn = n * n, half = n >> 1;
  const NNu = uint(nn), mask = uint(n - 1);
  const slotStride = uint(Lg * nn);
  const normalLayer = 2 * Lg;
  const curLayer = u.parity.mul(uint(Lg));
  const dk = cascades.map((d) => (2 * Math.PI) / d.domain);
  const cell = cascades.map((d) => d.domain / n);
  const h0Attr = new StorageBufferAttribute(new Float32Array(2 * Lg * nn * 2), 2);
  h0Attr.name = `ocean.fft.${name}.h0`;
  const omegaAttr = new StorageBufferAttribute(omega, 1);
  const tw = new Float32Array(n);
  for (let j = 0; j < half; j++) { tw[2 * j] = Math.cos((2 * Math.PI * j) / n); tw[2 * j + 1] = Math.sin((2 * Math.PI * j) / n); }
  const twAttr = new StorageBufferAttribute(tw, 2);
  const fieldAttr = new StorageBufferAttribute(Lg * nn, 4);
  fieldAttr.name = `ocean.fft.${name}.field`;
  const redOff: number[] = [];
  redOff[3] = 0;
  redOff[7] = (n >> 3) * (n >> 3) * Lg;
  const redCount = redOff[7] + Math.max(1, (n >> 7) * (n >> 7) * Lg);
  const redD = new StorageBufferAttribute(redCount, 4), redN = new StorageBufferAttribute(redCount, 4);
  const fields = makeArrayTexture(n, 3 * Lg, `ocean.fft.${name}`);

  const h0 = storage(h0Attr, 'vec2', h0Attr.count).toReadOnly();
  const om = storage(omegaAttr, 'float', omegaAttr.count).toReadOnly();
  const twn = storage(twAttr, 'vec2', twAttr.count).toReadOnly();
  const work = storage(fieldAttr, 'vec4', fieldAttr.count);
  const rD = storage(redD, 'vec4', redD.count), rN = storage(redN, 'vec4', redN.count);
  const kernels: any[] = [];

  function evolveAt(x: any, y: any, c: any) {
    const i = c.mul(NNu).add(y.mul(uint(n))).add(x);
    const j = c.mul(NNu).add(uint(n).sub(y).bitAnd(mask).mul(uint(n))).add(uint(n).sub(x).bitAnd(mask));
    const fa = u.fromGpu.mul(slotStride), ta = u.toGpu.mul(slotStride);
    const h = mix(h0.element(fa.add(i)), h0.element(ta.add(i)), u.blend).toVar();
    const hmr = mix(h0.element(fa.add(j)), h0.element(ta.add(j)), u.blend).toVar();
    const hm = vec2(hmr.x, hmr.y.negate()).toVar(); // conj(h0(-k))
    const m = om.element(i).toVar();
    const p = uint(m).mul(u.timeInt).bitAnd(uint(LOOP - 1));
    const cyc = fract(float(p).add(m.mul(u.timeFrac)).div(LOOP)).toVar();
    const th = cyc.sub(floor(cyc.add(0.5))).mul(2 * Math.PI).toVar();
    const cs = cos(th).toVar(), sn = sin(th).toVar();
    const htr = h.x.mul(cs).add(h.y.mul(sn)).add(hm.x.mul(cs).sub(hm.y.mul(sn))).toVar();
    const hti = h.y.mul(cs).sub(h.x.mul(sn)).add(hm.y.mul(cs).add(hm.x.mul(sn))).toVar();
    const dkc = perLayer(c, dk).toVar();
    const kx = float(int(x).sub(int(half))).mul(dkc).toVar();
    const kz = float(int(y).sub(int(half))).mul(dkc).toVar();
    const kl = sqrt(kx.mul(kx).add(kz.mul(kz))).add(1e-4).toVar();
    const kxn = kx.div(kl), kzn = kz.div(kl).toVar();
    const one = kxn.add(1).toVar();
    return vec4(htr.mul(one), hti.mul(one), kzn.mul(hti), kzn.mul(htr).negate());
  }

  function lineFft(columns: boolean) {
    return Fn(() => {
      const j = localId.x;
      const line = workgroupId.x, c = workgroupId.y;
      const sh = workgroupArray('vec4', n);
      const base = columns ? c.mul(NNu).add(line) : c.mul(NNu).add(line.mul(uint(n)));
      const stride = uint(columns ? n : 1);
      const j2 = j.add(uint(half));
      if (!columns) {
        sh.element(bitReverse(j, log2n)).assign(evolveAt(j, line, c));
        sh.element(bitReverse(j2, log2n)).assign(evolveAt(j2, line, c));
      } else {
        sh.element(bitReverse(j, log2n)).assign(work.element(base.add(j.mul(stride))));
        sh.element(bitReverse(j2, log2n)).assign(work.element(base.add(j2.mul(stride))));
      }
      workgroupBarrier();
      for (let s = 0; s < log2n; s++) {
        const hs = 1 << s;
        const k = j.bitAnd(uint(hs - 1));
        const a = j.shiftRight(uint(s)).shiftLeft(uint(s + 1)).add(k);
        const b = a.add(uint(hs));
        const w = twn.element(k.shiftLeft(uint(log2n - 1 - s))).toVar();
        const xa = sh.element(a).toVar();
        const xb = sh.element(b).toVar();
        const t = vec4(cmul(w, xb.xy), cmul(w, xb.zw)).toVar();
        sh.element(b).assign(xa.sub(t));
        sh.element(a).assign(xa.add(t));
        workgroupBarrier();
      }
      if (!columns) {
        work.element(base.add(j)).assign(sh.element(j));
        work.element(base.add(j2)).assign(sh.element(j2));
      } else {
        for (const yy of [j, j2]) {
          const v = sh.element(yy).toVar();
          const sg = float(1).sub(float(line.add(yy).bitAnd(uint(1))).mul(2));
          work.element(base.add(yy.mul(stride))).assign(vec4(v.x.mul(sg).negate(), v.y.mul(sg).mul(u.chop), v.z.mul(sg).mul(u.chop).negate(), 0));
        }
      }
    })().compute([n, Lg, 1], [half, 1, 1]);
  }
  kernels.push(lineFft(false), lineFft(true));

  function derive(x: any, y: any, c: any) {
    const at = (xx: any, yy: any) => work.element(c.mul(NNu).add(yy.bitAnd(mask).mul(uint(n))).add(xx.bitAnd(mask)));
    const ctr = at(x, y).toVar();
    const px = at(x.add(uint(1)), y).toVar(), mx = at(x.add(uint(n - 1)), y).toVar();
    const py = at(x, y.add(uint(1))).toVar(), my = at(x, y.add(uint(n - 1))).toVar();
    const inv2c = float(1).div(perLayer(c, cell).mul(2));
    const dA = vec3(px.y.sub(mx.y), px.x.sub(mx.x), mx.z.sub(px.z)).mul(inv2c).toVar(); // d(east, up, south)/da
    const dB = vec3(py.y.sub(my.y), py.x.sub(my.x), my.z.sub(py.z)).mul(inv2c).toVar(); // d/db
    const tx = vec3(1, 0, 0).add(dA).toVar();
    const tz = vec3(0, 0, 1).add(dB).toVar();
    const nr = normalize(cross(tz, tx)).toVar();
    const dd = tx.x.sub(tz.z);
    const eig = tx.x.add(tz.z).sub(sqrt(max(dd.mul(dd).add(tx.z.mul(tz.x).mul(4)), 0))).mul(0.5);
    const comp = clamp(float(1).sub(eig), 0, 1).mul(smoothstep(0, 0.2, dot(vec2(nr.x, nr.z), u.windAB)));
    return { disp: vec4(ctr.x, ctr.y, ctr.z, dB.x.negate()), nrm: vec4(nr.x, nr.z.negate(), nr.y, float(1).sub(comp)) };
  }

  function reduceKernel(src: number, T: number, inOff: number, outOff: number) {
    const k = Math.round(Math.log2(T));
    const first = src < 0 ? 0 : src + 1;
    const lastLevel = src < 0 ? k : src + k;
    const tex: any[] = [];
    for (let lv = first; lv <= lastLevel; lv++) tex[lv] = mipStore(fields, lv);
    const ns = src < 0 ? n : n >> src;
    return Fn(() => {
      const lx = localId.x, ly = localId.y, c = workgroupId.z;
      const gx = globalId.x, gy = globalId.y;
      const shD = workgroupArray('vec4', T * T), shN = workgroupArray('vec4', T * T);
      const li = ly.mul(uint(T)).add(lx);
      const d = vec4(0).toVar(), m4 = vec4(0).toVar();
      if (src < 0) {
        const r = derive(gx, gy, c);
        d.assign(r.disp); m4.assign(r.nrm);
        const xy = uvec2(gx, gy);
        textureStore(tex[0].depth(int(c.add(curLayer))), xy, d);
        textureStore(tex[0].depth(int(c.add(uint(normalLayer)))), xy, m4);
      } else {
        const idx = uint(inOff).add(c.mul(uint(ns * ns))).add(gy.mul(uint(ns))).add(gx);
        d.assign(rD.element(idx)); m4.assign(rN.element(idx));
      }
      shD.element(li).assign(d); shN.element(li).assign(m4);
      workgroupBarrier();
      const lv0 = src < 0 ? 0 : src;
      for (let s = 1; s <= k; s++) {
        const m = T >> s;
        const on = lx.lessThan(uint(m)).and(ly.lessThan(uint(m)));
        const a = vec4(0).toVar(), b = vec4(0).toVar();
        If(on, () => {
          const i00 = ly.mul(uint(2 * T)).add(lx.mul(uint(2)));
          a.assign(shD.element(i00).add(shD.element(i00.add(uint(1)))).add(shD.element(i00.add(uint(T)))).add(shD.element(i00.add(uint(T + 1)))).mul(0.25));
          b.assign(shN.element(i00).add(shN.element(i00.add(uint(1)))).add(shN.element(i00.add(uint(T)))).add(shN.element(i00.add(uint(T + 1)))).mul(0.25));
        });
        workgroupBarrier();
        const lv = lv0 + s;
        If(on, () => {
          shD.element(li).assign(a); shN.element(li).assign(b);
          const xy = uvec2(workgroupId.x.mul(uint(m)).add(lx), workgroupId.y.mul(uint(m)).add(ly));
          textureStore(tex[lv].depth(int(c.add(curLayer))), xy, a);
          textureStore(tex[lv].depth(int(c.add(uint(normalLayer)))), xy, b);
          if (s === k && lv < log2n) {
            const no = n >> lv;
            const o = uint(outOff).add(c.mul(uint(no * no))).add(xy.y.mul(uint(no))).add(xy.x);
            rD.element(o).assign(a); rN.element(o).assign(b);
          }
        });
        workgroupBarrier();
      }
    })().compute([ns / T, ns / T, Lg], [T, T, 1]);
  }
  kernels.push(reduceKernel(-1, 8, 0, redOff[3]));
  let lv = 3;
  while (lv < log2n) {
    const T = Math.min(16, n >> lv);
    const k = Math.round(Math.log2(T));
    kernels.push(reduceKernel(lv, T, redOff[lv], redOff[lv + k] ?? 0));
    lv += k;
  }
  let texels = 0;
  for (let l = 0; l <= log2n; l++) texels += (n >> l) * (n >> l);
  const bytes = h0Attr.array.byteLength + omega.byteLength + tw.byteLength + Lg * nn * 16 + 2 * redCount * 16 + texels * 8 * 3 * Lg;
  return { n, Lg, cascades, fields, fieldAttr, h0Attr, omega, kernels, bytes };
}

function disposeGroup(g: Group) {
  g.fields.dispose();
  for (const k of g.kernels) k.dispose?.();
}

// ------------------------------------------------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------------------------------------------------
export interface OceanFftTextures {
  /** Base group texture (cascades 0, 1; 256^2): displacement + normal layers (both nodes = one texture). */
  readonly displacementBase: any;
  readonly normalBase: any;
  /** Detail group texture (cascade 2 [+ capillary band]; N2^2). */
  readonly displacementDetail: any;
  readonly normalDetail: any;
}

export interface OceanFft {
  /** Render-graph pass (stage compute). execute() allocates nothing in steady state. */
  readonly pass: Pass;
  /** Resolves when the first tier is compiled and has run once. */
  readonly ready: Promise<void>;
  readonly tier: OceanTier;
  readonly pendingTier: OceanTier | null;
  readonly capillary: boolean;
  /** Texture NODES (detail values swap on tier change; build samples from them, e.g. via createFftSampling). */
  readonly nodes: OceanFftTextures;
  readonly uniforms: {
    /** Displacement ping-pong parity of this frame / last frame (0 or 1), float. Layer = i + parity * Lg. */
    readonly parity: any;
    readonly prevParity: any;
    /** Detail group grid size N2 (float), its layer count Lg (1, or 2 with the capillary band), int N2 - 1. */
    readonly detailSize: any;
    readonly detailLayers: any;
    readonly detailMask: any;
    /** Weight of the capillary band (0 when off). */
    readonly capillaryGain: any;
    /** @deprecated always 0. */
    readonly fineEnabled: any;
    /** Slope variance of the master spectrum the tier does not draw (low: c2 above its grid), m^2/m^2. */
    readonly unresolvedSlopeVariance: any;
    /** Wind direction (towards) of the drawn (crossfaded) spectrum in the tile frame (east, south). */
    readonly windAB: any;
  };
  setTier(t: OceanTier): void;
  /** Enable / disable the capillary band (only drawn on tiers that allow it); rebuilds the detail group async. */
  setCapillary(on: boolean): void;
  readonly debug: {
    /** f32 (up, east, north, 0) fields of a group, layer-major (resolved after this frame's GPU work). */
    readFields(group: 'base' | 'detail'): Promise<Float32Array>;
    size(group: 'base' | 'detail'): number;
    layers(group: 'base' | 'detail'): number;
    /** h0 (h0-only layout) per layer of a group that the GPU uses for core slot s, or null. */
    spectrum(group: 'base' | 'detail', slot: number): Float32Array[] | null;
    /** Omega table (layers concatenated) of a group. */
    omega(group: 'base' | 'detail'): Float32Array | null;
    memoryBytes(): number;
    lastCompileMs(): number;
    lastBuildMainThreadMs(): number;
  };
  dispose(): void;
}

interface TierState {
  readonly tier: OceanTier;
  readonly capillary: boolean;
  readonly detail: Group;
  readonly kernels: any[];
  /** GPU slot entries (shared by both groups): core slot index / version held (-1 = none). */
  readonly entrySlot: Int32Array;
  readonly entryVer: Int32Array;
}

export function createOceanFft(opts: OceanFftOptions): OceanFft {
  const { renderer, ocean } = opts;
  const u = createUniforms();
  const st0 = ocean.getState();
  const g = st0.gravity as number;
  const seed = (st0.seed ?? 1) as number;

  // ---- worker (capillary band, dropped-slope sums); synchronous fallback where Worker is unavailable
  let worker: Worker | null = null;
  try {
    if (typeof Worker !== 'undefined') worker = new Worker(new URL('./fft-worker.ts', import.meta.url), { type: 'module' });
  } catch { worker = null; }
  let jobId = 0;
  const jobs = new Map<number, (r: any) => void>();
  if (worker) {
    worker.onmessage = (e: MessageEvent) => {
      const cb = jobs.get(e.data.id);
      jobs.delete(e.data.id);
      if (e.data.error) console.error('[ocean.fft] worker job failed', e.data.error);
      cb?.(e.data);
    };
  }
  function job(msg: any, transfer: Transferable[] = []): Promise<any> {
    const id = ++jobId;
    msg.id = id;
    if (!worker) return Promise.resolve(handleFftJob(msg).result);
    return new Promise((resolve) => { jobs.set(id, resolve); worker!.postMessage(msg, transfer); });
  }

  // ---- per (core slot, version) caches: capillary h0 (worker), dropped c2 slope (worker)
  interface CapEntry { version: number; h0: Float32Array | null }
  const cap: CapEntry[] = [];
  let capOmega: Float32Array | null = null;
  function capillaryFor(s: number, state: any): Float32Array | null {
    const sl = state.slots[s];
    if (!sl || !(sl.version > 0)) return null;
    const e = cap[s];
    if (e && e.version === sl.version) return e.h0;
    const entry: CapEntry = { version: sl.version, h0: null };
    cap[s] = entry;
    job({ kind: 'capillary', params: { ...sl.params }, seed, g, n: CAPILLARY_SIZE }).then((r) => {
      if (!capOmega && r.omega) capOmega = r.omega;
      if (cap[s] === entry && r.h0) entry.h0 = r.h0;
    });
    return null;
  }
  const dropped = new Map<string, number>();
  function droppedFor(s: number, n: number, state: any): number {
    const sl = state.slots[s];
    if (!sl || n >= REFERENCE_SIZE) return 0;
    const k = `${s}:${sl.version}:${n}`;
    const v = dropped.get(k);
    if (v !== undefined) return v < 0 ? 0 : v;
    dropped.set(k, -1);
    job({ kind: 'dropped', h0: sl.h0[2].slice(), n, domain: CASCADES[2].domain }).then((r) => { dropped.set(k, r.slope ?? 0); });
    return 0;
  }

  // ---- groups
  const baseOmega = new Float32Array(2 * REFERENCE_SIZE * REFERENCE_SIZE);
  baseOmega.set(st0.omegaIndex[0], 0); baseOmega.set(st0.omegaIndex[1], REFERENCE_SIZE * REFERENCE_SIZE);
  const base = buildGroup(u, REFERENCE_SIZE, [CASCADES[0], CASCADES[1]], baseOmega, 'base');
  const baseEntrySlot = new Int32Array([-1, -1]), baseEntryVer = new Int32Array([-1, -1]);

  let active: TierState | null = null;
  let pending: TierState | null = null;
  let pendingTier: OceanTier | null = null;
  let buildToken = 0;
  let compileMs = 0, buildMainMs = 0;
  let requested: OceanTier = opts.tier ?? 'high';
  let wantCapillary = !!opts.capillary;
  let retired: TierState | null = null;
  let retireFrames = 0;
  let baseCompiled = false;

  const detailCascades = (t: OceanTier, capOn: boolean) => (FFT_TIERS[t].capillary && capOn ? [CASCADES[2], CAPILLARY_CASCADE] : [CASCADES[2]]);

  async function buildTier(t: OceanTier, capOn: boolean, token: number, prebuilt: Group | null): Promise<TierState | null> {
    let main = 0;
    const tick = () => performance.now();
    const cs = prebuilt ? prebuilt.cascades : detailCascades(t, capOn);
    const n2 = FFT_TIERS[t].sizes[2];
    let detail = prebuilt;
    if (!detail) {
      if (cs.length > 1) { // capillary omega (worker; also warms the spectrum of the slots in use)
        while (!capOmega) {
          const s = ocean.getState();
          capillaryFor(s.toSlot, s);
          await new Promise((r) => setTimeout(r, 16));
          if (token !== buildToken) return null;
        }
      }
      let t0 = tick();
      const s = ocean.getState();
      const nn2 = n2 * n2;
      const om = new Float32Array(cs.length * nn2);
      om.set(masterOmegaSubset(s.omegaIndex[2], n2), 0);
      if (cs.length > 1) om.set(capOmega!, nn2);
      main += tick() - t0; await nextTask(); if (token !== buildToken) return null; t0 = tick();
      detail = buildGroup(u, n2, cs, om, `detail.${t}${cs.length > 1 ? '.cap' : ''}`);
      main += tick() - t0; await nextTask(); if (token !== buildToken) { disposeGroup(detail); return null; }
    }
    // the slots in use must be uploadable (capillary spectra generated) before the tier goes live
    if (cs.length > 1) {
      for (;;) {
        const s = ocean.getState();
        const a = capillaryFor(s.toSlot, s), b = capillaryFor(s.fromSlot, s);
        if (a && b) break;
        await new Promise((r) => setTimeout(r, 16));
        if (token !== buildToken) { disposeGroup(detail); return null; }
      }
    }
    const kernels = [...base.kernels, ...detail.kernels];
    const c0 = tick();
    await renderer.compileComputeAsync(baseCompiled ? detail.kernels : kernels);
    baseCompiled = true;
    compileMs = tick() - c0;
    buildMainMs = main;
    if (token !== buildToken) { disposeGroup(detail); return null; }
    return { tier: t, capillary: cs.length > 1, detail, kernels, entrySlot: new Int32Array([-1, -1]), entryVer: new Int32Array([-1, -1]) };
  }

  function startBuild(t: OceanTier, prebuilt: Group | null = null) {
    const token = ++buildToken;
    pendingTier = t;
    const capOn = wantCapillary;
    buildTier(t, capOn, token, prebuilt).then((ts) => {
      if (!ts) return;
      if (token !== buildToken) { disposeGroup(ts.detail); return; }
      if (pending) disposeGroup(pending.detail);
      pending = ts;
    }, (e) => { console.error('[ocean.fft] tier build failed', e); if (token === buildToken) pendingTier = null; });
  }

  // Texture nodes: the first tier's detail texture exists from the start (three merges texture nodes sharing a
  // texture into one binding when a shader is built, so no shared placeholder).
  const firstDetail = (() => {
    const t = requested, cs = detailCascades(t, false), n2 = FFT_TIERS[t].sizes[2];
    return buildGroup(u, n2, cs, masterOmegaSubset(st0.omegaIndex[2], n2), `detail.${t}`);
  })();
  const nodes: OceanFftTextures = {
    displacementBase: texture(base.fields), normalBase: texture(base.fields),
    displacementDetail: texture(firstDetail.fields), normalDetail: texture(firstDetail.fields),
  };
  const uniforms = {
    parity: uniform(0),
    prevParity: uniform(0),
    detailSize: uniform(firstDetail.n),
    detailLayers: uniform(firstDetail.Lg),
    detailMask: uniform(firstDetail.n - 1, 'int'),
    capillaryGain: uniform(0),
    fineEnabled: uniform(0),
    unresolvedSlopeVariance: uniform(0),
    windAB: u.windAB,
  };

  let resolveReady: () => void = () => {};
  const ready = new Promise<void>((r) => { resolveReady = r; });
  let readyResolved = false;
  let parity = 0;
  let freshHistory = true;

  /** Upload core slot s into GPU entry e of the base group (h0-only copy of the master arrays). */
  function uploadBase(e: number, s: number, state: any) {
    const nn = REFERENCE_SIZE * REFERENCE_SIZE, arr = base.h0Attr.array as Float32Array;
    for (let c = 0; c < 2; c++) {
      const src = state.slots[s].h0[c], off = (e * 2 + c) * nn * 2;
      for (let i = 0; i < nn; i++) { arr[off + 2 * i] = src[4 * i]; arr[off + 2 * i + 1] = src[4 * i + 1]; }
    }
    base.h0Attr.addUpdateRange(e * 2 * nn * 2, 2 * nn * 2);
    base.h0Attr.needsUpdate = true;
  }
  /** Upload core slot s into entry e of the detail group; false if its capillary band is not generated yet. */
  function uploadDetail(ts: TierState, e: number, s: number, state: any): boolean {
    const d = ts.detail, n = d.n, nn = n * n, arr = d.h0Attr.array as Float32Array;
    let capH0: Float32Array | null = null;
    if (d.Lg > 1) { capH0 = capillaryFor(s, state); if (!capH0) return false; }
    arr.set(masterH0Subset(state.slots[s].h0[2], n), (e * d.Lg) * nn * 2);
    if (capH0) arr.set(capH0, (e * d.Lg + 1) * nn * 2);
    d.h0Attr.addUpdateRange(e * d.Lg * nn * 2, d.Lg * nn * 2);
    d.h0Attr.needsUpdate = true;
    return true;
  }
  /** Make core slot s resident in an entry (not `keep`) of both groups; returns the entry or -1. */
  function resident(ts: TierState, s: number, keep: number, state: any): number {
    const ver = state.slots[s]?.version ?? -1;
    if (!(ver > 0)) return -1;
    for (let e = 0; e < 2; e++) if (ts.entrySlot[e] === s && ts.entryVer[e] === ver) return e;
    const e = keep === 0 ? 1 : keep === 1 ? 0 : (ts.entrySlot[0] === s ? 0 : ts.entrySlot[1] === s ? 1 : ts.entrySlot[0] === -1 ? 0 : 1);
    if (!uploadDetail(ts, e, s, state)) { // capillary band not generated yet: keep an older copy of this slot
      for (let q = 0; q < 2; q++) if (ts.entrySlot[q] === s) return q;
      return -1;
    }
    if (baseEntrySlot[e] !== s || baseEntryVer[e] !== ver) { uploadBase(e, s, state); baseEntrySlot[e] = s; baseEntryVer[e] = ver; }
    ts.entrySlot[e] = s; ts.entryVer[e] = ver;
    return e;
  }

  function activate(ts: TierState) {
    if (retired) { disposeGroup(retired.detail); retired = null; }
    if (active) { retired = active; retireFrames = 3; }
    active = ts;
    pending = null;
    pendingTier = null;
    // the base group is shared: its entries stay valid; the detail entries of the new tier start empty and are
    // filled with the same entry indices as the base group (resident() keeps them aligned)
    for (let e = 0; e < 2; e++) {
      if (baseEntrySlot[e] >= 0) {
        const st = ocean.getState();
        const s = baseEntrySlot[e];
        if (st.slots[s] && st.slots[s].version === baseEntryVer[e] && uploadDetail(ts, e, s, st)) { ts.entrySlot[e] = s; ts.entryVer[e] = baseEntryVer[e]; }
      }
    }
    nodes.displacementDetail.value = ts.detail.fields;
    nodes.normalDetail.value = ts.detail.fields;
    uniforms.detailSize.value = ts.detail.n;
    uniforms.detailLayers.value = ts.detail.Lg;
    uniforms.detailMask.value = ts.detail.n - 1;
    uniforms.capillaryGain.value = ts.capillary ? 1 : 0;
    freshHistory = true;
    // the first tier is built without the capillary band (its omega comes from the worker): add it now if wanted
    if (!ts.capillary && FFT_TIERS[ts.tier].capillary && wantCapillary && !pendingTier) startBuild(ts.tier);
  }

  startBuild(requested, firstDetail);

  const pass: Pass = {
    desc: { id: 'ocean.fft', module: 'world/ocean', stage: 'compute', reads: [], writes: ['ocean.displacement', 'ocean.normal'] },
    execute(ctx: PassContext) {
      if (opts.perf) opts.perf.begin('world/ocean', 'ocean.fft');
      if (retired && --retireFrames <= 0) { disposeGroup(retired.detail); retired = null; }
      if (pending) activate(pending);
      if (!active) { if (opts.perf) opts.perf.end(); return; }
      const state = ocean.getState();
      const t = ctx.time;
      const crossing = state.fromSlot !== state.toSlot;
      const w = crossing ? ocean.blendAt(t) : 1;
      const needFrom = crossing && w < 1;
      let holdFrom = -1;
      if (needFrom) for (let e = 0; e < 2; e++) if (active.entrySlot[e] === state.fromSlot) holdFrom = e;
      const eTo = resident(active, state.toSlot, holdFrom, state);
      let eFrom = eTo;
      if (needFrom && eTo >= 0) { const e = resident(active, state.fromSlot, eTo, state); if (e >= 0) eFrom = e; }
      else if (state.pendingSlot >= 0 && eTo >= 0) resident(active, state.pendingSlot, eTo, state); // prefetch
      if (eTo >= 0) {
        u.fromGpu.value = eFrom;
        u.toGpu.value = eTo;
        u.blend.value = eFrom === eTo ? 1 : w;
      } else if (active.entrySlot[0] < 0 && active.entrySlot[1] < 0) { if (opts.perf) opts.perf.end(); return; }
      // (else: the new slot's capillary band is still being generated: keep drawing the resident spectra)
      let tm = t % LOOP;
      if (tm < 0) tm += LOOP;
      const ti = Math.floor(tm);
      u.timeInt.value = ti;
      u.timeFrac.value = tm - ti;
      u.chop.value = ocean.choppinessAt(t);
      const wa = state.slots[state.fromSlot].params.windDirection as number, wb = state.slots[state.toSlot].params.windDirection as number;
      let dw = (wb - wa) % (2 * Math.PI);
      if (dw > Math.PI) dw -= 2 * Math.PI; else if (dw < -Math.PI) dw += 2 * Math.PI;
      const wd = wa + dw * w;
      (u.windAB.value as Vector2).set(Math.cos(wd), -Math.sin(wd));
      const n2 = active.detail.n;
      const dA = droppedFor(state.fromSlot, n2, state), dB = droppedFor(state.toSlot, n2, state);
      uniforms.unresolvedSlopeVariance.value = dA + (dB - dA) * (u.blend.value as number);
      parity ^= 1;
      u.parity.value = parity;
      uniforms.parity.value = parity;
      uniforms.prevParity.value = freshHistory ? parity : 1 - parity;
      freshHistory = false;
      ctx.renderer.compute(active.kernels);
      if (!readyResolved) { readyResolved = true; resolveReady(); }
      if (opts.perf) opts.perf.end();
    },
  };

  const groupOf = (gname: 'base' | 'detail') => (gname === 'base' ? base : active?.detail ?? null);
  return {
    pass,
    ready,
    get tier() { return active ? active.tier : requested; },
    get pendingTier() { return pendingTier; },
    get capillary() { return active ? active.capillary : false; },
    nodes,
    uniforms,
    setTier(t: OceanTier) {
      if (!FFT_TIERS[t]) throw new Error(`ocean.fft: unknown tier ${t}`);
      requested = t;
      if (pendingTier === t) return;
      const capOn = FFT_TIERS[t].capillary && wantCapillary;
      if (!pendingTier && active && active.tier === t && active.capillary === capOn) return;
      if (pending) { disposeGroup(pending.detail); pending = null; }
      if (active && active.tier === t && active.capillary === capOn) { buildToken++; pendingTier = null; return; }
      startBuild(t);
    },
    setCapillary(on: boolean) {
      if (wantCapillary === on) return;
      wantCapillary = on;
      const t = requested;
      if (pending) { disposeGroup(pending.detail); pending = null; }
      if (active && active.tier === t && active.capillary === (FFT_TIERS[t].capillary && on)) { buildToken++; pendingTier = null; return; }
      startBuild(t);
    },
    debug: {
      async readFields(gname) {
        const gr = groupOf(gname);
        if (!gr) throw new Error('no active tier');
        return new Float32Array(await renderer.getArrayBufferAsync(gr.fieldAttr));
      },
      size(gname) { return groupOf(gname)?.n ?? 0; },
      layers(gname) { return groupOf(gname)?.Lg ?? 0; },
      spectrum(gname, s) {
        const st = ocean.getState(), sl = st.slots[s];
        if (!sl || !active) return null;
        if (gname === 'base') return [masterH0Subset(sl.h0[0], REFERENCE_SIZE), masterH0Subset(sl.h0[1], REFERENCE_SIZE)];
        const out = [masterH0Subset(sl.h0[2], active.detail.n)];
        if (active.detail.Lg > 1) { const c = cap[s]; if (!c?.h0 || c.version !== sl.version) return null; out.push(c.h0); }
        return out;
      },
      omega(gname) { return groupOf(gname)?.omega ?? null; },
      memoryBytes() { return base.bytes + (active ? active.detail.bytes : 0); },
      lastCompileMs() { return compileMs; },
      lastBuildMainThreadMs() { return buildMainMs; },
    },
    dispose() {
      buildToken++;
      if (active) disposeGroup(active.detail);
      if (pending) disposeGroup(pending.detail);
      if (retired) disposeGroup(retired.detail);
      disposeGroup(base);
      active = pending = retired = null;
      pendingTier = null;
      worker?.terminate();
    },
  };
}

// ------------------------------------------------------------------------------------------------------------
// Sampling helpers (TSL) for the material / LOD / foam.
// ------------------------------------------------------------------------------------------------------------
export interface FftSampleOptions {
  /** Footprint in metres (vertex: grid cell size; fragment: max |d rest / d screen|). Default 0 = LOD 0. */
  footprint?: any;
  /** Sample LAST frame's displacement (motion vectors). */
  previous?: boolean;
  /** Catmull-Rom where LOD < 0.5 (default: true for sampleDisplacement, false for sampleSurface's chaining). */
  cubic?: boolean;
  /** Only the 3 master cascades (no capillary band), e.g. to compare with the CPU query. */
  threeCascades?: boolean;
}

export interface FftSurfaceSample {
  /** Chained displacement, local mesh axes (x east, y up, z south), metres. */
  displacement: any;
  /** Normalised surface normal, local mesh axes (x east, y up, z south). */
  normal: any;
  /** Whitecap compression (reference: max(C - sum(1 - compression_i), 0)), >= 0. */
  compression: any;
  /** Normal shortening from the mips (reference spectralVariance), 0..1 (+ unresolvedSlopeVariance not included). */
  variance: any;
}

export interface FftSampling {
  /** Chained displacement at the rest point, local mesh axes (x east, y up, z south), metres. */
  sampleDisplacement(tileCoord: (period: number) => any, opts?: FftSampleOptions): any;
  /** Displacement + per-cascade normals (fragment / foam). */
  sampleSurface(tileCoord: (period: number) => any, opts?: FftSampleOptions): FftSurfaceSample;
  /** Raw normal texel (n.east, n.north, n.up, 1 - compression) of master cascade c (0..2) at tile coordinate tc. */
  normal(c: number, tc: any, lod: any): any;
  /** Hardware uv of cascade c at tile coordinate tc. */
  uv(c: number, tc: any): any;
  /** Number of master cascades (3); the capillary band is cascade 3 when enabled. */
  readonly cascades: number;
}

export function createFftSampling(fft: OceanFft): FftSampling {
  const { nodes, uniforms } = fft;
  const C = [...CASCADES, CAPILLARY_CASCADE];
  const isBase = (c: number) => c < 2;
  const sizeOf = (c: number) => (isBase(c) ? float(BASE_SIZE) : uniforms.detailSize);
  const maskOf = (c: number) => (isBase(c) ? int(BASE_SIZE - 1) : uniforms.detailMask);
  const lgOf = (c: number) => (isBase(c) ? float(2) : uniforms.detailLayers);
  const local = (c: number) => (isBase(c) ? c : c - 2);
  const dispTex = (c: number) => (isBase(c) ? nodes.displacementBase : nodes.displacementDetail);
  const nrmTex = (c: number) => (isBase(c) ? nodes.normalBase : nodes.normalDetail);
  const uvOf = (c: number, tc: any) => tc.add(C[c].offset).div(C[c].domain).add(sizeOf(c).reciprocal().mul(0.5));
  const lodOf = (c: number, footprint: any) => (footprint ? max(log2(max(footprint, 1e-6).mul(sizeOf(c)).div(C[c].domain)), 0) : float(0));
  const normalLayer = (c: number) => int(lgOf(c).mul(2).add(local(c)));

  function catmullRom(c: number, tc: any, layer: any) {
    const s = tc.add(C[c].offset).div(C[c].domain).mul(sizeOf(c)).toVar();
    const f = floor(s).toVar();
    const t = s.sub(f).toVar();
    const t2 = t.mul(t).toVar(), t3 = t2.mul(t).toVar();
    const W = [
      t3.mul(-0.5).add(t2).sub(t.mul(0.5)).toVar(),
      t3.mul(1.5).sub(t2.mul(2.5)).add(1).toVar(),
      t3.mul(-1.5).add(t2.mul(2)).add(t.mul(0.5)).toVar(),
      t3.mul(0.5).sub(t2.mul(0.5)).toVar(),
    ];
    const b = ivec2(f).toVar();
    let sum: any = null;
    for (let r = 0; r < 4; r++) {
      let row: any = null;
      const yy = b.y.add(r - 1).bitAnd(maskOf(c)).toVar();
      for (let q = 0; q < 4; q++) {
        const xx = b.x.add(q - 1).bitAnd(maskOf(c)).toVar();
        const v = dispTex(c).load(ivec2(xx, yy)).depth(layer).level(int(0)).mul(W[q].x);
        row = row === null ? v : row.add(v);
      }
      const rr = row.mul(W[r].y);
      sum = sum === null ? rr : sum.add(rr);
    }
    return sum;
  }

  function fetchDisp(c: number, tc: any, lod: any, previous: boolean, cubic: boolean) {
    const layer = int((previous ? uniforms.prevParity : uniforms.parity).mul(lgOf(c)).add(local(c))).toVar();
    const tri = dispTex(c).sample(uvOf(c, tc)).depth(layer).level(lod);
    if (!cubic || c > 2) return tri;
    return Fn(() => {
      const v = vec4(tri).toVar();
      If(lod.lessThan(0.5), () => { v.assign(mix(catmullRom(c, tc, layer), tri, smoothstep(0, 0.5, lod))); });
      return v;
    })();
  }

  function chain(tileCoord: (p: number) => any, o: FftSampleOptions | undefined, withNormal: boolean, cubicDefault: boolean) {
    const previous = !!o?.previous;
    const cubic = o?.cubic ?? cubicDefault;
    const count = o?.threeCascades ? 3 : 4;
    let sum: any = null, nSum: any = null, foamSum: any = null, varSum: any = null;
    for (let c = 0; c < count; c++) {
      if (c === 3 && !tileCoord(C[3].domain)) break; // caller has no tile coordinate for the capillary tile
      const lod = lodOf(c, o?.footprint).toVar();
      const tc = (sum === null ? tileCoord(C[c].domain) : tileCoord(C[c].domain).add(vec2(sum.y, sum.z.negate()))).toVar();
      const gain = c === 3 ? uniforms.capillaryGain : null;
      let d: any = vec4(fetchDisp(c, tc, lod, previous, cubic)).toVar();
      if (gain) d = d.mul(gain);
      sum = (sum === null ? d.xyz : sum.add(d.xyz)).toVar();
      if (withNormal) {
        const nm = nrmTex(c).sample(uvOf(c, tc)).depth(normalLayer(c)).level(lod);
        let tilt: any = nm.xyz.sub(vec3(0, 0, 1));
        let foam: any = float(1).sub(nm.w);
        let short: any = max(float(1).sub(length(nm.xyz)), 0);
        if (gain) { tilt = tilt.mul(gain); foam = foam.mul(gain); short = short.mul(gain); }
        nSum = nSum === null ? tilt : nSum.add(tilt);
        foamSum = foamSum === null ? foam : foamSum.add(foam);
        varSum = varSum === null ? short : varSum.add(short);
      }
    }
    return { sum, nSum, foamSum, varSum };
  }

  return {
    cascades: 3,
    uv: uvOf,
    normal(c, tc, lod) { return nrmTex(c).sample(uvOf(c, tc)).depth(normalLayer(c)).level(lod); },
    sampleDisplacement(tileCoord, o) {
      const { sum } = chain(tileCoord, o, false, true);
      return vec3(sum.y, sum.x, sum.z.negate());
    },
    sampleSurface(tileCoord, o) {
      const { sum, nSum, foamSum, varSum } = chain(tileCoord, o, true, false);
      const nLocal = nSum.add(vec3(0, 0, 1)); // (east, north, up)
      return {
        displacement: vec3(sum.y, sum.x, sum.z.negate()),
        normal: normalize(vec3(nLocal.x, nLocal.z, nLocal.y.negate())),
        compression: max(foamSum, 0),
        variance: clamp(varSum, 0, 1),
      };
    },
  };
}
