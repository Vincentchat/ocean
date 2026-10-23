// GPU FFT ocean (three r186, WebGPU compute in TSL). Per frame, per cascade: spectrum evolution at the explicit
// draw time, crossfade of two spectrum slots, packed inverse FFT, derived fields (normal + foam compression) and
// compute mip chains. Quality tiers, band layout and the pure maths live in fft-tiers.ts (no three import).
//
// Conventions: EXACTLY the core's (spectrum.ts header), for the tier's band layout:
//   h0 = lerp(h0[from], h0[to], w),  w = ocean.blendAt(t),  chop = ocean.choppinessAt(t)
//   phase/(2 pi) = fract(((m * T_int) mod 8192 + m * T_frac) / 8192)   (m = integer dispersion index, u32 math)
//   ht = h0 e^{-i w t} + h0m e^{+i w t}, h0m = conj(h0(-k)) read at the mirrored cell;  kl = |k| + 1e-4
//   packed IFFT (unnormalised, exp(+i k x)):  F1 = ht (1 + kx/kl) -> Hraw + i Da,   F2 = -i (kz/kl) ht -> Db
//   checkerboard s = (-1)^(x+y);  up = -Hraw s,  east = chop Da s,  north = -chop Db s
//   texel (x, y) = tile position a = x L/N (east), b = y L/N (south);  hardware uv = (a_tile + OFF)/L + 0.5/N
// FFT: radix-2, one workgroup per line (N/2 invocations, whole line in workgroup memory, bit-reversed load,
// f64-built twiddles): 2 dispatches for all rows / columns of all cascades.
//
// Per frame: row FFT (spectrum evolution fused into its load), column FFT (+ finalize in place), derive level 0 +
// mips 1-3 in 8x8 workgroup tiles, then reduce kernels of up to 4 mip levels each: low 4 dispatches, others 5.
//
// OUTPUT: ONE rgba16float 2D-ARRAY storage texture per tier ("fields", N x N, 9 layers, Repeat, LinearMipmapLinear,
// full mip chain written by compute, box-averaged, UNNORMALISED) = one binding for the material:
//   layers 0-2 / 3-5  displacement of cascades 0-2, this frame / last frame (ping-pong: layer = c + curLayer,
//                     prev = c + prevLayer = the field at time - frameDt, for motion vectors)
//                     texel = (up, east, north, dEast/dNorth)  metres (derivative wrt the REST position)
//   layers 6-8        normal of cascade c (NORMAL_LAYER + c): (n.east, n.north, n.up, 1 - compression) (reference
//                     tex1 layout: per-cascade normal of the displaced surface; compression = clamp(1 - minEig(J), 0, 1)
//                     masked to downwind-facing slopes). Mip-averaged unnormalised vectors: |n| < 1 = unresolved slope.
// fft.nodes.displacement and fft.nodes.normal are two texture nodes of that same texture.
// (Round 1's 'derivatives' textures, base/detail groups, 4th cascade and f32 mip pyramids are gone.)
//
// Displacement sampling (createFftSampling): chained lookup (reference spectralField), explicit LOD from the
// footprint, and Catmull-Rom bicubic (16 exact loads) where LOD < 0.5, blended to trilinear by
// smoothstep(0, 0.5, LOD) like the reference's B-spline. Catmull-Rom (interpolating) instead of the reference's
// B-spline (smoothing): it is the core query's 'cubic' filter, so the CPU can match the drawn heights exactly; a
// B-spline shrinks the 47 m swell by ~5 % (transfer (2 + cos kh)/3), i.e. a few % of Hs of parity error.
//
// COST (RTX 2070, compute timestamps, p50 of 600 frames, best of 3 runs, GPU shared with other jobs, 2026-10-23;
// tools/fft-check.mjs --no-check):  low 0.05-0.07 ms, medium 0.14-0.18 ms, high / ultra 0.65-0.68 ms.
// Memory (textures + buffers): low 3.2 MB, medium 12.8 MB, high / ultra 51.4 MB (reference FFT ~20 MB).
// Traffic ~185 B per texel-layer per frame: low 9 MB, medium 36 MB, high / ultra 145 MB per frame; on a ~25 GB/s
// integrated GPU that is ~0.4 / 1.5 / 6 ms (+ ~0.1 ms of dispatch latency). On the 2070 low is not 1/4 of medium in
// time only because 4 dependent small dispatches are latency-bound there.
// Tier switch: setTier() returns in < 1 ms; build steps on the main thread total 1-3 ms (largest frame during a
// medium -> ultra switch 5 ms CPU, no long task); compile 0.3-1 s async; the old tier is drawn meanwhile.
//
// Spectrum slots on the GPU: 2 entries (from / to) of h0 only (h0m read mirrored) = 8 bytes per cell per slot.
// The reference tier (medium) uploads the core's slot arrays; the other tiers' spectra are generated in a Web
// Worker (fft-worker.ts) for the core slot's parameters, prefetched when the core announces a pending slot.
// Tier change: everything heavy is async (worker omega + spectra, resources created across frames,
// renderer.compileComputeAsync); the old tier stays drawn until the new one is ready.
import {
  HalfFloatType, LinearFilter, LinearMipmapLinearFilter, RGBAFormat, RepeatWrapping, StorageArrayTexture,
  StorageBufferAttribute, StorageTextureNode, Vector2,
} from 'three/webgpu';
import {
  Fn, If, clamp, cos, cross, dot, float, floor, fract, globalId, int, ivec2, length, localId, log2, max, mix,
  nodeObject, normalize, select, sin, smoothstep, sqrt, storage, texture, textureStore, uint, uniform,
  uvec2, vec2, vec3, vec4, workgroupArray, workgroupBarrier, workgroupId,
} from 'three/tsl';
import { CASCADES } from '../spectrum.ts';
import type { CascadeDef } from '../spectrum.ts';
import { FFT_TIERS, NUM_CASCADES, tierCascades, stripMasterH0, type OceanTier, type FftTierSpec } from './fft-tiers.ts';
import { handleFftJob } from './fft-worker.ts';
import type { Pass, PassContext } from './pass-types.ts';

export { FFT_TIERS, TIER_NAMES, tierCascades, tierQueryLayout } from './fft-tiers.ts';
export type { OceanTier, FftTierSpec } from './fft-tiers.ts';
/** @deprecated round-1 name. */
export type FftTierConfig = FftTierSpec;
/** Tile sizes and offsets (identical on every tier; the band split per tier is tierCascades(t)). */
export const FFT_CASCADES: readonly CascadeDef[] = CASCADES;
/** @deprecated round-1 constants (no base / detail groups and no 4th cascade any more). */
export const BASE_SIZE = 256;
export const FINE_DOMAIN = 9 / (1024 / 96);

const LOOP = 8192;
const L = NUM_CASCADES;

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
  perf?: { begin(module: string, phase: string): void; end(): void };
}

interface TierState {
  readonly tier: OceanTier;
  readonly n: number;
  /** The fields array texture (displacement cur / prev layers + normal layers). */
  readonly fields: any;
  readonly fieldAttr: any;
  readonly h0Attr: any;
  readonly kernels: any[];
  readonly bytes: number;
  /** GPU slot entries: core slot index / version held (-1 = none). */
  readonly entrySlot: Int32Array;
  readonly entryVer: Int32Array;
  readonly entryUnres: Float64Array;
  readonly omega: Float32Array;
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
const perLayer = (c: any, v: readonly number[]) => select(c.equal(uint(0)), float(v[0]), select(c.equal(uint(1)), float(v[1]), float(v[2])));
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
    curLayer: uniform(0, 'uint'),
  };
}
type Uniforms = ReturnType<typeof createUniforms>;

/** First layer of the normals in the fields array (displacement cur/prev occupy layers 0 .. 2L-1). */
export const NORMAL_LAYER = 2 * L;

/**
 * TSL kernels of one tier (built synchronously; cheap JS graph construction):
 *   1 row FFT with the spectrum evolution fused into its load, 2 column FFT + finalize (in place),
 *   3 derive level 0 + mips 1..3 in an 8x8 tile (workgroup memory), 4.. reduce kernels writing up to 4 mip levels
 *   each from the previous kernel's last level (f32 buffer). low: 4 dispatches, medium / high / ultra: 5.
 */
function buildKernels(u: Uniforms, n: number, cascades: readonly CascadeDef[], h0Attr: any, omegaAttr: any, twAttr: any, fieldAttr: any, redD: any, redN: any, redOff: number[], fields: any) {
  const log2n = Math.round(Math.log2(n));
  const nn = n * n, half = n >> 1;
  const NNu = uint(nn), mask = uint(n - 1);
  const slotStride = uint(L * nn);
  const dk = cascades.map((d) => (2 * Math.PI) / d.domain);
  const cell = cascades.map((d) => d.domain / n);
  const h0 = storage(h0Attr, 'vec2', h0Attr.count).toReadOnly();
  const om = storage(omegaAttr, 'float', omegaAttr.count).toReadOnly();
  const twn = storage(twAttr, 'vec2', twAttr.count).toReadOnly();
  const work = storage(fieldAttr, 'vec4', fieldAttr.count);
  const rD = storage(redD, 'vec4', redD.count), rN = storage(redN, 'vec4', redN.count);
  const kernels: any[] = [];

  /** Evolved packed spectra (F1, F2) of cell (x, y) of cascade c. */
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
    })().compute([n, L, 1], [half, 1, 1]);
  }
  kernels.push(lineFft(false), lineFft(true));

  // derived fields (central differences at the cell size, reference BLEND_FRAGMENT)
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

  /**
   * Tile reduction: T x T threads each hold one texel of level `src` (disp, normal); writes levels src+1 ..
   * src+log2(T) of the texture (src = -1: the tile also derives and writes level 0) and, if levels remain, the
   * last level into the f32 buffer section `outOff`.
   */
  function reduceKernel(src: number, T: number, inOff: number, outOff: number) {
    const k = Math.round(Math.log2(T));
    const first = src < 0 ? 0 : src + 1;
    const last = (src < 0 ? 0 : src) + k;          // last level written (src -1: levels 0..k with T = 8 -> 0..3)
    const lastLevel = src < 0 ? k : last;
    const tex: any[] = [];
    for (let lv = first; lv <= lastLevel; lv++) tex[lv] = mipStore(fields, lv);
    const ns = src < 0 ? n : n >> src;                // texels per side of the source level
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
        textureStore(tex[0].depth(int(c.add(u.curLayer))), xy, d);
        textureStore(tex[0].depth(int(c.add(uint(NORMAL_LAYER)))), xy, m4);
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
          textureStore(tex[lv].depth(int(c.add(u.curLayer))), xy, a);
          textureStore(tex[lv].depth(int(c.add(uint(NORMAL_LAYER)))), xy, b);
          if (s === k && lv < log2n) {
            const no = n >> lv;
            const o = uint(outOff).add(c.mul(uint(no * no))).add(xy.y.mul(uint(no))).add(xy.x);
            rD.element(o).assign(a); rN.element(o).assign(b);
          }
        });
        workgroupBarrier();
      }
    })().compute([ns / T, ns / T, L], [T, T, 1]);
  }
  // level 0..3 in 8x8 tiles, then 16x16 (or smaller) tiles from the last written level
  kernels.push(reduceKernel(-1, 8, 0, redOff[3]));
  let lv = 3;
  while (lv < log2n) {
    const ns = n >> lv;
    const T = Math.min(16, ns);
    const k = Math.round(Math.log2(T));
    kernels.push(reduceKernel(lv, T, redOff[lv], redOff[lv + k] ?? 0));
    lv += k;
  }
  return kernels;
}
// ------------------------------------------------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------------------------------------------------
export interface OceanFftTextures {
  /** 2D-array textures, N x N per tier; layer = cascade (+ curLayer / prevLayer for the displacement). */
  readonly displacement: any;
  readonly normal: any;
}

export interface OceanFft {
  /** Render-graph pass (stage compute). execute() allocates nothing in steady state. */
  readonly pass: Pass;
  /** Resolves when the first tier is compiled and has run once. */
  readonly ready: Promise<void>;
  readonly tier: OceanTier;
  readonly pendingTier: OceanTier | null;
  /** Texture NODES (their .value swaps on tier change; build samples from them, e.g. via createFftSampling). */
  readonly nodes: OceanFftTextures;
  readonly uniforms: {
    /** First displacement layer of this frame / last frame (0 or 3), float. */
    readonly dispCurLayer: any;
    readonly dispPrevLayer: any;
    /** Grid size N of the active tier (float); detailSize is the round-1 alias. */
    readonly size: any;
    readonly detailSize: any;
    /** int N - 1 (texel wrap mask for exact loads). */
    readonly sizeMask: any;
    /** @deprecated always 0 (no 4th cascade). */
    readonly fineEnabled: any;
    /** Slope variance of the full spectrum the tier does not draw (low: above its band top), m^2/m^2. */
    readonly unresolvedSlopeVariance: any;
    /** Wind direction (towards) of the drawn (crossfaded) spectrum in the tile frame (east, south). */
    readonly windAB: any;
  };
  setTier(t: OceanTier): void;
  readonly debug: {
    /** f32 (up, east, north, 0) fields of the active tier, layer-major (resolved after this frame's GPU work). */
    readFields(): Promise<Float32Array>;
    size(): number;
    /** The tier's h0 (h0-only layout, per cascade) that the GPU uses for core slot s, or null. */
    spectrum(slot: number): Float32Array[] | null;
    /** Omega index table (per cascade concatenated) of the active tier. */
    omega(): Float32Array | null;
    memoryBytes(): number;
    lastCompileMs(): number;
    /** Main-thread ms spent in the last tier build (sum of its synchronous steps). */
    lastBuildMainThreadMs(): number;
  };
  dispose(): void;
}

export function createOceanFft(opts: OceanFftOptions): OceanFft {
  const { renderer, ocean } = opts;
  const u = createUniforms();
  const st0 = ocean.getState();
  const g = st0.gravity as number;
  const seed = (st0.seed ?? 1) as number;

  // ---- worker (spectra + omega tables); synchronous fallback where Worker is unavailable
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
  function job(msg: any): Promise<any> {
    const id = ++jobId;
    msg.id = id;
    if (!worker) return Promise.resolve(handleFftJob(msg).result);
    return new Promise((resolve) => { jobs.set(id, resolve); worker!.postMessage(msg); });
  }

  // ---- spectra per (tier, core slot, version)
  interface SpecEntry { version: number; h0: Float32Array[] | null; unresolved: number; pending: boolean }
  const specs = new Map<string, SpecEntry>();
  const key = (t: OceanTier, s: number) => `${t}:${s}`;
  /** Spectrum of core slot s for tier t if ready (requests it otherwise). */
  function spectrumFor(t: OceanTier, s: number, state: any): SpecEntry | null {
    const sl = state.slots[s];
    if (!sl || !(sl.version > 0)) return null;
    const k = key(t, s);
    let e = specs.get(k);
    if (e && e.version === sl.version) return e.h0 ? e : null;
    if (FFT_TIERS[t].reference) {
      const n = FFT_TIERS[t].size;
      const h0 = [0, 1, 2].map((c) => stripMasterH0(sl.h0[c], new Float32Array(2 * n * n)));
      e = { version: sl.version, h0, unresolved: 0, pending: false };
      specs.set(k, e);
      return e;
    }
    if (e && e.pending && e.version === sl.version) return null;
    const entry: SpecEntry = { version: sl.version, h0: null, unresolved: 0, pending: true };
    specs.set(k, entry);
    job({ kind: 'spectrum', tier: t, params: { ...sl.params }, seed, g }).then((r) => {
      if (specs.get(k) !== entry || !r.h0) return;
      entry.h0 = r.h0; entry.unresolved = r.unresolved; entry.pending = false;
    });
    return null;
  }

  // ---- tiers
  let active: TierState | null = null;
  let pending: TierState | null = null;
  let pendingTier: OceanTier | null = null;
  let buildToken = 0;
  let compileMs = 0, buildMainMs = 0;
  let requested: OceanTier = opts.tier ?? 'high';
  let retired: TierState | null = null;
  let retireFrames = 0;

  function disposeTier(ts: TierState) {
    ts.fields.dispose();
    for (const k of ts.kernels) k.dispose?.();
  }

  function createTextures(t: OceanTier) {
    const n = FFT_TIERS[t].size;
    return makeArrayTexture(n, 3 * L, `ocean.fft.fields.${t}`);
  }

  async function buildTier(t: OceanTier, token: number, tex: any): Promise<TierState | null> {
    const spec = FFT_TIERS[t], n = spec.size, nn = n * n, log2n = Math.round(Math.log2(n));
    const cascades = tierCascades(t);
    let main = 0;
    const tick = () => performance.now();
    // omega table (worker)
    const om = await job({ kind: 'omega', tier: t, g });
    if (token !== buildToken) return null;
    // spectra of the slots in use (worker; poll until generated)
    for (let tries = 0; ; tries++) {
      const state = ocean.getState();
      const a = spectrumFor(t, state.toSlot, state);
      const b = state.fromSlot === state.toSlot ? a : spectrumFor(t, state.fromSlot, state);
      if (a && b) break;
      await new Promise((r) => setTimeout(r, 16));
      if (token !== buildToken) return null;
      if (tries > 2000) throw new Error('ocean.fft: spectrum generation timed out');
    }
    // resources, a few per task so no single main-thread step is long
    let t0 = tick();
    const h0Attr = new StorageBufferAttribute(new Float32Array(2 * L * nn * 2), 2);
    h0Attr.name = `ocean.fft.${t}.h0`;
    const omegaAttr = new StorageBufferAttribute(om.omega as Float32Array, 1);
    const tw = new Float32Array(n);
    for (let j = 0; j < n / 2; j++) { tw[2 * j] = Math.cos((2 * Math.PI * j) / n); tw[2 * j + 1] = Math.sin((2 * Math.PI * j) / n); }
    const twAttr = new StorageBufferAttribute(tw, 2);
    main += tick() - t0; await nextTask(); if (token !== buildToken) return null; t0 = tick();
    const fieldAttr = new StorageBufferAttribute(L * nn, 4);
    fieldAttr.name = `ocean.fft.${t}.field`;
    main += tick() - t0; await nextTask(); if (token !== buildToken) return null; t0 = tick();
    // f32 hand-over levels between the mip reduction kernels (3 and 7)
    const redOff: number[] = [];
    redOff[3] = 0;
    redOff[7] = (n >> 3) * (n >> 3) * L;
    const redCount = redOff[7] + Math.max(1, (n >> 7) * (n >> 7) * L);
    const redD = new StorageBufferAttribute(redCount, 4), redN = new StorageBufferAttribute(redCount, 4);
    const fields = tex ?? createTextures(t);
    main += tick() - t0; await nextTask(); if (token !== buildToken) return null; t0 = tick();
    const kernels = buildKernels(u, n, cascades, h0Attr, omegaAttr, twAttr, fieldAttr, redD, redN, redOff, fields);
    main += tick() - t0; await nextTask(); if (token !== buildToken) { for (const k of kernels) k.dispose?.(); return null; }
    const c0 = tick();
    await renderer.compileComputeAsync(kernels);
    compileMs = tick() - c0;
    buildMainMs = main;
    if (token !== buildToken) { for (const k of kernels) k.dispose?.(); return null; }
    let texels = 0;
    for (let lv = 0; lv <= log2n; lv++) texels += (n >> lv) * (n >> lv);
    const bytes = h0Attr.array.byteLength + omegaAttr.array.byteLength + tw.byteLength + L * nn * 16 + 2 * redCount * 16 + texels * 8 * 3 * L;
    return {
      tier: t, n, fields, fieldAttr, h0Attr, kernels, bytes,
      entrySlot: new Int32Array([-1, -1]), entryVer: new Int32Array([-1, -1]), entryUnres: new Float64Array(2),
      omega: omegaAttr.array as Float32Array,
    };
  }

  function startBuild(t: OceanTier, tex: any = null) {
    const token = ++buildToken;
    pendingTier = t;
    buildTier(t, token, tex).then((ts) => {
      if (!ts) return;
      if (token !== buildToken) { disposeTier(ts); return; }
      if (pending) disposeTier(pending);
      pending = ts;
    }, (e) => { console.error('[ocean.fft] tier build failed', e); if (token === buildToken) pendingTier = null; });
  }

  // Texture nodes: the first tier's textures exist from the start (three merges texture nodes sharing a texture
  // into one binding when a shader is built, so no shared placeholder).
  const firstTex = createTextures(requested);
  // displacement and normal are layers of ONE array texture (both nodes share its single binding).
  const nodes: OceanFftTextures = { displacement: texture(firstTex), normal: texture(firstTex) };
  const sizeU = uniform(FFT_TIERS[requested].size);
  const uniforms = {
    dispCurLayer: uniform(0),
    dispPrevLayer: uniform(0),
    size: sizeU,
    detailSize: sizeU,
    sizeMask: uniform(FFT_TIERS[requested].size - 1, 'int'),
    fineEnabled: uniform(0),
    unresolvedSlopeVariance: uniform(0),
    windAB: u.windAB,
  };

  let resolveReady: () => void = () => {};
  const ready = new Promise<void>((r) => { resolveReady = r; });
  let readyResolved = false;
  let parity = 0;
  let freshHistory = true;

  /** Make core slot s resident in a GPU entry (not `keep`); returns the entry or -1 if its spectrum is not ready. */
  function resident(ts: TierState, s: number, keep: number, state: any): number {
    const ver = state.slots[s]?.version ?? -1;
    for (let e = 0; e < 2; e++) if (ts.entrySlot[e] === s && ts.entryVer[e] === ver) return e;
    const sp = spectrumFor(ts.tier, s, state);
    if (!sp) { // not generated yet: keep using an older copy of this slot if resident
      for (let e = 0; e < 2; e++) if (ts.entrySlot[e] === s) return e;
      return -1;
    }
    const e = keep === 0 ? 1 : keep === 1 ? 0 : (ts.entrySlot[0] === -1 ? 0 : ts.entrySlot[1] === -1 ? 1 : (ts.entrySlot[0] === s ? 0 : 1));
    const nn = ts.n * ts.n, arr = ts.h0Attr.array as Float32Array;
    for (let c = 0; c < L; c++) arr.set(sp.h0![c], (e * L + c) * nn * 2);
    ts.h0Attr.addUpdateRange(e * L * nn * 2, L * nn * 2);
    ts.h0Attr.needsUpdate = true;
    ts.entrySlot[e] = s; ts.entryVer[e] = ver; ts.entryUnres[e] = sp.unresolved;
    return e;
  }

  function activate(ts: TierState) {
    if (retired) { disposeTier(retired); retired = null; }
    if (active) { retired = active; retireFrames = 3; }
    active = ts;
    pending = null;
    pendingTier = null;
    nodes.displacement.value = ts.fields;
    nodes.normal.value = ts.fields;
    sizeU.value = ts.n;
    uniforms.sizeMask.value = ts.n - 1;
    freshHistory = true;
  }

  startBuild(requested, firstTex);

  const pass: Pass = {
    desc: { id: 'ocean.fft', module: 'world/ocean', stage: 'compute', reads: [], writes: ['ocean.displacement', 'ocean.normal'] },
    execute(ctx: PassContext) {
      if (opts.perf) opts.perf.begin('world/ocean', 'ocean.fft');
      if (retired && --retireFrames <= 0) { disposeTier(retired); retired = null; }
      if (pending) activate(pending);
      if (!active) { if (opts.perf) opts.perf.end(); return; }
      const state = ocean.getState();
      const t = ctx.time;
      const crossing = state.fromSlot !== state.toSlot;
      const w = crossing ? ocean.blendAt(t) : 1;
      // spectrum slots -> GPU entries (uploads only when a slot / version changes)
      const needFrom = crossing && w < 1;
      let holdFrom = -1;
      if (needFrom) for (let e = 0; e < 2; e++) if (active.entrySlot[e] === state.fromSlot) holdFrom = e;
      const eTo = resident(active, state.toSlot, holdFrom, state);
      let eFrom = eTo;
      if (needFrom && eTo >= 0) { const e = resident(active, state.fromSlot, eTo, state); if (e >= 0) eFrom = e; }
      else if (state.pendingSlot >= 0) resident(active, state.pendingSlot, eTo, state); // prefetch the next one
      if (eTo < 0) { if (opts.perf) opts.perf.end(); return; }
      u.fromGpu.value = eFrom;
      u.toGpu.value = eTo;
      u.blend.value = eFrom === eTo ? 1 : w;
      // time / chop
      let tm = t % LOOP;
      if (tm < 0) tm += LOOP;
      const ti = Math.floor(tm);
      u.timeInt.value = ti;
      u.timeFrac.value = tm - ti;
      u.chop.value = ocean.choppinessAt(t);
      // foam mask wind = the drawn (crossfaded) spectrum's direction
      const wa = state.slots[state.fromSlot].params.windDirection as number, wb = state.slots[state.toSlot].params.windDirection as number;
      let dw = (wb - wa) % (2 * Math.PI);
      if (dw > Math.PI) dw -= 2 * Math.PI; else if (dw < -Math.PI) dw += 2 * Math.PI;
      const wd = wa + dw * w;
      (u.windAB.value as Vector2).set(Math.cos(wd), -Math.sin(wd));
      uniforms.unresolvedSlopeVariance.value = active.entryUnres[eFrom] + (active.entryUnres[eTo] - active.entryUnres[eFrom]) * (u.blend.value as number);
      // history ping-pong
      parity ^= 1;
      const cur = parity * L;
      u.curLayer.value = cur;
      uniforms.dispCurLayer.value = cur;
      uniforms.dispPrevLayer.value = freshHistory ? cur : L - cur;
      freshHistory = false;
      ctx.renderer.compute(active.kernels);
      if (!readyResolved) { readyResolved = true; resolveReady(); }
      if (opts.perf) opts.perf.end();
    },
  };

  return {
    pass,
    ready,
    get tier() { return active ? active.tier : requested; },
    get pendingTier() { return pendingTier; },
    nodes,
    uniforms,
    setTier(t: OceanTier) {
      if (!FFT_TIERS[t]) throw new Error(`ocean.fft: unknown tier ${t}`);
      requested = t;
      if (pendingTier === t) return;
      if (!pendingTier && active && active.tier === t) return;
      if (pending) { disposeTier(pending); pending = null; }
      if (active && active.tier === t) { buildToken++; pendingTier = null; return; }
      startBuild(t);
    },
    debug: {
      async readFields() {
        if (!active) throw new Error('no active tier');
        return new Float32Array(await renderer.getArrayBufferAsync(active.fieldAttr));
      },
      size() { return active ? active.n : 0; },
      spectrum(slot: number) {
        if (!active) return null;
        const e = specs.get(key(active.tier, slot));
        return e?.h0 ?? null;
      },
      omega() { return active ? active.omega : null; },
      memoryBytes() { return active ? active.bytes : 0; },
      lastCompileMs() { return compileMs; },
      lastBuildMainThreadMs() { return buildMainMs; },
    },
    dispose() {
      buildToken++;
      if (active) disposeTier(active);
      if (pending) disposeTier(pending);
      if (retired) disposeTier(retired);
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
  /** @deprecated (no 4th cascade any more). */
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
  /** Raw normal texel (n.east, n.north, n.up, 1 - compression) of cascade c at tile coordinate tc (east, south). */
  normal(c: number, tc: any, lod: any): any;
  /** Hardware uv of cascade c at tile coordinate tc. */
  uv(c: number, tc: any): any;
  readonly cascades: number;
}

export function createFftSampling(fft: OceanFft): FftSampling {
  const { nodes, uniforms } = fft;
  const C = FFT_CASCADES;
  const invN = uniforms.size.reciprocal();
  const uvOf = (c: number, tc: any) => tc.add(C[c].offset).div(C[c].domain).add(invN.mul(0.5));
  const lodOf = (c: number, footprint: any) => (footprint ? max(log2(max(footprint, 1e-6).mul(uniforms.size).div(C[c].domain)), 0) : float(0));

  /** Catmull-Rom of the displacement texture, 16 exact level-0 loads (core 'cubic' weights and texel mapping). */
  function catmullRom(c: number, tc: any, layer: any) {
    // (every reused intermediate is a var: TSL inlines shared sub-expressions otherwise, which blew the WGSL up)
    const s = tc.add(C[c].offset).div(C[c].domain).mul(uniforms.size).toVar();
    const f = floor(s).toVar();
    const t = s.sub(f).toVar();
    const t2 = t.mul(t).toVar(), t3 = t2.mul(t).toVar();
    const w0 = t3.mul(-0.5).add(t2).sub(t.mul(0.5)).toVar();
    const w1 = t3.mul(1.5).sub(t2.mul(2.5)).add(1).toVar();
    const w2 = t3.mul(-1.5).add(t2.mul(2)).add(t.mul(0.5)).toVar();
    const w3 = t3.mul(0.5).sub(t2.mul(0.5)).toVar();
    const W = [w0, w1, w2, w3];
    const b = ivec2(f).toVar();
    let sum: any = null;
    for (let r = 0; r < 4; r++) {
      let row: any = null;
      const yy = b.y.add(r - 1).bitAnd(uniforms.sizeMask).toVar();
      for (let q = 0; q < 4; q++) {
        const xx = b.x.add(q - 1).bitAnd(uniforms.sizeMask).toVar();
        const v = nodes.displacement.load(ivec2(xx, yy)).depth(layer).level(int(0)).mul(W[q].x);
        row = row === null ? v : row.add(v);
      }
      const rr = row.mul(W[r].y);
      sum = sum === null ? rr : sum.add(rr);
    }
    return sum;
  }

  function fetchDisp(c: number, tc: any, lod: any, previous: boolean, cubic: boolean) {
    const layer = int((previous ? uniforms.dispPrevLayer : uniforms.dispCurLayer).add(c));
    const tri = nodes.displacement.sample(uvOf(c, tc)).depth(layer).level(lod);
    if (!cubic) return tri;
    return Fn(() => {
      const v = vec4(tri).toVar();
      If(lod.lessThan(0.5), () => { v.assign(mix(catmullRom(c, tc, layer), tri, smoothstep(0, 0.5, lod))); });
      return v;
    })();
  }

  function chain(tileCoord: (p: number) => any, o: FftSampleOptions | undefined, withNormal: boolean, cubicDefault: boolean) {
    const previous = !!o?.previous;
    const cubic = o?.cubic ?? cubicDefault;
    let sum: any = null, nSum: any = null, foamSum: any = null, varSum: any = null;
    for (let c = 0; c < L; c++) {
      const lod = lodOf(c, o?.footprint).toVar();
      const tc = (sum === null ? tileCoord(C[c].domain) : tileCoord(C[c].domain).add(vec2(sum.y, sum.z.negate()))).toVar();
      const d = vec4(fetchDisp(c, tc, lod, previous, cubic)).toVar();
      sum = (sum === null ? d.xyz : sum.add(d.xyz)).toVar();
      if (withNormal) {
        const nm = nodes.normal.sample(uvOf(c, tc)).depth(int(c + NORMAL_LAYER)).level(lod);
        const tilt = nm.xyz.sub(vec3(0, 0, 1));
        const foam = float(1).sub(nm.w);
        const short = max(float(1).sub(length(nm.xyz)), 0);
        nSum = nSum === null ? tilt : nSum.add(tilt);
        foamSum = foamSum === null ? foam : foamSum.add(foam);
        varSum = varSum === null ? short : varSum.add(short);
      }
    }
    return { sum, nSum, foamSum, varSum };
  }

  return {
    cascades: L,
    uv: uvOf,
    normal(c, tc, lod) { return nodes.normal.sample(uvOf(c, tc)).depth(int(c + NORMAL_LAYER)).level(lod); },
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
