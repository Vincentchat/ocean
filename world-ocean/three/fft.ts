// GPU FFT ocean (three r186, WebGPU compute in TSL). Owns the per-frame spectrum evolution, the packed
// inverse FFTs, the derived fields (normals, Jacobian terms, foam compression) and their mip chains.
//
// Conventions: EXACTLY the core's (spectrum.ts header). Per cascade, per frame, at the explicit draw time t:
//   h0 = lerp(h0[fromSlot], h0[toSlot], w),  w = ocean.blendAt(t),  chop = ocean.choppinessAt(t)
//   phase/(2 pi) = fract(((m * T_int) mod 8192 + m * T_frac) / 8192)   (m = integer dispersion index, u32 math)
//   ht = h0 e^{-i w t} + h0m e^{+i w t};  kl = |k| + 1e-4
//   packed IFFT (unnormalised, exp(+i k x)):  F1 = ht (1 + kx/kl) -> Hraw + i Da,   F2 = -i (kz/kl) ht -> Db
//   checkerboard s = (-1)^(x+y);  up = -Hraw s,  east = chop Da s,  north = -chop Db s
//   texel (x, y) = tile position a = x L/N (east), b = y L/N (south);  sample uv = (a_tile + OFF)/L + 0.5/N
// FFT: radix-2 Cooley-Tukey, one workgroup per row (then per column), N/2 invocations, the whole line in
// workgroup shared memory (vec4 = two complex values), bit-reversed load, log2 N barriers, f64-built twiddle
// table. 2 dispatches per cascade group instead of 2 log2 N ping-pong passes.
//
// =====================================================================================================
// TIERS (decided with numbers; see also the report in docs/port):
//   cascade  tile      band k (rad/m)    lambda          group   low    medium  high   ultra
//   c0       1024 m    0     .. 0.52     inf .. 12 m     base    256    256     256    256
//   c1       96 m      0.52  .. 5.59     12  .. 1.1 m    base    256    256     256    256
//   c2       9 m       5.59  .. 59.6     1.1 .. 0.105 m  detail  128    256     256    512
//   c3       0.84375 m 59.6  .. 635      0.105 .. 0.01 m detail  -      -       256    512
// Why c0 / c1 are 256 on EVERY tier:
//   * Band completeness: the core fixes every band at the 256 master grid (upper = pi 256 / L / 1.5, fade to the
//     256 Nyquist). A 128 grid for c1 drops k in [4.2, 8.4] = 37 % (64: 75 %) of c1's band edge while c2 only fades
//     in from 3.7: a visible hole between cascades. Same for c0 (17 % of its slope variance at 128, breeze).
//   * CPU parity: the core's WATER_QUERY samples c0 / c1 bilinearly on 256 grids. Measured (core CascadeSim,
//     3000 random points, scratch parity512.ts): drawn 256/256/256 vs CPU: max 0.29 % Hs (breeze), 0.75 % (storm);
//     drawn 512/512/512 (exact band-limited 2x upsample): p99.9 7.0 % Hs, max 8.2 % (breeze) -> FAILS 2 %.
//     drawn 256/256/512: max 0.33 %; drawn 256/256/128: max 0.24 % (breeze) / 0.70 % (storm). So only the
//     detail group may change size.
//   * 64 grids would need ~6 cascades to cover the same band (each covers only N/6 of k-range ratio), more work.
// What the detail size does:
//   * 128 (low): c2 keeps k < 44.7 rad/m (lambda > 0.14 m): loses 11 % of c2's slope variance (0.5 % of its height
//     variance; 0.006 m^2/m^2 of slope variance, all presets). The lost part is exposed as the uniform
//     `unresolvedSlopeVariance` (add it to the roughness: it is unresolved micro-slope, like the mip shortening).
//   * 512 (ultra): the 256-master band zero-padded to 512: same waves, exact band-limited 2x interpolation of the
//     finest normals for close-ups (replaces bicubic on c2/c3).
//   * c3 (high, ultra): a 4th tile L3 = 9 / (1024/96) continuing the band (lower = c2 upper, complementary
//     smoothstep cross-over, so no gap and no double counting). Generated here from spectrum.ts with the
//     ocean's seed (gaussianField(seed, 3)), same JONSWAP + spreading, gravity dispersion (quantised like the
//     core). Height sigma ~1.7 mm (CPU parity unaffected: < 0.05 % Hs); slope variance ~= c2's (0.04-0.055):
//     it adds real micro roughness (the material sees it through the mip shortening / `fineGain`).
//   GPU cost (RTX 2070, timestamps, see fft-debug ?bench): reported in the module report / TUNING.md.
//
// =====================================================================================================
// OUTPUT TEXTURES (all rgba16float 2D-ARRAY storage textures, Repeat, LinearMipmapLinear, full mip chains
// written by compute, box-averaged, UNNORMALISED): 2 groups x 3 kinds = 6 textures. The material needs 4
// (displacement + normal of both groups); `derivatives` is optional (foam / debug).
//   displacement  layers [cur c_a, cur c_b, prev c_a, prev c_b]   (history ping-pong: layer = i + curLayer)
//                 texel = (up, east, north, dEast/dNorth)  metres; the derivative is wrt the REST position.
//   normal        layers [c_a, c_b]  texel = (n.east, n.north, n.up, 1 - compression)   (reference tex1 layout:
//                 per-cascade normal of the displaced surface, foam = 1 - clamp(1 - minEig(J), 0, 1) masked to
//                 downwind-facing slopes). Mips average the unnormalised vectors: |n| < 1 = unresolved slope.
//   derivatives   layers [c_a, c_b]  texel = (dUp/dEast, dUp/dNorth, dEast/dEast, dNorth/dNorth)
//                 (with dEast/dNorth in displacement.w the full Jacobian: J = (1+Exx)(1+Nyy) - Exy^2).
// Base group = cascades 0, 1 (256^2, every tier). Detail group = cascades 2, 3 (layer b is zero / unused when the
// tier has 3 cascades; `fineEnabled` = 0).
// Derivatives are central finite differences at the cascade's cell size like the reference blend pass
// (spectral-ocean.js BLEND_FRAGMENT), on the f32 FFT result, chop applied.
//
// Sampling (material / LOD / foam): use createFftSampling(fft).sampleDisplacement(tileCoord, footprint?) etc.
// tileCoord(P) = the LOD's periodic (east, south) tile coordinate for period P (lod.nodes.tileCoord).
import {
  HalfFloatType, LinearFilter, LinearMipmapLinearFilter, RGBAFormat, RepeatWrapping, StorageArrayTexture,
  StorageBufferAttribute, Vector2,
} from 'three/webgpu';
import {
  Fn, If, clamp, cos, cross, dot, float, floor, fract, globalId, int, length, localId, log2, max, mix,
  normalize, select, sin, smoothstep, sqrt, storage, storageTexture, texture, textureStore, uint, uniform,
  uvec2, vec2, vec3, vec4, workgroupArray, workgroupBarrier, workgroupId,
} from 'three/tsl';
import {
  referenceCascades, resampleCentred, omegaIndicesSized, gaussianField, spectrumScalars, generateRawRows,
  packInitialSpectrum,
} from '../spectrum.ts';
import type { CascadeDef, SpectrumScalars } from '../spectrum.ts';
import type { Pass, PassContext } from './pass-types.ts';

export type OceanTier = 'low' | 'medium' | 'high' | 'ultra';

/** Per-tier FFT layout. Base group (cascades 0, 1) is always 256^2. */
export interface FftTierConfig {
  /** Grid size of the detail group (cascades 2 and 3). */
  readonly detailSize: number;
  /** 3 or 4 cascades (4 = the extra 0.84 m tile). */
  readonly cascades: 3 | 4;
}

export const FFT_TIERS: Readonly<Record<OceanTier, FftTierConfig>> = Object.freeze({
  low: Object.freeze({ detailSize: 128, cascades: 3 as const }),
  medium: Object.freeze({ detailSize: 256, cascades: 3 as const }),
  high: Object.freeze({ detailSize: 256, cascades: 4 as const }),
  ultra: Object.freeze({ detailSize: 512, cascades: 4 as const }),
});

export const BASE_SIZE = 256;
const MASTER = 256;
/** Tile of the extra fine cascade: the reference's ratio 1024/96 applied once more below 9 m. */
export const FINE_DOMAIN = 9 / (1024 / 96);
/** The four cascade definitions (the first three are the core's CASCADES, bit-identical). */
export const FFT_CASCADES: readonly CascadeDef[] = Object.freeze(referenceCascades([1024, 96, 9, FINE_DOMAIN]));
const LOOP = 8192;
const NUM_SLOTS = 3;
/** Rows of the extra cascade's spectrum generated per frame (amortised regeneration). */
const FINE_ROWS_PER_FRAME = 32;

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

/** One cascade group's GPU resources + kernels (see header). */
interface Group {
  readonly n: number;
  readonly log2n: number;
  /** Global cascade indices of the layers (length 2). */
  readonly cascades: readonly number[];
  /** Layers actually computed (1 or 2). */
  readonly active: number;
  readonly displacement: any;
  readonly normal: any;
  readonly derivatives: any;
  /** f32 field buffer (up, east, north, 0) per texel, layer-major: readback for parity checks. */
  readonly fieldAttr: any;
  readonly h0Attr: any;
  readonly h0Size: number;
  readonly kernels: any[];
  readonly bytes: number;
  /** Version of each core slot uploaded (per layer). */
  readonly uploaded: Int32Array;
}

interface TierState {
  readonly tier: OceanTier;
  readonly config: FftTierConfig;
  readonly detail: Group;
  readonly list: any[];
  readonly bytes: number;
}

// ------------------------------------------------------------------------------------------------------
// Shared uniforms (set once per frame in execute; used by every tier's kernels).
// ------------------------------------------------------------------------------------------------------
function createUniforms() {
  return {
    timeInt: uniform(0, 'uint'),
    timeFrac: uniform(0),
    blend: uniform(0),
    chop: uniform(1),
    fromSlot: uniform(0, 'uint'),
    toSlot: uniform(0, 'uint'),
    /** Wind direction (towards) in the tile frame (a = east, b = south), unit. */
    windAB: uniform(new Vector2(0.540302306, 0.841470985)),
    /** First displacement layer written this frame (0 or 2). */
    curLayer: uniform(0, 'uint'),
  };
}
type Uniforms = ReturnType<typeof createUniforms>;

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

/** Bit reversal of a uint node over `bits` bits (unrolled). */
function bitReverse(i: any, bits: number) {
  let r: any = null;
  for (let b = 0; b < bits; b++) {
    const bit = i.shiftRight(uint(b)).bitAnd(uint(1)).shiftLeft(uint(bits - 1 - b));
    r = r === null ? bit : r.bitOr(bit);
  }
  return r;
}

const cmul = (a: any, b: any) => vec2(a.x.mul(b.x).sub(a.y.mul(b.y)), a.x.mul(b.y).add(a.y.mul(b.x)));

/** Per-layer constant (2 layers) as a node. */
const perLayer = (c: any, v0: number, v1: number) => select(c.equal(uint(0)), float(v0), float(v1));

// ------------------------------------------------------------------------------------------------------
// Group builder
// ------------------------------------------------------------------------------------------------------
function buildGroup(u: Uniforms, n: number, cascades: readonly number[], active: number, label: string, omegaFor: (c: number, n: number, out: Float32Array) => void): Group {
  const log2n = Math.round(Math.log2(n));
  if (1 << log2n !== n || n < 16 || n > 512) throw new RangeError(`fft: unsupported size ${n}`);
  const L = 2; // layers per group (fixed so the material code is tier-agnostic)
  const nn = n * n;
  const half = n >> 1;
  const h0Size = Math.min(n, MASTER);
  const defs = cascades.map((c) => FFT_CASCADES[c]);
  const dk = defs.map((d) => (2 * Math.PI) / d.domain);
  const cell = defs.map((d) => d.domain / n);

  // ---- buffers
  const h0Attr = new StorageBufferAttribute(new Float32Array(NUM_SLOTS * L * h0Size * h0Size * 4), 4);
  h0Attr.name = `ocean.fft.${label}.h0`;
  const omega = new Float32Array(L * nn);
  for (let l = 0; l < L; l++) omegaFor(cascades[l], n, omega.subarray(l * nn, (l + 1) * nn));
  const omegaAttr = new StorageBufferAttribute(omega, 1);
  const tw = new Float32Array(half * 2);
  for (let j = 0; j < half; j++) { tw[2 * j] = Math.cos((2 * Math.PI * j) / n); tw[2 * j + 1] = Math.sin((2 * Math.PI * j) / n); }
  const twAttr = new StorageBufferAttribute(tw, 2);
  const workAttr = new StorageBufferAttribute(L * nn, 4);
  workAttr.name = `ocean.fft.${label}.work`;
  const fieldAttr = new StorageBufferAttribute(L * nn, 4);
  fieldAttr.name = `ocean.fft.${label}.field`;
  // Mip pyramid (levels 1..log2n) of the 3 derived textures, f32 (source of the next level).
  const levelOff: number[] = [0, 0];
  let pyr = 0;
  for (let lv = 1; lv <= log2n; lv++) { levelOff[lv] = pyr; pyr += (n >> lv) * (n >> lv) * L; }
  const pyrAttr = [0, 1, 2].map(() => new StorageBufferAttribute(Math.max(pyr, 1), 4));

  const h0 = storage(h0Attr, 'vec4', h0Attr.count).toReadOnly();
  const om = storage(omegaAttr, 'float', omegaAttr.count).toReadOnly();
  const twn = storage(twAttr, 'vec2', twAttr.count).toReadOnly();
  const work = storage(workAttr, 'vec4', workAttr.count);
  const field = storage(fieldAttr, 'vec4', fieldAttr.count);
  const pyrN = pyrAttr.map((a) => storage(a, 'vec4', a.count));

  // ---- textures
  const displacement = makeArrayTexture(n, 2 * L, `ocean.displacement.${label}`);
  const normal = makeArrayTexture(n, L, `ocean.normal.${label}`);
  const derivatives = makeArrayTexture(n, L, `ocean.derivatives.${label}`);

  const kernels: any[] = [];
  const NNu = uint(nn);
  const mask = uint(n - 1);

  // ---- 1. evolve: (x, y, layer) -> packed spectra in natural order
  const evolve = Fn(() => {
    const x = globalId.x, y = globalId.y, c = globalId.z;
    const i = c.mul(NNu).add(y.mul(uint(n))).add(x);
    const h = vec4(0).toVar();
    const slot = uint(L * h0Size * h0Size);
    if (n <= MASTER) {
      const hi = c.mul(uint(h0Size * h0Size)).add(y.mul(uint(h0Size))).add(x);
      h.assign(mix(h0.element(u.fromSlot.mul(slot).add(hi)), h0.element(u.toSlot.mul(slot).add(hi)), u.blend));
    } else {
      const off = (n - MASTER) / 2;
      const mx = int(x).sub(int(off)), my = int(y).sub(int(off));
      If(mx.greaterThanEqual(int(0)).and(mx.lessThan(int(MASTER))).and(my.greaterThanEqual(int(0))).and(my.lessThan(int(MASTER))), () => {
        const hi = c.mul(uint(MASTER * MASTER)).add(uint(my).mul(uint(MASTER))).add(uint(mx));
        h.assign(mix(h0.element(u.fromSlot.mul(slot).add(hi)), h0.element(u.toSlot.mul(slot).add(hi)), u.blend));
      });
    }
    const m = om.element(i);
    const p = uint(m).mul(u.timeInt).bitAnd(uint(LOOP - 1));
    const cyc = fract(float(p).add(m.mul(u.timeFrac)).div(LOOP));
    const th = cyc.sub(floor(cyc.add(0.5))).mul(2 * Math.PI).toVar();
    const cs = cos(th).toVar(), sn = sin(th).toVar();
    // a = h0 e^{-i th}, b = h0m e^{+i th}
    const htr = h.x.mul(cs).add(h.y.mul(sn)).add(h.z.mul(cs).sub(h.w.mul(sn))).toVar();
    const hti = h.y.mul(cs).sub(h.x.mul(sn)).add(h.w.mul(cs).add(h.z.mul(sn))).toVar();
    const dkc = perLayer(c, dk[0], dk[1]);
    const kx = float(int(x).sub(int(half))).mul(dkc);
    const kz = float(int(y).sub(int(half))).mul(dkc);
    const kl = sqrt(kx.mul(kx).add(kz.mul(kz))).add(1e-4).toVar();
    const kxn = kx.div(kl), kzn = kz.div(kl).toVar();
    const one = kxn.add(1);
    work.element(i).assign(vec4(htr.mul(one), hti.mul(one), kzn.mul(hti), kzn.mul(htr).negate()));
  })().compute([Math.ceil(n / 8), Math.ceil(n / 8), active], [8, 8, 1]);
  kernels.push(evolve);

  // ---- 2./3. line FFTs in shared memory (rows, then columns + finalize)
  function lineFft(columns: boolean) {
    return Fn(() => {
      const j = localId.x;
      const line = workgroupId.x, c = workgroupId.y;
      const sh = workgroupArray('vec4', n);
      const base = columns ? c.mul(NNu).add(line) : c.mul(NNu).add(line.mul(uint(n)));
      const stride = uint(columns ? n : 1);
      const j2 = j.add(uint(half));
      sh.element(bitReverse(j, log2n)).assign(work.element(base.add(j.mul(stride))));
      sh.element(bitReverse(j2, log2n)).assign(work.element(base.add(j2.mul(stride))));
      workgroupBarrier();
      for (let s = 0; s < log2n; s++) {
        const h = 1 << s;
        const k = j.bitAnd(uint(h - 1));
        const a = j.shiftRight(uint(s)).shiftLeft(uint(s + 1)).add(k);
        const b = a.add(uint(h));
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
        // finalize: checkerboard, fields, chop  -> f32 field buffer
        const x = line;
        for (const yy of [j, j2]) {
          const v = sh.element(yy).toVar();
          const s = float(1).sub(float(x.add(yy).bitAnd(uint(1))).mul(2));
          field.element(base.add(yy.mul(stride))).assign(vec4(
            v.x.mul(s).negate(), v.y.mul(s).mul(u.chop), v.z.mul(s).mul(u.chop).negate(), 0,
          ));
        }
      }
    })().compute([n, active, 1], [half, 1, 1]);
  }
  kernels.push(lineFft(false), lineFft(true));

  // ---- 4. derived fields at level 0 (central differences, reference BLEND_FRAGMENT)
  /** Returns { disp, nrm, der } vec4 nodes for texel (x, y) of layer c (uint nodes, wrapped). */
  function derive(x: any, y: any, c: any) {
    const at = (xx: any, yy: any) => field.element(c.mul(NNu).add(yy.bitAnd(mask).mul(uint(n))).add(xx.bitAnd(mask)));
    const ctr = at(x, y).toVar();
    const px = at(x.add(uint(1)), y), mx = at(x.add(uint(n - 1)), y);
    const py = at(x, y.add(uint(1))), my = at(x, y.add(uint(n - 1)));
    const inv2c = float(1).div(perLayer(c, cell[0], cell[1]).mul(2));
    // reference displacement vector d = (east, up, south) in the tile frame (a = east, b = south)
    const dA = vec3(px.y.sub(mx.y), px.x.sub(mx.x), mx.z.sub(px.z)).mul(inv2c).toVar(); // d/da
    const dB = vec3(py.y.sub(my.y), py.x.sub(my.x), my.z.sub(py.z)).mul(inv2c).toVar(); // d/db
    const tx = vec3(1, 0, 0).add(dA).toVar();
    const tz = vec3(0, 0, 1).add(dB).toVar();
    const nr = normalize(cross(tz, tx)).toVar();
    const dd = tx.x.sub(tz.z);
    const eig = tx.x.add(tz.z).sub(sqrt(max(dd.mul(dd).add(tx.z.mul(tz.x).mul(4)), 0))).mul(0.5);
    const comp = clamp(float(1).sub(eig), 0, 1).mul(smoothstep(0, 0.2, dot(vec2(nr.x, nr.z), u.windAB)));
    return {
      // (up, east, north, dEast/dNorth = -d east/db)
      disp: vec4(ctr.x, ctr.y, ctr.z, dB.x.negate()),
      // (n.east, n.north, n.up, 1 - compression)
      nrm: vec4(nr.x, nr.z.negate(), nr.y, float(1).sub(comp)),
      // (dUp/dEast, dUp/dNorth, dEast/dEast, dNorth/dNorth)
      der: vec4(dA.y, dB.y.negate(), dA.x, dB.z),
    };
  }
  const tDisp0 = storageTexture(displacement).setMipLevel(0);
  const tNrm0 = storageTexture(normal).setMipLevel(0);
  const tDer0 = storageTexture(derivatives).setMipLevel(0);
  kernels.push(Fn(() => {
    const x = globalId.x, y = globalId.y, c = globalId.z;
    const d = derive(x, y, c);
    const xy = uvec2(x, y);
    textureStore(tDisp0.depth(int(c.add(u.curLayer))), xy, d.disp);
    textureStore(tNrm0.depth(int(c)), xy, d.nrm);
    textureStore(tDer0.depth(int(c)), xy, d.der);
  })().compute([Math.ceil(n / 8), Math.ceil(n / 8), active], [8, 8, 1]));

  // ---- 5. mip chain: level 1 from 2x2 derived texels, levels >= 2 from the f32 pyramid
  for (let lv = 1; lv <= log2n; lv++) {
    const m = n >> lv;
    const tD = storageTexture(displacement).setMipLevel(lv);
    const tN = storageTexture(normal).setMipLevel(lv);
    const tR = storageTexture(derivatives).setMipLevel(lv);
    const wg = Math.min(8, m);
    kernels.push(Fn(() => {
      const x = globalId.x, y = globalId.y, c = globalId.z;
      const sD = vec4(0).toVar(), sN = vec4(0).toVar(), sR = vec4(0).toVar();
      for (let jj = 0; jj < 2; jj++) for (let ii = 0; ii < 2; ii++) {
        const sx = x.mul(uint(2)).add(uint(ii)), sy = y.mul(uint(2)).add(uint(jj));
        if (lv === 1) {
          const d = derive(sx, sy, c);
          sD.addAssign(d.disp); sN.addAssign(d.nrm); sR.addAssign(d.der);
        } else {
          const ms = m * 2;
          const idx = uint(levelOff[lv - 1]).add(c.mul(uint(ms * ms))).add(sy.mul(uint(ms))).add(sx);
          sD.addAssign(pyrN[0].element(idx)); sN.addAssign(pyrN[1].element(idx)); sR.addAssign(pyrN[2].element(idx));
        }
      }
      sD.mulAssign(0.25); sN.mulAssign(0.25); sR.mulAssign(0.25);
      const xy = uvec2(x, y);
      textureStore(tD.depth(int(c.add(u.curLayer))), xy, sD);
      textureStore(tN.depth(int(c)), xy, sN);
      textureStore(tR.depth(int(c)), xy, sR);
      if (lv < log2n) {
        const o = uint(levelOff[lv]).add(c.mul(uint(m * m))).add(y.mul(uint(m))).add(x);
        pyrN[0].element(o).assign(sD); pyrN[1].element(o).assign(sN); pyrN[2].element(o).assign(sR);
      }
    })().compute([Math.ceil(m / wg), Math.ceil(m / wg), active], [wg, wg, 1]));
  }

  // ---- memory estimate (bytes)
  let texels = 0;
  for (let lv = 0; lv <= log2n; lv++) texels += (n >> lv) * (n >> lv);
  const bytes = h0Attr.array.byteLength + omega.byteLength + tw.byteLength + 2 * L * nn * 16 + 3 * pyr * 16 +
    texels * 8 * (2 * L + L + L);
  return {
    n, log2n, cascades, active, displacement, normal, derivatives, fieldAttr, h0Attr, h0Size, kernels, bytes,
    uploaded: new Int32Array(NUM_SLOTS * L).fill(-1),
  };
}

function disposeGroup(g: Group) {
  g.displacement.dispose(); g.normal.dispose(); g.derivatives.dispose();
  for (const k of g.kernels) k.dispose?.();
}

// ------------------------------------------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------------------------------------------
export interface OceanFftTextures {
  /** 2D-array textures (see header). base: cascades 0, 1; detail: cascades 2, 3. */
  readonly displacementBase: any;
  readonly displacementDetail: any;
  readonly normalBase: any;
  readonly normalDetail: any;
  readonly derivativesBase: any;
  readonly derivativesDetail: any;
}

export interface OceanFft {
  /** Render-graph pass (stage compute). execute() allocates nothing. */
  readonly pass: Pass;
  /** Resolves when the first tier is compiled and has been uploaded. */
  readonly ready: Promise<void>;
  /** Tier currently drawn (the old tier stays live while a new one compiles). */
  readonly tier: OceanTier;
  readonly pendingTier: OceanTier | null;
  /** Texture NODES (their .value swaps on tier change; build samples from them, e.g. via createFftSampling). */
  readonly nodes: OceanFftTextures;
  /** Uniform nodes the sampling helpers / material use. */
  readonly uniforms: {
    /** First layer of this frame's / last frame's displacement (0 or 2), float. */
    readonly dispCurLayer: any;
    readonly dispPrevLayer: any;
    /** Detail group grid size (float) and texel size of c2 / c3 (m). */
    readonly detailSize: any;
    /** 1 when the 4th cascade is drawn (high / ultra), else 0. */
    readonly fineEnabled: any;
    /** Slope variance not resolved by the detail grid (low tier: c2 truncated at 128), m^2/m^2. */
    readonly unresolvedSlopeVariance: any;
    /** Wind direction (towards) in the tile frame (east, south). */
    readonly windAB: any;
  };
  setTier(t: OceanTier): void;
  /** Readback helpers for the parity check (async, never awaited on the render path). */
  readonly debug: {
    /** f32 (up, east, north, 0) fields: base layers 0,1 then detail. Group n via groupSize(). */
    readFields(group: 'base' | 'detail'): Promise<Float32Array>;
    groupSize(group: 'base' | 'detail'): number;
    /** Cascade definitions used for each layer (global index). */
    readonly cascades: readonly CascadeDef[];
    /** Master-size h0 (256^2 x 4) of cascade 3 for slot s, as uploaded (for CPU comparisons). */
    fineH0(slot: number): Float32Array;
    /** Bytes of GPU memory (textures + buffers) of the active tier, approximately. */
    memoryBytes(): number;
    /** Last tier compile time, ms. */
    lastCompileMs(): number;
  };
  dispose(): void;
}

export function createOceanFft(opts: OceanFftOptions): OceanFft {
  const { renderer, ocean } = opts;
  const u = createUniforms();
  const st0 = ocean.getState();
  const g = st0.gravity as number;
  const seed = (st0.seed ?? 1) as number;

  const omegaFor = (c: number, n: number, out: Float32Array) => { omegaIndicesSized(FFT_CASCADES[c], g, n, out); };
  // Shared base group (identical on every tier).
  const base = buildGroup(u, BASE_SIZE, [0, 1], 2, 'base', omegaFor);

  // ---- cascade 3 spectrum (adapter-generated, per core slot, amortised)
  const fineXi = gaussianField(seed, 3, new Float64Array(2 * MASTER * MASTER));
  const fineRaw = new Float64Array(2 * MASTER * MASTER);
  const fineH0: Float32Array[] = [];
  for (let s = 0; s < NUM_SLOTS; s++) fineH0.push(new Float32Array(MASTER * MASTER * 4));
  const fineVersion = new Int32Array(NUM_SLOTS).fill(-1);   // slot version the fineH0 holds
  const fineScalars: SpectrumScalars = { g: 0, U: 0, dirGrid: 0, wp: 0 };
  let fineJobSlot = -1, fineJobRow = 0, fineJobVersion = 0;
  function fineGenerateSync(s: number, version: number, params: any) {
    spectrumScalars(params, g, fineScalars);
    generateRawRows(params, fineScalars, FFT_CASCADES[3], fineXi, fineRaw, 0, MASTER);
    packInitialSpectrum(fineRaw, params.standing, fineH0[s]);
    fineVersion[s] = version;
  }
  function fineStep(state: any) {
    if (fineJobSlot < 0) {
      for (let s = 0; s < NUM_SLOTS; s++) {
        const sl = state.slots[s];
        if (sl.version !== fineVersion[s] && sl.version > 0) {
          fineJobSlot = s; fineJobRow = 0; fineJobVersion = sl.version;
          spectrumScalars(sl.params, g, fineScalars);
          break;
        }
      }
      if (fineJobSlot < 0) return;
    }
    const sl = state.slots[fineJobSlot];
    if (sl.version !== fineJobVersion) { fineJobSlot = -1; return; } // regenerated meanwhile: restart next frame
    const r1 = Math.min(MASTER, fineJobRow + FINE_ROWS_PER_FRAME);
    generateRawRows(sl.params, fineScalars, FFT_CASCADES[3], fineXi, fineRaw, fineJobRow, r1);
    fineJobRow = r1;
    if (r1 >= MASTER) {
      packInitialSpectrum(fineRaw, sl.params.standing, fineH0[fineJobSlot]);
      fineVersion[fineJobSlot] = fineJobVersion;
      fineJobSlot = -1;
    }
  }
  // Initial: synchronous for the slots in use.
  for (let s = 0; s < NUM_SLOTS; s++) if (st0.slots[s].version > 0) fineGenerateSync(s, st0.slots[s].version, st0.slots[s].params);

  // ---- unresolved slope variance (low tier): per slot, master c2 energy outside the detail grid
  const lostSlope = new Float64Array(NUM_SLOTS);
  function computeLostSlope(s: number, n: number, masterH0: Float32Array) {
    if (n >= MASTER) { lostSlope[s] = 0; return; }
    const dk = (2 * Math.PI) / FFT_CASCADES[2].domain, h = n / 2;
    let sum = 0;
    for (let y = 0; y < MASTER; y++) {
      const ky = y - MASTER / 2;
      for (let x = 0; x < MASTER; x++) {
        const kx = x - MASTER / 2;
        if (kx >= -h + 1 && kx < h && ky >= -h + 1 && ky < h) continue;
        const i = 4 * (y * MASTER + x);
        const e = masterH0[i] ** 2 + masterH0[i + 1] ** 2 + masterH0[i + 2] ** 2 + masterH0[i + 3] ** 2;
        sum += e * (kx * kx + ky * ky) * dk * dk;
      }
    }
    lostSlope[s] = sum;
  }

  // ---- upload of h0 for a group when core slot versions change (setup / regeneration only)
  const tmp512 = new Float32Array(MASTER * MASTER * 4);
  function uploadGroup(grp: Group, state: any, force: boolean) {
    const L = 2, hs = grp.h0Size, slotStride = L * hs * hs * 4;
    let any = false;
    for (let s = 0; s < NUM_SLOTS; s++) {
      for (let l = 0; l < L; l++) {
        const c = grp.cascades[l];
        let ver: number, src: Float32Array;
        if (c < 3) { ver = state.slots[s].version; src = state.slots[s].h0[c]; }
        else { ver = fineVersion[s]; src = fineH0[s]; }
        if (ver <= 0 || (!force && grp.uploaded[s * L + l] === ver)) continue;
        if (c === 3 && grp.active < 2) continue;
        const off = s * slotStride + l * hs * hs * 4;
        const dst = (grp.h0Attr.array as Float32Array).subarray(off, off + hs * hs * 4);
        if (hs === MASTER) dst.set(src);
        else { resampleCentred(src, tmp512, hs, 4, true); dst.set(tmp512.subarray(0, hs * hs * 4)); }
        if (c === 2) computeLostSlope(s, grp.n, src);
        grp.h0Attr.addUpdateRange(off, hs * hs * 4);
        grp.uploaded[s * L + l] = ver;
        any = true;
      }
    }
    if (any) grp.h0Attr.needsUpdate = true;
  }

  // ---- tiers
  let active: TierState | null = null;
  let pending: TierState | null = null;
  let pendingTier: OceanTier | null = null;
  let compileMs = 0;
  let requested: OceanTier = opts.tier ?? 'high';
  let retired: TierState | null = null;
  let retireFrames = 0;

  async function buildTier(t: OceanTier): Promise<TierState> {
    const config = FFT_TIERS[t];
    const detail = buildGroup(u, config.detailSize, [2, 3], config.cascades === 4 ? 2 : 1, `detail${config.detailSize}`, omegaFor);
    const list = [...base.kernels, ...detail.kernels];
    const t0 = performance.now();
    await renderer.compileComputeAsync(list);
    compileMs = performance.now() - t0;
    return { tier: t, config, detail, list, bytes: base.bytes + detail.bytes };
  }

  function startBuild(t: OceanTier) {
    pendingTier = t;
    buildTier(t).then((ts) => {
      if (pendingTier !== t) { disposeGroup(ts.detail); return; } // superseded
      pending = ts;
    }, (e) => { console.error('[ocean.fft] tier build failed', e); if (pendingTier === t) pendingTier = null; });
  }

  // Texture nodes for the material (values swap on tier change).
  const nodes: OceanFftTextures = {
    displacementBase: texture(base.displacement),
    normalBase: texture(base.normal),
    derivativesBase: texture(base.derivatives),
    displacementDetail: texture(base.displacement), // placeholder value until the first tier is ready
    normalDetail: texture(base.normal),
    derivativesDetail: texture(base.derivatives),
  };
  const uniforms = {
    dispCurLayer: uniform(0),
    dispPrevLayer: uniform(0),
    detailSize: uniform(256),
    fineEnabled: uniform(0),
    unresolvedSlopeVariance: uniform(0),
    windAB: u.windAB,
  };

  let resolveReady: () => void = () => {};
  const ready = new Promise<void>((r) => { resolveReady = r; });
  let readyResolved = false;
  let parity = 0;
  let freshHistory = true;

  function activate(ts: TierState, state: any) {
    if (retired) { disposeGroup(retired.detail); retired = null; }
    if (active) { retired = active; retireFrames = 3; }
    active = ts;
    pending = null;
    pendingTier = null;
    ts.detail.uploaded.fill(-1);
    uploadGroup(ts.detail, state, true);
    nodes.displacementDetail.value = ts.detail.displacement;
    nodes.normalDetail.value = ts.detail.normal;
    nodes.derivativesDetail.value = ts.detail.derivatives;
    uniforms.detailSize.value = ts.config.detailSize;
    uniforms.fineEnabled.value = ts.config.cascades === 4 ? 1 : 0;
    freshHistory = true;
  }

  startBuild(requested);

  const pass: Pass = {
    desc: {
      id: 'ocean.fft', module: 'world/ocean', stage: 'compute', reads: [],
      writes: ['ocean.displacement', 'ocean.normal', 'ocean.derivatives'],
    },
    execute(ctx: PassContext) {
      if (opts.perf) opts.perf.begin('world/ocean', 'ocean.fft');
      const state = ocean.getState();
      if (retired && --retireFrames <= 0) { disposeGroup(retired.detail); retired = null; }
      if (pending) activate(pending, state);
      if (!active) { if (opts.perf) opts.perf.end(); return; }
      // spectrum slots -> GPU (only on version changes)
      fineStep(state);
      uploadGroup(base, state, false);
      uploadGroup(active.detail, state, false);
      // time / crossfade uniforms
      const t = ctx.time;
      let tm = t % LOOP;
      if (tm < 0) tm += LOOP;
      const ti = Math.floor(tm);
      u.timeInt.value = ti;
      u.timeFrac.value = tm - ti;
      u.blend.value = state.fromSlot === state.toSlot ? 1 : ocean.blendAt(t);
      u.chop.value = ocean.choppinessAt(t);
      u.fromSlot.value = state.fromSlot;
      u.toSlot.value = state.toSlot;
      const wd = state.windDirection as number;
      (u.windAB.value as any).set(Math.cos(wd), -Math.sin(wd));
      const w = u.blend.value as number;
      uniforms.unresolvedSlopeVariance.value = lostSlope[state.fromSlot] + (lostSlope[state.toSlot] - lostSlope[state.fromSlot]) * w;
      // history ping-pong
      parity ^= 1;
      const cur = parity * 2;
      u.curLayer.value = cur;
      uniforms.dispCurLayer.value = cur;
      uniforms.dispPrevLayer.value = freshHistory ? cur : 2 - cur;
      freshHistory = false;
      ctx.renderer.compute(active.list);
      if (!readyResolved) { readyResolved = true; resolveReady(); }
      if (opts.perf) opts.perf.end();
    },
  };

  const fft: OceanFft = {
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
      if (active && active.tier === t && !pendingTier) return;
      if (pending) { disposeGroup(pending.detail); pending = null; }
      if (active && active.tier === t) { pendingTier = null; return; }
      startBuild(t);
    },
    debug: {
      async readFields(group) {
        const grp = group === 'base' ? base : active?.detail;
        if (!grp) throw new Error('no active tier');
        const buf = await renderer.getArrayBufferAsync(grp.fieldAttr);
        return new Float32Array(buf);
      },
      groupSize(group) { return group === 'base' ? base.n : (active?.detail.n ?? 0); },
      cascades: FFT_CASCADES,
      fineH0(slot: number) { return fineH0[slot]; },
      memoryBytes() { return active ? active.bytes : base.bytes; },
      lastCompileMs() { return compileMs; },
    },
    dispose() {
      if (active) disposeGroup(active.detail);
      if (pending) disposeGroup(pending.detail);
      if (retired) disposeGroup(retired.detail);
      disposeGroup(base);
      active = pending = retired = null;
      pendingTier = null;
    },
  };
  return fft;
}

// ------------------------------------------------------------------------------------------------------
// Sampling helpers for the material / LOD / foam (TSL). Chained lookup of the reference (spectralField /
// spectralSurface), explicit LOD from a footprint, bilinear at LOD 0 (what the CPU query assumes).
// ------------------------------------------------------------------------------------------------------
export interface FftSampleOptions {
  /** Footprint in metres (vertex: grid cell size; fragment: max |d rest / d screen|). Default 0 = LOD 0. */
  footprint?: any;
  /** Sample LAST frame's displacement (motion vectors). */
  previous?: boolean;
  /** Use only the first 3 cascades (diagnostics: compare with the CPU query, which has no cascade 3). */
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
}

export function createFftSampling(fft: OceanFft): FftSampling {
  const { nodes, uniforms } = fft;
  const C = FFT_CASCADES;
  const baseTexel = (c: number) => C[c].domain / BASE_SIZE;
  const detailTexel = (c: number) => uniforms.detailSize.reciprocal().mul(C[c].domain);

  function uvOf(c: number, tc: any) {
    const halfTexel = c < 2 ? float(0.5 / BASE_SIZE) : uniforms.detailSize.reciprocal().mul(0.5);
    return tc.add(C[c].offset).div(C[c].domain).add(halfTexel);
  }
  function lodOf(c: number, footprint: any) {
    if (!footprint) return float(0);
    const texel = c < 2 ? float(baseTexel(c)) : detailTexel(c);
    return max(log2(max(footprint, 1e-6).div(texel)), 0);
  }
  function dispTex(c: number) { return c < 2 ? nodes.displacementBase : nodes.displacementDetail; }
  function nrmTex(c: number) { return c < 2 ? nodes.normalBase : nodes.normalDetail; }
  const layerOf = (c: number) => c & 1;

  function fetchDisp(c: number, tc: any, lod: any, previous: boolean) {
    const layer = (previous ? uniforms.dispPrevLayer : uniforms.dispCurLayer).add(layerOf(c));
    return dispTex(c).sample(uvOf(c, tc)).depth(int(layer)).level(lod);
  }

  function chain(tileCoord: (p: number) => any, o: FftSampleOptions | undefined, withNormal: boolean) {
    const previous = !!o?.previous;
    let sum: any = null;          // (up, east, north) accumulated
    let nSum: any = null, foamSum: any = null, varSum: any = null;
    const count = o?.threeCascades ? 3 : 4;
    for (let c = 0; c < count; c++) {
      const lod = lodOf(c, o?.footprint);
      // chained tile coordinate: rest + horizontal displacement of the larger cascades (east, south)
      const tc = sum === null ? tileCoord(C[c].domain) : tileCoord(C[c].domain).add(vec2(sum.y, sum.z.negate()));
      const d = fetchDisp(c, tc, lod, previous);
      const gate = c === 3 ? uniforms.fineEnabled : null;
      const dv = gate ? d.xyz.mul(gate) : d.xyz;
      sum = sum === null ? dv : sum.add(dv);
      if (withNormal) {
        const nm = nrmTex(c).sample(uvOf(c, tc)).depth(int(layerOf(c))).level(lod);
        // tilt = n - up (so the sum of C normals minus (C-1) up), foam term 1 - a, shortening 1 - |n|
        let tilt: any = nm.xyz.sub(vec3(0, 0, 1));
        let foam: any = float(1).sub(nm.w);
        let short: any = max(float(1).sub(length(nm.xyz)), 0);
        if (gate) { tilt = tilt.mul(gate); foam = foam.mul(gate); short = short.mul(gate); }
        nSum = nSum === null ? tilt : nSum.add(tilt);
        foamSum = foamSum === null ? foam : foamSum.add(foam);
        varSum = varSum === null ? short : varSum.add(short);
      }
    }
    return { sum, nSum, foamSum, varSum };
  }

  return {
    sampleDisplacement(tileCoord, o) {
      const { sum } = chain(tileCoord, o, false);
      return vec3(sum.y, sum.x, sum.z.negate());
    },
    sampleSurface(tileCoord, o) {
      const { sum, nSum, foamSum, varSum } = chain(tileCoord, o, true);
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

