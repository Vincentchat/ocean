// GPU FFT quality tiers + the pure (three-free, Node-testable) maths shared by the adapter, its worker,
// the debug page and the tests: band layout per tier, tier spectrum generation, an independent exact CPU
// evaluation of the drawn fields, and the CPU-side replica of the GPU displacement filter (parity analysis).
//
// TIERS (one grid size N for every cascade; tiles 1024 / 96 / 9 m on every tier, so the tile coordinates and
// the material's varyings never change). Band split as a function of N (spectrum.ts bandCascades):
//     upper_c = pi N / L_c / 1.5,  lower_c = upper_{c-1}   -> contiguous bands at any N, no holes.
//   tier    N    band top (k, rad/m / lambda)            dispersion          spectrum source
//   low     128  29.8 (fades out by 44.7) / 0.21 m        gravity             adapter (worker), tier layout
//   medium  256  59.6 (89.4) / 0.105 m  = THE REFERENCE   gravity             core slots (bit-identical)
//   high    512  59.6 (89.4) / 0.105 m  (top capped)      gravity             adapter (worker), tier layout
//   ultra   512  119  (179)  / 0.053 m                    gravity-capillary   adapter (worker), tier layout
//   * low draws 1/4 of medium's texels; the slope variance above its band top is reported as
//     unresolvedSlopeVariance (roughness).
//   * high: same spectrum extent as the reference, re-cut so every cascade has 2x the resolution (c0 down to
//     lambda 6 m on 2 m cells, c1 down to 0.56 m): sharper crests / normals near the camera, no new top band.
//   * ultra: high + the 5-10 cm ripple octave, with the gravity-capillary dispersion w^2 = g k + S k^3
//     (S = 7.4e-5 m^3/s^2; at k = 179 the capillary term is 24 % of w^2). It is the only tier that draws MORE
//     than the reference spectrum.
//   Cascade count stays 3: at N = 128 three tiles are needed to reach lambda 0.2 m from the 1024 m swell
//   (fewer would drop the 1-12 m chop or the swell); at 512 a 4th tile only adds sub-5 cm detail.
// The old 4th "c3" cascade (0.84 m tile, gravity dispersion) is gone (review fft-r1: wrong dispersion, speckle).
//
// CPU parity: a re-cut band moves energy between tiles (a different random realisation of those waves), so the
// core query (reference layout) does not match low / high / ultra (measured: see tests/fft-tiers.test.ts and
// the module report). The core must then evaluate the SAME layout: tierQueryLayout(t) describes it.
import {
  CASCADES, REFERENCE_SIZE, OMEGA_UNIT, LOOP_PERIOD, bandCascades, gaussianField, gaussianFieldSized,
  generateRawRowsSized, packInitialSpectrumSized, omegaIndicesGrid, spectrumScalars,
} from '../spectrum.ts';
import type { CascadeDef, SpectrumParams, SpectrumScalars } from '../spectrum.ts';

export type OceanTier = 'low' | 'medium' | 'high' | 'ultra';

export interface FftTierSpec {
  /** Grid size of every cascade. */
  readonly size: number;
  /** Upper k cap of the last cascade (rad/m), Infinity = the grid's own band top. */
  readonly topUpper: number;
  /** Gravity-capillary dispersion (else the core's quantised deep-water gravity dispersion). */
  readonly capillary: boolean;
  /** True when the layout IS the reference / core layout (the adapter then uploads the core's slots as is). */
  readonly reference: boolean;
}

export const FFT_TIERS: Readonly<Record<OceanTier, FftTierSpec>> = Object.freeze({
  low: Object.freeze({ size: 128, topUpper: Infinity, capillary: false, reference: false }),
  medium: Object.freeze({ size: 256, topUpper: Infinity, capillary: false, reference: true }),
  high: Object.freeze({ size: 512, topUpper: CASCADES[2].upper, capillary: false, reference: false }),
  ultra: Object.freeze({ size: 512, topUpper: Infinity, capillary: true, reference: false }),
});

export const TIER_NAMES: readonly OceanTier[] = ['low', 'medium', 'high', 'ultra'];
export const NUM_CASCADES = 3;

/** The tier's cascades (domains 1024 / 96 / 9 m, bands cut for its N). medium === core CASCADES. */
export function tierCascades(t: OceanTier): CascadeDef[] {
  const s = FFT_TIERS[t];
  return s.reference ? CASCADES.map((c) => ({ ...c })) : bandCascades(s.size, undefined, s.topUpper);
}

/**
 * What the CPU query (core) needs to evaluate exactly what tier t draws. The core's current query uses the
 * reference layout (= medium); for other tiers it must generate its slots with these cascades and grid sizes.
 *   cascades[c]  band (domain, lower, upper, offset) to use INSTEAD of CASCADES[c] in generateRawRows
 *   size         spectral grid size N (generateRawRowsSized / gaussianFieldSized / packInitialSpectrumSized)
 *   capillary    dispersion (omegaIndicesGrid(cascade, g, N, capillary))
 *   filter       what the GPU uses for displacement near the camera ('cubic' = Catmull-Rom, the core's 'cubic')
 *   querySize    the grid the query must sample (= size: the drawn grid; a smaller query grid re-introduces
 *                the interpolation error measured in the report)
 */
export function tierQueryLayout(t: OceanTier) {
  const s = FFT_TIERS[t];
  return { tier: t, size: s.size, capillary: s.capillary, cascades: tierCascades(t), filter: 'cubic' as const, querySize: s.size };
}

// ------------------------------------------------------------------------------------------------------------
// Spectrum generation for a tier (setup / worker; allocates)
// ------------------------------------------------------------------------------------------------------------
export interface TierSpectrum {
  readonly size: number;
  /** Per cascade: h0 only, 2 floats per cell [Re, Im] (h0m = conj(h0(-k)) is read at the mirrored cell). */
  readonly h0: Float32Array[];
  /** Time-averaged slope variance of the drawn spectrum, sum_k k^2 (|h0|^2 + |h0m|^2). */
  readonly slopeVariance: number;
}

/** Master (256) gaussian draws per cascade, cached by seed (setup-time). */
const xiCache = new Map<string, Float64Array>();
function masterXi(seed: number, c: number): Float64Array {
  const key = `${seed}:${c}`;
  let x = xiCache.get(key);
  if (!x) { x = gaussianField(seed, c, new Float64Array(2 * REFERENCE_SIZE * REFERENCE_SIZE)); xiCache.set(key, x); }
  return x;
}

/** Slope variance of a packed [h0, h0m] (4 floats / cell) size-n spectrum of a tile of side L. */
function slopeVarianceOf(packed: Float32Array, n: number, L: number): number {
  const dk = (2 * Math.PI) / L;
  let s = 0;
  for (let y = 0; y < n; y++) {
    const ky = (y - n / 2) * dk;
    for (let x = 0; x < n; x++) {
      const kx = (x - n / 2) * dk, i = 4 * (y * n + x);
      const e = packed[i] * packed[i] + packed[i + 1] * packed[i + 1] + packed[i + 2] * packed[i + 2] + packed[i + 3] * packed[i + 3];
      s += e * (kx * kx + ky * ky);
    }
  }
  return s;
}

/** Generate the tier's h0 (all cascades) for one spectrum slot. ~10-60 ms: run in the worker. */
export function generateTierSpectrum(spec: FftTierSpec, cascades: readonly CascadeDef[], params: Readonly<SpectrumParams>, seed: number, g: number): TierSpectrum {
  const n = spec.size;
  const sc: SpectrumScalars = spectrumScalars(params, g, { g: 0, U: 0, dirGrid: 0, wp: 0 });
  const xi = new Float64Array(2 * n * n), raw = new Float64Array(2 * n * n), packed = new Float32Array(4 * n * n);
  const h0: Float32Array[] = [];
  let slope = 0;
  for (let c = 0; c < cascades.length; c++) {
    gaussianFieldSized(seed, c, n, xi, masterXi(seed, c));
    generateRawRowsSized(params, sc, cascades[c], n, xi, raw, 0, n);
    packInitialSpectrumSized(raw, params.standing, n, packed);
    slope += slopeVarianceOf(packed, n, cascades[c].domain);
    const out = new Float32Array(2 * n * n);
    for (let i = 0; i < n * n; i++) { out[2 * i] = packed[4 * i]; out[2 * i + 1] = packed[4 * i + 1]; }
    h0.push(out);
  }
  return { size: n, h0, slopeVariance: slope };
}

/** h0-only layout of a core master slot (4 floats / cell -> 2). */
export function stripMasterH0(master: Float32Array, out: Float32Array): Float32Array {
  const nn = master.length / 4;
  for (let i = 0; i < nn; i++) { out[2 * i] = master[4 * i]; out[2 * i + 1] = master[4 * i + 1]; }
  return out;
}

/** Slope variance of the reference spectrum (all master cascades) from core master slots. */
export function referenceSlopeVariance(masterH0: readonly Float32Array[]): number {
  let s = 0;
  for (let c = 0; c < CASCADES.length; c++) s += slopeVarianceOf(masterH0[c], REFERENCE_SIZE, CASCADES[c].domain);
  return s;
}

/** Omega index table (n^2 per cascade, concatenated) for a tier. */
export function tierOmega(spec: FftTierSpec, cascades: readonly CascadeDef[], g: number): Float32Array {
  const n = spec.size, nn = n * n;
  const out = new Float32Array(cascades.length * nn);
  for (let c = 0; c < cascades.length; c++) omegaIndicesGrid(cascades[c], g, n, spec.capillary, out.subarray(c * nn, (c + 1) * nn));
  return out;
}

// ------------------------------------------------------------------------------------------------------------
// Exact CPU evaluation of the fields the GPU draws (independent of cpu-fft.ts; for checks / tests)
// ------------------------------------------------------------------------------------------------------------
function fft1(re: Float64Array, im: Float64Array, off: number, stride: number, n: number) {
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const a = off + i * stride, b = off + j * stride;
      let t = re[a]; re[a] = re[b]; re[b] = t; t = im[a]; im[a] = im[b]; im[b] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (2 * Math.PI) / len; // inverse transform: e^{+i}
    for (let i = 0; i < n; i += len) for (let j = 0; j < len / 2; j++) {
      const c = Math.cos(ang * j), s = Math.sin(ang * j);
      const a = off + (i + j) * stride, b = a + (len / 2) * stride;
      const tr = re[b] * c - im[b] * s, ti = re[b] * s + im[b] * c;
      re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
    }
  }
}

/**
 * Fields (up, east, north; chop applied) of one cascade at time t, exactly as specified in spectrum.ts, with
 * h0 = lerp(from, to, w) (h0-only layout, h0m read at the mirrored cell). Output n^2 x 3 (f64). Allocates.
 */
export function evaluateFields(from: Float32Array, to: Float32Array, w: number, omega: Float32Array, n: number, domain: number, t: number, chop: number): Float64Array {
  const nn = n * n;
  const r1 = new Float64Array(nn), i1 = new Float64Array(nn), r2 = new Float64Array(nn), i2 = new Float64Array(nn);
  const dk = (2 * Math.PI) / domain;
  let tm = t % LOOP_PERIOD; if (tm < 0) tm += LOOP_PERIOD;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = y * n + x, j = ((n - y) & (n - 1)) * n + ((n - x) & (n - 1));
    const hr = from[2 * i] + (to[2 * i] - from[2 * i]) * w, hi = from[2 * i + 1] + (to[2 * i + 1] - from[2 * i + 1]) * w;
    const jr = from[2 * j] + (to[2 * j] - from[2 * j]) * w, ji = from[2 * j + 1] + (to[2 * j + 1] - from[2 * j + 1]) * w;
    const mr = jr, mi = -ji; // h0m = conj(h0(-k))
    const m = omega[i];
    const ph = OMEGA_UNIT * ((m * tm) % LOOP_PERIOD);
    const c = Math.cos(ph), s = Math.sin(ph);
    const htr = hr * c + hi * s + mr * c - mi * s, hti = hi * c - hr * s + mi * c + mr * s;
    const kx = Math.fround((x - n / 2) * dk), kz = Math.fround((y - n / 2) * dk);
    const kl = Math.hypot(kx, kz) + 1e-4, kxn = kx / kl, kzn = kz / kl;
    r1[i] = htr * (1 + kxn); i1[i] = hti * (1 + kxn);
    r2[i] = kzn * hti; i2[i] = -kzn * htr;
  }
  for (let y = 0; y < n; y++) { fft1(r1, i1, y * n, 1, n); fft1(r2, i2, y * n, 1, n); }
  for (let x = 0; x < n; x++) { fft1(r1, i1, x, n, n); fft1(r2, i2, x, n, n); }
  const out = new Float64Array(3 * nn);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = y * n + x, sg = ((x + y) & 1) === 0 ? 1 : -1;
    out[3 * i] = -r1[i] * sg; out[3 * i + 1] = chop * i1[i] * sg; out[3 * i + 2] = -chop * r2[i] * sg;
  }
  return out;
}

// ------------------------------------------------------------------------------------------------------------
// CPU replica of the GPU displacement lookup at LOD 0 (parity analysis). filter: 'bilinear' | 'cubic'
// (Catmull-Rom, same weights / texel mapping as the GPU kernel and the core's sampleCubic).
// ------------------------------------------------------------------------------------------------------------
export function sampleField(f: ArrayLike<number>, n: number, L: number, a: number, b: number, cubic: boolean, out: Float64Array): void {
  const sa = (a / L - Math.floor(a / L)) * n, sb = (b / L - Math.floor(b / L)) * n;
  const xf = Math.floor(sa), yf = Math.floor(sb), fx = sa - xf, fy = sb - yf, mask = n - 1;
  out[0] = out[1] = out[2] = 0;
  if (!cubic) {
    for (let r = 0; r < 2; r++) for (let q = 0; q < 2; q++) {
      const wgt = (q ? fx : 1 - fx) * (r ? fy : 1 - fy), i = 3 * (((yf + r) & mask) * n + ((xf + q) & mask));
      out[0] += wgt * f[i]; out[1] += wgt * f[i + 1]; out[2] += wgt * f[i + 2];
    }
    return;
  }
  const cr = (t: number, k: number) => {
    const t2 = t * t, t3 = t2 * t;
    return k === 0 ? -0.5 * t3 + t2 - 0.5 * t : k === 1 ? 1.5 * t3 - 2.5 * t2 + 1 : k === 2 ? -1.5 * t3 + 2 * t2 + 0.5 * t : 0.5 * t3 - 0.5 * t2;
  };
  for (let r = 0; r < 4; r++) for (let q = 0; q < 4; q++) {
    const wgt = cr(fx, q) * cr(fy, r), i = 3 * (((yf - 1 + r) & mask) * n + ((xf - 1 + q) & mask));
    out[0] += wgt * f[i]; out[1] += wgt * f[i + 1]; out[2] += wgt * f[i + 2];
  }
}

/**
 * Chained lookup of the drawn surface at local rest point (east e, north nn) (reference spectralField):
 * returns [up, east displacement, north displacement] in out.
 */
export function sampleChained(fields: readonly ArrayLike<number>[], n: number, cascades: readonly CascadeDef[], e: number, nn: number, cubic: boolean, out: Float64Array, tmp: Float64Array): void {
  let U = 0, E = 0, N = 0;
  for (let c = 0; c < cascades.length; c++) {
    const L = cascades[c].domain, off = cascades[c].offset;
    sampleField(fields[c], n, L, e + E + off, -(nn + N) + off, cubic, tmp);
    U += tmp[0]; E += tmp[1]; N += tmp[2];
  }
  out[0] = U; out[1] = E; out[2] = N;
}
