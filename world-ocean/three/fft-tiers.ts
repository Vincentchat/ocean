// GPU FFT quality tiers + the pure (three-free, Node-testable) maths shared by the adapter, its worker, the debug
// page and the tests.
//
// TIER INVARIANCE (round 3): every wave component of the reference master layout (the core's slots: 256^2 per
// cascade, reference band edges, same k / amplitude / phase / dispersion) is drawn bit-identically on every tier.
// A tier may only DROP short waves (a smaller grid takes the CENTRED SUB-BLOCK of the master grid of c1 / c2, like
// spectrum.resampleCentred) or ADD a band above the master top (ultra's optional capillary cascade, its own tile,
// draws and dispersion; the master components are untouched). So the sea is the same on every tier, a tier switch
// only changes fine detail, and the CPU query (master layout) is tier-independent.
//
//   tier    c0 (1024 m)  c1 (96 m)  c2 (9 m)  extra                       drops (-> unresolvedSlopeVariance)
//   low     256          256        64        -                           c2 k > 22.3 (lambda < 0.28 m)
//   medium  256          256        256       -                           nothing (= the reference)
//   high    256          256        256       -                           nothing (same FFT as medium; see note)
//   ultra   256          256        256       capillary band (opt-in)     nothing
// c0 AND c1 stay 256 on every tier (measured against the master CPU query, cubic c0/c1, p99.9 % Hs, see
// tests/fft-tiers.test.ts): c1 at 128 drops lambda 1.1-1.5 m waves = 2.7 % (breeze, Hs 1.46 m) by truncation alone,
// 4.3 % with the 0.75 m-cell interpolation; windy 1.7 %, storm 0.8 %. c0 at 128: 35 / 17 / 8 %. Only c2 (heights
// of cm) can shrink: 256/256/64 gives p99.9 1.05 / 0.76 / 0.32 % (breeze / windy / storm; medium 0.91 / 0.74 / 0.24). So low costs ~70 % of medium's texels, not 25 %:
// that is the price of drawing the same sea on every tier. high: a finer (zero-padded) grid for c2 only
// re-samples the same band (no new waves) and changes the finite-difference normals the reference look relies on,
// so high spends nothing extra on the FFT (its quality comes from LOD / foam / material tiers).
// ultra's capillary band: tile L3 = 9 / (1024 / 96) m, band from the master top (k 59.6, complementary to c2's
// fade-out) to k 635 (lambda 1 cm), random draws gaussianField(seed, 3), gravity-capillary dispersion
// w^2 = g k + S k^3 (S = 7.4e-5 m^3/s^2). Opt-in (createOceanFft({ capillary: true }) or setCapillary), and its
// normals are weighted by `capillaryGain` (default 1 when enabled). With the current material (fineNormals off) it is
// invisible (live toggle on a paused C2 shot: 0.005 % of pixels change), so it stays OFF by default.
//
// COST (RTX 2070, FFT pass incl. derived fields + mips, compute timestamps, best p50 of 3 x 600 frames, GPU shared
// with other jobs 33-44 %, 2026-10-23): low 0.118 ms (75 % of medium), medium 0.158 ms, high = ultra = medium,
// ultra + capillary 0.20 ms. Memory (textures + buffers): low 8.8 MB (69 %), medium / high / ultra 12.9 MB, + capillary
// 17.1 MB (reference FFT ~20 MB). Texels per frame: low 135 k (69 %), medium 197 k.
import {
  CASCADES, REFERENCE_SIZE, OMEGA_UNIT, LOOP_PERIOD, referenceCascades, resampleCentred, gaussianField,
  generateRawRows, packInitialSpectrum, omegaIndicesGrid, spectrumScalars,
} from '../spectrum.ts';
import type { CascadeDef, SpectrumParams } from '../spectrum.ts';

export type OceanTier = 'low' | 'medium' | 'high' | 'ultra';

export interface FftTierSpec {
  /** Grid size of cascades 0, 1, 2 (c0 / c1 are always 256: they share the base texture; c2 is the detail texture). */
  readonly sizes: readonly [number, number, number];
  /** The tier allows the capillary band (still opt-in). */
  readonly capillary: boolean;
}

export const FFT_TIERS: Readonly<Record<OceanTier, FftTierSpec>> = Object.freeze({
  low: Object.freeze({ sizes: [256, 256, 64] as const, capillary: false }),
  medium: Object.freeze({ sizes: [256, 256, 256] as const, capillary: false }),
  high: Object.freeze({ sizes: [256, 256, 256] as const, capillary: false }),
  ultra: Object.freeze({ sizes: [256, 256, 256] as const, capillary: true }),
});
export const TIER_NAMES: readonly OceanTier[] = ['low', 'medium', 'high', 'ultra'];
export const NUM_CASCADES = 3;
/** The capillary cascade: tile, band (lower = the master top), offset. */
export const CAPILLARY_CASCADE: CascadeDef = Object.freeze(referenceCascades([1024, 96, 9, 9 / (1024 / 96)])[3]);
export const CAPILLARY_SIZE = 256;

/** Cascades drawn by a tier (always the core's CASCADES; tiers only change grid sizes). */
export function tierCascades(_t?: OceanTier): CascadeDef[] { return CASCADES.map((c) => ({ ...c })); }

// ------------------------------------------------------------------------------------------------------------
// Master subsets (exact: same cells, same values)
// ------------------------------------------------------------------------------------------------------------
/** h0-only (2 floats / cell) centred size-n subset of a core master slot array (4 floats / cell). Allocates. */
export function masterH0Subset(master: Float32Array, n: number): Float32Array {
  const packed = n === REFERENCE_SIZE ? master : resampleCentred(master, new Float32Array(4 * n * n), n, 4, true);
  const out = new Float32Array(2 * n * n);
  for (let i = 0; i < n * n; i++) { out[2 * i] = packed[4 * i]; out[2 * i + 1] = packed[4 * i + 1]; }
  return out;
}
/** Centred size-n subset of a master omega-index table. */
export function masterOmegaSubset(master: Float32Array, n: number): Float32Array {
  return n === REFERENCE_SIZE ? master.slice() : resampleCentred(master, new Float32Array(n * n), n, 1, false);
}

/** Slope variance (sum k^2 (|h0|^2 + |h0m|^2)) of the master cells OUTSIDE the centred size-n block. */
export function droppedSlopeVariance(master: Float32Array, n: number, domain: number): number {
  if (n >= REFERENCE_SIZE) return 0;
  const m = REFERENCE_SIZE, h = n / 2, dk = (2 * Math.PI) / domain;
  let s = 0;
  for (let y = 0; y < m; y++) {
    const ky = y - m / 2;
    for (let x = 0; x < m; x++) {
      const kx = x - m / 2;
      if (kx > -h && kx < h && ky > -h && ky < h) continue; // kept (the new Nyquist line -h is dropped)
      const i = 4 * (y * m + x);
      const e = master[i] * master[i] + master[i + 1] * master[i + 1] + master[i + 2] * master[i + 2] + master[i + 3] * master[i + 3];
      s += e * (kx * kx + ky * ky);
    }
  }
  return s * dk * dk;
}

/** Capillary band spectrum (h0-only, n^2 x 2) + its omega table, for one slot's parameters. Worker / setup. */
export function generateCapillary(params: Readonly<SpectrumParams>, seed: number, g: number, n = CAPILLARY_SIZE): { h0: Float32Array; omega: Float32Array } {
  const m = REFERENCE_SIZE;
  const xi = gaussianField(seed, 3, new Float64Array(2 * m * m));
  const raw = new Float64Array(2 * m * m);
  generateRawRows(params, spectrumScalars(params, g, { g: 0, U: 0, dirGrid: 0, wp: 0 }), CAPILLARY_CASCADE, xi, raw, 0, m);
  const packed = packInitialSpectrum(raw, params.standing, new Float32Array(4 * m * m));
  const h0 = masterH0Subset(packed, n);
  const omega = omegaIndicesGrid(CAPILLARY_CASCADE, g, n, true, new Float32Array(n * n));
  return { h0, omega };
}

// ------------------------------------------------------------------------------------------------------------
// Exact CPU evaluation of the drawn fields (independent of cpu-fft.ts; for checks / tests)
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
    const ang = (2 * Math.PI) / len;
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
    const mr = jr, mi = -ji;
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
// CPU replica of the GPU displacement lookup at LOD 0 (parity analysis). cubic = Catmull-Rom (same weights /
// texel mapping as the GPU kernel and the core's sampleCubic).
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

/** Chained lookup (reference spectralField) at local rest (e, nn): out = [up, east disp, north disp]. */
export function sampleChained(fields: readonly ArrayLike<number>[], sizes: readonly number[], cascades: readonly CascadeDef[], e: number, nn: number, cubic: readonly boolean[], out: Float64Array, tmp: Float64Array): void {
  let U = 0, E = 0, N = 0;
  for (let c = 0; c < fields.length; c++) {
    const L = cascades[c].domain, off = cascades[c].offset;
    sampleField(fields[c], sizes[c], L, e + E + off, -(nn + N) + off, cubic[c], tmp);
    U += tmp[0]; E += tmp[1]; N += tmp[2];
  }
  out[0] = U; out[1] = E; out[2] = N;
}
