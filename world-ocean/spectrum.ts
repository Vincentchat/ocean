// JONSWAP + directional spreading spectrum, cascades and seeding. Pure functions, no three import.
//
// Ported from the WebGL reference `src/spectral-ocean.js` (nagi-ocean-sim fork, MIT). Every
// constant below is the reference's unless stated otherwise; see "REFERENCE CONSTANTS".
//
// =====================================================================================
// CONVENTIONS (the GPU FFT MUST follow these exactly; the CPU query in index.ts does)
// =====================================================================================
//
// Grid frame. Each cascade c is a periodic tile of side L_c metres. Its spectral grid is N x N
// (N = power of two). Spectral cell (x, y), x,y in [0, N), array index i = y * N + x, holds the
// wave vector
//     kx = (x - N/2) * dk,   kz = (y - N/2) * dk,   dk = 2*pi / L_c      (rad/m)
// i.e. the spectrum is stored CENTRED (k = 0 at (N/2, N/2)), exactly like the reference.
// The tile's two horizontal axes are  a = local EAST  and  b = local SOUTH (= -north). This is the
// reference's (world +X, world +Z) on the canonical flat frame (east +X, north -Z), so the flat
// case reproduces the reference waves wave-for-wave AT THE REFERENCE GRAVITY 9.81 (verified to float32).
// With the shipped g = shape.surfaceGravity (9.80665, spec section 3) 84 % of the integer dispersion
// indices differ by up to 8, so the waves are identical at t = 0 and drift apart in phase (2.5 % rms
// of height at 60 s, decorrelated after ~1 h): same statistics, not the same waves at late times.
//
// Initial spectrum layout  (Float32Array, 4 floats per cell, index 4*i):
//     [ Re h0(k), Im h0(k), Re h0m(k), Im h0m(k) ]     with  h0m(k) = conj(h0(-k))
// where -k is the cell ((N - x) % N, (N - y) % N). Row x = 0 / column y = 0 (the Nyquist line)
// is always 0. The amplitude multiplier is already folded in (metres).
//
// Angular frequency: omega(k) = OMEGA_UNIT * m(k), m an integer stored (exactly) in a
// Float32Array `omegaIndex`, OMEGA_UNIT = 2*pi / LOOP_PERIOD. So every wave repeats after
// LOOP_PERIOD = 8192 s and the phase can be computed without float32 precision loss:
//     phase/(2*pi) = fract( ((m * T_int) mod LOOP_PERIOD + m * T_frac) / LOOP_PERIOD )
// with T = time mod LOOP_PERIOD = T_int (integer seconds, u32) + T_frac (float in [0,1)).
//
// Time evolution (Tessendorf, waves travel along +k):
//     ht(k, t)  = h0(k) * exp(-i w t) + h0m(k) * exp(+i w t)
//     dht(k, t) = d/dt ht = -i w h0(k) exp(-i w t) + i w h0m(k) exp(+i w t)
// Let kl = |k| + 1e-4 (the reference's `length(k) + .0001`; kx, kz are the float32 values).
//
// Fields (inverse FFT = sum_k F(k) exp(+i k.x), UNNORMALISED, see cpu-fft.ts; then the centred
// spectrum needs the checkerboard factor s(x,y) = (-1)^(x+y) on the output texel (x, y)):
//     Hraw = IFFT( ht )                         (real)
//     Da   = IFFT( -i (kx/kl) ht )              (real)  displacement along a (east)
//     Db   = IFFT( -i (kz/kl) ht )              (real)  displacement along b (south)
// A convenient packing (the reference's): IFFT( ht * (1 + kx/kl) ) = Hraw + i Da.
// Output texel (x, y) of cascade c is the field at tile position
//     a = x * L/N,  b = y * L/N   (metres, modulo L)
// and the displacement the surface point at rest (a, b) receives is (metres):
//     up    = -Hraw * s                 <- NOTE the minus sign (reference: vec3(f.y*chop, -f.x, f.z*chop))
//     east  =  chop * Da * s
//     north = -chop * Db * s            (b is south)
// Velocities: the same with dht in place of ht (m/s).
//
// Tile mapping. A local tangent-plane point (east e, north n) samples cascade c at tile position
//     a = e + OFF_c,   b = -n + OFF_c,     OFF_c = L_c * (0.5 - 0.5 / 256)
// i.e. texel coordinate s = a * N / L_c (bilinear between floor(s) and floor(s)+1, wrapping).
// For N = 256 this is exactly the reference's `uv = rest / L + .5` with texel centres at
// (p + .5) / N; OFF_c is N-independent so every tier draws the same surface.
// With hardware sampling (texel centres at (p + .5)/N):  uv = a / L_c + 0.5 / N.
//
// Cascade chaining (reference `spectralField`): smaller cascades are sampled at the point already
// displaced horizontally by the larger ones:
//     A = F0(r);  q = r + A.horiz;  B = F1(q);  C = F2(q + B.horiz);  D = A + B + C
// The surface point at rest r is drawn at r + D.horiz, height D.up.
//
// Crossfade (wind / preset changes, no pop): the drawn h0 is
//     h0 = lerp(h0[fromSlot], h0[toSlot], w(t))   (both h0 and h0m),   chop = lerp(chop_from, chop_to, w(t))
//     w(t) = smoothstep(clamp((t - blendStart) / blendDuration, 0, 1))      (w = 1 if fromSlot == toSlot)
// fromSlot / toSlot / blendStart / blendDuration come from ocean.getState() (they change when
// state.blendVersion bumps; blendStart may lie up to ~2 s in the future: w = 0 until then).
// Evaluate w and chop at the draw time with ocean.blendAt(t) / ocean.choppinessAt(t).
//
// GPU checklist for CPU = GPU parity (what index.ts's WATER_QUERY assumes is drawn):
//  1. Upload state.slots[s].h0[c] (256 master layout; spectrum.resampleCentred() for tier size N)
//     whenever slots[s].version changes, plus state.omegaIndex[c] once. Size-N grids of the same
//     cascade are exact subsets / zero-paddings of the 256 master: same waves at every tier.
//  2. Evolve at the EXACT draw time (FrameInfo.interpolatedTime) with the integer-index phase above
//     (no snapshot interpolation in time; the reference's 20-60 Hz snapshots + lerp are dropped).
//  3. h0 = lerp(slot[fromSlot], slot[toSlot], ocean.blendAt(t)); chop = ocean.choppinessAt(t).
//  4. Displacement texture per cascade = (up, east, north) as defined above, chop applied to east /
//     north only. rgba16float is fine (~2 mm quantisation at 4 m amplitudes).
//  5. Vertex displacement: chained lookup (above), CATMULL-ROM (a = -0.5) on the 256 master grid at LOD 0:
//     texel coordinate s = a * N / L (a = east + OFF, b = -north + OFF, wrapped with origin.wrapped(L)
//     on the GPU), taps floor(s) - 1 .. floor(s) + 2 (texel x holds the field at x * L / N). The CPU
//     query samples cascades 0 / 1 the same way (filter 'cubic'); cascade 2 bilinear at 128.
//     Mip levels for distant vertices are fine (parity matters near the camera).
//  6. Measured CPU vs "256^3 bilinear exact-time" parity: tests/core-parity.test.ts.
//
// =====================================================================================
// REFERENCE CONSTANTS (src/spectral-ocean.js unless noted)
// =====================================================================================
//  FFT_SIZE 256 .............. REFERENCE_SIZE: the master grid; all band limits are fixed at it.
//  DOMAINS [1024, 96, 9] ..... cascade tile sizes, m.
//  gravity 9.81 .............. replaced by shape.surfaceGravity (spec ??3: 9.80665 default).
//  windSpeed 15 m/s, windDirection 1 rad (towards +X..+Z = towards east..south), peakWavelength 47 m,
//  gamma 3.3, spread 1.57, standing 0.13, choppiness 1.5 (createSpectralOcean `chop`), period 8192 s.
//  alpha = 0.076 * (U * wp / (22 g))^0.66  (JONSWAP alpha with fetch eliminated through wp).
//  sigma = 0.07 (w < wp) / 0.09 (w >= wp); peak-shape denominator 2 sigma^2 wp^2 + 1e-4.
//  energy = alpha g^2 / w^5 * exp(-1.25 (wp/w)^4) * gamma^r * (0.5 sqrt(g/k) / k)   (S(w) -> S(kx,kz)).
//  spreading: s = max(0.5, spread * 9.77 * (w/wp)^(w < wp ? 5 : -2.5)),
//             D = cos(theta/2)^(2s) * sqrt(s + .25) / (2 sqrt(pi)) * (m0 + (1 - m0) (q + 1)/2),
//             m0 = 0.07 + 0.93 * standing, q = cos(theta), cos(theta/2) = sqrt(max((q+1)/2, 1e-4)).
//  band: smooth(lower/1.5, lower*1.5, k) * (1 - smooth(upper/1.5, upper*1.5, k)),
//        upper_c = pi * 256 / L_c / 1.5, lower_0 = 1e-9, lower_c = upper_{c-1}.
//  P(k) = energy * D * (2 pi / L)^2 * band;  k clamped >= 1e-4.
//  random: mulberry32-style generator seeded 71831 + band * 13007, consumed row-major over the
//          256 x 256 grid (y outer, x inner), two draws per cell: radius = sqrt(-2 ln max(u1, 1e-4)),
//          phase = 2 pi u2; raw = radius (cos, sin) sqrt(P / 2).
//  standing waves: h0(k) = (1 - st/2) raw(k) + (st/2) conj(raw(-k)).
//  dispersion: w = round(sqrt(g (k + 1e-4)) / unit) * unit, unit = 2 pi / 8192 (deep water, no depth term).
//  Evolution kernel `EVOLVE_FRAGMENT`: ht = h0 e^{-iwt} + conj(h0(-k)) e^{+iwt}, k + .0001, packing above.
//  Display: field = (height, dx, dz) = (-f.x, f.y chop, f.z chop) * amplitude  (BLEND_FRAGMENT).
//  Page amplitude = uWave * ((5 + 12.5 uWind) / 15)^0.33 (src/ocean.js spectralAmplitude); the
//  reference page presets are golden (0.9, 0.6), azure (1, 0.8), storm (1.5, 1).
//
// NEW in the port (documented choices):
//  - Wind speed + FETCH drive the peak: wp = max(22 (g^2 / (U F))^(1/3), 0.855 g / U)
//    (JONSWAP fetch law, Hasselmann et al. 1973, capped at the Pierson-Moskowitz fully developed
//    peak). With U = 15, F = fetchForPeakWavelength(15, 47) (~45.5 km) this is the reference's 47 m.
//  - Seeds: seed 1 reproduces the reference realisation; seed s adds (s - 1) * 1000003 to every
//    cascade's generator seed.
//  - The random draws are generated on the 256 master grid and a size-N grid takes the centred
//    sub-block (N < 256) or zero-pads (N > 256; the band is exactly 0 beyond the 256 Nyquist).
//    So every tier (and the CPU query) shares the same waves.

export const REFERENCE_SIZE = 256;
export const REFERENCE_DOMAINS: readonly number[] = [1024, 96, 9];
export const LOOP_PERIOD = 8192; // s
export const OMEGA_UNIT = (2 * Math.PI) / LOOP_PERIOD; // rad/s
export const REFERENCE_GRAVITY = 9.81; // m/s^2 (reference); the port uses shape.surfaceGravity
export const REFERENCE_SEED_BASE = 71831;
export const REFERENCE_SEED_BAND_STRIDE = 13007;
export const SEED_STRIDE = 1000003;
/** Reference page default amplitude: wave 1.5 * ((5 + 12.5 * wind 1) / 15)^0.33 (src/ocean.js spectralAmplitude, preset storm). */
export const REFERENCE_AMPLITUDE = 1.5 * Math.pow((5 + 12.5 * 1) / 15, 0.33);
/** Fully developed (Pierson-Moskowitz) peak: wp >= PM_PEAK * g / U. */
export const PM_PEAK = 0.855;
/** JONSWAP fetch law constant: wp = 22 (g^2 / (U F))^(1/3). */
export const JONSWAP_FETCH_CONSTANT = 22;

/** Parameters of one spectrum (one crossfade slot). */
export interface SpectrumParams {
  /** Wind speed at 10 m, m/s. */
  windSpeed: number;
  /** Direction the wind blows TOWARDS, radians, CCW from local east towards local north. */
  windDirection: number;
  /** Fetch, metres (distance over which the wind has blown). */
  fetch: number;
  /** JONSWAP peak enhancement (1 = Pierson-Moskowitz shape). */
  gamma: number;
  /** Directional spreading multiplier (reference `spread`). Higher = narrower around the wind. */
  spread: number;
  /** Fraction of energy in counter-propagating (standing) waves, 0..1. */
  standing: number;
  /** Horizontal (choppy) displacement multiplier. */
  choppiness: number;
  /** Overall height multiplier (artistic; 1 = physical spectrum). Folded into h0. */
  amplitude: number;
}

export interface CascadeDef {
  /** Tile side, metres. */
  readonly domain: number;
  /** Band lower k (rad/m), smoothstep from lower/1.5 to lower*1.5. */
  readonly lower: number;
  /** Band upper k (rad/m), fade-out smoothstep from upper/1.5 to upper*1.5. */
  readonly upper: number;
  /** Tile offset OFF = L (0.5 - 0.5/256), metres: a = east + OFF, b = -north + OFF. */
  readonly offset: number;
}

/** The reference's three cascades with band limits fixed at the 256 master grid. */
export function referenceCascades(domains: readonly number[] = REFERENCE_DOMAINS): CascadeDef[] {
  const out: CascadeDef[] = [];
  let lower = 1e-9;
  for (let c = 0; c < domains.length; c++) {
    const domain = domains[c];
    const upper = (Math.PI * REFERENCE_SIZE) / domain / 1.5;
    out.push({ domain, lower, upper, offset: domain * (0.5 - 0.5 / REFERENCE_SIZE) });
    lower = upper;
  }
  return out;
}

export const CASCADES: readonly CascadeDef[] = Object.freeze(referenceCascades());

// ---------------------------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------------------------
function clamp(x: number, a: number, b: number): number { return x < a ? a : x > b ? b : x; }
function smooth(a: number, b: number, x: number): number {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
}

/**
 * Hard caps (sea-state sanity + tile resolution): wind speed <= 35 m/s, fetch <= 200 km and peak
 * wavelength <= 160 m (>= 6.4 peak waves per 1024 m tile; also keeps the peak well resolved by
 * cascade 0). With the reference calibration (amplitude 1.578) the highest reachable Hs is ~14.9 m
 * (35 m/s, peak capped at 160 m); without the caps 35 m/s over 2000 km gave Hs 68 m, lambda_p 1 km.
 */
export const MAX_WIND_SPEED = 35;      // m/s
export const MAX_FETCH = 200000;       // m
export const MAX_PEAK_WAVELENGTH = 160; // m

/** Wind speed actually used by the spectrum (clamped to [0.1, MAX_WIND_SPEED]), m/s. */
export function effectiveWindSpeed(windSpeed: number): number {
  return windSpeed < 0.1 ? 0.1 : windSpeed > MAX_WIND_SPEED ? MAX_WIND_SPEED : windSpeed;
}

/** Peak angular frequency, rad/s, from wind speed (m/s) and fetch (m), with the caps above. */
export function peakOmega(windSpeed: number, fetch: number, g: number): number {
  const U = effectiveWindSpeed(windSpeed);
  const F = Math.min(Math.max(fetch, 1), MAX_FETCH);
  const fetchLimited = JONSWAP_FETCH_CONSTANT * Math.cbrt((g * g) / (U * F));
  const minOmega = Math.sqrt((2 * Math.PI * g) / MAX_PEAK_WAVELENGTH);
  return Math.max(fetchLimited, (PM_PEAK * g) / U, minOmega);
}

/** Peak wavelength, m (deep water: lambda = 2 pi g / wp^2). */
export function peakWavelength(windSpeed: number, fetch: number, g: number): number {
  const wp = peakOmega(windSpeed, fetch, g);
  return (2 * Math.PI * g) / (wp * wp);
}

/** Peak period, s. */
export function peakPeriod(windSpeed: number, fetch: number, g: number): number {
  return (2 * Math.PI) / peakOmega(windSpeed, fetch, g);
}

/** Fetch (m) that makes the fetch law give `wavelength` (m) at `windSpeed` (inverse of peakWavelength). */
export function fetchForPeakWavelength(windSpeed: number, wavelength: number, g: number): number {
  const wp = Math.sqrt((2 * Math.PI * g) / wavelength);
  const r = wp / JONSWAP_FETCH_CONSTANT; // = (g^2/(U F))^(1/3)
  return (g * g) / (Math.max(windSpeed, 0.1) * r * r * r);
}

/** Quantised deep-water dispersion (reference `dispersion`): integer multiple m of OMEGA_UNIT. */
export function omegaIndexOf(k: number, g: number): number {
  return Math.round(Math.sqrt(g * (k + 1e-4)) / OMEGA_UNIT);
}

/**
 * Directional wave-number spectrum P(kx, kz) (m^2 per spectral cell), reference `spectralDensity`.
 * (kx, kz) in the grid frame (a = east, b = south). `dirGrid` = wind direction in the grid frame
 * (radians from +a towards +b) = -windDirection.
 */
export function spectralDensity(
  kx: number, kz: number, domain: number, lower: number, upper: number,
  g: number, U: number, dirGrid: number, wp: number, gamma: number, spread: number, standing: number,
): number {
  const k = Math.max(Math.hypot(kx, kz), 1e-4);
  const omega = Math.sqrt(g * k);
  const ratio = omega / wp;
  const alpha = 0.076 * Math.pow((Math.max(U, 0.1) * wp) / (22 * g), 0.66);
  const sigma = omega < wp ? 0.07 : 0.09;
  const peakShape = Math.exp(-((omega - wp) * (omega - wp)) / (2 * sigma * sigma * wp * wp + 1e-4));
  const energy =
    ((alpha * g * g) / Math.pow(omega, 5)) *
    Math.exp(-1.25 / Math.pow(ratio, 4)) *
    Math.pow(gamma, peakShape) *
    ((0.5 * Math.sqrt(g / k)) / k);
  const q = clamp((kx * Math.cos(dirGrid) + kz * Math.sin(dirGrid)) / k, -1, 1);
  const s = Math.max(0.5, spread * 9.77 * Math.pow(ratio, omega < wp ? 5 : -2.5));
  const halfCos = Math.sqrt(Math.max((q + 1) * 0.5, 1e-4));
  const m0 = 0.07 + 0.93 * standing;
  const directional =
    ((Math.pow(halfCos, 2 * s) * Math.sqrt(s + 0.25)) / (2 * Math.sqrt(Math.PI))) *
    (m0 + (1 - m0) * (q + 1) * 0.5);
  const band = smooth(lower / 1.5, lower * 1.5, k) * (1 - smooth(upper / 1.5, upper * 1.5, k));
  const dk = (2 * Math.PI) / domain;
  return energy * directional * dk * dk * band;
}

// ---------------------------------------------------------------------------------------------
// Seeding
// ---------------------------------------------------------------------------------------------

/** The reference's generator (mulberry32 variant), returns [0, 1). Allocates a closure: setup only. */
export function randomGenerator(seed: number): () => number {
  let state = seed;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function cascadeSeed(seed: number, cascade: number): number {
  return (REFERENCE_SEED_BASE + cascade * REFERENCE_SEED_BAND_STRIDE + (seed - 1) * SEED_STRIDE) | 0;
}

/**
 * Unit complex Gaussian draws (E|xi|^2 = 2, as the reference: radius = sqrt(-2 ln u)) on the
 * 256 x 256 master grid of one cascade, row-major. `out` length 2 * 256^2: [re, im] per cell.
 * Setup-time (once per seed).
 */
export function gaussianField(seed: number, cascade: number, out: Float64Array): Float64Array {
  const n = REFERENCE_SIZE;
  if (out.length < 2 * n * n) throw new RangeError('gaussianField: out too small');
  const rng = randomGenerator(cascadeSeed(seed, cascade));
  for (let i = 0; i < n * n; i++) {
    const radius = Math.sqrt(-2 * Math.log(Math.max(rng(), 1e-4)));
    const phase = 2 * Math.PI * rng();
    out[2 * i] = radius * Math.cos(phase);
    out[2 * i + 1] = radius * Math.sin(phase);
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Spectrum generation into the GPU layout
// ---------------------------------------------------------------------------------------------

/** Derived per-slot constants (hoisted out of the per-cell loop). */
export interface SpectrumScalars { g: number; U: number; dirGrid: number; wp: number }

export function spectrumScalars(p: Readonly<SpectrumParams>, g: number, out: SpectrumScalars): SpectrumScalars {
  out.g = g;
  out.U = effectiveWindSpeed(p.windSpeed);
  out.dirGrid = -p.windDirection; // local (east->north) to grid (east->south)
  out.wp = peakOmega(p.windSpeed, p.fetch, g);
  return out;
}

/**
 * Raw (pre standing-mix) amplitudes sqrt(P/2) * amplitude * xi for master rows [rowStart, rowEnd)
 * of one cascade into `raw` (2 floats per master cell). Allocation-free; call repeatedly with
 * row ranges to amortise a regeneration over several ticks.
 */
export function generateRawRows(
  p: Readonly<SpectrumParams>, sc: Readonly<SpectrumScalars>, cascade: Readonly<CascadeDef>,
  xi: Float64Array, raw: Float64Array, rowStart: number, rowEnd: number,
): void {
  const n = REFERENCE_SIZE, half = n / 2;
  const dk = (2 * Math.PI) / cascade.domain;
  const kMax = cascade.upper * 1.5; // band is exactly 0 beyond
  const amp = p.amplitude;
  // Same maths as spectralDensity(), with the per-slot constants hoisted and pow() replaced by
  // exp/log/multiplies (~4x faster; agrees to ~1e-13 relative, see tests/core-spectrum.test.ts).
  const g = sc.g, wp = sc.wp;
  const alphaG2 = 0.076 * Math.pow((Math.max(sc.U, 0.1) * wp) / (22 * g), 0.66) * g * g;
  const lnGamma = Math.log(p.gamma);
  const cd = Math.cos(sc.dirGrid), sd = Math.sin(sc.dirGrid);
  const s9 = p.spread * 9.77;
  const m0 = 0.07 + 0.93 * p.standing;
  const inv2sqrtPi = 1 / (2 * Math.sqrt(Math.PI));
  const lo0 = cascade.lower / 1.5, lo1 = cascade.lower * 1.5, up0 = cascade.upper / 1.5, up1 = kMax;
  const dk2 = dk * dk;
  for (let y = rowStart; y < rowEnd; y++) {
    const kz = Math.fround((y - half) * dk);
    for (let x = 0; x < n; x++) {
      const i = y * n + x;
      const kx = Math.fround((x - half) * dk);
      let P = 0;
      const kr = Math.sqrt(kx * kx + kz * kz);
      if (x !== 0 && y !== 0 && kr < kMax) {
        const k = kr > 1e-4 ? kr : 1e-4;
        const omega = Math.sqrt(g * k);
        const ratio = omega / wp, r2 = ratio * ratio, r4 = r2 * r2;
        const below = omega < wp;
        const sigma = below ? 0.07 : 0.09;
        const dw = omega - wp;
        const peakShape = Math.exp(-(dw * dw) / (2 * sigma * sigma * wp * wp + 1e-4));
        const o2 = omega * omega;
        const energy = (alphaG2 / (o2 * o2 * omega)) * Math.exp(-1.25 / r4 + peakShape * lnGamma) *
          ((0.5 * Math.sqrt(g / k)) / k);
        let q = (kx * cd + kz * sd) / k;
        q = q < -1 ? -1 : q > 1 ? 1 : q;
        let s = s9 * (below ? r4 * ratio : 1 / (r2 * Math.sqrt(ratio)));
        if (s < 0.5) s = 0.5;
        const hc2 = (q + 1) * 0.5;
        const directional = Math.exp(s * Math.log(hc2 > 1e-4 ? hc2 : 1e-4)) * Math.sqrt(s + 0.25) * inv2sqrtPi *
          (m0 + (1 - m0) * hc2);
        let bandLo = (k - lo0) / (lo1 - lo0); bandLo = bandLo < 0 ? 0 : bandLo > 1 ? 1 : bandLo;
        let bandUp = (k - up0) / (up1 - up0); bandUp = bandUp < 0 ? 0 : bandUp > 1 ? 1 : bandUp;
        const band = bandLo * bandLo * (3 - 2 * bandLo) * (1 - bandUp * bandUp * (3 - 2 * bandUp));
        P = energy * directional * dk2 * band;
      }
      const a = Math.sqrt(P * 0.5) * amp;
      raw[2 * i] = xi[2 * i] * a;
      raw[2 * i + 1] = xi[2 * i + 1] * a;
    }
  }
}

/**
 * Standing-wave mix + pack into the GPU layout at the master size (256):
 * out[4i..4i+3] = [h0(k), conj(h0(-k))], h0(k) = (1 - st/2) raw(k) + (st/2) conj(raw(-k)).
 * Allocation-free.
 */
export function packInitialSpectrum(raw: Float64Array, standing: number, out: Float32Array): Float32Array {
  const n = REFERENCE_SIZE;
  const a = 1 - standing * 0.5, b = standing * 0.5;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = y * n + x;
    const j = ((n - y) % n) * n + ((n - x) % n);
    // h0(k)
    const hr = a * raw[2 * i] + b * raw[2 * j];
    const hi = a * raw[2 * i + 1] - b * raw[2 * j + 1];
    // h0(-k)
    const mr = a * raw[2 * j] + b * raw[2 * i];
    const mi = a * raw[2 * j + 1] - b * raw[2 * i + 1];
    out[4 * i] = hr;
    out[4 * i + 1] = hi;
    out[4 * i + 2] = mr;   // conj(h0(-k))
    out[4 * i + 3] = -mi;
  }
  return out;
}

/** Integer dispersion indices m (omega = m * OMEGA_UNIT) at master size, float32 (exact integers). */
export function omegaIndices(cascade: Readonly<CascadeDef>, g: number, out: Float32Array): Float32Array {
  const n = REFERENCE_SIZE, half = n / 2;
  const dk = (2 * Math.PI) / cascade.domain;
  for (let y = 0; y < n; y++) {
    const kz = Math.fround((y - half) * dk);
    for (let x = 0; x < n; x++) {
      const kx = Math.fround((x - half) * dk);
      out[y * n + x] = omegaIndexOf(Math.hypot(kx, kz), g);
    }
  }
  return out;
}

/**
 * Copy a master-size (256) per-cell array (`stride` floats per cell) into a size-N grid with the
 * same centred convention: N < 256 takes the centred sub-block (k index -N/2 .. N/2-1) and zeroes
 * the new Nyquist line (x = 0 or y = 0) for layouts flagged `zeroNyquist`; N > 256 zero-pads.
 * Allocation-free. Used by the GPU adapter for its tier size and by the CPU query.
 */
export function resampleCentred(
  src: Float32Array, dst: Float32Array, n: number, stride: number, zeroNyquist: boolean,
): Float32Array {
  const m = REFERENCE_SIZE;
  if (dst.length < n * n * stride) throw new RangeError('resampleCentred: dst too small');
  for (let y = 0; y < n; y++) {
    const ky = y - n / 2; // signed k index
    const sy = ky + m / 2;
    for (let x = 0; x < n; x++) {
      const kx = x - n / 2;
      const sx = kx + m / 2;
      const d = (y * n + x) * stride;
      const inside = sx >= 0 && sx < m && sy >= 0 && sy < m && !(zeroNyquist && (x === 0 || y === 0));
      if (inside) {
        const s = (sy * m + sx) * stride;
        for (let c = 0; c < stride; c++) dst[d + c] = src[s + c];
      } else {
        for (let c = 0; c < stride; c++) dst[d + c] = 0;
      }
    }
  }
  return dst;
}

/**
 * Omega indices for a size-N grid computed directly (needed for N > 256 where the master has no
 * entry; identical to the master values inside). Setup-time.
 */
export function omegaIndicesSized(cascade: Readonly<CascadeDef>, g: number, n: number, out: Float32Array): Float32Array {
  const dk = (2 * Math.PI) / cascade.domain;
  for (let y = 0; y < n; y++) {
    const kz = Math.fround((y - n / 2) * dk);
    for (let x = 0; x < n; x++) {
      const kx = Math.fround((x - n / 2) * dk);
      out[y * n + x] = omegaIndexOf(Math.hypot(kx, kz), g);
    }
  }
  return out;
}

/**
 * Convenience (setup-time, allocates): full initial spectrum of one cascade at master size.
 * Returns { h0: Float32Array(256^2 * 4), omegaIndex: Float32Array(256^2), power } where power is
 * sum P * amplitude^2 (m^2) before the standing mix.
 */
export function generateCascade(
  params: Readonly<SpectrumParams>, cascadeIndex: number, seed: number, g: number,
  cascades: readonly CascadeDef[] = CASCADES,
): { h0: Float32Array; omegaIndex: Float32Array; power: number } {
  const n = REFERENCE_SIZE;
  const xi = gaussianField(seed, cascadeIndex, new Float64Array(2 * n * n));
  const raw = new Float64Array(2 * n * n);
  const sc = spectrumScalars(params, g, { g: 0, U: 0, dirGrid: 0, wp: 0 });
  generateRawRows(params, sc, cascades[cascadeIndex], xi, raw, 0, n);
  const h0 = packInitialSpectrum(raw, params.standing, new Float32Array(4 * n * n));
  const omegaIndex = omegaIndices(cascades[cascadeIndex], g, new Float32Array(n * n));
  return { h0, omegaIndex, power: cascadePower(params, cascades[cascadeIndex], g) };
}

/** Sum of P(k) * amplitude^2 over the master grid of one cascade (m^2). Allocation-free. */
export function cascadePower(p: Readonly<SpectrumParams>, cascade: Readonly<CascadeDef>, g: number): number {
  const n = REFERENCE_SIZE, half = n / 2;
  const dk = (2 * Math.PI) / cascade.domain;
  const kMax = cascade.upper * 1.5;
  const wp = peakOmega(p.windSpeed, p.fetch, g);
  let sum = 0;
  for (let y = 1; y < n; y++) {
    const kz = Math.fround((y - half) * dk);
    for (let x = 1; x < n; x++) {
      const kx = Math.fround((x - half) * dk);
      if (Math.hypot(kx, kz) >= kMax) continue;
      sum += spectralDensity(kx, kz, cascade.domain, cascade.lower, cascade.upper,
        g, effectiveWindSpeed(p.windSpeed), -p.windDirection, wp, p.gamma, p.spread, p.standing);
    }
  }
  return sum * p.amplitude * p.amplitude;
}

/**
 * Expected (time-averaged, ensemble) variance of the surface height, m^2, summed over cascades:
 * sum_c power_c * (1 + (1 - standing)^2)  (reference `expectedVariance`).
 */
export function expectedHeightVariance(p: Readonly<SpectrumParams>, g: number, cascades: readonly CascadeDef[] = CASCADES): number {
  let v = 0;
  for (let c = 0; c < cascades.length; c++) v += cascadePower(p, cascades[c], g);
  return v * (1 + (1 - p.standing) * (1 - p.standing));
}

/** Expected significant wave height Hs = 4 sigma, m. */
export function expectedHs(p: Readonly<SpectrumParams>, g: number, cascades: readonly CascadeDef[] = CASCADES): number {
  return 4 * Math.sqrt(expectedHeightVariance(p, g, cascades));
}

// =============================================================================================
// Gravity-capillary dispersion (ADDITIVE; used only by the GPU adapter's optional capillary band above the
// master spectrum top; the master cascades keep the reference's quantised gravity dispersion).
// =============================================================================================

/** Surface tension / density of sea water, m^3/s^2 (gravity-capillary dispersion w^2 = g k + S k^3). */
export const CAPILLARY_S = 7.4e-5;

/** Quantised gravity-capillary dispersion: round(sqrt(g (k + 1e-4) + S k^3) / OMEGA_UNIT). */
export function omegaIndexOfCapillary(k: number, g: number): number {
  return Math.round(Math.sqrt(g * (k + 1e-4) + CAPILLARY_S * k * k * k) / OMEGA_UNIT);
}

/** Omega indices of a size-n grid (gravity, or gravity-capillary when `capillary`). Setup-time. */
export function omegaIndicesGrid(cascade: Readonly<CascadeDef>, g: number, n: number, capillary: boolean, out: Float32Array): Float32Array {
  const dk = (2 * Math.PI) / cascade.domain;
  for (let y = 0; y < n; y++) {
    const kz = Math.fround((y - n / 2) * dk);
    for (let x = 0; x < n; x++) {
      const kx = Math.fround((x - n / 2) * dk);
      const k = Math.hypot(kx, kz);
      out[y * n + x] = capillary ? omegaIndexOfCapillary(k, g) : omegaIndexOf(k, g);
    }
  }
  return out;
}
