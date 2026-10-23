// @dreamp/world-ocean ??? public entry. Render-agnostic core: NEVER import three here.
//
//   const ocean = createOcean({ preset: 'breeze', seed: 1 });   // WorldSystem, provides WATER_QUERY
//   world.use(ocean); world.start();
//   const h = world.require(WATER_QUERY).heightAt({ x: 10, y: 0, z: 5 }, world.time);
//
// CPU / GPU agreement path (spec ??1 rule 8): the CPU runs the SAME spectrum + FFT as the GPU
// (cpu-fft.ts CascadeSim), following the conventions documented at the top of spectrum.ts.
// Cascade 0 (96 % of the height variance) is evaluated at 256^2 with bilinear sampling, i.e. the
// same grid + filter as the GPU at 256; cascades 1 and 2 at lower sizes (exact spectral subsets).
//
// Wind coupling + crossfade (no pops): the WIND_QUERY sample (10 m above sea level at the focus
// point) is low-passed (EMA, windTimeConstant). When the smoothed wind drifts beyond tolerance,
// a new spectrum is generated into a spare slot, amortised over ticks (regenRowsPerTick), and the
// drawn h0 crossfades from the old slot to the new one:  h0 = lerp(h0_from, h0_to, w(t)),
// w(t) = smoothstep(clamp((t - blendStart) / blendDuration, 0, 1)); choppiness is lerped the same
// way. Both slots share the same random draws (same seed), so only amplitudes change: no phase
// cancellation, heights stay continuous (C1 in time).
import {
  WATER_QUERY, WIND_QUERY, ENVIRONMENT, TERRAIN_QUERY, TERRAIN_GPU, SKY_LIGHT, DEFAULT_SIM_PARAMS,
  DEFAULT_QUERY_CASCADES,
} from './types.ts';
import type {
  Vec3, TangentFrame, WorldShape, WaterQuery, WindQuery, World, WorldSystem, TickPhase, TickInfo,
  OceanOptions, PresetName, SimParams, SpectrumParams, QueryCascadeConfig, PerfSink, CascadeDef,
} from './types.ts';
import {
  CASCADES, REFERENCE_SIZE, LOOP_PERIOD, OMEGA_UNIT, gaussianField, spectrumScalars, generateRawRows,
  packInitialSpectrum, omegaIndices, expectedHs,
} from './spectrum.ts';
import type { SpectrumScalars } from './spectrum.ts';
import { CascadeSim, SAMPLE, GRAD, sampleCubic, sampleCubicGrad } from './cpu-fft.ts';
import { PRESETS } from './presets/index.ts';

export * from './types.ts';
export * as spectrum from './spectrum.ts';
export { PRESETS, PRESET_NAMES, REFERENCE_SCENE_AMPLITUDE } from './presets/index.ts';
export { CASCADES, REFERENCE_SIZE, LOOP_PERIOD, OMEGA_UNIT } from './spectrum.ts';

export const MODULE_ID = 'world/ocean';
const NUM_SLOTS = 3;
const SNAPSHOT_POOL = 10;
/** Cascades whose gradient enters the Newton Jacobian of the heightAt inversion (see forward()). */
const JACOBIAN_CASCADES = 2;

/** One crossfade slot as the GPU adapter sees it. */
export interface OceanSlot {
  /** The parameters this slot was generated with. */
  readonly params: Readonly<SpectrumParams>;
  /** Bumps when h0 is regenerated: re-upload h0[c] for every cascade. */
  readonly version: number;
  /** Per cascade: master-size (256) initial spectrum, layout [Re h0, Im h0, Re h0m, Im h0m] (spectrum.ts). */
  readonly h0: readonly Float32Array[];
}

/** Everything the GPU adapter needs (stable object; read it in frame()). */
export interface OceanState {
  readonly masterSize: number;              // 256 (use spectrum.resampleCentred for other sizes)
  readonly cascades: readonly CascadeDef[];
  readonly loopPeriod: number;              // s
  readonly omegaUnit: number;               // rad/s; omega = omegaIndex * omegaUnit
  readonly gravity: number;                 // m/s^2
  /** Per cascade, master size: integer dispersion index m (float32, exact). */
  readonly omegaIndex: readonly Float32Array[];
  readonly slots: readonly OceanSlot[];
  /** Bumps whenever any slot's h0 changes. */
  readonly version: number;
  /** Bumps whenever fromSlot / toSlot / blendStart change. */
  readonly blendVersion: number;
  readonly fromSlot: number;
  readonly toSlot: number;
  /** Slot generated but not yet blended in (-1 = none). Upload it ahead of time. */
  readonly pendingSlot: number;
  readonly blendStart: number;              // s (simulation time)
  readonly blendDuration: number;           // s
  /** Values at the last tick (use ocean.blendAt(t) / choppinessAt(t) at draw time). */
  readonly blend: number;
  readonly choppiness: number;
  readonly time: number;
  /** Current target wind (after smoothing), m/s and radians (towards, CCW from east). */
  readonly windSpeed: number;
  readonly windDirection: number;
  readonly focus: Readonly<Vec3>;
  readonly anchor: Readonly<Vec3>;
}

/** Result of sampling the surface by REST point (no inversion), local tangent-plane terms. */
export interface RestSample {
  /** Displaced position, local east / north, m. */
  east: number;
  north: number;
  /** Height above sea level, m. */
  up: number;
  /** Velocity of that surface point, m/s (east, north, up). */
  vEast: number;
  vNorth: number;
  vUp: number;
}

export interface Ocean extends WorldSystem {
  readonly id: string;
  /** The WATER_QUERY implementation (also provided to the world on install). */
  readonly query: WaterQuery;
  /** The camera / area of interest, world space: wind is sampled 10 m above sea level here. */
  setFocus(p: Readonly<Vec3>): void;
  /** Switch preset (crossfades unless immediate). Overrides passed to createOcean are dropped. */
  setPreset(name: PresetName, immediate?: boolean): void;
  /** Change spectrum and / or sim parameters (crossfades unless immediate). */
  setParams(p: Partial<SpectrumParams & SimParams>, immediate?: boolean): void;
  /** Target parameters (preset + overrides + smoothed wind). */
  readonly params: Readonly<SpectrumParams>;
  readonly sim: Readonly<SimParams>;
  getState(): Readonly<OceanState>;
  /** Crossfade weight of toSlot at time t (s). Allocation-free. */
  blendAt(t: number): number;
  /** Choppiness at time t (lerped during crossfades). */
  choppinessAt(t: number): number;
  /** Expected Hs (m) of the target spectrum. */
  expectedHs(): number;
  /** Local (east, north) metres -> height at time t; same as heightAt but without the world mapping. */
  heightAtLocal(east: number, north: number, time: number): number;
  /**
   * Diagnostics / tests: invert the displacement at local (east, north). Writes the REST point to
   * out.east / out.north, the height and velocity to out.up / out.v*; returns the residual |P(r) - p| (m).
   */
  invertLocal(east: number, north: number, time: number, out: RestSample): number;
  /** Surface point that rests at local (east, north): displaced position, height, velocity. */
  sampleRest(east: number, north: number, time: number, out: RestSample): RestSample;
  /** Diagnostics: CPU snapshots computed per cascade so far. */
  snapshotCounts(out: number[]): number[];
}

interface SlotData {
  params: SpectrumParams;
  version: number;
  h0: Float32Array[];
}

function copyParams(src: Readonly<SpectrumParams>, dst: SpectrumParams): SpectrumParams {
  dst.windSpeed = src.windSpeed; dst.windDirection = src.windDirection; dst.fetch = src.fetch;
  dst.gamma = src.gamma; dst.spread = src.spread; dst.standing = src.standing;
  dst.choppiness = src.choppiness; dst.amplitude = src.amplitude;
  return dst;
}
function emptyParams(): SpectrumParams {
  return { windSpeed: 0, windDirection: 0, fetch: 0, gamma: 0, spread: 0, standing: 0, choppiness: 0, amplitude: 0 };
}
const SPECTRUM_KEYS: readonly (keyof SpectrumParams)[] = ['windSpeed', 'windDirection', 'fetch', 'gamma', 'spread', 'standing', 'choppiness', 'amplitude'];
const SIM_KEYS: readonly (keyof SimParams)[] = ['blendTime', 'windTimeConstant', 'windSpeedTolerance', 'windDirectionToleranceDeg', 'regenRowsPerTick', 'inversionIterations', 'inversionTolerance', 'windSampleHeight'];

function wrapAngle(a: number): number {
  a = (a + Math.PI) % (2 * Math.PI);
  if (a < 0) a += 2 * Math.PI;
  return a - Math.PI;
}

export function createOcean(opts: OceanOptions = {}): Ocean {
  const presetName: PresetName = opts.preset ?? 'breeze';
  if (!PRESETS[presetName]) throw new Error(`Unknown ocean preset '${presetName}'`);
  const seed = opts.seed ?? 1;
  const useWind = opts.useWind ?? true;
  const perf: PerfSink | undefined = opts.perf;
  const anchor: Vec3 = { x: opts.anchor?.x ?? 0, y: opts.anchor?.y ?? 0, z: opts.anchor?.z ?? 0 };
  const queryCfg: readonly QueryCascadeConfig[] = opts.query ?? DEFAULT_QUERY_CASCADES;
  if (queryCfg.length !== CASCADES.length) throw new Error(`query must list ${CASCADES.length} cascades`);

  // ---- parameters -------------------------------------------------------------------------
  /** Preset + explicit overrides (no wind). */
  const base: SpectrumParams = copyParams(PRESETS[presetName].params, emptyParams());
  for (const k of SPECTRUM_KEYS) if (typeof opts[k] === 'number') base[k] = opts[k] as number;
  const sim: SimParams = { ...DEFAULT_SIM_PARAMS };
  for (const k of SIM_KEYS) if (typeof opts[k] === 'number') sim[k] = opts[k] as number;
  /** Target = base with the smoothed wind applied. */
  const target: SpectrumParams = copyParams(base, emptyParams());

  // ---- world bits -------------------------------------------------------------------------
  let shape: WorldShape | undefined = opts.shape;
  let wind: WindQuery | undefined;
  let initialised = false;
  let g = 9.80665;
  const frame: TangentFrame = { east: { x: 1, y: 0, z: 0 }, north: { x: 0, y: 0, z: -1 }, up: { x: 0, y: 1, z: 0 } };
  const focus: Vec3 = { x: anchor.x, y: anchor.y, z: anchor.z };
  const windPoint: Vec3 = { x: 0, y: 0, z: 0 };
  const windVec: Vec3 = { x: 0, y: 0, z: 0 };
  let windE = 0, windN = 0, windInit = false;

  // ---- spectrum slots -----------------------------------------------------------------------
  const nm = REFERENCE_SIZE;
  const xi: Float64Array[] = [];
  const omegaIndex: Float32Array[] = [];
  const slots: SlotData[] = [];
  for (let s = 0; s < NUM_SLOTS; s++) {
    const h0: Float32Array[] = [];
    for (let c = 0; c < CASCADES.length; c++) h0.push(new Float32Array(nm * nm * 4));
    slots.push({ params: emptyParams(), version: 0, h0 });
  }
  let version = 0, blendVersion = 0;
  let fromSlot = 0, toSlot = 0, pendingSlot = -1;
  let blendStart = 0, blendDuration = sim.blendTime;
  let lastTickTime = 0;
  let stamp = 0; // bumps when anything that changes materialised frames changes
  // Regeneration job
  const regenRaw = new Float64Array(2 * nm * nm);
  const regenScalars: SpectrumScalars = { g: 0, U: 0, dirGrid: 0, wp: 0 };
  let regenSlot = -1, regenCascade = 0, regenRow = 0;

  // ---- CPU query ---------------------------------------------------------------------------
  const sims: (CascadeSim | null)[] = [];
  const simDomain = new Float64Array(CASCADES.length);
  const simOffset = new Float64Array(CASCADES.length);
  const simN = new Int32Array(CASCADES.length);
  const simCubic = new Uint8Array(CASCADES.length);
  let curFrame = 0;
  // Compact list of the cascades the CPU evaluates (hot path of the queries).
  let K = 0;
  const kSim: CascadeSim[] = [], kCascade: number[] = [], kData: Float32Array[] = [], kVel: Float32Array[] = [];
  const kN: number[] = [], kInv: number[] = [], kL: number[] = [], kOff: number[] = [], kCubic: number[] = [];
  function selectFrame(f: number): void {
    curFrame = f;
    for (let k = 0; k < K; k++) { kData[k] = kSim[k].frames[f]; kVel[k] = kSim[k].velFrames[f]; }
  }

  function init(): void {
    if (initialised) return;
    if (!shape) throw new Error('createOcean: no world shape yet; install() the system or pass options.shape');
    g = shape.surfaceGravity;
    for (let c = 0; c < CASCADES.length; c++) {
      xi.push(gaussianField(seed, c, new Float64Array(2 * nm * nm)));
      omegaIndex.push(omegaIndices(CASCADES[c], g, new Float32Array(nm * nm)));
    }
    for (let c = 0; c < CASCADES.length; c++) {
      const q = queryCfg[c];
      if (q.size > 0) {
        sims.push(new CascadeSim(CASCADES[c], q, omegaIndex[c], NUM_SLOTS, SNAPSHOT_POOL));
        simN[c] = q.size; simDomain[c] = CASCADES[c].domain; simOffset[c] = CASCADES[c].offset;
        simCubic[c] = q.filter === 'cubic' ? 1 : 0;
      } else sims.push(null);
    }
    for (let c = 0; c < sims.length; c++) {
      const sm = sims[c];
      if (!sm) continue;
      kSim.push(sm); kCascade.push(c); kData.push(sm.frames[0]); kVel.push(sm.velFrames[0]);
      kN.push(sm.n); kInv.push(sm.n / sm.domain); kL.push(sm.domain); kOff.push(sm.offset);
      kCubic.push(sm.cubic ? 1 : 0);
    }
    K = kSim.length;
    initialised = true;
    generateNow(0, target);
    fromSlot = toSlot = 0;
    pendingSlot = -1;
    blendVersion++;
    stamp++;
  }

  /** Synchronous generation into a slot (setup / immediate changes). */
  function generateNow(slot: number, p: Readonly<SpectrumParams>): void {
    const sd = slots[slot];
    copyParams(p, sd.params);
    spectrumScalars(sd.params, g, regenScalars);
    for (let c = 0; c < CASCADES.length; c++) {
      generateRawRows(sd.params, regenScalars, CASCADES[c], xi[c], regenRaw, 0, nm);
      packInitialSpectrum(regenRaw, sd.params.standing, sd.h0[c]);
    }
    sd.version++;
    version++;
    for (let c = 0; c < sims.length; c++) sims[c]?.setSlot(slot, sd.h0[c], sd.version);
  }

  function startRegen(): void {
    regenSlot = (toSlot + 1) % NUM_SLOTS;
    if (regenSlot === fromSlot) regenSlot = (regenSlot + 1) % NUM_SLOTS;
    copyParams(target, slots[regenSlot].params);
    spectrumScalars(slots[regenSlot].params, g, regenScalars);
    regenCascade = 0;
    regenRow = 0;
  }

  /** Advance the amortised regeneration by `rows` master rows. */
  function stepRegen(rows: number): void {
    if (regenSlot < 0) return;
    const sd = slots[regenSlot];
    while (rows > 0 && regenSlot >= 0) {
      if (regenRow < nm) {
        const r1 = Math.min(nm, regenRow + rows);
        generateRawRows(sd.params, regenScalars, CASCADES[regenCascade], xi[regenCascade], regenRaw, regenRow, r1);
        rows -= r1 - regenRow;
        regenRow = r1;
      } else {
        packInitialSpectrum(regenRaw, sd.params.standing, sd.h0[regenCascade]);
        rows -= 8; // packing costs about as much as 8 rows
        regenCascade++;
        regenRow = 0;
        if (regenCascade >= CASCADES.length) {
          sd.version++;
          version++;
          for (let c = 0; c < sims.length; c++) sims[c]?.setSlot(regenSlot, sd.h0[c], sd.version);
          pendingSlot = regenSlot;
          regenSlot = -1;
        }
      }
    }
  }

  function differs(a: Readonly<SpectrumParams>, b: Readonly<SpectrumParams>): boolean {
    const du = Math.abs(a.windSpeed - b.windSpeed);
    if (du > sim.windSpeedTolerance * Math.max(b.windSpeed, 1)) return true;
    if (Math.min(a.windSpeed, b.windSpeed) > 0.5 &&
      Math.abs(wrapAngle(a.windDirection - b.windDirection)) > (sim.windDirectionToleranceDeg * Math.PI) / 180) return true;
    if (Math.abs(a.fetch - b.fetch) > 1e-6 * (1 + b.fetch)) return true;
    if (a.gamma !== b.gamma || a.spread !== b.spread || a.standing !== b.standing) return true;
    if (a.choppiness !== b.choppiness || a.amplitude !== b.amplitude) return true;
    return false;
  }

  function smoothstep01(x: number): number {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    return x * x * (3 - 2 * x);
  }
  function blendAt(t: number): number {
    if (fromSlot === toSlot) return 1;
    return smoothstep01((t - blendStart) / blendDuration);
  }
  function blendRateAt(t: number): number {
    if (fromSlot === toSlot) return 0;
    const x = (t - blendStart) / blendDuration;
    if (x <= 0 || x >= 1) return 0;
    return (6 * x * (1 - x)) / blendDuration;
  }
  function choppinessAt(t: number): number {
    const w = blendAt(t);
    const a = slots[fromSlot].params.choppiness, b = slots[toSlot].params.choppiness;
    return a + (b - a) * w;
  }

  /**
   * Make the CPU frames hold the field at time t (positions; velocities too when withVel) and
   * select them for forward(). Two frames are cached (keyed by time + state stamp). Allocation-free.
   */
  function prepare(t: number, withVel: boolean): number {
    if (!initialised) init();
    if (K === 0) return 0;
    const s0 = kSim[0];
    if (s0.frameTime[curFrame] === t && s0.frameStamp[curFrame] === stamp && (!withVel || s0.frameHasVel[curFrame] === 1)) return curFrame;
    const other = 1 - curFrame;
    if (s0.frameTime[other] === t && s0.frameStamp[other] === stamp && (!withVel || s0.frameHasVel[other] === 1)) {
      selectFrame(other);
      return curFrame;
    }
    // Recompute into the current frame if it already holds t (adding velocities), else the other one.
    const f = s0.frameTime[curFrame] === t && s0.frameStamp[curFrame] === stamp ? curFrame : other;
    const w = blendAt(t), dw = blendRateAt(t);
    const chop = choppinessAt(t);
    const dchop = (slots[toSlot].params.choppiness - slots[fromSlot].params.choppiness) * dw;
    for (let k = 0; k < K; k++) {
      const sm = kSim[k];
      if (fromSlot === toSlot) sm.materialise(f, t, stamp, withVel, toSlot, 1, 0, -1, 0, 0, chop, 0);
      else sm.materialise(f, t, stamp, withVel, toSlot, w, dw, fromSlot, 1 - w, -dw, chop, dchop);
    }
    selectFrame(f);
    return f;
  }

  // ---- surface evaluation (local tangent-plane coordinates) --------------------------------
  // Scratch (module-closure, reused; never allocate per call)
  let outH = 0, outE = 0, outN = 0, outVH = 0, outVE = 0, outVN = 0;

  // Jacobian d(displaced east, north)/d(rest east, north) of the whole chain (Newton inversion):
  // J = (I + G_1)(I + G_0), G_c = gradient of cascade c's horizontal displacement at its
  // (chained) sample point, in local terms.
  let j00 = 1, j01 = 0, j10 = 0, j11 = 1;

  /**
   * Chained forward evaluation at rest point (re, rn) (reference spectralField), fast path:
   * bilinear cascades are inlined. Uses the frames selected by the last prepare().
   */
  function forward(f: number, re: number, rn: number, wantVel: boolean, wantGrad: boolean): void {
    let H = 0, E = 0, N = 0, VH = 0, VE = 0, VN = 0;
    if (wantGrad) { j00 = 1; j01 = 0; j10 = 0; j11 = 1; }
    // The Newton Jacobian only includes the first JACOBIAN_CASCADES cascades: the 9 m cascade moves
    // points by ~1-3 cm but has steep, quickly varying gradients that would only destabilise the
    // steps; its displacement is still part of the solved map (it converges as a fixed point).
    for (let k = 0; k < K; k++) {
      const c = kCascade[k];
      const data = kData[k];
      const off = kOff[k];
      if (kCubic[k] === 0) {
        const n = kN[k], inv = kInv[k], mask = n - 1;
        let sa = (re + E + off) * inv, sb = (off - rn - N) * inv; // texel coords of a = e + OFF, b = -n + OFF
        if (sa > 1e9 || sa < -1e9) sa %= n;
        if (sb > 1e9 || sb < -1e9) sb %= n;
        let x0 = Math.floor(sa), y0 = Math.floor(sb);
        const fx = sa - x0, fy = sb - y0;
        x0 &= mask; y0 &= mask;
        const x1 = (x0 + 1) & mask, y1 = (y0 + 1) & mask;
        const i00 = (y0 * n + x0) * 3, i10 = (y0 * n + x1) * 3, i01 = (y1 * n + x0) * 3, i11 = (y1 * n + x1) * 3;
        const gx = 1 - fx, gy = 1 - fy;
        const w00 = gx * gy, w10 = fx * gy, w01 = gx * fy, w11 = fx * fy;
        H += w00 * data[i00] + w10 * data[i10] + w01 * data[i01] + w11 * data[i11];
        const e00 = data[i00 + 1], e10 = data[i10 + 1], e01 = data[i01 + 1], e11 = data[i11 + 1];
        const n00 = data[i00 + 2], n10 = data[i10 + 2], n01 = data[i01 + 2], n11 = data[i11 + 2];
        if (wantGrad && c < JACOBIAN_CASCADES) {
          const dEa = (gy * (e10 - e00) + fy * (e11 - e01)) * inv, dEb = (gx * (e01 - e00) + fx * (e11 - e10)) * inv;
          const dNa = (gy * (n10 - n00) + fy * (n11 - n01)) * inv, dNb = (gx * (n01 - n00) + fx * (n11 - n10)) * inv;
          // local gradient: d/de = d/da, d/dn = -d/db
          const g00 = 1 + dEa, g01 = -dEb, g10 = dNa, g11 = 1 - dNb;
          const t00 = g00 * j00 + g01 * j10, t01 = g00 * j01 + g01 * j11;
          const t10 = g10 * j00 + g11 * j10, t11 = g10 * j01 + g11 * j11;
          j00 = t00; j01 = t01; j10 = t10; j11 = t11;
        }
        E += w00 * e00 + w10 * e10 + w01 * e01 + w11 * e11;
        N += w00 * n00 + w10 * n10 + w01 * n01 + w11 * n11;
        if (wantVel) {
          const v = kVel[k];
          VH += w00 * v[i00] + w10 * v[i10] + w01 * v[i01] + w11 * v[i11];
          VE += w00 * v[i00 + 1] + w10 * v[i10 + 1] + w01 * v[i01 + 1] + w11 * v[i11 + 1];
          VN += w00 * v[i00 + 2] + w10 * v[i10 + 2] + w01 * v[i01 + 2] + w11 * v[i11 + 2];
        }
        continue;
      }
      const n = kN[k], L = kL[k];
      const a = re + E + off, b = -(rn + N) + off; // tile coords: a = east + OFF, b = -north + OFF
      if (wantVel) {
        sampleCubic(kVel[k], n, L, a, b);
        VH += SAMPLE[0]; VE += SAMPLE[1]; VN += SAMPLE[2];
      }
      if (wantGrad && c < JACOBIAN_CASCADES) {
        sampleCubicGrad(data, n, L, a, b);
        // local gradient: d/de = d/da, d/dn = -d/db
        const g00 = 1 + GRAD[0], g01 = -GRAD[1], g10 = GRAD[2], g11 = 1 - GRAD[3];
        const t00 = g00 * j00 + g01 * j10, t01 = g00 * j01 + g01 * j11;
        const t10 = g10 * j00 + g11 * j10, t11 = g10 * j01 + g11 * j11;
        j00 = t00; j01 = t01; j10 = t10; j11 = t11;
      } else sampleCubic(data, n, L, a, b);
      H += SAMPLE[0]; E += SAMPLE[1]; N += SAMPLE[2];
    }
    outH = H; outE = re + E; outN = rn + N;
    outVH = VH; outVE = VE; outVN = VN;
  }

  /**
   * Find the rest point r displaced onto local (pe, pn), i.e. solve r + D(r) = p (D = chained
   * horizontal displacement), with Newton steps on the exact Jacobian of the sampled (bilinear /
   * Catmull-Rom) chain. Where the surface is close to folding (det J < 0.1) a plain fixed-point step
   * is taken instead. Stops when |residual| < 0.1 mm or after inversionIterations steps.
   * Where crests fold over (several rest points map to p) any of the sheets may be returned.
   * Leaves the forward() results for the found rest point in out*.
   */
  let lastRestE = 0, lastRestN = 0;
  function invert(f: number, pe: number, pn: number, wantVel: boolean): void {
    let re = pe, rn = pn;
    if (K > 0 && kCubic[0] === 0) {
      // Initial guess: undo the largest cascade's displacement at p (cheap single bilinear fetch).
      const data = kData[0], n = kN[0], inv = kInv[0], off = kOff[0], mask = n - 1;
      let sa = (pe + off) * inv, sb = (off - pn) * inv;
      if (sa > 1e9 || sa < -1e9) sa %= n;
      if (sb > 1e9 || sb < -1e9) sb %= n;
      let x0 = Math.floor(sa), y0 = Math.floor(sb);
      const fx = sa - x0, fy = sb - y0;
      x0 &= mask; y0 &= mask;
      const x1 = (x0 + 1) & mask, y1 = (y0 + 1) & mask;
      const i00 = (y0 * n + x0) * 3, i10 = (y0 * n + x1) * 3, i01 = (y1 * n + x0) * 3, i11 = (y1 * n + x1) * 3;
      const gx = 1 - fx, gy = 1 - fy;
      re -= gx * gy * data[i00 + 1] + fx * gy * data[i10 + 1] + gx * fy * data[i01 + 1] + fx * fy * data[i11 + 1];
      rn -= gx * gy * data[i00 + 2] + fx * gy * data[i10 + 2] + gx * fy * data[i01 + 2] + fx * fy * data[i11 + 2];
    }
    let pre = re, prn = rn, prevRes = Infinity; // last accepted iterate (backtracking safeguard)
    let bestE = pe, bestN = pn, bestRes = Infinity;
    const iters = sim.inversionIterations;
    const tol2 = sim.inversionTolerance * sim.inversionTolerance;
    for (let it = 0; it < iters; it++) {
      forward(f, re, rn, false, true);
      const ee = outE - pe, en = outN - pn;
      const res = ee * ee + en * en;
      if (res < tol2) { // converged
        if (wantVel) forward(f, re, rn, true, false);
        lastRestE = re; lastRestN = rn;
        return;
      }
      if (res < bestRes) { bestRes = res; bestE = re; bestN = rn; }
      if (res >= prevRes) { // step made things worse: halve it
        re = pre + 0.5 * (re - pre);
        rn = prn + 0.5 * (rn - prn);
        continue;
      }
      pre = re; prn = rn; prevRes = res;
      const det = j00 * j11 - j01 * j10;
      if (det > 0.1) {
        re -= (j11 * ee - j01 * en) / det;
        rn -= (j00 * en - j10 * ee) / det;
      } else {
        re -= ee; rn -= en;
      }
    }
    re = bestE; rn = bestN;
    forward(f, re, rn, wantVel, false);
    lastRestE = re; lastRestN = rn;
  }

  let localE = 0, localN = 0;
  function toLocal(p: Readonly<Vec3>): void {
    (shape as WorldShape).tangentFrame(p, frame);
    const dx = p.x - anchor.x, dy = p.y - anchor.y, dz = p.z - anchor.z;
    localE = dx * frame.east.x + dy * frame.east.y + dz * frame.east.z;
    localN = dx * frame.north.x + dy * frame.north.y + dz * frame.north.z;
  }

  const query: WaterQuery = {
    heightAt(p: Readonly<Vec3>, time: number): number {
      const f = prepare(time, false);
      toLocal(p);
      invert(f, localE, localN, false);
      return outH;
    },
    velocityAt(p: Readonly<Vec3>, time: number, out: Vec3): Vec3 {
      const f = prepare(time, true);
      toLocal(p); // also fills `frame`
      invert(f, localE, localN, true);
      const e = frame.east, n = frame.north, u = frame.up;
      out.x = outVE * e.x + outVN * n.x + outVH * u.x;
      out.y = outVE * e.y + outVN * n.y + outVH * u.y;
      out.z = outVE * e.z + outVN * n.z + outVH * u.z;
      return out;
    },
  };

  // ---- wind --------------------------------------------------------------------------------
  function sampleWind(dt: number, snap: boolean): void {
    if (!wind || !useWind || !shape) return;
    shape.atAltitude(focus, sim.windSampleHeight, windPoint);
    wind.at(windPoint, windVec);
    shape.tangentFrame(windPoint, frame);
    const we = windVec.x * frame.east.x + windVec.y * frame.east.y + windVec.z * frame.east.z;
    const wn = windVec.x * frame.north.x + windVec.y * frame.north.y + windVec.z * frame.north.z;
    if (snap || !windInit) { windE = we; windN = wn; windInit = true; }
    else {
      const a = sim.windTimeConstant > 0 ? 1 - Math.exp(-dt / sim.windTimeConstant) : 1;
      windE += (we - windE) * a;
      windN += (wn - windN) * a;
    }
    const speed = Math.hypot(windE, windN);
    target.windSpeed = speed;
    if (speed > 1e-3) target.windDirection = Math.atan2(windN, windE);
  }

  function rebuildTarget(): void {
    const ws = target.windSpeed, wd = target.windDirection;
    copyParams(base, target);
    if (wind && useWind && windInit) { target.windSpeed = ws; target.windDirection = wd; }
  }

  function applyImmediate(): void {
    if (!initialised) { init(); return; }
    regenSlot = -1;
    pendingSlot = -1;
    const slot = (toSlot + 1) % NUM_SLOTS;
    generateNow(slot, target);
    fromSlot = toSlot = slot;
    blendVersion++;
    stamp++;
  }

  // ---- state for the GPU adapter -------------------------------------------------------------
  const slotViews: OceanSlot[] = slots as unknown as OceanSlot[];
  const state = {
    masterSize: REFERENCE_SIZE,
    cascades: CASCADES,
    loopPeriod: LOOP_PERIOD,
    omegaUnit: OMEGA_UNIT,
    gravity: g,
    omegaIndex: omegaIndex as readonly Float32Array[],
    slots: slotViews as readonly OceanSlot[],
    version: 0, blendVersion: 0, fromSlot: 0, toSlot: 0, pendingSlot: -1,
    blendStart: 0, blendDuration: sim.blendTime, blend: 1, choppiness: base.choppiness, time: 0,
    windSpeed: target.windSpeed, windDirection: target.windDirection,
    focus: focus as Readonly<Vec3>, anchor: anchor as Readonly<Vec3>,
  };
  function syncState(): void {
    state.gravity = g;
    state.version = version; state.blendVersion = blendVersion;
    state.fromSlot = fromSlot; state.toSlot = toSlot; state.pendingSlot = pendingSlot;
    state.blendStart = blendStart; state.blendDuration = blendDuration;
    state.blend = blendAt(lastTickTime); state.choppiness = choppinessAt(lastTickTime);
    state.time = lastTickTime;
    state.windSpeed = target.windSpeed; state.windDirection = target.windDirection;
  }

  // ---- the system ----------------------------------------------------------------------------
  const ocean: Ocean = {
    id: MODULE_ID,
    provides: [WATER_QUERY],
    optional: [WIND_QUERY, ENVIRONMENT, TERRAIN_QUERY, TERRAIN_GPU, SKY_LIGHT],
    phases: ['simulate'] as readonly TickPhase[],
    query,
    get params() { return target; },
    get sim() { return sim; },

    install(world: World): void {
      if (!opts.shape) shape = world.shape;
      world.provide(WATER_QUERY, query);
    },

    start(world: World): void {
      wind = world.optional(WIND_QUERY);
      if (wind && useWind && shape) {
        sampleWind(0, true);
        if (initialised) { if (differs(target, slots[toSlot].params)) applyImmediate(); }
        else init();
      } else if (!initialised) init();
      syncState();
    },

    tick(phase: TickPhase, info: TickInfo): void {
      if (phase !== 'simulate') return;
      if (perf) perf.begin(MODULE_ID, 'sim');
      if (!initialised) init();
      const T = info.time;
      lastTickTime = T;
      sampleWind(info.dt, false);
      // Crossfade end.
      if (fromSlot !== toSlot && T >= blendStart + blendDuration) {
        fromSlot = toSlot; blendVersion++; stamp++;
      }
      // Regeneration.
      if (regenSlot >= 0) stepRegen(sim.regenRowsPerTick);
      else if (pendingSlot < 0 && fromSlot === toSlot && differs(target, slots[toSlot].params)) {
        startRegen();
        stepRegen(sim.regenRowsPerTick);
      }
      // Start the crossfade once the CPU has the pending slot's snapshots for this interval.
      if (pendingSlot >= 0 && fromSlot === toSlot) {
        let ready = true;
        for (let c = 0; c < sims.length; c++) {
          const sm = sims[c];
          if (!sm) continue;
          const j0 = Math.floor(T / sm.snapshotDt);
          if (!sm.has(pendingSlot, j0) || !sm.has(pendingSlot, j0 + 1)) { ready = false; break; }
        }
        if (ready) {
          fromSlot = toSlot; toSlot = pendingSlot; pendingSlot = -1;
          blendStart = T; blendDuration = sim.blendTime;
          blendVersion++; stamp++;
        }
      }
      // CPU snapshots (amortised) + the field at T for this step's queries.
      for (let c = 0; c < sims.length; c++) {
        const sm = sims[c];
        if (!sm) continue;
        sm.schedule(T, info.dt, toSlot, fromSlot !== toSlot ? fromSlot : -1, pendingSlot);
      }
      prepare(T, false);
      syncState();
      if (perf) perf.end();
    },

    setFocus(p: Readonly<Vec3>): void { focus.x = p.x; focus.y = p.y; focus.z = p.z; },

    setPreset(name: PresetName, immediate = false): void {
      const pr = PRESETS[name];
      if (!pr) throw new Error(`Unknown ocean preset '${name}'`);
      copyParams(pr.params, base);
      rebuildTarget();
      if (immediate) applyImmediate();
      syncState();
    },

    setParams(p: Partial<SpectrumParams & SimParams>, immediate = false): void {
      for (const k of SPECTRUM_KEYS) if (typeof p[k] === 'number') base[k] = p[k] as number;
      for (const k of SIM_KEYS) if (typeof p[k] === 'number') sim[k] = p[k] as number;
      rebuildTarget();
      if (immediate) applyImmediate();
      syncState();
    },

    getState(): Readonly<OceanState> {
      if (!initialised && shape) { init(); syncState(); }
      return state;
    },
    blendAt,
    choppinessAt,
    expectedHs(): number { return expectedHs(target, g); },

    heightAtLocal(east: number, north: number, time: number): number {
      const f = prepare(time, false);
      invert(f, east, north, false);
      return outH;
    },

    invertLocal(east: number, north: number, time: number, out: RestSample): number {
      const f = prepare(time, true);
      invert(f, east, north, true);
      const residual = Math.hypot(outE - east, outN - north);
      out.east = lastRestE; out.north = lastRestN; out.up = outH;
      out.vEast = outVE; out.vNorth = outVN; out.vUp = outVH;
      return residual;
    },

    sampleRest(east: number, north: number, time: number, out: RestSample): RestSample {
      const f = prepare(time, true);
      forward(f, east, north, true, false);
      out.east = outE; out.north = outN; out.up = outH;
      out.vEast = outVE; out.vNorth = outVN; out.vUp = outVH;
      return out;
    },

    snapshotCounts(out: number[]): number[] {
      for (let c = 0; c < sims.length; c++) out[c] = sims[c] ? (sims[c] as CascadeSim).computed : 0;
      return out;
    },

    dispose(): void {
      sims.length = 0;
      initialised = false;
      wind = undefined;
    },
  };
  return ocean;
}
