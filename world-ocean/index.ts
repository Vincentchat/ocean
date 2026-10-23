// @dreamp/world-ocean - public entry. Render-agnostic core: NEVER import three here.
//
//   const ocean = createOcean({ preset: 'windy', seed: 1 });    // WorldSystem, provides WATER_QUERY
//   world.use(ocean); world.start();
//   const h = world.require(WATER_QUERY).heightAt({ x: 10, y: 0, z: 5 }, world.time);
//
// CPU / GPU agreement path (spec section 1 rule 8): the CPU runs the SAME spectrum + FFT as the GPU
// (cpu-fft.ts CascadeSim), following the conventions documented at the top of spectrum.ts.
// Default: cascades 0 and 1 (>99.9 % of the height variance) are evaluated at 256^2 with bilinear
// sampling, i.e. the same grid + filter the GPU uses at 256; cascade 2 at 128^2 (an exact spectral
// subset). Time: cubic Hermite between snapshots (exact analytic velocities). heightAt inverts the
// chained horizontal displacement with Newton steps, so it returns the height of the surface point
// that ends up AT p. velocityAt is the material derivative of that surface point (analytic dD/dt of
// every cascade + advection of the chained sample points).
//
// Wind coupling + crossfade (no pops): the WIND_QUERY sample (10 m above sea level at the focus
// point) is low-passed (EMA, windTimeConstant). When the smoothed wind drifts beyond tolerance,
// a new spectrum is generated into a spare slot, amortised over ticks (regenRowsPerTick), and the
// drawn h0 crossfades from the old slot to the new one:  h0 = lerp(h0_from, h0_to, w(t)),
// w(t) = smoothstep(clamp((t - blendStart) / blendDuration, 0, 1)); choppiness is lerped the same
// way. Both slots share the same random draws (same seed), so only amplitudes change: no phase
// cancellation, heights stay continuous (C1 in time). Wind speed and direction are smoothed
// separately (a reversal never passes through 0 m/s) and the spectrum turns by at most
// maxTurnPerBlendDeg per crossfade (no mid-blend energy loss: a 180 deg turn keeps Hs >= 95 %).
//
// Threads (option worker, default true when SharedArrayBuffer + workers exist): one worker per cascade
// (ocean-worker.ts) computes the snapshot FFTs, prefills the next tick's position frames and
// regenerates the spectrum; the main thread only schedules, adopts and answers queries. Same code,
// bit-identical results with or without workers (tested).
//
// Query semantics and costs (60 Hz budget):
//  - Exact within [lastTick - timeWindow, lastTick] (default 0.1 s: tick time and the interpolated
//    render time, which lies between the last two ticks). Other times are CLAMPED to that window (bounded cost; an exact
//    far time needs ~6 fresh snapshots, 50-70 ms). timeWindow = Infinity: exact anywhere, at that
//    cost. Before the first tick every time is exact. Non-finite times throw RangeError.
//  - Heights reflect the CURRENT spectrum sources: a time before the last crossfade start is
//    evaluated with today's spectra (history is not kept).
//  - Where crests fold over (rare, storm), several surface sheets cover p; any of them is returned.
//  - Allocation: tick, velocityAt and heightsAt allocate nothing. heightAt returns a JS number,
//    which V8 boxes (16 B) when the call is not inlined into the caller: use heightsAt on hot paths.
//  - Not supported yet: a curved (sphere) world. Local coordinates are (p - anchor) projected on the
//    tangent frame at p, which is valid for flat worlds only.
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
import { CascadeSim, SAMPLE, SAMPLE_IN, CUBIC_IN, GRAD, cubicTaps, sampleCubic, sampleCubicGrad } from './cpu-fft.ts';
import { PRESETS } from './presets/index.ts';

export * from './types.ts';
export * as spectrum from './spectrum.ts';
export { PRESETS, PRESET_NAMES, REFERENCE_SCENE_AMPLITUDE } from './presets/index.ts';
export { CASCADES, REFERENCE_SIZE, LOOP_PERIOD, OMEGA_UNIT } from './spectrum.ts';

export const MODULE_ID = 'world/ocean';
const NUM_SLOTS = 3;
const SNAPSHOT_POOL = 10;
function gcd(a: number, b: number): number { while (b) { const t = a % b; a = b; b = t; } return a; }
/**
 * The whole surface repeats every SURFACE_PERIOD metres along east and north: lcm of the tile sizes
 * (1024, 96, 9 -> 9216 m). Query coordinates are wrapped to it, so tile lookups stay in small numbers
 * at any distance from the anchor (float64 world coordinates of any size are fine).
 */
export const SURFACE_PERIOD = CASCADES.reduce((p, c) => (Number.isInteger(c.domain) ? (p * c.domain) / gcd(p, c.domain) : p), 1);
function wrapPeriod(x: number): number { return x - SURFACE_PERIOD * Math.floor(x / SURFACE_PERIOD); }

/** Cascades whose gradient enters the Newton Jacobian of the heightAt inversion (see forward()). */
const JACOBIAN_CASCADES = 2;
/** Fixed-point steps on the largest cascade alone that form the Newton initial guess. */
const GUESS_STEPS = 2;
/** Extra damped fixed-point steps when Newton did not converge (rare). */
const RESCUE_ITERATIONS = 12;
/** Newton steps on the exact Catmull-Rom surface after the bilinear solve (each ~cm -> sub-mm). */
const CUBIC_REFINE_ITERATIONS = 3;
/** Cost of packing one cascade's h0 in units of generated master rows (amortised regeneration budget). */
const PACK_ROWS_EQUIVALENT = 8;

// ---- snapshot workers --------------------------------------------------------------------------
type Proc = { versions?: { node?: string }; getBuiltinModule?: (id: string) => unknown };
function nodeProcess(): Proc | undefined {
  const p = (globalThis as { process?: Proc }).process;
  return p?.versions?.node && p.getBuiltinModule ? p : undefined;
}
/** Workers need SharedArrayBuffer (cross-origin isolated pages in browsers) and a Worker API. */
function canUseWorkers(): boolean {
  if (typeof SharedArrayBuffer === 'undefined' || typeof Atomics === 'undefined') return false;
  if (nodeProcess()) return true;
  const g = globalThis as { Worker?: unknown; crossOriginIsolated?: boolean };
  return typeof g.Worker === 'function' && g.crossOriginIsolated === true;
}
/** Start ocean-worker.ts for one cascade (null if it cannot start: the main thread then does the work). */
function spawnSnapshotWorker(data: unknown): { terminate(): unknown } | null {
  try {
    const url = new URL('./ocean-worker.ts', import.meta.url);
    const proc = nodeProcess();
    if (proc) {
      const wt = proc.getBuiltinModule!('node:worker_threads') as { Worker: new (u: URL, o: object) => { unref(): void; terminate(): unknown } };
      const w = new wt.Worker(url, { workerData: data });
      w.unref(); // never keeps a Node process alive
      return w;
    }
    const W = (globalThis as unknown as { Worker: new (u: URL, o: object) => { postMessage(d: unknown): void; terminate(): unknown } }).Worker;
    const w = new W(url, { type: 'module' });
    w.postMessage(data);
    return w;
  } catch {
    return null;
  }
}

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
  /** Random seed of the spectrum (spectrum.gaussianField(seed, cascade)); lets the GPU adapter add extra cascades with the same seeding. */
  readonly seed: number;
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
   * Diagnostics / tests (exact at any time, no time window: may compute snapshots synchronously):
   * invert the displacement at local (east, north). Writes the REST point to
   * out.east / out.north, the height and velocity to out.up / out.v*; returns the residual |P(r) - p| (m).
   */
  invertLocal(east: number, north: number, time: number, out: RestSample): number;
  /**
   * Allocation-free batch heightAt: points = [x0, y0, z0, x1, ...] world metres, out[i] = height of
   * point i (m above sea level) at 	ime. Same results as heightAt; use it on hot paths (heightAt
   * returns a JS number, which V8 boxes (16 B) when the call is not inlined into the caller).
   */
  heightsAt(points: Float64Array, time: number, out: Float64Array, count?: number): Float64Array;
  /** Diagnostics (exact at any time, like invertLocal): surface point resting at local (east, north). */
  sampleRest(east: number, north: number, time: number, out: RestSample): RestSample;
  /**
   * Diagnostics per cascade c: out[5c] snapshots computed, out[5c+1] of them synchronously (prefetch
   * missed or cold time), out[5c+2] of them by the worker, out[5c+3] / out[5c+4] lazy position /
   * velocity blocks filled.
   */
  cpuStats(out: number[]): number[];
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
const SIM_KEYS: readonly (keyof SimParams)[] = ['blendTime', 'windTimeConstant', 'windSpeedTolerance', 'windDirectionToleranceDeg', 'regenRowsPerTick', 'inversionIterations', 'inversionTolerance', 'windSampleHeight', 'timeWindow', 'maxTurnPerBlendDeg'];

function failTickTime(): never { throw new RangeError('world/ocean: tick time must be finite'); }

function wrapAngle(a: number): number {
  a = (a + Math.PI) % (2 * Math.PI);
  if (a < 0) a += 2 * Math.PI;
  return a - Math.PI;
}

export function createOcean(opts: OceanOptions = {}): Ocean {
  const presetName: PresetName = opts.preset ?? 'windy';
  if (!PRESETS[presetName]) throw new Error(`Unknown ocean preset '${presetName}'`);
  const seed = opts.seed ?? 1;
  const useWind = opts.useWind ?? true;
  const useWorker = (opts.worker ?? true) && canUseWorkers();
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
  let windInit = false;
  /** True once tick() ran: queries are then limited to the time window around the last tick. */
  let ticked = false;
  let disposed = false;
  const workers: { terminate(): unknown }[] = [];
  let workersStarted = false;
  /** Start the snapshot workers on the first tick (standalone, never-ticked oceans do not need them). */
  function startWorkers(): void {
    workersStarted = true;
    if (!useWorker) return;
    for (let c = 0; c < sims.length; c++) {
      const sm = sims[c];
      if (!sm || !sm.shared) continue;
      const q = queryCfg[c];
      const masterH0: SharedArrayBuffer[] = [];
      for (let s = 0; s < NUM_SLOTS; s++) masterH0.push(slots[s].h0[c].buffer as SharedArrayBuffer);
      const w = spawnSnapshotWorker({ cascade: c, cfg: { size: q.size, snapshotDt: q.snapshotDt, filter: q.filter }, g, seed, numSlots: NUM_SLOTS, poolSize: SNAPSHOT_POOL, shared: sm.shared, masterH0 });
      if (w) workers.push(w);
    }
  }

  // ---- spectrum slots -----------------------------------------------------------------------
  const nm = REFERENCE_SIZE;
  const xi: Float64Array[] = [];
  const omegaIndex: Float32Array[] = [];
  const slots: SlotData[] = [];
  for (let s = 0; s < NUM_SLOTS; s++) {
    const h0: Float32Array[] = [];
    // Shared when workers may run (they write regenerated spectra straight into it).
    for (let c = 0; c < CASCADES.length; c++) h0.push(useWorker ? new Float32Array(new SharedArrayBuffer(nm * nm * 4 * 4)) : new Float32Array(nm * nm * 4));
    slots.push({ params: emptyParams(), version: 0, h0 });
  }
  let version = 0, blendVersion = 0;
  // Crossfade sources (cpu-fft.ts CascadeSim): the ACTIVE source is what is drawn,
  // h0 = lerp(h0[fromSlot], h0[toSlot], w(t)); the PENDING source is the next crossfade, defined when
  // a regeneration finishes with a start time pendingLead in the future, so the CPU can compute its
  // snapshots in the background and switch to it before its crossfade begins (no stall, no pop).
  let srcCounter = 0;
  let activeId = 0, fromSlot = 0, toSlot = 0;
  let pendingId = -1, pendingFrom = 0, pendingSlot = -1;
  // Per-tick doubles live in a typed array (closure-captured mutable doubles would be boxed = allocated per write).
  const TS = new Float64Array(9);
  const T_WIND_SPEED = 0, T_WIND_DIR = 1, T_BLEND_START = 2, T_BLEND_DURATION = 3, T_LAST_TICK = 4, T_DT = 5;
  const T_PEND_START = 6, T_PEND_DURATION = 7, T_PEND_SWITCH = 8;
  TS[T_BLEND_DURATION] = 1;

  let stamp = 0; // bumps when anything that changes materialised frames changes
  // Regeneration job
  const regenRaw = new Float64Array(2 * nm * nm);
  const regenScalars: SpectrumScalars = { g: 0, U: 0, dirGrid: 0, wp: 0 };
  let regenSlot = -1, regenCascade = 0, regenRow = 0, regenRemote = false, regenTicks = 0, regenDue = 1;
  /** Regeneration params handed to the workers (wind, dir, fetch, gamma, spread, standing, chop, amp). */
  const RP = new Float64Array(8);

  // ---- CPU query ---------------------------------------------------------------------------
  const sims: (CascadeSim | null)[] = [];
  let curFrame = 0;
  let anyCubic = false;
  /** Lead time between the end of a regeneration and the start of its crossfade, s (CPU warm-up). */
  let pendingLead = 1.6;
  /** The CPU switches to the pending source this long before its crossfade starts, s. */
  let switchMargin = 0.4;
  // Compact list of the cascades the CPU evaluates (hot path of the queries).
  let K = 0;
  const kSim: CascadeSim[] = [], kCascade: number[] = [], kData: Float32Array[] = [], kVel: Float32Array[] = [];
  const kBlock: Uint8Array[] = [], kVelBlock: Uint8Array[] = [], kBSh: number[] = [], kNb: number[] = [];
  const kN: number[] = [], kShift: number[] = [], kInv: number[] = [], kL: number[] = [], kOff: number[] = [], kCubic: number[] = [];
  function selectFrame(f: number): void {
    curFrame = f;
    for (let k = 0; k < K; k++) {
      const sm = kSim[k];
      kData[k] = sm.frames[f]; kVel[k] = sm.velFrames[f]; kBlock[k] = sm.blockValid[f]; kVelBlock[k] = sm.velValid[f];
    }
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
        const sm = new CascadeSim(CASCADES[c], q, omegaIndex[c], NUM_SLOTS, SNAPSHOT_POOL, useWorker ? true : undefined);
        sims.push(sm);
      } else sims.push(null);
    }
    for (let c = 0; c < sims.length; c++) {
      const sm = sims[c];
      if (!sm) continue;
      kSim.push(sm); kCascade.push(c); kData.push(sm.frames[0]); kVel.push(sm.velFrames[0]);
      kBlock.push(sm.blockValid[0]); kVelBlock.push(sm.velValid[0]); kBSh.push(sm.blockShift); kNb.push(sm.nb);
      kN.push(sm.n); kInv.push(sm.n / sm.domain);
      kShift.push(sm.n * (Math.ceil(SURFACE_PERIOD / sm.domain) + 4)); kL.push(sm.domain); kOff.push(sm.offset);
      kCubic.push(sm.cubic ? 1 : 0);
    }
    K = kSim.length;
    for (let k = 0; k < K; k++) if (kCubic[k] === 1) anyCubic = true;
    for (let k = 0; k < K; k++) {
      pendingLead = Math.max(pendingLead, 4 * kSim[k].snapshotDt);
      switchMargin = Math.max(switchMargin, kSim[k].snapshotDt);
    }
    initialised = true;
    generateNow(0, target);
    setActive(++srcCounter, 0, 0, 0, 1);
  }

  /** Make (from -> to, start, duration) the active source on every cascade (and clear the pending one). */
  function setActive(id: number, from: number, to: number, start: number, duration: number): void {
    activeId = id; fromSlot = from; toSlot = to;
    TS[T_BLEND_START] = start; TS[T_BLEND_DURATION] = duration;
    pendingId = -1; pendingSlot = -1;
    for (let k = 0; k < K; k++) {
      const sm = kSim[k];
      sm.args[0] = start; sm.args[1] = duration;
      sm.setSource(0, id, from, to); sm.setSource(1, -1, 0, 0);
    }
    blendVersion++;
    stamp++;
  }

  /** The slot neither the active source nor the pending one uses (where the next spectrum goes). */
  function spareSlot(): number {
    for (let s = 0; s < NUM_SLOTS; s++) if (s !== fromSlot && s !== toSlot && s !== pendingSlot) return s;
    return (toSlot + 1) % NUM_SLOTS;
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
    for (let c = 0; c < sims.length; c++) sims[c]?.setSlot(slot, sd.h0[c]);
  }

  function startRegen(): void {
    regenSlot = spareSlot();
    const p = slots[regenSlot].params;
    copyParams(target, p);
    // Turn the sea by at most maxTurnPerBlendDeg per crossfade: the crossfade lerps complex
    // amplitudes, and two spectra far apart in direction would partly cancel mid-blend (Hs dip).
    const cur = slots[toSlot].params.windDirection;
    let dd = p.windDirection - cur;
    while (dd > Math.PI) dd -= 2 * Math.PI;
    while (dd < -Math.PI) dd += 2 * Math.PI;
    const maxTurn = (sim.maxTurnPerBlendDeg * Math.PI) / 180;
    if (dd > maxTurn) p.windDirection = cur + maxTurn; else if (dd < -maxTurn) p.windDirection = cur - maxTurn;
    spectrumScalars(slots[regenSlot].params, g, regenScalars);
    regenCascade = 0;
    regenRow = 0;
    regenTicks = 0;
    const rpt = sim.regenRowsPerTick >= 1 ? sim.regenRowsPerTick : 1;
    regenDue = Math.ceil((CASCADES.length * (nm + PACK_ROWS_EQUIVALENT)) / rpt) + 1;
    // Offload to the cascade workers when every cascade has a ready one (each regenerates its own
    // cascade: master h0 for the GPU + its slot grid); the main thread then only polls.
    regenRemote = false;
    if (workers.length > 0 && K === CASCADES.length) {
      let ready = true;
      for (let k = 0; k < K; k++) if (!kSim[k].remoteReady()) ready = false;
      if (ready) {
        RP[0] = p.windSpeed; RP[1] = p.windDirection; RP[2] = p.fetch; RP[3] = p.gamma;
        RP[4] = p.spread; RP[5] = p.standing; RP[6] = p.choppiness; RP[7] = p.amplitude;
        let ok = true;
        for (let k = 0; k < K; k++) if (!kSim[k].requestRegen(regenSlot, RP)) ok = false;
        regenRemote = ok; // (if a worker refused, the main thread regenerates: same numbers)
      }
    }
  }

  /** Regenerated spectrum is in slot regenSlot (all cascades): publish it and define the pending source. */
  function finishRegen(): void {
    const sd = slots[regenSlot];
    sd.version++;
    version++;
    // Pending source: crossfade from the current spectrum to the new one, starting later.
    pendingId = ++srcCounter; pendingFrom = toSlot; pendingSlot = regenSlot;
    TS[T_PEND_START] = TS[T_LAST_TICK] + pendingLead; TS[T_PEND_DURATION] = sim.blendTime;
    TS[T_PEND_SWITCH] = TS[T_PEND_START] - switchMargin;
    for (let k = 0; k < K; k++) {
      const sm = kSim[k];
      sm.args[0] = TS[T_PEND_START]; sm.args[1] = TS[T_PEND_DURATION];
      sm.setSource(1, pendingId, pendingFrom, pendingSlot);
    }
    regenSlot = -1;
    regenRemote = false;
  }

  /** Advance the amortised regeneration by `rows` master rows. */
  // The new spectrum is published on a DETERMINISTIC tick, regenDue ticks after the start, whether
  // the main thread or the workers did the work: the crossfade schedule must not depend on thread
  // timing (same inputs, same waves).
  function stepRegen(rows: number): void {
    if (regenSlot < 0) return;
    regenTicks++;
    if (regenRemote) {
      if (regenTicks < regenDue) return;
      // Due: wait for the workers if they are late (they have ~3 s for ~0.1 s of work).
      for (let k = 0; k < K; k++) { while (!kSim[k].regenDone()) { /* spin */ } }
      for (let k = 0; k < K; k++) { kSim[k].ackRegen(); kSim[k].slotChanged(regenSlot); }
      finishRegen();
      return;
    }
    const sd = slots[regenSlot];
    while (rows > 0 && regenCascade < CASCADES.length) {
      if (regenRow < nm) {
        const r1 = Math.min(nm, regenRow + rows);
        generateRawRows(sd.params, regenScalars, CASCADES[regenCascade], xi[regenCascade], regenRaw, regenRow, r1);
        rows -= r1 - regenRow;
        regenRow = r1;
      } else {
        packInitialSpectrum(regenRaw, sd.params.standing, sd.h0[regenCascade]);
        rows -= PACK_ROWS_EQUIVALENT;
        regenCascade++;
        regenRow = 0;
        if (regenCascade >= CASCADES.length) {
          for (let c = 0; c < sims.length; c++) sims[c]?.setSlot(regenSlot, sd.h0[c]);
        }
      }
    }
    if (regenCascade >= CASCADES.length && regenTicks >= regenDue) finishRegen();
  }

  /** Does the target differ enough from slot params b to regenerate? (no Math.* calls returning boxed doubles) */
  function differs(a: Readonly<SpectrumParams>, b: Readonly<SpectrumParams>): boolean {
    let du = a.windSpeed - b.windSpeed;
    if (du < 0) du = -du;
    const ref = b.windSpeed > 1 ? b.windSpeed : 1;
    if (du > sim.windSpeedTolerance * ref) return true;
    if (a.windSpeed > 0.5 && b.windSpeed > 0.5) {
      let dd = a.windDirection - b.windDirection;
      while (dd > Math.PI) dd -= 2 * Math.PI;
      while (dd < -Math.PI) dd += 2 * Math.PI;
      if (dd < 0) dd = -dd;
      if (dd > (sim.windDirectionToleranceDeg * Math.PI) / 180) return true;
    }
    let df = a.fetch - b.fetch;
    if (df < 0) df = -df;
    if (df > 1e-6 * (1 + b.fetch)) return true;
    if (a.gamma !== b.gamma || a.spread !== b.spread || a.standing !== b.standing) return true;
    if (a.choppiness !== b.choppiness || a.amplitude !== b.amplitude) return true;
    return false;
  }

  /**
   * Crossfade state at time ST[S_T]: ST[S_W] = w(t) = smoothstep((t - blendStart) / blendDuration),
   * ST[S_DW] = dw/dt, ST[S_CHOP] = lerp(chop_from, chop_to, w), ST[S_DCHOP] = d chop / dt.
   * (Inputs / outputs through the typed array: no boxed doubles.)
   */
  function blendInto(): void {
    const t = ST[S_T];
    const a = slots[fromSlot].params.choppiness, b = slots[toSlot].params.choppiness;
    if (fromSlot === toSlot) { ST[S_W] = 1; ST[S_DW] = 0; ST[S_CHOP] = b; ST[S_DCHOP] = 0; return; }
    const x = (t - TS[T_BLEND_START]) / TS[T_BLEND_DURATION];
    const w = x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x); // smoothstep inlined (a call returning a double may box)
    const dw = x <= 0 || x >= 1 ? 0 : (6 * x * (1 - x)) / TS[T_BLEND_DURATION];
    ST[S_W] = w; ST[S_DW] = dw; ST[S_CHOP] = a + (b - a) * w; ST[S_DCHOP] = (b - a) * dw;
  }
  function blendAt(t: number): number { ST[S_T] = t; blendInto(); return ST[S_W]; }
  function choppinessAt(t: number): number { ST[S_T] = t; blendInto(); return ST[S_CHOP]; }

  /**
   * Make the (lazy) CPU frames describe the field at query time ST[S_T] and select them for
   * forward(). Two frames are cached (keyed by time + state stamp). The query time is first
   * validated and limited to the query window (see clampQueryTime). Allocation-free.
   */
  function prepare(exact = false): number {
    if (disposed) throw new Error('world/ocean: disposed');
    if (!initialised) init();
    if (exact) { const t = ST[S_T]; if (!(t - t === 0)) throw new RangeError('world/ocean: query time must be a finite number of seconds'); }
    else clampQueryTime();
    const t = ST[S_T];
    if (K === 0) return 0;
    const s0 = kSim[0];
    if (s0.frameTime[curFrame] === t && s0.frameStamp[curFrame] === stamp) return curFrame;
    const other = 1 - curFrame;
    if (s0.frameTime[other] === t && s0.frameStamp[other] === stamp) {
      selectFrame(other);
      return curFrame;
    }
    blendInto();
    const chop = ST[S_CHOP], dchop = ST[S_DCHOP];
    for (let k = 0; k < K; k++) {
      const sm = kSim[k];
      sm.args[0] = t; sm.args[1] = chop; sm.args[2] = dchop;
      sm.materialise(other, stamp, activeId);
    }
    selectFrame(other);
    return other;
  }

  /**
   * Query time policy. Non-finite times throw. Once the system ticks, queries are exact within
   * [lastTick - timeWindow, lastTick] (the cached snapshots: no stall; covers the interpolated render
   * time, which lies between the last two ticks); times outside
   * are clamped to that window, which keeps every query bounded in cost (an arbitrary time would
   * otherwise need six fresh 256^2 snapshots, ~50-70 ms). timeWindow = Infinity disables the clamp
   * (exact at any time, synchronous snapshot cost). Before the first tick (standalone use) every
   * time is exact. Deterministic: depends on the time and the last tick time only.
   */
  function clampQueryTime(): void {
    const t = ST[S_T];
    if (!(t - t === 0)) throw new RangeError('world/ocean: query time must be a finite number of seconds');
    if (!ticked) return;
    const lo = TS[T_LAST_TICK] - sim.timeWindow, hi = sim.timeWindow === Infinity ? Infinity : TS[T_LAST_TICK];
    if (t < lo) ST[S_T] = lo; else if (t > hi) ST[S_T] = hi;
  }

  // ---- surface evaluation (local tangent-plane coordinates) --------------------------------
  // Scratch (module-closure, reused; never allocate per call)
  // Hot scratch state in a typed array: mutable doubles captured in a closure would be boxed
  // (one heap allocation per write), typed-array slots are not.

  // Jacobian d(displaced east, north)/d(rest east, north) of the whole chain (Newton inversion):
  // J = (I + G_1)(I + G_0), G_c = gradient of cascade c's horizontal displacement at its
  // (chained) sample point, in local terms.
  const ST = new Float64Array(23);
  const S_H = 0, S_E = 1, S_N = 2, S_VH = 3, S_VE = 4, S_VN = 5, S_J00 = 6, S_J01 = 7, S_J10 = 8, S_J11 = 9;
  const S_RE = 10, S_RN = 11, S_LE = 12, S_LN = 13;
  // Arguments of invert() / forward() (doubles passed as call arguments would be boxed by V8).
  const S_PE = 14, S_PN = 15, S_FE = 16, S_FN = 17;
  // Time argument of prepare() / blendInto() and the crossfade state it computes.
  const S_T = 18, S_W = 19, S_DW = 20, S_CHOP = 21, S_DCHOP = 22;

  /**
   * Chained forward evaluation at rest point (re, rn) (reference spectralField), fast path:
   * bilinear cascades are inlined. Uses the frames selected by the last prepare().
   */
  function forward(f: number, wantVel: boolean, wantGrad: boolean, fast: boolean): void {
    const re = ST[S_FE], rn = ST[S_FN];
    let H = 0, E = 0, N = 0, VH = 0, VE = 0, VN = 0;
    let J00 = 1, J01 = 0, J10 = 0, J11 = 1;
    // The Newton Jacobian only includes the first JACOBIAN_CASCADES cascades: the 9 m cascade moves
    // points by ~1-3 cm but has steep, quickly varying gradients that would only destabilise the
    // steps; its displacement is still part of the solved map (it converges as a fixed point).
    for (let k = 0; k < K; k++) {
      const c = kCascade[k];
      const data = kData[k];
      const off = kOff[k];
      if (kCubic[k] === 0 || fast) {
        const n = kN[k], inv = kInv[k], mask = n - 1;
        // texel coords of a = e + OFF, b = -n + OFF, shifted by a whole number of tiles (kShift) so they
        // are positive (inputs are wrapped to [0, SURFACE_PERIOD) and displacements are far below 4 tiles):
        // |0 is then floor.
        const sh = kShift[k];
        const sa = (re + E + off) * inv + sh, sb = (off - rn - N) * inv + sh;
        let x0 = sa | 0, y0 = sb | 0;
        const fx = sa - x0, fy = sb - y0;
        x0 &= mask; y0 &= mask;
        const x1 = (x0 + 1) & mask, y1 = (y0 + 1) & mask;
        // Lazy blocks: make sure the 4 texels (and, for velocity, their velocities) are materialised.
        const bs = kBSh[k], bn = kNb[k];
        const r0 = (y0 >> bs) * bn, r1 = (y1 >> bs) * bn, c0 = x0 >> bs, c1 = x1 >> bs;
        let bv = kBlock[k];
        if (bv[r0 + c0] === 0 || bv[r0 + c1] === 0 || bv[r1 + c0] === 0 || bv[r1 + c1] === 0) kSim[k].ensurePos(f, x0, y0, x1, y1);
        if (wantVel) {
          bv = kVelBlock[k];
          if (bv[r0 + c0] === 0 || bv[r0 + c1] === 0 || bv[r1 + c0] === 0 || bv[r1 + c1] === 0) kSim[k].ensureVel(f, x0, y0, x1, y1);
        }
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
          const t00 = g00 * J00 + g01 * J10, t01 = g00 * J01 + g01 * J11;
          const t10 = g10 * J00 + g11 * J10, t11 = g10 * J01 + g11 * J11;
          J00 = t00; J01 = t01; J10 = t10; J11 = t11;
        }
        E += w00 * e00 + w10 * e10 + w01 * e01 + w11 * e11;
        N += w00 * n00 + w10 * n10 + w01 * n01 + w11 * n11;
        if (wantVel) {
          // Material derivative of the chained surface: this cascade's own velocity plus advection of
          // its sample point, which moves with the horizontal velocity of the cascades before it
          // (VE, VN so far): d/dt X(q(t), t) = dX/dt + grad X . dq/dt.
          const v = kVel[k];
          const h00 = data[i00], h10 = data[i10], h01 = data[i01], h11 = data[i11];
          const gHe = (gy * (h10 - h00) + fy * (h11 - h01)) * inv, gHn = -(gx * (h01 - h00) + fx * (h11 - h10)) * inv;
          const gEe = (gy * (e10 - e00) + fy * (e11 - e01)) * inv, gEn = -(gx * (e01 - e00) + fx * (e11 - e10)) * inv;
          const gNe = (gy * (n10 - n00) + fy * (n11 - n01)) * inv, gNn = -(gx * (n01 - n00) + fx * (n11 - n10)) * inv;
          const vh = w00 * v[i00] + w10 * v[i10] + w01 * v[i01] + w11 * v[i11] + gHe * VE + gHn * VN;
          const ve = w00 * v[i00 + 1] + w10 * v[i10 + 1] + w01 * v[i01 + 1] + w11 * v[i11 + 1] + gEe * VE + gEn * VN;
          const vn = w00 * v[i00 + 2] + w10 * v[i10 + 2] + w01 * v[i01 + 2] + w11 * v[i11 + 2] + gNe * VE + gNn * VN;
          VH += vh; VE += ve; VN += vn;
        }
        continue;
      }
      const n = kN[k], L = kL[k], inv = kInv[k], mask = n - 1, sh = kShift[k];
      const a = re + E + off, b = -(rn + N) + off; // tile coords: a = east + OFF, b = -north + OFF
      // Catmull-Rom taps span texels xi-1 .. xi+2: materialise their (at most 2 x 2) blocks.
      const sa = a * inv + sh, sb = b * inv + sh;
      const xi = sa | 0, yi = sb | 0;
      {
        const bs = kBSh[k], bn = kNb[k], bv = kBlock[k];
        const xa = ((xi - 1) & mask) >> bs, xb = ((xi + 2) & mask) >> bs, ya = ((yi - 1) & mask) >> bs, yb = ((yi + 2) & mask) >> bs;
        if (bv[ya * bn + xa] === 0 || bv[ya * bn + xb] === 0 || bv[yb * bn + xa] === 0 || bv[yb * bn + xb] === 0) kSim[k].ensurePos(f, xi - 1, yi - 1, xi + 2, yi + 2);
      }
      if (!wantVel && !(wantGrad && c < JACOBIAN_CASCADES)) {
        CUBIC_IN[0] = xi; CUBIC_IN[1] = yi; CUBIC_IN[2] = sa - xi; CUBIC_IN[3] = sb - yi;
        cubicTaps(data, n);
        H += SAMPLE[0]; E += SAMPLE[1]; N += SAMPLE[2];
        continue;
      }
      SAMPLE_IN[0] = a; SAMPLE_IN[1] = b;
      if (wantVel) {
        kSim[k].ensureVel(f, xi - 1, yi - 1, xi + 2, yi + 2);
        sampleCubic(kVel[k], n, L);
        const v0 = SAMPLE[0], v1 = SAMPLE[1], v2 = SAMPLE[2];
        sampleCubicGrad(data, n, L); // local: d/de = d/da, d/dn = -d/db
        const vh = v0 + GRAD[4] * VE - GRAD[5] * VN;
        const ve = v1 + GRAD[0] * VE - GRAD[1] * VN;
        const vn = v2 + GRAD[2] * VE - GRAD[3] * VN;
        VH += vh; VE += ve; VN += vn;
      }
      if (wantGrad && c < JACOBIAN_CASCADES) {
        sampleCubicGrad(data, n, L);
        // local gradient: d/de = d/da, d/dn = -d/db
        const g00 = 1 + GRAD[0], g01 = -GRAD[1], g10 = GRAD[2], g11 = 1 - GRAD[3];
        const t00 = g00 * J00 + g01 * J10, t01 = g00 * J01 + g01 * J11;
        const t10 = g10 * J00 + g11 * J10, t11 = g10 * J01 + g11 * J11;
        J00 = t00; J01 = t01; J10 = t10; J11 = t11;
      } else sampleCubic(data, n, L);
      H += SAMPLE[0]; E += SAMPLE[1]; N += SAMPLE[2];    }
    ST[S_H] = H; ST[S_E] = re + E; ST[S_N] = rn + N;
    ST[S_VH] = VH; ST[S_VE] = VE; ST[S_VN] = VN;
    if (wantGrad) { ST[S_J00] = J00; ST[S_J01] = J01; ST[S_J10] = J10; ST[S_J11] = J11; }
  }

  /**
   * Find the rest point r displaced onto local (pe, pn), i.e. solve r + D(r) = p (D = chained
   * horizontal displacement), with Newton steps on the exact Jacobian of the sampled (bilinear /
   * Catmull-Rom) chain. Where the surface is close to folding (det J < 0.1) a plain fixed-point step
   * is taken instead. Stops when |residual| < 0.1 mm or after inversionIterations steps.
   * Where crests fold over (several rest points map to p) any of the sheets may be returned.
   * Leaves the forward() results for the found rest point in out*.
   */

  function invert(f: number, wantVel: boolean): void {
    let pe = ST[S_PE], pn = ST[S_PN];
    // The whole surface repeats every SURFACE_PERIOD m: wrap so all tile coordinates stay small.
    const se = pe - wrapPeriod(pe), sn = pn - wrapPeriod(pn);
    pe -= se; pn -= sn;
    let re = pe, rn = pn;
    if (K > 0) {
      // Initial guess: GUESS_STEPS cheap fixed-point steps on the largest cascade alone
      // (r = p - D0(r)); it carries most of the displacement. Bilinear on the frame texels even when the
      // cascade is sampled with Catmull-Rom: it is only a starting point.
      const data = kData[0], n = kN[0], inv = kInv[0], off = kOff[0], mask = n - 1;
      for (let step = 0; step < GUESS_STEPS; step++) {
        const sa = (re + off) * inv, sb = (off - rn) * inv;
        let x0 = Math.floor(sa), y0 = Math.floor(sb);
        const fx = sa - x0, fy = sb - y0;
        x0 &= mask; y0 &= mask;
        const x1 = (x0 + 1) & mask, y1 = (y0 + 1) & mask;
        const bv = kBlock[0], bs = kBSh[0], bn = kNb[0];
        const r0 = (y0 >> bs) * bn, r1 = (y1 >> bs) * bn, c0 = x0 >> bs, c1 = x1 >> bs;
        if (bv[r0 + c0] === 0 || bv[r0 + c1] === 0 || bv[r1 + c0] === 0 || bv[r1 + c1] === 0) kSim[0].ensurePos(f, x0, y0, x1, y1);
        const i00 = (y0 * n + x0) * 3, i10 = (y0 * n + x1) * 3, i01 = (y1 * n + x0) * 3, i11 = (y1 * n + x1) * 3;
        const gx = 1 - fx, gy = 1 - fy;
        re = pe - (gx * gy * data[i00 + 1] + fx * gy * data[i10 + 1] + gx * fy * data[i01 + 1] + fx * fy * data[i11 + 1]);
        rn = pn - (gx * gy * data[i00 + 2] + fx * gy * data[i10 + 2] + gx * fy * data[i01 + 2] + fx * fy * data[i11 + 2]);
      }
    }
    // Phase A: Newton on the BILINEAR reading of the same frame texels (cheap; gradients exact for it).
    let pre = re, prn = rn, prevRes = Infinity; // last accepted iterate (backtracking safeguard)
    let bestE = re, bestN = rn, bestRes = Infinity;
    const iters = sim.inversionIterations;
    const tol2 = sim.inversionTolerance * sim.inversionTolerance;
    let converged = false;
    for (let it = 0; it < iters; it++) {
      ST[S_FE] = re; ST[S_FN] = rn;
      forward(f, false, true, true);
      const ee = ST[S_E] - pe, en = ST[S_N] - pn;
      const res = ee * ee + en * en;
      if (res < bestRes) { bestRes = res; bestE = re; bestN = rn; }
      if (res < tol2) { converged = true; break; }
      if (res >= prevRes) { // step made things worse: halve it
        re = pre + 0.5 * (re - pre);
        rn = prn + 0.5 * (rn - prn);
        continue;
      }
      pre = re; prn = rn; prevRes = res;
      const det = ST[S_J00] * ST[S_J11] - ST[S_J01] * ST[S_J10];
      if (det > 0.1) {
        re -= (ST[S_J11] * ee - ST[S_J01] * en) / det;
        rn -= (ST[S_J00] * en - ST[S_J10] * ee) / det;
      } else {
        re -= ee; rn -= en;
      }
    }
    // Rescue (rare: < 0.1 % of queries, near folding crests): damped fixed-point steps from the best
    // point so far, which converge where the Newton steps oscillate between cells.
    if (!converged && bestRes > 25 * tol2) {
      re = bestE; rn = bestN;
      for (let it = 0; it < RESCUE_ITERATIONS; it++) {
        ST[S_FE] = re; ST[S_FN] = rn;
        forward(f, false, false, true);
        const ee = ST[S_E] - pe, en = ST[S_N] - pn;
        const res = ee * ee + en * en;
        if (res < bestRes) { bestRes = res; bestE = re; bestN = rn; }
        if (res < tol2) break;
        re -= 0.5 * ee; rn -= 0.5 * en;
      }
    }
    re = bestE; rn = bestN;
    // Phase B: the exact (Catmull-Rom) surface differs from the bilinear reading by ~cm: a few
    // Newton steps with the phase-A Jacobian (still in ST) land on it. Without cubic cascades the
    // phase-A point is already exact.
    let exactAt = false; // the last exact forward() was at (re, rn)
    if (anyCubic) {
      const det = ST[S_J00] * ST[S_J11] - ST[S_J01] * ST[S_J10];
      for (let it = 0; it < CUBIC_REFINE_ITERATIONS; it++) {
        ST[S_FE] = re; ST[S_FN] = rn;
        forward(f, false, false, false);
        const ee = ST[S_E] - pe, en = ST[S_N] - pn;
        if (ee * ee + en * en < tol2) { exactAt = true; break; }
        if (det > 0.1) {
          re -= (ST[S_J11] * ee - ST[S_J01] * en) / det;
          rn -= (ST[S_J00] * en - ST[S_J10] * ee) / det;
        } else { re -= ee; rn -= en; }
      }
    }
    if (!exactAt || wantVel) {
      ST[S_FE] = re; ST[S_FN] = rn;
      forward(f, wantVel, false, false);
    }
    ST[S_RE] = re + se; ST[S_RN] = rn + sn;
  }


  const P3: Vec3 = { x: 0, y: 0, z: 0 };
  function toLocal(p: Readonly<Vec3>): void {
    (shape as WorldShape).tangentFrame(p, frame);
    const dx = p.x - anchor.x, dy = p.y - anchor.y, dz = p.z - anchor.z;
    ST[S_LE] = dx * frame.east.x + dy * frame.east.y + dz * frame.east.z;
    ST[S_LN] = dx * frame.north.x + dy * frame.north.y + dz * frame.north.z;
  }

  const query: WaterQuery = {
    heightAt(p: Readonly<Vec3>, time: number): number {
      ST[S_T] = time;
      const f = prepare();
      toLocal(p);
      ST[S_PE] = ST[S_LE]; ST[S_PN] = ST[S_LN];
      invert(f, false);
      return ST[S_H];
    },
    velocityAt(p: Readonly<Vec3>, time: number, out: Vec3): Vec3 {
      ST[S_T] = time;
      const f = prepare();
      toLocal(p); // also fills `frame`
      ST[S_PE] = ST[S_LE]; ST[S_PN] = ST[S_LN];
      invert(f, true);
      const e = frame.east, n = frame.north, u = frame.up;
      out.x = ST[S_VE] * e.x + ST[S_VN] * n.x + ST[S_VH] * u.x;
      out.y = ST[S_VE] * e.y + ST[S_VN] * n.y + ST[S_VH] * u.y;
      out.z = ST[S_VE] * e.z + ST[S_VN] * n.z + ST[S_VH] * u.z;
      return out;
    },
  };

  // ---- wind --------------------------------------------------------------------------------
  function sampleWind(snap: boolean): void {
    if (!wind || !useWind || !shape) return;
    const dt = TS[T_DT];
    shape.atAltitude(focus, sim.windSampleHeight, windPoint);
    wind.at(windPoint, windVec);
    shape.tangentFrame(windPoint, frame);
    const we = windVec.x * frame.east.x + windVec.y * frame.east.y + windVec.z * frame.east.z;
    const wn = windVec.x * frame.north.x + windVec.y * frame.north.y + windVec.z * frame.north.z;
    // Speed and direction are low-passed SEPARATELY (a vector EMA would pass through ~0 m/s when
    // the wind reverses, and the sea would collapse). Direction follows the shortest arc.
    const speedNow = Math.sqrt(we * we + wn * wn);
    if (snap || !windInit) {
      TS[T_WIND_SPEED] = speedNow;
      if (speedNow > 1e-3) TS[T_WIND_DIR] = Math.atan2(wn, we);
      windInit = true;
    } else {
      const a = sim.windTimeConstant > 0 ? 1 - Math.exp(-dt / sim.windTimeConstant) : 1;
      TS[T_WIND_SPEED] += (speedNow - TS[T_WIND_SPEED]) * a;
      if (speedNow > 0.5) {
        let dd = Math.atan2(wn, we) - TS[T_WIND_DIR];
        while (dd > Math.PI) dd -= 2 * Math.PI;
        while (dd < -Math.PI) dd += 2 * Math.PI;
        let d = TS[T_WIND_DIR] + dd * a;
        while (d > Math.PI) d -= 2 * Math.PI;
        while (d < -Math.PI) d += 2 * Math.PI;
        TS[T_WIND_DIR] = d;
      }
    }
    target.windSpeed = TS[T_WIND_SPEED];
    target.windDirection = TS[T_WIND_DIR];
  }

  function rebuildTarget(): void {
    const ws = target.windSpeed, wd = target.windDirection;
    copyParams(base, target);
    if (wind && useWind && windInit) { target.windSpeed = ws; target.windDirection = wd; }
  }

  function applyImmediate(): void {
    if (!initialised) { init(); return; }
    if (regenRemote) {
      // A worker may still be writing a slot: let it finish (setup-time call, rare) before reusing slots.
      for (let k = 0; k < K; k++) { while (!kSim[k].regenDone()) { /* spin */ } kSim[k].ackRegen(); }
      regenRemote = false;
    }
    regenSlot = -1;
    pendingId = -1; pendingSlot = -1;
    const slot = (toSlot + 1) % NUM_SLOTS;
    generateNow(slot, target);
    setActive(++srcCounter, slot, slot, 0, 1);
  }

  // ---- state for the GPU adapter -------------------------------------------------------------
  const slotViews: OceanSlot[] = slots as unknown as OceanSlot[];
  const state = {
    masterSize: REFERENCE_SIZE,
    seed,
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
    state.blendStart = TS[T_BLEND_START]; state.blendDuration = TS[T_BLEND_DURATION];
    ST[S_T] = TS[T_LAST_TICK]; blendInto();
    state.blend = ST[S_W]; state.choppiness = ST[S_CHOP];
    state.time = TS[T_LAST_TICK];
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
        TS[T_DT] = 0;
        sampleWind(true);
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
      if (!(T - T === 0)) failTickTime(); // (throw kept out of line: keeps tick small for the JIT)
      TS[T_LAST_TICK] = T;
      ticked = true;
      if (!workersStarted) startWorkers();
      TS[T_DT] = info.dt;
      sampleWind(false);
      // Regeneration (only once the current crossfade has finished and nothing is pending).
      const blendDone = fromSlot === toSlot || T >= TS[T_BLEND_START] + TS[T_BLEND_DURATION];
      if (regenSlot >= 0) stepRegen(sim.regenRowsPerTick);
      else if (pendingId < 0 && blendDone) {
        // Inlined copy of differs() (a function called once per tick stays in the interpreter for a
        // long time, which boxes every double it computes = allocation per tick).
        const a = target, b = slots[toSlot].params;
        let du = a.windSpeed - b.windSpeed;
        if (du < 0) du = -du;
        let regen = du > sim.windSpeedTolerance * (b.windSpeed > 1 ? b.windSpeed : 1);
        if (!regen && a.windSpeed > 0.5 && b.windSpeed > 0.5) {
          let dd = a.windDirection - b.windDirection;
          while (dd > Math.PI) dd -= 2 * Math.PI;
          while (dd < -Math.PI) dd += 2 * Math.PI;
          if (dd < 0) dd = -dd;
          regen = dd > (sim.windDirectionToleranceDeg * Math.PI) / 180;
        }
        if (!regen) {
          let df = a.fetch - b.fetch;
          if (df < 0) df = -df;
          regen = df > 1e-6 * (1 + b.fetch) || a.gamma !== b.gamma || a.spread !== b.spread || a.standing !== b.standing ||
            a.choppiness !== b.choppiness || a.amplitude !== b.amplitude;
        }
        if (regen) {
          startRegen();
          stepRegen(sim.regenRowsPerTick);
        }
      }
      // Switch to the pending source once the CPU has its snapshots for this interval (normally
      // well before its crossfade starts; at the start time at the latest, computing them now).
      if (pendingId >= 0 && T >= TS[T_PEND_SWITCH]) {
        let ready = T >= TS[T_PEND_START];
        for (let k = 0; k < K && !ready; k++) {
          const sm = kSim[k];
          const j0 = Math.floor(T / sm.snapshotDt);
          if (!sm.has(pendingId, j0) || !sm.has(pendingId, j0 + 1)) break;
          if (k === K - 1) ready = true;
        }
        if (ready) setActive(pendingId, pendingFrom, pendingSlot, TS[T_PEND_START], TS[T_PEND_DURATION]);
      }
      // CPU snapshots (amortised) + the field at T for this step's queries.
      for (let k = 0; k < K; k++) {
        const sm = kSim[k];
        sm.args[0] = T; sm.args[1] = info.dt; sm.args[2] = TS[T_PEND_SWITCH];
        sm.schedule(activeId, pendingId);
      }
      ST[S_T] = T;
      prepare();
      // Ask the workers to fill the position frames of the NEXT tick (T + dt) ahead of time; the next
      // prepare() adopts them (bit-identical to filling here), so queries at tick time fill nothing.
      if (workers.length > 0) {
        ST[S_T] = T + info.dt;
        blendInto();
        for (let k = 0; k < K; k++) {
          const sm = kSim[k];
          sm.args[0] = ST[S_T]; sm.args[1] = ST[S_CHOP];
          sm.requestPrefill(activeId);
        }
      }
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
      ST[S_T] = time;
      const f = prepare();
      ST[S_PE] = east; ST[S_PN] = north;
      invert(f, false);
      return ST[S_H];
    },

    invertLocal(east: number, north: number, time: number, out: RestSample): number {
      ST[S_T] = time;
      const f = prepare(true);
      ST[S_PE] = east; ST[S_PN] = north;
      invert(f, true);
      const residual = Math.hypot(ST[S_E] - east, ST[S_N] - north);
      out.east = ST[S_RE]; out.north = ST[S_RN]; out.up = ST[S_H];
      out.vEast = ST[S_VE]; out.vNorth = ST[S_VN]; out.vUp = ST[S_VH];
      return residual;
    },

    sampleRest(east: number, north: number, time: number, out: RestSample): RestSample {
      ST[S_T] = time;
      const f = prepare(true);
      const se = east - wrapPeriod(east), sn = north - wrapPeriod(north);
      ST[S_FE] = east - se; ST[S_FN] = north - sn;
      forward(f, true, false, false);
      ST[S_E] += se; ST[S_N] += sn;
      out.east = ST[S_E]; out.north = ST[S_N]; out.up = ST[S_H];
      out.vEast = ST[S_VE]; out.vNorth = ST[S_VN]; out.vUp = ST[S_VH];
      return out;
    },

    cpuStats(out: number[]): number[] {
      for (let k = 0; k < K; k++) {
        const sm = kSim[k], c = kCascade[k];
        out[5 * c] = sm.computed; out[5 * c + 1] = sm.syncComputed; out[5 * c + 2] = sm.remoteComputed;
        out[5 * c + 3] = sm.posBlocksFilled; out[5 * c + 4] = sm.velBlocksFilled;
        out[5 * K + c] = sm.prefillsAdopted;
      }
      return out;
    },

    snapshotCounts(out: number[]): number[] {
      for (let c = 0; c < sims.length; c++) out[c] = sims[c] ? (sims[c] as CascadeSim).computed : 0;
      return out;
    },

    heightsAt(points: Float64Array, time: number, out: Float64Array, count?: number): Float64Array {
      const n = count ?? Math.min(out.length, (points.length / 3) | 0);
      ST[S_T] = time;
      const f = prepare();
      for (let i = 0; i < n; i++) {
        P3.x = points[3 * i]; P3.y = points[3 * i + 1]; P3.z = points[3 * i + 2];
        toLocal(P3);
        ST[S_PE] = ST[S_LE]; ST[S_PN] = ST[S_LN];
        invert(f, false);
        out[i] = ST[S_H];
      }
      return out;
    },

    dispose(): void {
      // Release the CPU buffers (slots, snapshot pools, frames) and the workers; the query throws afterwards.
      disposed = true;
      wind = undefined;
      for (let k = 0; k < K; k++) kSim[k].stopRemote();
      for (const w of workers) w.terminate();
      workers.length = 0;
      for (let k = 0; k < K; k++) kSim[k].dispose();
      K = 0;
      kSim.length = 0;
      sims.length = 0;
      slots.length = 0;
      xi.length = 0;
    },
  };
  return ocean;
}
