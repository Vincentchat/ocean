// Config + parameter definitions for world/ocean. No three import.
//
// Single import location for the engine contracts (spec section 9): on import into the engine, change the
// line below to `from '@dreamp/world-core'` and delete stubs/.
export type {
  Vec3, TangentFrame, WorldShape, FloatingOrigin, WaterQuery, WindQuery, EnvironmentState,
  TerrainQuery, GpuGridSource, TerrainGpuHeight, RGB, SkyLight, ProviderKey, TickPhase, TickInfo,
  FrameInfo, World, WorldSystem,
} from './stubs/world-core.ts';
export {
  WATER_QUERY, WIND_QUERY, ENVIRONMENT, TERRAIN_QUERY, TERRAIN_GPU, SKY_LIGHT,
} from './stubs/world-core.ts';

import type { Vec3, WorldShape } from './stubs/world-core.ts';
import type { SpectrumParams } from './spectrum.ts';
export type { SpectrumParams, CascadeDef } from './spectrum.ts';

export type PresetName = 'calm' | 'breeze' | 'windy' | 'storm';

/** A preset: the spectrum parameters plus the documentation row of the target table. */
export interface OceanPreset {
  readonly name: PresetName;
  readonly params: Readonly<SpectrumParams>;
  /** Target table (TUNING.md / tests). */
  readonly target: {
    /** Expected significant wave height Hs = 4 sigma, m (tests allow +-15 %). */
    readonly hs: number;
    /** Peak period, s. */
    readonly peakPeriod: number;
    /** Peak wavelength, m. */
    readonly peakWavelength: number;
    readonly reason: string;
  };
}

/** Tunables of the simulation (not of the spectrum shape). */
export interface SimParams {
  /** Crossfade duration between two spectra, s. */
  blendTime: number;
  /** Time constant of the wind low-pass (EMA on the wind vector), s. */
  windTimeConstant: number;
  /** Regenerate the spectrum when the smoothed wind speed drifts by more than this fraction. */
  windSpeedTolerance: number;
  /** ... or when the smoothed wind direction drifts by more than this, degrees. */
  windDirectionToleranceDeg: number;
  /** Spectrum regeneration budget: master rows (of 3 x 256) per tick. */
  regenRowsPerTick: number;
  /** Fixed-point iterations used to invert the horizontal (choppy) displacement in heightAt. */
  inversionIterations: number;
  /** Stop the inversion when the horizontal residual |P(r) - p| is below this, m. */
  inversionTolerance: number;
  /** Wind sampling height above sea level, m (spec: 10 m). */
  windSampleHeight: number;
  /**
   * Query time window, s: once ticking, query times are clamped to [lastTick - w, lastTick]
   * (the cached snapshots, bounded cost). Infinity = exact at any time (a cold time costs ~50-70 ms).
   */
  timeWindow: number;
  /** Largest change of the spectrum's wind direction per crossfade, degrees (avoids mid-blend energy loss). */
  maxTurnPerBlendDeg: number;
}

/** CPU query configuration (per cascade). Setup-time; see cpu-fft.ts. */
export interface QueryCascadeConfig {
  /** CPU FFT size for this cascade (power of two, 0 = cascade skipped by the CPU query). */
  size: number;
  /**
   * Snapshot spacing in time, s (cubic Hermite in time between snapshots, exact velocities). Should
   * divide the 8192 s loop period (e.g. 0.5, 0.25, 0.125) so that t and t + 8192 give identical heights.
   */
  snapshotDt: number;
  /** Spatial filter: 'bilinear' (matches GPU hardware/manual bilinear) or 'cubic' (Catmull-Rom). */
  filter: 'bilinear' | 'cubic';
}

export interface PerfSink {
  begin(module: string, phase: string): void;
  end(): void;
}

export interface OceanOptions extends Partial<SpectrumParams>, Partial<SimParams> {
  /** Preset name (default 'windy' = the reference's load-time scene). */
  preset?: PresetName;
  /** Random realisation, integer. 1 = the reference's waves. Default 1. */
  seed?: number;
  /** Use WIND_QUERY (when installed) to drive windSpeed / windDirection. Default true. */
  useWind?: boolean;
  /** World point (at sea level) that is the origin of the ocean's tangent-plane coordinates. Default (0,0,0). */
  anchor?: Readonly<Vec3>;
  /** CPU query cascades (defaults: DEFAULT_QUERY_CASCADES). */
  query?: readonly QueryCascadeConfig[];
  /** Optional perf sink (engine perf.begin/end). */
  perf?: PerfSink;
  /**
   * Run the CPU query's snapshot FFTs in worker threads (one per cascade; ocean-worker.ts) when
   * SharedArrayBuffer + workers are available (Node, or a cross-origin isolated page). Default true.
   * Results are bit-identical either way; the main-thread tick drops by the snapshot cost (~1.5 ms).
   */
  worker?: boolean;
  /** Shape to use before install() (standalone / tests). install() uses world.shape if not given. */
  shape?: WorldShape;
}

/** One tunable's definition (UI / docs / validation). */
export interface ParamDef {
  readonly key: string;
  readonly label: string;
  readonly unit: string;
  readonly min: number;
  readonly max: number;
  readonly default: number;
  readonly step?: number;
  readonly description: string;
}

/** Defaults of the sim tunables. */
export const DEFAULT_SIM_PARAMS: Readonly<SimParams> = Object.freeze({
  blendTime: 4,
  windTimeConstant: 8,
  windSpeedTolerance: 0.03,
  windDirectionToleranceDeg: 3,
  regenRowsPerTick: 4,
  inversionIterations: 6,
  inversionTolerance: 0.004,
  windSampleHeight: 10,
  timeWindow: 0.1,
  maxTurnPerBlendDeg: 15,
});

/**
 * CPU query cascades. Cascades 0 (1024 m, ~96 % of the height variance at 'windy') and 1 (96 m,
 * ~4 %) run at 256 with Catmull-Rom, i.e. the same grid + filter the GPU draws near the camera (identical numbers
 * up to float32 / Hermite-in-time rounding). Cascade 2 (9 m, ~0.03 %) runs at 128, an exact
 * spectral subset of the 256 grid (modes |k| < 64 dk). Snapshot spacing in time: 0.5 / 0.25 / 0.125 s
 * (binary fractions that divide the 8192 s loop: the snapshot grid repeats with the waves)
 * (Hermite-in-time error <= 0.45 % of Hs for every preset; snapshot cost ~1.5 ms per 60 Hz tick, run by the
 * snapshot workers when available, see OceanOptions.worker).
 * Cost and measured parity vs a full 256^3 evaluation: see tests/core-parity.test.ts and the report.
 * LITE alternative (half the CPU): [{256, .4}, {128, .25, 'cubic'}, {64, .1}] (p99 error ~1.2 % of Hs).
 */
export const DEFAULT_QUERY_CASCADES: readonly QueryCascadeConfig[] = Object.freeze([
  Object.freeze({ size: 256, snapshotDt: 0.5, filter: 'cubic' as const }),
  Object.freeze({ size: 256, snapshotDt: 0.25, filter: 'cubic' as const }),
  Object.freeze({ size: 128, snapshotDt: 0.125, filter: 'cubic' as const }),
]);

/** Every tunable with min / max / unit / label / default (default = the 'windy' preset = the reference). */
export const PARAM_DEFS: readonly ParamDef[] = Object.freeze([
  { key: 'windSpeed', label: 'Wind speed (10 m)', unit: 'm/s', min: 0.5, max: 35, default: 15, step: 0.5, // hard cap MAX_WIND_SPEED
    description: 'U10 for the JONSWAP spectrum. From WIND_QUERY when installed (useWind).' },
  { key: 'windDirection', label: 'Wind direction', unit: 'rad', min: -Math.PI, max: Math.PI, default: -1, step: 0.01,
    description: 'Direction the wind blows towards, CCW from local east to north. Reference: 1 rad towards +X..+Z = -1 rad here.' },
  { key: 'fetch', label: 'Fetch', unit: 'm', min: 1000, max: 200000, default: 45479, step: 1000,
    description: 'Distance the wind has blown over water. Sets the peak: wp = max(22 (g^2/(U F))^(1/3), 0.855 g/U, peak <= 160 m). Hard caps (spectrum.ts): fetch <= 200 km, U <= 35 m/s, peak wavelength <= 160 m, so Hs <= ~14.9 m.' },
  { key: 'gamma', label: 'Peak enhancement', unit: '', min: 1, max: 7, default: 3.3, step: 0.1,
    description: 'JONSWAP gamma. 1 = Pierson-Moskowitz, 3.3 = standard JONSWAP.' },
  { key: 'spread', label: 'Directional spread', unit: '', min: 0.1, max: 4, default: 1.57, step: 0.01,
    description: 'Multiplies the Mitsuyasu spreading exponent (9.77 at the peak). Higher = waves more aligned with the wind.' },
  { key: 'standing', label: 'Opposing waves', unit: '0..1', min: 0, max: 0.5, default: 0.13, step: 0.01,
    description: 'Fraction mixed in from counter-propagating waves (reference 0.13).' },
  { key: 'choppiness', label: 'Choppiness', unit: '', min: 0, max: 2.5, default: 1.5, step: 0.05,
    description: 'Horizontal displacement multiplier (sharp crests). Above ~2 crests fold over.' },
  { key: 'amplitude', label: 'Amplitude', unit: 'x', min: 0, max: 3, default: 1.578, step: 0.01,
    description: 'Height multiplier on the physical spectrum (also scales the horizontal displacement). 1.578 = the reference calibration used by every preset (Hs ~2x open-ocean JONSWAP).' },
  { key: 'blendTime', label: 'Spectrum crossfade', unit: 's', min: 0.5, max: 30, default: 4, step: 0.5,
    description: 'Crossfade duration when the spectrum is regenerated (wind / preset change).' },
  { key: 'windTimeConstant', label: 'Wind smoothing', unit: 's', min: 0, max: 120, default: 8, step: 1,
    description: 'EMA time constant on the sampled wind speed and (separately, shortest arc) direction: gusts do not reshape the sea instantly and a reversal never passes through 0 m/s.' },
  { key: 'windSpeedTolerance', label: 'Wind speed tolerance', unit: 'fraction', min: 0.005, max: 0.5, default: 0.03, step: 0.005,
    description: 'Relative drift of the smoothed wind speed that triggers a regeneration + crossfade.' },
  { key: 'windDirectionToleranceDeg', label: 'Wind direction tolerance', unit: 'deg', min: 0.5, max: 45, default: 3, step: 0.5,
    description: 'Drift of the smoothed wind direction that triggers a regeneration + crossfade.' },
  { key: 'regenRowsPerTick', label: 'Regeneration rows / tick', unit: 'rows', min: 1, max: 768, default: 4, step: 1,
    description: 'Amortisation of the spectrum regeneration (3 x 256 rows in total).' },
  { key: 'inversionIterations', label: 'Inversion iterations', unit: '', min: 0, max: 12, default: 6, step: 1,
    description: 'Max Newton iterations to find the rest point that is displaced onto the query point (reference: 3 plain fixed-point steps).' },
  { key: 'inversionTolerance', label: 'Inversion tolerance', unit: 'm', min: 0.0001, max: 0.05, default: 0.004, step: 0.0005,
    description: 'Horizontal residual at which the inversion stops. Height error is about slope x residual.' },
  { key: 'timeWindow', label: 'Query time window', unit: 's', min: 0.02, max: Infinity, default: 0.1, step: 0.01,
    description: 'Queries are exact within [lastTick - w, lastTick] (cached snapshots, no stall; covers the interpolated render time); other times are clamped (future times to the last tick). Infinity = always exact (synchronous snapshot cost ~50-70 ms per new time).' },
  { key: 'maxTurnPerBlendDeg', label: 'Max sea turn per crossfade', unit: 'deg', min: 1, max: 180, default: 15, step: 1,
    description: 'The spectrum direction follows the wind by at most this per crossfade (complex-amplitude crossfades between distant directions lose energy mid-blend).' },
  { key: 'windSampleHeight', label: 'Wind sample height', unit: 'm', min: 0, max: 100, default: 10, step: 1,
    description: 'Height above sea level at which WIND_QUERY is sampled near the focus point.' },
] as ParamDef[]);
