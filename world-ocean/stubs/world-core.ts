// Engine contracts copied VERBATIM from docs/Ocean-WebGPU-Port-Rules.md §3 (world/core ^0.10).
// On import into the engine this file is removed and imports switch to '@dreamp/world-core'.
// Do not edit: if a contract needs to change, note it in HANDOFF.md.

/** A 3D vector or point, metres. Mutable so it can be reused as an `out` parameter. */
export interface Vec3 { x: number; y: number; z: number }
export interface TangentFrame { east: Vec3; north: Vec3; up: Vec3 }

export interface WorldShape {
  readonly kind: 'flat' | 'sphere';
  readonly radius?: number;                  // sphere: sea-level radius, m
  readonly center?: Readonly<Vec3>;          // sphere centre, world space
  readonly seaLevel?: number;                // flat: sea level on +Y, m
  up(p: Readonly<Vec3>, out: Vec3): Vec3;    // +Y on flat, away from centre on sphere
  altitude(p: Readonly<Vec3>): number;       // height above sea level along up, m
  atAltitude(p: Readonly<Vec3>, altitude: number, out: Vec3): Vec3;
  // Flat: east +X, north -Z, up +Y. Right-handed: east x north = up.
  tangentFrame(p: Readonly<Vec3>, out: TangentFrame): TangentFrame;
  gravity(p: Readonly<Vec3>, out: Vec3): Vec3; // m/s^2, points down
  readonly surfaceGravity: number;           // 9.80665 default (use this as g in the dispersion relation)
}

export interface FloatingOrigin {
  readonly origin: Readonly<Vec3>;           // float64, world space. render = world - origin
  toRender(world: Readonly<Vec3>, out: Vec3): Vec3;
  toWorld(render: Readonly<Vec3>, out: Vec3): Vec3;
  onShift(fn: (delta: Readonly<Vec3>, origin: Readonly<Vec3>) => void): () => void;
  /** origin mod period per axis, in [0, period). Shaders use renderPos + wrapped. Never add the raw origin in a shader. */
  wrapped(period: number, out: Vec3): Vec3;
}

/** What the ocean PROVIDES. */
export interface WaterQuery {
  /** Water surface height along up at `p`, metres above sea level, at simulation time `time` (s). */
  heightAt(p: Readonly<Vec3>, time: number): number;
  /** Surface velocity at `p`, m/s, world space. Writes and returns `out`. */
  velocityAt(p: Readonly<Vec3>, time: number, out: Vec3): Vec3;
}

/** What the ocean READS (all optional). */
export interface WindQuery {
  /** Wind velocity at `p` (gusts included), m/s, world space. Sample at 10 m above sea level for the spectrum. */
  at(p: Readonly<Vec3>, out: Vec3): Vec3;
}

export interface EnvironmentState {
  readonly time: number;                     // simulation seconds
  readonly sunDir: Readonly<Vec3>;           // unit, towards the sun
  readonly moonDir: Readonly<Vec3>;
  readonly cloudCover: number;               // 0..1
  readonly rain: number;                     // 0..1 (later: rain ripples)
  readonly windBase: Readonly<Vec3>;         // m/s at 10 m
  // (more fields exist: dayFraction, temperature, fog, wetness, snow...)
}

export interface TerrainQuery {
  /** Ground height along up at `p`, metres above sea level. Depth = water height - this. */
  heightAt(p: Readonly<Vec3>): number;
}

/** GPU grid shared between modules. uv = (dot(p - center, east), dot(p - center, north)) / size + 0.5 */
export interface GpuGridSource {
  readonly texture: unknown;                 // a three.js Texture in the three adapter
  readonly center: Readonly<Vec3>;           // world space, texel-snapped
  readonly east: Readonly<Vec3>;
  readonly north: Readonly<Vec3>;
  readonly up?: Readonly<Vec3>;
  readonly size: number;                     // metres per side
  readonly resolution: number;               // texels per side
  readonly version: number;                  // bumps when contents / mapping change
}
/** R float = ground height above sea level, m, around the camera. For shore depth / foam / wave damping. */
export interface TerrainGpuHeight extends GpuGridSource {}

export interface RGB { r: number; g: number; b: number }   // linear, >= 0
export interface SkyLight {
  readonly sunIlluminance: number;           // lux
  readonly sunColor: Readonly<RGB>;          // normalised luminance 1
  readonly skyIlluminance: number;           // lux, up-facing
  readonly skyColor: Readonly<RGB>;
  readonly skySH: Float32Array;              // order-2 SH (27 numbers), cd/m^2 — use for diffuse / rough reflection
  readonly fogDensity: number; readonly fogHeightFalloff: number; readonly fogColor: Readonly<RGB>;
  readonly suggestedEV100: number;
  readonly version: number;
}

export interface ProviderKey<T> { readonly id: string; readonly _t?: T }
export type TickPhase = 'time' | 'weather' | 'wind' | 'simulate' | 'gameplay' | 'render';
export interface TickInfo { readonly dt: number; readonly time: number; readonly step: number }
export interface FrameInfo {
  readonly frameDt: number;                  // real seconds since last frame
  readonly time: number;                     // simulation time
  readonly alpha: number;                    // 0..1 between the last two fixed steps
  readonly interpolatedTime: number;         // DRAW the waves at this time (smooth at any frame rate)
  readonly frame: number;
}

export interface World {
  readonly shape: WorldShape;
  readonly origin: FloatingOrigin;
  readonly time: number;
  provide<T>(key: ProviderKey<T>, impl: T): void;
  require<T>(key: ProviderKey<T>): T;
  optional<T>(key: ProviderKey<T>): T | undefined;
  on(type: string, fn: (e: { type: string; data?: unknown }) => void): () => void;
}

export interface WorldSystem {
  readonly id: string;                                  // 'world/ocean'
  readonly provides?: readonly ProviderKey<unknown>[];  // [WATER_QUERY]
  readonly requires?: readonly ProviderKey<unknown>[];
  readonly optional?: readonly ProviderKey<unknown>[];
  install(world: World): void;                          // world.provide(WATER_QUERY, query) here
  start?(world: World): void;                           // world.optional(WIND_QUERY) etc. here
  readonly phases?: readonly TickPhase[];               // ['simulate']
  tick?(phase: TickPhase, info: TickInfo): void;        // fixed step: spectrum params from wind, CPU FFT
  frame?(info: FrameInfo): void;                        // variable rate: copy state to GPU uniforms
  dispose?(): void;
}

// Keys (ids are what matter: the engine defines them with these exact strings)
export const WATER_QUERY  = { id: 'WATER_QUERY' }  as ProviderKey<WaterQuery>;
export const WIND_QUERY   = { id: 'WIND_QUERY' }   as ProviderKey<WindQuery>;
export const ENVIRONMENT  = { id: 'ENVIRONMENT' }  as ProviderKey<EnvironmentState>;
export const TERRAIN_QUERY= { id: 'TERRAIN_QUERY' }as ProviderKey<TerrainQuery>;
export const TERRAIN_GPU  = { id: 'TERRAIN_GPU' }  as ProviderKey<TerrainGpuHeight>;
export const SKY_LIGHT    = { id: 'SKY_LIGHT' }    as ProviderKey<SkyLight>;
