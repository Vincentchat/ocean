// Tiny implementations of the world/core contracts for the side repo (tests + demo).
// No three.js import here: this file must run in Node with --experimental-strip-types.
// None of the per-call methods allocate.
import type {
  EnvironmentState,
  FloatingOrigin,
  FrameInfo,
  ProviderKey,
  RGB,
  SkyLight,
  TangentFrame,
  TerrainGpuHeight,
  TerrainQuery,
  TickInfo,
  TickPhase,
  Vec3,
  WindQuery,
  World,
  WorldShape,
  WorldSystem,
} from './world-core.ts';

export const STANDARD_GRAVITY = 9.80665; // m/s^2

// ---------------------------------------------------------------------------------------------
// Small vector helpers (allocation-free, out parameters)
// ---------------------------------------------------------------------------------------------
export function vec3(x = 0, y = 0, z = 0): Vec3 {
  return { x, y, z };
}
function set(out: Vec3, x: number, y: number, z: number): Vec3 {
  out.x = x; out.y = y; out.z = z;
  return out;
}

/** Unit quaternion, rotation of the canonical flat frame (east +X, up +Y, north -Z). */
export interface Quat { x: number; y: number; z: number; w: number }

/** Quaternion from an axis (need not be unit) and an angle in radians. Setup-time helper. */
export function quatFromAxisAngle(axis: Readonly<Vec3>, angleRad: number): Quat {
  const len = Math.hypot(axis.x, axis.y, axis.z) || 1;
  const s = Math.sin(angleRad / 2) / len;
  return { x: axis.x * s, y: axis.y * s, z: axis.z * s, w: Math.cos(angleRad / 2) };
}

/** out = q * v * q^-1 (v may alias out). */
export function rotateByQuat(q: Readonly<Quat>, v: Readonly<Vec3>, out: Vec3): Vec3 {
  const { x: qx, y: qy, z: qz, w: qw } = q;
  const vx = v.x, vy = v.y, vz = v.z;
  // t = 2 * cross(q.xyz, v)
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  // v' = v + w * t + cross(q.xyz, t)
  return set(out,
    vx + qw * tx + (qy * tz - qz * ty),
    vy + qw * ty + (qz * tx - qx * tz),
    vz + qw * tz + (qx * ty - qy * tx));
}

// ---------------------------------------------------------------------------------------------
// World shapes
// ---------------------------------------------------------------------------------------------
export interface FlatShapeOptions {
  /** Sea level along +Y, metres. Default 0. */
  seaLevel?: number;
  /** m/s^2. Default 9.80665. */
  surfaceGravity?: number;
}

/** Flat world: up +Y, east +X, north -Z, sea level at y = seaLevel. */
export function createFlatShape(opts: FlatShapeOptions = {}): WorldShape {
  const seaLevel = opts.seaLevel ?? 0;
  const g = opts.surfaceGravity ?? STANDARD_GRAVITY;
  return {
    kind: 'flat',
    seaLevel,
    surfaceGravity: g,
    up(_p, out) { return set(out, 0, 1, 0); },
    altitude(p) { return p.y - seaLevel; },
    atAltitude(p, altitude, out) { return set(out, p.x, seaLevel + altitude, p.z); },
    tangentFrame(_p, out: TangentFrame) {
      set(out.east, 1, 0, 0);
      set(out.north, 0, 0, -1);
      set(out.up, 0, 1, 0);
      return out;
    },
    gravity(_p, out) { return set(out, 0, -g, 0); },
  };
}

export interface RotatedFlatShape extends WorldShape {
  /** The rotation applied to the canonical flat frame. */
  readonly rotation: Readonly<Quat>;
  /** World point -> canonical flat coordinates (x = east, y = altitude + seaLevel, z = -north). */
  worldToLocal(p: Readonly<Vec3>, out: Vec3): Vec3;
  /** Canonical flat coordinates -> world point. */
  localToWorld(p: Readonly<Vec3>, out: Vec3): Vec3;
}

export interface RotatedFlatShapeOptions extends FlatShapeOptions {
  /** World-space point the rotation pivots about (the canonical origin). Default (0,0,0). */
  pivot?: Readonly<Vec3>;
}

/**
 * A flat world whose up is NOT +Y: the canonical flat frame (east +X, up +Y, north -Z) rotated by
 * `rotation` about `pivot`. Same physics as createFlatShape in local terms; used for acceptance check 8
 * (no hardcoded +Y up). `rotation` is a unit quaternion (see quatFromAxisAngle).
 */
export function createRotatedFlatShape(rotation: Readonly<Quat>, opts: RotatedFlatShapeOptions = {}): RotatedFlatShape {
  const n = Math.hypot(rotation.x, rotation.y, rotation.z, rotation.w) || 1;
  const q: Quat = { x: rotation.x / n, y: rotation.y / n, z: rotation.z / n, w: rotation.w / n };
  const qInv: Quat = { x: -q.x, y: -q.y, z: -q.z, w: q.w };
  const seaLevel = opts.seaLevel ?? 0;
  const g = opts.surfaceGravity ?? STANDARD_GRAVITY;
  const pivot = { x: opts.pivot?.x ?? 0, y: opts.pivot?.y ?? 0, z: opts.pivot?.z ?? 0 };
  const UP = rotateByQuat(q, { x: 0, y: 1, z: 0 }, vec3());
  const EAST = rotateByQuat(q, { x: 1, y: 0, z: 0 }, vec3());
  const NORTH = rotateByQuat(q, { x: 0, y: 0, z: -1 }, vec3());
  const tmp = vec3();
  const shape: RotatedFlatShape = {
    kind: 'flat',
    seaLevel,
    surfaceGravity: g,
    rotation: q,
    up(_p, out) { return set(out, UP.x, UP.y, UP.z); },
    altitude(p) {
      return (p.x - pivot.x) * UP.x + (p.y - pivot.y) * UP.y + (p.z - pivot.z) * UP.z - seaLevel;
    },
    atAltitude(p, altitude, out) {
      const d = altitude - shape.altitude(p);
      return set(out, p.x + UP.x * d, p.y + UP.y * d, p.z + UP.z * d);
    },
    tangentFrame(_p, out) {
      set(out.east, EAST.x, EAST.y, EAST.z);
      set(out.north, NORTH.x, NORTH.y, NORTH.z);
      set(out.up, UP.x, UP.y, UP.z);
      return out;
    },
    gravity(_p, out) { return set(out, -UP.x * g, -UP.y * g, -UP.z * g); },
    worldToLocal(p, out) {
      set(tmp, p.x - pivot.x, p.y - pivot.y, p.z - pivot.z);
      return rotateByQuat(qInv, tmp, out);
    },
    localToWorld(p, out) {
      rotateByQuat(q, p, out);
      return set(out, out.x + pivot.x, out.y + pivot.y, out.z + pivot.z);
    },
  };
  return shape;
}

// ---------------------------------------------------------------------------------------------
// Floating origin
// ---------------------------------------------------------------------------------------------
export interface StubFloatingOrigin extends FloatingOrigin {
  /** Move the origin to (x, y, z) world metres (float64) and notify listeners with the delta. */
  setOrigin(x: number, y: number, z: number): void;
  /** Shift the origin by (dx, dy, dz) metres. */
  shift(dx: number, dy: number, dz: number): void;
}

/** a mod period in [0, period), exact for any finite float64 (JS % is an exact fmod). */
export function wrapScalar(a: number, period: number): number {
  let r = a % period;            // exact, sign of a, |r| < period
  if (r < 0) r += period;        // may round up to exactly `period` when r is a tiny negative
  if (r >= period) r -= period;
  return r === 0 ? 0 : r;        // normalise -0
}

export function createFloatingOrigin(initial: Readonly<Vec3> = { x: 0, y: 0, z: 0 }): StubFloatingOrigin {
  const origin = vec3(initial.x, initial.y, initial.z);
  const delta = vec3();
  const listeners: ((delta: Readonly<Vec3>, origin: Readonly<Vec3>) => void)[] = [];
  const self: StubFloatingOrigin = {
    origin,
    toRender(world, out) { return set(out, world.x - origin.x, world.y - origin.y, world.z - origin.z); },
    toWorld(render, out) { return set(out, render.x + origin.x, render.y + origin.y, render.z + origin.z); },
    onShift(fn) {
      listeners.push(fn);
      return () => {
        const i = listeners.indexOf(fn);
        if (i >= 0) listeners.splice(i, 1);
      };
    },
    wrapped(period, out) {
      if (!(period > 0) || !Number.isFinite(period)) throw new RangeError('wrapped: period must be a positive finite number');
      return set(out, wrapScalar(origin.x, period), wrapScalar(origin.y, period), wrapScalar(origin.z, period));
    },
    setOrigin(x, y, z) {
      set(delta, x - origin.x, y - origin.y, z - origin.z);
      if (delta.x === 0 && delta.y === 0 && delta.z === 0) return;
      set(origin, x, y, z);
      for (let i = 0; i < listeners.length; i++) listeners[i](delta, origin);
    },
    shift(dx, dy, dz) { self.setOrigin(origin.x + dx, origin.y + dy, origin.z + dz); },
  };
  return self;
}

// ---------------------------------------------------------------------------------------------
// Wind
// ---------------------------------------------------------------------------------------------
export interface StubWind extends WindQuery {
  /** Wind speed m/s and direction in degrees: the direction the wind blows TOWARDS, measured
   *  counter-clockwise from east towards north in the local tangent plane (0 = towards east, 90 = towards north). */
  set(speed: number, dirDeg: number): void;
  readonly speed: number;
  readonly dirDeg: number;
}

/** Uniform wind (no gusts). Direction is in the tangent plane of `shape` (default flat). */
export function createStubWind(speed: number, dirDeg: number, shape: WorldShape = createFlatShape()): StubWind {
  let s = speed, d = dirDeg, c = Math.cos(dirDeg * Math.PI / 180), sn = Math.sin(dirDeg * Math.PI / 180);
  const frame: TangentFrame = { east: vec3(), north: vec3(), up: vec3() };
  return {
    get speed() { return s; },
    get dirDeg() { return d; },
    set(speed2, dirDeg2) {
      s = speed2; d = dirDeg2;
      c = Math.cos(dirDeg2 * Math.PI / 180); sn = Math.sin(dirDeg2 * Math.PI / 180);
    },
    at(p, out) {
      shape.tangentFrame(p, frame);
      const e = frame.east, n = frame.north;
      return set(out, s * (c * e.x + sn * n.x), s * (c * e.y + sn * n.y), s * (c * e.z + sn * n.z));
    },
  };
}

// ---------------------------------------------------------------------------------------------
// Environment + sky light
// ---------------------------------------------------------------------------------------------
export interface StubEnvironment extends EnvironmentState {
  time: number; cloudCover: number; rain: number;
  readonly sunDir: Vec3; readonly moonDir: Vec3; readonly windBase: Vec3;
}

/** Plain mutable EnvironmentState. Sun elevation/azimuth in degrees (azimuth CCW from east to north), flat frame. */
export function createStubEnvironment(opts: { sunElevationDeg?: number; sunAzimuthDeg?: number; cloudCover?: number; windBase?: Readonly<Vec3> } = {}): StubEnvironment {
  const env: StubEnvironment = {
    time: 0,
    sunDir: vec3(),
    moonDir: vec3(0, -1, 0),
    cloudCover: opts.cloudCover ?? 0.2,
    rain: 0,
    windBase: vec3(opts.windBase?.x ?? 0, opts.windBase?.y ?? 0, opts.windBase?.z ?? 0),
  };
  setSunDirection(env.sunDir, opts.sunElevationDeg ?? 38, opts.sunAzimuthDeg ?? 217);
  return env;
}

/** Writes the flat-frame direction towards the sun (east +X, north -Z, up +Y). */
export function setSunDirection(out: Vec3, elevationDeg: number, azimuthDeg: number): Vec3 {
  const el = elevationDeg * Math.PI / 180, az = azimuthDeg * Math.PI / 180;
  return set(out, Math.cos(el) * Math.cos(az), Math.sin(el), -Math.cos(el) * Math.sin(az));
}

export interface StubSkyLight extends SkyLight {
  sunIlluminance: number; skyIlluminance: number;
  sunColor: RGB; skyColor: RGB; fogColor: RGB;
  fogDensity: number; fogHeightFalloff: number; suggestedEV100: number; version: number;
  readonly skySH: Float32Array;
  /** Recompute a plausible clear/overcast sky for a sun elevation (degrees) and cloud cover 0..1; bumps version. */
  update(sunElevationDeg: number, cloudCover?: number): void;
}

/**
 * Rough physically-plausible daylight stand-in (lux, cd/m^2). skySH layout: 9 real SH coefficients
 * (Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22 with y = up), each as r,g,b (27 floats), radiance in cd/m^2.
 */
export function createStubSkyLight(sunElevationDeg = 38, cloudCover = 0.2): StubSkyLight {
  const sky: StubSkyLight = {
    sunIlluminance: 0, skyIlluminance: 0,
    sunColor: { r: 1, g: 1, b: 1 }, skyColor: { r: 0.6, g: 0.75, b: 1 },
    skySH: new Float32Array(27),
    fogDensity: 0.00005, fogHeightFalloff: 0.002, fogColor: { r: 0.6, g: 0.7, b: 0.8 },
    suggestedEV100: 14, version: 0,
    update(elevDeg, cover = cloudCover) {
      const s = Math.max(Math.sin(elevDeg * Math.PI / 180), 0);
      const clear = 1 - 0.85 * Math.min(Math.max(cover, 0), 1);
      // Direct sun on a surface facing the sun: ~110 klux at zenith, attenuated by air mass + clouds.
      const airMass = 1 / Math.max(s + 0.15 * Math.pow(elevDeg + 3.885, -1.253), 1e-3);
      sky.sunIlluminance = s > 0 ? 127500 * Math.pow(0.7, Math.pow(Math.min(airMass, 40), 0.678)) * clear : 0;
      sky.skyIlluminance = 2000 + 18000 * Math.sqrt(s) * (0.6 + 0.4 * clear);
      const warm = Math.min(Math.max((elevDeg - 4) / 40, 0), 1);
      const r = 1, g = 0.62 + 0.36 * warm, b = 0.32 + 0.6 * warm;
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      sky.sunColor.r = r / lum; sky.sunColor.g = g / lum; sky.sunColor.b = b / lum;
      const sr = 0.55, sg = 0.72 + 0.05 * warm, sb = 1;
      const sl = 0.2126 * sr + 0.7152 * sg + 0.0722 * sb;
      sky.skyColor.r = sr / sl; sky.skyColor.g = sg / sl; sky.skyColor.b = sb / sl;
      // Mean radiance from up-facing illuminance (E = pi * L for a uniform hemisphere), brighter at zenith.
      const L = sky.skyIlluminance / Math.PI;
      const c0 = L * 3.5449077; // sqrt(4 pi): constant L -> Y00 coefficient
      const c1 = L * 0.6 * 2.0466534; // sqrt(4 pi / 3): linear gradient towards +up
      sky.skySH.fill(0);
      const col = [sky.skyColor.r, sky.skyColor.g, sky.skyColor.b];
      for (let k = 0; k < 3; k++) {
        sky.skySH[0 * 3 + k] = c0 * col[k];
        sky.skySH[2 * 3 + k] = c1 * col[k]; // Y10 ~ y (up)
      }
      sky.fogColor.r = 0.6 * sky.skyColor.r; sky.fogColor.g = 0.6 * sky.skyColor.g; sky.fogColor.b = 0.6 * sky.skyColor.b;
      // EV100 that maps the dominant illuminance to mid grey (incident-light meter, C = 250).
      sky.suggestedEV100 = Math.log2(Math.max(sky.sunIlluminance * s + sky.skyIlluminance, 1) * 100 / 250);
      sky.version++;
    },
  };
  sky.update(sunElevationDeg, cloudCover);
  return sky;
}

// ---------------------------------------------------------------------------------------------
// Terrain (TERRAIN_QUERY + TERRAIN_GPU stand-ins)
// ---------------------------------------------------------------------------------------------
/** TERRAIN_QUERY from a height function (world point -> ground height above sea level along up, m). */
export function createStubTerrain(heightAt: (p: Readonly<Vec3>) => number): TerrainQuery {
  return { heightAt };
}

export interface StubTerrainGrid extends TerrainGpuHeight {
  /** Set by the host (three adapter: a DataTexture over `data`). */
  texture: unknown;
  readonly center: Vec3;
  readonly east: Vec3;
  readonly north: Vec3;
  readonly up: Vec3;
  readonly size: number;
  readonly resolution: number;
  version: number;
  /** R heights (m above sea level), row j = north, column i = east: texel (i, j) centre at
   *  uv = ((i + .5) / res, (j + .5) / res) with uv = (dot(p - center, east), dot(p - center, north)) / size + .5. */
  readonly data: Float32Array;
  /** Re-centre on `p` (world, e.g. the camera) when it is more than `recenter` x size from the centre (or on the first
   *  call): snaps the centre to the texel lattice, refills `data` from the query, bumps `version`, calls onChange.
   *  Returns true when refilled. Allocation-free. */
  update(p: Readonly<Vec3>): boolean;
}

/**
 * TERRAIN_GPU stand-in: a camera-centred height grid filled on the CPU from a TerrainQuery (spec 3 GpuGridSource uv
 * convention). The host owns the GPU texture (no three import here) and uploads `data` in onChange.
 */
export function createStubTerrainGrid(query: TerrainQuery, shape: WorldShape, opts: {
  size?: number; resolution?: number; recenter?: number; onChange?: (grid: StubTerrainGrid) => void;
} = {}): StubTerrainGrid {
  const size = opts.size ?? 256, res = opts.resolution ?? 256, recenter = opts.recenter ?? 0.125;
  const texel = size / res;
  const frame: TangentFrame = { east: vec3(), north: vec3(), up: vec3() };
  const foot = vec3(), q = vec3();
  let first = true;
  const grid: StubTerrainGrid = {
    texture: null,
    center: vec3(), east: vec3(1, 0, 0), north: vec3(0, 0, -1), up: vec3(0, 1, 0),
    size, resolution: res, version: 0,
    data: new Float32Array(res * res),
    update(p) {
      shape.atAltitude(p, 0, foot);
      const c = grid.center;
      const dx = foot.x - c.x, dy = foot.y - c.y, dz = foot.z - c.z;
      if (!first && dx * dx + dy * dy + dz * dz < (recenter * size) ** 2) return false;
      first = false;
      shape.tangentFrame(foot, frame);
      set(grid.east, frame.east.x, frame.east.y, frame.east.z);
      set(grid.north, frame.north.x, frame.north.y, frame.north.z);
      set(grid.up, frame.up.x, frame.up.y, frame.up.z);
      // Snap the centre to the texel lattice along east / north (world-anchored, so texels never swim).
      const e = Math.round((foot.x * frame.east.x + foot.y * frame.east.y + foot.z * frame.east.z) / texel) * texel;
      const n = Math.round((foot.x * frame.north.x + foot.y * frame.north.y + foot.z * frame.north.z) / texel) * texel;
      const u = foot.x * frame.up.x + foot.y * frame.up.y + foot.z * frame.up.z;
      set(c, e * frame.east.x + n * frame.north.x + u * frame.up.x,
        e * frame.east.y + n * frame.north.y + u * frame.up.y,
        e * frame.east.z + n * frame.north.z + u * frame.up.z);
      const d = grid.data;
      for (let j = 0; j < res; j++) {
        const nn = ((j + 0.5) / res - 0.5) * size;
        for (let i = 0; i < res; i++) {
          const ee = ((i + 0.5) / res - 0.5) * size;
          set(q, c.x + ee * frame.east.x + nn * frame.north.x, c.y + ee * frame.east.y + nn * frame.north.y,
            c.z + ee * frame.east.z + nn * frame.north.z);
          d[j * res + i] = query.heightAt(q);
        }
      }
      grid.version++;
      opts.onChange?.(grid);
      return true;
    },
  };
  return grid;
}

// ---------------------------------------------------------------------------------------------
// World + fixed-step loop
// ---------------------------------------------------------------------------------------------
export const TICK_PHASES: readonly TickPhase[] = ['time', 'weather', 'wind', 'simulate', 'gameplay', 'render'];

export interface StubWorldOptions {
  shape?: WorldShape;
  origin?: FloatingOrigin;
  /** Fixed step, seconds. Default 1/60. */
  fixedDt?: number;
  /** Max fixed steps per advance() (spiral-of-death guard). Default 8. */
  maxSubSteps?: number;
}

export interface StubWorld extends World {
  readonly shape: WorldShape;
  readonly origin: FloatingOrigin;
  readonly time: number;
  readonly fixedDt: number;
  readonly step: number;
  /** Install a system (calls install(world)). */
  use(system: WorldSystem): void;
  /** Calls start(world) on every installed system, once. */
  start(): void;
  /**
   * Advance by `realDt` real seconds: runs whole fixed steps (each phase in TICK_PHASES order, for the
   * systems that listed that phase), then calls frame(info) on every system. Returns the FrameInfo
   * (the same object every call — do not keep it).
   */
  advance(realDt: number): FrameInfo;
  /** Run exactly one fixed step (no frame call). */
  stepOnce(): void;
  /** Jump to simulation time `t` seconds (deterministic shots): no ticks, interpolatedTime = t. */
  setTime(t: number): void;
  /** Call frame(info) on every system without advancing time (paused). */
  frameOnly(realDt: number): FrameInfo;
  emit(type: string, data?: unknown): void;
  dispose(): void;
}

export function createStubWorld(opts: StubWorldOptions = {}): StubWorld {
  const shape = opts.shape ?? createFlatShape();
  const origin = opts.origin ?? createFloatingOrigin();
  const fixedDt = opts.fixedDt ?? 1 / 60;
  const maxSubSteps = opts.maxSubSteps ?? 8;
  const providers = new Map<string, unknown>();
  const listeners = new Map<string, ((e: { type: string; data?: unknown }) => void)[]>();
  const systems: WorldSystem[] = [];
  const phaseSystems: WorldSystem[][] = TICK_PHASES.map(() => []);
  let started = false;
  let time = 0;        // s, simulation time at the last completed fixed step
  let step = 0;
  let accumulator = 0; // s
  let frame = 0;
  // Reused info objects (no allocation per tick / frame).
  const tick = { dt: fixedDt, time: 0, step: 0 };
  const info = { frameDt: 0, time: 0, alpha: 1, interpolatedTime: 0, frame: 0 };
  const event = { type: '', data: undefined as unknown };

  function runStep() {
    time += fixedDt;
    step++;
    tick.time = time;
    tick.step = step;
    for (let p = 0; p < TICK_PHASES.length; p++) {
      const list = phaseSystems[p];
      for (let i = 0; i < list.length; i++) list[i].tick!(TICK_PHASES[p], tick as TickInfo);
    }
  }
  function runFrame(realDt: number, alpha: number): FrameInfo {
    frame++;
    info.frameDt = realDt;
    info.time = time;
    info.alpha = alpha;
    // The state is at `time`; the previous step was at time - fixedDt. Draw between them.
    info.interpolatedTime = time - fixedDt + alpha * fixedDt;
    info.frame = frame;
    for (let i = 0; i < systems.length; i++) systems[i].frame?.(info);
    return info;
  }

  const world: StubWorld = {
    shape,
    origin,
    get time() { return time; },
    get step() { return step; },
    fixedDt,
    provide(key, impl) {
      if (providers.has(key.id)) throw new Error(`Provider already registered: ${key.id}`);
      providers.set(key.id, impl);
    },
    require<T>(key: ProviderKey<T>): T {
      if (!providers.has(key.id)) throw new Error(`Missing required provider: ${key.id}`);
      return providers.get(key.id) as T;
    },
    optional<T>(key: ProviderKey<T>): T | undefined {
      return providers.get(key.id) as T | undefined;
    },
    on(type, fn) {
      let list = listeners.get(type);
      if (!list) listeners.set(type, (list = []));
      list.push(fn);
      return () => {
        const l = listeners.get(type);
        const i = l ? l.indexOf(fn) : -1;
        if (l && i >= 0) l.splice(i, 1);
      };
    },
    emit(type, data) {
      const list = listeners.get(type);
      if (!list) return;
      event.type = type;
      event.data = data;
      for (let i = 0; i < list.length; i++) list[i](event);
    },
    use(system) {
      systems.push(system);
      if (system.tick && system.phases) {
        for (const ph of system.phases) {
          const idx = TICK_PHASES.indexOf(ph);
          if (idx < 0) throw new Error(`Unknown tick phase ${ph}`);
          phaseSystems[idx].push(system);
        }
      }
      system.install(world);
      if (started) system.start?.(world);
    },
    start() {
      if (started) return;
      started = true;
      for (const s of systems) {
        for (const k of s.requires ?? []) world.require(k);
        s.start?.(world);
      }
    },
    advance(realDt) {
      if (!(realDt > 0) || !Number.isFinite(realDt)) return runFrame(0, info.alpha);
      accumulator += realDt;
      let n = 0;
      while (accumulator >= fixedDt && n < maxSubSteps) {
        accumulator -= fixedDt;
        runStep();
        n++;
      }
      if (n === maxSubSteps && accumulator >= fixedDt) accumulator = accumulator % fixedDt; // drop backlog
      return runFrame(realDt, accumulator / fixedDt);
    },
    stepOnce() { runStep(); },
    setTime(t) {
      time = t;
      accumulator = 0;
      tick.time = t;
      info.time = t;
      info.alpha = 1;
      info.interpolatedTime = t;
    },
    frameOnly(realDt) { return runFrame(realDt, 1); },
    dispose() {
      for (let i = systems.length - 1; i >= 0; i--) systems[i].dispose?.();
      systems.length = 0;
      for (const l of phaseSystems) l.length = 0;
      providers.clear();
      listeners.clear();
    },
  };
  return world;
}
