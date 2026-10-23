# Ocean WebGPU port: rules for the side repo

**Goal:** port the FFT ocean prototype (`reference/ocean`, a fork of [noxellab/nagi-ocean-sim](https://github.com/noxellab/nagi-ocean-sim), MIT) from WebGL2 to WebGPU in a separate repo, so it drops into the DreamP engine as the `world/ocean` module (package `@dreamp/world-ocean`) without a rewrite.

This file is self-contained: the side repo does not have the engine. The contracts you code against are copied below. Treat them as fixed: if one needs to change, note it in the handoff and the engine side will make the change.

Written 2026-10-23 against engine contract `world/core ^0.10` and three.js r186.

---

## 1. Hard rules (breaking any of these means a rewrite on import)

| # | Rule | What it means in practice |
|---|---|---|
| 1 | **WebGPU only, TSL only** | `WebGPURenderer` from `three/webgpu`, nodes from `three/tsl`. No GLSL, no `ShaderMaterial` / `RawShaderMaterial`, no `WebGLRenderer`, no WebGL2 fallback. If `navigator.gpu` is missing, show a clear "WebGPU required" message. The FFT runs as **compute** (`Fn(...)().compute(n)` + `renderer.compute`), not as fragment ping-pong passes. |
| 2 | **three.js r186 exactly** | `"three": "0.186.x"` (the prototype is on 0.180). Do not use APIs added after r186. |
| 3 | **Render-agnostic core** | The spectrum, cascade setup, CPU FFT, height / velocity queries and presets live in files that **never import three**. three.js code lives only in `three/` (and `demo/`). The core must run in Node. |
| 4 | **Never assume +Y is up** | Get "up" from the world shape: `shape.up(p, out)`, `shape.tangentFrame(p, out)`, `shape.altitude(p)`. Waves displace along the local up and the local east / north tangent frame. Flat must work now; the code must not prevent a sphere (planet) later. Wave heights are **metres above sea level along up**. |
| 5 | **Floating origin: render camera-relative** | World positions are float64 JS numbers; the GPU only sees camera-relative float32. Never put a raw world position or the raw origin in a shader. For periodic patterns (the FFT tiles are periodic: 1024 m, 96 m, 9 m), pass `origin.wrapped(period, out)` as a uniform and sample with `renderPos + wrapped`. The surface must look identical 100 km from the origin. |
| 6 | **Fixed timestep, explicit time** | Never read `performance.now()` / `Date.now()` / `clock.getElapsedTime()` in the simulation. Time comes in as a number (seconds of simulation time). The same `time` must give the same waves (deterministic, seeded). |
| 7 | **No allocation per frame** | No `new`, no array / object literals, no closures, no `.clone()` in `tick` / `frame` / per-frame GPU code / query calls. Preallocate; use `out` parameters. Queries (`heightAt`, `velocityAt`) must be allocation-free. |
| 8 | **CPU and GPU agree** | What gameplay reads (`heightAt`, `velocityAt`) must match what is drawn. Either run the same spectrum + FFT on the CPU (prototype has a 64² CPU FFT), or read a small GPU result back with **one frame of delay, never a stall**. Pick one, document it, and measure the error (target: < 1–2 % of wave height, or state the number). |
| 9 | **Physical units, HDR** | Light comes in lux and cd/m² (see `SkyLight` below). Output linear HDR radiance; no tonemapping or sRGB encoding in the water shader. Multiply the radiance you write by pre-exposure (§5). |
| 10 | **Motion vectors** | The water vertices move, so the material must provide previous-frame positions (§5). Without this, TAA and motion blur will smear the ocean. |
| 11 | **Systems never import each other** | Wind, terrain and sky come **only** through the contracts in §3. Do not import a wind / sky module. In the side repo, write tiny stubs that implement the contracts. |
| 12 | **Licences** | Keep `LICENSE` and `THIRD_PARTY_NOTICES.txt` from nagi-ocean-sim with the ported code. Whale model is CC BY 4.0: attribution must show if it ships in a demo. Do **not** copy code from the Minecraft shader packs (Pyvtron, Rethinking Voxels); ideas only, written fresh from papers. |

---

## 2. Module shape to produce

Lay the side repo out so the folder can be copied to `packages/world/ocean/` as is:

```
ocean/
  package.json        name "@dreamp/world-ocean", "type": "module", exports "." -> index.ts, "./three" -> three/index.ts
  module.json         manifest (template below)
  index.ts            public entry: createOcean(...), types, presets; NO three import
  types.ts            config + param defs (min, max, unit, label for every tunable)
  spectrum.ts         JONSWAP + directional spreading, cascades, seeding (pure functions)
  cpu-fft.ts          CPU FFT for the WATER_QUERY (if that is the chosen agreement path)
  presets/            calm.ts, breeze.ts, windy.ts, storm.ts (+ index.ts)
  three/              index.ts (createOceanGpu), fft.ts (TSL compute), material.ts (TSL surface), passes.ts
  demo/               index.html + demo.ts (standalone page, free camera, preset buttons, ?parity, ?shot)
  tests/              *.test.ts, run with: node --experimental-strip-types --test tests/
  README.md           agent quick-start first (see §8)
  TUNING.md           every parameter + frame cost per tier (see §8)
  LICENSE, THIRD_PARTY_NOTICES.txt
```

Code style:

- TypeScript, **erasable syntax only** (it runs with `node --experimental-strip-types`): no `enum`, no `namespace`, no constructor parameter properties, no decorators. Use `import type` for types and include the `.ts` extension in relative imports.
- Plain structs in public APIs (`{ x, y, z }`), no three.js types in `index.ts` / `types.ts`.
- Units in names or comments everywhere (metres, m/s, seconds, radians or degrees).
- ES modules, no bundler-only features. In the engine, imports of other modules go by package name (`@dreamp/world-core`), never a relative path into another package.
- Assets (foam texture, etc.): reference them by logical ID through a small loader hook, not hard paths. They become an asset pack (`assets/<pack>/pack.json`) on import. Keep large binaries out of the core.

`module.json` template:

```json
{
  "id": "world/ocean",
  "package": "@dreamp/world-ocean",
  "name": "Ocean",
  "category": "world",
  "version": "0.1.0",
  "status": "experimental",
  "summary": "FFT ocean: JONSWAP spectrum, cascades, foam; CPU water query matching the GPU surface; calm to storm presets.",
  "entry": "index.ts",
  "three": "three/index.ts",
  "dependsOn": ["world/core"],
  "worksWith": ["rendering/render-graph"],
  "provides": ["WATER_QUERY"],
  "requires": [],
  "optional": ["WIND_QUERY", "ENVIRONMENT", "TERRAIN_QUERY", "TERRAIN_GPU", "SKY_LIGHT"],
  "contract": "^0.10",
  "tickPhases": ["simulate"],
  "targets": ["desktop"],
  "renderer": "webgpu",
  "platform": ["any"],
  "devDependencies": []
}
```

(`WIND_QUERY` is optional so the ocean still runs, from its preset, when no wind module is installed. The plan lists it as the main input: when present, wind drives the spectrum.)

---

## 3. Contracts (copy into the side repo as `stubs/world-core.ts`)

These are the engine's real types, trimmed to what the ocean touches. Code against them exactly.

```ts
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
```

Contract rules: plain structs, `out` parameters, no allocation per call, explicit time, units in comments.

**Tick order in the engine** (one fixed step): time → weather → wind → **simulate (ocean here)** → gameplay (reads `WATER_QUERY`: buoyancy) → render adapters copy state to the GPU in `frame()`.

The public entry should look like this (mirrors the wind module):

```ts
import { createOcean } from '@dreamp/world-ocean';
import { createOceanGpu } from '@dreamp/world-ocean/three';
const ocean = createOcean({ preset: 'breeze', seed: 1 });     // WorldSystem, provides WATER_QUERY
world.use(ocean); world.start();
const gpu = createOceanGpu({ ocean, renderer, scene, origin: world.origin, tier: 'high' });
graph.add(gpu.fftPass);                                       // render-graph pass(es), see §5
const h = world.require(WATER_QUERY).heightAt({ x: 10, y: 0, z: 5 }, world.time);
ocean.setPreset('storm');
```

---

## 4. What to port, what to drop

| Prototype part | Keep | Change |
|---|---|---|
| `spectral-ocean.js` spectrum | JONSWAP + directional spreading, wind speed / direction, 3 cascades (1024 m, 96 m, 9 m), seeded, looping | Pure functions in `spectrum.ts` with tests. Wind comes from `WIND_QUERY` (sample at 10 m above sea level near the camera; fall back to the preset). Presets calm / breeze / windy / storm. |
| GPU FFT | 256² per cascade, choppy displacement, foam from surface compression (Jacobian) | WebGPU **compute** in TSL (storage textures / buffers). Tiers scale the grid (64 → 512) and the number of cascades. |
| Capillary ripples, CPU FFT (64²) | Small ripples, CPU fallback | The CPU FFT feeds `WATER_QUERY` so gameplay matches the drawn waves. |
| Buoyancy, float readback probes | GPU buoyancy, readback | Becomes the "floating object matches the drawn waves" test (§7). Readback must be async (one frame late), never `await` on the render path. |
| Refraction, environment lighting | Working shading | Redone in TSL. Read scene colour / depth through render-graph targets; lighting from `SKY_LIGHT` (+ `ENVIRONMENT.sunDir`). |
| Sky (`atmosphere.js`, `photo-sky.js`), prefiltered environment | — | **Not part of the ocean.** The engine has `world/sky`. Keep a photo sky only in the demo, as a stand-in `SkyLight` + env map. |
| Whale, seabed, HUD, i18n, free camera, FPS counter | — | Demo only (or drop). The engine has its own camera, menu and settings modules. Seabed later reads `TERRAIN_QUERY`. |

---

## 5. Rendering rules (engine render graph)

The engine draws everything through a render graph (`@dreamp/render-graph/three`). You do not have it in the side repo, so **structure the GPU work as passes** that are easy to register later. Code each piece as an object of this shape:

```ts
// Frame pass: runs every frame, no allocation. Compute dispatches go here.
const fft = {
  desc: { id: 'ocean.fft', module: 'world/ocean', stage: 'compute', reads: [], writes: ['ocean.displacement'] },
  execute(ctx /* { renderer, dt, frame, width, height, tier } */) { ctx.renderer.compute(fftNode); },
};
```

- Stages, in order: `compute, shadow, scene, ao, lighting, atmosphere, volumetrics, transparent, temporal, motionblur, dof, bloom, exposure, tonemap, ldr, overlay`. The FFT and foam go in `compute`. The water surface is a normal mesh drawn by the scene pass (stage `scene`), or in `transparent` if it needs refraction of `scene.color` / `scene.depth`.
- Shared targets you may read: `scene.color` (HDR linear RGBA16F, radiance × pre-exposure), `scene.depth`, `scene.normal` (view space, -1..1), `scene.velocity` (NDC, current − previous). Your own targets are named `ocean.<name>`.
- **Declare every read and write.** A pass that writes nothing someone reads gets culled.
- Main loop: in the engine you never call `renderer.render` yourself; the graph does. In the side repo, keep all `renderer.render` / `renderer.compute` calls inside these pass objects so the demo loop is just "run passes in stage order".
- **Pre-exposure.** Radiance in physical units overflows RGBA16F. A mesh drawn by the scene pass gets this for free. Anything you composite yourself into `scene.color` (spray, a separate refraction composite) must multiply the added radiance by the pre-exposure uniform. Expose it as a `preExposure` TSL uniform input (default 1) so the engine can plug in its own.
- **Motion vectors for displaced vertices.** The engine API is `graph.setMaterialPreviousPosition(material, previousLocalPositionNode)`. So the water material must be able to compute **last frame's displaced local position** (sample the displacement at `time − frameDt`, or keep last frame's displacement texture). Build that node and expose it (e.g. `gpu.previousPositionNode`). In the side repo, verify it by rendering a velocity debug view.
- **Floating origin shifts.** Subscribe with `origin.onShift(...)`: move the water mesh and update the `wrapped` uniforms; the waves must not jump.
- **Tiers** are `low | medium | high | ultra`. Build **low first** (smallest grid / fewest cascades that still looks like an ocean), then add tiers. Every tier is an FFT ocean (no Gerstner-only tier). Changing tier must not stall: create compute pipelines with `renderer.compileComputeAsync` / async pipelines, and keep the old tier on screen until the new one is ready.
- **Mesh / LOD:** camera-centred grid (projected grid, concentric rings or quadtree), snapped so it never swims, drawn camera-relative. The engine will later have a shared `world/lod-tree`; keep the LOD logic isolated so it can be swapped.
- Shader precision: no world-space coordinates in float32. Use `renderPos + wrapped(period)` for the periodic FFT tiles.
- Don't modify three.js global prototypes.

---

## 6. Performance rules

- No allocation per frame anywhere (CPU sim, queries, GPU adapter, readback handling).
- `heightAt` / `velocityAt`: allocation-free and fast (aim ≥ 1 M calls/s in Node; state the number).
- Generation / heavy precompute (spectrum init, LUTs) may run once at start or on preset / tier change, never per frame. Big precompute can go to a worker (the engine pages are cross-origin isolated, so `SharedArrayBuffer` is available).
- Measure and write down, per tier: GPU ms (timestamp queries: `new WebGPURenderer({ trackTimestamp: true })`), CPU ms per frame, memory (textures + buffers), at 1080p. Wrap CPU work in named markers so it maps to the engine's `perf.begin('world/ocean', 'sim')` / `perf.end()` (accept an optional `perf: { begin(module, phase), end() }` sink in the options).

---

## 7. Acceptance checks (what "done" means for import)

1. **Runs on WebGPU** with three r186; no WebGL, GLSL or `ShaderMaterial` left (grep for them).
2. **Core runs in Node**: `node --experimental-strip-types --test tests/` passes; no three import outside `three/` and `demo/`.
3. **Determinism**: same seed + same time → identical heights (test).
4. **Presets**: calm → storm visibly and measurably change significant wave height (test checks Hs per preset against your target table).
5. **Wind coupling**: with a stub `WIND_QUERY`, changing the wind speed / direction changes the spectrum (test), and blends smoothly (no pop).
6. **CPU = GPU**: a floating test object driven by `WATER_QUERY.heightAt` sits on the drawn surface. Demo `?parity` reads back GPU heights at N points and compares with the CPU (report max error; target < 2 % of Hs).
7. **Far from origin**: with the origin moved to (100 km, 0, 100 km), the surface looks identical and parity still passes.
8. **Up is not hardcoded**: grep finds no `y` used as height in the core except through the shape. A test runs the queries with a stub shape whose up is not +Y (a rotated flat frame) and gets the same result in local terms.
9. **Motion vectors**: velocity debug view shows wave motion; a still camera over calm water shows small, correct vectors, not zero or noise.
10. **No per-frame allocation**: a test calls the queries / tick in a loop and checks heap growth stays flat.
11. **Cost table** filled in TUNING.md for each tier you built.
12. Licence files present.

---

## 8. Docs to write (the engine requires them)

**README.md**, in this order:

- One-line summary.
- **Agent quick-start**: a 5-line code block (import, create, `world.use`, wire the GPU adapter, verify), like the snippet in §3.
- Bullets: Provides / Requires / Optional keys, Targets (desktop), Renderer (WebGPU, three r186, TSL); how to verify (test command, demo URL and query flags); link to TUNING.md.
- Then: how the model works, API reference, config reference, which CPU / GPU agreement path you chose and its measured error, events (if any), testing, technique and references (papers: Tessendorf "Simulating Ocean Water", JONSWAP / Hasselmann 1973, etc.), known gaps.

**TUNING.md**: every tunable as a table row: param, default, unit, effect, safe range and what breaks outside it, value per tier (low / medium / high / ultra). Then presets table (wind speed, fetch, Hs, peak period, reason), frame cost per tier (GPU ms at 1080p, CPU ms, memory, which GPU it was measured on, date), linked parameters (what must change together), known gaps, what to test after a change. When a value changes on purpose, change the number **and** the TUNING row (value, reason, date) together.

---

## 9. Handoff to the engine repo

When it's ready, bring over:

- The module folder (layout from §2), with `stubs/` **removed**: imports switch from `./stubs/world-core.ts` to `@dreamp/world-core`. Keep that import in one place (e.g. re-export it from `types.ts`) so the switch is a one-line change.
- A short `HANDOFF.md`: what was built, which tiers exist, measured costs, the parity error, anything in the contracts you needed changed (the engine side changes `world/core`; don't fork the types), and open issues.
- Asset files listed with their source and licence (they become an asset pack).

On the engine side we will then: drop it into `packages/world/ocean/`, wire the real render-graph passes and `setMaterialPreviousPosition`, run `npm run modules:catalog` and `npm run check`, and add it to the World Demo. That's milestone 3: one storm preset raises the waves, the wind drives them, and a floating object matches the drawn waves.
