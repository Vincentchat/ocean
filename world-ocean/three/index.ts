// PLACEHOLDER three adapter (infra scaffold). The real WebGPU/TSL ocean replaces the material + fft pass
// (fft.ts, material.ts, passes.ts ...). Keep the createOceanGpu signature (spec Â§3/Â§5).
//
// What it does now: the real camera-centred LOD mesh (three/lod.ts: snapped clipmap, geomorph, horizon skirt,
// camera-relative, floating-origin tile anchors) displaced by a TEMPORARY test pattern â€” a sum of six
// Gerstner-like waves with integer wave numbers per tile of 1024 / 96 / 9 m, sampled through
// lod.nodes.tileCoord(P), so it is periodic in WORLD space exactly like the FFT tiles will be. The FFT agent
// replaces `testWaves` with the FFT displacement and keeps the lod wiring.
// Shading: Fresnel env reflection + Gaussian distance haze (reference Â§2.11), radiance x preExposure, no tonemap.
//
// Debug views (opts.debug, comma separated; demo ?debug=...):
//   lod        level colours (hue per level, darker = geomorph), lit by facet normals, wireframe overlay,
//              magenta background (any magenta below the horizon = crack / gap), back faces culled.
//   facets     grey facet shading of the displaced geometry (shows vertex positions; for swim tests), no haze.
//   nomesh     hide the ocean (perf baseline).
//   nosnap     disable LOD snapping (control for the swim test: the surface crawls).
//   lodoff:u:v offset the LOD's camera foot by (u east, v south) metres (swim test without moving the view).
import { DoubleSide, FrontSide, Mesh, MeshBasicNodeMaterial } from 'three/webgpu';
import {
  cameraPosition, clamp, cos, cross, floor, dFdx, dFdy, dot, exp, float, max, mix, modelWorldMatrix,
  normalize, pmremTexture, pow, positionWorld, reflect, sin, smoothstep, uniform, varying, vec2, vec3, vec4,
} from 'three/tsl';
import type { FloatingOrigin, SkyLight, Vec3, WorldShape } from '../stubs/world-core.ts';
import type { Pass, PassContext } from './pass-types.ts';
import { createOceanLod, type OceanLod } from './lod.ts';

export type { Pass, PassContext, PassDesc, Stage } from './pass-types.ts';
export { STAGES } from './pass-types.ts';
export { createOceanLod } from './lod.ts';
export type { OceanLod, OceanLodNodes, OceanLodOptions, OceanLodStats } from './lod.ts';

export type OceanTier = 'low' | 'medium' | 'high' | 'ultra';

export interface PerfSink {
  begin(module: string, phase: string): void;
  end(): void;
}

export interface OceanGpuOptions {
  /** The core WorldSystem from createOcean() (may be null while the core is not built). */
  ocean: unknown;
  /** three/webgpu WebGPURenderer (must be the WebGPU backend). */
  renderer: any;
  scene: any;
  camera: any;
  origin: FloatingOrigin;
  /** World shape (up / tangent frame). Default: flat, sea level 0 (stand-in until the engine passes world.shape). */
  shape?: WorldShape;
  tier?: OceanTier;
  skyLight?: SkyLight;
  /** PMREM (CubeUV) env texture, linear HDR. */
  envMap?: any;
  /** Multiply env samples by this to get cd/m^2 (demo: REF_NITS). Default 1. */
  envIntensity?: number;
  /** TSL uniform node (or number) multiplied into the written radiance. Default 1. */
  preExposure?: any;
  perf?: PerfSink;
  /** Demo debug view name (?debug=). */
  debug?: string;
}

export interface OceanGpu {
  readonly passes: Pass[];
  readonly fftPass: Pass;
  readonly mesh: any;
  readonly material: any;
  /** The LOD (mesh, TSL nodes, stats). */
  readonly lod: OceanLod;
  /** Last frame's displaced local position node (motion vectors). */
  readonly previousPositionNode: any;
  setTier(t: OceanTier): void;
  dispose(): void;
}

/** Minimal flat shape (east +X, north -Z, up +Y, sea level 0) used only when no shape is passed. */
function defaultFlatShape(): WorldShape {
  const set = (o: Vec3, x: number, y: number, z: number) => { o.x = x; o.y = y; o.z = z; return o; };
  return {
    kind: 'flat', seaLevel: 0, surfaceGravity: 9.80665,
    up: (_p, o) => set(o, 0, 1, 0),
    altitude: (p) => p.y,
    atAltitude: (p, a, o) => set(o, p.x, a, p.z),
    tangentFrame: (_p, o) => { set(o.east, 1, 0, 0); set(o.north, 0, 0, -1); set(o.up, 0, 1, 0); return o; },
    gravity: (_p, o) => set(o, 0, -9.80665, 0),
  };
}

// TEMPORARY test waves: integer wave numbers (kx, kz) per tile so each is exactly periodic over its tile.
const TEST_WAVES = [
  { P: 1024, kx: 20, kz: 10, A: 0.55, phase: 0.3 },  // lambda 45.8 m (reference peak)
  { P: 1024, kx: 7, kz: -3, A: 0.7, phase: 1.1 },    // 134 m swell
  { P: 96, kx: 5, kz: 3, A: 0.13, phase: 2.0 },      // 16.5 m
  { P: 96, kx: -2, kz: 7, A: 0.09, phase: 0.7 },     // 13.2 m
  { P: 9, kx: 3, kz: 4, A: 0.028, phase: 1.7 },      // 1.8 m
  { P: 9, kx: -4, kz: 1, A: 0.018, phase: 0.2 },     // 2.2 m
];
const G = 9.80665;
const CHOP = 0.9;

export function createOceanGpu(opts: OceanGpuOptions): OceanGpu {
  const { origin, camera, scene } = opts;
  const shape = opts.shape ?? defaultFlatShape();
  const preExposure = typeof opts.preExposure === 'object' && opts.preExposure ? opts.preExposure : uniform(opts.preExposure ?? 1);
  const envIntensity = uniform(opts.envIntensity ?? 1);
  const dbg = new Set((opts.debug ?? '').split(',').map((s) => s.trim()).filter(Boolean));
  let lodOffU = 0, lodOffV = 0;
  for (const d of dbg) if (d.startsWith('lodoff:')) { const [, u, v] = d.split(':'); lodOffU = Number(u) || 0; lodOffV = Number(v) || 0; }

  const lod = createOceanLod({ tier: opts.tier ?? 'high', shape, origin });
  lod.debug.noSnap = dbg.has('nosnap');
  lod.debug.footOffsetU = lodOffU;
  lod.debug.footOffsetV = lodOffV;
  const n = lod.nodes;

  // ------------------------------------------------------------------ temporary test displacement
  const phaseNow = TEST_WAVES.map(() => uniform(0));
  const phasePrev = TEST_WAVES.map(() => uniform(0));
  const fade = float(1).sub(smoothstep(200, 1200, n.distanceToCamera)); // reference farFade
  function testWaves(phases: any[], tileOf: (P: number) => any, cell: any, fadeNode: any) {
    let disp: any = vec3(0, 0, 0);
    let slope: any = vec2(0, 0);
    TEST_WAVES.forEach((w, i) => {
      const kx = (2 * Math.PI * w.kx) / w.P, kz = (2 * Math.PI * w.kz) / w.P;
      const k = Math.hypot(kx, kz), lambda = (2 * Math.PI) / k;
      const theta = dot(tileOf(w.P), vec2(kx, kz)).sub(phases[i]);
      // Footprint LOD: drop waves the local grid cannot resolve (cell > lambda / 4).
      const att = float(1).sub(smoothstep(lambda / 6, lambda / 3, cell)).mul(fadeNode).mul(w.A);
      const c = cos(theta), s = sin(theta);
      disp = disp.add(vec3(s.mul(-CHOP * kx / k), c, s.mul(-CHOP * kz / k)).mul(att));
      slope = slope.add(vec2(kx, kz).mul(s.negate()).mul(att));
    });
    return { disp, slope };
  }
  const vtx = testWaves(phaseNow, (P) => n.tileCoord(P), n.cellSize, fade);
  const vtxPrev = testWaves(phasePrev, (P) => n.tileCoord(P), n.cellSize, fade);
  const positionNode = n.localPos.add(vtx.disp);
  const previousPositionNode = n.previousLocalPos.add(vtxPrev.disp);

  // ------------------------------------------------------------------ materials
  const upW = normalize(modelWorldMatrix.mul(vec4(0, 1, 0, 0)).xyz);
  const toFrag = positionWorld.sub(cameraPosition);
  const dist = toFrag.length();
  const V = toFrag.div(max(dist, 1e-4));
  const envAt = (dir: any, rough: any) => (opts.envMap ? pmremTexture(opts.envMap, dir, rough).rgb : vec3(0.3, 0.45, 0.6));
  const facetN = (() => {
    const nf = normalize(cross(dFdx(positionWorld), dFdy(positionWorld)));
    return nf.mul(dot(nf, V).greaterThan(0).select(float(-1), float(1))); // face the viewer
  })();

  let material: any;
  let overlay: any = null;
  let savedBackground: any;
  const isLod = dbg.has('lod'), isFacets = dbg.has('facets');
  if (isLod) {
    material = new MeshBasicNodeMaterial({ side: FrontSide });
    material.name = 'ocean.debug.lod';
    const lvl = varying(n.level);
    const m = varying(n.morphFactor);
    // 6-colour palette without magenta (magenta = background = crack detector).
    const PALETTE = [[0.9, 0.35, 0.2], [0.9, 0.8, 0.2], [0.3, 0.8, 0.3], [0.2, 0.75, 0.8], [0.25, 0.4, 0.95], [0.85, 0.85, 0.85]];
    const idx = lvl.sub(floor(lvl.div(6)).mul(6));
    let pal: any = vec3(...PALETTE[5]);
    for (let i = 4; i >= 0; i--) pal = idx.lessThan(i + 0.5).select(vec3(...PALETTE[i]), pal);
    const base = lvl.lessThan(-0.5).select(vec3(0.35, 0.35, 0.35), pal);
    const L = normalize(upW.add(vec3(0.4, 0, 0.25)));
    const lit = float(0.45).add(clamp(dot(facetN, L), 0, 1).mul(0.55));
    material.colorNode = base.mul(float(1).sub(m.mul(0.45))).mul(lit).mul(envIntensity).mul(0.6).mul(preExposure);
    material.polygonOffset = true; material.polygonOffsetFactor = 1; material.polygonOffsetUnits = 1;
    const wire = new MeshBasicNodeMaterial({ side: FrontSide, wireframe: true });
    wire.name = 'ocean.debug.lod.wire';
    // Lift the lines towards the camera (local camera = (0, h, 0)) so they win the depth test over the fill.
    const toCam = n.cameraLocal.sub(positionNode);
    wire.positionNode = positionNode.add(normalize(toCam).mul(toCam.length().mul(0.004)));
    wire.colorNode = base.mul(0.18).mul(envIntensity).mul(preExposure);
    overlay = new Mesh(lod.mesh.geometry, wire);
    overlay.frustumCulled = false;
    overlay.name = 'ocean.debug.lod.wire';
    savedBackground = scene.backgroundNode;
    scene.backgroundNode = vec3(1, 0, 1);
  } else if (isFacets) {
    material = new MeshBasicNodeMaterial({ side: DoubleSide });
    material.name = 'ocean.debug.facets';
    // Grazing light so small facet tilts give strong contrast (swim tests diff these images).
    const L = normalize(upW.mul(0.25).add(vec3(0.8, 0, 0.55)));
    const g = float(0.03).add(pow(clamp(dot(facetN, L), 0, 1), 2).mul(1.6));
    material.colorNode = vec3(g).mul(envIntensity).mul(0.5).mul(preExposure);
  } else {
    material = new MeshBasicNodeMaterial({ side: DoubleSide });
    material.name = 'ocean.placeholder';
    // Fragment normal from the analytic slope at the (world-periodic) tile coordinates: no swimming.
    const frag = testWaves(phaseNow, (P) => varying(n.tileCoord(P)), varying(n.cellSize).mul(0.5), float(1));
    const nLocal = normalize(vec3(frag.slope.x.negate(), 1, frag.slope.y.negate()));
    let N: any = normalize(modelWorldMatrix.mul(vec4(nLocal, 0)).xyz);
    N = normalize(N.add(V.negate().mul(max(float(0.025).sub(dot(N, V.negate())), 0)))); // keep N facing the viewer
    const NoV = clamp(dot(N, V.negate()), 0.015, 1);
    const fresnel = float(0.018).add(float(0.982).mul(pow(float(1).sub(NoV), 5)));
    const deep = vec3(0.0159963, 0.0612461, 0.0998987);
    const sky = envAt(reflect(V, N), float(0.24));
    const diffuse = envAt(upW, float(1));
    let color: any = mix(deep.mul(diffuse).mul(0.8), sky, fresnel);
    // Gaussian distance haze towards the env colour just above the horizon (reference Â§2.11).
    const horiz = toFrag.sub(upW.mul(dot(toFrag, upW)));
    const hazeDir = normalize(horiz.add(upW.mul(dist.mul(0.002))));
    const haze = float(1).sub(exp(dist.div(280).pow(2).negate()));
    color = mix(color, envAt(hazeDir, float(0)), haze);
    material.colorNode = color.mul(envIntensity).mul(preExposure);
  }
  material.positionNode = positionNode;
  lod.mesh.material = material;
  lod.mesh.visible = !dbg.has('nomesh');
  scene.add(lod.mesh);
  if (overlay) lod.mesh.add(overlay); // child: shares the LOD mesh transform

  // ------------------------------------------------------------------ pass
  const omega = TEST_WAVES.map((w) => Math.sqrt(G * Math.hypot(2 * Math.PI * w.kx / w.P, 2 * Math.PI * w.kz / w.P)));
  const TWO_PI = Math.PI * 2;
  let lastTime = NaN;
  function setPhases(target: any[], t: number) {
    for (let i = 0; i < TEST_WAVES.length; i++) target[i].value = ((omega[i] * t - TEST_WAVES[i].phase) % TWO_PI + TWO_PI) % TWO_PI;
  }

  const fftPass: Pass = {
    desc: { id: 'ocean.fft', module: 'world/ocean', stage: 'compute', reads: [], writes: ['ocean.displacement'] },
    execute(ctx: PassContext) {
      opts.perf?.begin('world/ocean', 'ocean.lod');
      lod.update(camera);
      if (overlay && overlay.geometry !== lod.mesh.geometry) overlay.geometry = lod.mesh.geometry;
      setPhases(phasePrev, Number.isFinite(lastTime) ? lastTime : ctx.time);
      setPhases(phaseNow, ctx.time);
      lastTime = ctx.time;
      opts.perf?.end();
    },
  };

  return {
    passes: [fftPass],
    fftPass,
    mesh: lod.mesh,
    material,
    lod,
    previousPositionNode,
    setTier(t) { lod.setTier(t); },
    dispose() {
      lod.dispose();
      material.dispose();
      if (overlay) { overlay.removeFromParent(); overlay.material.dispose(); scene.backgroundNode = savedBackground; }
    },
  };
}
