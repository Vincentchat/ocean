// three adapter (WebGPU / TSL): createOceanGpu wires the GPU FFT (three/fft.ts) to the camera-centred LOD mesh
// (three/lod.ts). Keep the createOceanGpu signature (spec Â§3/Â§5).
//
// Displacement: the FFT's chained cascade lookup (createFftSampling().sampleDisplacement, reference spectralField)
// at lod.nodes.tileCoord(P) (world-periodic, floating-origin safe), explicit LOD from the vertex footprint
// (lod.nodes.cellSize), faded to flat between 200 m and 1200 m (reference farFade). The core's spectrum anchor is
// the world origin (createOcean anchor 0), which is what tileCoord measures from.
// Motion vectors: previousPositionNode = lod.nodes.previousLocalPos + the PREVIOUS frame's displacement (the FFT
// keeps last frame's displacement layers = the field at time - frameDt), same lookup.
// Shading: three/material.ts (createOceanMaterial), the full port of the reference water fragment shader, lit only
// through SKY_LIGHT / ENVIRONMENT (+ env map). Hooks: foamCoverage, transmission (refraction), haze, waveScale.
// Radiance x preExposure, no tonemap.
// Refraction (options.refraction, three/refraction.ts): the water leaves `scene` and is drawn by the 'ocean.water'
// pass, stage 'transparent', reads scene.color + scene.depth (+ ocean.displacement), writes scene.color. Optional
// TERRAIN_GPU (options.terrainGpu) for shore / contact thickness.
// Foam (three/foam.ts): history compute pass 'ocean.foam' after 'ocean.fft'; material foamCoverage = foam coverage.
// Passes: [ocean.fft (compute), ocean.foam (compute), ocean.water (transparent, with refraction)].
//
// Debug views (opts.debug, comma separated; demo ?debug=...):
//   lod        level colours (hue per level, darker = geomorph), lit by facet normals, wireframe overlay,
//              magenta background (any magenta below the horizon = crack / gap), back faces culled.
//   facets     grey facet shading of the displaced geometry (shows vertex positions; for swim tests), no haze.
//   nomesh     hide the ocean (perf baseline).
//   nosnap     disable LOD snapping (control for the swim test: the surface crawls).
//   lodoff:u:v offset the LOD's camera foot by (u east, v south) metres (swim test without moving the view).
//   rough      water material: R = roughness / .8, G = mip shortening x 4, B = cascade-1 LOD / 8.
//   footprint  water material: R = derivative / analytic footprint / 4, G = sqrt(dN variance) x 4, B = silhouette gate.
//   velocity   screen-space motion of the displaced surface (current - previous position), pixels / frame, log brightness (16 px saturates), hue = direction.
import { DoubleSide, FrontSide, Mesh, MeshBasicNodeMaterial, Scene } from 'three/webgpu';
import {
  abs, cameraPosition, cameraProjectionMatrix, cameraViewMatrix, clamp, cross, dFdx, dFdy, dot, float, floor,
  length, log2, max, modelWorldMatrix, normalize, pow, positionWorld, screenSize, smoothstep, uniform, varying, vec3, vec4,
} from 'three/tsl';
import type { EnvironmentState, FloatingOrigin, SkyLight, TerrainGpuHeight, Vec3, WorldShape } from '../stubs/world-core.ts';
import { createOceanMaterial, type OceanMaterial, type OceanMaterialOptions } from './material.ts';
import { createOceanRefraction, type OceanRefraction, type OceanRefractionOptions } from './refraction.ts';
import { createOceanFoam, type OceanFoam } from './foam.ts';
import type { Pass, PassContext } from './pass-types.ts';
import { createOceanLod, type OceanLod } from './lod.ts';
import { createOceanFft, createFftSampling, FFT_CASCADES, type OceanFft, type FftOceanSource } from './fft.ts';

export type { Pass, PassContext, PassDesc, Stage } from './pass-types.ts';
export { STAGES } from './pass-types.ts';
export { createOceanLod } from './lod.ts';
export type { OceanLod, OceanLodNodes, OceanLodOptions, OceanLodStats } from './lod.ts';
export { createOceanFft, createFftSampling, FFT_TIERS, FFT_CASCADES, BASE_SIZE, FINE_DOMAIN } from './fft.ts';
export type { OceanFft, FftSampling, FftSampleOptions, FftSurfaceSample, FftTierConfig } from './fft.ts';
export { createOceanMaterial } from './material.ts';
export type { OceanMaterial, OceanMaterialOptions, OceanFoamContext, OceanTransmission, OceanTransmissionContext } from './material.ts';
export { createOceanRefraction } from './refraction.ts';
export { createOceanFoam, FOAM_TIERS, FOAM_PARAMS } from './foam.ts';
export type { OceanFoam, OceanFoamNodes, OceanFoamOptions } from './foam.ts';
export type { OceanRefraction, OceanRefractionOptions } from './refraction.ts';

export type OceanTier = 'low' | 'medium' | 'high' | 'ultra';

export interface PerfSink {
  begin(module: string, phase: string): void;
  end(): void;
}

export interface OceanGpuOptions {
  /** The core WorldSystem from createOcean() (required: the FFT reads its spectrum slots). */
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
  /** ENVIRONMENT state (sunDir, cloudCover). */
  environment?: EnvironmentState;
  /** PMREM (CubeUV) env texture, linear HDR. */
  envMap?: any;
  /** Multiply env samples by this to get cd/m^2 (demo: REF_NITS). Default 1. */
  envIntensity?: number;
  /** TSL uniform node (or number) multiplied into the written radiance. Default 1. */
  preExposure?: any;
  /** Material hooks (three/material.ts): foam coverage, refraction transmission, distance haze, crest normaliser. */
  foamCoverage?: OceanMaterialOptions['foamCoverage'];
  transmission?: OceanMaterialOptions['transmission'];
  haze?: OceanMaterialOptions['haze'];
  waveScale?: number;
  /**
   * Refraction of the opaque scene (three/refraction.ts). When set, the water is drawn by the 'ocean.water' pass in
   * the 'transparent' stage (reads scene.color + scene.depth from PassContext.targets, writes scene.color) instead of
   * being added to `scene`. Without opaque geometry under the water the result equals the deep-water look exactly.
   */
  refraction?: boolean | Omit<OceanRefractionOptions, 'origin' | 'shape' | 'terrainGpu'>;
  /** TERRAIN_GPU height grid (world.optional(TERRAIN_GPU)): shore / contact thickness where scene.depth has no ground. */
  terrainGpu?: TerrainGpuHeight;
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
  /** The GPU FFT (textures, uniforms, tiers, readback). */
  readonly fft: OceanFft;
  /** Resolves once the first FFT tier is compiled and has run. */
  readonly ready: Promise<void>;
  /** Last frame's displaced local position node (motion vectors). */
  readonly previousPositionNode: any;
  /** Refraction piece (null when options.refraction is off). */
  readonly refraction: OceanRefraction | null;
  /** 'ocean.water' transparent-stage pass (null when refraction is off: the mesh is then in `scene`). */
  readonly waterPass: Pass | null;
  /** Whitecap foam (history compute pass 'ocean.foam' + material nodes, three/foam.ts). */
  readonly foam: OceanFoam;
  /** The water material wrapper (uniforms / overrides / update); null in debug views. */
  readonly water: OceanMaterial | null;
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

export function createOceanGpu(opts: OceanGpuOptions): OceanGpu {
  const { origin, camera, scene, renderer } = opts;
  if (!opts.ocean) throw new Error('createOceanGpu: options.ocean (the core createOcean() system) is required');
  const shape = opts.shape ?? defaultFlatShape();
  const preExposure = typeof opts.preExposure === 'object' && opts.preExposure ? opts.preExposure : uniform(opts.preExposure ?? 1);
  const envIntensity = uniform(opts.envIntensity ?? 1);
  const dbg = new Set((opts.debug ?? '').split(',').map((s) => s.trim()).filter(Boolean));
  let lodOffU = 0, lodOffV = 0;
  for (const d of dbg) if (d.startsWith('lodoff:')) { const [, u, v] = d.split(':'); lodOffU = Number(u) || 0; lodOffV = Number(v) || 0; }
  const tier = opts.tier ?? 'high';

  const lod = createOceanLod({ tier, shape, origin });
  lod.debug.noSnap = dbg.has('nosnap');
  lod.debug.footOffsetU = lodOffU;
  lod.debug.footOffsetV = lodOffV;
  const n = lod.nodes;

  // ?debug=fftcapillary: opt into ultra's capillary band (fft r3)
  const fft = createOceanFft({ renderer, ocean: opts.ocean as FftOceanSource, tier, perf: opts.perf, capillary: dbg.has('fftcapillary') });
  const sampling = createFftSampling(fft);

  // ------------------------------------------------------------------ displacement (vertex)
  const fade = float(1).sub(smoothstep(200, 1200, n.distanceToCamera)); // reference farFade
  const tile = (P: number) => n.tileCoord(P);
  const cubic = !dbg.has('fftbilinear'); // ?debug=fftbilinear: plain trilinear displacement (A/B of the Catmull-Rom)
  const disp = sampling.sampleDisplacement(tile, { footprint: n.cellSize, cubic }).mul(fade);
  const dispPrev = sampling.sampleDisplacement(tile, { footprint: n.cellSize, previous: true, cubic }).mul(fade);
  const positionNode = n.localPos.add(disp);
  const previousPositionNode = n.previousLocalPos.add(dispPrev);

  // ------------------------------------------------------------------ materials
  const upW = normalize(modelWorldMatrix.mul(vec4(0, 1, 0, 0)).xyz);
  const toFrag = positionWorld.sub(cameraPosition);
  const dist = toFrag.length();
  const V = toFrag.div(max(dist, 1e-4));
  const facetN = (() => {
    const nf = normalize(cross(dFdx(positionWorld), dFdy(positionWorld)));
    return nf.mul(dot(nf, V).greaterThan(0).select(float(-1), float(1))); // face the viewer
  })();

  // ------------------------------------------------------------------ foam (whitecap history, compute; three/foam.ts)
  const foam: OceanFoam = createOceanFoam({
    renderer, fft, lod, origin, shape, tier, camera, ocean: opts.ocean as any, perf: opts.perf,
  });

  // ------------------------------------------------------------------ refraction (transparent stage, optional)
  const refraction: OceanRefraction | null = opts.refraction
    ? createOceanRefraction({
      // contact foam lace = pattern R at rest / 18 m, the same binding as the whitecaps
      contactPattern: (c) => foam.nodes.pattern(c.restLocal, 18).x,
      ...(typeof opts.refraction === 'object' ? opts.refraction : {}), origin, shape, terrainGpu: opts.terrainGpu,
    })
    : null;

  let material: any;
  let water: OceanMaterial | null = null;
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
  } else if (dbg.has('velocity')) {
    material = new MeshBasicNodeMaterial({ side: DoubleSide });
    material.name = 'ocean.debug.velocity';
    // NDC motion of the displaced vertex (current - previous), both with this frame's camera (object motion only).
    const toClip = (p: any) => cameraProjectionMatrix.mul(cameraViewMatrix).mul(modelWorldMatrix).mul(vec4(p, 1));
    const c0 = toClip(positionNode), c1 = toClip(previousPositionNode);
    const vel = varying(c0.xy.div(c0.w).sub(c1.xy.div(c1.w)));
    // Readable scale: motion in PIXELS per frame; direction as hue (red/green = signed x/y, 0.5 = none),
    // brightness log2(1 + px) / log2(1 + 16): 1 px -> 0.24, 4 px -> 0.57, >= 16 px saturates. (fft r2, minimal edit)
    const px = vel.mul(screenSize).mul(0.5);
    const mag = length(px);
    const dir = px.div(max(mag, 1e-6));
    const bright = clamp(log2(mag.add(1)).div(Math.log2(17)), 0, 1);
    material.colorNode = vec3(dir.x.mul(0.5).add(0.5), dir.y.mul(0.5).add(0.5), 0.25).mul(bright).mul(envIntensity).mul(preExposure);
  } else {
    // The water material (three/material.ts): every shading term of the reference fragment shader.
    water = createOceanMaterial({
      fft, sampling, nodes: n, positionNode, displacement: disp,
      skyLight: opts.skyLight, environment: opts.environment, envMap: opts.envMap, envIntensity: opts.envIntensity ?? 1,
      preExposure, foamCoverage: opts.foamCoverage ?? ((c) => foam.nodes.coverage(c.restLocal)), transmission: opts.transmission ?? refraction?.transmission, haze: opts.haze,
      waveScale: opts.waveScale, ocean: opts.ocean as any,
      debug: dbg.has('rough') ? 'rough' : dbg.has('footprint') ? 'footprint' : undefined,
    });
    material = water.material;
  }
  material.positionNode = positionNode;
  lod.mesh.material = material;
  lod.mesh.visible = !dbg.has('nomesh');
  // With refraction the water is a transparent-stage draw (ocean.water pass, own draw list); otherwise a scene mesh.
  const waterScene = refraction ? new Scene() : null;
  if (waterScene) waterScene.name = 'ocean.water';
  (waterScene ?? scene).add(lod.mesh);
  const waterPass: Pass | null = refraction && waterScene ? refraction.createPass({ waterScene, camera }) : null;
  if (overlay) lod.mesh.add(overlay); // child: shares the LOD mesh transform

  // ------------------------------------------------------------------ pass
  const fftPass: Pass = {
    desc: fft.pass.desc,
    execute(ctx: PassContext) {
      opts.perf?.begin('world/ocean', 'ocean.lod');
      lod.update(camera);
      if (overlay && overlay.geometry !== lod.mesh.geometry) overlay.geometry = lod.mesh.geometry;
      water?.update();
      opts.perf?.end();
      fft.pass.execute(ctx);
    },
  };

  return {
    passes: waterPass ? [fftPass, foam.pass, waterPass] : [fftPass, foam.pass],
    fftPass,
    mesh: lod.mesh,
    material,
    lod,
    fft,
    ready: Promise.all([fft.ready, foam.ready]).then(() => undefined),
    previousPositionNode,
    refraction,
    waterPass,
    foam,
    water,
    setTier(t) { lod.setTier(t); fft.setTier(t); foam.setTier(t); },
    dispose() {
      lod.mesh.removeFromParent();
      lod.dispose();
      foam.dispose();
      fft.dispose();
      material.dispose();
      refraction?.dispose();
      if (overlay) { overlay.removeFromParent(); overlay.material.dispose(); scene.backgroundNode = savedBackground; }
    },
  };
}

