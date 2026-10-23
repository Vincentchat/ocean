// three adapter (WebGPU / TSL): createOceanGpu wires the GPU FFT (three/fft.ts) to the camera-centred LOD mesh
// (three/lod.ts). Keep the createOceanGpu signature (spec §3/§5).
//
// Displacement: the FFT's chained cascade lookup (createFftSampling().sampleDisplacement, reference spectralField)
// at lod.nodes.tileCoord(P) (world-periodic, floating-origin safe), explicit LOD from the vertex footprint
// (lod.nodes.cellSize), faded to flat between 200 m and 1200 m (reference farFade). The core's spectrum anchor is
// the world origin (createOcean anchor 0), which is what tileCoord measures from.
// Motion vectors: previousPositionNode = lod.nodes.previousLocalPos + the PREVIOUS frame's displacement (the FFT
// keeps last frame's displacement layers = the field at time - frameDt), same lookup.
// Shading (TEMPORARY simple lit material; the real material builder replaces it): per-fragment normal from the
// FFT normal textures (reference spectralSurface: chained, footprint LOD from d(rest)/d(screen)), roughness from
// the mip shortening, Fresnel env reflection, deep-water body colour, grey whitecaps from the Jacobian
// compression, Gaussian distance haze (reference §2.11). Radiance x preExposure, no tonemap.
//
// Debug views (opts.debug, comma separated; demo ?debug=...):
//   lod        level colours (hue per level, darker = geomorph), lit by facet normals, wireframe overlay,
//              magenta background (any magenta below the horizon = crack / gap), back faces culled.
//   facets     grey facet shading of the displaced geometry (shows vertex positions; for swim tests), no haze.
//   nomesh     hide the ocean (perf baseline).
//   nosnap     disable LOD snapping (control for the swim test: the surface crawls).
//   lodoff:u:v offset the LOD's camera foot by (u east, v south) metres (swim test without moving the view).
//   velocity   screen-space motion of the displaced surface (current - previous position), |NDC v| x 200 per frame.
import { DoubleSide, FrontSide, Mesh, MeshBasicNodeMaterial } from 'three/webgpu';
import {
  abs, cameraPosition, cameraProjectionMatrix, cameraViewMatrix, clamp, cross, dFdx, dFdy, dot, exp, float, floor,
  max, mix, modelWorldMatrix, normalize, pmremTexture, pow, positionWorld, reflect, smoothstep, sqrt, uniform,
  varying, vec3, vec4,
} from 'three/tsl';
import type { FloatingOrigin, SkyLight, Vec3, WorldShape } from '../stubs/world-core.ts';
import type { Pass, PassContext } from './pass-types.ts';
import { createOceanLod, type OceanLod } from './lod.ts';
import { createOceanFft, createFftSampling, FFT_CASCADES, type OceanFft, type FftOceanSource } from './fft.ts';

export type { Pass, PassContext, PassDesc, Stage } from './pass-types.ts';
export { STAGES } from './pass-types.ts';
export { createOceanLod } from './lod.ts';
export type { OceanLod, OceanLodNodes, OceanLodOptions, OceanLodStats } from './lod.ts';
export { createOceanFft, createFftSampling, FFT_TIERS, FFT_CASCADES, BASE_SIZE, FINE_DOMAIN } from './fft.ts';
export type { OceanFft, FftSampling, FftSampleOptions, FftSurfaceSample, FftTierConfig } from './fft.ts';

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
  /** The GPU FFT (textures, uniforms, tiers, readback). */
  readonly fft: OceanFft;
  /** Resolves once the first FFT tier is compiled and has run. */
  readonly ready: Promise<void>;
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

  const fft = createOceanFft({ renderer, ocean: opts.ocean as FftOceanSource, tier, perf: opts.perf });
  const sampling = createFftSampling(fft);

  // ------------------------------------------------------------------ displacement (vertex)
  const fade = float(1).sub(smoothstep(200, 1200, n.distanceToCamera)); // reference farFade
  const tile = (P: number) => n.tileCoord(P);
  const disp = sampling.sampleDisplacement(tile, { footprint: n.cellSize }).mul(fade);
  const dispPrev = sampling.sampleDisplacement(tile, { footprint: n.cellSize, previous: true }).mul(fade);
  const positionNode = n.localPos.add(disp);
  const previousPositionNode = n.previousLocalPos.add(dispPrev);

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
  } else if (dbg.has('velocity')) {
    material = new MeshBasicNodeMaterial({ side: DoubleSide });
    material.name = 'ocean.debug.velocity';
    // NDC motion of the displaced vertex (current - previous), both with this frame's camera (object motion only).
    const toClip = (p: any) => cameraProjectionMatrix.mul(cameraViewMatrix).mul(modelWorldMatrix).mul(vec4(p, 1));
    const c0 = toClip(positionNode), c1 = toClip(previousPositionNode);
    const vel = varying(c0.xy.div(c0.w).sub(c1.xy.div(c1.w)));
    material.colorNode = vec3(abs(vel.x).mul(200), abs(vel.y).mul(200), 0.05).mul(envIntensity).mul(preExposure);
  } else {
    material = new MeshBasicNodeMaterial({ side: DoubleSide });
    material.name = 'ocean.fft.simple';
    // Fragment surface from the FFT normal textures at the (world-periodic) rest tile coordinates.
    const restLocal = varying(n.localPos.xz);
    const footprint = max(max(abs(dFdx(restLocal.x)), abs(dFdx(restLocal.y))), max(abs(dFdy(restLocal.x)), abs(dFdy(restLocal.y))));
    const tileV = new Map<number, any>(FFT_CASCADES.map((c) => [c.domain, varying(n.tileCoord(c.domain))] as [number, any]));
    const tileF = (P: number) => tileV.get(P);
    const surf = sampling.sampleSurface(tileF, { footprint });
    let N: any = normalize(modelWorldMatrix.mul(vec4(surf.normal, 0)).xyz);
    N = normalize(N.add(V.negate().mul(max(float(0.025).sub(dot(N, V.negate())), 0)))); // keep N facing the viewer
    const NoV = clamp(dot(N, V.negate()), 0.015, 1);
    const fresnel = float(0.018).add(float(0.982).mul(pow(float(1).sub(NoV), 5)));
    const rough = clamp(float(0.018).add(surf.variance.mul(0.8))
      .add(sqrt(float(1).add(fft.uniforms.unresolvedSlopeVariance)).sub(1).mul(0.8)), 0.018, 0.8);
    const deep = vec3(0.0159963, 0.0612461, 0.0998987);
    const sky = envAt(reflect(V, N), clamp(rough.add(0.22), 0, 1));
    const diffuse = envAt(upW, float(1));
    let color: any = mix(deep.mul(diffuse).mul(0.8), sky, fresnel);
    // Whitecaps (simple): grey foam where the surface compresses (reference foam source smoothstep(.15,.5,c)).
    const foam = smoothstep(0.4, 1.0, surf.compression).mul(0.5);
    color = mix(color, diffuse.mul(0.6), foam);
    // Gaussian distance haze towards the env colour just above the horizon (reference §2.11).
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
  const fftPass: Pass = {
    desc: fft.pass.desc,
    execute(ctx: PassContext) {
      opts.perf?.begin('world/ocean', 'ocean.lod');
      lod.update(camera);
      if (overlay && overlay.geometry !== lod.mesh.geometry) overlay.geometry = lod.mesh.geometry;
      opts.perf?.end();
      fft.pass.execute(ctx);
    },
  };

  return {
    passes: [fftPass],
    fftPass,
    mesh: lod.mesh,
    material,
    lod,
    fft,
    ready: fft.ready,
    previousPositionNode,
    setTier(t) { lod.setTier(t); fft.setTier(t); },
    dispose() {
      lod.dispose();
      fft.dispose();
      material.dispose();
      if (overlay) { overlay.removeFromParent(); overlay.material.dispose(); scene.backgroundNode = savedBackground; }
    },
  };
}

