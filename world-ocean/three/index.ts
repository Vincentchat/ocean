// PLACEHOLDER three adapter (infra scaffold). The real WebGPU/TSL ocean replaces this file
// (fft.ts, material.ts, passes.ts, lod.ts ...). Keep the createOceanGpu signature (spec §3/§5).
//
// What it does now: a flat TSL-shaded plane at sea level, drawn camera-relative, with a 10 m grid
// computed from renderPos + origin.wrapped(10) so floating-origin behaviour is visible, and an env
// reflection (Fresnel) when an envMap is given. Radiance in cd/m^2 x preExposure; no tonemapping.
import { Mesh, MeshBasicNodeMaterial, PlaneGeometry, Vector3, DoubleSide } from 'three/webgpu';
import {
  abs, cameraPosition, clamp, dot, float, fract, fwidth, max, min, mix, normalize, pmremTexture, pow,
  positionWorld, reflect, uniform, vec2, vec3, positionLocal,
} from 'three/tsl';
import type { FloatingOrigin, SkyLight, Vec3 } from '../stubs/world-core.ts';
import type { Pass, PassContext } from './pass-types.ts';

export type { Pass, PassContext, PassDesc, Stage } from './pass-types.ts';
export { STAGES } from './pass-types.ts';

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
  /** Sea level along +Y in this placeholder (flat only). */
  seaLevel?: number;
}

export interface OceanGpu {
  readonly passes: Pass[];
  readonly fftPass: Pass;
  readonly mesh: any;
  readonly material: any;
  /** Last frame's displaced local position node (motion vectors). */
  readonly previousPositionNode: any;
  setTier(t: OceanTier): void;
  dispose(): void;
}

export function createOceanGpu(opts: OceanGpuOptions): OceanGpu {
  const { origin, camera } = opts;
  const seaLevel = opts.seaLevel ?? 0;
  let tier: OceanTier = opts.tier ?? 'high';
  const preExposure = typeof opts.preExposure === 'object' && opts.preExposure ? opts.preExposure : uniform(opts.preExposure ?? 1);
  const wrapped10 = uniform(new Vector3());
  const envIntensity = uniform(opts.envIntensity ?? 1);

  const geometry = new PlaneGeometry(40000, 40000, 1, 1);
  geometry.rotateX(-Math.PI / 2);
  const material = new MeshBasicNodeMaterial({ side: DoubleSide });
  material.name = 'ocean.placeholder';

  // Grid lines every 10 m, sampled at renderPos + wrapped (floating-origin safe).
  const p = positionWorld.xz.add(wrapped10.xz);
  const cell = abs(fract(p.div(10).sub(0.5)).sub(0.5)).mul(10).div(max(fwidth(p), vec2(1e-4)));
  const line = float(1).sub(clamp(min(cell.x, cell.y), 0, 1));
  const V = normalize(positionWorld.sub(cameraPosition));
  const N = vec3(0, 1, 0);
  const NoV = clamp(dot(N, V.negate()), 0, 1);
  const fresnel = float(0.02).add(float(0.98).mul(pow(float(1).sub(NoV), 5)));
  const deep = vec3(0.0159963, 0.0612461, 0.0998987);
  const sky = opts.envMap ? pmremTexture(opts.envMap, reflect(V, N), float(0.08)).rgb : vec3(0.3, 0.45, 0.6);
  // Reference-unit colours x envIntensity -> cd/m^2, then x preExposure.
  const radiance = mix(deep.mul(0.6), sky, fresnel).add(vec3(0.05, 0.08, 0.09).mul(line)).mul(envIntensity);
  material.colorNode = radiance.mul(preExposure);

  const mesh = new Mesh(geometry, material);
  mesh.name = 'ocean.placeholder';
  mesh.frustumCulled = false;
  opts.scene.add(mesh);

  const tmp: Vec3 = { x: 0, y: 0, z: 0 };
  function update() {
    // Follow the camera in render space, snapped to the 10 m grid so nothing swims.
    mesh.position.set(Math.round(camera.position.x / 10) * 10, seaLevel - origin.origin.y, Math.round(camera.position.z / 10) * 10);
    origin.wrapped(10, tmp);
    (wrapped10.value as Vector3).set(tmp.x, tmp.y, tmp.z);
  }
  const unsubscribe = origin.onShift(update);

  const fftPass: Pass = {
    desc: { id: 'ocean.fft', module: 'world/ocean', stage: 'compute', reads: [], writes: ['ocean.displacement'] },
    execute(_ctx: PassContext) {
      opts.perf?.begin('world/ocean', 'gpu-update');
      update();
      opts.perf?.end();
    },
  };

  return {
    passes: [fftPass],
    fftPass,
    mesh,
    material,
    previousPositionNode: positionLocal, // flat placeholder: no displacement, previous == current
    setTier(t) { tier = t; void tier; },
    dispose() {
      unsubscribe();
      opts.scene.remove(mesh);
      geometry.dispose();
      material.dispose();
    },
  };
}
