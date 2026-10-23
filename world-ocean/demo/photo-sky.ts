// DEMO ONLY: photographic sky stand-in for world/sky (spec §4: "Keep a photo sky only in the demo, as a
// stand-in SkyLight + env map"). TSL port of the reference's src/photo-sky.js (photographicSky, sky decode)
// and src/atmosphere.js (atmosphereRadiance -> HDR cube) + src/prefiltered-environment.js (PMREM).
//
// Units. The reference shades in "reference units" (RU): its tonemap(x) expects RU directly. Here:
//   - env textures (cube + PMREM) store RU (rgb) and sun visibility (a), like the reference cube.
//   - REF_NITS converts RU -> cd/m^2. SkyLight values are physical (lux, cd/m^2) = RU * REF_NITS.
//   - the demo uses preExposure = 1 / REF_NITS, so scene.color holds RU and the demo tonemap
//     (the reference's ACES fit + gamma 2.2) gives the reference look.
// So an ocean shader that computes radiance in cd/m^2 (env sample * envIntensity, sun from SkyLight)
// and multiplies by preExposure matches the WebGL reference exposure.
import {
  CubeCamera, CubeRenderTarget, HalfFloatType, LinearFilter, LinearMipmapLinearFilter, Mesh,
  MeshBasicNodeMaterial, BackSide, BoxGeometry, NoColorSpace, PMREMGenerator, RepeatWrapping,
  ClampToEdgeWrapping, Scene, TextureLoader, Vector3, type Texture, type WebGPURenderer, type RenderTarget,
} from 'three/webgpu';
import {
  Fn, abs, acos, asin, atan, clamp, cos, dot, exp2, float, floor, max, min, mix, normalize, positionLocal,
  pow, select, sin, smoothstep, tan, texture, uniform, vec2, vec3, vec4, positionWorldDirection,
} from 'three/tsl';
import type { RGB, SkyLight, Vec3 } from '../stubs/world-core.ts';
import { DEMO_ASSETS, resolveAsset } from '../assets.ts';

/** cd/m^2 per reference unit. */
export const REF_NITS = 10000;

/** Reference scene presets (src/ocean.js `presets`): sun elevation (degrees) + mood (0 golden, 1 azure, 2 storm). */
export const SKY_PRESETS = {
  golden: { sunDeg: 10, mood: 0 },
  azure: { sunDeg: 50, mood: 1 },
  storm: { sunDeg: 38, mood: 2 },
} as const;
export type SkyPresetName = keyof typeof SKY_PRESETS;

const PI = Math.PI;
const SKY_GAIN_WEIGHT = 0.42399845327747504;
const SKY_SUN_DIRECTION = [0.5542598486859208, 0.7418088572248551, 0.3775124361890805];
const SKY_SUN_BASELINE = [9.655837059020996, 10.09691047668457, 12.554890632629395];
const ROTATION = 3.093747;
const PHOTO_SUN_TAN = Math.tan(0.835763);
const DEEP = [0.0159963, 0.0612461, 0.0998987];

/** Reference sun direction for elevation e (radians): flat frame, +Y up. */
export function referenceSunDir(e: number, out: Vec3): Vec3 {
  out.x = -0.79863551 * Math.cos(e); out.y = Math.sin(e); out.z = -0.60181502 * Math.cos(e);
  return out;
}
/** src/atmosphere.js solarColor. */
export function referenceSolarColor(sun: number, out: RGB): RGB {
  const warm = Math.max(0, Math.min(1, (sun - 0.08) / 0.7));
  out.r = 1; out.g = 0.58 + 0.39 * warm; out.b = 0.3 + 0.62 * warm;
  return out;
}

// ------------------------------------------------------------------------------------------------
// TSL
// ------------------------------------------------------------------------------------------------
const codeToLinear = Fn(([code]: [any]) => {
  const v = code.div(255);
  const lookup = select(code.lessThan(1024), floor(code), code);
  return select(v.lessThan(0.04045), v.mul(0.0773993808), pow(lookup.div(255).mul(0.9478672986).add(0.0521327014), 2.4));
});

function makeSkyNodes(sdrTex: Texture, gainTex: Texture) {
  const uSun = uniform(SKY_PRESETS.storm.sunDeg * PI / 180);   // radians
  const uMood = uniform(SKY_PRESETS.storm.mood);
  const uSunDir = uniform(new Vector3());
  const uSolar = uniform(new Vector3(1, 0.7, 0.3));

  const decode = Fn(([sdr255, gain01]: [any, any]) => {
    const boosted = sdr255.add(1).mul(exp2(gain01.mul(15.99 * SKY_GAIN_WEIGHT))).sub(1);
    return clamp(vec3(codeToLinear(boosted.x), codeToLinear(boosted.y), codeToLinear(boosted.z)), 0, 65504);
  });
  const sampleFast = Fn(([q]: [any]) => {
    const d = normalize(q);
    const uv = vec2(atan(d.z, d.x).div(2 * PI).add(0.5), asin(clamp(d.y, -1, 1)).div(PI).add(0.5));
    const lin = decode(texture(sdrTex, uv).level(0).rgb.mul(255), texture(gainTex, uv).level(0).rgb);
    // removeSkySun
    const angle = acos(clamp(dot(d, vec3(...SKY_SUN_DIRECTION)), -1, 1)).mul(180 / PI);
    const t = clamp(angle.sub(1).div(0.25), 0, 1);
    const amount = float(1).sub(t.mul(t).mul(float(3).sub(t.mul(2))));
    return lin.sub(max(lin.sub(vec3(...SKY_SUN_BASELINE)), 0).mul(amount));
  });
  /** Shared part of photographicSky / atmosphereRadiance (no sun disc, no below-horizon fresnel). */
  const gradedSky = Fn(([direction]: [any]) => {
    const rd = normalize(direction);
    const q0 = normalize(vec3(rd.x, max(abs(rd.y), 0.0001), rd.z));
    const c = Math.cos(ROTATION), s = Math.sin(ROTATION);
    const qx = q0.x.mul(c).sub(q0.z.mul(s));
    const qz = q0.x.mul(s).add(q0.z.mul(c));
    const q = normalize(vec3(qx, q0.y.mul(PHOTO_SUN_TAN).div(max(tan(uSun), 0.025)), qz));
    const color0 = sampleFast(q).mul(0.7);
    const sunset = float(1).sub(smoothstep(0.12, 0.58, uSun));
    const color1 = color0.mul(mix(vec3(1, 1, 1), vec3(1.28, 0.75, 0.48), sunset));
    const storm = clamp(uMood.sub(1), 0, 1);
    const lum = dot(color1, vec3(0.2126, 0.7152, 0.0722));
    const color2 = mix(color1, vec3(0.72, 0.79, 0.84).mul(lum.mul(0.35).add(0.15)), storm.mul(0.92));
    const haze0 = mix(vec3(0.258, 0.275, 0.287), vec3(0.33, 0.206, 0.138), sunset);
    const haze = mix(haze0, vec3(0.18, 0.205, 0.22), storm);
    return mix(haze, color2, smoothstep(0, mix(0.025, 0.065, sunset), max(rd.y, 0)));
  });
  /** src/photo-sky.js photographicSky: visible sky incl. sun disc, RU. */
  const photographicSky = Fn(([direction]: [any]) => {
    const rd = normalize(direction);
    const storm = clamp(uMood.sub(1), 0, 1);
    const disc = smoothstep(0.999955, 0.999985, dot(rd, uSunDir)).mul(float(1).sub(storm.mul(0.97)));
    return gradedSky(rd).add(uSolar.mul(4.5).mul(disc));
  });
  /** src/atmosphere.js atmosphereRadiance: env capture, rgb RU + a = sun visibility. */
  const atmosphereRadiance = Fn(([direction]: [any]) => {
    const rd = normalize(direction);
    const below = max(rd.y.negate(), 0);
    const col = gradedSky(rd);
    const F = float(0.0180094).add(float(0.9819906).mul(pow(float(1).sub(min(below, 1)), 5)));
    const under = col.mul(F).add(vec3(...DEEP).mul(float(1).sub(F)));
    const storm = clamp(uMood.sub(1), 0, 1);
    return vec4(max(select(below.greaterThan(0), under, col), 0), float(1).sub(storm.mul(0.97)));
  });
  return { uSun, uMood, uSunDir, uSolar, photographicSky, atmosphereRadiance };
}

// ------------------------------------------------------------------------------------------------
// CPU twin (for SkyLight SH / illuminance): same math, on a downsampled copy of the photo.
// ------------------------------------------------------------------------------------------------
function cpuCodeToLinear(code: number): number {
  const v = code / 255;
  if (v < 0.04045) return v * 0.0773993808;
  const lookup = code < 1024 ? Math.floor(code) : code;
  return Math.pow((lookup / 255) * 0.9478672986 + 0.0521327014, 2.4);
}

interface CpuImage { w: number; h: number; sdr: Uint8ClampedArray; gain: Uint8ClampedArray }

function cpuAtmosphere(img: CpuImage, sun: number, mood: number, dx: number, dy: number, dz: number, out: number[]): void {
  const len = Math.hypot(dx, dy, dz);
  const rx = dx / len, ry = dy / len, rz = dz / len;
  let qx = rx, qy = Math.max(Math.abs(ry), 1e-4), qz = rz;
  let l = Math.hypot(qx, qy, qz); qx /= l; qy /= l; qz /= l;
  const c = Math.cos(ROTATION), s = Math.sin(ROTATION);
  const tx = c * qx - s * qz, tz = s * qx + c * qz;
  qx = tx; qz = tz; qy = qy * PHOTO_SUN_TAN / Math.max(Math.tan(sun), 0.025);
  l = Math.hypot(qx, qy, qz); qx /= l; qy /= l; qz /= l;
  const u = Math.atan2(qz, qx) / (2 * PI) + 0.5, v = Math.asin(Math.max(-1, Math.min(1, qy))) / PI + 0.5;
  const px = Math.min(img.w - 1, Math.max(0, Math.floor(u * img.w)));
  const py = Math.min(img.h - 1, Math.max(0, Math.floor((1 - v) * img.h))); // image rows top-down
  const i = (py * img.w + px) * 4;
  const angle = Math.acos(Math.max(-1, Math.min(1, qx * SKY_SUN_DIRECTION[0] + qy * SKY_SUN_DIRECTION[1] + qz * SKY_SUN_DIRECTION[2]))) * 180 / PI;
  const t = Math.max(0, Math.min(1, (angle - 1) / 0.25)), amount = 1 - t * t * (3 - 2 * t);
  const sunset = 1 - smooth(0.12, 0.58, sun), storm = Math.max(0, Math.min(1, mood - 1));
  const tint = [1 + 0.28 * sunset, 1 - 0.25 * sunset, 1 - 0.52 * sunset];
  for (let k = 0; k < 3; k++) {
    const boosted = (img.sdr[i + k] + 1) * Math.pow(2, (img.gain[i + k] / 255) * 15.99 * SKY_GAIN_WEIGHT) - 1;
    let lin = Math.min(65504, Math.max(0, cpuCodeToLinear(boosted)));
    lin -= Math.max(lin - SKY_SUN_BASELINE[k], 0) * amount;
    out[k] = lin * 0.7 * tint[k];
  }
  const lum = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
  const grey = [0.72, 0.79, 0.84], h0 = [0.258, 0.275, 0.287], h1 = [0.33, 0.206, 0.138], h2 = [0.18, 0.205, 0.22];
  const a = smooth(0, 0.025 + (0.065 - 0.025) * sunset, Math.max(ry, 0));
  const below = Math.max(-ry, 0), F = 0.0180094 + 0.9819906 * Math.pow(1 - Math.min(below, 1), 5);
  for (let k = 0; k < 3; k++) {
    let col = out[k] + (grey[k] * (0.15 + lum * 0.35) - out[k]) * storm * 0.92;
    let haze = h0[k] + (h1[k] - h0[k]) * sunset; haze += (h2[k] - haze) * storm;
    col = haze + (col - haze) * a;
    if (below > 0) col = col * F + DEEP[k] * (1 - F);
    out[k] = Math.max(col, 0);
  }
}
function smooth(e0: number, e1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

async function loadCpuImage(url: string, w: number, h: number): Promise<Uint8ClampedArray> {
  const blob = await (await fetch(url)).blob();
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', resizeWidth: w, resizeHeight: h, resizeQuality: 'medium' });
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  return ctx.getImageData(0, 0, w, h).data;
}

// ------------------------------------------------------------------------------------------------
// Public stand-in
// ------------------------------------------------------------------------------------------------
export interface PhotoSkyLight extends SkyLight {
  sunIlluminance: number; skyIlluminance: number; sunColor: RGB; skyColor: RGB; fogColor: RGB;
  fogDensity: number; fogHeightFalloff: number; suggestedEV100: number; version: number;
  /** 9 real SH coefficients x RGB (27 floats), radiance projection in cd/m^2, flat frame with y = up. Order:
   *  0 Y00, 1 x, 2 y, 3 z, 4 xy, 5 yz, 6 (3y^2-1), 7 xz, 8 (x^2-z^2) (basis constants 0.282095 / 0.488603 / 1.092548 / 0.315392 / 0.546274). */
  readonly skySH: Float32Array;
  /** Demo extension: direction towards the sun (flat frame) and the raw reference parameters. */
  readonly sunDir: Vec3;
  readonly reference: { sunElevationRad: number; mood: number; storminess: number; sunVisibility: number; solarColor: RGB };
}

export interface PhotoSky {
  /** Assign to scene.backgroundNode (RU x REF_NITS x preExposure). */
  readonly backgroundNode: any;
  readonly skyLight: PhotoSkyLight;
  /** Prefiltered env (PMREM, CubeUV mapping). Sample with pmremTexture(envMap, dir, roughness). RU; alpha = sun visibility.
   *  null until envPass has executed once (the demo runs it before creating the ocean). */
  readonly envMap: Texture | null;
  /** Raw HDR cube (with mips), RU. */
  readonly envCube: Texture;
  /** Multiply env samples by this to get cd/m^2. */
  readonly envIntensity: number;
  /** compute-stage pass: re-captures the env cube + PMREM when the sky changed. */
  readonly envPass: { desc: any; execute(ctx: any): void };
  setPreset(name: SkyPresetName): void;
  /** Resolves once photo textures + CPU SkyLight are ready and the first env capture ran. */
  readonly ready: Promise<void>;
  readonly envVersion: number;
  dispose(): void;
}

export async function createPhotoSky(renderer: WebGPURenderer, opts: { preExposure: any; preset?: SkyPresetName; envSize?: number }): Promise<PhotoSky> {
  const loader = new TextureLoader();
  const [sdr, gain] = await Promise.all([
    loader.loadAsync(resolveAsset(DEMO_ASSETS.skyBase)),
    loader.loadAsync(resolveAsset(DEMO_ASSETS.skyGain)),
  ]);
  for (const t of [sdr, gain]) {
    t.colorSpace = NoColorSpace;
    t.flipY = true;
    t.wrapS = RepeatWrapping;
    t.wrapT = ClampToEdgeWrapping;
    t.minFilter = LinearFilter;
    t.magFilter = LinearFilter;
    t.generateMipmaps = false;
    t.needsUpdate = true;
  }
  const nodes = makeSkyNodes(sdr, gain);
  const backgroundNode = nodes.photographicSky(positionWorldDirection).mul(REF_NITS).mul(opts.preExposure);

  // Env capture scene (atmosphereRadiance on a back-faced box), cube camera, PMREM.
  const envSize = opts.envSize ?? 256;
  const cubeTarget = new CubeRenderTarget(envSize, {
    type: HalfFloatType, generateMipmaps: true, minFilter: LinearMipmapLinearFilter, magFilter: LinearFilter, depthBuffer: false,
  });
  const envScene = new Scene();
  const envMaterial = new MeshBasicNodeMaterial({ side: BackSide, depthTest: false, depthWrite: false });
  envMaterial.fragmentNode = nodes.atmosphereRadiance(positionLocal);
  const envBox = new Mesh(new BoxGeometry(2, 2, 2), envMaterial);
  envScene.add(envBox);
  const cubeCamera = new CubeCamera(0.1, 10, cubeTarget);
  const pmrem = new PMREMGenerator(renderer);
  let pmremTarget: RenderTarget | null = null;
  let dirty = true;
  let envVersion = 0;

  // CPU copy for SkyLight.
  const W = 512, H = 256;
  const [cpuSdr, cpuGain] = await Promise.all([
    loadCpuImage(resolveAsset(DEMO_ASSETS.skyBase), W, H),
    loadCpuImage(resolveAsset(DEMO_ASSETS.skyGain), W, H),
  ]);
  const img: CpuImage = { w: W, h: H, sdr: cpuSdr, gain: cpuGain };

  const skyLight: PhotoSkyLight = {
    sunIlluminance: 0, skyIlluminance: 0,
    sunColor: { r: 1, g: 1, b: 1 }, skyColor: { r: 1, g: 1, b: 1 },
    skySH: new Float32Array(27),
    fogDensity: 1 / 250, fogHeightFalloff: 0, fogColor: { r: 0, g: 0, b: 0 },
    suggestedEV100: 0, version: 0,
    sunDir: { x: 0, y: 1, z: 0 },
    reference: { sunElevationRad: 0, mood: 0, storminess: 0, sunVisibility: 1, solarColor: { r: 1, g: 1, b: 1 } },
  };

  function computeSkyLight(sun: number, mood: number) {
    const sh = skyLight.skySH;
    sh.fill(0);
    const rgb = [0, 0, 0];
    let upE = [0, 0, 0];
    const NT = 48, NP = 96;
    let fogAcc = [0, 0, 0], fogW = 0;
    for (let i = 0; i < NT; i++) {
      const theta = ((i + 0.5) / NT) * PI; // from +Y
      const st = Math.sin(theta), ct = Math.cos(theta);
      const dOmega = st * (PI / NT) * (2 * PI / NP);
      for (let j = 0; j < NP; j++) {
        const phi = ((j + 0.5) / NP) * 2 * PI;
        const x = st * Math.cos(phi), y = ct, z = st * Math.sin(phi);
        cpuAtmosphere(img, sun, mood, x, y, z, rgb);
        // Real SH basis, order 2 (y = up).
        const b = [0.282095, 0.488603 * x, 0.488603 * y, 0.488603 * z, 1.092548 * x * y, 1.092548 * y * z,
          0.315392 * (3 * y * y - 1), 1.092548 * x * z, 0.546274 * (x * x - z * z)];
        for (let k = 0; k < 9; k++) for (let c = 0; c < 3; c++) sh[k * 3 + c] += rgb[c] * REF_NITS * b[k] * dOmega;
        if (y > 0) for (let c = 0; c < 3; c++) upE[c] += rgb[c] * REF_NITS * y * dOmega;
        if (Math.abs(y) < 0.08) { for (let c = 0; c < 3; c++) fogAcc[c] += rgb[c]; fogW++; }
      }
    }
    const solar = referenceSolarColor(sun, skyLight.reference.solarColor);
    const storm = Math.max(0, Math.min(1, mood - 1));
    const visibility = 1 - storm * 0.97;
    const lumS = 0.2126 * solar.r + 0.7152 * solar.g + 0.0722 * solar.b;
    skyLight.sunColor.r = solar.r / lumS; skyLight.sunColor.g = solar.g / lumS; skyLight.sunColor.b = solar.b / lumS;
    // Reference sun term: sunlight() * BRDF * (4.5 - 3.5 storm) * visibility  ->  E_sun in lux (x REF_NITS).
    skyLight.sunIlluminance = (4.5 - 3.5 * storm) * visibility * lumS * REF_NITS;
    const lumE = 0.2126 * upE[0] + 0.7152 * upE[1] + 0.0722 * upE[2];
    skyLight.skyIlluminance = lumE;
    skyLight.skyColor.r = upE[0] / lumE; skyLight.skyColor.g = upE[1] / lumE; skyLight.skyColor.b = upE[2] / lumE;
    skyLight.fogColor.r = fogAcc[0] / fogW * REF_NITS; skyLight.fogColor.g = fogAcc[1] / fogW * REF_NITS; skyLight.fogColor.b = fogAcc[2] / fogW * REF_NITS;
    // Reference haze: 1 - exp(-(d / L)^2), L = mix(280, 220, storm) m. Exposed as a density ~ 1/L.
    skyLight.fogDensity = 1 / (280 - 60 * storm);
    skyLight.fogHeightFalloff = 0;
    skyLight.suggestedEV100 = Math.log2(Math.max(skyLight.sunIlluminance * Math.max(Math.sin(sun), 0) + lumE, 1) * 100 / 250);
    referenceSunDir(sun, skyLight.sunDir);
    skyLight.reference.sunElevationRad = sun;
    skyLight.reference.mood = mood;
    skyLight.reference.storminess = storm;
    skyLight.reference.sunVisibility = visibility;
    skyLight.version++;
  }

  function applyPreset(name: SkyPresetName) {
    const p = SKY_PRESETS[name];
    if (!p) throw new Error(`Unknown sky preset ${name}`);
    const sun = p.sunDeg * PI / 180;
    nodes.uSun.value = sun;
    nodes.uMood.value = p.mood;
    referenceSunDir(sun, nodes.uSunDir.value as Vector3);
    const sc = referenceSolarColor(sun, { r: 0, g: 0, b: 0 });
    (nodes.uSolar.value as Vector3).set(sc.r, sc.g, sc.b);
    computeSkyLight(sun, p.mood);
    dirty = true;
  }
  applyPreset(opts.preset ?? 'storm');

  let resolveReady!: () => void;
  const ready = new Promise<void>((r) => (resolveReady = r));

  const envPass = {
    desc: { id: 'demo.sky-env', module: 'demo', stage: 'compute', reads: [], writes: ['demo.env'] },
    execute(ctx: any) {
      if (!dirty) return;
      dirty = false;
      const r = ctx.renderer;
      cubeCamera.update(r, envScene);
      pmremTarget = pmrem.fromCubemap(cubeTarget.texture, pmremTarget);
      envVersion++;
      resolveReady();
    },
  };

  const sky: PhotoSky = {
    backgroundNode,
    skyLight,
    get envMap() { return pmremTarget ? pmremTarget.texture : null; },
    envCube: cubeTarget.texture,
    envIntensity: REF_NITS,
    envPass,
    setPreset: applyPreset,
    ready,
    get envVersion() { return envVersion; },
    dispose() {
      pmrem.dispose();
      pmremTarget?.dispose();
      cubeTarget.dispose();
      envMaterial.dispose();
      envBox.geometry.dispose();
      sdr.dispose();
      gain.dispose();
    },
  };
  return sky;
}
