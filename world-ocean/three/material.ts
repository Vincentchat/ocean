// Water surface material (three r186, WebGPU / TSL, MeshBasicNodeMaterial with a custom colorNode).
// Faithful port of the WebGL reference fragment shader (src/ocean.js:228-354, docs/port/REFERENCE.md Ã‚Â§2), every term
// in the same order with the same constants. Output: linear HDR radiance (cd/m^2) x preExposure, no tonemap.
//
// Term by term (reference line refs = src/ocean.js):
//   normal      FFT normal textures, chained cascades, explicit LOD from the fragment footprint
//               max(|d rest/dx|, |d rest/dy|) (:271, spectralSurface :176-182; trilinear like normalAt). Not faded
//               with distance (the reference does not either: the mips + roughness do the distance filtering).
//               N = facing fix N + V max(.025 - N.V, 0) (:273). R = reflect(-V, N), NoV >= .015 (:274).
//   roughness   .018 + .8 spectralVariance (mip shortening) + .8 (sqrt(1 + slopeVar) - 1) (slopeVar = the FFT's
//               unresolvedSlopeVariance) + .8 sqrt(max(|dN/dx|^2, |dN/dy|^2)), clamped .018..8 (:276-277).
//               Derivatives are taken at the top level of the shader (uniform control flow in WGSL).
//   sun spec    GGX D, Smith-Schlick G (k = (r+1)^2/8), Schlick F0 .0180094, r floored at .12 (sun glitter path),
//               spec = D G F / (4 NoV + .04), capped at 22 (:278-281, :298).
//   Fresnel     split-sum env BRDF (Karis mobile fit), F0 .0180094 (= IOR 1.31) (:114-118, :297).
//   reflection  prefiltered env (PMREM) at roughness min(.22 + rough, 1) (:131-132, :295).
//   water body  deep (.0159963,.0612461,.0998987) x mediumLight, mediumLight = diffuseSky .8 + sun (.12 + .18 NoL);
//               transmitted scene x Beer-Lambert exp(-sigma path) + in-scatter deep x mediumLight x (1 - exp) when a
//               `transmission` input is plugged in (three/refraction.ts) and a background exists; else (open water,
//               sky behind) exactly the reference's no-refraction path (valid = 0) (:282-290).
//   contact     transmission.contact(color) after foam, before haze: edge fade to the raw scene + contact foam (:337-348).
//   SSS / crest transmissionTint (.0159963,.1356333,.0908417) x smoothstep(-.3,.85, height / waveScale)
//               x (.12 + .6 backLight) x (diffuseSky .4 + sun 2.2), backLight = max(V . -norm(L + .4 N), 0)^2 (:283, :291-293).
//   foam        coverage from the `foamCoverage` input (foam piece; default: the FFT compression); grey foam
//               .3 + .7 sat(surfaceNormal . L) at the foam white level, sun specular removed under foam (:333-336).
//   haze        Gaussian 1 - exp(-(dist / D)^2), D = 1 / SkyLight.fogDensity (demo: 280 m, storm 220 m), towards
//               the env colour just above the horizon (direction (horizontal, dist .002)) (:350-352). Input.
//   back faces  DoubleSide with the same shader (the reference has no underwater path); the facing fix flips N.
//
// Units (rule 9). Light comes only through SKY_LIGHT / ENVIRONMENT (+ the env map stand-in):
//   E_sun  = SkyLight.sunColor * SkyLight.sunIlluminance (lux) = the reference's sunlight() * 4.5 * sunVisibility in
//            cd/m^2 (4.5 = the reference's sun-disc / specular gain: the sun is a 4.5 RU irradiance).
//            Reference sun terms that use sunlight() * visibility without the 4.5 gain use E_sun / 4.5.
//   sunSpecularScale = the reference's storm dimming of the sun specular, (4.5 - 3.5 storm) / 4.5. Default from
//            ENVIRONMENT.cloudCover (1 - 7/9 cloudCover); the demo maps the reference storminess to cloudCover.
//   env    = pmremTexture(envMap) * envIntensity (cd/m^2). Without an env map: SkyLight.skySH.
//   foam white = envIntensity (1 reference unit: the reference foam is unlit grey .3..1 RU).
//   With the demo (REF_NITS, preExposure = 1 / REF_NITS) the written value equals the reference's pre-tonemap colour.
import { DoubleSide, MeshBasicNodeMaterial, Vector3 } from 'three/webgpu';
import {
  cameraPosition, clamp, dFdx, dFdy, dot, exp, exp2, float, length, max, min, mix, modelWorldMatrix, normalize,
  pmremTexture, pow, positionWorld, reflect, smoothstep, sqrt, uniform, varying, vec2, vec3, vec4,
} from 'three/tsl';
import type { EnvironmentState, SkyLight } from '../stubs/world-core.ts';
import type { FftSampling, OceanFft } from './fft.ts';
import { FFT_CASCADES } from './fft.ts';
import type { OceanLodNodes } from './lod.ts';

/** What the foam hook gets (all fragment-stage TSL nodes). */
export interface OceanFoamContext {
  /** Periodic rest-tile coordinate for period P (m), (east, south), varying. */
  tileCoord(period: number): any;
  /** Undisplaced local position (x east, z south), metres, varying (camera-foot relative). */
  restLocal: any;
  /** Fragment footprint on the rest grid, metres. */
  footprint: any;
  /** FFT whitecap compression (reference max(3 - sum a_i, 0)) at the fragment's LOD. */
  compression: any;
  /** Surface normal (local mesh axes x east, y up, z south). */
  normalLocal: any;
  /** Wind direction (towards) in the tile frame (east, south). */
  windAB: any;
}

/** What the transmission (refraction) hook gets (fragment-stage TSL nodes). */
export interface OceanTransmissionContext {
  /** Refraction normal (reference mix(surfaceNormal, N, .22), viewer-facing), world axes. */
  normalWorld: any;
  /** Unit vector fragment -> camera, world axes. */
  viewDir: any;
  positionWorld: any;
  /** Local up (shape), world axes. */
  upWorld: any;
  /** Periodic rest-tile coordinate for period P (m), varying (contact pattern). */
  tileCoord(period: number): any;
  /** Undisplaced local position (x east, z south), m, varying (same as OceanFoamContext.restLocal). */
  restLocal: any;
  /** The material's preExposure node (scene.color = radiance x preExposure). */
  preExposure: any;
  /** Foam / contact white level (cd/m^2 per reference unit). */
  foamWhite: any;
}

/** What the transmission hook returns. */
export interface OceanTransmission {
  /** Transmitted scene radiance (cd/m^2, NOT pre-exposed). */
  color: any;
  /** Optical path through the water, m. */
  path: any;
  /** 1 where a background exists under the water, 0 = open water (deep-water path, exactly). */
  valid: any;
  /** Optional contact / shoreline composite, applied after foam and before haze (radiance in, radiance out). */
  contact?: (color: any) => any;
}

export interface OceanMaterialOptions {
  fft: OceanFft;
  sampling: FftSampling;
  /** LOD nodes (lod.nodes). */
  nodes: OceanLodNodes;
  /** Displaced local position (vertex stage). */
  positionNode: any;
  /** Displacement actually applied (local axes, y = up), vertex stage: crest height for the SSS term. */
  displacement: any;
  skyLight?: SkyLight;
  environment?: EnvironmentState;
  /** PMREM (CubeUV) env texture, linear. */
  envMap?: any;
  /** env sample x envIntensity = cd/m^2. Default 1. */
  envIntensity?: number;
  /** TSL node multiplied into the written radiance. */
  preExposure: any;
  /** Foam coverage 0..1 (node, or a builder receiving the surface context). Default: from the FFT compression. */
  foamCoverage?: any | ((ctx: OceanFoamContext) => any);
  /** Transmitted scene (refraction piece, three/refraction.ts) built from the context. */
  transmission?: (ctx: OceanTransmissionContext) => OceanTransmission;
  /** Distance haze. false = none (engine atmosphere does it). Default: reference Gaussian haze to the env horizon. */
  haze?: false | { distance?: any; color?: (dir: any) => any };
  /** Wave height normaliser of the crest glow (reference uWave). Default: expectedHs / 2.575 (Hs at amplitude 1). */
  waveScale?: number;
  /** Shade with the 4th (0.84 m) cascade's normals too (high / ultra). Default false = the reference's 3 cascades. */
  fineNormals?: boolean;
  /** Ocean core (for expectedHs). */
  ocean?: { expectedHs(): number };
  name?: string;
}

export interface OceanMaterial {
  readonly material: any;
  /** Uniform nodes (tunables, all set by update() from the contracts unless overridden). */
  readonly uniforms: {
    sunDir: any; sunE: any; sunSpecularScale: any; hazeDistance: any; waveScale: any; diffuseSH: any;
    envIntensity: any; foamWhite: any;
  };
  /** Copy SKY_LIGHT / ENVIRONMENT into the uniforms. Allocation-free; call once per frame. */
  update(): void;
  /** Override (set) or release (null) a contract-driven value. */
  readonly overrides: { sunSpecularScale: number | null; hazeDistance: number | null; waveScale: number | null };
  dispose(): void;
}

// Reference constants (src/ocean.js).
const DEEP = [0.0159963, 0.0612461, 0.0998987];
const TINT = [0.0159963, 0.1356333, 0.0908417];
const SIGMA = [0.29613827, 0.10461648, 0.09530747];
const F0 = 0.0180094;
const SKY_REFLECTION_BLUR = 0.22;
const SUN_GAIN = 4.5;
const HS_AT_AMP1 = 2.575;

export function createOceanMaterial(o: OceanMaterialOptions): OceanMaterial {
  const { fft, sampling, nodes: n } = o;
  const envI = uniform(o.envIntensity ?? 1);
  const foamWhite = uniform(o.envIntensity ?? 1);
  const sunDir = uniform(new Vector3(-0.6, 0.6, -0.45).normalize());
  const sunE = uniform(new Vector3(4.5, 4.5, 4.5));
  const sunSpecularScale = uniform(1);
  const hazeDistance = uniform(280);
  const waveScale = uniform(o.waveScale ?? 1);
  const diffuseSH = uniform(new Vector3(0.12, 0.2, 0.3));
  const overrides = { sunSpecularScale: null as number | null, hazeDistance: null as number | null, waveScale: o.waveScale ?? null };

  // ---------------------------------------------------------------- geometry inputs (varyings)
  const restLocal = varying(n.localPos.xz);
  const height = varying(o.displacement.y);                   // reference vFoam.z (displaced height, m)
  const tileV = new Map<number, any>(FFT_CASCADES.map((c) => [c.domain, varying(n.tileCoord(c.domain))] as [number, any]));
  const tileF = (P: number) => tileV.get(P) ?? varying(n.tileCoord(P));

  // ---------------------------------------------------------------- derivatives (top level: uniform control flow)
  const footprint = max(length(dFdx(restLocal)), length(dFdy(restLocal)));
  // The reference shades with the 3 reference cascades. The extra 0.84 m cascade (high / ultra) adds sub-0.1 m
  // ripples the reference does not have: it turns the broad glitter blobs into fine sparkle. Opt-in.
  const surf = sampling.sampleSurface(tileF, { footprint, threeCascades: !o.fineNormals });
  const toWorld = (v: any) => normalize(modelWorldMatrix.mul(vec4(v, 0)).xyz);
  const upW = toWorld(vec3(0, 1, 0));
  const surfaceNormal = toWorld(surf.normal);
  const camToP = positionWorld.sub(cameraPosition);
  const dist = length(camToP);
  const V = camToP.negate().div(max(dist, 1e-6));             // reference V = normalize(cam - P)
  const faceV = (nn: any) => normalize(nn.add(V.mul(max(float(0.025).sub(dot(nn, V)), 0))));
  const N = faceV(surfaceNormal);
  const dNx = dFdx(N), dNy = dFdy(N);
  const variance = max(dot(dNx, dNx), dot(dNy, dNy));

  const R = normalize(reflect(V.negate(), N));
  const NoV = max(dot(N, V), 0.015);
  const L = normalize(sunDir);
  const H = normalize(V.add(L));
  const NoH = max(dot(N, H), 0);
  const NoL = max(dot(N, L), 0);
  const rough = clamp(
    float(0.018).add(surf.variance.mul(0.8))
      .add(sqrt(float(1).add(fft.uniforms.unresolvedSlopeVariance)).sub(1).mul(0.8))
      .add(sqrt(variance).mul(0.8)),
    0.018, 0.8);

  // ---------------------------------------------------------------- sun specular (GGX)
  const rs = max(rough, 0.12);
  const a2 = pow(rs, 4);
  const denom = NoH.mul(NoH).mul(a2.sub(1)).add(1);
  const D = a2.div(denom.mul(denom).mul(Math.PI));
  const k = rs.add(1).mul(rs.add(1)).mul(0.125);
  const G = NoV.div(NoV.mul(float(1).sub(k)).add(k)).mul(NoL.div(NoL.mul(float(1).sub(k)).add(k)));
  const sunF = float(F0).add(float(1 - F0).mul(pow(float(1).sub(max(dot(V, H), 0)), 5)));
  const spec = D.mul(G).mul(sunF).div(NoV.mul(4).add(0.04));
  const sunSpec = sunE.mul(min(spec, 22)).mul(sunSpecularScale);  // = sunlight * min(spec,22) * (4.5-3.5s) * vis
  const sunRef = sunE.div(SUN_GAIN);                               // = sunlight() * solarVisibility

  // ---------------------------------------------------------------- environment
  const env = (dir: any, r: any) => (o.envMap ? pmremTexture(o.envMap, dir, r).rgb.mul(envI) : null);
  const diffuseSky = env(upW, float(1)) ?? diffuseSH;

  // ---------------------------------------------------------------- water body (absorption + scatter)
  const deep = vec3(...DEEP);
  const mediumLight = diffuseSky.mul(0.8).add(sunRef.mul(NoL.mul(0.18).add(0.12)));
  let water: any = deep.mul(mediumLight); // reference with refractionValid = 0 (open water / no pre-pass)
  let transmitted: OceanTransmission | null = null;
  if (o.transmission) {
    const refractionNormal = faceV(normalize(mix(surfaceNormal, N, 0.22)));
    transmitted = o.transmission({
      normalWorld: refractionNormal, viewDir: V, positionWorld, upWorld: upW, tileCoord: tileF, restLocal,
      preExposure: o.preExposure, foamWhite,
    });
    const ext = exp(vec3(...SIGMA).negate().mul(transmitted.path));
    // mix with a 0/1 mask (mix(a, b, 0) == a exactly: open water stays bit-identical to the no-refraction path).
    // Not select(): TSL emits an if/else for it, putting the derivative terms below in non-uniform control flow.
    water = mix(water, transmitted.color.mul(ext).add(water.mul(float(1).sub(ext))), transmitted.valid);
  }
  // crest translucency (subsurface glow)
  const crest = smoothstep(-0.3, 0.85, height.div(max(waveScale, 0.1)));
  const backLight = pow(max(dot(V, normalize(L.add(N.mul(0.4))).negate()), 0), 2);
  const crestTransport = crest.mul(backLight.mul(0.6).add(0.12));
  water = water.add(vec3(...TINT).mul(crestTransport).mul(diffuseSky.mul(0.4).add(sunRef.mul(2.2))));

  // ---------------------------------------------------------------- reflection + Fresnel
  const reflected = env(R, min(rough.add(SKY_REFLECTION_BLUR), 1)) ?? diffuseSH;
  const rr = rough.mul(vec4(-1, -0.0275, -0.572, 0.022)).add(vec4(1, 0.0425, 1.04, -0.04));
  const ea = min(rr.x.mul(rr.x), exp2(NoV.mul(-9.28))).mul(rr.x).add(rr.y);
  const brdf = vec2(-1.04, 1.04).mul(ea).add(rr.zw);
  const integratedF = clamp(brdf.x.mul(F0).add(brdf.y), 0, 1);
  let color: any = water.mul(float(1).sub(integratedF)).add(reflected.mul(integratedF)).add(sunSpec);

  // ---------------------------------------------------------------- foam (coverage from the foam piece)
  const foamCtx: OceanFoamContext = {
    tileCoord: tileF, restLocal, footprint, compression: surf.compression, normalLocal: surf.normal, windAB: fft.uniforms.windAB,
  };
  let foam: any;
  if (typeof o.foamCoverage === 'function') foam = o.foamCoverage(foamCtx);
  else if (o.foamCoverage) foam = o.foamCoverage;
  else foam = smoothstep(0.4, 1.0, surf.compression).mul(0.5); // placeholder until the foam piece plugs in
  foam = clamp(foam, 0, 1);
  color = color.sub(sunSpec.mul(float(1).sub(clamp(float(1).sub(foam.mul(2)), 0, 1))));
  const foamColor = vec3(clamp(dot(surfaceNormal, L), 0, 1).mul(0.7).add(0.3)).mul(foamWhite);
  color = mix(color, foamColor, foam);
  // contact / shoreline (refraction piece; reference :337-348): soft intersection + contact foam over rocks.
  if (transmitted?.contact) color = transmitted.contact(color);

  // ---------------------------------------------------------------- distance haze (Gaussian, to the horizon env)
  if (o.haze !== false) {
    const hz = o.haze || {};
    const dHaze = hz.distance ?? hazeDistance;
    const haze = float(1).sub(exp(dist.div(dHaze).pow(2).negate()));
    const horiz = camToP.sub(upW.mul(dot(camToP, upW)));
    const hazeDir = normalize(horiz.add(upW.mul(dist.mul(0.002))));
    const hazeColor = hz.color ? hz.color(hazeDir) : (env(hazeDir, float(0)) ?? diffuseSH);
    color = mix(color, hazeColor, haze);
  }

  const material = new MeshBasicNodeMaterial({ side: DoubleSide });
  material.name = o.name ?? 'ocean.water';
  material.positionNode = o.positionNode;
  material.colorNode = max(color, vec3(0)).mul(o.preExposure);

  // ---------------------------------------------------------------- per-frame contract copy (allocation-free)
  let shVersion = -1;
  function update() {
    const sl = o.skyLight, en = o.environment;
    const sd = en?.sunDir ?? (sl as any)?.sunDir;
    if (sd) (sunDir.value as Vector3).set(sd.x, sd.y, sd.z);
    if (sl) {
      const e = sl.sunIlluminance;
      (sunE.value as Vector3).set(sl.sunColor.r * e, sl.sunColor.g * e, sl.sunColor.b * e);
      if (overrides.hazeDistance === null) hazeDistance.value = sl.fogDensity > 0 ? 1 / sl.fogDensity : 1e9;
      if (sl.version !== shVersion) {
        shVersion = sl.version;
        // Radiance average over the upper hemisphere ~ irradiance(up) / pi from the order-2 SH (y = up).
        const c = sl.skySH;
        const v = diffuseSH.value as Vector3;
        const f = (i: number) => c[0 * 3 + i] * 0.282095 + (2 / 3) * c[2 * 3 + i] * 0.488603 + 0.25 * c[6 * 3 + i] * 0.315392 * 2;
        v.set(Math.max(f(0), 0), Math.max(f(1), 0), Math.max(f(2), 0));
      }
    }
    if (overrides.hazeDistance !== null) hazeDistance.value = overrides.hazeDistance;
    if (overrides.sunSpecularScale !== null) sunSpecularScale.value = overrides.sunSpecularScale;
    else sunSpecularScale.value = 1 - (3.5 / 4.5) * Math.min(Math.max(en?.cloudCover ?? 0, 0), 1);
    if (overrides.waveScale !== null) waveScale.value = overrides.waveScale;
    else if (o.ocean) waveScale.value = o.ocean.expectedHs() / HS_AT_AMP1;
  }
  update();

  return {
    material,
    uniforms: { sunDir, sunE, sunSpecularScale, hazeDistance, waveScale, diffuseSH, envIntensity: envI, foamWhite },
    update,
    overrides,
    dispose() { material.dispose(); },
  };
}
