// Water surface material (three r186, WebGPU / TSL, MeshBasicNodeMaterial with a custom colorNode).
// Port of the WebGL reference fragment shader (src/ocean.js:228-354, docs/port/REFERENCE.md section 2), every term
// in the same order with the same constants. Output: linear HDR radiance (cd/m^2) x preExposure, no tonemap.
//
// Term by term (reference line refs = src/ocean.js):
//   view        V / dist from the VIEW-space position (modelViewMatrix composed on the CPU in float64): exact at any
//               floating-origin offset (no large f32 world positions are subtracted).
//   normal      FFT normal textures, chained cascades, trilinear at the explicit LOD log2(footprint / texel)
//               (:271, spectralSurface :176-182). Footprint = max(|d rest/dx|, |d rest/dy|) like the reference, but
//               clamped to FOOTPRINT_CLAMP x the analytic pixel footprint dist x pixelAngle / |V.up|: across a crest
//               silhouette a 2x2 quad straddles two waves and the derivatives explode (flickering speckle rows).
//               N = facing fix N + V max(.025 - N.V, 0) (:273). R = reflect(-V, N), NoV >= .015 (:274).
//   roughness   .018 + .8 spectralVariance (mip shortening) + .8 (sqrt(1 + slopeVar) - 1) (slopeVar = the FFT's
//               unresolvedSlopeVariance) + .8 sqrt(max(|dN/dx|^2, |dN/dy|^2)), clamped .018..8 (:276-277); the dN term
//               is gated off on silhouette quads (derivative footprint > 2x analytic).
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
//   foam        coverage from the `foamCoverage` input (three/foam.ts); diffuse white of albedo foamAlbedo (0.8) lit by
//               SKY_LIGHT: (skyColor skyIlluminance (.5 + .5 N.up) + sunColor sunIlluminance sat(N.L)) / pi (the
//               reference used an unlit grey .3 + .7 N.L in display units). Sun specular removed under foam (:333-336).
//   haze        Gaussian 1 - exp(-(dist / D)^2), D = 1 / SkyLight.fogDensity (demo: 280 m, storm 220 m), towards
//               the env colour just above the horizon (direction (horizontal, dist .002)) (:350-352). Input.
//   back faces  DoubleSide with the same shader (the reference has no underwater path); the facing fix flips N.
//
// Units (rule 9). Light comes only through SKY_LIGHT / ENVIRONMENT (+ the env map stand-in):
//   E_sun  = SkyLight.sunColor * SkyLight.sunIlluminance (lux) = the reference's sunlight() * 4.5 * sunVisibility in
//            cd/m^2 (4.5 = the reference's sun-disc / specular gain: the sun is a 4.5 RU irradiance).
//            Reference sun terms that use sunlight() * visibility without the 4.5 gain use E_sun / 4.5.
//            No extra cloud dimming here: a sky module lowers sunIlluminance itself (the demo stand-in folds the
//            reference's storm specular dimming into its sunIlluminance).
//   env    = pmremTexture(envMap) * envIntensity (cd/m^2). Without an env map: SkyLight.skySH.
//   foam / contact rim = Lambert radiance from skyIlluminance + sunIlluminance (dark at night).
//   With the demo (REF_NITS, preExposure = 1 / REF_NITS) the water (not foam) equals the reference's pre-tonemap colour.
import { DoubleSide, MeshBasicNodeMaterial, Vector3 } from 'three/webgpu';
import {
  abs, cameraWorldMatrix, clamp, dFdx, dFdy, dot, exp, exp2, float, length, log2, max, min, mix, modelWorldMatrix,
  normalize, pmremTexture, positionView, positionWorld, pow, reflect, smoothstep, sqrt, uniform, varying, vec2, vec3, vec4,
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
  /** Lambert radiance (cd/m^2) of a unit-albedo flat white surface at the fragment, from SKY_LIGHT (sky + sun). */
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
  /** Optional 0/1 mask: opaque geometry in front of the water at this pixel -> output `background` instead. */
  occlusion?: any;
  /** Pre-exposed scene colour at the pixel (used with `occlusion`). */
  background?: any;
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
  /** Diffuse albedo of the foam (lit by SKY_LIGHT). Default 0.8. */
  foamAlbedo?: number;
  /** Debug view: 'rough' (roughness / mip shortening / LOD) or 'footprint' (derivative vs analytic footprint). */
  debug?: 'rough' | 'footprint';
  /** Ocean core (for expectedHs). */
  ocean?: { expectedHs(): number };
  name?: string;
}

export interface OceanMaterial {
  readonly material: any;
  /** Uniform nodes (tunables, all set by update() from the contracts unless overridden). */
  readonly uniforms: {
    sunDir: any; sunE: any; skyE: any; hazeDistance: any; waveScale: any; diffuseSH: any;
    envIntensity: any; foamAlbedo: any;
  };
  /** Copy SKY_LIGHT / ENVIRONMENT into the uniforms. Allocation-free; call once per frame. */
  update(): void;
  /** Override (set) or release (null) a contract-driven value. */
  readonly overrides: { hazeDistance: number | null; waveScale: number | null };
  dispose(): void;
}

/**
 * Material of the LOD's horizon closure (lod.enableClosure): the band between the last ring and the eye-level
 * horizon. At those distances the water is fully hazed, so it is the haze colour = the env radiance just above the
 * horizon in the view direction (the water material's haze target). Never reads scene.color / scene.depth.
 * Radiance x preExposure. (The closure sets positionNode / vertexNode itself.)
 */
export function createOceanHorizonMaterial(o: { envMap?: any; envIntensity?: number; preExposure: any; upNode?: any }): any {
  const up = o.upNode ?? vec3(0, 1, 0);
  const ray = normalize(cameraWorldMatrix.mul(vec4(normalize(positionView), 0)).xyz);
  const horiz = ray.sub(up.mul(dot(ray, up)));
  const dir = normalize(horiz.add(up.mul(length(horiz).mul(0.002))));      // = the water haze direction at distance
  const color = o.envMap ? pmremTexture(o.envMap, dir, float(0)).rgb.mul(o.envIntensity ?? 1) : vec3(0.3, 0.45, 0.6);
  const m = new MeshBasicNodeMaterial({ side: DoubleSide, fog: false });
  m.name = 'ocean.horizon';
  m.colorNode = max(color, vec3(0)).mul(o.preExposure);
  return m;
}

// Reference constants (src/ocean.js).
const DEEP = [0.0159963, 0.0612461, 0.0998987];
const TINT = [0.0159963, 0.1356333, 0.0908417];
const SIGMA = [0.29613827, 0.10461648, 0.09530747];
const F0 = 0.0180094;
const SKY_REFLECTION_BLUR = 0.22;
const SUN_GAIN = 4.5;
const HS_AT_AMP1 = 2.575;
/** Max ratio of the derivative footprint to the analytic pixel footprint before it is treated as a silhouette quad. */
const FOOTPRINT_CLAMP = 2;
/** Diffuse albedo of whitecap foam (fresh dense foam ~0.6-0.9). */
const FOAM_ALBEDO = 0.8;

export function createOceanMaterial(o: OceanMaterialOptions): OceanMaterial {
  const { fft, sampling, nodes: n } = o;
  const envI = uniform(o.envIntensity ?? 1);
  const foamAlbedo = uniform(o.foamAlbedo ?? FOAM_ALBEDO);
  const sunDir = uniform(new Vector3(-0.6, 0.6, -0.45).normalize());
  const sunE = uniform(new Vector3(4.5, 4.5, 4.5));
  const skyE = uniform(new Vector3(1, 1, 1));
  const hazeDistance = uniform(280);
  const waveScale = uniform(o.waveScale ?? 1);
  const diffuseSH = uniform(new Vector3(0.12, 0.2, 0.3));
  const overrides = { hazeDistance: null as number | null, waveScale: o.waveScale ?? null };

  // ---------------------------------------------------------------- geometry inputs (varyings)
  const restLocal = varying(n.localPos.xz);
  const height = varying(o.displacement.y);                   // reference vFoam.z (displaced height, m)
  const tileV = new Map<number, any>(FFT_CASCADES.map((c) => [c.domain, varying(n.tileCoord(c.domain))] as [number, any]));
  const tileF = (P: number) => tileV.get(P) ?? varying(n.tileCoord(P));

  // ---------------------------------------------------------------- view vector (view space: floating-origin exact)
  // positionView = modelViewMatrix (composed on the CPU in float64) x the small local position: no large world-space
  // f32 values are subtracted, so V / dist are identical 100 km from the origin (rule 5).
  const toWorld = (v: any) => normalize(modelWorldMatrix.mul(vec4(v, 0)).xyz);
  const upW = toWorld(vec3(0, 1, 0));
  const pView = positionView;
  const dist = length(pView);
  const rayView = pView.div(max(dist, 1e-6));                         // camera -> fragment, view space
  const V = normalize(cameraWorldMatrix.mul(vec4(rayView.negate(), 0)).xyz); // reference V = normalize(cam - P), world axes
  const camToP = V.negate().mul(dist);

  // ---------------------------------------------------------------- footprint + derivatives (top level: uniform control flow)
  // The reference footprint is max(|d rest/dx|, |d rest/dy|). Across a crest silhouette a 2x2 quad straddles two
  // different waves and that derivative explodes (and so does dN): the LOD and roughness of those quads toggle from
  // frame to frame = rows of flickering single-pixel speckles on distant crest lines. Bound both by the analytic
  // footprint of a pixel on the mean water plane: dist x pixelAngle / |V.up| (the ray direction is continuous
  // across silhouettes, so its derivative gives the pixel angle robustly).
  const pixelAngle = max(length(dFdx(rayView)), length(dFdy(rayView)));
  const cosUp = max(abs(dot(V, upW)), 0.02);
  const footprintAnalytic = dist.mul(pixelAngle).div(cosUp);
  const footprintDeriv = max(length(dFdx(restLocal)), length(dFdy(restLocal)));
  const footprint = min(footprintDeriv, footprintAnalytic.mul(FOOTPRINT_CLAMP));
  // 0 where the quad is coherent, -> 1 where it straddles a silhouette (derivatives meaningless there).
  const straddle = smoothstep(FOOTPRINT_CLAMP, FOOTPRINT_CLAMP * 2, footprintDeriv.div(max(footprintAnalytic, 1e-6)));
  // The reference shades with the 3 reference cascades. The extra 0.84 m cascade (high / ultra) adds sub-0.1 m
  // ripples the reference does not have: it turns the broad glitter blobs into fine sparkle. Opt-in.
  const surf = sampling.sampleSurface(tileF, { footprint, threeCascades: !o.fineNormals });
  const surfaceNormal = toWorld(surf.normal);
  const faceV = (nn: any) => normalize(nn.add(V.mul(max(float(0.025).sub(dot(nn, V)), 0))));
  const N = faceV(surfaceNormal);
  const dNx = dFdx(N), dNy = dFdy(N);
  const variance = max(dot(dNx, dNx), dot(dNy, dNy)).mul(float(1).sub(straddle));

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
  const sunSpec = sunE.mul(min(spec, 22));                         // = sunlight * min(spec,22) * gain * vis
  const sunRef = sunE.div(SUN_GAIN);                               // = sunlight() * solarVisibility

  // ---------------------------------------------------------------- environment
  const env = (dir: any, r: any) => (o.envMap ? pmremTexture(o.envMap, dir, r).rgb.mul(envI) : null);
  const diffuseSky = env(upW, float(1)) ?? diffuseSH;

  // Lambert radiance per unit albedo of a flat sea-level white surface (contact rim, shoreline foam): SKY_LIGHT.
  const foamLightFlat = skyE.add(sunE.mul(max(dot(upW, L), 0))).div(Math.PI);

  // ---------------------------------------------------------------- water body (absorption + scatter)
  const deep = vec3(...DEEP);
  const mediumLight = diffuseSky.mul(0.8).add(sunRef.mul(NoL.mul(0.18).add(0.12)));
  let water: any = deep.mul(mediumLight); // reference with refractionValid = 0 (open water / no pre-pass)
  let transmitted: OceanTransmission | null = null;
  if (o.transmission) {
    const refractionNormal = faceV(normalize(mix(surfaceNormal, N, 0.22)));
    transmitted = o.transmission({
      normalWorld: refractionNormal, viewDir: V, positionWorld, upWorld: upW, tileCoord: tileF, restLocal,
      preExposure: o.preExposure, foamWhite: foamLightFlat,
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
  // Foam is a diffuse white (Lambert) lit by SKY_LIGHT: sky irradiance (up-facing lux, cosine-weighted for the foam
  // normal's tilt) + sun irradiance x N.L. (The reference used an unlit grey .3 + .7 N.L in its display units.)
  const foamLight = skyE.mul(dot(surfaceNormal, upW).mul(0.5).add(0.5))
    .add(sunE.mul(clamp(dot(surfaceNormal, L), 0, 1))).div(Math.PI);    // cd/m^2 per unit albedo
  const foamColor = foamLight.mul(foamAlbedo);
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

  // fog = false: the haze above is the water's own (reference) aerial perspective; with an engine fog pass use
  // haze: false instead of stacking both.
  const material = new MeshBasicNodeMaterial({ side: DoubleSide, fog: false });
  material.name = o.name ?? 'ocean.water';
  material.positionNode = o.positionNode;
  let out: any = max(color, vec3(0)).mul(o.preExposure);
  if (transmitted?.occlusion && transmitted.background) out = mix(out, transmitted.background, transmitted.occlusion);
  if (o.debug === 'rough') {
    // R = roughness / .8, G = mip shortening (spectral variance) x 4, B = LOD of cascade 1 (96 m) / 8. Not exposed.
    const lod1 = max(log2(max(footprint, 1e-6).mul(256 / 96)), 0);
    out = vec3(rough.div(0.8), surf.variance.mul(4), lod1.div(8));
  } else if (o.debug === 'footprint') {
    // R = derivative / analytic footprint / 4, G = sqrt(dN variance) (after the silhouette gate) x 4, B = straddle.
    out = vec3(footprintDeriv.div(max(footprintAnalytic, 1e-6)).div(4), sqrt(variance).mul(4), straddle);
  }
  material.colorNode = out;

  // ---------------------------------------------------------------- per-frame contract copy (allocation-free)
  let shVersion = -1;
  function update() {
    const sl = o.skyLight, en = o.environment;
    const sd = en?.sunDir;
    if (sd) (sunDir.value as Vector3).set(sd.x, sd.y, sd.z);
    if (sl) {
      const e = sl.sunIlluminance;
      (sunE.value as Vector3).set(sl.sunColor.r * e, sl.sunColor.g * e, sl.sunColor.b * e);
      const ks = sl.skyIlluminance;
      (skyE.value as Vector3).set(sl.skyColor.r * ks, sl.skyColor.g * ks, sl.skyColor.b * ks);
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
    if (overrides.waveScale !== null) waveScale.value = overrides.waveScale;
    else if (o.ocean) waveScale.value = o.ocean.expectedHs() / HS_AT_AMP1;
  }
  update();

  return {
    material,
    uniforms: { sunDir, sunE, skyE, hazeDistance, waveScale, diffuseSH, envIntensity: envI, foamAlbedo },
    update,
    overrides,
    dispose() { material.dispose(); },
  };
}
