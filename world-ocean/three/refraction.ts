// Screen-space refraction of the opaque scene under the water (three r186, WebGPU / TSL).
// Port of the reference's refraction pre-pass consumer (src/refraction.js transmittedScene, src/ocean.js:286-290 and
// the contact / shoreline block :337-348; docs/port/REFERENCE.md 2.7, 2.10).
//
// Render-graph shape (spec 5). The water is drawn by its own pass in the 'transparent' stage:
//   ocean.water  stage 'transparent'
//     reads   scene.color   HDR linear radiance x preExposure of the OPAQUE scene (sky, terrain / seabed, objects)
//             scene.depth   its depth buffer (perspective depth, standard or reversed)
//             ocean.displacement (FFT layers, vertex + fragment)
//     writes  scene.color   (the water composited over the opaque scene; depth tested against scene.depth)
//   Host targets (PassContext.targets):
//     'scene.color'              RenderTarget whose .texture is scene.color (single-sample, sampled by the water)
//     'scene.depth'              its DepthTexture (sampled)                     [default: targets['scene.color'].depthTexture]
//     'scene.color.transparent'  optional RenderTarget the transparent stage draws into (e.g. the demo's 4x MSAA HDR
//                                target). When given and different, the pass first resolves scene.color + scene.depth
//                                into it with one full-screen triangle pair (colour + depth), then draws the water.
//                                When absent, the water draws into 'scene.color' itself after copying colour + depth
//                                into ocean-owned textures (ocean.refraction.color / .depth) so it never samples a
//                                texture it is rendering to.
//
// Units / pre-exposure (rule 9, spec 5): scene.color holds radiance x preExposure. The water material works in
// radiance and multiplies its output by preExposure once, so the transmitted scene is divided by the same preExposure
// node here (bottom = scene.color / preExposure). The composite therefore stays consistent for any preExposure.
//
// Water body (reference): opticalPath = view-depth difference (background eye-Z - water eye-Z), clamped 0..120 m;
// UV offset = view-space xy of refract(-V, N, 1/ior) - refract(-V, up, 1/ior) x strength x smoothstep(0,4,thickness)
// x clamp(6 / waterZ, .15, 1), soft-limited, faded at the screen edge, 2 depth-continuity iterations, fallback to the
// unrefracted uv when the displaced sample is in front of the water. Beer-Lambert + in-scattering are applied by the
// material (sigma, deep x mediumLight). Where the background is sky (no opaque geometry: open ocean) valid = 0 and the
// material takes exactly its deep-water path (select, not a multiply: bit-identical to the no-refraction shader).
//
// Contact / shore (reference :337-348): where the background exists, edgeFade = 1 - smoothstep(0, range/2, thickness)
// blends to the raw scene colour (soft intersection) and contact foam smoothstep(thr, thr+.15, pattern.r(p/18)) x .2
// (thr = mix(1, d + .23, smoothstep(0, .3, d)), d = thickness / range) mixes to (.846873,.947307,.982251) x foamWhite.
// TERRAIN_GPU (optional): where scene.depth has no background (the terrain is not in the depth buffer, or beyond it)
// the contact thickness comes from the terrain height grid instead: vertical water depth / max(V.up, .25). Where the
// depth buffer has the ground the reference's view-thickness is used unchanged (parity with the reference).
//
// Floating origin: no world coordinates on the GPU. The terrain grid centre is passed render-relative
// (center - origin, float64 on the CPU), the contact pattern uses the LOD's periodic tile coordinate (period 18 m).
// No per-frame allocation (update() and the pass only copy numbers / swap texture references).
import {
  ClampToEdgeWrapping, DataTexture, DataUtils, DepthTexture, FloatType, HalfFloatType, LinearFilter,
  LinearMipmapLinearFilter, NoColorSpace, NodeMaterial, QuadMesh, RedFormat, RenderTarget, RepeatWrapping, Scene,
  TextureLoader, Vector3,
} from 'three/webgpu';
import {
  abs, cameraFar, cameraNear, cameraViewMatrix, clamp, dot, float, length, max, min, mix, perspectiveDepthToViewZ,
  positionView, positionWorld, refract, screenUV, select, smoothstep, texture, uniform, uv, vec2, vec3, vec4,
} from 'three/tsl';
import type { FloatingOrigin, TerrainGpuHeight, Vec3, WorldShape } from '../stubs/world-core.ts';
import { OCEAN_ASSETS, resolveAsset } from '../assets.ts';
import type { Pass, PassContext } from './pass-types.ts';
import type { OceanTransmission, OceanTransmissionContext } from './material.ts';

export interface OceanRefractionOptions {
  /** Water index of refraction. Default 1.31 (reference uWaterIOR). */
  ior?: number;
  /** Screen-space offset gain. Default 0.25 (reference uRefractionStrength). */
  strength?: number;
  /** Contact / edge-fade range, metres of view thickness. Default 3.95 (reference). */
  contactRange?: number;
  /** Contact foam gain. Default 0.2 (reference). */
  contactGain?: number;
  /** Optional terrain height grid (TERRAIN_GPU). */
  terrainGpu?: TerrainGpuHeight;
  /** Needed with terrainGpu (altitude of the grid centre). */
  shape?: WorldShape;
  origin: FloatingOrigin;
  /** Contact lace (pattern R) sampler; default: the module's foam pattern asset loaded here. */
  contactPattern?: (uv: any) => any;
  /** Asset hook (same as the foam piece). */
  assets?: { resolve?: (id: string) => string | null; foamPattern?: any };
}

export interface OceanRefraction {
  /** The material hook (three/material.ts `transmission`). */
  readonly transmission: (ctx: OceanTransmissionContext) => OceanTransmission;
  readonly uniforms: { ior: any; strength: any; contactRange: any; contactGain: any; terrainEnabled: any };
  /** Bind scene.color / scene.depth textures (the pass does this from ctx.targets; allocation-free). */
  bind(color: any, depth: any): void;
  /** Copy TERRAIN_GPU mapping into uniforms (render-relative centre). Allocation-free; once per frame. */
  update(): void;
  /** Creates the 'ocean.water' transparent pass drawing `waterScene` with `camera`. */
  createPass(o: { waterScene: any; camera: any; perf?: { begin(m: string, p: string): void; end(): void } }): Pass;
  dispose(): void;
}

const NO_GROUND = -1e4; // m: placeholder terrain height (no ground)
const CONTACT_COLOR = [0.846873, 0.947307, 0.982251];

export function createOceanRefraction(o: OceanRefractionOptions): OceanRefraction {
  const ior = uniform(o.ior ?? 1.31);
  const strength = uniform(o.strength ?? 0.25);
  const contactRange = uniform(o.contactRange ?? 3.95);
  const contactGain = uniform(o.contactGain ?? 0.2);
  const disposers: (() => void)[] = [];

  // ------------------------------------------------------------------ scene.color / scene.depth bindings
  const colorPlaceholder = new DataTexture(new Uint16Array(4), 1, 1);
  colorPlaceholder.type = HalfFloatType;
  colorPlaceholder.needsUpdate = true;
  colorPlaceholder.name = 'ocean.refraction.color.placeholder';
  const depthPlaceholder = new DepthTexture(1, 1);
  depthPlaceholder.type = FloatType;
  depthPlaceholder.name = 'ocean.refraction.depth.placeholder';
  const colorNode = texture(colorPlaceholder);
  const depthNode = texture(depthPlaceholder);
  let boundColor: any = null, boundDepth: any = null;
  function bind(color: any, depth: any) {
    if (color && color !== boundColor) { boundColor = color; colorNode.value = color; }
    if (depth && depth !== boundDepth) { boundDepth = depth; depthNode.value = depth; }
  }

  // ------------------------------------------------------------------ terrain grid (TERRAIN_GPU, optional)
  const terrain = o.terrainGpu;
  const terrainPlaceholder = new DataTexture(new Uint16Array([DataUtils.toHalfFloat(NO_GROUND)]), 1, 1, RedFormat, HalfFloatType);
  terrainPlaceholder.needsUpdate = true;
  terrainPlaceholder.name = 'ocean.terrain.placeholder';
  const terrainNode = texture(terrainPlaceholder);
  const terrainEnabled = uniform(terrain ? 1 : 0);
  const tCenter = uniform(new Vector3());   // render-relative centre (m)
  const tEast = uniform(new Vector3(1, 0, 0));
  const tNorth = uniform(new Vector3(0, 0, -1));
  const tUp = uniform(new Vector3(0, 1, 0));
  const tSize = uniform(1);
  const tCenterAlt = uniform(0);
  let terrainVersion = -1;
  const centerRender: Vec3 = { x: 0, y: 0, z: 0 };
  function update() {
    if (!terrain) return;
    if (terrain.version !== terrainVersion) {
      terrainVersion = terrain.version;
      if (terrain.texture && terrainNode.value !== terrain.texture) terrainNode.value = terrain.texture;
      (tEast.value as Vector3).set(terrain.east.x, terrain.east.y, terrain.east.z);
      (tNorth.value as Vector3).set(terrain.north.x, terrain.north.y, terrain.north.z);
      const up = terrain.up;
      if (up) (tUp.value as Vector3).set(up.x, up.y, up.z);
      else (tUp.value as Vector3).crossVectors(tEast.value as Vector3, tNorth.value as Vector3).normalize();
      tSize.value = terrain.size;
      tCenterAlt.value = o.shape ? o.shape.altitude(terrain.center) : terrain.center.y;
    }
    // Render-relative centre in float64 (never the raw world position on the GPU). Done every frame: the origin moves.
    o.origin.toRender(terrain.center, centerRender);
    (tCenter.value as Vector3).set(centerRender.x, centerRender.y, centerRender.z);
  }
  update();

  // ------------------------------------------------------------------ contact pattern (pattern R at p / 18 m)
  let patternTex: any = null;
  const patternPlaceholder = new DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  patternPlaceholder.needsUpdate = true;
  const patternNode = texture(patternPlaceholder);
  if (!o.contactPattern) {
    (async () => {
      try {
        let tex = o.assets?.foamPattern ?? null;
        if (!tex) {
          const url = o.assets?.resolve ? o.assets.resolve(OCEAN_ASSETS.foamPattern) : resolveAsset(OCEAN_ASSETS.foamPattern);
          if (!url) throw new Error(`no URL for ${OCEAN_ASSETS.foamPattern}`);
          tex = await new TextureLoader().loadAsync(url);
        }
        tex.wrapS = RepeatWrapping; tex.wrapT = RepeatWrapping;
        tex.colorSpace = NoColorSpace;
        tex.minFilter = LinearMipmapLinearFilter; tex.magFilter = LinearFilter;
        tex.generateMipmaps = true;
        tex.needsUpdate = true;
        tex.name = 'ocean.contact.pattern';
        patternTex = tex;
        patternNode.value = tex;
      } catch (e) {
        console.warn('[ocean.refraction] foam pattern unavailable: no contact foam', e);
      }
    })();
  }
  const contactRed = o.contactPattern ?? ((c: OceanTransmissionContext) => patternNode.sample(c.tileCoord(18).div(18)).r);

  // ------------------------------------------------------------------ helpers (fragment stage)
  const viewDepthAt = (uvn: any) => perspectiveDepthToViewZ(depthNode.sample(uvn).x, cameraNear, cameraFar).negate();
  const isBackground = (z: any) => z.lessThan(cameraFar.mul(0.99));

  // ------------------------------------------------------------------ material hook
  function transmission(ctx: OceanTransmissionContext): OceanTransmission {
    const V = ctx.viewDir;
    const waterZ = positionView.z.negate();
    const suv = screenUV;
    const baseZ = viewDepthAt(suv);
    const hasBackground = isBackground(baseZ).and(baseZ.greaterThan(waterZ));
    const thickness = max(baseZ.sub(waterZ), 0);

    // Refraction offset (reference refraction.js:14-18). View-space xy; screen uv has y down in WebGPU.
    const eta = float(1).div(ior);
    const bent = refract(V.negate(), ctx.normalWorld, eta);
    const flat = refract(V.negate(), ctx.upWorld, eta);
    const delta = cameraViewMatrix.mul(vec4(bent.sub(flat), 0)).xyz;
    let offset: any = vec2(delta.x, delta.y.negate())
      .mul(strength).mul(smoothstep(0, 4, thickness)).mul(clamp(float(6).div(max(waterZ, 1)), 0.15, 1));
    offset = offset.mul(float(0.035).div(length(offset).add(0.035)));
    const edge = min(min(suv.x, suv.y), min(float(1).sub(suv.x), float(1).sub(suv.y)));
    offset = offset.mul(smoothstep(0, 0.055, edge));
    // Two depth-continuity iterations (refraction.js:19-26).
    for (let i = 0; i < 2; i++) {
      const probe = clamp(suv.add(offset), 0.001, 0.999);
      const probeZ = viewDepthAt(probe);
      const tolerance = thickness.mul(0.12).add(0.45);
      const continuity = min(1, tolerance.div(max(abs(probeZ.sub(baseZ)), 0.0001)));
      const behind = smoothstep(waterZ.add(0.02), waterZ.add(0.2), probeZ);
      offset = offset.mul(continuity.mul(behind));
    }
    const displaced0 = clamp(suv.add(offset), 0.001, 0.999);
    const displacedZ0 = viewDepthAt(displaced0);
    const fallback = isBackground(displacedZ0).not().or(displacedZ0.lessThanEqual(waterZ.add(0.03)));
    const displaced = select(fallback, suv, displaced0);
    const displacedZ = select(fallback, baseZ, displacedZ0);
    const path = clamp(displacedZ.sub(waterZ), 0, 120);
    const color = colorNode.sample(displaced).rgb.div(ctx.preExposure);

    // Contact / shoreline (reference :337-348), applied by the material after foam, before haze.
    const contact = (c: any) => {
      // TERRAIN_GPU thickness (only where the depth buffer has no background).
      const rel = positionWorld.sub(tCenter);
      const tuv = vec2(dot(rel, tEast), dot(rel, tNorth)).div(tSize).add(0.5);
      const inside = tuv.x.greaterThan(0).and(tuv.x.lessThan(1)).and(tuv.y.greaterThan(0)).and(tuv.y.lessThan(1));
      const ground = terrainNode.sample(tuv).r;
      const waterAlt = tCenterAlt.add(dot(rel, tUp));
      const terrainThickness = waterAlt.sub(ground).div(max(dot(V, ctx.upWorld), 0.25));
      const terrainValid = terrainEnabled.greaterThan(0.5).and(inside).and(ground.greaterThan(NO_GROUND * 0.5));
      const t = select(hasBackground, thickness, terrainThickness);
      const active = hasBackground.or(terrainValid);
      const range = contactRange;
      const d = t.div(range.add(0.001));
      const thr = mix(float(1), d.add(0.23), smoothstep(0, 0.3, d));
      const red = contactRed(ctx);
      const foam = smoothstep(thr, thr.add(0.15), red).mul(contactGain);
      const edgeFade = select(hasBackground, float(1).sub(smoothstep(0, range.mul(0.5), thickness)), float(0));
      const raw = colorNode.sample(suv).rgb.div(ctx.preExposure);
      const mixed = mix(mix(c, raw, edgeFade), vec3(...CONTACT_COLOR).mul(ctx.foamWhite), foam);
      return select(active, mixed, c);
    };

    return { color, path, valid: select(hasBackground, float(1), float(0)), contact };
  }

  // ------------------------------------------------------------------ the transparent-stage pass
  function createPass(po: { waterScene: any; camera: any; perf?: { begin(m: string, p: string): void; end(): void } }): Pass {
    // Resolve quad: scene.color + scene.depth -> the transparent target (colour + depth), used when the host gives
    // the transparent stage its own (e.g. MSAA) target.
    const resolveMat = new NodeMaterial();
    resolveMat.name = 'ocean.water.resolve';
    resolveMat.colorNode = colorNode.sample(uv());
    resolveMat.depthNode = depthNode.sample(uv()).x;
    resolveMat.depthTest = true;
    resolveMat.depthWrite = true;
    const resolveQuad = new QuadMesh(resolveMat);
    // Copies for the single-target path.
    let copyRT: any = null;
    let copyInit = false;
    const ensureCopy = (w: number, h: number) => {
      if (copyRT && copyRT.width === w && copyRT.height === h) return;
      copyRT?.dispose();
      copyInit = false;
      copyRT = new RenderTarget(w, h, { type: HalfFloatType, depthBuffer: true });
      copyRT.texture.name = 'ocean.refraction.color';
      copyRT.depthTexture = new DepthTexture(w, h);
      copyRT.depthTexture.type = FloatType;
      copyRT.depthTexture.name = 'ocean.refraction.depth';
      copyRT.texture.minFilter = LinearFilter; copyRT.texture.magFilter = LinearFilter;
      copyRT.texture.wrapS = ClampToEdgeWrapping; copyRT.texture.wrapT = ClampToEdgeWrapping;
    };
    disposers.push(() => { resolveMat.dispose(); copyRT?.dispose(); });
    return {
      desc: {
        id: 'ocean.water', module: 'world/ocean', stage: 'transparent',
        reads: ['scene.color', 'scene.depth', 'ocean.displacement'], writes: ['scene.color'],
      },
      execute(ctx: PassContext) {
        const r = ctx.renderer;
        const src = ctx.targets['scene.color'];
        const depth = ctx.targets['scene.depth'] ?? src?.depthTexture;
        const dst = ctx.targets['scene.color.transparent'] ?? src;
        update();
        const autoClear = r.autoClear;
        if (dst !== src) {
          bind(src.texture, depth);
          r.setRenderTarget(dst);
          r.autoClear = true;
          resolveQuad.render(r);              // clear + scene.color / scene.depth into dst
        } else {
          // Same target: never sample what we render to. Copy colour + depth into ocean-owned textures first.
          ensureCopy(src.width, src.height);
          if (!copyInit) { r.initRenderTarget(copyRT); copyInit = true; }
          r.copyTextureToTexture(src.texture, copyRT.texture);
          r.copyTextureToTexture(depth, copyRT.depthTexture);
          bind(copyRT.texture, copyRT.depthTexture);
          r.setRenderTarget(dst);
        }
        r.autoClear = false;
        r.render(po.waterScene, po.camera);
        r.autoClear = autoClear;
      },
    };
  }

  return {
    transmission,
    uniforms: { ior, strength, contactRange, contactGain, terrainEnabled },
    bind,
    update,
    createPass,
    dispose() {
      colorPlaceholder.dispose(); depthPlaceholder.dispose(); terrainPlaceholder.dispose(); patternPlaceholder.dispose();
      patternTex?.dispose();
      for (const d of disposers) d();
      disposers.le