// Ocean foam (whitecaps): persistent foam history (compute, ping-pong) + the reference's foam-pattern threshold.
// three r186, WebGPU compute + TSL only. Port of src/ocean.js foamFragment (:355-367), updateFoam (:924-984) and
// the water shader's foam block (:299-317); see docs/port/REFERENCE.md §1.5 and §2.9.
//
// =====================================================================================================
// MODEL (reference numbers unchanged)
//   * History: a 400 m x 400 m window around an anchor, R x R texels (R per tier, see FOAM_TIERS), f32 ping-pong
//     storage buffer + an rgba16float output texture that the material samples (R = history, G = this frame's
//     source). Each frame, per texel at its REST position x:
//        src = .99 smoothstep(.15, .5, compression) + 1.10 sqrt(max(dot(n.xz, wind), 0))
//        history = mix(src, old, exp(-dt / 1.48 s))                       (exponential moving average, tau 1.48 s)
//     compression = max(sum_c (1 - normal_c.w), 0) and n = normalize(sum_c n_c.xz, 1): the FFT normal textures
//     (fft.ts: .w = 1 - compression, compression = clamp(1 - minEig(J)) masked to DOWNWIND-facing slopes,
//     smoothstep(0, .2, n.wind), exactly the reference blend pass), point-sampled bilinearly at LOD 0 like the
//     reference foamSurface, NOT chained (the reference samples rest/domain + .5 for every cascade).
//     Cascades: the 3 reference cascades always; the 4th (0.84 m, high/ultra) is drawn but NOT summed into the
//     source by default (`fineInSource` = 0, see "Tier invariance" below; measured, not assumed).
//   * Anchor: the camera's look ray hit on sea level, offset clamped to +-152 m per tangent axis, snapped to the
//     history cell (400 m / R) in WORLD lattice coordinates (float64 on the CPU: anchor = integer cell index x cell).
//     A snap moves the window by an INTEGER number of texels, so re-projection is an exact texel shift (no
//     resampling blur, no smearing); texels that enter the window take the instantaneous source (the reference
//     used 0 there, which only shows beyond the 152 m edge fade). A jump > 200 m (teleport) or a backwards / > 2 s
//     time step warm-starts the history at the source (steady state of the EMA; the reference cleared to 0 and
//     faded in over ~1.5 s).
//   * Time: dt = this frame's draw time - last frame's draw time (PassContext.time = FrameInfo.interpolatedTime,
//     simulation seconds), never wall clock. Paused (dt = 0): history held, only re-projected if the anchor snaps.
//     Deterministic for a fixed step; frame-rate independent (the EMA uses exp(-dt / tau)). The reference also
//     clamped dt to 0.05 s (identical at >= 20 fps; not done here so slow frame rates keep the same decay).
//   * Coverage (material, fragment): the reference foam block, unchanged:
//        history  = texture(history, w / 400 + .5).r * edge,   w = rest - anchor,  edge = prod(1 - smoothstep(152, 200, |w|))
//        red      = pattern(breakingUV).r,  breakingUV = (wind dot(x, wind) .53 + cross dot(x, cross)) / 9.1
//                   (lace stretched 1.9x along the wind: a 17.2 m x 9.1 m tile)
//        surface  = pattern(x / 30).g
//        breaking = clamp(red smoothstep(t, t + .6, red) .6, 0, 1),  t = 1 - 1.6 max(history, 0)
//        foam     = clamp(breaking + surface smoothstep(.79, .94, surface) .3, 0, 1)
//     x = the REST (undisplaced) position: the pattern is fixed on the water's rest grid. NO DRIFT: the reference
//     computes uFoamOffset but never uses it (REFERENCE.md §0 item 10d), and the checklist (F16) lists "the pattern
//     drifting or swimming" as a failure. The lace moves with the rest grid only, streaks come from the history.
//   * Floating origin: nothing world-sized reaches the GPU. The anchor and the camera foot are float64 lattice
//     coordinates on the CPU; the material gets anchor - foot (small) and builds w from lod.nodes.localPos.xz
//     (= rest - foot, small). Pattern phases are fract(M foot + c) (float64) per tile, the source pass gets
//     anchor mod P per FFT tile. Identical at 100 km (or 1e7 m) from the origin.
//   * Wind: direction from fft.uniforms.windAB (core state, towards, east/south). The pattern's direction follows
//     it with a 1 s low-pass and is re-pivoted at the camera foot when it turns (no far-field sliding).
//
// Tier invariance (history resolution scales, the look does not):
//   * The source is the same function on every tier (c0..c2 at LOD 0, the reference's look incl. its c2
//     point sampling); R only changes how finely it is stored. FOAM_TIERS keeps >= 512 so the bilinear history
//     stays sharper than the 9.1 m lace tile (256 would visibly soften crest edges; measured below).
//   * c3 (0.84 m tile, lambda 1-10 cm, high/ultra only) has slope and strain variance ~= c2's: summing it into
//     the source raises the compression term by a biased positive amount and the downwind slope term by its
//     noise, i.e. ~+X % more foam on high/ultra than on low/medium (see the measured table in the handoff).
//     Capillary ripples do not make whitecaps and the reference never had them, so the source excludes c3 by
//     default (`fineInSource` uniform, 0). Set foam.uniforms.fineInSource.value = 1 to include it.
//
// =====================================================================================================
// API
//   const foam = createOceanFoam({ renderer, fft, lod, origin, shape?, tier?, camera?, assets?, perf? });
//   runner/graph.add(foam.pass);   // stage 'compute', id 'ocean.foam', AFTER 'ocean.fft' (reads ocean.normal)
//   per frame: foam.update(camera) (or pass `camera` in the options) before the pass runs; lod.update(camera)
//              must have run this frame before foam.pass (createOceanGpu's fft pass does it).
//   await foam.ready;             // pattern loaded + first tier compiled (the pass is a no-op until then)
//   foam.setTier('low');           // async compile, old tier keeps running, history resampled across the switch
//   foam.dispose();
//
// MATERIAL INTEGRATION (exact recipe; 2 texture bindings: history + pattern, 2 samplers)
//   // fragment-stage rest position in the LOD's local tangent frame (x east, z south), metres:
//   const rest = varying(lod.nodes.localPos.xz);             // the SAME varying the material uses for its rest pos
//   const f = foam.nodes.layers(rest);                        // { foam, breaking, surface, history, red }
//   // (or foam.nodes.coverage(rest) for f.foam alone)
//   // reference composition (src/ocean.js :333-336), after sun specular + reflection, before contact / haze:
//   color = color - sunSpecular * (1 - clamp(1 - 2 * f.foam, 0, 1));          // no specular on foam
//   const foamColor = vec3(float(.3).add(clamp(dot(Nshade, L), 0, 1).mul(.7)));  // GREY, NOT sun/sky tinted;
//                       // Nshade = the shading normal before the view-facing fix, L = direction to the sun;
//                       // same radiance units as the reference colour (x envIntensity x preExposure like the rest)
//   color = mix(color, foamColor, f.foam);
//   // contact foam (reference :343, pattern.r at p / 18): foam.nodes.pattern(rest, 18).r — same texture binding.
//   // the material must keep foam.nodes.coverage OUTSIDE non-uniform branches (implicit-derivative samples).
//
// Debug view (demo only): attachFoamDebugView(...) below, demo ?debug=foam (simple water + reference foam) or
// ?debug=foamraw (R = history, G = final coverage, B = window mask).
import {
  ClampToEdgeWrapping, DataTexture, DoubleSide, HalfFloatType, LinearFilter, LinearMipmapLinearFilter,
  MeshBasicNodeMaterial, NoColorSpace, RGBAFormat, RepeatWrapping, StorageBufferAttribute, StorageTexture,
  TextureLoader, Vector2, Vector3,
} from 'three/webgpu';
import {
  Fn, If, abs, clamp, dFdx, dFdy, dot, exp, float, globalId, int, max, mix, modelWorldMatrix, normalize, pmremTexture,
  select, smoothstep, sqrt, storage, texture, textureStore, uint, uniform, uvec2, varying, vec2, vec3, vec4,
  cameraPosition, positionWorld, reflect, pow,
} from 'three/tsl';
import type { FloatingOrigin, TangentFrame, Vec3, WorldShape } from '../stubs/world-core.ts';
import type { Pass, PassContext } from './pass-types.ts';
import type { OceanFft } from './fft.ts';
import { BASE_SIZE, FFT_CASCADES, createFftSampling } from './fft.ts';
import type { OceanLod } from './lod.ts';
import { OCEAN_ASSETS, resolveAsset } from '../assets.ts';

export type FoamTier = 'low' | 'medium' | 'high' | 'ultra';

/** History resolution per tier (texels per side of the 400 m window). Cell = 400 / R metres. */
export const FOAM_TIERS: Readonly<Record<FoamTier, number>> = Object.freeze({
  low: 512, medium: 1024, high: 1024, ultra: 1024,
});

/** Reference constants (src/ocean.js), metres / seconds. Changing any of these changes the look. */
export const FOAM_PARAMS = Object.freeze({
  /** History window side, m. */
  window: 400,
  /** Max look-ahead of the anchor from the camera, m per tangent axis. */
  anchorLead: 152,
  /** Anchor jump that resets the history (teleport), m per axis. */
  resetJump: 200,
  /** Window edge fade (|rest - anchor| per axis), m. */
  edgeStart: 152, edgeEnd: 200,
  /** EMA time constant, s. */
  tau: 1.48,
  /** Source: gain * smoothstep(lo, hi, compression) + slopeGain * sqrt(downwind slope). */
  compressionGain: 0.99, compressionLo: 0.15, compressionHi: 0.5, slopeGain: 1.1,
  /** Threshold = 1 - thresholdGain * history; breaking = red * smoothstep(t, t + softness, red) * breakingGain. */
  thresholdGain: 1.6, breakingSoftness: 0.6, breakingGain: 0.6,
  /** Breaking lace tile: 9.1 m across the wind, 9.1 / 0.53 = 17.2 m along it (1.9x stretch). */
  laceTile: 9.1, laceAlong: 0.53,
  /** Surface lace (pattern G): tile m, smoothstep lo / hi, gain. */
  surfaceTile: 30, surfaceLo: 0.79, surfaceHi: 0.94, surfaceGain: 0.3,
  /** Pattern wind-direction low-pass, s. */
  windTau: 1,
});

export interface OceanFoamOptions {
  renderer: any;
  fft: OceanFft;
  lod: OceanLod;
  origin: FloatingOrigin;
  /** Default: flat (east +X, south +Z, up +Y). Should be world.shape. */
  shape?: WorldShape;
  tier?: FoamTier;
  /** Camera used for the anchor (or call foam.update(camera) every frame). */
  camera?: any;
  /** The core ocean (createOcean()): lets the source keep the reference statistics on the low / ultra FFT
   *  detail grids (128 / 512 instead of 256). Without it those two tiers get ~9 % less / ~6 % more foam. */
  ocean?: { getState(): any; blendAt(t: number): number };
  /** Asset hook: URL resolver for logical ids (default assets.ts resolveAsset) or a preloaded pattern texture. */
  assets?: { resolve?: (id: string) => string | null; foamPattern?: any };
  perf?: { begin(module: string, phase: string): void; end(): void };
}

export interface FoamLayers {
  /** Final foam coverage 0..1 (breaking + surface): what the reference mixes the grey foam colour with. */
  foam: any;
  /** Whitecap part (history-thresholded lace). */
  breaking: any;
  /** Sparse static surface lace part. */
  surface: any;
  /** Edge-faded history value (>= 0, ~0..2). */
  history: any;
  /** Raw breaking-lace pattern value (pattern R). */
  red: any;
}

export interface OceanFoamNodes {
  /** Foam coverage 0..1 at a rest position (vec2, LOD local tangent frame: x east, y south, m). */
  coverage(rest: any): any;
  /** All foam terms at a rest position (one history + two pattern samples). */
  layers(rest: any): FoamLayers;
  /** Foam pattern (rgba) sampled at rest / period, world-anchored (floating-origin safe). One texture binding. */
  pattern(rest: any, period: number): any;
  /** History texture node (rgba16float: R history, G source). Its .value swaps on tier change. */
  readonly historyTexture: any;
  /** Pattern texture node. Its .value swaps when the PNG has loaded. */
  readonly patternTexture: any;
}

export interface OceanFoam {
  readonly pass: Pass;
  readonly nodes: OceanFoamNodes;
  /** Pattern loaded and first tier compiled. */
  readonly ready: Promise<void>;
  readonly tier: FoamTier;
  readonly pendingTier: FoamTier | null;
  readonly uniforms: {
    /** anchor - LOD camera foot (east, south), m. */
    readonly anchorRel: any;
    /** 1 = sum the 4th (0.84 m) cascade into the source (default 0, see header). */
    readonly fineInSource: any;
    /** c2 gain keeping the 256-grid source statistics on the 128 / 512 detail grids (1 on medium / high). */
    readonly detailGain: any;
  };
  /** Store the camera for this frame's anchor (allocation-free). */
  update(camera: any): void;
  setTier(t: FoamTier): void;
  readonly debug: {
    /** Current history (R x R floats, row = south index) read back asynchronously. */
    readHistory(): Promise<Float32Array>;
    resolution(): number;
    /** Anchor in world lattice coordinates (east, south), m. */
    anchor(out: { x: number; y: number }): { x: number; y: number };
    memoryBytes(): number;
    lastCompileMs(): number;
    /** False until the foam pattern PNG has loaded (no whitecaps without it). */
    patternLoaded(): boolean;
    /** id of the active compute node (matches three's timestamp query uids 'c:<n>:<id>:f<frame>'). */
    kernelId(): number;
    /** Dispatches executed (frames the history advanced). */
    readonly steps: number;
  };
  dispose(): void;
}

/** Minimal flat shape when none is passed. */
function flatShape(): WorldShape {
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

const fract64 = (x: number) => x - Math.floor(x);
const mod64 = (x: number, p: number) => { const r = x % p; return r < 0 ? r + p : r; };

interface FoamTierState {
  readonly tier: FoamTier;
  readonly res: number;
  readonly cell: number;
  readonly histAttr: any;
  readonly out: any;
  readonly kernel: any;
  readonly bytes: number;
}

export function createOceanFoam(opts: OceanFoamOptions): OceanFoam {
  const { renderer, fft, lod } = opts;
  const origin = opts.origin;
  const shape = opts.shape ?? flatShape();
  const P = FOAM_PARAMS;
  const C = FFT_CASCADES;

  // ------------------------------------------------------------------ shared uniforms
  const uAnchorMod = [0, 1, 2, 3].map(() => uniform(new Vector2()));   // anchor mod tile P_c (east, south), m
  const uShift = uniform(new Vector2());        // texel shift: new texel i reads old texel i + shift
  const uRetention = uniform(0);                // exp(-dt / tau), 0 = warm reset
  const uPrevBase = uniform(0, 'uint');
  const uCurBase = uniform(0, 'uint');
  const uResample = uniform(0);                 // 1 on the first frame of a new tier: old = previous tier's texture
  const uResampleOffset = uniform(new Vector2()); // new anchor - old tier's anchor (m)
  const fineInSource = uniform(0);
  const uDetailGain = uniform(1);
  const anchorRel = uniform(new Vector2());     // anchor - lod foot, m (material)
  const uAlong = uniform(new Vector2(0.540302306, 0.841470985)); // pattern wind (east, south)
  const uLacePhase = uniform(new Vector2());    // fract(M foot + c)
  const uSurfacePhase = uniform(new Vector2()); // fract(foot / surfaceTile)
  const windAB = fft.uniforms.windAB;

  // Placeholder textures: distinct objects (three merges texture nodes sharing a texture into one binding).
  const prevTierPlaceholder = new StorageTexture(4, 4);
  prevTierPlaceholder.type = HalfFloatType; prevTierPlaceholder.format = RGBAFormat;
  prevTierPlaceholder.minFilter = LinearFilter; prevTierPlaceholder.magFilter = LinearFilter;
  prevTierPlaceholder.generateMipmaps = false;
  prevTierPlaceholder.name = 'ocean.foam.placeholder';
  const prevTierNode = texture(prevTierPlaceholder);
  const patternPlaceholder = new DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  patternPlaceholder.needsUpdate = true;
  patternPlaceholder.name = 'ocean.foam.pattern.placeholder';

  // ------------------------------------------------------------------ source (shared by every tier's kernel)
  /** Instantaneous foam source at window offset o (east, south, m from the anchor). Reference foamSurface + source. */
  function sourceAt(o: any) {
    let slope: any = null, comp: any = null;
    for (let c = 0; c < 4; c++) {
      const tc = vec2(uAnchorMod[c]).add(o);
      const half = c < 2 ? float(0.5 / BASE_SIZE) : fft.uniforms.detailSize.reciprocal().mul(0.5);
      const uvc = tc.add(C[c].offset).div(C[c].domain).add(half);
      const tex = c < 2 ? fft.nodes.normalBase : fft.nodes.normalDetail;
      const n = tex.sample(uvc).depth(int(c & 1)).level(0);
      let s: any = vec2(n.x, n.y.negate());                  // (n.east, n.south)
      let k: any = float(1).sub(n.w);                          // compression_c (downwind-masked)
      // c2 on a detail grid other than the reference's 256 (low 128, ultra 512): rescaled to the 256-grid
      // finite-difference slope variance (uDetailGain, computed from the spectrum, see detailGainFor).
      if (c === 2) { s = s.mul(uDetailGain); k = k.mul(uDetailGain); }
      if (c === 3) { const g = fft.uniforms.fineEnabled.mul(fineInSource); s = s.mul(g); k = k.mul(g); }
      slope = slope === null ? s : slope.add(s);
      comp = comp === null ? k : comp.add(k);
    }
    const nrm = normalize(vec3(slope.x, 1, slope.y));
    const downwind = max(dot(vec2(nrm.x, nrm.z), windAB), 0);
    return smoothstep(P.compressionLo, P.compressionHi, max(comp, 0)).mul(P.compressionGain).add(sqrt(downwind).mul(P.slopeGain));
  }

  function buildTier(t: FoamTier): FoamTierState {
    const R = FOAM_TIERS[t];
    if (R % 8 !== 0) throw new RangeError(`foam: resolution ${R} must be a multiple of 8`);
    const cell = P.window / R;
    const histAttr = new StorageBufferAttribute(new Float32Array(2 * R * R), 1);
    histAttr.name = `ocean.foam.history${R}`;
    const hist = storage(histAttr, 'float', 2 * R * R);
    const out = new StorageTexture(R, R);
    out.type = HalfFloatType; out.format = RGBAFormat;
    out.minFilter = LinearFilter; out.magFilter = LinearFilter;
    out.wrapS = ClampToEdgeWrapping; out.wrapT = ClampToEdgeWrapping;
    out.generateMipmaps = false;
    out.name = `ocean.foam.${R}`;
    const kernel = Fn(() => {
      const i = globalId.x, j = globalId.y;
      const o = vec2(float(i), float(j)).add(0.5 - R / 2).mul(cell).toVar();
      const src = sourceAt(o).toVar();
      const old = src.toVar();
      If(uResample.greaterThan(0.5), () => {
        const uvOld = o.add(uResampleOffset).div(P.window).add(0.5).toVar();
        const inside = uvOld.x.greaterThanEqual(0).and(uvOld.x.lessThanEqual(1)).and(uvOld.y.greaterThanEqual(0)).and(uvOld.y.lessThanEqual(1));
        If(inside, () => { old.assign(prevTierNode.sample(uvOld).level(0).x); });
      }).Else(() => {
        const pi = int(i).add(int(uShift.x)), pj = int(j).add(int(uShift.y));
        const inside = pi.greaterThanEqual(0).and(pi.lessThan(R)).and(pj.greaterThanEqual(0)).and(pj.lessThan(R));
        If(inside, () => { old.assign(hist.element(uPrevBase.add(uint(pj.mul(R).add(pi))))); });
      });
      const h = max(mix(src, old, uRetention), 0);
      hist.element(uCurBase.add(j.mul(uint(R)).add(i))).assign(h);
      textureStore(out, uvec2(i, j), vec4(h, src, 0, 1));
    })().compute([R / 8, R / 8, 1], [8, 8, 1]);
    const bytes = 2 * R * R * 4 + R * R * 8;
    return { tier: t, res: R, cell, histAttr, out, kernel, bytes };
  }
  function disposeTier(ts: FoamTierState) { ts.out.dispose(); ts.kernel.dispose?.(); }

  // ------------------------------------------------------------------ tiers (async compile, old tier live)
  let requested: FoamTier = opts.tier ?? 'high';
  let active: FoamTierState | null = null;
  let pending: FoamTierState | null = null;
  let pendingTier: FoamTier | null = null;
  let retired: FoamTierState | null = null;
  let retireFrames = 0;
  let compileMs = 0;
  const first = buildTier(requested);
  const historyNode = texture(first.out);
  const patternNode = texture(patternPlaceholder);

  function startBuild(t: FoamTier, prebuilt: FoamTierState | null) {
    pendingTier = t;
    const ts = prebuilt ?? buildTier(t);
    const t0 = performance.now();
    return renderer.compileComputeAsync([ts.kernel]).then(() => {
      compileMs = performance.now() - t0;
      if (pendingTier !== t) { disposeTier(ts); return; }
      pending = ts;
    }, (e: unknown) => { console.error('[ocean.foam] tier build failed', e); if (pendingTier === t) pendingTier = null; disposeTier(ts); });
  }
  const firstBuild = startBuild(requested, first);

  // ------------------------------------------------------------------ pattern (asset hook)
  let patternLoaded = false;
  let patternTex: any = null;
  const patternReady: Promise<void> = (async () => {
    try {
      let tex = opts.assets?.foamPattern ?? null;
      if (!tex) {
        const url = opts.assets?.resolve ? opts.assets.resolve(OCEAN_ASSETS.foamPattern) : resolveAsset(OCEAN_ASSETS.foamPattern);
        if (!url) throw new Error(`no URL for ${OCEAN_ASSETS.foamPattern}`);
        tex = await new TextureLoader().loadAsync(url);
      }
      tex.wrapS = RepeatWrapping; tex.wrapT = RepeatWrapping;
      tex.colorSpace = NoColorSpace;
      tex.minFilter = LinearMipmapLinearFilter; tex.magFilter = LinearFilter;
      tex.generateMipmaps = true;
      tex.needsUpdate = true;
      tex.name = 'ocean.foam.pattern';
      patternTex = tex;
      patternNode.value = tex;
      patternLoaded = true;
    } catch (e) {
      console.warn('[ocean.foam] foam pattern unavailable: whitecaps disabled (no procedural fallback)', e);
    }
  })();
  let fftReady = false;
  void fft.ready.then(() => { fftReady = true; });
  // (does not wait for fft.ready: that resolves only once the FFT pass has run, i.e. inside the frame loop)
  const ready = Promise.all([patternReady, firstBuild]).then(() => undefined);

  // ------------------------------------------------------------------ material nodes
  const periods: number[] = [];
  const phases: any[] = [];
  function phaseFor(period: number) {
    let i = periods.indexOf(period);
    if (i < 0) { i = periods.length; periods.push(period); phases.push(uniform(new Vector2())); updatePhases(); }
    return phases[i];
  }
  function patternAt(rest: any, period: number) {
    if (!(period > 0)) throw new RangeError('foam.nodes.pattern: period must be > 0');
    return patternNode.sample(vec2(rest).div(period).add(phaseFor(period)));
  }
  function layers(rest: any): FoamLayers {
    const r = vec2(rest);
    const w = r.sub(anchorRel);
    const uvH = clamp(w.div(P.window).add(0.5), 0.001, 0.999);
    const e = float(1).sub(smoothstep(P.edgeStart, P.edgeEnd, abs(w)));
    const history = max(historyNode.sample(uvH).x, 0).mul(e.x.mul(e.y));
    const along = vec2(uAlong), cross = vec2(uAlong.y.negate(), uAlong.x);
    const laceUV = along.mul(dot(r, along).mul(P.laceAlong / P.laceTile)).add(cross.mul(dot(r, cross).div(P.laceTile))).add(uLacePhase);
    const red = patternNode.sample(laceUV).x;
    const surfaceRed = patternNode.sample(r.div(P.surfaceTile).add(uSurfacePhase)).y;
    const threshold = float(1).sub(history.mul(P.thresholdGain));
    const breaking = clamp(red.mul(smoothstep(threshold, threshold.add(P.breakingSoftness), red)).mul(P.breakingGain), 0, 1);
    const surface = surfaceRed.mul(smoothstep(P.surfaceLo, P.surfaceHi, surfaceRed)).mul(P.surfaceGain);
    return { foam: clamp(breaking.add(surface), 0, 1), breaking, surface, history, red };
  }
  const nodes: OceanFoamNodes = {
    coverage: (rest) => layers(rest).foam,
    layers,
    pattern: patternAt,
    historyTexture: historyNode,
    patternTexture: patternNode,
  };

  // ------------------------------------------------------------------ per-frame CPU state (preallocated)
  let camera: any = opts.camera ?? null;
  const camR: Vec3 = { x: 0, y: 0, z: 0 }, camW: Vec3 = { x: 0, y: 0, z: 0 }, footW: Vec3 = { x: 0, y: 0, z: 0 };
  const frame: TangentFrame = { east: { x: 1, y: 0, z: 0 }, north: { x: 0, y: 0, z: -1 }, up: { x: 0, y: 1, z: 0 } };
  let initialized = false;
  let lastTime = 0;
  let anchorIU = 0, anchorIV = 0;       // integer cell indices (float64)
  let anchorU = 0, anchorV = 0;         // m, world lattice
  let parity = 0;
  let steps = 0;
  // pattern wind state (float64)
  let windAngle = NaN;
  let laceCU = 0, laceCV = 0;           // phase constant c (fract-reduced)
  let laceM00 = 0, laceM01 = 0, laceM10 = 0, laceM11 = 0;

  function laceMatrix(angle: number) {
    // uv = along (x.along) k1 + cross (x.cross) k2, along = (ca, sa), cross = (-sa, ca)  (east, south)
    const ca = Math.cos(angle), sa = Math.sin(angle);
    const k1 = P.laceAlong / P.laceTile, k2 = 1 / P.laceTile;
    laceM00 = ca * ca * k1 + sa * sa * k2; laceM01 = ca * sa * (k1 - k2);
    laceM10 = laceM01;                    laceM11 = sa * sa * k1 + ca * ca * k2;
  }
  function updatePhases() {
    const fu = lod.state.footU, fv = lod.state.footV;
    (uLacePhase.value as Vector2).set(
      fract64(laceM00 * fu + laceM01 * fv + laceCU),
      fract64(laceM10 * fu + laceM11 * fv + laceCV),
    );
    (uSurfacePhase.value as Vector2).set(fract64(fu / P.surfaceTile), fract64(fv / P.surfaceTile));
    for (let i = 0; i < periods.length; i++) (phases[i].value as Vector2).set(fract64(fu / periods[i]), fract64(fv / periods[i]));
  }
  function updateWind(dt: number, reset: boolean) {
    const w = windAB.value as Vector2;
    const target = Math.atan2(w.y, w.x);
    let next: number;
    if (reset || !(windAngle === windAngle)) next = target;
    else {
      let d = target - windAngle;
      d -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
      next = windAngle + d * (1 - Math.exp(-Math.max(dt, 0) / P.windTau));
    }
    if (next === windAngle) return;
    const fu = lod.state.footU, fv = lod.state.footV;
    if (windAngle === windAngle) {
      // re-pivot at the camera foot: uv(foot) continuous across the turn
      const u0 = laceM00 * fu + laceM01 * fv, v0 = laceM10 * fu + laceM11 * fv;
      laceMatrix(next);
      laceCU = fract64(laceCU + u0 - (laceM00 * fu + laceM01 * fv));
      laceCV = fract64(laceCV + v0 - (laceM10 * fu + laceM11 * fv));
    } else laceMatrix(next);
    windAngle = next;
    (uAlong.value as Vector2).set(Math.cos(next), Math.sin(next));
  }

  // ---- c2 detail-grid gain: the reference's c2 normals are central differences on a 256 grid (cell L/256).
  // The low tier's 128 grid keeps only |k| < 128 dk and differences over L/128; ultra's 512 grid (same band,
  // zero-padded) differences over L/512. The foam source scales c2's slope / compression by
  // g = sqrt(S_256 / S_n), S_n = sum_k E(k) [(sin(kx h_n)/h_n)^2 + (sin(ky h_n)/h_n)^2] over the band kept
  // on that grid (E = |h0|^2 + |h0m|^2 of the master spectrum). Computed on the CPU when a slot regenerates.
  const gainVer = new Float64Array(8).fill(-1), gainN = new Float64Array(8).fill(-1), gainVal = new Float64Array(8).fill(1);
  function detailGainFor(h0: Float32Array, n: number): number {
    if (n === BASE_SIZE) return 1;
    const M = BASE_SIZE, half = M / 2, L = C[2].domain, dk = (2 * Math.PI) / L;
    const hR = L / M, hN = L / n, nh = n / 2;
    let sR = 0, sN = 0;
    for (let y = 0; y < M; y++) {
      const ky = y - half, ay = ky * dk;
      const dyR = Math.sin(ay * hR) / hR, dyN = Math.sin(ay * hN) / hN;
      for (let x = 0; x < M; x++) {
        const kx = x - half, ax = kx * dk;
        const i = 4 * (y * M + x);
        const e = h0[i] * h0[i] + h0[i + 1] * h0[i + 1] + h0[i + 2] * h0[i + 2] + h0[i + 3] * h0[i + 3];
        if (e === 0) continue;
        const dxR = Math.sin(ax * hR) / hR;
        sR += e * (dxR * dxR + dyR * dyR);
        if (n < M && !(kx >= -nh + 1 && kx < nh && ky >= -nh + 1 && ky < nh)) continue;
        const dxN = Math.sin(ax * hN) / hN;
        sN += e * (dxN * dxN + dyN * dyN);
      }
    }
    // Exponent 1 (variance ratio), not 1/2: measured, the history mean responds to c2 roughly like ln(g) x 2
    // (the compression smoothstep and the sqrt slope term are nonlinear); see the header table.
    return sN > 0 ? sR / sN : 1;
  }
  function updateDetailGain(t: number) {
    const oc = opts.ocean;
    if (!oc) return;
    const st = oc.getState();
    const n = fft.uniforms.detailSize.value as number;
    const a = st.fromSlot as number, b = st.toSlot as number;
    for (let q = 0; q < 2; q++) {
      const s = q === 0 ? a : b;
      const sl = st.slots[s];
      if (!sl || !(sl.version > 0)) continue;
      if (gainVer[s] !== sl.version || gainN[s] !== n) { gainVal[s] = detailGainFor(sl.h0[2], n); gainVer[s] = sl.version; gainN[s] = n; }
    }
    const w = a === b ? 1 : oc.blendAt(t);
    uDetailGain.value = gainVal[a] + (gainVal[b] - gainVal[a]) * w;
  }

  function activate(ts: FoamTierState) {
    const old = active;
    if (retired) { disposeTier(retired); retired = null; }
    active = ts;
    pending = null;
    pendingTier = null;
    historyNode.value = ts.out;
    if (old) {
      retired = old; retireFrames = 3;
      prevTierNode.value = old.out;
      resampleFrom = old;
    }
  }
  let resampleFrom: FoamTierState | null = null;

  const pass: Pass = {
    desc: { id: 'ocean.foam', module: 'world/ocean', stage: 'compute', reads: ['ocean.normal'], writes: ['ocean.foam'] },
    execute(ctx: PassContext) {
      opts.perf?.begin('world/ocean', 'ocean.foam');
      if (retired && --retireFrames <= 0) { disposeTier(retired); retired = null; prevTierNode.value = prevTierPlaceholder; }
      if (pending) activate(pending);
      const ts = active;
      // Nothing to accumulate before the FFT has produced its first normals (empty textures would read as
      // full compression everywhere and seed the history with a white sheet).
      if (!ts || !camera || !lod.state.hasPrev || !fftReady) { opts.perf?.end(); return; }

      // ---- time
      const t = ctx.time;
      const dt = initialized ? t - lastTime : 0;
      lastTime = t;
      let reset = !initialized || dt < 0 || dt > 2;

      // ---- anchor: look ray hit on sea level, clamped lead, snapped to the cell (float64 lattice coordinates)
      const e = camera.matrixWorld.elements;
      camR.x = e[12]; camR.y = e[13]; camR.z = e[14];
      let fx = -e[8], fy = -e[9], fz = -e[10];
      const fl = Math.hypot(fx, fy, fz) || 1; fx /= fl; fy /= fl; fz /= fl;
      origin.toWorld(camR, camW);
      const h = shape.altitude(camW);
      shape.atAltitude(camW, 0, footW);
      shape.tangentFrame(footW, frame);
      const E = frame.east, N = frame.north, U = frame.up;
      const fu = footW.x * E.x + footW.y * E.y + footW.z * E.z;
      const fv = -(footW.x * N.x + footW.y * N.y + footW.z * N.z);
      const dU = fx * U.x + fy * U.y + fz * U.z;
      const dE = fx * E.x + fy * E.y + fz * E.z;
      const dS = -(fx * N.x + fy * N.y + fz * N.z);
      const dist = dU < -0.01 ? -h / dU : 0;
      const lead = P.anchorLead;
      const tu = fu + Math.max(-lead, Math.min(lead, dE * dist));
      const tv = fv + Math.max(-lead, Math.min(lead, dS * dist));
      const cell = ts.cell;
      const iu = Math.round(tu / cell), iv = Math.round(tv / cell);
      const du = iu - anchorIU, dv = iv - anchorIV;
      if (!resampleFrom && (Math.abs(du) * cell > P.resetJump || Math.abs(dv) * cell > P.resetJump)) reset = true;
      const newU = iu * cell, newV = iv * cell;
      if (resampleFrom) {
        uResample.value = reset ? 0 : 1;
        (uResampleOffset.value as Vector2).set(newU - anchorU, newV - anchorV);
      } else uResample.value = 0;
      anchorIU = iu; anchorIV = iv; anchorU = newU; anchorV = newV;
      (uShift.value as Vector2).set(du, dv);
      uRetention.value = reset ? 0 : Math.exp(-Math.max(dt, 0) / P.tau);
      for (let c = 0; c < 4; c++) {
        const Pc = C[c].domain;
        (uAnchorMod[c].value as Vector2).set(mod64(anchorU, Pc), mod64(anchorV, Pc));
      }

      // ---- material uniforms (relative to the LOD's foot of THIS frame)
      (anchorRel.value as Vector2).set(anchorU - lod.state.footU, anchorV - lod.state.footV);
      updateWind(dt, reset);
      updatePhases();
      updateDetailGain(t);

      // ---- dispatch (skipped when nothing changes: paused and the anchor did not move)
      const idle = !reset && !resampleFrom && dt === 0 && du === 0 && dv === 0;
      if (!idle) {
        const R2 = ts.res * ts.res;
        uPrevBase.value = parity * R2;
        parity ^= 1;
        uCurBase.value = parity * R2;
        ctx.renderer.compute(ts.kernel);
        steps++;
      }
      resampleFrom = null;
      initialized = true;
      opts.perf?.end();
    },
  };

  const foam: OceanFoam = {
    pass,
    nodes,
    ready,
    get tier() { return active ? active.tier : requested; },
    get pendingTier() { return pendingTier; },
    uniforms: { anchorRel, fineInSource, detailGain: uDetailGain },
    update(cam: any) { camera = cam; },
    setTier(t: FoamTier) {
      if (!(t in FOAM_TIERS)) throw new Error(`ocean.foam: unknown tier ${t}`);
      requested = t;
      if (pendingTier === t) return;
      if (pending) { disposeTier(pending); pending = null; }
      if (active && FOAM_TIERS[active.tier] === FOAM_TIERS[t]) { pendingTier = null; return; } // same resolution
      void startBuild(t, null);
    },
    debug: {
      async readHistory() {
        if (!active) throw new Error('ocean.foam: no active tier');
        const R2 = active.res * active.res;
        const cur = parity * R2;
        const buf = new Float32Array(await renderer.getArrayBufferAsync(active.histAttr));
        return buf.slice(cur, cur + R2);
      },
      resolution() { return active ? active.res : FOAM_TIERS[requested]; },
      anchor(out) { out.x = anchorU; out.y = anchorV; return out; },
      memoryBytes() { return (active ? active.bytes : 0) + (patternTex ? 1024 * 1024 * 4 * 4 / 3 : 0); },
      lastCompileMs() { return compileMs; },
      patternLoaded() { return patternLoaded; },
      kernelId() { return active ? active.kernel.id : -1; },
      get steps() { return steps; },
    },
    dispose() {
      if (active) disposeTier(active);
      if (pending) disposeTier(pending);
      if (retired) disposeTier(retired);
      active = pending = retired = null;
      pendingTier = null;
      prevTierPlaceholder.dispose();
      patternPlaceholder.dispose();
      patternTex?.dispose();
    },
  };
  return foam;
}

// ======================================================================================================
// DEMO-ONLY debug view: replaces the ocean mesh material with a simple water shading + the reference foam
// composition, so the foam can be judged against the WebGL reference before the real material lands.
//   mode 'foam'    : simple water (deep body colour + env reflection + haze) + grey foam (.3 + .7 N.L)
//   mode 'foamraw' : R = edge-faded history / 2, G = final coverage, B = 0.15 inside the 400 m window
// ======================================================================================================
export interface FoamDebugOptions {
  foam: OceanFoam;
  /** The createOceanGpu() result (uses .mesh, .material.positionNode, .lod, .fft). */
  gpu: any;
  envMap?: any;
  envIntensity?: number;
  preExposure?: any;
  /** Direction towards the sun (flat frame), read every frame. */
  sunDir: () => Vec3;
  mode: 'foam' | 'foamraw';
}

export function attachFoamDebugView(o: FoamDebugOptions) {
  const { foam, gpu } = o;
  const lod = gpu.lod as OceanLod;
  const n = lod.nodes;
  const sampling = createFftSampling(gpu.fft);
  const envIntensity = uniform(o.envIntensity ?? 1);
  const preExposure = o.preExposure ?? uniform(1);
  const uSun = uniform(new Vector3(0, 1, 0));
  const upW = normalize(modelWorldMatrix.mul(vec4(0, 1, 0, 0)).xyz);
  const toFrag = positionWorld.sub(cameraPosition);
  const dist = toFrag.length();
  const V = toFrag.div(max(dist, 1e-4));
  const envAt = (dir: any, rough: any) => (o.envMap ? pmremTexture(o.envMap, dir, rough).rgb : vec3(0.3, 0.45, 0.6));

  const material = new MeshBasicNodeMaterial({ side: DoubleSide });
  material.name = `ocean.debug.${o.mode}`;
  const rest = varying(n.localPos.xz);
  const f = foam.nodes.layers(rest);
  if (o.mode === 'foamraw') {
    const inWin = abs(rest.sub(foam.uniforms.anchorRel)).lessThan(200);
    const win = select(inWin.x.and(inWin.y), float(0.15), float(0));
    material.colorNode = vec3(f.history.mul(0.5), f.foam, win).mul(envIntensity).mul(preExposure);
  } else {
    const footprint = max(max(abs(dFdx(rest.x)), abs(dFdx(rest.y))), max(abs(dFdy(rest.x)), abs(dFdy(rest.y))));
    const tileV = new Map<number, any>(FFT_CASCADES.map((c) => [c.domain, varying(n.tileCoord(c.domain))] as [number, any]));
    const surf = sampling.sampleSurface((Pp: number) => tileV.get(Pp), { footprint });
    const Ns = normalize(modelWorldMatrix.mul(vec4(surf.normal, 0)).xyz);
    const N = normalize(Ns.add(V.negate().mul(max(float(0.025).sub(dot(Ns, V.negate())), 0))));
    const NoV = clamp(dot(N, V.negate()), 0.015, 1);
    const fresnel = float(0.018).add(float(0.982).mul(pow(float(1).sub(NoV), 5)));
    const rough = clamp(float(0.018).add(surf.variance.mul(0.8)), 0.018, 0.8);
    const deep = vec3(0.0159963, 0.0612461, 0.0998987);
    const sky = envAt(reflect(V, N), clamp(rough.add(0.22), 0, 1));
    const diffuse = envAt(upW, float(1));
    let color: any = mix(deep.mul(diffuse).mul(0.8), sky, fresnel);
    const foamColor = vec3(float(0.3).add(clamp(dot(Ns, uSun), 0, 1).mul(0.7)));
    color = mix(color, foamColor, f.foam);
    const horiz = toFrag.sub(upW.mul(dot(toFrag, upW)));
    const hazeDir = normalize(horiz.add(upW.mul(dist.mul(0.002))));
    const haze = float(1).sub(exp(dist.div(280).pow(2).negate()));
    color = mix(color, envAt(hazeDir, float(0)), haze);
    material.colorNode = color.mul(envIntensity).mul(preExposure);
  }
  const saved = gpu.mesh.material;
  material.positionNode = saved.positionNode;
  gpu.mesh.material = material;
  const sun = o.sunDir();
  (uSun.value as Vector3).set(sun.x, sun.y, sun.z);
  return {
    material,
    /** Call per frame (cheap) to follow sky changes. */
    update() { const s = o.sunDir(); (uSun.value as Vector3).set(s.x, s.y, s.z); },
    dispose() { gpu.mesh.material = saved; material.dispose(); },
  };
}
