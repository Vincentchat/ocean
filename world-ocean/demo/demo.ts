// world/ocean standalone demo (WebGPU only).
//
// URL flags
//   ?cam=x,y,z,yawDeg,pitchDeg   camera world position (m, flat +Y up) and orientation. yaw 0 looks towards -Z,
//                                positive yaw turns towards +X; pitch positive looks up. Same convention as the
//                                WebGL reference's ?cam= hook, so canonical shots line up.
//   ?shot                        deterministic capture: sim time from ?t= (default 0), warm-up of ?warm= seconds
//                                (default 6, fixed steps) before t so history effects (foam) exist, then pause,
//                                hide UI and set window.__shotReady = true once frames are settled.
//   ?t=seconds                   simulation time (also without ?shot: start time).
//   ?paused                      start paused.
//   ?tier=low|medium|high|ultra  ocean quality tier (default high).
//   ?preset=calm|breeze|windy|storm  ocean preset (core; default windy = the reference's load-time sea).
//   ?sky=golden|azure|storm      photo-sky stand-in preset (reference scenes; default storm = reference default).
//   ?scene=golden|azure|storm    match a WebGL reference scene: sky=<scene> + ocean preset 'windy' with
//                                amplitude REFERENCE_SCENE_AMPLITUDE[scene] (core). Used by tools/shots.json.
//   ?origin=x,y,z                initial floating origin (m). Only the origin moves: the camera keeps its WORLD
//                                position (?cam), so the view must be identical at (100000,0,100000) (acceptance
//                                check 7). Disables the 2 km auto-rebase unless ?rebase=1.  ?rebase=0|1 forces it.
//   ?wind=speed,dirDeg           provide a stub WIND_QUERY (m/s, direction blown towards, CCW from east).
//   ?debug=name                  debug view name handed to createOceanGpu (e.g. velocity).
//   ?exposure=x                  extra tonemap exposure (default 1).  ?dpr=x pixel ratio cap (default 1.75).
//   ?parity                      reserved for the ocean parity check (handed through as window.__demo.flags).
import {
  HalfFloatType, LinearSRGBColorSpace, NoToneMapping, NodeMaterial, PerspectiveCamera, QuadMesh, RenderTarget,
  Scene, Vector3, WebGPURenderer,
} from 'three/webgpu';
import { Fn, clamp, max, pow, texture, uniform, uv, vec3, vec4 } from 'three/tsl';
import { setAssetResolver, OCEAN_ASSETS, DEMO_ASSETS } from '../assets.ts';
import { createOceanGpu, type OceanTier, type Pass, type PassContext, type OceanGpu } from '../three/index.ts';
import {
  createFloatingOrigin, createStubWorld, createStubWind, createStubEnvironment,
} from '../stubs/impl.ts';
import { ENVIRONMENT, SKY_LIGHT, WIND_QUERY, type FrameInfo } from '../stubs/world-core.ts';
import { createPassRunner } from './pass-runner.ts';
import { createFreeCamera } from './free-camera.ts';
import { createPhotoSky, REF_NITS, SKY_PRESETS, type SkyPresetName } from './photo-sky.ts';

const $ = (id: string) => document.getElementById(id)!;
/** Reference Wave slider per scene (src/ocean.js presets). */
const REFERENCE_WAVE: Record<string, number> = { golden: 0.9, azure: 1.0, storm: 1.5 };
const params = new URLSearchParams(location.search);
const flags = {
  shot: params.has('shot'),
  parity: params.has('parity'),
  t: Number(params.get('t') ?? 0) || 0,
  warm: params.has('warm') ? Math.max(0, Number(params.get('warm')) || 0) : 6,
  paused: params.has('paused'),
  tier: (params.get('tier') ?? 'high') as OceanTier,
  preset: params.get('preset') ?? 'windy',
  scene: params.get('scene') as SkyPresetName | null,
  sky: (params.get('sky') ?? params.get('scene') ?? 'storm') as SkyPresetName,
  debug: params.get('debug') ?? '',
  exposure: Number(params.get('exposure') ?? 1) || 1,
  dpr: Number(params.get('dpr') ?? 1.75) || 1.75,
  cam: parseNumbers(params.get('cam'), 5),
  origin: parseNumbers(params.get('origin'), 3),
  wind: parseNumbers(params.get('wind'), 2),
};
function parseNumbers(s: string | null, n: number): number[] | null {
  if (!s) return null;
  const v = s.split(',').map(Number);
  if (v.length < n || v.slice(0, n).some((x) => !Number.isFinite(x))) {
    console.error(`[demo] bad URL flag value "${s}" (expected ${n} numbers)`);
    return null;
  }
  return v.slice(0, n);
}

function fatal(title: string, body: string) {
  $('fatal-title').textContent = title;
  $('fatal-body').textContent = body;
  $('fatal').hidden = false;
  $('ui').style.display = 'none';
  (window as any).__demoFatal = `${title}: ${body}`;
}

// Asset ids -> URLs (Vite serves the reference repo's public/ at the site root).
setAssetResolver((id) => ({
  [OCEAN_ASSETS.foamPattern]: '/foam-pattern.png',
  [DEMO_ASSETS.skyBase]: '/sky-base.jpg',
  [DEMO_ASSETS.skyGain]: '/sky-gain.jpg',
  [DEMO_ASSETS.skyRadiance]: '/sky-radiance.bin',
} as Record<string, string>)[id] ?? null);

// Allocation-free perf sink (maps to the engine's perf.begin(module, phase) / perf.end()).
function createPerfSink() {
  const MAX = 32;
  const keys: string[] = new Array(MAX).fill('');
  const starts = new Float64Array(MAX);
  let depth = 0;
  const totals = new Map<string, number>(); // ms accumulated this frame
  return {
    totals,
    begin(module: string, phase: string) {
      if (depth >= MAX) return;
      keys[depth] = phase; // pass ids / phases are already namespaced ('ocean.fft', 'demo.scene', ...)
      void module;
      starts[depth++] = performance.now();
    },
    end() {
      if (depth <= 0) return;
      depth--;
      const k = keys[depth];
      totals.set(k, (totals.get(k) ?? 0) + performance.now() - starts[depth]);
    },
    resetFrame() { for (const k of totals.keys()) totals.set(k, 0); },
  };
}

async function main() {
  if (!('gpu' in navigator) || !navigator.gpu) {
    fatal('WebGPU required', 'This demo needs a browser with WebGPU (navigator.gpu). There is no WebGL fallback.');
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    fatal('WebGPU required', 'navigator.gpu.requestAdapter() returned no adapter. Enable hardware acceleration / WebGPU.');
    return;
  }
  const adapterInfo = (adapter as any).info ?? {};

  // ---------------------------------------------------------------------------------- renderer
  const renderer = new WebGPURenderer({ antialias: false, trackTimestamp: true });
  await renderer.init();
  if (!(renderer.backend as any).isWebGPUBackend) {
    fatal('WebGPU required', 'three.js fell back to its WebGL backend; this demo is WebGPU only.');
    return;
  }
  const device: GPUDevice = (renderer.backend as any).device;
  device.addEventListener('uncapturederror', (ev: any) => console.error(`[WebGPU] uncaptured error: ${ev.error?.message ?? ev.error}`));
  device.lost.then((info) => console.error(`[WebGPU] device lost (${info.reason}): ${info.message}`));
  renderer.outputColorSpace = LinearSRGBColorSpace; // the tonemap pass writes display-ready values
  renderer.toneMapping = NoToneMapping;
  renderer.setPixelRatio(Math.min(devicePixelRatio, flags.dpr));
  renderer.setSize(innerWidth, innerHeight);
  $('scene').appendChild(renderer.domElement);
  const hasTimestamps = renderer.hasFeature('timestamp-query');

  // ---------------------------------------------------------------------------------- world
  const originStart = flags.origin ?? [0, 0, 0];
  const origin = createFloatingOrigin({ x: originStart[0], y: originStart[1], z: originStart[2] });
  const world = createStubWorld({ origin, fixedDt: 1 / 60 });
  const preExposure = uniform(1 / REF_NITS);
  const exposure = uniform(flags.exposure);

  const scene = new Scene();
  const camera = new PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 20000);
  const sky = await createPhotoSky(renderer, { preExposure, preset: SKY_PRESETS[flags.sky] ? flags.sky : 'storm' });
  scene.backgroundNode = sky.backgroundNode;

  const env = createStubEnvironment();
  // ENVIRONMENT follows the photo-sky stand-in: same sun direction; cloud cover = the reference storminess.
  const syncEnv = () => {
    const s = sky.skyLight.sunDir;
    env.sunDir.x = s.x; env.sunDir.y = s.y; env.sunDir.z = s.z;
    env.cloudCover = sky.skyLight.reference.storminess;
  };
  syncEnv();
  world.provide(SKY_LIGHT, sky.skyLight);
  world.provide(ENVIRONMENT, env);
  if (flags.wind) world.provide(WIND_QUERY, createStubWind(flags.wind[0], flags.wind[1], world.shape));

  // Core ocean (written by the core builder). Loaded dynamically so the shell runs before it exists.
  let core: any = null, ocean: any = null;
  try {
    // Ask the dev server first so a missing core does not show up as a 404 console error.
    const probe = await fetch('/__exists?path=index.ts').then((r) => (r.ok ? r.json() : { exists: true }), () => ({ exists: true }));
    if (!probe.exists) throw new Error('world-ocean/index.ts does not exist yet');
    core = await import(/* @vite-ignore */ new URL('../index.ts', import.meta.url).href);
    if (typeof core.createOcean === 'function') {
      const amp = flags.scene ? core.REFERENCE_SCENE_AMPLITUDE?.[flags.scene] : undefined;
      ocean = core.createOcean(amp !== undefined ? { preset: 'windy', seed: 1, amplitude: amp } : { preset: flags.preset, seed: 1 });
      world.use(ocean);
    } else console.info('[demo] ../index.ts has no createOcean yet; running without the core');
  } catch (e) {
    console.info(`[demo] ocean core not available yet (${(e as Error).message}); placeholder only`);
  }
  world.start();

  // ---------------------------------------------------------------------------------- targets + passes
  const perf = createPerfSink();
  const runner = createPassRunner(perf);
  const size = { w: 1, h: 1 };
  const sceneColor = new RenderTarget(1, 1, { type: HalfFloatType, samples: 4, depthBuffer: true });
  sceneColor.texture.name = 'scene.color';
  const ctx: PassContext = {
    renderer, dt: 0, frame: 0, width: 1, height: 1, tier: flags.tier, time: 0,
    targets: { 'scene.color': sceneColor },
  };
  function resize() {
    renderer.setPixelRatio(Math.min(devicePixelRatio, flags.dpr));
    renderer.setSize(innerWidth, innerHeight);
    const pr = renderer.getPixelRatio();
    size.w = Math.max(1, Math.floor(innerWidth * pr));
    size.h = Math.max(1, Math.floor(innerHeight * pr));
    sceneColor.setSize(size.w, size.h);
    ctx.width = size.w; ctx.height = size.h;
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
  }
  resize();
  addEventListener('resize', resize);

  runner.add(sky.envPass as Pass);
  // Run the env capture once now (it is a pass) so the ocean gets a ready env map.
  runner.execute(ctx);

  let gpu: OceanGpu = createOceanGpu({
    ocean, renderer, scene, camera, origin, shape: world.shape, tier: flags.tier, skyLight: sky.skyLight, envMap: sky.envMap,
    environment: env, envIntensity: sky.envIntensity, preExposure, perf, debug: flags.debug,
    // Reference scenes: the crest glow is normalised by the reference Wave slider (golden .9, azure 1, storm 1.5).
    waveScale: flags.scene ? REFERENCE_WAVE[flags.scene] : undefined,
  } as any);
  for (const p of gpu.passes) runner.add(p);

  // [foam builder] opt-in foam debug view (?debug=foam | ?debug=foamraw): three/foam.ts history pass + a debug
  // material with the reference foam composition. Only active with the debug flag; remove once the material
  // integrates foam.nodes. window.__foam = { foam, view } for tools/foam-check.mjs.
  const foamMode = flags.debug.split(',').includes('foamraw') ? 'foamraw' : flags.debug.split(',').includes('foam') ? 'foam' : null;
  if (foamMode && ocean) {
    const { createOceanFoam, attachFoamDebugView } = await import('../three/foam.ts');
    const foam = createOceanFoam({ renderer, fft: gpu.fft, lod: gpu.lod, origin, shape: world.shape, tier: flags.tier, camera, ocean });
    runner.add(foam.pass);
    if (flags.debug.split(',').includes('foamfine')) foam.uniforms.fineInSource.value = 1; // c3 in the source (A/B)
    const view = attachFoamDebugView({
      foam, gpu, envMap: sky.envMap, envIntensity: sky.envIntensity, preExposure, mode: foamMode,
      sunDir: () => (sky.skyLight as any).sunDir,
    });
    $('tier').addEventListener('change', () => foam.setTier(($('tier') as HTMLSelectElement).value as OceanTier));
    (window as any).__foam = { foam, view };
    await foam.ready;
  }

  const scenePass: Pass = {
    desc: { id: 'demo.scene', module: 'demo', stage: 'scene', reads: ['ocean.displacement', 'demo.env'], writes: ['scene.color', 'scene.depth'] },
    execute(c) {
      c.renderer.setRenderTarget(sceneColor);
      c.renderer.render(scene, camera);
    },
  };
  // HDR -> display: exposure, then the reference's tonemap (ACES fit + gamma 2.2, src/ocean.js `tonemap`).
  const tonemapMaterial = new NodeMaterial();
  tonemapMaterial.name = 'demo.tonemap';
  tonemapMaterial.fragmentNode = Fn(() => {
    const x = max(texture(sceneColor.texture, uv()).rgb.mul(exposure), vec3(0));
    const mapped = clamp(x.mul(x.mul(2.51).add(0.03)).div(x.mul(x.mul(2.43).add(0.59)).add(0.14)), 0, 1);
    return vec4(pow(mapped, vec3(1 / 2.2)), 1);
  })();
  const quad = new QuadMesh(tonemapMaterial);
  const tonemapPass: Pass = {
    desc: { id: 'demo.tonemap', module: 'demo', stage: 'tonemap', reads: ['scene.color'], writes: ['output'] },
    execute(c) {
      c.renderer.setRenderTarget(null);
      quad.render(c.renderer);
    },
  };
  runner.add(scenePass);
  runner.add(tonemapPass);

  // ---------------------------------------------------------------------------------- camera
  const freeCam = createFreeCamera();
  if (flags.cam) {
    freeCam.x = flags.cam[0]; freeCam.y = flags.cam[1]; freeCam.z = flags.cam[2];
    freeCam.yaw = flags.cam[3] * Math.PI / 180; freeCam.pitch = flags.cam[4] * Math.PI / 180;
  }
  // ?origin moves ONLY the floating origin: the camera keeps its WORLD position (?cam), so the view must be identical
  // (acceptance check 7). The automatic rebase is off while an explicit ?origin is given (?rebase=1 forces it on,
  // ?rebase=0 forces it off), otherwise it would move the origin back next to the camera on the first frame.
  freeCam.attach(renderer.domElement);
  freeCam.enabled = !flags.shot;
  const fwd = { x: 0, y: 0, z: 0 };
  const lookTarget = new Vector3();
  const REBASE_DISTANCE = 2000; // m: shift the floating origin when the camera gets this far from it
  const autoRebase = params.has('rebase') ? params.get('rebase') !== '0' : !flags.origin;
  function updateCamera() {
    let rx = freeCam.x - origin.origin.x, rz = freeCam.z - origin.origin.z;
    if (autoRebase && (Math.abs(rx) > REBASE_DISTANCE || Math.abs(rz) > REBASE_DISTANCE)) {
      origin.setOrigin(Math.round(freeCam.x), origin.origin.y, Math.round(freeCam.z));
      rx = freeCam.x - origin.origin.x; rz = freeCam.z - origin.origin.z;
    }
    camera.position.set(rx, freeCam.y - origin.origin.y, rz);
    freeCam.forward(fwd);
    lookTarget.set(camera.position.x + fwd.x, camera.position.y + fwd.y, camera.position.z + fwd.z);
    camera.lookAt(lookTarget);
    camera.updateMatrixWorld();
  }

  // ---------------------------------------------------------------------------------- UI
  let paused = flags.paused;
  const setPaused = (v: boolean) => { paused = v; $('pause').textContent = v ? 'play' : 'pause'; };
  setPaused(paused);
  $('pause').onclick = () => setPaused(!paused);
  const markActive = (attr: string, value: string) =>
    document.querySelectorAll<HTMLElement>(`[data-${attr}]`).forEach((b) => b.classList.toggle('active', b.dataset[attr] === value));
  document.querySelectorAll<HTMLElement>('[data-preset]').forEach((b) => (b.onclick = () => {
    const name = b.dataset.preset!;
    if (ocean?.setPreset) { ocean.setPreset(name); markActive('preset', name); }
    else console.info(`[demo] preset ${name}: ocean core not available yet`);
  }));
  document.querySelectorAll<HTMLElement>('[data-sky]').forEach((b) => (b.onclick = () => {
    const name = b.dataset.sky as SkyPresetName;
    sky.setPreset(name);
    syncEnv();
    markActive('sky', name);
  }));
  markActive('preset', flags.preset);
  markActive('sky', flags.sky);
  const tierSelect = $('tier') as HTMLSelectElement;
  tierSelect.value = flags.tier;
  tierSelect.onchange = () => { ctx.tier = tierSelect.value as OceanTier; gpu.setTier(ctx.tier); };
  addEventListener('keydown', (e) => {
    if ((e.target as HTMLElement)?.matches?.('input,select,textarea,button')) return;
    if (e.code === 'Space') { e.preventDefault(); if (!flags.shot) setPaused(!paused); }
    else if (e.code === 'KeyH') document.body.classList.toggle('hide-ui');
  });
  if (flags.shot) document.body.classList.add('shot');

  // ---------------------------------------------------------------------------------- perf readout
  const RING = 1024;
  const perfState = {
    adapter: { vendor: adapterInfo.vendor, architecture: adapterInfo.architecture, description: adapterInfo.description },
    timestamps: hasTimestamps,
    frames: 0,
    count: 0,
    head: 0,
    cpuMs: new Float64Array(RING),     // JS time spent in the frame callback (sim + passes + submit)
    frameMs: new Float64Array(RING),   // rAF interval
    gpuMs: new Float64Array(RING),     // GPU render timestamp total (last resolved frame)
    gpuComputeMs: new Float64Array(RING),
    reset() { perfState.count = 0; perfState.head = 0; },
    snapshot() {
      const n = Math.min(perfState.count, RING), out: Record<string, number[]> = { cpuMs: [], frameMs: [], gpuMs: [], gpuComputeMs: [] };
      for (let i = 0; i < n; i++) {
        const j = (perfState.head - n + i + RING) % RING;
        out.cpuMs.push(perfState.cpuMs[j]); out.frameMs.push(perfState.frameMs[j]);
        out.gpuMs.push(perfState.gpuMs[j]); out.gpuComputeMs.push(perfState.gpuComputeMs[j]);
      }
      return { ...out, count: n, timestamps: hasTimestamps, adapter: perfState.adapter, tier: ctx.tier, width: ctx.width, height: ctx.height };
    },
    passes() {
      const o: Record<string, number> = {};
      runner.passes.forEach((p, i) => (o[p.desc.id] = runner.cpuMs[i]));
      return o;
    },
  };
  (window as any).__perf = perfState;
  let gpuRender = 0, gpuCompute = 0, pendingR = false, pendingC = false;
  const onRender = (ms: number) => { gpuRender = ms || 0; pendingR = false; };
  const onCompute = (ms: number) => { gpuCompute = ms || 0; pendingC = false; };
  const onFail = () => { pendingR = false; pendingC = false; };
  let statsTimer = 0, statsFrames = 0, statsCpu = 0, statsInterval = 0;
  const stats = $('stats');

  (window as any).__demo = { renderer, scene, camera, world, origin, ocean, core, get gpu() { return gpu; }, sky, runner, freeCam, flags, perf, adapterInfo };

  // ---------------------------------------------------------------------------------- shot state
  const fixedDt = world.fixedDt;
  let shotPhase: 'off' | 'warmup' | 'settle' | 'done' = 'off';
  let warmSteps = 0, settleFrames = 0;
  let assetsReady = false;
  Promise.all([sky.ready, (gpu as any).ready]).then(() => { assetsReady = true; }, (e) => console.error('[demo] asset init failed', e));
  if (flags.shot) {
    const start = Math.max(0, flags.t - flags.warm);
    world.setTime(start);
    warmSteps = Math.round((flags.t - start) / fixedDt);
    shotPhase = warmSteps > 0 ? 'warmup' : 'settle';
    if (shotPhase === 'settle') { world.setTime(flags.t); paused = true; }
    (window as any).__shotReady = false;
  } else {
    world.setTime(flags.t);
  }

  // ---------------------------------------------------------------------------------- loop
  let last = performance.now();
  function frame(now: number) {
    requestAnimationFrame(frame);
    const t0 = performance.now();
    const interval = now - last;
    last = now;
    let realDt = Math.min(Math.max(interval / 1000, 0), 0.1);
    perf.resetFrame();

    let info: FrameInfo;
    if (shotPhase === 'warmup') {
      info = world.advance(fixedDt);
      if (--warmSteps <= 0) { world.setTime(flags.t); paused = true; shotPhase = 'settle'; }
      realDt = fixedDt;
    } else if (shotPhase === 'settle' || shotPhase === 'done') {
      info = world.frameOnly(fixedDt);
      realDt = fixedDt;
    } else {
      freeCam.update(realDt);
      info = paused ? world.frameOnly(realDt) : world.advance(realDt);
    }
    env.time = info.time;
    updateCamera();
    ctx.dt = realDt;
    ctx.frame++;
    ctx.time = info.interpolatedTime;
    runner.execute(ctx);

    if (hasTimestamps) {
      if (!pendingR) { pendingR = true; renderer.resolveTimestampsAsync('render').then(onRender, onFail); }
      if (!pendingC) { pendingC = true; renderer.resolveTimestampsAsync('compute').then(onCompute, onFail); }
    }
    const cpu = performance.now() - t0;
    const h = perfState.head;
    perfState.cpuMs[h] = cpu; perfState.frameMs[h] = interval; perfState.gpuMs[h] = gpuRender; perfState.gpuComputeMs[h] = gpuCompute;
    perfState.head = (h + 1) % RING; perfState.count++; perfState.frames++;

    if (shotPhase === 'settle' && assetsReady && ++settleFrames >= 8) {
      shotPhase = 'done';
      // Let the last submitted frame reach the canvas before signalling.
      device.queue.onSubmittedWorkDone().then(() => { (window as any).__shotReady = true; });
    }

    statsFrames++; statsCpu += cpu; statsInterval += interval;
    if (now - statsTimer > 500 && !flags.shot) {
      const fps = (statsFrames * 1000) / Math.max(statsInterval, 1e-3);
      stats.textContent =
        `${fps.toFixed(0)} fps  ${(statsInterval / statsFrames).toFixed(2)} ms/frame\n` +
        `cpu ${(statsCpu / statsFrames).toFixed(2)} ms  gpu ${hasTimestamps ? (gpuRender + gpuCompute).toFixed(2) + ' ms' : 'n/a'}` +
        (hasTimestamps ? ` (r ${gpuRender.toFixed(2)} c ${gpuCompute.toFixed(2)})` : '') + `\n` +
        `t ${info.interpolatedTime.toFixed(2)} s  tier ${ctx.tier}  ${ctx.width}x${ctx.height}\n` +
        `cam ${freeCam.x.toFixed(1)},${freeCam.y.toFixed(1)},${freeCam.z.toFixed(1)} yaw ${(freeCam.yaw * 180 / Math.PI).toFixed(1)} pitch ${(freeCam.pitch * 180 / Math.PI).toFixed(1)}\n` +
        `core ${ocean ? 'yes' : 'no (placeholder)'}  ${adapterInfo.description || adapterInfo.vendor || ''}`;
      statsTimer = now; statsFrames = 0; statsCpu = 0; statsInterval = 0;
    }
  }
  requestAnimationFrame(frame);

  // Replace the ocean adapter at runtime (used when the real adapter lands / for tier rebuilds).
  (window as any).__demo.replaceOcean = (next: OceanGpu) => {
    for (const p of gpu.passes) runner.remove(p);
    gpu.dispose();
    gpu = next;
    for (const p of gpu.passes) runner.add(p);
  };
}

main().catch((e) => {
  console.error('[demo] fatal', e);
  fatal('Demo failed to start', String((e as Error)?.message ?? e));
});
