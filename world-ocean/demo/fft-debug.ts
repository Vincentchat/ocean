// GPU FFT debug page (WebGPU only): shows every cascade's textures and runs a numeric GPU-vs-CPU check.
//
// URL flags
//   ?tier=low|medium|high|ultra   FFT tier (default high)
//   ?preset=calm|breeze|windy|storm  core preset (default breeze)
//   ?t=seconds                    fixed draw time (default: running clock from 0)
//   ?lod=n                        mip level shown in the panels
//   ?check                        run the GPU-vs-CPU check automatically; results in window.__fftCheck
//   ?bench                        FFT only (no panels): GPU timestamps in window.__perf (tools/perf.mjs format)
//
// Panels (rows = cascades 0..3, columns): height (up, red +, blue -), per-cascade normal (east/north tilt in
// red/green, blue = mip shortening x10), Jacobian J = (1+Exx)(1+Nyy)-Exy^2 (grey, red where J < 0) with the
// foam compression in green.
//
// GPU-vs-CPU check: the f32 FFT fields (before the rgba16f store) are read back asynchronously (the frame loop
// never waits; results arrive one or more frames later) and compared texel by texel with the core's CPU FFT
// (cpu-fft.ts CascadeSim at the same grid size, exact time: snapshotDt = t so the Hermite weight is exactly 1)
// for t = 12.5 s, 8191.7 s (loop wrap), 100000.37 s and in the middle of a preset crossfade. A second check
// samples the TEXTURES through the material helper (createFftSampling().sampleDisplacement, chained lookup,
// bilinear LOD 0) at 4096 random rest points and compares with the core's sampleRest (same grids, exact time).
import { NodeMaterial, QuadMesh, StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import { Fn, clamp, float, floor, fract, instanceIndex, int, length, select, storage, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';
import { createOcean } from '../index.ts';
import { createFlatShape } from '../stubs/impl.ts';
import { CascadeSim, CALL } from '../cpu-fft.ts';
import { omegaIndices } from '../spectrum.ts';
import { createOceanFft, createFftSampling, FFT_CASCADES, FFT_TIERS, type OceanTier } from '../three/fft.ts';
import type { PassContext } from '../three/pass-types.ts';

const $ = (id: string) => document.getElementById(id)!;
const params = new URLSearchParams(location.search);
const tierParam = (params.get('tier') ?? 'high') as OceanTier;
const preset = (params.get('preset') ?? 'breeze') as any;
const fixedT = params.has('t') ? Number(params.get('t')) : null;
const bench = params.has('bench');
const autoCheck = params.has('check');

async function main() {
  if (!navigator.gpu) { $('stats').textContent = 'WebGPU required (navigator.gpu missing)'; return; }
  const renderer = new WebGPURenderer({ antialias: false, trackTimestamp: true });
  await renderer.init();
  if (!(renderer.backend as any).isWebGPUBackend) { $('stats').textContent = 'WebGPU backend required'; return; }
  const device: GPUDevice = (renderer.backend as any).device;
  device.addEventListener('uncapturederror', (ev: any) => console.error(`[WebGPU] uncaptured error: ${ev.error?.message ?? ev.error}`));
  renderer.setPixelRatio(1);
  renderer.setSize(innerWidth, innerHeight);
  $('view').appendChild(renderer.domElement);
  addEventListener('resize', () => renderer.setSize(innerWidth, innerHeight));

  const shape = createFlatShape();
  const ocean = createOcean({ shape, preset, seed: 1 });
  const state = ocean.getState();
  const g = state.gravity;
  const fft = createOceanFft({ renderer, ocean, tier: tierParam });
  (window as any).__fft = { fft, ocean, renderer };
  const tierSel = $('tier') as HTMLSelectElement;
  tierSel.value = tierParam;
  tierSel.onchange = () => fft.setTier(tierSel.value as OceanTier);

  // ---------------------------------------------------------------- panels
  // Per-cascade colour scale: 2.5 sigma of the cascade's height (time average of |h0|^2 + |h0m|^2).
  const sigma = [0, 1, 2, 3].map((c) => {
    const h0 = c < 3 ? state.slots[state.toSlot].h0[c] : fft.debug.fineH0(state.toSlot);
    let s = 0;
    for (let i = 0; i < h0.length; i++) s += h0[i] * h0[i];
    return Math.sqrt(s);
  });
  const uLod = uniform(Number(params.get('lod') ?? 0));
  ($('lod') as HTMLSelectElement).value = String(uLod.value);
  ($('lod') as HTMLSelectElement).onchange = (e) => { uLod.value = Number((e.target as HTMLSelectElement).value); };
  const N = fft.nodes, U = fft.uniforms;
  const mat = new NodeMaterial();
  mat.fragmentNode = Fn(() => {
    const p = uv();
    const col = floor(p.x.mul(3)), row = floor(float(1).sub(p.y).mul(4));
    const q = vec2(fract(p.x.mul(3)), fract(float(1).sub(p.y).mul(4)));
    let out: any = vec3(0);
    for (let r = 0; r < 4; r++) {
      const isBase = r < 2;
      const layer = r & 1;
      const disp = (isBase ? N.displacementBase : N.displacementDetail).sample(q).depth(int(U.dispCurLayer.add(layer))).level(uLod);
      const nrm = (isBase ? N.normalBase : N.normalDetail).sample(q).depth(int(layer)).level(uLod);
      const der = (isBase ? N.derivativesBase : N.derivativesDetail).sample(q).depth(int(layer)).level(uLod);
      const v = disp.x.div(2.5 * sigma[r] + 1e-9);
      const cH = vec3(clamp(v, 0, 1), clamp(v.abs(), 0, 1).mul(0.15), clamp(v.negate(), 0, 1));
      const cN = vec3(nrm.x.mul(2).add(0.5), nrm.y.mul(2).add(0.5), float(1).sub(length(nrm.xyz)).mul(10));
      const J = der.z.add(1).mul(der.w.add(1)).sub(disp.w.mul(disp.w));
      const grey = clamp(J.mul(0.5), 0, 1);
      const cJ = select(J.lessThan(0), vec3(1, 0, 0), vec3(grey, grey.add(float(1).sub(nrm.w).mul(2)), grey));
      const c = select(col.equal(0), cH, select(col.equal(1), cN, cJ));
      out = select(row.equal(r), c, out);
    }
    // thin separators
    const edge = q.x.lessThan(0.004).or(q.y.lessThan(0.006));
    return vec4(select(edge, vec3(0.1), out), 1);
  })();
  const quad = new QuadMesh(mat);
  const labels = $('labels');
  const names = ['height (up)', 'normal tilt / mip shortening', 'Jacobian (red < 0), foam (green)'];
  for (let r = 0; r < 4; r++) for (let c = 0; c < 3; c++) {
    const d = document.createElement('div');
    d.style.left = `${(c / 3) * 100}%`; d.style.top = `${(r / 4) * 100}%`;
    d.textContent = `c${r} L=${FFT_CASCADES[r].domain.toFixed(r === 3 ? 4 : 0)} m: ${names[c]}`;
    labels.appendChild(d);
  }
  if (bench) { labels.style.display = 'none'; }

  // ---------------------------------------------------------------- chained-sampling check kernel
  const K = 4096;
  const ptsArr = new Float32Array(K * 2);
  let s = 12345;
  const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
  for (let i = 0; i < K; i++) { ptsArr[2 * i] = rnd() * 400 - 200; ptsArr[2 * i + 1] = rnd() * 400 - 200; }
  const ptsAttr = new StorageBufferAttribute(ptsArr, 2);
  const outAttr = new StorageBufferAttribute(K, 4);
  const sampling = createFftSampling(fft);
  const pts = storage(ptsAttr, 'vec2', K).toReadOnly();
  const outBuf = storage(outAttr, 'vec4', K);
  const chainKernel = Fn(() => {
    const pt = pts.element(instanceIndex);
    // tile coordinate (east, south) of a rest point (east e, north n) on the flat frame anchored at 0
    const d = sampling.sampleDisplacement(() => vec2(pt.x, pt.y.negate()), { threeCascades: true });
    outBuf.element(instanceIndex).assign(vec4(d, 0));
  })().compute(K);

  // ---------------------------------------------------------------- perf state (tools/perf.mjs format)
  const RING = 2048;
  const perfState = {
    frames: 0, count: 0, head: 0,
    cpuMs: new Float64Array(RING), frameMs: new Float64Array(RING), gpuMs: new Float64Array(RING), gpuComputeMs: new Float64Array(RING),
    reset() { perfState.count = 0; perfState.head = 0; },
    snapshot() {
      const n = Math.min(perfState.count, RING), o: Record<string, number[]> = { cpuMs: [], frameMs: [], gpuMs: [], gpuComputeMs: [] };
      for (let i = 0; i < n; i++) {
        const j = (perfState.head - n + i + RING) % RING;
        o.cpuMs.push(perfState.cpuMs[j]); o.frameMs.push(perfState.frameMs[j]); o.gpuMs.push(0); o.gpuComputeMs.push(perfState.gpuComputeMs[j]);
      }
      return { ...o, count: n, timestamps: renderer.hasFeature('timestamp-query'), tier: fft.tier, memoryMB: fft.debug.memoryBytes() / 1048576, compileMs: fft.debug.lastCompileMs() };
    },
  };
  (window as any).__perf = perfState;

  // ---------------------------------------------------------------- check state machine
  type Row = { cascade: number; n: number; maxUp: number; maxEast: number; maxNorth: number; rmsUp: number; maxAbsM: number };
  const results: any = { tier: null, hs: ocean.expectedHs(), tests: [] as any[], done: false, error: null };
  (window as any).__fftCheck = results;
  const TESTS: { name: string; t: number | 'xfade' }[] = [
    { name: 't=12.5', t: 12.5 }, { name: 't=8191.7 (loop wrap)', t: 8191.7 }, { name: 't=100000.37', t: 100000.37 },
    { name: 'crossfade breeze->storm, mid', t: 'xfade' },
  ];
  let checkIdx = -1, checkWait = 0, checkBusy = false, checkTime = 0, checkSetup = false;
  const startCheck = () => { if (checkIdx < 0 || results.done) { results.tests = []; results.done = false; results.tier = fft.tier; checkIdx = 0; checkWait = 4; checkBusy = false; checkSetup = false; } };
  $('check').onclick = startCheck;

  function cpuField(c: number, n: number, T: number, chop: number): Float32Array {
    const def = FFT_CASCADES[c];
    const masterOmega = c < 3 ? state.omegaIndex[c] : omegaIndices(def, g, new Float32Array(256 * 256));
    const sim = new CascadeSim(def, { size: n, snapshotDt: T, filter: 'bilinear' }, masterOmega, 3, 4);
    for (let sl = 0; sl < 3; sl++) sim.setSlot(sl, c < 3 ? state.slots[sl].h0[c] : fft.debug.fineH0(sl));
    CALL[0] = state.blendStart; CALL[1] = state.blendDuration;
    sim.setSource(0, 1, state.fromSlot, state.toSlot);
    CALL[0] = T; CALL[1] = chop; CALL[2] = 0;
    sim.materialise(0, 1, true, 1);
    return sim.frames[0];
  }
  function compare(gpu: Float32Array, layer: number, cpu: Float32Array, n: number, c: number, hs: number): Row {
    let mu = 0, me = 0, mn = 0, su = 0;
    const nn = n * n;
    for (let i = 0; i < nn; i++) {
      const o = (layer * nn + i) * 4, q = i * 3;
      const du = Math.abs(gpu[o] - cpu[q]), de = Math.abs(gpu[o + 1] - cpu[q + 1]), dn = Math.abs(gpu[o + 2] - cpu[q + 2]);
      if (du > mu) mu = du; if (de > me) me = de; if (dn > mn) mn = dn;
      su += du * du;
    }
    return { cascade: c, n, maxUp: mu / hs, maxEast: me / hs, maxNorth: mn / hs, rmsUp: Math.sqrt(su / nn) / hs, maxAbsM: Math.max(mu, me, mn) };
  }

  let cpuOcean: any = null, cpuOceanTier = '';
  function chainedCompare(out: Float32Array, T: number, hs: number) {
    const nd = FFT_TIERS[fft.tier].detailSize;
    if (!cpuOcean || cpuOceanTier !== `${fft.tier}:${T}`) {
      cpuOcean = createOcean({ shape, preset, seed: 1, query: [
        { size: 256, snapshotDt: T, filter: 'bilinear' }, { size: 256, snapshotDt: T, filter: 'bilinear' }, { size: nd, snapshotDt: T, filter: 'bilinear' },
      ] });
      cpuOceanTier = `${fft.tier}:${T}`;
    }
    const r = { east: 0, north: 0, up: 0, vEast: 0, vNorth: 0, vUp: 0 };
    let mu = 0, me = 0, mn = 0, su = 0;
    for (let i = 0; i < K; i++) {
      const e = ptsArr[2 * i], nn = ptsArr[2 * i + 1];
      cpuOcean.sampleRest(e, nn, T, r);
      const du = Math.abs(out[4 * i + 1] - r.up), de = Math.abs(out[4 * i] - (r.east - e)), dn = Math.abs(-out[4 * i + 2] - (r.north - nn));
      if (du > mu) mu = du; if (de > me) me = de; if (dn > mn) mn = dn;
      su += du * du;
    }
    return { points: K, maxUp: mu / hs, maxEast: me / hs, maxNorth: mn / hs, rmsUp: Math.sqrt(su / K) / hs };
  }

  function runXfadeSetup(): number {
    ocean.setPreset('storm');
    let t = 20, step = 0;
    const dt = 1 / 60;
    for (let i = 0; i < 4000; i++) {
      ocean.tick('simulate', { dt, time: t, step: step++ });
      const st = ocean.getState();
      if (st.fromSlot !== st.toSlot && t >= st.blendStart + st.blendDuration * 0.5) break;
      t += dt;
    }
    const st = ocean.getState();
    return st.blendStart + st.blendDuration * 0.5;
  }

  function checkFrame() {
    if (checkIdx < 0 || checkIdx >= TESTS.length || checkBusy) return;
    const test = TESTS[checkIdx];
    if (!checkSetup) {
      checkSetup = true;
      checkTime = test.t === 'xfade' ? runXfadeSetup() : test.t;
      if (test.t === 'xfade') checkWait = 16; // let the amortised cascade-3 spectrum + uploads finish
    }
    if (--checkWait > 0) return;
    // this frame's FFT ran at checkTime (see loop); request readbacks (resolved later, never awaited here)
    checkBusy = true;
    console.log('[fft-debug] check readback', test.name, checkTime);
    const T = checkTime, chop = ocean.choppinessAt(T), hs = results.hs;
    const st = ocean.getState();
    const info = { name: test.name, t: T, blend: st.fromSlot === st.toSlot ? 1 : ocean.blendAt(T), chop, from: st.fromSlot, to: st.toSlot };
    const tier = fft.tier;
    const nb = fft.debug.groupSize('base'), nd = fft.debug.groupSize('detail');
    const wantChain = test.t !== 'xfade';
    if (wantChain) renderer.compute(chainKernel);
    Promise.all([
      fft.debug.readFields('base'), fft.debug.readFields('detail'),
      wantChain ? renderer.getArrayBufferAsync(outAttr).then((b: ArrayBuffer) => new Float32Array(b)) : Promise.resolve(null),
    ]).then(([gb, gd, chainOut]) => {
      console.log('[fft-debug] readback done', test.name);
      const rows: Row[] = [];
      rows.push(compare(gb, 0, cpuField(0, nb, T, chop), nb, 0, hs));
      rows.push(compare(gb, 1, cpuField(1, nb, T, chop), nb, 1, hs));
      rows.push(compare(gd, 0, cpuField(2, nd, T, chop), nd, 2, hs));
      if (FFT_TIERS[tier].cascades === 4) rows.push(compare(gd, 1, cpuField(3, nd, T, chop), nd, 3, hs));
      const entry: any = { ...info, fields: rows };
      if (chainOut) entry.chained = chainedCompare(chainOut, T, hs);
      results.tests.push(entry);
      checkIdx++; checkWait = 4; checkBusy = false; checkSetup = false;
      if (checkIdx >= TESTS.length) { results.done = true; ocean.setPreset(preset, true); }
      showResults();
    }).catch((e) => { results.error = String(e?.stack ?? e); results.done = true; checkBusy = false; showResults(); });
  }

  function showResults() {
    const f = (x: number) => x.toExponential(2);
    let s = `GPU vs CPU (tier ${results.tier}, Hs ${results.hs.toFixed(3)} m), max |err| / Hs per field:\n`;
    for (const t of results.tests) {
      s += `${t.name}  (t=${t.t.toFixed(3)}, w=${t.blend.toFixed(3)}, chop=${t.chop.toFixed(3)})\n`;
      for (const r of t.fields) s += `  c${r.cascade} ${r.n}^2  up ${f(r.maxUp)}  east ${f(r.maxEast)}  north ${f(r.maxNorth)}  (rms up ${f(r.rmsUp)}, max ${f(r.maxAbsM)} m)\n`;
      if (t.chained) s += `  chained texture lookup (rgba16f, bilinear, ${t.chained.points} pts): up ${f(t.chained.maxUp)}  east ${f(t.chained.maxEast)}  north ${f(t.chained.maxNorth)}\n`;
    }
    if (results.error) s += `ERROR ${results.error}\n`;
    if (results.done) s += 'done';
    $('results').textContent = s;
  }

  // ---------------------------------------------------------------- loop
  const ctx: PassContext = { renderer, dt: 0, frame: 0, width: innerWidth, height: innerHeight, tier: tierParam, time: 0, targets: {} };
  let gpuCompute = 0, pendingC = false;
  const onC = (ms: number) => { gpuCompute = ms || 0; pendingC = false; };
  const onFail = () => { pendingC = false; };
  const t0 = performance.now();
  let last = t0, statT = 0, acc = 0, accN = 0;
  let checkArmed = autoCheck;
  function frame(now: number) {
    requestAnimationFrame(frame);
    const c0 = performance.now();
    const interval = now - last; last = now;
    ctx.dt = interval / 1000; ctx.frame++;
    ctx.time = checkIdx >= 0 && checkIdx < TESTS.length ? checkTime : fixedT ?? (now - t0) / 1000;
    ctx.tier = fft.tier;
    fft.pass.execute(ctx);
    if (checkArmed && fft.pendingTier === null && ctx.frame > 10) { checkArmed = false; startCheck(); }
    checkFrame();
    if (!bench) { renderer.setRenderTarget(null); quad.render(renderer); }
    if (!pendingC && renderer.hasFeature('timestamp-query')) { pendingC = true; renderer.resolveTimestampsAsync('compute').then(onC, onFail); }
    if (!bench) renderer.resolveTimestampsAsync('render');
    const h = perfState.head;
    perfState.cpuMs[h] = performance.now() - c0; perfState.frameMs[h] = interval; perfState.gpuComputeMs[h] = gpuCompute;
    perfState.head = (h + 1) % RING; perfState.count++; perfState.frames++;
    acc += gpuCompute; accN++;
    if (now - statT > 500) {
      $('stats').textContent = `tier ${fft.tier}${fft.pendingTier ? ` (compiling ${fft.pendingTier})` : ''}  t ${ctx.time.toFixed(2)} s\n` +
        `GPU compute ${(acc / Math.max(accN, 1)).toFixed(3)} ms  (FFT pass${checkIdx >= 0 && !results.done ? ' + check' : ''})\n` +
        `memory ~${(fft.debug.memoryBytes() / 1048576).toFixed(1)} MB  compile ${fft.debug.lastCompileMs().toFixed(0)} ms\n` +
        `unresolved slope var ${(fft.uniforms.unresolvedSlopeVariance.value as number).toExponential(2)}`;
      statT = now; acc = 0; accN = 0;
    }
  }
  requestAnimationFrame(frame);
}

main().catch((e) => { console.error('[fft-debug] fatal', e); $('stats').textContent = `fatal: ${e?.message ?? e}`; });
