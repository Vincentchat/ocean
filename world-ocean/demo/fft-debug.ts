// GPU FFT debug page (WebGPU only): shows every cascade's textures and runs a numeric GPU-vs-CPU check.
// Uses only public APIs: the core's createOcean (getState / blendAt / choppinessAt / tick / setPreset) and the
// adapter's fft-tiers.ts (an independent exact CPU evaluation of the drawn fields).
//
// URL flags
//   ?tier=low|medium|high|ultra   FFT tier (default high)      ?capillary  enable ultra's capillary band
//   ?preset=calm|breeze|windy|storm  core preset (default windy = the reference)
//   ?t=seconds                    fixed draw time (default: running clock from 0)
//   ?lod=n                        mip level shown in the panels
//   ?check                        run the GPU-vs-CPU check automatically; results in window.__fftCheck
//   ?bench                        FFT only (no panels): GPU timestamps in window.__perf (tools/perf.mjs format)
//
// Panels (rows = cascades 0..2 (+3), columns): height (up, red +, blue -), per-cascade normal (east/north tilt in
// red/green, blue = mip shortening x10), foam compression (1 - normal.w).
//
// Check 1 (fields): the f32 FFT output of each group (before the rgba16f store) is read back asynchronously and
// compared texel by texel with fft-tiers.evaluateFields (f64, exact time) for the same h0 / omega / blend / chop, at
// t = 12.5 s, 8191.7 s (loop wrap), 100000.37 s and mid-crossfade. Check 2: displacement mips vs the CPU box
// average. Check 3 (textures + sampling kernel): createFftSampling().sampleDisplacement at 4096 random rest points
// (3 master cascades), chained, LOD 0, Catmull-Rom and bilinear vs fft-tiers.sampleChained of the exact fields.
import { NodeMaterial, QuadMesh, StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import { Fn, clamp, float, floor, fract, instanceIndex, int, ivec2, length, select, storage, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';
import { createOcean } from '../index.ts';
import { createFlatShape } from '../stubs/impl.ts';
import { createOceanFft, createFftSampling, FFT_CASCADES, CAPILLARY_CASCADE, type OceanTier } from '../three/fft.ts';
import { evaluateFields, sampleChained } from '../three/fft-tiers.ts';
import type { PassContext } from '../three/pass-types.ts';

const $ = (id: string) => document.getElementById(id)!;
const params = new URLSearchParams(location.search);
const tierParam = (params.get('tier') ?? 'high') as OceanTier;
const preset = (params.get('preset') ?? 'windy') as any;
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
  const fft = createOceanFft({ renderer, ocean, tier: tierParam, capillary: params.has('capillary') });
  (window as any).__fft = { fft, ocean, renderer };
  const tierSel = $('tier') as HTMLSelectElement;
  tierSel.value = tierParam;
  tierSel.onchange = () => fft.setTier(tierSel.value as OceanTier);
  const ALL = [...FFT_CASCADES, CAPILLARY_CASCADE];

  // ---------------------------------------------------------------- panels
  const sigma = [0.5, 0.08, 0.008, 0.001].map((x) => x * Math.max(ocean.expectedHs(), 0.1));
  const uLod = uniform(Number(params.get('lod') ?? 0));
  ($('lod') as HTMLSelectElement).value = String(uLod.value);
  ($('lod') as HTMLSelectElement).onchange = (e) => { uLod.value = Number((e.target as HTMLSelectElement).value); };
  const N = fft.nodes, U = fft.uniforms;
  const ROWS = 4;
  const mat = new NodeMaterial();
  mat.fragmentNode = Fn(() => {
    const p = uv();
    const col = floor(p.x.mul(3)), row = floor(float(1).sub(p.y).mul(ROWS));
    const q = vec2(fract(p.x.mul(3)), fract(float(1).sub(p.y).mul(ROWS)));
    let out: any = vec3(0);
    for (let r = 0; r < ROWS; r++) {
      const base = r < 2, li = base ? r : r - 2, Lg = base ? float(2) : U.detailLayers;
      const tex = base ? N.displacementBase : N.displacementDetail;
      const disp = tex.sample(q).depth(int(U.parity.mul(Lg).add(li))).level(uLod);
      const nrm = (base ? N.normalBase : N.normalDetail).sample(q).depth(int(Lg.mul(2).add(li))).level(uLod);
      const v = disp.x.div(sigma[r]);
      const cH = vec3(clamp(v, 0, 1), clamp(v.abs(), 0, 1).mul(0.15), clamp(v.negate(), 0, 1));
      const cN = vec3(nrm.x.mul(2).add(0.5), nrm.y.mul(2).add(0.5), float(1).sub(length(nrm.xyz)).mul(10));
      const comp = float(1).sub(nrm.w);
      const cF = vec3(comp, comp, comp.mul(0.6).add(0.1));
      let c: any = select(col.equal(0), cH, select(col.equal(1), cN, cF));
      if (r === 3) c = c.mul(U.capillaryGain);
      out = select(row.equal(r), c, out);
    }
    const edge = q.x.lessThan(0.004).or(q.y.lessThan(0.006));
    return vec4(select(edge, vec3(0.1), out), 1);
  })();
  const quad = new QuadMesh(mat);
  const labels = $('labels');
  const names = ['height (up)', 'normal tilt / mip shortening', 'foam compression'];
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < 3; c++) {
    const d = document.createElement('div');
    d.style.left = `${(c / 3) * 100}%`; d.style.top = `${(r / ROWS) * 100}%`;
    d.textContent = `c${r} L=${+ALL[r].domain.toFixed(4)} m: ${names[c]}${r === 3 ? ' (capillary band, if enabled)' : ''}`;
    labels.appendChild(d);
  }
  if (bench) labels.style.display = 'none';

  // ---------------------------------------------------------------- texture-lookup check kernels
  const K = 4096;
  const ptsArr = new Float32Array(K * 2);
  let s = 12345;
  const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
  for (let i = 0; i < K; i++) { ptsArr[2 * i] = rnd() * 400 - 200; ptsArr[2 * i + 1] = rnd() * 400 - 200; }
  const ptsAttr = new StorageBufferAttribute(ptsArr, 2);
  const outCubic = new StorageBufferAttribute(K, 4), outBilin = new StorageBufferAttribute(K, 4);
  const sampling = createFftSampling(fft);
  const pts = storage(ptsAttr, 'vec2', K).toReadOnly();
  const mkKernel = (attr: any, cubic: boolean) => {
    const o = storage(attr, 'vec4', K);
    return Fn(() => {
      const pt = pts.element(instanceIndex);
      const d = sampling.sampleDisplacement(() => vec2(pt.x, pt.y.negate()), { cubic, threeCascades: true });
      o.element(instanceIndex).assign(vec4(d, 0));
    })().compute(K);
  };
  const chainKernels = [mkKernel(outCubic, true), mkKernel(outBilin, false)];
  // Mip check: displacement mip levels 1, 3, 4, 7 of each master cascade at 256 texels vs the CPU box average.
  const MIP_LEVELS = [1, 3, 4, 7], MK = 256;
  const mipOut = new StorageBufferAttribute(MIP_LEVELS.length * 3 * MK, 4);
  const mipIdx = new Int32Array(MK * 2);
  for (let i = 0; i < MK; i++) { mipIdx[2 * i] = Math.floor(rnd() * 4096); mipIdx[2 * i + 1] = Math.floor(rnd() * 4096); }
  const mipIdxAttr = new StorageBufferAttribute(mipIdx, 2, Int32Array);
  const mipKernels = new Map<number, any>();
  function mipKernel(n2: number) {
    let k = mipKernels.get(n2);
    if (k) return k;
    const ids = storage(mipIdxAttr, 'ivec2', MK).toReadOnly();
    const o = storage(mipOut, 'vec4', mipOut.count);
    k = Fn(() => {
      const i = instanceIndex;
      const p = ids.element(i);
      for (let li = 0; li < MIP_LEVELS.length; li++) {
        const lv = MIP_LEVELS[li];
        for (let c = 0; c < 3; c++) {
          const n = c < 2 ? 256 : n2, m = Math.max(1, n >> lv), base = c < 2;
          const Lg = base ? float(2) : U.detailLayers, local = base ? c : 0;
          const xy = ivec2(p.x.mod(m), p.y.mod(m));
          const tex = base ? N.displacementBase : N.displacementDetail;
          o.element(i.add((li * 3 + c) * MK)).assign(tex.load(xy).depth(int(U.parity.mul(Lg).add(local))).level(int(Math.min(lv, Math.log2(n)))));
        }
      }
    })().compute(MK);
    mipKernels.set(n2, k);
    return k;
  }

  // ---------------------------------------------------------------- perf state (tools/perf.mjs format)
  const RING = 2048;
  const perfState = {
    frames: 0, count: 0, head: 0,
    cpuMs: new Float64Array(RING), frameMs: new Float64Array(RING), gpuComputeMs: new Float64Array(RING),
    reset() { perfState.count = 0; perfState.head = 0; },
    snapshot() {
      const n = Math.min(perfState.count, RING), o: Record<string, number[]> = { cpuMs: [], frameMs: [], gpuMs: [], gpuComputeMs: [] };
      for (let i = 0; i < n; i++) {
        const j = (perfState.head - n + i + RING) % RING;
        o.cpuMs.push(perfState.cpuMs[j]); o.frameMs.push(perfState.frameMs[j]); o.gpuMs.push(0); o.gpuComputeMs.push(perfState.gpuComputeMs[j]);
      }
      return { ...o, count: n, timestamps: renderer.hasFeature('timestamp-query'), tier: fft.pendingTier ? null : fft.tier, capillary: fft.capillary,
        memoryMB: fft.debug.memoryBytes() / 1048576, compileMs: fft.debug.lastCompileMs(), buildMainMs: fft.debug.lastBuildMainThreadMs() };
    },
  };
  (window as any).__perf = perfState;

  // ---------------------------------------------------------------- check state machine
  const results: any = { tier: null, hs: ocean.expectedHs(), tests: [] as any[], done: false, error: null };
  (window as any).__fftCheck = results;
  const TESTS: { name: string; t: number | 'xfade' }[] = [
    { name: 't=12.5', t: 12.5 }, { name: 't=8191.7 (loop wrap)', t: 8191.7 }, { name: 't=100000.37', t: 100000.37 },
    { name: 'crossfade, mid', t: 'xfade' },
  ];
  let checkIdx = -1, checkWait = 0, checkBusy = false, checkTime = 0, checkSetup = false;
  const startCheck = () => { if (checkIdx < 0 || results.done) { results.tests = []; results.done = false; results.tier = fft.tier; checkIdx = 0; checkWait = 6; checkBusy = false; checkSetup = false; } };
  $('check').onclick = startCheck;

  function runXfadeSetup(): number {
    ocean.setPreset(preset === 'storm' ? 'calm' : 'storm');
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
      if (test.t === 'xfade') checkWait = 40;
    }
    if (--checkWait > 0) return;
    {
      const s0 = ocean.getState();
      if (!fft.debug.spectrum('detail', s0.fromSlot) || !fft.debug.spectrum('detail', s0.toSlot)) { checkWait = 10; return; }
    }
    checkBusy = true;
    const T = checkTime, st = ocean.getState(), hs = results.hs;
    const w = st.fromSlot === st.toSlot ? 1 : ocean.blendAt(T), chop = ocean.choppinessAt(T);
    const info = { name: test.name, t: T, blend: w, chop, from: st.fromSlot, to: st.toSlot };
    const groups = (['base', 'detail'] as const).map((gname) => ({
      gname, n: fft.debug.size(gname), Lg: fft.debug.layers(gname), omega: fft.debug.omega(gname),
      from: fft.debug.spectrum(gname, st.fromSlot), to: fft.debug.spectrum(gname, st.toSlot),
      cascades: gname === 'base' ? [0, 1] : fft.debug.layers('detail') > 1 ? [2, 3] : [2],
    }));
    for (const k of chainKernels) renderer.compute(k);
    const n2 = groups[1].n;
    renderer.compute(mipKernel(n2));
    Promise.all([
      fft.debug.readFields('base'), fft.debug.readFields('detail'),
      renderer.getArrayBufferAsync(outCubic).then((b: ArrayBuffer) => new Float32Array(b)),
      renderer.getArrayBufferAsync(outBilin).then((b: ArrayBuffer) => new Float32Array(b)),
      renderer.getArrayBufferAsync(mipOut).then((b: ArrayBuffer) => new Float32Array(b)),
    ]).then(([gfb, gfd, gc, gb, gm]) => {
      const rows: any[] = [];
      const fields: Float64Array[] = [];
      const gpuOf: Float32Array[] = [gfb, gfb, gfd, gfd];
      for (const gr of groups) {
        if (!gr.from || !gr.to || !gr.omega) throw new Error('spectrum / omega not available');
        const nn = gr.n * gr.n;
        gr.cascades.forEach((c, li) => {
          const f = evaluateFields(gr.from![li], gr.to![li], w, gr.omega!.subarray(li * nn, (li + 1) * nn), gr.n, ALL[c].domain, T, chop);
          fields[c] = f;
          const gf = gpuOf[c];
          let mu = 0, me = 0, mn = 0;
          for (let i = 0; i < nn; i++) {
            const o = (li * nn + i) * 4;
            mu = Math.max(mu, Math.abs(gf[o] - f[3 * i])); me = Math.max(me, Math.abs(gf[o + 1] - f[3 * i + 1])); mn = Math.max(mn, Math.abs(gf[o + 2] - f[3 * i + 2]));
          }
          rows.push({ cascade: c, n: gr.n, maxUp: mu / hs, maxEast: me / hs, maxNorth: mn / hs, maxAbsM: Math.max(mu, me, mn) });
        });
      }
      const sizes = [256, 256, n2];
      const chained = (g: Float32Array, cubic: boolean) => {
        const out = new Float64Array(3), tmp = new Float64Array(3);
        let mu = 0, me = 0, mn = 0;
        for (let i = 0; i < K; i++) {
          sampleChained(fields.slice(0, 3), sizes, FFT_CASCADES, ptsArr[2 * i], ptsArr[2 * i + 1], [cubic, cubic, cubic], out, tmp);
          mu = Math.max(mu, Math.abs(g[4 * i + 1] - out[0])); me = Math.max(me, Math.abs(g[4 * i] - out[1])); mn = Math.max(mn, Math.abs(-g[4 * i + 2] - out[2]));
        }
        return { points: K, maxUp: mu / hs, maxEast: me / hs, maxNorth: mn / hs };
      };
      let mipMax = 0;
      for (let li = 0; li < MIP_LEVELS.length; li++) for (let c = 0; c < 3; c++) {
        const n = sizes[c], lv = Math.min(MIP_LEVELS[li], Math.log2(n)), m = n >> lv, B = 1 << lv, f = fields[c];
        for (let i = 0; i < MK; i++) {
          const mx = mipIdx[2 * i] % m, my = mipIdx[2 * i + 1] % m;
          const acc = [0, 0, 0];
          for (let y = 0; y < B; y++) for (let x = 0; x < B; x++) {
            const o = 3 * ((my * B + y) * n + mx * B + x);
            acc[0] += f[o]; acc[1] += f[o + 1]; acc[2] += f[o + 2];
          }
          const g4 = ((li * 3 + c) * MK + i) * 4;
          for (let q = 0; q < 3; q++) mipMax = Math.max(mipMax, Math.abs(gm[g4 + q] - acc[q] / (B * B)));
        }
      }
      results.tests.push({ ...info, fields: rows, chainedCubic: chained(gc, true), chainedBilinear: chained(gb, false), mipMaxErr: mipMax / hs });
      checkIdx++; checkWait = 6; checkBusy = false; checkSetup = false;
      if (checkIdx >= TESTS.length) { results.done = true; ocean.setPreset(preset, true); }
      showResults();
    }).catch((e) => { results.error = String(e?.stack ?? e); results.done = true; checkBusy = false; showResults(); });
  }

  function showResults() {
    const f = (x: number) => x.toExponential(2);
    let s = `GPU vs CPU (tier ${results.tier}, Hs ${results.hs.toFixed(3)} m), max |err| / Hs:\n`;
    for (const t of results.tests) {
      s += `${t.name}  (t=${t.t.toFixed(3)}, w=${t.blend.toFixed(3)}, chop=${t.chop.toFixed(3)})\n`;
      for (const r of t.fields) s += `  c${r.cascade} ${r.n}^2 field  up ${f(r.maxUp)}  east ${f(r.maxEast)}  north ${f(r.maxNorth)}  (max ${f(r.maxAbsM)} m)\n`;
      s += `  displacement mips 1/3/4/7 vs CPU box average: max ${f(t.mipMaxErr)}\n`;
      for (const [k, v] of [['catmull-rom', t.chainedCubic], ['bilinear', t.chainedBilinear]] as const) {
        s += `  chained rgba16f lookup ${k}: up ${f(v.maxUp)}  east ${f(v.maxEast)}  north ${f(v.maxNorth)}\n`;
      }
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
    if (checkArmed && fft.pendingTier === null && fft.debug.size('detail') > 0 && ctx.frame > 10) { checkArmed = false; startCheck(); }
    checkFrame();
    if (!bench) { renderer.setRenderTarget(null); quad.render(renderer); }
    if (!pendingC && renderer.hasFeature('timestamp-query')) { pendingC = true; renderer.resolveTimestampsAsync('compute').then(onC, onFail); }
    if (!bench) renderer.resolveTimestampsAsync('render');
    const h = perfState.head;
    perfState.cpuMs[h] = performance.now() - c0; perfState.frameMs[h] = interval; perfState.gpuComputeMs[h] = gpuCompute;
    perfState.head = (h + 1) % RING; perfState.count++; perfState.frames++;
    acc += gpuCompute; accN++;
    if (now - statT > 500) {
      $('stats').textContent = `tier ${fft.tier}${fft.capillary ? ' + capillary' : ''}${fft.pendingTier ? ` (building ${fft.pendingTier})` : ''}  N2 ${fft.debug.size('detail')}  t ${ctx.time.toFixed(2)} s\n` +
        `GPU compute ${(acc / Math.max(accN, 1)).toFixed(3)} ms  (FFT pass${checkIdx >= 0 && !results.done ? ' + check' : ''})\n` +
        `memory ~${(fft.debug.memoryBytes() / 1048576).toFixed(1)} MB  compile ${fft.debug.lastCompileMs().toFixed(0)} ms  build main-thread ${fft.debug.lastBuildMainThreadMs().toFixed(1)} ms\n` +
        `unresolved slope var ${(fft.uniforms.unresolvedSlopeVariance.value as number).toExponential(2)}`;
      statT = now; acc = 0; accN = 0;
    }
  }
  requestAnimationFrame(frame);
}

main().catch((e) => { console.error('[fft-debug] fatal', e); $('stats').textContent = `fatal: ${e?.message ?? e}`; });
