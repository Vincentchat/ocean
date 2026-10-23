#!/usr/bin/env node
// Frame-time harness: open a URL in headless Edge (hardware GPU), warm up, sample N frames, print JSON.
//   node tools/perf.mjs --url <url> [--frames 600] [--warmup 5] [--width 1920 --height 1080] [--vsync]
//        [--out file.json] [--label name]
// Measures, for any page, via an injected requestAnimationFrame wrapper:
//   frameMs = rAF interval, cpuMs = time spent inside rAF callbacks (main-thread JS per frame).
// If the page exposes window.__perf (the world-ocean demo does), GPU timestamp ms (render + compute,
// from WebGPURenderer trackTimestamp) are reported too.
// By default the frame rate is uncapped (--disable-gpu-vsync --disable-frame-rate-limit) so fps reflects
// throughput rather than the 60 Hz display; pass --vsync to keep the cap.
import { writeFileSync } from 'node:fs';
import { ADAPTER_PROBE, isSoftwareAdapter, launch, parseArgs, watchPage } from './lib/browser.mjs';

const opts = parseArgs(process.argv.slice(2));
if (!opts.url) {
  console.error('usage: node tools/perf.mjs --url <url> [--frames 600] [--warmup 5] [--out file.json]');
  process.exit(2);
}
const width = Number(opts.width ?? 1920), height = Number(opts.height ?? 1080);
const frames = Number(opts.frames ?? 600), warmup = Number(opts.warmup ?? 5);
const extraArgs = opts.vsync ? [] : ['--disable-gpu-vsync', '--disable-frame-rate-limit'];

const RAF_PROBE = () => {
  const N = 8192;
  const st = { interval: new Float64Array(N), cpu: new Float64Array(N), n: 0, last: 0, on: false };
  window.__rafPerf = st;
  const raf = window.requestAnimationFrame.bind(window);
  // Group all callbacks of one frame: the first callback of a frame records the interval,
  // and the callback CPU time is summed per frame.
  let frameStamp = -1, idx = -1;
  window.requestAnimationFrame = (cb) => raf((now) => {
    const t0 = performance.now();
    if (now !== frameStamp) {
      frameStamp = now;
      if (st.on) {
        idx = st.n % N;
        st.interval[idx] = st.last ? now - st.last : 0;
        st.cpu[idx] = 0;
        st.n++;
      }
      st.last = now;
    }
    try { cb(now); } finally { if (st.on && idx >= 0) st.cpu[idx] += performance.now() - t0; }
  });
};

function stats(arr) {
  const a = arr.filter((x) => Number.isFinite(x) && x > 0).sort((x, y) => x - y);
  if (!a.length) return null;
  const q = (p) => a[Math.min(a.length - 1, Math.floor(p * a.length))];
  const mean = a.reduce((s, x) => s + x, 0) / a.length;
  const r = (x) => Math.round(x * 1000) / 1000;
  return { mean: r(mean), p50: r(q(0.5)), p95: r(q(0.95)), p99: r(q(0.99)), min: r(a[0]), max: r(a[a.length - 1]), n: a.length };
}

const { browser, context } = await launch({ width, height, extraArgs });
let exit = 0;
try {
  await context.addInitScript(RAF_PROBE);
  const page = await context.newPage();
  const watch = watchPage(page, { label: opts.label ?? 'perf', quiet: !!opts.quiet });
  await page.goto(opts.url, { waitUntil: 'load', timeout: 120000 });
  const adapter = await page.evaluate(ADAPTER_PROBE);
  if (!adapter.ok || isSoftwareAdapter(adapter)) throw new Error(`no hardware WebGPU adapter: ${JSON.stringify(adapter)}`);
  const webgl = await page.evaluate(() => {
    const c = document.createElement('canvas').getContext('webgl2');
    const ext = c && c.getExtension('WEBGL_debug_renderer_info');
    return ext ? c.getParameter(ext.UNMASKED_RENDERER_WEBGL) : null;
  });
  if (opts['wait-ready']) await page.waitForFunction(() => window.__shotReady === true, null, { timeout: 120000 });
  await page.waitForTimeout(warmup * 1000);
  await page.evaluate(() => { window.__rafPerf.n = 0; window.__rafPerf.on = true; window.__perf?.reset?.(); });
  await page.waitForFunction((n) => window.__rafPerf.n >= n, frames, { timeout: 600000, polling: 250 });
  const raw = await page.evaluate(() => {
    const st = window.__rafPerf; st.on = false;
    const n = Math.min(st.n, st.interval.length);
    const demo = window.__perf?.snapshot?.() ?? null;
    return { interval: Array.from(st.interval.subarray(1, n)), cpu: Array.from(st.cpu.subarray(1, n)), demo,
      canvas: (() => { const c = document.querySelector('canvas'); return c ? [c.width, c.height] : null; })() };
  });
  const frameMs = stats(raw.interval);
  const result = {
    label: opts.label ?? null,
    url: opts.url,
    date: new Date().toISOString(),
    adapter: { vendor: adapter.vendor, architecture: adapter.architecture, description: adapter.description },
    webglRenderer: webgl,
    viewport: [width, height],
    canvas: raw.canvas,
    vsync: !!opts.vsync,
    framesSampled: raw.interval.length,
    fps: frameMs ? Math.round((1000 / frameMs.mean) * 10) / 10 : null,
    frameMs,
    cpuMs: stats(raw.cpu),
    gpuMs: raw.demo?.timestamps ? stats(raw.demo.gpuMs.map((r, i) => r + (raw.demo.gpuComputeMs[i] || 0))) : null,
    gpuRenderMs: raw.demo?.timestamps ? stats(raw.demo.gpuMs) : null,
    gpuComputeMs: raw.demo?.timestamps ? stats(raw.demo.gpuComputeMs) : null,
    demoCpuMs: raw.demo ? stats(raw.demo.cpuMs) : null,
    tier: raw.demo?.tier ?? null,
    consoleErrors: watch.problems.filter((p) => p.type !== 'warning').length,
  };
  const json = JSON.stringify(result, null, 2);
  console.log(json);
  if (opts.out) writeFileSync(opts.out, json + '\n');
  if (watch.webgpuErrors.length) { console.error(`perf: ${watch.webgpuErrors.length} WebGPU/page error(s)`); exit = 1; }
} catch (e) {
  console.error(`perf: ${e.message || e}`);
  exit = 1;
} finally {
  await browser.close();
}
process.exit(exit);
