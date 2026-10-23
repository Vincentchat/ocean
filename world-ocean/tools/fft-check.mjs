#!/usr/bin/env node
// GPU FFT verification driver: opens demo/fft-debug.html per tier in headless Edge (hardware WebGPU), runs the
// GPU-vs-CPU field check (?check) and a short GPU timestamp bench (?bench), prints JSON.
//   node tools/fft-check.mjs --base http://127.0.0.1:5182 [--tiers low,medium,high,ultra] [--preset breeze]
//        [--bench-frames 600] [--out file.json] [--shots dir]
import { writeFileSync } from 'node:fs';
import { ADAPTER_PROBE, isSoftwareAdapter, launch, parseArgs, watchPage } from './lib/browser.mjs';

const opts = parseArgs(process.argv.slice(2));
const base = opts.base ?? 'http://127.0.0.1:5182';
const tiers = String(opts.tiers ?? 'low,medium,high,ultra,ultra+cap').split(','); // 'ultra+cap' = ultra with the capillary band
const q = (tier) => { const [t, cap] = tier.split('+'); return `tier=${t}${cap ? '&capillary' : ''}`; };
const preset = opts.preset ?? 'windy';
const benchFrames = Number(opts['bench-frames'] ?? 600);

const { browser, context } = await launch({ width: 1600, height: 900, extraArgs: ['--disable-gpu-vsync', '--disable-frame-rate-limit'] });
const out = { date: new Date().toISOString(), preset, adapter: null, tiers: {} };
let exit = 0;
try {
  for (const tier of tiers) {
    let check = null;
    const watch = { problems: [], webgpuErrors: [] };
    if (!opts['no-check']) {
      const page = await context.newPage();
      const w = watchPage(page, { label: `fft-${tier}` });
      await page.goto(`${base}/demo/fft-debug.html?${q(tier)}&preset=${preset}&check`, { waitUntil: 'load', timeout: 120000 });
      if (!out.adapter) {
        const a = await page.evaluate(ADAPTER_PROBE);
        if (!a.ok || isSoftwareAdapter(a)) throw new Error(`no hardware WebGPU adapter: ${JSON.stringify(a)}`);
        out.adapter = a;
      }
      await page.waitForFunction(() => window.__fftCheck?.done === true, null, { timeout: 300000, polling: 250 });
      check = await page.evaluate(() => window.__fftCheck);
      if (opts.shots) await page.screenshot({ path: `${opts.shots}/fft-debug-${tier}.png` });
      await page.close();
      watch.problems = w.problems; watch.webgpuErrors = w.webgpuErrors;
    }
    // bench: FFT pass only
    const bp = await context.newPage();
    const bw = watchPage(bp, { label: `bench-${tier}` });
    await bp.goto(`${base}/demo/fft-debug.html?${q(tier)}&preset=${preset}&bench`, { waitUntil: 'load', timeout: 120000 });
    // (optional chaining everywhere: the dev server may full-reload the page when other files change)
    await bp.waitForFunction(() => window.__perf?.snapshot?.().tier && window.__perf.count > 60, null, { timeout: 120000 });
    await bp.waitForTimeout(1500);
    await bp.evaluate(() => window.__perf?.reset());
    await bp.waitForFunction((n) => (window.__perf?.count ?? 0) >= n, benchFrames, { timeout: 300000, polling: 250 });
    const snap = await bp.evaluate(() => window.__perf.snapshot());
    await bp.close();
    const g = snap.gpuComputeMs.filter((x) => x > 0).sort((a, b) => a - b);
    const nullFrames = snap.gpuComputeMs.length - g.length;
    const mean = g.reduce((s, x) => s + x, 0) / Math.max(g.length, 1);
    out.tiers[tier] = {
      check,
      gpuComputeMs: { mean: +mean.toFixed(4), p50: g[Math.floor(g.length / 2)], p95: g[Math.floor(g.length * 0.95)], min: g[0], n: g.length, missing: nullFrames },
      memoryMB: +snap.memoryMB.toFixed(2), compileMs: Math.round(snap.compileMs), buildMainMs: +(snap.buildMainMs ?? 0).toFixed(1),
      consoleErrors: watch.problems.length + bw.problems.length,
    };
    if (watch.webgpuErrors.length || bw.webgpuErrors.length) exit = 1;
    console.error(`[fft-check] ${tier}: gpu mean ${mean.toFixed(3)} p50 ${(g[Math.floor(g.length / 2)] ?? 0).toFixed(3)} ms (${g.length} frames), check ${!check ? 'skipped' : check.error ? 'ERROR' : 'ok'}`);
  }
} catch (e) {
  console.error(`fft-check: ${e.message || e}`);
  exit = 1;
} finally {
  await browser.close();
}
const json = JSON.stringify(out, null, 2);
console.log(json);
if (opts.out) writeFileSync(opts.out, json + '\n');
process.exit(exit);
